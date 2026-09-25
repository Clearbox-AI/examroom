"""Extract solved exercises from OpenStax books (HTML edition).

OpenStax renders every book both as PDF and as server-side HTML at
openstax.org/books/<slug>/pages/<page>.  The HTML is the better source:
answers typeset as equations are vector drawings in the PDF (invisible to
text extraction) but full MathML in the HTML.  Problems live on
``<chapter>-problems-exercises`` pages, answers on the ``chapter-<n>``
answer-key pages, and the two are joined by problem number (only
odd-numbered problems have published answers).

Everything here is stdlib-only: the pages are machine-generated CNX markup,
regular enough that a real HTML parser is not needed.
"""

from __future__ import annotations

import html as _html
import math
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

BASE_URL = "https://openstax.org/books/{book}/pages/{page}"
_UA = {"User-Agent": "Mozilla/5.0 (examroom authoring)"}


def fetch_page(book: str, page: str, cache_dir: str | Path | None = None) -> str:
    """Fetch one book page, optionally caching the raw HTML on disk."""
    if cache_dir is not None:
        cached = Path(cache_dir) / f"{book}__{page}.html"
        if cached.exists():
            return cached.read_text()
    req = urllib.request.Request(BASE_URL.format(book=book, page=page), headers=_UA)
    with urllib.request.urlopen(req) as r:
        text = r.read().decode("utf-8")
    if cache_dir is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(text)
    return text


# --------------------------------------------------------------------------
# MathML -> plain text
# --------------------------------------------------------------------------

_MATH_RE = re.compile(r"<math[\s>].*?</math>", re.S)


def flatten_math(fragment: str) -> str:
    """Render a MathML element as plain text ("3.0 × 10^4 m/s")."""
    try:
        el = ET.fromstring(fragment)
    except ET.ParseError:
        return re.sub(r"<[^>]+>", " ", fragment)
    return _squash(_flatten(el))


def _flatten(el: ET.Element) -> str:
    tag = el.tag.rsplit("}", 1)[-1]  # drop namespace if present
    if tag in ("annotation", "annotation-xml"):
        return ""
    kids = [_flatten(c) + (c.tail or "") for c in el]
    body = (el.text or "") + "".join(kids)
    if tag == "msup":
        base, exp = _pair(el)
        return f"{base}^{exp}"
    if tag == "msub":
        base, sub = _pair(el)
        return f"{base}_{sub}"
    if tag == "mfrac":
        num, den = _pair(el)
        return f"({num})/({den})"
    if tag == "msqrt":
        return f"sqrt({body})"
    if tag == "mspace":
        return " "
    return body


def _pair(el: ET.Element) -> tuple[str, str]:
    kids = [_flatten(c) for c in el]
    kids += [""] * (2 - len(kids))
    return _squash(kids[0]), _squash(kids[1])


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def strip_html(fragment: str) -> str:
    """Plain text of an HTML fragment, with MathML flattened, not dropped."""
    fragment = _MATH_RE.sub(lambda m: " " + flatten_math(m.group()) + " ", fragment)
    fragment = re.sub(r"</p>", "\n", fragment)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    fragment = _html.unescape(fragment)
    fragment = fragment.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    lines = [_squash(l) for l in fragment.split("\n")]
    return "\n".join(l for l in lines if l)


# --------------------------------------------------------------------------
# Page parsing
# --------------------------------------------------------------------------


@dataclass
class Exercise:
    number: int
    text: str
    section: str = ""          # e.g. "2.1 Displacement"
    has_solution: bool = False
    has_figure: bool = False


_H3_RE = re.compile(
    r"<h3[^>]*>\s*<span[^>]*os-number[^>]*>(.*?)</span>.*?os-text[^>]*>(.*?)</span>", re.S
)
_H3_PLAIN_RE = re.compile(r"<h3[^>]*>(.*?)</h3>", re.S)
_NUMBER_RE = re.compile(r'class="os-number"[^>]*>(\d+)<')


def parse_exercises(page_html: str) -> list[Exercise]:
    """Problems on a ``…-problems-exercises`` page, with their section titles."""
    sections: list[tuple[int, str]] = []
    for m in _H3_PLAIN_RE.finditer(page_html):
        title = _squash(re.sub(r"<[^>]+>", " ", m.group(1)))
        if re.match(r"\d+\.\d+", title):
            sections.append((m.start(), title))

    out: list[Exercise] = []
    chunks = re.split(r'(?=<div data-type="exercise")', page_html)
    pos = 0
    for chunk in chunks[1:]:
        pos = page_html.find(chunk, pos)
        m = _NUMBER_RE.search(chunk)
        if not m:
            continue
        end = chunk.find("</section>")
        body = chunk[_after_divider(chunk, m.end()):end if end != -1 else None]
        section = ""
        for start, title in sections:
            if start < pos:
                section = title
        text = strip_html(body)
        out.append(
            Exercise(
                number=int(m.group(1)),
                text=text,
                section=section,
                has_solution="os-hasSolution" in chunk[: chunk.find(">")],
                has_figure="<figure" in chunk or "<img" in chunk
                or bool(re.search(r"\bFigure\s+\d", text)),
            )
        )
    return out


def _after_divider(chunk: str, default: int) -> int:
    """Index just past the '<span class="os-divider">. </span>' header."""
    d = chunk.find("os-divider", default)
    if d == -1:
        return default
    close = chunk.find("</span>", d)
    return close + len("</span>") if close != -1 else default


def parse_solutions(page_html: str, problems_slug: str) -> dict[int, str]:
    """Answer-key page -> {problem number: answer text} for one exercises page."""
    out: dict[int, str] = {}
    for chunk in re.split(r'(?=<div data-type="solution")', page_html)[1:]:
        head = chunk[: chunk.find("os-divider")]
        if f'data-page-slug="{problems_slug}"' not in head:
            continue
        m = _NUMBER_RE.search(head)
        if not m:
            continue
        end = chunk.find("</div>\n</div>")
        start = chunk.find(">", chunk.find("os-solution-container")) + 1
        body = chunk[start:end if end != -1 else None]
        out[int(m.group(1))] = strip_html(body)
    return out


# --------------------------------------------------------------------------
# Answers -> (value, unit, tolerance)
# --------------------------------------------------------------------------

_PART_RE = re.compile(r"\(([a-h])\)")


def split_parts(text: str) -> dict[str, str]:
    """Split "(a) … (b) …" into {"a": …, "b": …}; {"": text} when unlabelled."""
    marks = list(_PART_RE.finditer(text))
    seen: dict[str, str] = {}
    expected = "a"
    starts: list[tuple[str, int, int]] = []
    for m in marks:
        if m.group(1) == expected:  # ignore stray "(a)" references out of order
            starts.append((m.group(1), m.start(), m.end()))
            expected = chr(ord(expected) + 1)
    if not starts:
        return {"": text.strip()}
    for i, (letter, _, body_start) in enumerate(starts):
        body_end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        seen[letter] = text[body_start:body_end].strip()
    return seen


@dataclass
class Quantity:
    value: float
    unit: str
    sigfigs: int
    rest: str = ""  # trailing text after the quantity (qualifiers, second results…)


_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")

_QTY_RE = re.compile(
    r"""^[\s~]*(?:about\s+|approximately\s+|≈\s*)?
    (?:[A-Za-z][\w¯' ]{0,24}[=:]\s*)?          # leading label: "v_0 =", "average speed:"
    (?P<sign>[-+−]?)\s*
    (?P<mantissa>\d[\d,]*(?:\.\d+)?)
    (?:
        \s*(?:[×xX*·]|\\times)\s*10\s*(?:\^|\*\*)?\s*\{?(?P<exp>[-+−]?\d+)\}?
      | [eE](?P<eexp>[-+]?\d+)
    )?
    \s*(?P<unit>(?:°|º|%|[A-Za-zµΩ°º/·.\^0-9\-]|\s)*?)
    (?P<rest>[,;(=].*|\.\s.*|)$""",
    re.S | re.X,
)

_UNIT_OK = re.compile(r"^[A-Za-zµΩ°º%][A-Za-zµΩ°º%/·.\^0-9\- ]*$")


def parse_quantity(text: str) -> Quantity | None:
    """Parse the leading physical quantity of an answer string.

    Handles "3.0 × 10^4 m/s", "27 cm/s", "384,000 km", "2 × 10⁷ years",
    "0 m/s", "-2.50 m/s^2".  Returns None when the answer does not start
    with a number (prose, figure description, algebraic expression).
    """
    text = text.translate(_SUPERSCRIPT).replace("−", "-").strip()
    # MathML pretty-printing can put spaces around a decimal point ("34 . 689")
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
    m = _QTY_RE.match(text)
    if not m:
        return None
    mantissa = m.group("mantissa").replace(",", "")
    value = float(mantissa)
    exp = m.group("exp") or m.group("eexp")
    if exp:
        value *= 10.0 ** int(exp.replace("−", "-"))
    if m.group("sign") in ("-", "−"):
        value = -value
    digits = mantissa.replace(".", "").lstrip("0")
    if "." not in mantissa:
        digits = digits.rstrip("0")  # trailing zeros of an integer are not significant
    sigfigs = len(digits) if digits else 1
    unit = _squash(m.group("unit") or "")
    if unit and not _UNIT_OK.match(unit):
        unit = ""
    return Quantity(value=value, unit=unit, sigfigs=sigfigs, rest=_squash(m.group("rest") or ""))


def sigfig_tolerance(q: Quantity) -> float:
    """Half a unit in the last significant digit (0 for an exact zero)."""
    if q.value == 0:
        return 0.0
    exp10 = math.floor(math.log10(abs(q.value)))
    return 0.5 * 10.0 ** (exp10 - q.sigfigs + 1)
