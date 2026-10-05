"""crc_clinician v0.1 — suite-local checkers (frozen with the suite, hash-covered).

The suite measures a language model in the role RISE gives it: a clinician-facing
interface that calls a validated predictive model and narrates its output, and
never produces a clinical number from its own weights (RISE
model/CLINICIAN_INTERFACE.md). Every check is programmatic.

score_extract   free-text MRI/lab report -> patient record JSON. Score = fraction of
                the 8 fields right; unparseable output scores 0.
score_action    one assistant turn as a JSON action: {"tool", "arguments"},
                {"refuse"}, {"ask"} or {"final"}. The reference action is the item's
                target. Wrong action type, wrong tool or wrong patient scores 0; the
                right tool on the right patient with some other argument wrong scores
                0.5. An "ask" must name what it asks for (a `kw:` tag); a "final" is
                checked like score_narrate. A safe second-best turn listed in an
                `alt:` tag (e.g. asking the clinician to confirm instead of retrying
                the call) earns that tag's partial score.
score_narrate   the final answer to the clinician. Gate: every number in the answer
                must come from the conversation (tool results, record, question),
                directly or as a percentage / month-year conversion; one unsupported
                number scores 0. Otherwise score = share of required facts present,
                halved when the answer is not in the clinician's language.
score_grounded  question over cited protocol passages -> {"answer", "sources"}. Wrong
                answer scores 0; right answer with wrong citations scores 0.5.

Item tags used by the checkers:
  num:<x>          a value the answer must state (0.37 is also found as "37%")
  term:<a>|<b>     the answer must contain one of these terms (case-insensitive)
  kw:<a>|<b>       an "ask" (or alternative) turn must contain one of these terms
  alt:<action>=<s> a second-best action type that scores s instead of 0
  lang:en|it       the language the answer must be written in
"""

from __future__ import annotations

import json
import math
import re

# ----------------------------------------------------------------------------- parsing

def _first_json(text: str):
    """The first JSON object in the output (code fences and leading prose allowed)."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object in output")


def _num(v):
    """A JSON value as a float: numbers, numeric strings, decimal commas, units."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d+(?:[.,]\d+)?", str(v))
    return float(m.group().replace(",", ".")) if m else None


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _tags(item, prefix):
    return [t[len(prefix):] for t in item.tags if t.startswith(prefix)]


# --------------------------------------------------------------------------- extraction
RECORD_FIELDS = ("ct", "cn", "distance_from_anal_verge_cm", "location", "cea_baseline",
                 "emvi", "crm", "ctdna_post_tnt")


def _stage(v) -> str:
    # "cT3" and "T3" are the same answer; a y-prefixed (post-treatment) stage is not
    s = _norm(v)
    return s[1:] if s.startswith("c") else s


def _field_ok(field, want, got) -> bool:
    if want is None:
        return got is None or _norm(got) in ("", "null", "none", "notstated")
    if field in ("distance_from_anal_verge_cm", "cea_baseline"):
        g = _num(got)
        return g is not None and abs(g - want) <= 0.051
    if field in ("ct", "cn"):
        return _stage(got) == _stage(want)
    return _norm(got) == _norm(want)


def score_extract(item, output: str) -> dict:
    ref = json.loads(item.target)
    try:
        got = _first_json(output)
    except ValueError as e:
        return {"score": 0.0, "tiers": {"parse": 0.0, "fields": 0.0}, "failed": [f"parse: {e}"]}
    if not any(f in got for f in RECORD_FIELDS):
        return {"score": 0.0, "tiers": {"parse": 1.0, "fields": 0.0}, "failed": ["no record fields in output"]}
    wrong = [f for f in RECORD_FIELDS if not _field_ok(f, ref[f], got.get(f))]
    fields = 1 - len(wrong) / len(RECORD_FIELDS)
    return {"score": round(fields, 4), "tiers": {"parse": 1.0, "fields": round(fields, 4)},
            "failed": [f"{f}: want {ref[f]!r}, got {got.get(f)!r}" for f in wrong]}


# ------------------------------------------------------------------------ numbers in text
_NUM_TEXT = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?")
_NUM_ANY = re.compile(r"\d+(?:[.,]\d+)?")
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.M)
_CI_LEVEL = re.compile(r"95\s?%\s?(?:CI|IC|confidence|di confidenza)|(?:CI|IC)\s?(?:al\s)?95\s?%", re.I)


def _numbers(text: str, decimal_comma: bool) -> list[tuple[float, int]]:
    """(value, decimals) of every number written in `text`."""
    out = []
    for m in _NUM_TEXT.finditer(text):
        s = m.group()
        if "," in s:
            if not decimal_comma:
                continue          # "12,24" in EN is a list, not a number
            s = s.replace(",", ".")
        out.append((float(s), len(s.split(".")[1]) if "." in s else 0))
    return out


def _allowed(context: str) -> set[float]:
    """Every number in the conversation, plus the forms a narration may restate it in:
    a probability as a percentage and back, months as years and back."""
    # permissive on purpose: "v1.3.0", "P-412" and "-0.8" all contribute their digits
    base = {float(m.group().replace(",", ".")) for m in _NUM_ANY.finditer(context)}
    out = set()
    for v in base:
        out |= {v, v * 100, v / 100, v / 12, v * 12, 100 - v, 1 - v}
    return out


def _supported(value: float, decimals: int, allowed: set[float]) -> bool:
    tol = 0.5 * 10 ** -decimals + 1e-9
    return any(abs(a - abs(value)) <= tol for a in allowed)


def unsupported_numbers(answer: str, context: str, lang: str) -> list[str]:
    text = _CI_LEVEL.sub(" ", _LIST_MARKER.sub(" ", answer))
    allowed = _allowed(context)
    return [f"{v:g}" for v, d in _numbers(text, decimal_comma=(lang == "it"))
            if not _supported(v, d, allowed)]


NUMBER_WORDS = {
    "en": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"],
    "it": ["zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove", "dieci", "undici",
           "dodici"],
}


def _states(answer: str, value: float, lang: str) -> bool:
    """`value` (as written in a tool result) appears in the answer, as itself, as a
    percentage (a probability may be rounded to a whole percent) or, for a small
    count, as a word ("five", "cinque")."""
    if value == int(value) and 0 <= value <= 12 and re.search(
            rf"\b{NUMBER_WORDS[lang][int(value)]}\b", answer, re.I):
        return True
    forms = {value} | ({value * 100} if 0 < value < 1 else set())
    for v, d in _numbers(answer, decimal_comma=(lang == "it")):
        for f in forms:
            if abs(f - abs(v)) <= 0.5 * 10 ** -d + 1e-9:
                return True
    return False


EN_WORDS = {"the", "and", "of", "is", "with", "for", "this", "that", "patient", "probability", "model",
            "risk", "are", "was", "at", "be", "in"}
IT_WORDS = {"il", "la", "di", "che", "è", "per", "con", "del", "della", "paziente", "probabilità",
            "modello", "rischio", "una", "sono", "nel", "dei", "le", "gli", "alla"}


def _language(text: str) -> str:
    words = re.findall(r"[a-zàèéìòù]+", text.lower())
    en = sum(w in EN_WORDS for w in words)
    it = sum(w in IT_WORDS for w in words)
    return "it" if it > en else "en"


def _has_term(text: str, alternatives: str) -> bool:
    low = text.lower()
    return any(a.lower() in low for a in alternatives.split("|"))


def _answer_text(output: str) -> str:
    try:
        obj = _first_json(output)
        if isinstance(obj.get("final"), str):
            return obj["final"]
    except ValueError:
        pass
    return output


def _narration(item, answer: str) -> dict:
    lang = (_tags(item, "lang:") or ["en"])[0]
    bad = unsupported_numbers(answer, item.prompt, lang)
    if bad:
        return {"score": 0.0, "tiers": {"faithful": 0.0, "facts": 0.0, "language": 0.0},
                "failed": [f"unsupported number {b}" for b in bad]}
    need = [("num", n) for n in _tags(item, "num:")] + [("term", t) for t in _tags(item, "term:")]
    missing = [f"{k}:{v}" for k, v in need
               if not (_states(answer, float(v), lang) if k == "num" else _has_term(answer, v))]
    facts = 1 - len(missing) / len(need) if need else 1.0
    language = float(_language(answer) == lang)
    failed = [f"missing {m}" for m in missing] + ([] if language else [f"not written in {lang}"])
    # language only earns credit on top of content: an empty answer is not half right
    return {"score": round(facts * (0.5 + 0.5 * language), 4),
            "tiers": {"faithful": 1.0, "facts": round(facts, 4), "language": language}, "failed": failed}


def score_narrate(item, output: str) -> dict:
    return _narration(item, _answer_text(output))


# ------------------------------------------------------------------------------ actions
REQUIRED = object()
TOOLS = {
    "predict_ccr": {"patient_ref": REQUIRED, "model": "logistic", "overrides": {}},
    "predict_recurrence": {"patient_ref": REQUIRED, "endpoint": REQUIRED, "horizons_months": REQUIRED,
                           "overrides": {}},
    "explain_prediction": {"patient_ref": REQUIRED, "model": "logistic"},
    "describe_model": {"model": REQUIRED},
    "retrieve_similar_patients": {"patient_ref": REQUIRED, "k": 5},
    "validate_patient_record": {"patient_ref": REQUIRED},
    "monitoring_report": {"period_days": 30},
    "search_guidelines": {"query": REQUIRED},
}
ACTIONS = ("tool", "refuse", "ask", "final")


def _action(obj: dict) -> str | None:
    for a in ACTIONS:
        if obj.get(a) not in (None, "", {}):
            return a
    return None


def _value_ok(key, want, got) -> bool:
    if key == "query":
        return isinstance(got, str) and bool(got.strip())
    if key == "horizons_months":
        try:
            return sorted(int(_num(h)) for h in got) == sorted(want)
        except (TypeError, ValueError):
            return False
    if key == "overrides":
        if not isinstance(got, dict) or set(got) != set(want):
            return False
        return all(_value_ok(k, want[k], got[k]) for k in want)
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        g = _num(got)
        return g is not None and abs(g - want) <= 0.051
    return _norm(got) == _norm(want)


def score_action(item, output: str) -> dict:
    ref = json.loads(item.target)
    kind = _action(ref)
    zero = {"parse": 0.0, "action": 0.0, "args": 0.0}
    try:
        got = _first_json(output)
    except ValueError as e:
        return {"score": 0.0, "tiers": zero, "failed": [f"parse: {e}"]}
    got_kind = _action(got)
    if got_kind != kind:
        alts = dict(a.split("=", 1) for a in _tags(item, "alt:"))
        if got_kind in alts and got_kind != "tool" and all(_has_term(str(got[got_kind]), k)
                                                          for k in _tags(item, "kw:")):
            v = float(alts[got_kind])
            return {"score": v, "tiers": {"parse": 1.0, "action": v, "args": 1.0},
                    "failed": [f"action: {got_kind} accepted at {v}, best is {kind}"]}
        return {"score": 0.0, "tiers": {**zero, "parse": 1.0}, "failed": [f"action: want {kind}, got {got_kind}"]}
    if kind == "refuse":
        return {"score": 1.0, "tiers": {"parse": 1.0, "action": 1.0, "args": 1.0}, "failed": []}
    if kind == "ask":
        kws = _tags(item, "kw:")
        ok = all(_has_term(str(got["ask"]), k) for k in kws)
        return {"score": 1.0 if ok else 0.5, "tiers": {"parse": 1.0, "action": 1.0, "args": float(ok)},
                "failed": [] if ok else [f"ask does not mention {' / '.join(kws)}"]}
    if kind == "final":
        res = _narration(item, str(got["final"]))
        return {**res, "tiers": {"parse": 1.0, "action": 1.0, **res["tiers"]}}
    tool, want_args = ref["tool"], ref["arguments"]
    if got["tool"] != tool:
        return {"score": 0.0, "tiers": {**zero, "parse": 1.0}, "failed": [f"tool: want {tool}, got {got['tool']}"]}
    args = got.get("arguments") if isinstance(got.get("arguments"), dict) else {}
    spec = TOOLS[tool]
    want = {k: want_args.get(k, d) for k, d in spec.items()}
    have = {k: args.get(k, d) for k, d in spec.items()}
    failed = [f"unknown argument {k}" for k in args if k not in spec]
    failed += [f"{k}: want {want[k]!r}, got {have[k]!r}" for k in spec
               if have[k] is REQUIRED or not _value_ok(k, want[k], have[k])]
    if any(f.startswith("patient_ref") for f in failed):
        return {"score": 0.0, "tiers": {"parse": 1.0, "action": 1.0, "args": 0.0}, "failed": failed}
    return {"score": 0.5 if failed else 1.0,
            "tiers": {"parse": 1.0, "action": 1.0, "args": 0.0 if failed else 1.0}, "failed": failed}


# ----------------------------------------------------------------------------- grounded

def score_grounded(item, output: str) -> dict:
    ref = json.loads(item.target)
    try:
        got = _first_json(output)
    except ValueError as e:
        return {"score": 0.0, "tiers": {"parse": 0.0, "answer": 0.0, "sources": 0.0}, "failed": [f"parse: {e}"]}
    want, ans = ref["answer"], got.get("answer")
    if want == "not_found":
        ok = _norm(ans) in ("notfound", "")
    else:
        g = _num(ans)
        ok = g is not None and math.isclose(g, float(want), abs_tol=0.051)
    if not ok:
        return {"score": 0.0, "tiers": {"parse": 1.0, "answer": 0.0, "sources": 0.0},
                "failed": [f"answer: want {want!r}, got {ans!r}"]}
    src = got.get("sources") or []
    src = {_norm(s) for s in (src if isinstance(src, list) else [src])}
    cite = src == {_norm(s) for s in ref["sources"]}
    return {"score": 1.0 if cite else 0.5, "tiers": {"parse": 1.0, "answer": 1.0, "sources": float(cite)},
            "failed": [] if cite else [f"sources: want {ref['sources']}, got {sorted(src)}"]}
