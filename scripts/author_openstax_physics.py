"""Author a numeric suite from OpenStax College Physics 2e solved exercises.

Problems come from the book's ``<n>-problems-exercises`` HTML page, answers
from the ``chapter-<n>`` answer-key page (odd-numbered problems only), joined
by problem number.  Only answers that parse as a physical quantity become
items; problems that depend on a figure, a sketch, or a graph are skipped.

    python scripts/author_openstax_physics.py --chapter 2

License: the book is CC BY-NC-SA 4.0 (OpenStax, Rice University); the suite
carries the attribution in its metadata.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

from benchsmith.authoring.openstax import (
    Exercise,
    fetch_page,
    parse_exercises,
    parse_quantity,
    parse_solutions,
    sigfig_tolerance,
    split_parts,
)
from benchsmith.items import Item, Kind, Provenance, SuiteMeta, Task, save_suite

BOOK = "college-physics-2e"
AUTHOR = "scripts/author_openstax_physics.py"
LICENSE = "CC BY-NC-SA 4.0"
ATTRIBUTION = (
    "Problems and answers derived from OpenStax College Physics 2e, "
    "openstax.org/details/books/college-physics-2e, © Rice University, CC BY-NC-SA 4.0."
)

# problems whose statement needs an artifact the model cannot see
_VISUAL_RE = re.compile(r"\b(sketch|graph|plot|draw|diagram|photograph|shown below)\b", re.I)


def build_items(chapter: int, exercises: list[Exercise], solutions: dict[int, str]) -> list[Item]:
    items: list[Item] = []
    for ex in exercises:
        answer = solutions.get(ex.number)
        if answer is None or ex.has_figure or _VISUAL_RE.search(ex.text):
            continue
        parts = split_parts(answer)
        multi = list(parts) != [""]
        for part, part_answer in parts.items():
            q = parse_quantity(part_answer)
            if q is None:
                continue
            if multi and f"({part})" not in ex.text:
                continue  # answer has parts the problem statement does not label
            if multi:
                ask = f"The answer to part ({part})"
            else:
                ask = "The answer"
            ask += f", in {q.unit}, is" if q.unit else " is the number"
            items.append(
                Item(
                    id=f"osp2e-c{chapter:02d}-{ex.number:03d}{part}",
                    kind=Kind.NUMERIC,
                    prompt=f"{ex.text}\n\n{ask}",
                    target=repr(q.value),
                    tolerance=sigfig_tolerance(q),
                    category=ex.section or f"chapter {chapter}",
                    tags=(["multipart"] if multi else []) + (["extra-answer-text"] if q.rest else []),
                    notes=f"book answer: {part_answer}",
                    provenance=Provenance(
                        method="book-html",
                        source=f"openstax:{BOOK}/pages/{chapter}-problems-exercises#{ex.number}",
                        author=AUTHOR,
                    ),
                    max_tokens=24,
                )
            )
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter", type=int, required=True)
    ap.add_argument("--book", default=BOOK)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None, help="directory to cache fetched HTML")
    args = ap.parse_args()

    ch = args.chapter
    out = args.out or f"suites/openstax_physics_ch{ch}_v1"
    problems_slug = f"{ch}-problems-exercises"
    probs_html = fetch_page(args.book, problems_slug, args.cache)
    key_html = fetch_page(args.book, f"chapter-{ch}", args.cache)

    exercises = parse_exercises(probs_html)
    solutions = parse_solutions(key_html, problems_slug)
    items = build_items(ch, exercises, solutions)

    by_section: dict[str, int] = {}
    for it in items:
        by_section[it.category] = by_section.get(it.category, 0) + 1
    for sec, n in sorted(by_section.items()):
        print(f"{sec:60s} {n}")

    task = Task(
        name=f"ch{ch}",
        file=f"ch{ch}.jsonl",
        kind=Kind.NUMERIC,
        scoring={"temperature": 0.0},
        description=f"OpenStax College Physics 2e chapter {ch}: first number in the "
        "generation must be within a half-unit of the answer's last significant digit",
    )
    meta = SuiteMeta(
        name=f"openstax_physics_ch{ch}",
        version="1.0",
        description=f"Solved end-of-chapter exercises, OpenStax College Physics 2e ch. {ch}.",
        tasks=[task],
        meta={
            "book": args.book,
            "chapter": ch,
            "source_pages": [problems_slug, f"chapter-{ch}"],
            "fetched": dt.date.today().isoformat(),
            "license": LICENSE,
            "attribution": ATTRIBUTION,
            "exercises_on_page": len(exercises),
            "answered": len(solutions),
        },
    )
    save_suite(out, meta, {task.name: items})
    Path(out, "README.md").write_text(
        f"# openstax_physics_ch{ch}\n\n"
        f"{len(items)} numeric items from the solved (odd-numbered) end-of-chapter\n"
        f"exercises of OpenStax College Physics 2e, chapter {ch}.  Problems that\n"
        "depend on a figure, sketch, or graph are excluded, as are answers that do\n"
        "not parse as a single physical quantity.  Multi-part problems yield one\n"
        "item per part; the prompt names the part and the expected unit.\n\n"
        "Tolerance is half a unit in the last significant digit of the book answer.\n\n"
        f"{ATTRIBUTION}\n\n"
        f"Authored by `{AUTHOR}`; re-running it refetches the live pages and may\n"
        "differ if OpenStax edits the book — treat the frozen version as canonical.\n"
    )
    print(f"total {len(items)} -> {out}")


if __name__ == "__main__":
    main()
