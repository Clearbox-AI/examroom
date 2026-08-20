"""Benchmark data model: Suite -> Task -> Item.

On disk a suite is a directory:

    suite.json        # name, version, description, tasks: [{name, file, kind, ...}]
    <task>.jsonl      # one Item per line
    README.md         # human description (optional)
    MANIFEST.json     # written by `benchsmith freeze` (counts, provenance, content hash)

The *kind* of an item decides how it is scored (see benchsmith.scoring), and is
the primary axis of the design: topics are just tags.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field, model_validator


class Kind(StrEnum):
    CHOICE_LOGLIK = "choice_loglik"      # closed set, length-normalised logprob per option
    PAIR_LOGLIK = "pair_loglik"          # good vs bad continuation
    CHOICE_GENERATE = "choice_generate"  # MCQ answered by letter, parsed from generation
    EXACT = "exact"                      # generated text == target (after normalisation)
    REGEX = "regex"                      # target is a regex the generation must match
    NUMERIC = "numeric"                  # first number in generation within tolerance
    PROGRAM = "program"                  # python checker "module:function(item, output) -> score"


LIKELIHOOD_KINDS = {Kind.CHOICE_LOGLIK, Kind.PAIR_LOGLIK}
GENERATIVE_KINDS = {Kind.CHOICE_GENERATE, Kind.EXACT, Kind.REGEX, Kind.NUMERIC, Kind.PROGRAM}


class Provenance(BaseModel):
    method: str = "hand"          # hand | template | generator | yourbench | llm
    source: str | None = None     # template id, document, generator script…
    seed: int | None = None
    author: str | None = None


class Item(BaseModel):
    id: str
    kind: Kind
    prompt: str
    # choice_* kinds
    options: list[str] | None = None
    answer_idx: int | None = None
    # pair_loglik
    good: str | None = None
    bad: str | None = None
    # exact / regex / numeric / program
    target: str | None = None
    tolerance: float | None = None
    checker: str | None = None
    # bookkeeping
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    provenance: Provenance = Field(default_factory=Provenance)
    # generation settings for generative kinds
    max_tokens: int = 32
    samples: int = 1

    @model_validator(mode="after")
    def _check_kind_fields(self) -> "Item":
        k = self.kind
        if k in (Kind.CHOICE_LOGLIK, Kind.CHOICE_GENERATE):
            if not self.options or len(self.options) < 2:
                raise ValueError(f"{self.id}: {k} needs >= 2 options")
            if self.answer_idx is None or not 0 <= self.answer_idx < len(self.options):
                raise ValueError(f"{self.id}: answer_idx out of range")
            if len(set(self.options)) != len(self.options):
                raise ValueError(f"{self.id}: duplicate options")
        elif k == Kind.PAIR_LOGLIK:
            if not self.good or not self.bad:
                raise ValueError(f"{self.id}: pair_loglik needs good and bad")
            if self.good == self.bad:
                raise ValueError(f"{self.id}: good == bad")
        elif k in (Kind.EXACT, Kind.REGEX, Kind.NUMERIC):
            if self.target is None:
                raise ValueError(f"{self.id}: {k} needs target")
        elif k == Kind.PROGRAM:
            if not self.checker:
                raise ValueError(f"{self.id}: program needs checker 'module:function'")
        return self

    @property
    def answer(self) -> str | None:
        if self.options is not None and self.answer_idx is not None:
            return self.options[self.answer_idx]
        return self.target

    def canonical(self) -> str:
        return self.model_dump_json(exclude_none=True, exclude={"notes"})


class Task(BaseModel):
    name: str
    file: str
    kind: Kind
    description: str | None = None
    # per-task scorer config, e.g. {"normalize": "mean"} or {"prefix": "Answer:"}
    scoring: dict[str, Any] = Field(default_factory=dict)


class SuiteMeta(BaseModel):
    name: str
    version: str
    description: str | None = None
    tasks: list[Task]


class Suite:
    def __init__(self, root: Path, meta: SuiteMeta, items: dict[str, list[Item]]):
        self.root = root
        self.meta = meta
        self.items = items  # task name -> items

    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def version(self) -> str:
        return self.meta.version

    @property
    def tasks(self) -> list[Task]:
        return self.meta.tasks

    def task(self, name: str) -> Task:
        for t in self.meta.tasks:
            if t.name == name:
                return t
        raise KeyError(name)

    def iter_items(self) -> Iterator[tuple[Task, Item]]:
        for t in self.meta.tasks:
            for it in self.items[t.name]:
                yield t, it

    def __len__(self) -> int:
        return sum(len(v) for v in self.items.values())

    def content_hash(self) -> str:
        """Stable hash of every item (sorted by id) + task config. Changes if the benchmark moves."""
        h = hashlib.sha256()
        for t in sorted(self.meta.tasks, key=lambda t: t.name):
            h.update(t.model_dump_json().encode())
            for it in sorted(self.items[t.name], key=lambda i: i.id):
                h.update(it.canonical().encode())
                h.update(b"\n")
        return h.hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON: {e}") from e
    return rows


def write_jsonl(path: Path, items: list[Item]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for it in items:
            f.write(it.model_dump_json(exclude_none=True) + "\n")


def load_suite(root: str | Path) -> Suite:
    root = Path(root)
    meta = SuiteMeta.model_validate_json((root / "suite.json").read_text())
    items: dict[str, list[Item]] = {}
    for t in meta.tasks:
        rows = read_jsonl(root / t.file)
        parsed = []
        for r in rows:
            r.setdefault("kind", t.kind)
            it = Item.model_validate(r)
            if it.kind != t.kind:
                raise ValueError(f"{t.file}: item {it.id} kind {it.kind} != task kind {t.kind}")
            parsed.append(it)
        items[t.name] = parsed
    return Suite(root, meta, items)


def save_suite(root: str | Path, meta: SuiteMeta, items: dict[str, list[Item]]) -> Suite:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "suite.json").write_text(meta.model_dump_json(indent=2, exclude_none=True) + "\n")
    for t in meta.tasks:
        write_jsonl(root / t.file, items[t.name])
    return Suite(root, meta, items)
