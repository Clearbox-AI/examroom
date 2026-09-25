"""Streaming n-gram leakage check: benchmark items vs a (large) training corpus.

Indexes every word-level n-gram of the benchmark (small), then streams the
corpus documents and reports any document sharing an n-gram with an item.
Memory is bounded by the benchmark, not the corpus, so it works on ~1B-token
shards. Supports .txt/.md/.jsonl/.json and .parquet (text column) files.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from examroom.items import Suite

_WORD = re.compile(r"\w+")


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


def ngram_hashes(words: list[str], n: int) -> Iterator[tuple[int, int]]:
    for i in range(len(words) - n + 1):
        yield hash(tuple(words[i:i + n])), i


def item_texts(it) -> list[str]:
    """The strings of an item that must not appear in training data."""
    texts = [it.prompt]
    if it.options:
        texts += [it.prompt + o for o in it.options]
    if it.good:
        texts += [it.prompt + it.good, it.prompt + it.bad]
    # `target` of program items is a reference document the model never sees; it is
    # covered by the exact-document check, not by n-grams of its boilerplate.
    return texts


def build_index(suite: Suite, n: int) -> tuple[dict[int, set[str]], int, int]:
    """n-gram hash -> item ids. Returns (index, n_items_indexed, n_items_too_short)."""
    index: dict[int, set[str]] = defaultdict(set)
    indexed = short = 0
    for _, it in suite.iter_items():
        any_ngram = False
        for t in item_texts(it):
            for h, _ in ngram_hashes(tokens(t), n):
                index[h].add(it.id)
                any_ngram = True
        if any_ngram:
            indexed += 1
        else:
            short += 1
    return index, indexed, short


def iter_corpus(paths: Iterable[Path], text_column: str = "text") -> Iterator[tuple[str, str]]:
    """Yield (doc_ref, text) for every document in the given files/directories."""
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        files += sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [p]
    for f in files:
        if f.suffix == ".parquet":
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(f)
            row = 0
            for batch in pf.iter_batches(columns=[text_column], batch_size=2048):
                for t in batch.column(0).to_pylist():
                    yield f"{f.name}:{row}", t or ""
                    row += 1
        elif f.suffix == ".jsonl":
            with f.open() as fh:
                for i, line in enumerate(fh):
                    if line.strip():
                        d = json.loads(line)
                        yield f"{f.name}:{i}", d.get(text_column) or d.get("prompt") or json.dumps(d)
        elif f.suffix in (".txt", ".md", ".json"):
            yield f.name, f.read_text(errors="ignore")


@dataclass
class LeakReport:
    n: int
    items_indexed: int
    items_too_short: int
    docs_scanned: int = 0
    collisions: dict[str, list[tuple[str, str]]] = field(default_factory=lambda: defaultdict(list))  # item -> [(doc, ngram)]

    @property
    def ok(self) -> bool:
        return not self.collisions


def _norm(text: str) -> str:
    # drop a trailing prose paragraph (the generator's summary) and normalise whitespace
    body = text.strip().split("\n\n")[0]
    return "\n".join(ln.rstrip() for ln in body.split("\n"))


def check_exact(suite: Suite, corpus: Iterable[Path], text_column: str = "text", progress=None) -> LeakReport:
    """Document-identity check for generated/structured suites: flags a corpus doc that
    equals an item's reference (`target`) or starts with an item's prompt prefix."""
    targets: dict[str, str] = {}
    prefixes: dict[int, dict[str, str]] = defaultdict(dict)   # len -> {prefix: item}
    for _, it in suite.iter_items():
        if it.target:
            targets[_norm(it.target)] = it.id
        prefixes[len(it.prompt)][it.prompt] = it.id
    rep = LeakReport(0, len(suite), 0)
    lens = sorted(prefixes)
    for ref, text in iter_corpus(corpus, text_column):
        rep.docs_scanned += 1
        nt = _norm(text)
        if nt in targets:
            rep.collisions[targets[nt]].append((ref, "identical document"))
        for L in lens:
            hit = prefixes[L].get(text[:L])
            if hit:
                rep.collisions[hit].append((ref, f"shares the {L}-char prompt prefix"))
        if progress and rep.docs_scanned % 10000 == 0:
            progress(rep.docs_scanned, len(rep.collisions))
    return rep


def check_leakage(suite: Suite, corpus: Iterable[Path], n: int = 13, text_column: str = "text",
                  progress=None, mode: str | None = None) -> LeakReport:
    mode = mode or suite.meta.meta.get("leakage_check", "ngram")
    if mode == "exact":
        return check_exact(suite, corpus, text_column, progress)
    index, indexed, short = build_index(suite, n)
    rep = LeakReport(n, indexed, short)
    for ref, text in iter_corpus(corpus, text_column):
        rep.docs_scanned += 1
        words = tokens(text)
        if len(words) >= n:
            for h, i in ngram_hashes(words, n):
                if h in index:
                    gram = " ".join(words[i:i + n])
                    for iid in index[h]:
                        if len(rep.collisions[iid]) < 5:
                            rep.collisions[iid].append((ref, gram))
        if progress and rep.docs_scanned % 10000 == 0:
            progress(rep.docs_scanned, len(rep.collisions))
    return rep
