"""Smallest good-enough model: per-task acceptance gates over the recorded runs of one suite.

"Good enough" is a minimum per task, not a mean over the suite: an average is
dominated by the easy, saturated tiers, while a deployment fails on its weakest
required capability.  Minimums come from the suite's `meta.acceptance`
({task_glob: minimum}), a criteria file, or `--min` flags; when several patterns
match a task the last one wins (like .gitignore), so a broad "*" can be refined.

By default a task passes when the 95% Wilson *lower bound* of its score clears
the minimum, so a small task cannot pass on luck: 12/12 only demonstrates ≥ 0.76.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field

from examroom.diff import wilson
from examroom.items import Suite
from examroom.ledger import Ledger

SIZE_KEYS = ("n_params", "active_params", "weights_mb")
Z = 1.96
_EPS = 1e-9  # means of 0/1 scores are floats: 0.9 must clear a 0.9 gate

_COUNT = re.compile(r"^\s*(\d*\.?\d+(?:e[+-]?\d+)?)\s*([kmbt]?)\s*$", re.I)
_MULT = {"": 1, "k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def parse_count(s) -> int:
    """'4B' -> 4_000_000_000; also '600M', '30.5b', '1.2e9', '4000'."""
    m = _COUNT.match(str(s))
    if not m:
        raise ValueError(f"not a parameter count: {s!r} (use e.g. 4B, 600M, 1.2e9)")
    return int(round(float(m.group(1)) * _MULT[m.group(2).lower()]))


def parse_size(key: str, value) -> int | float:
    if key == "weights_mb":
        return float(value)
    if key in ("n_params", "active_params"):
        return parse_count(value)
    raise ValueError(f"unknown size key {key!r}; expected one of {', '.join(SIZE_KEYS)}")


def fmt_size(v, key: str = "n_params") -> str:
    if v is None:
        return "?"
    if key == "weights_mb":
        return f"{v / 1024:.1f}GB" if v >= 1024 else f"{v:.0f}MB"
    for unit, d in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if v >= d:
            return f"{v / d:.3g}{unit}"
    return str(v)


def minimum_for(task: str, criteria: list[tuple[str, float]]) -> float | None:
    m = None
    for pat, v in criteria:
        if fnmatch.fnmatchcase(task, pat):
            m = v
    return m


def max_lower_bound(n: int) -> float:
    """Wilson lower bound of a perfect score: the most n items can demonstrate."""
    return n / (n + Z * Z) if n else 0.0


@dataclass
class Cell:
    score: float | None   # None: the run has no results for this task
    n: int
    lower: float | None
    ok: bool | None       # None: task not gated


@dataclass
class Candidate:
    run_id: str
    model: str
    label: str | None
    quantization: str | None
    size: float | None
    cells: dict[str, Cell]
    passed: bool | None   # None when nothing is gated


@dataclass
class Selection:
    suite: str
    version: str
    suite_hash: str
    size_key: str
    gate: str
    tasks: list[str]
    minimums: dict[str, float | None]
    candidates: list[Candidate]          # smallest first, unknown sizes last
    skipped_runs: int = 0                # same suite name, different content hash
    unmatched_patterns: list[str] = field(default_factory=list)
    unreachable: list[tuple[str, int, float, float]] = field(default_factory=list)  # task, n, max lower, min

    @property
    def gated(self) -> bool:
        return any(m is not None for m in self.minimums.values())

    @property
    def best(self) -> Candidate | None:
        return next((c for c in self.candidates if c.passed and c.size is not None), None)


def select(ledger: Ledger, suite: Suite, criteria: list[tuple[str, float]],
           gate: str = "lower", size_key: str = "n_params") -> Selection:
    if gate not in ("lower", "point"):
        raise ValueError("gate must be 'lower' or 'point'")
    if size_key not in SIZE_KEYS:
        raise ValueError(f"size key must be one of {', '.join(SIZE_KEYS)}")
    h = suite.content_hash()
    tasks = [t.name for t in suite.tasks]
    minimums = {t: minimum_for(t, criteria) for t in tasks}
    sel = Selection(suite.name, suite.version, h, size_key, gate, tasks, minimums, [],
                    unmatched_patterns=[p for p, _ in criteria if not any(fnmatch.fnmatchcase(t, p) for t in tasks)])
    if gate == "lower":
        for t in tasks:
            n = len(suite.items[t])
            if minimums[t] is not None and max_lower_bound(n) + _EPS < minimums[t]:
                sel.unreachable.append((t, n, max_lower_bound(n), minimums[t]))

    for row in ledger.runs(suite=suite.name):
        if row["suite_hash"] != h:
            sel.skipped_runs += 1
            continue
        scores = ledger.task_scores(row["run_id"])
        cells = {}
        for t in tasks:
            if t not in scores:
                cells[t] = Cell(None, 0, None, None if minimums[t] is None else False)
                continue
            s, n = scores[t]
            lo = wilson(s, n)[0]
            ok = None if minimums[t] is None else (lo if gate == "lower" else s) + _EPS >= minimums[t]
            cells[t] = Cell(s, n, lo, ok)
        gates = [c.ok for c in cells.values() if c.ok is not None]
        size = json.loads(row["backend_json"]).get(size_key)
        sel.candidates.append(Candidate(row["run_id"], row["model"], row["label"], row["quantization"],
                                        size, cells, all(gates) if gates else None))
    sel.candidates.sort(key=lambda c: (c.size is None, c.size or 0,
                                       -sum(x.score or 0 for x in c.cells.values())))
    return sel


def render(sel: Selection) -> str:
    what = "95% Wilson lower bound" if sel.gate == "lower" else "score"
    out = [f"{sel.suite}@{sel.version}  hash {sel.suite_hash[:8]}  {len(sel.candidates)} runs  "
           f"gate: {what} >= min  size: {sel.size_key}"]
    if sel.skipped_runs:
        out.append(f"({sel.skipped_runs} runs of other {sel.suite} content skipped: different suite hash)")
    if not sel.candidates:
        out.append("no recorded runs of this suite content")
        return "\n".join(out)
    cols = sel.candidates
    tw = max(len(t) for t in sel.tasks + ["verdict"])
    out.append("")
    out.append(f"{'task':{tw}s}  {'min':>5s} {'n':>4s} " + "".join(f"{f'[{i}]':>7s}" for i in range(1, len(cols) + 1)))
    for t in sel.tasks:
        m = sel.minimums[t]
        line = f"{t:{tw}s}  {(f'{m:.2f}' if m is not None else '-'):>5s} {max(c.cells[t].n for c in cols):4d} "
        for c in cols:
            x = c.cells[t]
            mark = "x" if x.ok is False else " "
            line += f"{'-' if x.score is None else f'{x.score:.2f}':>6s}{mark}"
        out.append(line)
    if sel.gated:
        out.append(f"{'verdict':{tw}s}  {'':>5s} {'':>4s} "
                   + "".join(f"{'PASS' if c.passed else 'fail':>6s} " for c in cols))
    out.append(f"{'size':{tw}s}  {'':>5s} {'':>4s} " + "".join(f"{fmt_size(c.size, sel.size_key):>6s} " for c in cols))
    out.append("")
    for i, c in enumerate(cols, 1):
        extra = "  ".join(x for x in (c.label, c.quantization and f"quant={c.quantization}") if x)
        out.append(f"[{i}] {c.run_id}  {c.model}  {sel.size_key}={fmt_size(c.size, sel.size_key)}  {extra}".rstrip())
    out.append("")
    for p in sel.unmatched_patterns:
        out.append(f"! pattern {p!r} matches no task of this suite")
    for t, n, lo, m in sel.unreachable:
        out.append(f"! {t}: {n} items can demonstrate at most {lo:.2f} (a perfect score), below the {m:.2f} "
                   f"minimum, so no run can pass; add items or use --gate point")
    if not sel.gated:
        out.append("no acceptance criteria: add meta.acceptance to suite.json or pass --min TASK_GLOB=MIN")
    elif sel.best:
        b = sel.best
        out.append(f"smallest passing: [{cols.index(b) + 1}] {b.model}  {sel.size_key}={fmt_size(b.size, sel.size_key)}"
                   f"  (run {b.run_id})")
    else:
        unsized = [cols.index(c) + 1 for c in cols if c.passed]
        out.append("no run passes every gate" if not unsized else
                   f"passing runs have no recorded {sel.size_key}: {unsized}; "
                   f"set it with `examroom annotate <run_id> {sel.size_key}=…`")
    return "\n".join(out)
