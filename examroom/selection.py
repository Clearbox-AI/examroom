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
from dataclasses import dataclass, field

from examroom.diff import wilson
from examroom.items import Suite
from examroom.ledger import Ledger
from examroom.profiles import get_profile
from examroom.registry import SIZE_KEYS, TIERS, Registry, fmt_size, parse_count, parse_size  # noqa: F401 (re-exported)

Z = 1.96
TIER_NAMES = [t[0] for t in TIERS]
_EPS = 1e-9  # means of 0/1 scores are floats: 0.9 must clear a 0.9 gate


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
    size_source: str | None = None   # registry | run | estimate
    tiers: list[str] = field(default_factory=list)
    profile: str | None = None


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
    filtered_runs: int = 0               # outside the requested tier or profile
    tier: str | None = None
    profile: str | None = None
    unmatched_patterns: list[str] = field(default_factory=list)
    unreachable: list[tuple[str, int, float, float]] = field(default_factory=list)  # task, n, max lower, min

    @property
    def gated(self) -> bool:
        return any(m is not None for m in self.minimums.values())

    @property
    def best(self) -> Candidate | None:
        return next((c for c in self.candidates if c.passed and c.size is not None), None)


def select(ledger: Ledger, suite: Suite, criteria: list[tuple[str, float]],
           gate: str = "lower", size_key: str = "n_params", registry: Registry | None = None,
           tier: str | None = None, profile: str | None = None) -> Selection:
    """`registry` supplies model sizes at read time (registry facts beat what a run
    recorded); `tier` and `profile` restrict the candidates, as a leaderboard column does."""
    if gate not in ("lower", "point"):
        raise ValueError("gate must be 'lower' or 'point'")
    if size_key not in SIZE_KEYS:
        raise ValueError(f"size key must be one of {', '.join(SIZE_KEYS)}")
    h = suite.content_hash()
    tasks = [t.name for t in suite.tasks]
    minimums = {t: minimum_for(t, criteria) for t in tasks}
    registry = registry or Registry()
    if tier is not None and tier not in TIER_NAMES:
        raise ValueError(f"tier must be one of {', '.join(TIER_NAMES)}")
    # a bare name means the current version: runs under an older version are not comparable
    want_profile = profile if not profile or "@" in profile else get_profile(profile).id
    sel = Selection(suite.name, suite.version, h, size_key, gate, tasks, minimums, [], tier=tier, profile=profile,
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
        backend = json.loads(row["backend_json"])
        ms = registry.size(backend)
        run_profile = backend.get("profile")
        if (tier and tier not in ms.tiers) or (want_profile and run_profile != want_profile):
            sel.filtered_runs += 1
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
        sel.candidates.append(Candidate(row["run_id"], row["model"], row["label"], ms.precision,
                                        getattr(ms, size_key), cells, all(gates) if gates else None,
                                        ms.weights_source if size_key == "weights_mb" else None,
                                        ms.tiers, run_profile))
    sel.candidates.sort(key=lambda c: (c.size is None, c.size or 0,
                                       -sum(x.score or 0 for x in c.cells.values())))
    return sel


def _size(c: Candidate, key: str) -> str:
    return fmt_size(c.size, key) + ("~" if c.size_source == "estimate" else "")


def render(sel: Selection) -> str:
    what = "95% Wilson lower bound" if sel.gate == "lower" else "score"
    out = [f"{sel.suite}@{sel.version}  hash {sel.suite_hash[:8]}  {len(sel.candidates)} runs  "
           f"gate: {what} >= min  size: {sel.size_key}"]
    if sel.tier or sel.profile:
        out.append("restricted to " + "  ".join(x for x in (sel.tier and f"tier={sel.tier}",
                                                         sel.profile and f"profile={sel.profile}") if x)
                   + f"  ({sel.filtered_runs} other runs left out)")
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
    out.append(f"{'size':{tw}s}  {'':>5s} {'':>4s} " + "".join(f"{_size(c, sel.size_key):>6s} " for c in cols))
    out.append("")
    for i, c in enumerate(cols, 1):
        extra = "  ".join(x for x in (c.label, c.quantization and f"precision={c.quantization}",
                                      c.tiers and "tiers=" + ",".join(c.tiers), c.profile and f"profile={c.profile}") if x)
        out.append(f"[{i}] {c.run_id}  {c.model}  {sel.size_key}={_size(c, sel.size_key)}  {extra}".rstrip())
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
        out.append(f"smallest passing: [{cols.index(b) + 1}] {b.model}  {sel.size_key}={_size(b, sel.size_key)}"
                   f"  (run {b.run_id})")
    else:
        unsized = [cols.index(c) + 1 for c in cols if c.passed]
        out.append("no run passes every gate" if not unsized else
                   f"passing runs have no known {sel.size_key}: {unsized}; register the model "
                   f"(`examroom models add`) or `examroom annotate <run_id> {sel.size_key}=…`")
    if any(c.size_source == "estimate" for c in cols):
        out.append("~ weights estimated from parameters and precision (8 bits/weight when the precision is unknown)")
    return "\n".join(out)
