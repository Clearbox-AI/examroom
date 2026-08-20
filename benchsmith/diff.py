"""Compare two runs item by item, with honest uncertainty."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from benchsmith.ledger import Ledger


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant counts b (A right, B wrong) and c (A wrong, B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


@dataclass
class TaskDiff:
    task: str
    n: int
    score_a: float
    score_b: float
    fixed: list[str] = field(default_factory=list)    # wrong in A, right in B
    broken: list[str] = field(default_factory=list)   # right in A, wrong in B
    p_value: float = 1.0

    @property
    def delta(self) -> float:
        return self.score_b - self.score_a


@dataclass
class RunDiff:
    run_a: str
    run_b: str
    same_benchmark: bool
    tasks: list[TaskDiff]

    @property
    def total_fixed(self) -> int:
        return sum(len(t.fixed) for t in self.tasks)

    @property
    def total_broken(self) -> int:
        return sum(len(t.broken) for t in self.tasks)


def diff_runs(ledger: Ledger, run_a: str, run_b: str) -> RunDiff:
    ra, rb = ledger.run(run_a), ledger.run(run_b)
    res_a, res_b = ledger.results(ra["run_id"]), ledger.results(rb["run_id"])
    common = sorted(set(res_a) & set(res_b))
    by_task: dict[str, list[str]] = {}
    for iid in common:
        by_task.setdefault(res_a[iid].task, []).append(iid)
    tasks = []
    for task, ids in sorted(by_task.items()):
        td = TaskDiff(task, len(ids),
                      sum(res_a[i].score for i in ids) / len(ids),
                      sum(res_b[i].score for i in ids) / len(ids))
        for i in ids:
            a_ok, b_ok = res_a[i].score >= 0.5, res_b[i].score >= 0.5
            if not a_ok and b_ok:
                td.fixed.append(i)
            elif a_ok and not b_ok:
                td.broken.append(i)
        td.p_value = mcnemar_exact_p(len(td.broken), len(td.fixed))
        tasks.append(td)
    return RunDiff(ra["run_id"], rb["run_id"], ra["suite_hash"] == rb["suite_hash"], tasks)
