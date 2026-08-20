from __future__ import annotations

import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Callable

from benchsmith.backends.base import Backend
from benchsmith.items import Suite
from benchsmith.scoring import score_item


@dataclass
class ItemResult:
    task: str
    item_id: str
    score: float
    protocol: str
    detail: dict
    seconds: float


@dataclass
class RunResult:
    run_id: str
    ts: str
    suite: str
    suite_version: str
    suite_hash: str
    backend: dict
    label: str | None
    git_commit: str | None
    stats: dict = field(default_factory=dict)
    results: list[ItemResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        return sum(r.score for r in self.results) / len(self.results) if self.results else float("nan")

    def task_scores(self) -> dict[str, tuple[float, int]]:
        acc: dict[str, list[float]] = {}
        for r in self.results:
            acc.setdefault(r.task, []).append(r.score)
        return {t: (sum(v) / len(v), len(v)) for t, v in acc.items()}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def run_suite(
    suite: Suite,
    backend: Backend,
    tasks: list[str] | None = None,
    limit: int | None = None,
    label: str | None = None,
    progress: Callable[[int, int, ItemResult], None] | None = None,
) -> RunResult:
    run = RunResult(
        run_id=uuid.uuid4().hex[:12],
        ts=datetime.now(UTC).isoformat(timespec="seconds"),
        suite=suite.name, suite_version=suite.version, suite_hash=suite.content_hash(),
        backend=backend.info.as_dict(), label=label, git_commit=_git_commit(),
    )
    todo = [(t, it) for t, it in suite.iter_items() if tasks is None or t.name in tasks]
    if limit:
        per_task: dict[str, int] = {}
        kept = []
        for t, it in todo:
            if per_task.get(t.name, 0) < limit:
                kept.append((t, it))
                per_task[t.name] = per_task.get(t.name, 0) + 1
        todo = kept
    for i, (t, it) in enumerate(todo, 1):
        t0 = time.perf_counter()
        score, detail, protocol = score_item(it, t, backend)
        r = ItemResult(t.name, it.id, score, str(protocol), detail, time.perf_counter() - t0)
        run.results.append(r)
        if progress:
            progress(i, len(todo), r)
    run.stats = backend.stats.finalize()
    return run
