from __future__ import annotations

import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Callable

from examroom.backends.base import Backend
from examroom.items import Suite
from examroom.scoring import load_checker, score_item


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
    overrides: dict | None = None,
    workers: int = 1,
) -> RunResult:
    """Evaluate `suite` on `backend`.

    `overrides` (e.g. {"max_tokens": 512}) are merged into every task's scoring
    config for this run only; they are recorded in the run metadata and do NOT
    affect the suite content hash.

    `workers` > 1 scores items concurrently in threads (item order in the results
    is preserved). Useful for API backends; a backend can opt out by setting
    `parallel_safe = False` (the hf backend does — the GPU is the bottleneck).
    """
    run = RunResult(
        run_id=uuid.uuid4().hex[:12],
        ts=datetime.now(UTC).isoformat(timespec="seconds"),
        suite=suite.name, suite_version=suite.version, suite_hash=suite.content_hash(),
        backend=backend.info.as_dict(), label=label, git_commit=_git_commit(),
    )
    if overrides:
        run.backend["overrides"] = dict(overrides)
        tasks_eff = {t.name: t.model_copy(update={"scoring": {**t.scoring, **overrides}}) for t in suite.tasks}
    else:
        tasks_eff = {t.name: t for t in suite.tasks}
    todo = [(t, it) for t, it in suite.iter_items() if tasks is None or t.name in tasks]
    if limit:
        per_task: dict[str, int] = {}
        kept = []
        for t, it in todo:
            if per_task.get(t.name, 0) < limit:
                kept.append((t, it))
                per_task[t.name] = per_task.get(t.name, 0) + 1
        todo = kept
    # preflight: resolve every checker (and run its environment probe, if it declares
    # one) before the first model call, so a missing checker file or library fails
    # here instead of after a paid run has spent its budget on the other tasks
    for spec in sorted({it.checker for _, it in todo if it.checker}):
        hook = getattr(load_checker(spec, suite.root), "preflight", None)
        if hook is not None:
            hook()

    def one(t, it) -> ItemResult:
        t0 = time.perf_counter()
        score, detail, protocol = score_item(it, tasks_eff[t.name], backend, root=suite.root)
        return ItemResult(t.name, it.id, score, str(protocol), detail, time.perf_counter() - t0)

    workers = min(workers, len(todo)) if getattr(backend, "parallel_safe", True) else 1
    if workers <= 1:
        for i, (t, it) in enumerate(todo, 1):
            r = one(t, it)
            run.results.append(r)
            if progress:
                progress(i, len(todo), r)
    else:
        from concurrent.futures import ThreadPoolExecutor

        results: list[ItemResult | None] = [None] * len(todo)
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(one, t, it): idx for idx, (t, it) in enumerate(todo)}
            from concurrent.futures import as_completed

            for f in as_completed(futs):
                results[futs[f]] = f.result()
                done += 1
                if progress:
                    progress(done, len(todo), results[futs[f]])
        run.results.extend(results)  # item order preserved regardless of completion order
    run.stats = backend.stats.finalize()
    return run
