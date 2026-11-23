"""Portable run records: the unit a results repo stores and the leaderboard reads.

`examroom export` writes one JSON file per run, <out>/<suite>@<version>/<run_id>.json,
so runs made on different machines can be collected in one place without merging
SQLite files.  Runs on a held-out suite (name ending in "-heldout") are exported
as per-task aggregates only: item ids, outputs and per-item scores would reveal
the items, which defeats holding them out.
"""

from __future__ import annotations

import json
from pathlib import Path

from examroom.ledger import Ledger
from examroom.splits import is_heldout

FORMAT = "examroom-run/1"


def export_run(ledger: Ledger, run_id: str) -> dict:
    row = ledger.run(run_id)
    rid = row["run_id"]
    heldout = is_heldout(row["suite"])
    rec = {
        "format": FORMAT,
        "run_id": rid, "ts": row["ts"], "suite": row["suite"], "suite_version": row["suite_version"],
        "suite_hash": row["suite_hash"], "label": row["label"], "git_commit": row["git_commit"],
        "backend": json.loads(row["backend_json"]), "stats": json.loads(row["stats_json"]),
        "n_items": row["n_items"], "score": row["score"],
        "tasks": {t: {"score": s, "n": n} for t, (s, n) in ledger.task_scores(rid).items()},
        "metrics": {k: {"value": v, "detail": d} for k, (v, d) in ledger.metrics(rid).items()},
        "heldout": heldout,
    }
    if not heldout:
        rec["results"] = [{"task": r.task, "item_id": r.item_id, "score": r.score, "protocol": r.protocol,
                           "seconds": r.seconds, "detail": r.detail}
                          for r in ledger.results(rid).values()]
    return rec


def write_exports(ledger: Ledger, run_ids: list[str], out_dir: str | Path) -> list[Path]:
    paths = []
    for run_id in run_ids:
        rec = export_run(ledger, run_id)
        p = Path(out_dir) / f"{rec['suite']}@{rec['suite_version']}" / f"{rec['run_id']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
        paths.append(p)
    return paths


def load_exports(results_dir: str | Path) -> list[dict]:
    recs = []
    for p in sorted(Path(results_dir).rglob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and rec.get("format") == FORMAT:
            recs.append(rec)
    return recs
