"""SQLite results ledger: one row per run, one row per (run, item)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from examroom.runner import ItemResult, RunResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    ts            TEXT NOT NULL,
    suite         TEXT NOT NULL,
    suite_version TEXT NOT NULL,
    suite_hash    TEXT NOT NULL,
    backend       TEXT NOT NULL,
    model         TEXT NOT NULL,
    dtype         TEXT,
    quantization  TEXT,
    device        TEXT,
    n_params      INTEGER,
    label         TEXT,
    git_commit    TEXT,
    backend_json  TEXT NOT NULL,
    stats_json    TEXT NOT NULL,
    n_items       INTEGER NOT NULL,
    score         REAL
);
CREATE TABLE IF NOT EXISTS results (
    run_id   TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    task     TEXT NOT NULL,
    item_id  TEXT NOT NULL,
    score    REAL NOT NULL,
    protocol TEXT NOT NULL,
    detail   TEXT,
    seconds  REAL,
    PRIMARY KEY (run_id, item_id)
);
CREATE INDEX IF NOT EXISTS results_item ON results(item_id);
CREATE TABLE IF NOT EXISTS metrics (
    run_id  TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    name    TEXT NOT NULL,
    value   REAL,
    detail  TEXT,
    PRIMARY KEY (run_id, name)
);
CREATE VIEW IF NOT EXISTS task_scores AS
    SELECT r.run_id, r.task, AVG(r.score) AS score, COUNT(*) AS n
    FROM results r GROUP BY r.run_id, r.task;
"""


class Ledger:
    def __init__(self, path: str | Path = "examroom.sqlite"):
        self.path = Path(path)
        self.con = sqlite3.connect(self.path)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys = ON")
        self.con.executescript(SCHEMA)

    def record(self, run: RunResult) -> None:
        b = run.backend
        with self.con:
            self.con.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run.run_id, run.ts, run.suite, run.suite_version, run.suite_hash,
                 b.get("backend"), b.get("model"), b.get("dtype"), b.get("quantization"),
                 b.get("device"), b.get("n_params"), run.label, run.git_commit,
                 json.dumps(b), json.dumps(run.stats), len(run.results), run.score),
            )
            self.con.executemany(
                "INSERT INTO results VALUES (?,?,?,?,?,?,?)",
                [(run.run_id, r.task, r.item_id, r.score, r.protocol, json.dumps(r.detail), r.seconds)
                 for r in run.results],
            )

    def runs(self, suite: str | None = None, model: str | None = None) -> list[sqlite3.Row]:
        q, args = "SELECT * FROM runs", []
        conds = []
        if suite:
            conds.append("suite = ?"); args.append(suite)
        if model:
            conds.append("model LIKE ?"); args.append(f"%{model}%")
        if conds:
            q += " WHERE " + " AND ".join(conds)
        return self.con.execute(q + " ORDER BY ts DESC", args).fetchall()

    def run(self, run_id: str) -> sqlite3.Row:
        row = self.con.execute("SELECT * FROM runs WHERE run_id LIKE ?", (run_id + "%",)).fetchall()
        if len(row) != 1:
            raise KeyError(f"run {run_id!r}: {len(row)} matches")
        return row[0]

    def results(self, run_id: str) -> dict[str, ItemResult]:
        rows = self.con.execute("SELECT * FROM results WHERE run_id = ?", (run_id,)).fetchall()
        return {r["item_id"]: ItemResult(r["task"], r["item_id"], r["score"], r["protocol"],
                                          json.loads(r["detail"] or "{}"), r["seconds"]) for r in rows}

    def task_scores(self, run_id: str) -> dict[str, tuple[float, int]]:
        rows = self.con.execute("SELECT task, score, n FROM task_scores WHERE run_id = ?", (run_id,))
        return {r["task"]: (r["score"], r["n"]) for r in rows}

    def set_metric(self, run_id: str, name: str, value: float | None, detail: dict | None = None) -> None:
        with self.con:
            self.con.execute("INSERT OR REPLACE INTO metrics VALUES (?,?,?,?)",
                             (run_id, name, value, json.dumps(detail) if detail is not None else None))

    def metrics(self, run_id: str) -> dict[str, tuple[float | None, dict | None]]:
        rows = self.con.execute("SELECT name, value, detail FROM metrics WHERE run_id = ?", (run_id,))
        return {r["name"]: (r["value"], json.loads(r["detail"]) if r["detail"] else None) for r in rows}

    def item_history(self, item_id: str) -> list[sqlite3.Row]:
        """Every run's result for one item — 'did the new model fix my car-wash question?'"""
        return self.con.execute(
            "SELECT runs.ts, runs.model, runs.quantization, runs.label, results.score, results.protocol "
            "FROM results JOIN runs USING(run_id) WHERE item_id = ? ORDER BY runs.ts", (item_id,)
        ).fetchall()
