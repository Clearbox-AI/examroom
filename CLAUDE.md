# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

examroom defines small, domain-specific LLM benchmarks and tracks per-item regressions across models, releases and quantisations in a SQLite ledger. (It was called `benchsmith` until 2026-09-25.) The README covers the CLI and suite authoring from the user's point of view.

## Commands

The venv has **CPU torch**, while `uv.lock` pins CUDA torch. Plain `uv sync` / `uv run` would replace it with CUDA wheels, so call the venv directly or pass `--no-sync`:

```bash
.venv/bin/python -m pytest -q                                   # full suite, ~1 min
.venv/bin/python -m pytest tests/test_core.py::test_item_validation -q    # single test
.venv/bin/python -m pytest tests/test_suites_conform.py -k crc_pilot_v0.4   # conformance for one suite
uv run --no-sync examroom <cmd>                                 # or .venv/bin/examroom
uv pip install -e ".[hf,dev]"                                   # extras: hf, anthropic, openai, crc, dev
```

If a uv build fails with `failed to read directory .../archive-v0/...`, the global uv cache has dangling entries. Point `UV_CACHE_DIR` at a fresh directory.

There is no linter or formatter config.

Use `examroom run <suite> dummy:uniform --dry-run --limit 3` for a cheap end-to-end check that doesn't write to the ledger.

## Architecture

Data flow: `items.load_suite` → `backends.load_backend("<type>:<model>")` → `runner.run_suite` → `scoring.score_item` per item → `ledger.Ledger.record`. `diff`, `selection`, `report` and `calibrate` read only from the ledger.

- **Kind is the primary axis.** Each task has one `Kind` (`items.py`), and every item in the task must match it. `scoring.SCORERS` maps each kind to a scorer that returns `(score in [0,1], detail)`. Topics are just tags or categories.
- **Protocol fallback.** `scoring.resolve_protocol` chooses a protocol based on what the backend can do. `choice_loglik` on a chat-only backend becomes `choice_generate`, and `pair_loglik` raises an error. The protocol actually used is stored on every ledger result row.
- **Task `scoring` dict** carries per-task options, such as `normalize`, `instruction`, `tolerance`, `relative`, `abs`, `min_rel_tol`, `answer_prefix`, `number`, `decimal_comma`, `temperature` and `max_tokens`. CLI `--max-tokens`/`--instruction` are passed as run `overrides`. They get merged into every task's scoring and recorded on the run, but they don't change the suite hash.
- **Backends** follow the `Backend` protocol in `backends/base.py`: `loglik`, `generate`, `info: BackendInfo`, `stats: Stats` (accumulated through `stats.tick`), plus `supports_logprobs`/`supports_generate`. Set `parallel_safe = False` to opt out of `--workers` threading (`hf` does this). Register new backends in `backends/__init__.load_backend`. `loglik` appends continuations verbatim, so the leading-space convention belongs to the suite, not the backend. The Anthropic backend must never send `temperature`.
- **Leaderboard model.** Rows are (model, precision) pairs in overlapping tiers defined in `registry.TIERS`: tiny ≤1B params, small ≤14B params, local ≤128 GB weights at the run's precision. Sizes are resolved **at read time** by `Registry.size(backend_json)` from `models.json` (committed; filled with `examroom models add-hub`/`quant-hub` from Hub metadata, with active params quoted from model cards). Registry facts win over what a run recorded, and missing footprints are estimated and flagged. Don't add new size facts to runs; put them in the registry. `--opt n_params/active_params/weights_mb/model_id` and `examroom annotate` are run-level fallbacks; `cmd_run` strips them out before building the backend.
- **Run profiles** (`profiles.py`, `nothink@1`, `think@1`) fix reasoning, temperature and a token-budget floor (`min_max_tokens`). Only runs with the same profile id are comparable. Never edit a profile; add a new version. A profiled run refuses `--max-tokens`, `--instruction`, the options in `PROFILE_FIXED_OPTIONS`, and partial runs (`--limit`/`--task`) unless `--dry-run` is set. Each backend maps `reasoning=off|on` into its own API. Generation output goes through `strip_reasoning` before scoring.

## Suites, freezing and content hashes

- A suite is a directory with `suite.json`, one JSONL per task, and optionally `checker.py`, `README.md` and `MANIFEST.json`. `Suite.content_hash()` covers task configs, every item (except `notes`) and the **bytes of suite-local checker files**. It also covers the file named by `meta["calibration"]`. The rest of `suite.json`'s `meta` (e.g. `meta.acceptance`) is excluded.
- `examroom freeze` writes `MANIFEST.json`. `examroom run` refuses unfrozen or changed suites unless given `--allow-unfrozen`.
- **Never edit a frozen suite in place, including its `checker.py`.** Create a new version directory instead (e.g. `crc_pilot_v0.4` → `v0.5`). Old versions stay in place so their ledger runs remain comparable. Frozen suite data and checkers still say "benchsmith" on purpose, because renaming would change their hashes.
- `program` items reference `"./checker.py:fn"`, which is resolved relative to the suite directory (`scoring.checker_module`), so a frozen suite carries its own scoring code. A checker gets `(item, output)` and returns a float or `{"score": …, ...}`; a `tiers` dict in that result is printed per task by `run`. A checker function may define a `preflight` attribute, which the runner calls once before the first model call so environment problems fail early.
- `tests/test_suites_conform.py` automatically checks every `suites/*` directory: it validates the suite, checks that the frozen hash matches, and checks that reference answers score 1.0 through the real scorer. It needs no per-suite test code. The legacy `schedules_v1`/`dialogue_v1` suites point at a checker module that no longer exists, so they skip there.
- `scripts/author_*.py` regenerate suites. `author_schedules.py` needs the external generator via `--generator`. Use `examroom.authoring.templates.Builder` for templated items with answer rebalancing, and `examroom.authoring.openstax` for textbook extraction.

## Ledger

`examroom.sqlite` (gitignored) has a `runs` table with the backend and stats as JSON, a `results` table with one row per (run, item) holding score, protocol and detail JSON (including raw outputs), and a `metrics` table for run-level values such as calibration KL. There is also a `task_scores` view. `diff.py` does per-item comparison with exact McNemar tests and Wilson CIs. `selection.py` gates on the Wilson lower bound by default, using per-task minimums from `meta.acceptance` (task globs, last match wins).
