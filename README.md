# benchsmith

Define custom, domain-specific LLM benchmarks and track improvements and
regressions — per item — across model releases, local vs cloud, and
quantisations.

Global leaderboards don't tell you whether the new Qwen release helps *your*
train-dynamics model or whether the Q4 quant broke *your* persona-schedule
generator. benchsmith is built around three things:

1. **An item format where the scoring kind is the primary axis.** Likelihood
   probes (`choice_loglik`, `pair_loglik`) for small models and anything
   factual; programmatic checks (`exact`, `regex`, `numeric`, `program`) for
   generation; `choice_generate` as the fallback for chat-only APIs.
2. **A per-item results ledger** (SQLite): model, dtype, quantisation, device,
   peak memory, tokens/s, git commit, suite content hash — and every item's
   score, so `diff` tells you *which* questions flipped, with a McNemar p-value
   instead of a vibe.
3. **Authoring helpers**: templates × slot-fills with automatic answer
   balancing, plus validation (schema, duplicates, balance, leading-space
   conventions, n-gram leakage against a training corpus).

## Quick start

```bash
uv pip install -e ".[hf,dev]"     # torch + transformers for local models
# (CPU-only torch: uv pip install torch --index-url https://download.pytorch.org/whl/cpu)

benchsmith validate suites/world_model_v1          # balance, dedup, conventions
benchsmith freeze   suites/world_model_v1          # write MANIFEST.json (content hash)
benchsmith run      suites/world_model_v1 hf:HuggingFaceTB/SmolLM2-135M --label base
benchsmith run      suites/world_model_v1 hf:HuggingFaceTB/SmolLM2-360M --label bigger
benchsmith runs
benchsmith diff <run_a> <run_b> --items            # per-task delta, fixed/broken items, p-values
benchsmith history calendar-3b7f31fb               # one item across every run
benchsmith run       suites/schedules_v1 hf:<ckpt> --max-tokens 1500
benchsmith calibrate suites/schedules_v1 <run_id>  # T4: KL per field vs reference_stats.json
benchsmith show     suites/world_model_v1 -n 10    # random sample for hand review
benchsmith leakage  suites/*_v1 --corpus data/pilot_v1/shards --ngram 13   # streaming, parquet-aware
```

Backends: `hf:<model-or-path>` (logprobs, generation, VRAM/RSS, tok/s; options
`--opt dtype=bf16 --opt quantization=4bit --opt batch_size=16`),
`openai:<model>` (any OpenAI-compatible endpoint via `--opt base_url=…`;
generation only, likelihood items fall back to `choice_generate` and the ledger
records it), `dummy:uniform` for dry runs.

## Suites

* `suites/world_model_v1/` — Track 1 of the tiny-model capability brief: 1223
  closed-set cloze probes in 5 sub-categories, scored by per-token-mean
  log-likelihood. See its README for the scoring spec.
* `suites/schedules_v1/` — Track 2: 900 household-week prompts (300 × XML/JSON/KV)
  sliced from held-out generator documents (seed 1000), scored programmatically
  by `benchsmith/checks/schedules.py` (T1 parse, T2 schema, T3 the generator's
  rules, each cited to a line of `generate_structured.py`), plus
  `benchsmith calibrate` for T4 (KL vs `reference_stats.json`, 50k held-out docs).
  Re-author with `scripts/author_schedules.py --generator <path>`; the generator
  is only needed at authoring time.
* `suites/dialogue_v1/` — Track 3: 491 `pair_loglik` minimal pairs (pragmatic /
  factual / social / temporal violations, length- and vocabulary-matched) plus
  101 generative openers scored leniently by `benchsmith/checks/dialogue.py`
  (qualitative). Out-of-distribution by design — read as a trend across the ladder.

## Writing your own

A suite is a directory with `suite.json` and one JSONL per task:

```json
{"name": "car_wash", "version": "1.0", "tasks": [
  {"name": "commute", "file": "commute.jsonl", "kind": "choice_loglik", "scoring": {"normalize": "mean"}}]}
```

```json
{"id": "carwash-50m", "prompt": "The car wash is fifty metres away, so I will", "options": [" walk", " drive"], "answer_idx": 0}
```

One item is a valid suite — that is the point. For generated answers use
`"kind": "numeric", "target": "42", "tolerance": 0.5` or
`"kind": "program", "checker": "mypkg.checks:valid_schedule"`.

Programmatic authoring: `benchsmith.authoring.templates.Builder` — see
`scripts/author_world_model.py`.

## Layout

```
benchsmith/
  items.py        Item / Task / Suite, JSONL IO, content hash
  scoring.py      one scorer per kind, protocol fallback
  checks/         program checkers: schedules (T1–T3), calibration (T4), dialogue (qualitative)
  backends/       hf (transformers), openai_compat, dummy
  runner.py       run a suite, collect per-item results + resource stats
  ledger.py       SQLite: runs, results, item history
  diff.py         run-vs-run comparison, Wilson CI, exact McNemar
  validate.py     schema, dedup, balance, conventions (+ leakage via --corpus)
  leakage.py      streaming n-gram / exact-document overlap vs training shards
  authoring/      template Builder with rebalancing
  cli.py
scripts/author_world_model.py   reproduces suites/world_model_v1
scripts/author_schedules.py     reproduces suites/schedules_v1 (needs the generator path)
scripts/author_dialogue.py      reproduces suites/dialogue_v1
tests/
```

## Roadmap

* `from-pdf --exercises`: extract solved exercises from a textbook chapter into `numeric`/`exact` items
* llama.cpp / GGUF backend with logprobs; lm-eval-harness task exporter
* YourBench adapter (document → `choice_generate` items) as an optional authoring backend
