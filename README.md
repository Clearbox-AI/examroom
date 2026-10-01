# examroom

Define custom, domain-specific LLM benchmarks and track improvements and
regressions — per item — across model releases, local vs cloud, and
quantisations.

Global leaderboards don't tell you whether the new Qwen release helps *your*
train-dynamics model or whether the Q4 quant broke *your* persona-schedule
generator. examroom is built around three things:

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

examroom validate suites/world_model_v1          # balance, dedup, conventions
examroom freeze   suites/world_model_v1          # write MANIFEST.json (content hash)
examroom run      suites/world_model_v1 hf:HuggingFaceTB/SmolLM2-135M --label base
examroom run      suites/world_model_v1 hf:HuggingFaceTB/SmolLM2-360M --label bigger
examroom runs
examroom diff <run_a> <run_b> --items            # per-task delta, fixed/broken items, p-values
examroom history calendar-3b7f31fb               # one item across every run
examroom report                                  # self-contained report.html: browse runs, outputs, item history, 2-run compare
examroom run       suites/schedules_v1 hf:<ckpt> --max-tokens 1500
examroom calibrate suites/schedules_v1 <run_id>  # T4: KL per field vs reference_stats.json
examroom show     suites/world_model_v1 -n 10    # random sample for hand review
examroom leakage  suites/*_v1 --corpus data/pilot_v1/shards --ngram 13   # streaming, parquet-aware
```

Backends: `hf:<model-or-path>` (logprobs, generation, VRAM/RSS, tok/s, parameter
count and weight footprint; options `--opt dtype=bf16 --opt quantization=4bit
--opt batch_size=16`; generation uses the tokenizer's chat template when it has
one, `--opt chat_template=off` for raw prompts, `--opt reasoning=off` for
Qwen3-style thinking models),
`anthropic:<model>` (Claude via the native SDK; `--opt thinking=off|adaptive|none
--opt effort=low…max`; temperature is never sent — current Claude models reject
sampling params), `openai:<model>` (any OpenAI-compatible endpoint via
`--opt base_url=…`), `dummy:uniform` for dry runs.  Chat backends are generation
only: likelihood items fall back to `choice_generate` (recorded in the ledger)
and `pair_loglik` refuses to run.

## Smallest good-enough model

"Good enough" is a minimum per task, not the suite mean: the mean is dominated by
easy, saturated tasks, while a deployment fails on its weakest required capability.
Put the minimums in the suite's `meta.acceptance` (task globs; suite `meta` is not
part of the content hash, so this does not unfreeze a suite), or keep per-use-case
profiles outside it:

```json
"meta": {"acceptance": {"*": 0.9, "mcq_*": 0.75, "tool_hard_*": 0.85}}
```

```bash
# record size with every run (API backends can't report it; hf fills n_params/weights_mb itself)
examroom run suites/crc_pilot_v0.4 openai:qwen3-4b --opt base_url=http://localhost:8080/v1 \
    --opt n_params=4B --opt quantization=Q4_K_M --opt weights_mb=2500 --label q4
examroom annotate <run_id> n_params=30.5B active_params=3.3B    # fix up past runs
examroom select suites/crc_pilot_v0.4                           # uses meta.acceptance
examroom select suites/crc_pilot_v0.4 --criteria rag_profile.json --min 'tool_hard_*=0.9' --size weights_mb
```

`select` prints a task × run matrix ordered by size and names the smallest run that
clears every gate (exit status 1 if none does). When several patterns match a task
the last one wins. A task passes when the 95% Wilson *lower bound* of its score
clears the minimum (`--gate point` compares the raw score instead), so small tasks
can't pass on luck: 12/12 only demonstrates 0.76, and `select` flags gates that a
task is too small to ever clear.

## Suites

* `suites/world_model_v1/` — Track 1 of the tiny-model capability brief: 1223
  closed-set cloze probes in 5 sub-categories, scored by per-token-mean
  log-likelihood. See its README for the scoring spec.
* `suites/schedules_v1.1/` — Track 2: 900 household-week prompts (300 × XML/JSON/KV)
  sliced from held-out generator documents (seed 1000), scored programmatically
  by the suite-local `checker.py` (T1 parse, T2 schema, T3 the generator's
  rules, each cited to a line of `generate_structured.py`), plus
  `examroom calibrate` for T4 (KL vs `reference_stats.json`, 50k held-out docs).
  Re-author with `scripts/author_schedules.py --generator <path>`; the generator
  is only needed at authoring time.
* `suites/dialogue_v1.1/` — Track 3: 491 `pair_loglik` minimal pairs (pragmatic /
  factual / social / temporal violations, length- and vocabulary-matched) plus
  101 generative openers scored leniently by the suite-local `checker.py`
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

Programmatic authoring: `examroom.authoring.templates.Builder` — see
`scripts/author_world_model.py`.

## Layout

```
examroom/
  items.py        Item / Task / Suite, JSONL IO, content hash
  scoring.py      one scorer per kind, protocol fallback
  checks/         program checkers: schedules (T1–T3), calibration (T4), dialogue (qualitative)
  backends/       hf (transformers), openai_compat, dummy
  runner.py       run a suite, collect per-item results + resource stats
  ledger.py       SQLite: runs, results, item history
  diff.py         run-vs-run comparison, Wilson CI, exact McNemar
  selection.py    per-task acceptance gates, smallest passing model
  validate.py     schema, dedup, balance, conventions (+ leakage via --corpus)
  leakage.py      streaming n-gram / exact-document overlap vs training shards
  authoring/      template Builder with rebalancing
  cli.py
scripts/author_world_model.py   reproduces suites/world_model_v1
scripts/author_schedules.py     reproduces suites/schedules_v1 (needs the generator path)
scripts/author_dialogue.py      reproduces suites/dialogue_v1
scripts/author_openstax_physics.py  numeric items from OpenStax solved exercises (--chapter N)
tests/
```

## Suites from textbooks

`examroom/authoring/openstax.py` extracts solved end-of-chapter exercises from
OpenStax books via their HTML edition (the PDFs render answer equations as vector
art; the HTML has real MathML).  Problems and the odd-numbered answer key are
joined by problem number; answers that parse as a physical quantity become
`numeric` items with a tolerance of half a unit in the last significant digit.
Figure-dependent problems and worked-solution/multi-quantity answers are skipped.
See `suites/openstax_physics_ch2_v1.1` (46 items, CC BY-NC-SA 4.0 attribution in
the suite metadata; v1 kept frozen).  Scoring compares magnitudes with a 0.5%
relative-tolerance floor (`abs` / `min_rel_tol` task scoring options) so unpinned
sign conventions and correct rounding don't count as errors.  For chat backends,
run with `--instruction "Give only the final numeric answer …"` and
`--opt thinking=adaptive --opt effort=low --max-tokens 3000`; Claude Sonnet 5
baseline: 0.957.

## Roadmap

* more book adapters (CLP problem books via PDF glyph decoding; arbitrary PDFs via LLM-assisted extraction)
* llama.cpp / GGUF backend with logprobs; lm-eval-harness task exporter
* YourBench adapter (document → `choice_generate` items) as an optional authoring backend
