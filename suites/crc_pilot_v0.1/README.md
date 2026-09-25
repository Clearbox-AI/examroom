# crc_pilot v0.1 — SLM capability ladder for colorectal-cancer drug-efficacy work

**Status: DRAFT.** Tier-1 facts and Italian phrasing are written from general oncology
knowledge and must be reviewed by a domain expert before this is frozen as v1.
Everything else (tiers 2–5) is template-generated with seeded jitter: no leakage
risk, ground truth known by construction, expandable to any size.

## Why this shape

The target use case is choosing a small local model (and a quantization of it) to
assist data analysis for colorectal-cancer drug-efficacy work: domain grounding for
RAG, paper reading, structured summarization, and MCP tool calls, in English and
Italian. Global benchmarks answer none of that; this suite answers exactly that.

Ten tasks = five tiers × two languages. The tiers are ordered by the kind of
capability that degrades under quantization, so comparing per-task scores of the
same architecture at Q4 / Q5 / Q8 / FP16 shows *where* a quantization starts to
hurt, not just that it does:

| tier | task | kind | measures | typically degrades |
|---|---|---|---|---|
| 1 | `mcq_*` (24×2) | choice_generate | drug/biomarker knowledge recall | last |
| 2 | `extract_*` (60×2) | numeric | find one number in a trial abstract | late |
| 3 | `compute_*` (24×2) | numeric | derive ARR / NNT / pooled ORR (`ANSWER:` line) | mid |
| 4 | `summary_*` (40×2) | program | abstract → fixed-schema JSON, field-checked | early |
| 5 | `tool_*` (20×2) | program | emit a valid `execute_python` MCP tool call | earliest |

The `_en`/`_it` twins are parallel item-for-item, so `score(task_en) − score(task_it)`
is a per-tier language-degradation number.

## Scoring notes

- All numbers use the decimal **point** in both languages and the prompts say so:
  the numeric parser reads `34,5` as 345 (thousands separator).
- `compute_*` allows working and parses the number after the final `ANSWER:` /
  `RISPOSTA:` line (task option `answer_prefix`); a bare number still scores.
- `summary_*` / `tool_*` are checked by the suite-local `checker.py`
  (`score_summary`, `score_tool_call`) — tiered: parse / schema / values and
  call-shape / compiles / content. The checker is covered by the content hash.
- Claude Sonnet 5 smoke test scores 1.0 on sampled items of every tier
  (2026-09-01, thinking off), so the ceiling is clean.

## Running the quantization matrix

```sh
# llama.cpp / Ollama / vLLM server per quantization, via the OpenAI-compatible backend:
llama-server -m qwen-27b-Q4_K_M.gguf --port 8080   # then:
benchsmith run suites/crc_pilot_v0.1 openai:qwen-27b --opt base_url=http://localhost:8080/v1 --label q4_k_m

# HF weights with bitsandbytes:
benchsmith run suites/crc_pilot_v0.1 hf:google/medgemma-... --opt quantization=4bit --label bnb4

# compare two quantizations of the same model, per task, with McNemar p-values:
benchsmith diff <run_q4> <run_fp16>
benchsmith report            # HTML leaderboard + per-item history
```

## Regenerating / extending

`uv run python scripts/author_crc_pilot.py` rewrites the item files (seeded, so
identical unless the script changes). Bump the version to a new directory for any
content change — the freeze hash also covers `checker.py`.
