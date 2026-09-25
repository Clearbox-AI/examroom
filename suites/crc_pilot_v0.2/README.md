# crc_pilot v0.2 — SLM capability ladder for colorectal-cancer drug-efficacy work

**Status: DRAFT.** All MCQ facts (base and hard band) and Italian phrasing must be
reviewed by a domain expert before freezing a v1. Non-MCQ tiers are template-generated
with seeded jitter: ground truth by construction, no leakage, expandable.

## Changes from v0.1

- **compute budget fix**: v0.1's `max_tokens=320` truncated verbose models mid-derivation
  (24 of 26–31 failures per model had the *correct* answer visible in the cut-off working);
  now 1200 tokens. v0.1 compute scores understate the models.
- **hard band added** (8 tasks, 164 items): full-precision 27B-class models ace the base
  band (three ledger baselines: gemma-4-31b 0.982, qwen3.8-flash 0.923, qwen3.8-27b 0.875 —
  the latter two depressed only by the compute truncation), so v0.2 restores top-end
  discrimination:
  - `mcq_hard_*` (16×2) — trial attribution (BEACON/CORRECT/RECOURSE/8HW/TRIBE…),
    metabolites, fine pharmacology; gemma-4-31b scores 0.875 here.
  - `extract_hard_*` (30×2) — TWO studies per prompt plus a subgroup sentence; the model
    must disambiguate study, arm, and overall-vs-subgroup.
  - `summary_hard_*` (24×2) — schema adds `orr_absolute_difference_pct`, a value NOT in
    the text (must be derived from two reported rates).
  - `tool_hard_*` (12×2) — the emitted `execute_python` code is **actually executed**
    (subprocess, 10s timeout) against the prompt's table; the printed result must match.
    ⚠ This runs model-generated Python on your machine with no sandbox beyond the
    subprocess and timeout — fine for local benchmarking of models you chose, but
    know that it happens.
- `score_summary` is now generic (expected keys/values read from the item target), so
  base and hard summaries share one scorer.

## Reading results

Base band ceiling + hard band + quantization = the emergence story: compare the same
architecture at Q4/Q5/Q8/FP16 per task; the tier where scores first drop tells you what
the quantization broke (tool-call fidelity and structured JSON go first, knowledge last).
`_en` vs `_it` twins give the language gap per tier. See v0.1's README for backend
commands; this suite is the default in `scripts/run_openrouter.sh`.
