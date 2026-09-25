# crc_pilot v0.4 — SLM capability ladder for colorectal-cancer drug-efficacy work

**Status: DRAFT.** All MCQ facts (base and hard band) and Italian phrasing must be
reviewed by a domain expert before freezing a v1. Non-MCQ tiers are template-generated
with seeded jitter: ground truth by construction, no leakage, expandable.

Same shape as v0.3 (five tiers × EN/IT, plus the hard band). See v0.1's README for the
tier table and backend commands, v0.3's for setup (`uv sync --extra crc`: the
`tool_hard_*` checker executes model code in a subprocess that needs pandas, and runs
model-generated Python on your machine with no sandbox beyond the subprocess and a
10 s timeout). This suite is the default in `scripts/run_openrouter.sh`.

## Changes from v0.3

Driven by the first full v0.3 batch (six OpenRouter models, ledger runs 9395a778…c8c6eb55):

- **`tool_hard_*` budget 300 → 1200 tokens.** Several models ignore "df is already
  loaded" and re-embed the ten-row table as a DataFrame literal, which needs ~600 tokens.
  At 300 they were cut off mid-list and scored as a malformed call: gemma-3-27b went
  0.33/0.42 → 1.00/1.00 and qwen3.8-27b 0.94/0.50 → 1.00/1.00 in a dry run at 1200,
  while qwen3-30b-a3b stayed near 0.55 (bare trailing expression instead of `print`,
  missing closing brace), so the tier now measures code, not verbosity. Same fix v0.2
  applied to `compute_*`.
- **Extraction reads the last number** (task option `number: last`). One v0.3 miss was
  `3{ "text": "30.9" }`, read as 3; the hard band's "In Study 2, …" echo has the same
  shape. A model that follows "answer with only the number" is unaffected.
- **Italian numeric tasks accept a decimal comma** (`decimal_comma: true` on
  `extract_it`, `extract_hard_it`, `compute_it`): "68,6" reads as 68.6, not 686. Two of
  the three v0.3 `compute_it` misses were correct answers in Italian locale. Prompts
  still ask for a point; a thousands group ("1,234") is untouched.
- **`compute_it` accepts `ANSWER:` as well as `RISPOSTA:`** (`answer_prefix` takes a
  list; the last occurrence of either wins), so the marker's language is not scored.
- **NNT authoring guard fixed.** It rejected 1/ARR near an integer (the safe cases) and
  kept values near a .5 boundary (the ambiguous ones); it now rejects within 0.08 of .5.
- **Oxaliplatin MCQ.** The base item's correct option no longer says the cumulative
  neuropathy is "triggered by cold"; that describes the acute form, which the hard-band
  item asks about. Both items now agree.
- The numeric parser accepts a leading-dot decimal (".69" for 0.69, a v0.3 miss of
  gemma-3-27b). Replaying every v0.3 extraction output under v0.4 scoring shows the
  last-number policy fixes one item and breaks none, while the leading-dot change fixes
  that one and breaks two gemma-3-27b glitch outputs of the form `.027.3`, which v0.3
  read as 27.3 only because the old pattern skipped the dot. Kept: that string is not a
  number. `extract_en`'s no-op `min_rel_tol: 0.0` is dropped so EN/IT task configs match.

Not changed: the adverse-event extraction question says "what percentage" while the
others say "in percent"; qwen3.8-27b answers it as a fraction (0.33) in EN only, in both
v0.1 and v0.3. Left as an observation for the expert review round.

## Reading results

Base band ceiling + hard band + quantization = the emergence story: compare the same
architecture at Q4/Q5/Q8/FP16 per task; the tier where scores first drop tells you what
the quantization broke (tool-call fidelity and structured JSON go first, knowledge last).
`_en` vs `_it` twins give the language gap per tier; `direction:` tags give the
read-vs-guess split for tiers 2 and 4. In v0.3 the base band saturated (≥ 0.97 for all
six models) and only `mcq_hard_*` and `tool_hard_*` separated them.
