# crc_pilot v0.3 — SLM capability ladder for colorectal-cancer drug-efficacy work

**Status: DRAFT.** All MCQ facts (base and hard band) and Italian phrasing must be
reviewed by a domain expert before freezing a v1. Non-MCQ tiers are template-generated
with seeded jitter: ground truth by construction, no leakage, expandable.

Same shape as v0.2 (five tiers × EN/IT, plus the hard band); see v0.1's README for the
tier table and backend commands. This suite is the default in `scripts/run_openrouter.sh`.

## Setup

The `tool_hard_*` checker executes the model's code in a subprocess that imports pandas,
so the interpreter running examroom needs it:

```sh
uv sync --extra crc          # or --extra dev, which includes it
```

Without it the run stops before the first model call with a clear error (the checker
declares a `preflight` hook the runner calls up front). v0.2 would instead have scored
every hard tool item 1/3 with "ModuleNotFoundError" as if the model had failed.

⚠ `tool_hard_*` runs model-generated Python on your machine with no sandbox beyond the
subprocess and a 10 s timeout. Fine for local benchmarking of models you chose, but know
that it happens.

## Changes from v0.2

Three review findings, all of which let a model score without doing the task:

- **Efficacy direction is now a coin flip.** In v0.2 every synthetic abstract favoured the
  experimental arm (larger ORR, longer PFS, HR < 1 in 100 of 100 abstracts), so "pick the
  larger number" answered every experimental-arm extraction and both summary ORR fields.
  Now the experimental arm (still always the first regimen mentioned) wins half the time,
  with HR > 1 when it loses; toxicity and the hard-band subgroup flip on independent
  coins. Extract and summary items carry a `direction:exp` / `direction:ctrl` tag, so a
  per-direction split of a run tells you whether a model reads or guesses.
  `orr_absolute_difference_pct` in `summary_hard_*` can therefore be negative.
- **Hard tool results are read from the last stdout line only.** v0.2 accepted the
  expected number or patient id anywhere in stdout, so `print(df)` or printing a boolean
  mask scored 1.0. Now the last non-empty line must carry the expected value and no other
  number (prose around it is fine: "Difference: -2.03").
- **The count task has a varying answer.** "Patients with pfs_months above the median" of
  10 rows is always 5; it is now "above the mean" (tables with a value on the mean are
  rejected at authoring time).
- pandas is declared (`crc` and `dev` extras) and a missing pandas raises, see Setup.

Still open from the same review, deferred to a later version: the oxaliplatin base MCQ
contradicts the hard-band acute-neurotoxicity item; the NNT authoring guard tests the
wrong rounding boundary; `extract_*` parse the first number (a "Study 2" echo scores 0);
`compute_it` accepts only `RISPOSTA:`.

## Reading results

Base band ceiling + hard band + quantization = the emergence story: compare the same
architecture at Q4/Q5/Q8/FP16 per task; the tier where scores first drop tells you what
the quantization broke (tool-call fidelity and structured JSON go first, knowledge last).
`_en` vs `_it` twins give the language gap per tier; `direction:` tags give the
read-vs-guess split for tiers 2 and 4.
