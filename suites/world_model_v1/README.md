# world_model v1.0 — Track 1 likelihood probes

Closed-set cloze items measuring everyday world knowledge a tiny model (50M–600M
params, ~1B tokens) could plausibly absorb. **Every item is scored by comparing
log-likelihoods over its candidate options — no text is generated or parsed.**

| task | n | what it probes |
|---|---|---|
| `calendar` | 303 | day/month adjacency and ordering, weekday vs weekend, days per month, month→season, counts (7 days, 12 months…) |
| `seasons_weather` | 212 | season identity, hot/cold, daylight, typical °C, weather↔activity/clothing |
| `time_conventions` | 219 | meals and parts of day, clock times for breakfast/school/office/sleep, durations, frequencies, ordering |
| `physical_social` | 212 | room↔activity, appliances, transport modes/speeds/durations, walk-vs-drive, occupations, household roles |
| `general_knowledge` | 277 | control: capitals, animal classes, basic science, colours, places (Simple-Wikipedia-like prose knowledge) |

Item counts per sub-category are ≥ 200 (the floor from the data brief); `calendar`
and `general_knowledge` exceed 270. `calendar`, `physical_social` and
`general_knowledge` are 100 % distinct facts. `seasons_weather` and
`time_conventions` were topped up to ≥ 200 with **prefixed paraphrases** of
existing items ("Usually, …", "For most people, …"), tagged `paraphrase` in
`tags` and `+prefix` in `provenance.source`; these add context variety, not new
facts — filter them out with `tags` if you want the strict-fact subset.

## Item schema

```json
{"id": "calendar-3b7f31fb", "kind": "choice_loglik",
 "prompt": "The shortest month of the year is",
 "options": [" December", " April", " February", " January"], "answer_idx": 2,
 "category": "calendar", "tags": [], "notes": null,
 "provenance": {"method": "template", "source": "calendar_counts", "seed": 1000,
                "author": "template:scripts/author_world_model.py"}}
```

`answer_idx` positions are shuffled with the authoring seed, so position carries
no signal (relevant if a run falls back to the generative `choice_generate`
protocol). `notes` records the implied continuation when the blank is not the
end of the sentence (e.g. "… degrees Celsius", "… days").

## Scoring spec

For each item and option `o`:

```
score(o) = (1 / n_tokens(o)) * Σ_t log p(o_t | prompt, o_<t)
pred     = argmax_o score(o);   item is correct iff pred == answer_idx and no tie
task accuracy = mean over items;  report Wilson 95 % CI (n≈200–300 → ±6 pp)
```

* **Length normalisation is per token** (`scoring.normalize = "mean"`), so
  multi-token answers (" living room", " 500 kilometres") are not penalised.
  `"byte"` (per UTF-8 byte) is available as a tokenizer-independent alternative;
  `"sum"` is raw. Record which one you used — scores are not comparable across
  modes.
* **Tokenisation.** The prompt is encoded with special tokens, each option
  **without**, and the two id lists are concatenated. Options therefore carry
  their own leading space (`" Tuesday"`), matching byte-level BPE's convention
  that a word-initial token includes the space. Prompts never end with
  whitespace. `benchsmith validate` enforces both.
* Distractors are the same word class and drawn from the same vocabulary as the
  answer (other days, other seasons, other appliances…).
* **Balance.** Answers cycle over the whole option vocabulary; the authoring
  script downsamples so no answer string is correct > 1.5× more often than the
  least frequent one in its option family. `validate` reports the
  leave-one-out frequency-prior accuracy (must be ≤ chance + 10 pp) and the
  maximum `P(correct | option shown)`.

## Provenance

* Authored by `scripts/author_world_model.py` (templates × slot-fills, seed
  1000), hand-reviewed via `benchsmith show`. Re-running the script reproduces
  the files byte-for-byte.
* No item text was copied from Simple Wikipedia / FineWeb-Edu. Before use on a
  specific model ladder, run the leakage check against its training shards:
  `benchsmith validate suites/world_model_v1 --corpus data/pilot_v1/shards --ngram 13`.

## Freeze policy

`MANIFEST.json` records the content hash (`3430bfa40a625ead`). Runs refuse a
suite whose hash differs from its manifest. Do not edit this directory; put
fixes in `suites/world_model_v1_1/` with `version: "1.1"`.
