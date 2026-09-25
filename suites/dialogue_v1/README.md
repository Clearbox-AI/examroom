# dialogue v1.0 — Track 3 everyday dialogue & interaction coherence

**Caveat honoured in the design:** the training corpus contains no
dialogue-formatted source. This track measures out-of-distribution
generalisation; expect low absolute scores at 50M. It exists to show what scale
adds, so read it as a *trend* across the ladder.

| task | kind | n | what a failure means |
|---|---|---|---|
| `pairs_pragmatic` | pair_loglik | 160 | the model prefers a fluent reply to a *different* question |
| `pairs_factual` | pair_loglik | 109 | prefers a wrong everyday fact (time of day, season, room, count…) |
| `pairs_social` | pair_loglik | 105 | prefers the wrong role/register (a shopkeeper tucking you into bed) |
| `pairs_temporal` | pair_loglik | 117 | prefers an impossible sequence (dinner, then breakfast; Sunday after Tuesday) |
| `openers` | program | 101 | **qualitative** — generated reply is off-format, off-topic or degenerate |

491 likelihood pairs ≥ the 400 target; ≥ 100 per violation type.

## Likelihood pairs

```json
{"id": "dlg-temp-1a2b3c4d", "kind": "pair_loglik",
 "prompt": "A: Yesterday was Saturday, so what is today?\nB:",
 "good": " Today is Sunday.", "bad": " Today is Tuesday.",
 "category": "temporal", "provenance": {"method": "template", "source": "day_order", "seed": 1000}}
```

The brief's field names map as `context → prompt`, `violation_type → category`.

**Score** = fraction of items where per-token-mean logprob(good) >
logprob(bad), each conditioned on the context. Same tokenisation convention as
Track 1: context ends in `B:` with no trailing space; both continuations begin
with one space; context and continuation are encoded separately and
concatenated. Chance = 50 %; Wilson 95 % CI at n≈100–160 is ±8–10 pp per type,
±4.5 pp on the 491-item total — compare *totals* across the ladder and use the
per-type split for diagnosis only.

**Not a fluency detector.** Every pair is a minimal pair from one template with
one slot changed, so `good`/`bad` share length and vocabulary: mean word-count
difference ≈ 0 in every task (`examroom validate` reports `len_diff`,
`good_longer` and the good/bad vocabulary Jaccard). Pragmatic pairs take `bad`
from another scenario's *coherent* reply in the same length bucket, so the bad
continuation is itself a fluent, in-distribution sentence.

Authoring: `scripts/author_dialogue.py` (hand-written scenario pool + slot
grids, seed 1000), hand-reviewed with `examroom show`.

## Generation openers (qualitative)

101 two-person openers (`"A: …\nB:"`), greedy decoding, 48 new tokens. Scored by
`examroom.checks.dialogue:score` as the mean of four 0/1 signals:
`format` (first line is a reply, any further lines are `A:`/`B:` turns),
`alternation` (turns alternate after B), `on_topic` (content-word overlap with
the opener, or an appropriate generic first word such as *yes/no/sure/thanks*),
`non_degenerate` (not an echo of the opener, ≥ 2 words, no repeated lines).
The per-signal means are printed by `examroom run`. **This sub-score is
qualitative**: the on-topic test is keyword overlap, not understanding, and a
model can satisfy `format` by emitting `A:`/`B:` noise. Do not rank models on
it; use it to see whether scale makes dialogue *shape* appear at all.

## Freeze policy

As for the other suites: `MANIFEST.json` carries the content hash; fixes go to
`dialogue_v1_1`.
