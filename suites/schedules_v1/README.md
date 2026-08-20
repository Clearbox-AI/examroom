# schedules v1.0 — Track 2 schedule coherence

Measures structure emission on the model's home format: given the persona block
of a household-week document, the model must write the seven days. Scored
**programmatically** against rules derived from the generator that produced the
training data (`bin/generate_structured.py`, commit `3f9d389`) — the generator
IS the spec.

| task | n | prompt | cut point |
|---|---|---|---|
| `schedule_xml` | 300 | `<household …><profile …>…</profile>\n  <week>\n` | after `<week>` |
| `schedule_json` | 300 | `{ "household_id": …, "profile": {…},\n "week": [\n` | after `"week": [` |
| `schedule_kv` | 300 | `household: …\nprofile: …\noccupant: …\n` | after the last `occupant:` line |

Prompts are held-out generator documents, **seed 1000, indices 0–899**
(training used seed 42; the generator seeds `random.Random(f"{seed}:{index}")`
per document, so the streams are disjoint). Documents were generated
`with_summary=False`; the parsers tolerate and strip a trailing prose summary,
since training documents end with one. `target` holds the full reference
document. `reference_stats.json` holds distributions over **50 000** further
held-out documents (seed 1000, indices 100 000–149 999).

## Item schema

```json
{"id": "sched-kv-0002", "kind": "program", "prompt": "<prefix>", "target": "<full reference document>",
 "checker": "benchsmith.checks.schedules:score", "category": "kv", "max_tokens": 1100,
 "provenance": {"method": "generator", "source": "generate_structured.py@3f9d389", "seed": 1000}}
```

`max_tokens` is sized from the reference length (~2.5 chars/token); override per
run with `benchsmith run … --max-tokens N` (recorded in the run, does not touch
the suite hash). Decoding is greedy (`temperature: 0`) by default; for T4 you may
prefer `samples > 1` at temperature 0.7–1.0 — set it per task in a v1.x.

## Scoring

The completion is appended to the prompt and the whole document parsed.

* **T1 parseability** — XML: `ElementTree` parses the text up to `</household>`;
  JSON: `raw_decode` of the leading object; KV: **every** non-blank line up to
  the first blank line matches the line grammar in `checks/schedules.py::KV_LINE`
  *and* at least one `day:` line exists (the prefix alone is grammatical).
  All-or-nothing; an unparseable document scores 0 on every tier.
* **T2 schema completeness** — fraction of checks passed: 7 days in
  Monday→Sunday order, profile fields + enums, every day has the six sections,
  weather fields, all times `HH:MM` (00:00–23:59, plus `24:00` for occupancy
  end), enum values (meal types, appliance names, trip purposes/modes),
  integer durations/people.
* **T3 internal consistency** — fraction of rule checks passed (inventory below).
* **Item score** = mean(T1, T2, T3). Task score = mean over items. The run
  printout also shows the per-tier means.
* **T4 calibration** — distribution-level, run after the fact:
  `benchsmith calibrate suites/schedules_v1 <run_id>` parses every stored
  completion, builds the same histograms as `reference_stats.json` and reports
  `KL(model ‖ reference)` (nats, additive smoothing 1e-4 over the union of
  support) and total variation per field, plus the mean KL, stored as run metric
  `t4_mean_kl`. Fields: occupation, household size/adults/children, heating,
  energy label, surface, PV/EV, breakfast/lunch/dinner time (15-min bins), lunch
  presence, trip mode/purpose, commute mode, per-day counts (meals, showers,
  appliances, trips, EV sessions, occupancy slots; weekday vs weekend trips),
  appliance names and durations (30-min bins), shower duration, cloud, temperature
  (5 °C bins). Sanity: the 900 references themselves score KL < 0.05 on every
  field against the 50k set.

### T3 rule inventory (generator line numbers, commit 3f9d389)

| rule | check | lines |
|---|---|---|
| `start_before_end` | every occupancy slot and trip has start < end | 136–142, 203–223 |
| `occupancy_contiguous` | slots run 00:00 → 24:00 with no gaps/overlaps | 136–142 |
| `occupancy_people_range` | 0 ≤ people ≤ n occupants | 137–139 |
| `occupancy_edges_full` | first and last slot have all n occupants | 137, 139 |
| `commute_count_weekday` | weekday commutes = number of working occupants (office_worker, teacher, shopkeeper, student, school_child) | 40–48, 123–131 |
| `commute_windows` | a one-to-one assignment exists: depart within window start ±15 min, return within window end ±20 | 41–45, 129–130 |
| `weekday_occupancy_dip` | with workers: exactly 3 slots, middle slot = n − workers | 136–140 |
| `no_worker_single_slot` | no workers on a weekday → a single 00:00–24:00 slot | 141–142 |
| `no_commute_weekend` / `weekend_single_slot` | weekends: no commutes, single slot | 123, 141–142 |
| `commute_mode_car_if_ev` | EV households commute by car | 206 |
| `breakfast_and_dinner_present` | every day | 147, 153 |
| `meal_times` | breakfast 07:15 ±25, lunch 13:00 ±20, dinner 19:30 ±25 | 147–153 |
| `meals_people` | breakfast/dinner people = n | 147, 153 |
| `lunch_iff_someone_home` | lunch present ⇔ someone home at 13:00 (not away on a commute spanning 13:00) | 144–152 |
| `lunch_people` | lunch people = number home at 13:00 | 144–152 |
| `meals_time_ordered` | breakfast < lunch < dinner | (implied) |
| `showers_one_per_occupant` | exactly n showers | 155–162 |
| `shower_duration` | 5–12 min | 161 |
| `appliance_unique_per_day` | each appliance at most once per day | 164–194 |
| `appliance_windows` | washing_machine 18:15 ±45, {60,90,120}; dishwasher 20:30 ±20, {50,70}; oven 18:45 ±20, 30–75; air_conditioner 13:30, 180–360; heating 06:00, 600–900 | 165–194 |
| `ac_iff_hot` | air conditioner ⇔ temp ≥ 27 °C | 183 |
| `heating_iff_cold` | heating ⇔ temp ≤ 7 °C | 189 |
| `dishwasher_if_large_home` | surface ≥ 75 m² → dishwasher every day | 171 |
| `ev_only_if_has_ev` | no sessions without an EV | 197 |
| `ev_sessions_valid` | ≤ 1 session, start 19:00 ±60, {120,180,240} | 196–201 |
| `ev_charges_after_commute` | EV + commute day → exactly one session | 197 |
| `shopping_window` / `shopping_only_if_home` | 10:30 ±45 → 11:45 ±45; only if someone is home at lunch | 210–216 |
| `leisure_weekend_dry_only` / `leisure_window` / `at_most_one_leisure` | weekend, precip ≤ 2 mm, 14:30 ±60 → 17:30 ±60 | 217–223 |
| `retired_age` / `child_ages` | retired > 60; school_child 4–14; student 15–19 | 79–88 |
| `occupant_count_matches` | declared occupants = listed | 240 |
| week-level `wash_days_2_or_3`, `oven_days_2_or_3`, `shopping_at_most_once` | per week | 116–118 |

Not checked (the prefix does not expose them): season/weather plausibility
(L90–99) — the model chooses the weather; only weather↔appliance consistency is
enforced. Per-person trips legitimately overlap (two commuters), so the brief's
"non-overlapping activities" is applied to occupancy slots and to same-name
appliances, not across trips.

**Generator quirk, kept as-is:** shopping trips jitter `depart` and `return`
independently (L212–214), so ~0.5 % of reference documents contain a shopping
trip that returns before it departs; 5 of the 900 references fail
`start_before_end` for this reason (ids `sched-xml-0204`, `sched-xml-0291`,
`sched-json-0295`, `sched-json-0733`, `sched-kv-0356`). The rule stays strict —
a model that learned the generator's marginal will hit the same ceiling (~99.9 %
of checks), which is fine for ladder comparisons.

## Leakage

`suite.json` declares `leakage_check: exact`: the format's boilerplate shares
13-grams with *any* generator output by construction, so n-gram overlap is not
the right test here. `benchsmith leakage suites/schedules_v1 --corpus <shards>`
instead flags a training document that is identical to an item's reference or
starts with an item's prompt prefix (i.e. a seed collision). Checked against a
50k-document regeneration of the seed-42 source (superset of the 30k used for
training): 0 collisions.

## Freeze policy

`MANIFEST.json` records the content hash, generator path/commit/seed and the
reference-stats index range. The scorer lives in `benchsmith/checks/` and is
versioned with the repo (`git_commit` in every run); changing a rule means a new
suite version, not an in-place edit.
