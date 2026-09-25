"""Author the Track 2 'schedules' suite from the structured-schedule generator.

The generator (`bin/generate_structured.py` in the corpus repo) is loaded HERE, at
authoring time only, from the path you pass; the resulting suite is frozen data
and nothing under benchsmith/ or suites/ imports the training pipeline.

For each format (xml, json, kv) it slices 300 held-out documents (seed 1000;
training used seed 42 — `random.Random(f"{seed}:{index}")` makes the streams
disjoint) at the cut point *after the persona block, before the first day*, and
stores the full reference document in `target`. It also builds
`reference_stats.json` from 50 000 held-out personas/weeks (seed 1000, a
different index range) — the T4 calibration targets.

Run:
  python scripts/author_schedules.py --generator ~/tiny_corpus_prep/bin/generate_structured.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

from benchsmith.items import Item, Kind, Provenance, SuiteMeta, Task, save_suite

SEED = 1000
N_PER_FORMAT = 300
N_REF = 50_000
REF_INDEX_OFFSET = 100_000   # reference docs use indices [100000, 150000) — disjoint from prompts [0, 900)
CHECKER = "./checker.py:score"
# canonical checker source, copied into the authored suite dir (suite-local, hash-covered);
# point this at the newest frozen suite when bumping versions
CHECKER_SRC = Path(__file__).resolve().parents[1] / "suites/schedules_v1.1/checker.py"


def load_generator(path: Path):
    # the script imports polars/pyarrow at module level only for parquet output; stub them
    for name in ("polars", "pyarrow", "pyarrow.parquet"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["pyarrow"].parquet = sys.modules["pyarrow.parquet"]  # type: ignore[attr-defined]
    spec = importlib.util.spec_from_file_location("generate_structured", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path.parent), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def cut_point(text: str, fmt: str) -> int:
    """Index just after the persona block / before the first day."""
    if fmt == "xml":
        marker = "  <week>\n"
    elif fmt == "json":
        marker = ' "week": [\n'
    else:  # kv: after the last 'occupant:' line
        marker = None
        lines = text.split("\n")
        last = max(i for i, ln in enumerate(lines) if ln.startswith("occupant: "))
        return len("\n".join(lines[: last + 1])) + 1
    i = text.index(marker)
    return i + len(marker)


def build_prompts(gen, commit: str | None) -> dict[str, list[Item]]:
    items: dict[str, list[Item]] = {f: [] for f in ("xml", "json", "kv")}
    for idx in range(3 * N_PER_FORMAT):
        fmt = ["xml", "json", "kv"][idx % 3]
        doc = gen.generate_doc(SEED, idx, fmt, with_summary=False)
        cut = cut_point(doc, fmt)
        prompt, reference = doc[:cut], doc
        # generous budget: tokenizer-dependent, so we size by characters (≈2.5 chars/token
        # on a corpus-trained BPE for this format) and let tasks override via scoring.max_tokens
        items[fmt].append(Item(
            id=f"sched-{fmt}-{idx:04d}", kind=Kind.PROGRAM, prompt=prompt, target=reference, checker=CHECKER,
            category=fmt, max_tokens=int(len(reference) / 2.5) + 128,
            provenance=Provenance(method="generator", source=f"generate_structured.py@{commit}", seed=SEED),
            notes=f"doc index {idx}; reference is the generator's own continuation",
        ))
    return items


def hist(counter: Counter) -> dict:
    tot = sum(counter.values())
    return {str(k): round(v / tot, 6) for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}


def bin15(hhmm: str) -> str:
    h, m = hhmm.split(":")
    return f"{int(h):02d}:{(int(m) // 15) * 15:02d}"


def build_reference_stats(gen) -> dict:
    c = {k: Counter() for k in (
        "occupation", "household_size", "n_adults", "n_children", "heating", "energy_label", "surface_m2", "has_pv", "has_ev",
        "breakfast_time", "lunch_time", "dinner_time", "lunch_present", "trip_mode", "trip_purpose", "commute_mode",
        "meals_per_day", "showers_per_day", "appliances_per_day", "trips_per_day", "ev_sessions_per_day",
        "appliance_name", "appliance_duration", "shower_duration", "occupancy_slots_per_day", "cloud", "temp_bin",
        "weekday_trips_per_day", "weekend_trips_per_day")}
    for i in range(REF_INDEX_OFFSET, REF_INDEX_OFFSET + N_REF):
        rng = random.Random(f"{SEED}:{i}")
        p = gen.make_persona(rng)
        week = gen.make_week(rng, p)
        occs = p["occupants"]
        c["household_size"][len(occs)] += 1
        c["n_adults"][sum(o["occupation"] not in ("school_child", "student") for o in occs)] += 1
        c["n_children"][sum(o["occupation"] in ("school_child", "student") for o in occs)] += 1
        for o in occs:
            c["occupation"][o["occupation"]] += 1
        c["heating"][p["heating"]] += 1; c["energy_label"][p["energy_label"]] += 1
        c["surface_m2"][p["surface_m2"]] += 1; c["has_pv"][p["has_pv"]] += 1; c["has_ev"][p["has_ev"]] += 1
        for d in week:
            c["cloud"][d["weather"]["cloud"]] += 1
            c["temp_bin"][int(d["weather"]["temp_c"] // 5) * 5] += 1
            for m in d["meals"]:
                c[f"{m['type']}_time"][bin15(m["time"])] += 1
            c["lunch_present"][any(m["type"] == "lunch" for m in d["meals"])] += 1
            c["meals_per_day"][len(d["meals"])] += 1
            c["showers_per_day"][len(d["showers"])] += 1
            c["appliances_per_day"][len(d["appliances"])] += 1
            c["trips_per_day"][len(d["trips"])] += 1
            c["ev_sessions_per_day"][len(d["ev_charging"])] += 1
            c["occupancy_slots_per_day"][len(d["occupancy"])] += 1
            (c["weekday_trips_per_day"] if d["day"] in gen.WEEKDAYS else c["weekend_trips_per_day"])[len(d["trips"])] += 1
            for s in d["showers"]:
                c["shower_duration"][s["duration_min"]] += 1
            for a in d["appliances"]:
                c["appliance_name"][a["name"]] += 1
                c["appliance_duration"][(a["duration_min"] // 30) * 30] += 1
            for t in d["trips"]:
                c["trip_mode"][t["mode"]] += 1; c["trip_purpose"][t["purpose"]] += 1
                if t["purpose"] == "commute":
                    c["commute_mode"][t["mode"]] += 1
    return {"n_docs": N_REF, "seed": SEED, "index_range": [REF_INDEX_OFFSET, REF_INDEX_OFFSET + N_REF],
            "note": "distributions over held-out generator output; times binned to 15 min, temps to 5 C, "
                    "appliance durations to 30 min; per-day counts are per (doc, day)",
            "fields": {k: hist(v) for k, v in c.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True, help="path to bin/generate_structured.py")
    ap.add_argument("--out", default="suites/schedules_v1")
    args = ap.parse_args()
    gpath = Path(args.generator).expanduser().resolve()
    gen = load_generator(gpath)
    commit = git_commit(gpath)
    out = Path(args.out)

    items = build_prompts(gen, commit)
    tasks = [Task(name=f"schedule_{f}", file=f"prompts_{f}.jsonl", kind=Kind.PROGRAM,
                  description=f"Complete a household week in {f.upper()} from the persona prefix; scored T1–T3 by {CHECKER}",
                  scoring={"temperature": 0.0}) for f in ("xml", "json", "kv")]
    meta = SuiteMeta(
        name="schedules", version="1.0",
        description="Track 2 — schedule coherence: structure emission on the model's home format, scored programmatically "
                    "(T1 parse, T2 schema, T3 generator rules) plus T4 calibration against reference_stats.json.",
        tasks=tasks,
        meta={"generator": str(gpath), "generator_commit": commit, "generator_seed": SEED, "training_seed": 42,
              "prompt_index_range": [0, 3 * N_PER_FORMAT], "reference_stats": "reference_stats.json",
              "calibration": "./checker.py:calibrate",
              "cut_point": "after the persona block, before the first day"},
    )
    save_suite(out, meta, {t.name: items[t.name.split("_")[1]] for t in tasks})
    shutil.copy(CHECKER_SRC, out / "checker.py")
    for t in tasks:
        print(f"{t.name:14s} {len(items[t.name.split('_')[1]])} prompts")
    print("building reference stats from", N_REF, "held-out docs …")
    stats = build_reference_stats(gen)
    (out / "reference_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print("wrote", out / "reference_stats.json", "->", len(stats["fields"]), "fields;", "generator commit", commit)


if __name__ == "__main__":
    main()
