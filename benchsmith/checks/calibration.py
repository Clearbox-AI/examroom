"""Track 2 T4 — calibration: KL(model ‖ reference) per field over all sampled completions.

This is a distribution-level metric, so it runs over a whole *run* (every
completion the ledger stored for the schedule tasks), not per item. Same field
definitions and binning as scripts/author_schedules.py::build_reference_stats.
"""

from __future__ import annotations

import math
from collections import Counter

from benchsmith.checks.schedules import PARSERS, WEEKDAYS, ParseError


def bin15(hhmm: str) -> str:
    h, m = hhmm.split(":")
    return f"{int(h):02d}:{(int(m) // 15) * 15:02d}"


def collect(docs: list[dict]) -> dict[str, Counter]:
    c: dict[str, Counter] = {}

    def add(k, v):
        c.setdefault(k, Counter())[str(v)] += 1

    for doc in docs:
        p = doc["profile"]
        occs = p.get("occupants", [])
        add("household_size", len(occs))
        add("n_adults", sum(o.get("occupation") not in ("school_child", "student") for o in occs))
        add("n_children", sum(o.get("occupation") in ("school_child", "student") for o in occs))
        for o in occs:
            add("occupation", o.get("occupation"))
        add("heating", p.get("heating")); add("energy_label", p.get("energy_label"))
        add("surface_m2", p.get("surface_m2")); add("has_pv", str(p.get("pv")) == "true"); add("has_ev", str(p.get("ev")) == "true")
        for d in doc["week"]:
            wx = d["weather"]
            if wx.get("cloud"):
                add("cloud", wx["cloud"])
            if isinstance(wx.get("temp_c"), (int, float)):
                add("temp_bin", int(wx["temp_c"] // 5) * 5)
            for m in d["meals"]:
                if m.get("type") in ("breakfast", "lunch", "dinner") and isinstance(m.get("time"), str) and ":" in m["time"]:
                    try:
                        add(f"{m['type']}_time", bin15(m["time"]))
                    except ValueError:
                        pass
            add("lunch_present", any(m.get("type") == "lunch" for m in d["meals"]))
            add("meals_per_day", len(d["meals"])); add("showers_per_day", len(d["showers"]))
            add("appliances_per_day", len(d["appliances"])); add("trips_per_day", len(d["trips"]))
            add("ev_sessions_per_day", len(d["ev_charging"])); add("occupancy_slots_per_day", len(d["occupancy"]))
            add("weekday_trips_per_day" if d.get("day") in WEEKDAYS else "weekend_trips_per_day", len(d["trips"]))
            for s in d["showers"]:
                add("shower_duration", s.get("duration_min"))
            for a in d["appliances"]:
                add("appliance_name", a.get("name"))
                if isinstance(a.get("duration_min"), int):
                    add("appliance_duration", (a["duration_min"] // 30) * 30)
            for t in d["trips"]:
                add("trip_mode", t.get("mode")); add("trip_purpose", t.get("purpose"))
                if t.get("purpose") == "commute":
                    add("commute_mode", t.get("mode"))
    return c


def kl(model: Counter, ref: dict[str, float], eps: float = 1e-4) -> tuple[float, float, int]:
    """KL(model ‖ ref) in nats with additive smoothing over the union of support, plus total variation."""
    keys = set(model) | set(ref)
    n = sum(model.values())
    p = {k: (model.get(k, 0) / n) for k in keys}
    q = {k: ref.get(k, 0.0) for k in keys}
    # smooth both so unseen categories are finite
    ps = {k: (p[k] + eps) / (1 + eps * len(keys)) for k in keys}
    qs = {k: (q[k] + eps) / (1 + eps * len(keys)) for k in keys}
    d = sum(ps[k] * math.log(ps[k] / qs[k]) for k in keys)
    tv = 0.5 * sum(abs(p[k] - q[k]) for k in keys)
    return d, tv, n


def calibrate(items_by_id: dict, results: dict, reference: dict) -> dict:
    """items_by_id: item_id -> Item; results: item_id -> ItemResult (with detail.outputs)."""
    docs, parsed, total = [], 0, 0
    for iid, r in results.items():
        it = items_by_id.get(iid)
        if it is None or "outputs" not in r.detail:
            continue
        for out in r.detail["outputs"]:
            total += 1
            try:
                docs.append(PARSERS[it.category](it.prompt + out))
                parsed += 1
            except (ParseError, Exception):
                pass
    model = collect(docs)
    fields = {}
    for name, ref in reference["fields"].items():
        if name in model:
            d, tv, n = kl(model[name], ref)
            fields[name] = {"kl": round(d, 4), "tv": round(tv, 4), "n": n}
        else:
            fields[name] = {"kl": None, "tv": None, "n": 0}
    kls = [f["kl"] for f in fields.values() if f["kl"] is not None]
    return {"completions": total, "parsed": parsed, "mean_kl": round(sum(kls) / len(kls), 4) if kls else None,
            "fields": fields}
