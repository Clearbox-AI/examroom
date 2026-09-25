"""Track 2 — schedule coherence scorer (T1 parseability, T2 schema, T3 consistency).

Suite-local checker: frozen with the suite and covered by its content hash.
The generator `bin/generate_structured.py` IS the spec. Every T3 rule below cites
the generator line (commit 3f9d389) it is derived from. Scoring is programmatic:
the model's completion is appended to the prompt (the document prefix) and the
whole document is parsed, then checked.

Item score = mean of the three tier scores, where each tier is the fraction of
its checks that pass (T1 is all-or-nothing and gates T2/T3: an unparseable
document scores 0 on every tier). `score()` returns
{"score": float, "tiers": {"t1": .., "t2": .., "t3": ..}, "failed": [...]}.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WEEKDAYS = set(DAYS[:5])
# generator L40-48
OCCUPATIONS = {
    "office_worker": ("08:00", "17:45"), "teacher": ("07:30", "16:00"), "shopkeeper": ("08:30", "19:15"),
    "student": ("08:15", "17:00"), "school_child": ("08:00", "16:15"), "remote_worker": None, "retired": None,
}
HEATING_TYPES = {"gas_boiler", "heat_pump", "district_heating", "electric_radiators"}   # L50
ENERGY_LABELS = set("ABCDEF")                                                          # L51
CLOUD = {"clear", "partly_cloudy", "overcast"}                                         # L55
TRIP_MODES = {"car", "bike", "bus", "walk"}                                            # L56
APPLIANCES = {"washing_machine", "dishwasher", "oven", "air_conditioner", "heating"}   # L164-194
MEAL_TYPES = {"breakfast", "lunch", "dinner"}
PURPOSES = {"commute", "shopping", "leisure"}

HHMM = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$|^24:00$")


def mins(t: str) -> int | None:
    if not isinstance(t, str) or not HHMM.match(t):
        return None
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def within(t: str, centre: str, spread: int) -> bool:
    a, c = mins(t), mins(centre)
    # generator _hhmm clamps to [00:00, 23:59] (L65) so allow the clamp
    return a is not None and (abs(a - c) <= spread or (a in (0, 23 * 60 + 59)))


# ----------------------------------------------------------------------------- parsers
class ParseError(Exception):
    pass


def _cut_xml(text: str) -> str:
    end = text.find("</household>")
    if end < 0:
        raise ParseError("no </household>")
    return text[: end + len("</household>")]


def parse_xml(text: str) -> dict:
    try:
        root = ET.fromstring(_cut_xml(text))
    except ET.ParseError as e:
        raise ParseError(str(e))
    if root.tag != "household":
        raise ParseError("root is not <household>")
    prof = root.find("profile")
    week = root.find("week")
    if prof is None or week is None:
        raise ParseError("missing <profile> or <week>")

    def num(v, cast):
        try:
            return cast(v)
        except (TypeError, ValueError):
            return v

    profile = {
        "occupants": [{"age": num(o.get("age"), int), "occupation": o.get("occupation")} for o in prof.findall("occupant")],
        "surface_m2": num(prof.get("surface_m2"), int), "energy_label": prof.get("energy_label"),
        "heating": prof.get("heating"), "pv": prof.get("pv"), "ev": prof.get("ev"),
        "declared_occupants": num(prof.get("occupants"), int),
    }
    days = []
    for d in week.findall("day"):
        days.append({
            "day": d.get("name"),
            "weather": {"temp_c": num(d.get("temp_c"), float), "cloud": d.get("cloud"), "precip_mm": num(d.get("precip_mm"), float)},
            "occupancy": [{"from": s.get("from"), "to": s.get("to"), "people": num(s.get("people"), int)}
                          for s in d.findall("occupancy/slot")],
            "meals": [{"type": m.get("type"), "time": m.get("time"), "people": num(m.get("people"), int)}
                      for m in d.findall("meals/meal")],
            "showers": [{"time": s.get("time"), "duration_min": num(s.get("duration_min"), int)} for s in d.findall("showers/shower")],
            "appliances": [{"name": a.get("name"), "start": a.get("start"), "duration_min": num(a.get("duration_min"), int)}
                           for a in d.findall("appliances/appliance")],
            "ev_charging": [{"start": s.get("start"), "duration_min": num(s.get("duration_min"), int)} for s in d.findall("ev_charging/session")],
            "trips": [{"purpose": t.get("purpose"), "depart": t.get("depart"), "return": t.get("return"), "mode": t.get("mode")}
                      for t in d.findall("trips/trip")],
            "_sections": {c.tag for c in d},
        })
    return {"profile": profile, "week": days}


def parse_json(text: str) -> dict:
    try:
        doc, _ = json.JSONDecoder().raw_decode(text.lstrip())
    except json.JSONDecodeError as e:
        raise ParseError(str(e))
    if not isinstance(doc, dict) or "profile" not in doc or "week" not in doc:
        raise ParseError("missing profile/week")
    p = doc["profile"]
    if not isinstance(p, dict) or not isinstance(doc["week"], list):
        raise ParseError("bad profile/week types")
    profile = dict(p)
    profile["pv"] = str(p.get("pv")).lower()
    profile["ev"] = str(p.get("ev")).lower()
    profile.setdefault("occupants", [])
    profile["declared_occupants"] = len(profile["occupants"]) if isinstance(profile["occupants"], list) else None
    days = []
    for d in doc["week"]:
        if not isinstance(d, dict):
            raise ParseError("day is not an object")
        dd = {k: d.get(k, []) for k in ("occupancy", "meals", "showers", "appliances", "ev_charging", "trips")}
        dd["day"] = d.get("day")
        dd["weather"] = d.get("weather") if isinstance(d.get("weather"), dict) else {}
        dd["_sections"] = set(d)
        for k in ("occupancy", "meals", "showers", "appliances", "ev_charging", "trips"):
            if not isinstance(dd[k], list) or not all(isinstance(x, dict) for x in dd[k]):
                raise ParseError(f"{k} is not a list of objects")
        days.append(dd)
    return {"profile": profile, "week": days}


KV_LINE = re.compile(
    r"^(household: \S+"
    r"|profile: occupants=\d+ surface_m2=\d+ energy_label=\w heating=\w+ pv=(true|false) ev=(true|false)"
    r"|occupant: age=\d+ occupation=\w+"
    r"|day: \w+ \| temp_c=-?\d+(\.\d+)? cloud=\w+ precip_mm=\d+(\.\d+)?"
    r"|  occupancy: \d\d:\d\d-\d\d:\d\d people=\d+"
    r"|  meal: \w+ time=\d\d:\d\d people=\d+"
    r"|  shower: time=\d\d:\d\d duration_min=\d+"
    r"|  appliance: \w+ start=\d\d:\d\d duration_min=\d+"
    r"|  ev_charging: start=\d\d:\d\d duration_min=\d+"
    r"|  trip: \w+ depart=\d\d:\d\d return=\d\d:\d\d mode=\w+"
    r")$"
)


def parse_kv(text: str) -> dict:
    lines = text.split("\n")
    body = []
    for ln in lines:
        if ln.strip() == "" and body:
            break  # a blank line ends the structure (the generator's summary follows it)
        if ln.strip():
            body.append(ln.rstrip("\r"))
    bad = [ln for ln in body if not KV_LINE.match(ln)]
    if bad:
        raise ParseError(f"{len(bad)} line(s) violate the KV grammar, e.g. {bad[0]!r}")
    if not any(ln.startswith("day: ") for ln in body):
        raise ParseError("no day lines (the prompt prefix alone is grammatical; T1 requires a week to start)")
    profile = {"occupants": [], "declared_occupants": None}
    days: list[dict] = []
    kv = lambda s: dict(p.split("=", 1) for p in s.split() if "=" in p)
    for ln in body:
        if ln.startswith("profile: "):
            f = kv(ln[9:])
            profile.update(surface_m2=int(f["surface_m2"]), energy_label=f["energy_label"], heating=f["heating"],
                           pv=f["pv"], ev=f["ev"], declared_occupants=int(f["occupants"]))
        elif ln.startswith("occupant: "):
            f = kv(ln[10:])
            profile["occupants"].append({"age": int(f["age"]), "occupation": f["occupation"]})
        elif ln.startswith("day: "):
            name, rest = ln[5:].split(" | ", 1)
            f = kv(rest)
            days.append({"day": name, "weather": {"temp_c": float(f["temp_c"]), "cloud": f["cloud"], "precip_mm": float(f["precip_mm"])},
                         "occupancy": [], "meals": [], "showers": [], "appliances": [], "ev_charging": [], "trips": [],
                         "_sections": {"occupancy", "meals", "showers", "appliances", "ev_charging", "trips"}})
        elif ln.startswith("  "):
            if not days:
                raise ParseError("activity line before any day")
            key, rest = ln.strip().split(": ", 1)
            d = days[-1]
            if key == "occupancy":
                rng_, ppl = rest.split(" ")
                a, b = rng_.split("-")
                d["occupancy"].append({"from": a, "to": b, "people": int(ppl.split("=")[1])})
            elif key == "meal":
                typ, f = rest.split(" ", 1); f = kv(f)
                d["meals"].append({"type": typ, "time": f["time"], "people": int(f["people"])})
            elif key == "shower":
                f = kv(rest); d["showers"].append({"time": f["time"], "duration_min": int(f["duration_min"])})
            elif key == "appliance":
                name, f = rest.split(" ", 1); f = kv(f)
                d["appliances"].append({"name": name, "start": f["start"], "duration_min": int(f["duration_min"])})
            elif key == "ev_charging":
                f = kv(rest); d["ev_charging"].append({"start": f["start"], "duration_min": int(f["duration_min"])})
            elif key == "trip":
                purpose, f = rest.split(" ", 1); f = kv(f)
                d["trips"].append({"purpose": purpose, "depart": f["depart"], "return": f["return"], "mode": f["mode"]})
    return {"profile": profile, "week": days}


PARSERS = {"xml": parse_xml, "json": parse_json, "kv": parse_kv}


def detect_format(text: str) -> str:
    t = text.lstrip()
    if t.startswith("<"):
        return "xml"
    if t.startswith("{"):
        return "json"
    return "kv"


# ----------------------------------------------------------------------------- checks
@dataclass
class Checks:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def __call__(self, rule: str, ok: bool):
        (self.passed if ok else self.failed).append(rule)

    @property
    def score(self) -> float:
        n = len(self.passed) + len(self.failed)
        return len(self.passed) / n if n else 1.0


def check_t2(doc: dict) -> Checks:
    """Schema completeness: 7 days in order, required fields, times parse as HH:MM."""
    c = Checks()
    week = doc["week"]
    names = [d.get("day") for d in week]
    c("t2.seven_days_in_order", names == DAYS)                                      # L120 zip(DAYS, weather)
    p = doc["profile"]
    c("t2.profile_fields", all(k in p and p[k] is not None for k in ("surface_m2", "energy_label", "heating", "pv", "ev")))
    c("t2.profile_enums", p.get("heating") in HEATING_TYPES and p.get("energy_label") in ENERGY_LABELS
      and p.get("pv") in ("true", "false") and p.get("ev") in ("true", "false"))
    c("t2.occupants_listed", isinstance(p.get("occupants"), list) and len(p["occupants"]) >= 1
      and all(o.get("occupation") in OCCUPATIONS and isinstance(o.get("age"), int) for o in p["occupants"]))
    for d in week:
        c("t2.day_sections", {"occupancy", "meals", "showers", "appliances", "ev_charging", "trips"} <= d["_sections"])
        wx = d["weather"]
        c("t2.weather_fields", isinstance(wx.get("temp_c"), (int, float)) and wx.get("cloud") in CLOUD
          and isinstance(wx.get("precip_mm"), (int, float)))
        times = ([s.get("from") for s in d["occupancy"]] + [s.get("to") for s in d["occupancy"]]
                 + [m.get("time") for m in d["meals"]] + [s.get("time") for s in d["showers"]]
                 + [a.get("start") for a in d["appliances"]] + [s.get("start") for s in d["ev_charging"]]
                 + [t.get("depart") for t in d["trips"]] + [t.get("return") for t in d["trips"]])
        c("t2.times_hhmm", all(mins(t) is not None for t in times))
        c("t2.enums", all(m.get("type") in MEAL_TYPES for m in d["meals"])
          and all(a.get("name") in APPLIANCES for a in d["appliances"])
          and all(t.get("purpose") in PURPOSES and t.get("mode") in TRIP_MODES for t in d["trips"]))
        c("t2.durations_int", all(isinstance(x.get("duration_min"), int) for x in d["showers"] + d["appliances"] + d["ev_charging"]))
        c("t2.people_int", all(isinstance(x.get("people"), int) for x in d["occupancy"] + d["meals"]))
    return c


def _match_commutes(commutes: list[dict], windows: list[tuple[str, str]]) -> bool:
    """Is there a one-to-one assignment of commutes to workers' jittered windows?

    Jitter (±15 depart, ±20 return; L129-130) can swap the order of two workers'
    departures, so pairing by sorted time is wrong; try all assignments (n ≤ 5).
    """
    from itertools import permutations

    if len(commutes) != len(windows):
        return False
    fits = [[within(t["depart"], w[0], 15) and within(t["return"], w[1], 20) for w in windows] for t in commutes]
    return any(all(fits[i][j] for i, j in enumerate(perm)) for perm in permutations(range(len(windows))))


def check_t3(doc: dict) -> Checks:
    """Internal consistency. Each rule cites generate_structured.py (commit 3f9d389)."""
    c = Checks()
    p = doc["profile"]
    occs = [o for o in p.get("occupants", []) if o.get("occupation") in OCCUPATIONS]
    n = len(occs)
    has_ev = p.get("ev") == "true"
    workers = [o for o in occs if OCCUPATIONS[o["occupation"]]]
    c("t3.occupant_count_matches", p.get("declared_occupants") in (None, n))       # L240 occupants="{len}"
    c("t3.retired_age", all(o.get("age", 0) > 60 for o in occs if o["occupation"] == "retired"))  # L81
    c("t3.child_ages", all(4 <= o.get("age", -1) < 15 for o in occs if o["occupation"] == "school_child")
      and all(15 <= o.get("age", -1) <= 19 for o in occs if o["occupation"] == "student"))       # L85-88
    wash_days = oven_days = shop_trips = 0
    commute_weekday_ok, weekend_has_leisure_or_quiet = True, True

    for d in doc["week"]:
        weekday = d.get("day") in WEEKDAYS
        wx = d["weather"]
        temp = wx.get("temp_c") if isinstance(wx.get("temp_c"), (int, float)) else None
        rainy = isinstance(wx.get("precip_mm"), (int, float)) and wx["precip_mm"] > 2.0      # L121
        trips = [t for t in d["trips"] if mins(t.get("depart")) is not None and mins(t.get("return")) is not None]
        commutes = [t for t in trips if t.get("purpose") == "commute"]
        occ = [s for s in d["occupancy"] if mins(s.get("from")) is not None and mins(s.get("to")) is not None]

        # generic ordering: start < end for every interval                        # L136-142, L203-223
        c("t3.start_before_end", all(mins(s["from"]) < mins(s["to"]) for s in occ)
          and all(mins(t["depart"]) < mins(t["return"]) for t in trips))
        # occupancy slots contiguous from 00:00 to 24:00 and non-overlapping      # L136-142
        c("t3.occupancy_contiguous", bool(occ) and occ[0]["from"] == "00:00" and occ[-1]["to"] == "24:00"
          and all(occ[i]["to"] == occ[i + 1]["from"] for i in range(len(occ) - 1)))
        c("t3.occupancy_people_range", all(isinstance(s.get("people"), int) and 0 <= s["people"] <= n for s in occ))
        c("t3.occupancy_edges_full", bool(occ) and occ[0].get("people") == n and occ[-1].get("people") == n)  # L137,139

        # commutes: weekdays only, one per working occupant, inside jittered windows  # L123-131, L203-209
        if weekday:
            c("t3.commute_count_weekday", len(commutes) == len(workers))
            wins = [OCCUPATIONS[o["occupation"]] for o in workers]
            c("t3.commute_windows", _match_commutes(commutes, wins))
            if workers:
                # someone is away -> the middle occupancy slot shows fewer people     # L138
                c("t3.weekday_occupancy_dip", len(occ) == 3 and occ[1].get("people") == n - len(workers))
            else:
                c("t3.no_worker_single_slot", len(occ) == 1)                              # L142
        else:
            c("t3.no_commute_weekend", not commutes)                                     # L123
            c("t3.weekend_single_slot", len(occ) == 1)                                   # L141-142
        c("t3.commute_mode_car_if_ev", all(t.get("mode") == "car" for t in commutes) if has_ev else True)  # L206

        # meals                                                                   # L144-153
        meals = {m.get("type"): m for m in d["meals"]}
        c("t3.breakfast_and_dinner_present", "breakfast" in meals and "dinner" in meals)
        c("t3.meal_times", ("breakfast" not in meals or within(meals["breakfast"].get("time", ""), "07:15", 25))
          and ("lunch" not in meals or within(meals["lunch"].get("time", ""), "13:00", 20))
          and ("dinner" not in meals or within(meals["dinner"].get("time", ""), "19:30", 25)))
        c("t3.meals_people", meals.get("breakfast", {}).get("people") == n and meals.get("dinner", {}).get("people") == n)
        home_at_lunch = n - sum(1 for t in commutes if mins(t["depart"]) < 13 * 60 < mins(t["return"]))
        c("t3.lunch_iff_someone_home", ("lunch" in meals) == (home_at_lunch > 0))
        c("t3.lunch_people", "lunch" not in meals or meals["lunch"].get("people") == home_at_lunch)
        mt = [mins(meals[k]["time"]) for k in ("breakfast", "lunch", "dinner") if k in meals and mins(meals[k].get("time")) is not None]
        c("t3.meals_time_ordered", mt == sorted(mt))

        # showers: one per occupant, 5–12 min                                     # L155-162
        c("t3.showers_one_per_occupant", len(d["showers"]) == n)
        c("t3.shower_duration", all(isinstance(s.get("duration_min"), int) and 5 <= s["duration_min"] <= 12 for s in d["showers"]))

        # appliances                                                              # L164-194
        ap = {a.get("name"): a for a in d["appliances"]}
        c("t3.appliance_unique_per_day", len(ap) == len(d["appliances"]))
        rules = {"washing_machine": ("18:15", 45, {60, 90, 120}), "dishwasher": ("20:30", 20, {50, 70}),
                 "oven": ("18:45", 20, set(range(30, 76))), "air_conditioner": ("13:30", 0, set(range(180, 361))),
                 "heating": ("06:00", 0, set(range(600, 901)))}
        c("t3.appliance_windows", all(
            within(a.get("start", ""), rules[nm][0], rules[nm][1]) and a.get("duration_min") in rules[nm][2]
            for nm, a in ap.items() if nm in rules))
        if temp is not None:
            c("t3.ac_iff_hot", ("air_conditioner" in ap) == (temp >= 27.0))               # L183
            c("t3.heating_iff_cold", ("heating" in ap) == (temp <= 7.0))                  # L189
        if isinstance(p.get("surface_m2"), int) and p["surface_m2"] >= 75:
            c("t3.dishwasher_if_large_home", "dishwasher" in ap)                          # L171
        wash_days += "washing_machine" in ap
        oven_days += "oven" in ap

        # EV                                                                      # L196-201
        c("t3.ev_only_if_has_ev", has_ev or not d["ev_charging"])
        c("t3.ev_sessions_valid", len(d["ev_charging"]) <= 1 and all(
            within(s.get("start", ""), "19:00", 60) and s.get("duration_min") in (120, 180, 240) for s in d["ev_charging"]))
        if has_ev and commutes:
            c("t3.ev_charges_after_commute", len(d["ev_charging"]) == 1)

        # other trips                                                             # L210-223
        shops = [t for t in trips if t.get("purpose") == "shopping"]
        leis = [t for t in trips if t.get("purpose") == "leisure"]
        shop_trips += len(shops)
        c("t3.shopping_window", all(within(t["depart"], "10:30", 45) and within(t["return"], "11:45", 45) for t in shops))
        c("t3.shopping_only_if_home", not shops or home_at_lunch > 0)
        c("t3.leisure_weekend_dry_only", not leis or (not weekday and not rainy))
        c("t3.leisure_window", all(within(t["depart"], "14:30", 60) and within(t["return"], "17:30", 60) for t in leis))
        c("t3.at_most_one_leisure", len(leis) <= 1)
        c("t3.trip_purpose_counts", len([t for t in trips if t.get("purpose") not in PURPOSES]) == 0)

    # week-level                                                                  # L116-118
    c("t3.wash_days_2_or_3", wash_days in (2, 3))
    c("t3.oven_days_2_or_3", oven_days in (2, 3))
    c("t3.shopping_at_most_once", shop_trips <= 1)
    return c


def score(item, output: str) -> dict:
    """benchsmith `program` checker entry point: checker = "./checker.py:score"."""
    text = item.prompt + output
    fmt = item.category or detect_format(text)
    try:
        doc = PARSERS[fmt](text)
        t1 = 1.0
    except ParseError as e:
        return {"score": 0.0, "tiers": {"t1": 0.0, "t2": 0.0, "t3": 0.0}, "failed": [f"t1.parse: {e}"]}
    t2 = check_t2(doc)
    t3 = check_t3(doc)
    tiers = {"t1": t1, "t2": round(t2.score, 4), "t3": round(t3.score, 4)}
    return {"score": round((t1 + t2.score + t3.score) / 3, 4), "tiers": tiers,
            "failed": sorted(set(t2.failed + t3.failed))}


# ============================================================================
# T4 calibration — KL(model ‖ reference) per field over all sampled completions.
#
# Distribution-level: runs over a whole *run* (every completion the ledger
# stored for the schedule tasks), not per item. Same field definitions and
# binning as scripts/author_schedules.py::build_reference_stats.
# Entry point for `benchsmith calibrate`: meta["calibration"] = "./checker.py:calibrate".
# ============================================================================

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
