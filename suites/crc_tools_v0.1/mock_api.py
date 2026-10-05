"""crc_tools v0.1 — mock surgery-risk model service (frozen with the suite, hash-covered).

A stand-in for the API / MCP endpoint a CRC decision-support model would expose: the
probability that a rectal cancer patient needs surgery (TME) after total neoadjuvant
therapy, its SHAP explanation, the most similar past patients (kNN) and the model
card.  It is deterministic and fully synthetic: each item's `fixture` holds the
patients, their kNN neighbours and the model/retriever versions, and this module
computes everything else.  The model is a fixed logistic regression, so its SHAP
values (interventional, against a fixed background) are exact:
    shap_f = term_f(x_f) - E[term_f],   logit(p) = base + sum_f shap_f

Interface used by examroom's `agent` scorer:
    TOOLS                          MCP-style tool definitions {name, description, inputSchema}
    call(name, arguments, fixture) -> JSON-serialisable result, {"error", "code", …} on failure
Arguments are validated strictly against the schemas (unknown or missing arguments,
wrong types and out-of-range values are errors the model can read and recover from).
"""

from __future__ import annotations

import math

FEATURES = {
    "age": {"label": "Age (years)", "type": "number", "min": 18, "max": 100},
    "ct": {"label": "Clinical T stage", "type": "category", "values": ["T2", "T3", "T4"]},
    "cn": {"label": "Clinical N stage", "type": "category", "values": ["N0", "N1", "N2"]},
    "distance_cm": {"label": "Distance from anal verge (cm)", "type": "number", "min": 0, "max": 20},
    "emvi": {"label": "EMVI on baseline MRI", "type": "category", "values": ["negative", "positive"]},
    "cea_post": {"label": "CEA after TNT (ng/mL)", "type": "number", "min": 0, "max": 100},
    "mrtrg": {"label": "mrTRG on restaging MRI", "type": "integer", "min": 1, "max": 5},
    "endoscopy": {"label": "Endoscopic response after TNT", "type": "category",
                  "values": ["complete", "near_complete", "incomplete"]},
    "ctdna_post": {"label": "ctDNA after TNT", "type": "category", "values": ["negative", "positive"]},
}
TRAINING_RANGE = {"age": (32, 84), "distance_cm": (0, 12), "cea_post": (0, 40)}

_CAT = {
    "ct": {"T2": 0.0, "T3": 0.4, "T4": 0.9},
    "cn": {"N0": 0.0, "N1": 0.2, "N2": 0.5},
    "emvi": {"negative": 0.0, "positive": 0.5},
    "endoscopy": {"complete": 0.0, "near_complete": 0.7, "incomplete": 1.6},
    "ctdna_post": {"negative": 0.0, "positive": 0.9},
}
_LIN = {"age": -0.02, "distance_cm": 0.07, "cea_post": 0.12, "mrtrg": 0.45}
# background (training cohort) distribution the SHAP values are taken against
_BG_MEAN = {"age": 64.0, "distance_cm": 6.5, "cea_post": 3.2, "mrtrg": 2.4}
_BG_FREQ = {
    "ct": {"T2": 0.15, "T3": 0.65, "T4": 0.20},
    "cn": {"N0": 0.30, "N1": 0.45, "N2": 0.25},
    "emvi": {"negative": 0.6, "positive": 0.4},
    "endoscopy": {"complete": 0.3, "near_complete": 0.3, "incomplete": 0.4},
    "ctdna_post": {"negative": 0.75, "positive": 0.25},
}
_INTERCEPT = {12: -3.6, 24: -3.0}
HORIZONS = (12, 24)
DEFAULTS = {"predict_surgery": {"horizon_months": 24, "overrides": {}},
            "explain_prediction": {"horizon_months": 24},
            "similar_patients": {"k": 5}}


def _term(f: str, v) -> float:
    if f in _CAT:
        return _CAT[f][v]
    return _LIN[f] * (min(v, 20.0) if f == "cea_post" else v)


def _expected(f: str) -> float:
    if f in _CAT:
        return sum(p * _CAT[f][c] for c, p in _BG_FREQ[f].items())
    return _term(f, _BG_MEAN[f])


def _expit(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def base_logit(h: int) -> float:
    return _INTERCEPT[h] + sum(_expected(f) for f in FEATURES)


def shap_values(x: dict) -> dict[str, float]:
    return {f: _term(f, x[f]) - _expected(f) for f in FEATURES}


def probability(x: dict, h: int) -> float:
    return _expit(base_logit(h) + sum(shap_values(x).values()))


# ------------------------------------------------------------------------------- schemas
_PATIENT = {"type": "string", "description": "Patient identifier as written in the request (P- and four digits)"}
_HORIZON = {"type": "integer", "enum": list(HORIZONS),
            "description": "Prediction horizon in months after TNT completion (default 24)"}
_OVERRIDE_PROPS = {
    f: ({"type": "string", "enum": s["values"]} if s["type"] == "category"
        else {"type": s["type"], "minimum": s["min"], "maximum": s["max"]}) | {"description": s["label"]}
    for f, s in FEATURES.items()
}

TOOLS = [
    {"name": "find_patients",
     "description": "Search the patients currently followed by the MDT after total neoadjuvant therapy (TNT). "
                    "Filters are optional and combined with AND. Returns patient_id, age, sex and TNT completion date.",
     "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
         "age": {"type": "integer", "description": "Age in years"},
         "sex": {"type": "string", "enum": ["F", "M"]}}}},
    {"name": "get_patient_journey",
     "description": "The patient's care pathway so far: diagnosis, baseline MRI, CEA, TNT regimen and dates, "
                    "restaging MRI, endoscopy and ctDNA, in date order.",
     "inputSchema": {"type": "object", "additionalProperties": False, "required": ["patient_id"],
                     "properties": {"patient_id": _PATIENT}}},
    {"name": "predict_surgery",
     "description": "Surgery-risk model: probability that the patient undergoes TME surgery within the horizon "
                    "after TNT completion, with a 95% confidence interval and any warnings. `overrides` replaces "
                    "model inputs with hypothetical values for a what-if analysis; leave it out for the patient's "
                    "actual prediction.",
     "inputSchema": {"type": "object", "additionalProperties": False, "required": ["patient_id"],
                     "properties": {"patient_id": _PATIENT, "horizon_months": _HORIZON,
                                    "overrides": {"type": "object", "additionalProperties": False,
                                                  "properties": _OVERRIDE_PROPS,
                                                  "description": "Hypothetical model inputs, {feature: value}"}}}},
    {"name": "explain_prediction",
     "description": "SHAP explanation of the surgery-risk prediction: each input's contribution on the log-odds "
                    "scale (positive raises the probability of surgery), sorted by absolute size, and the "
                    "cohort base probability.",
     "inputSchema": {"type": "object", "additionalProperties": False, "required": ["patient_id"],
                     "properties": {"patient_id": _PATIENT, "horizon_months": _HORIZON}}},
    {"name": "similar_patients",
     "description": "The k most similar past patients (kNN on the model inputs) with their outcome: surgery "
                    "(with months from TNT completion to surgery) or watch-and-wait (with months of follow-up "
                    "without surgery), and a count of each.",
     "inputSchema": {"type": "object", "additionalProperties": False, "required": ["patient_id"],
                     "properties": {"patient_id": _PATIENT,
                                    "k": {"type": "integer", "minimum": 1, "maximum": 20,
                                          "description": "Number of neighbours (default 5)"}}}},
    {"name": "model_card",
     "description": "The surgery-risk model's card: version, outcome definition, training cohort, performance, "
                    "intended use and limitations.",
     "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}}},
]
_SCHEMAS = {t["name"]: t["inputSchema"] for t in TOOLS}


def _err(code: str, message: str, **extra) -> dict:
    return {"error": message, "code": code, **extra}


def _type_ok(v, spec: dict) -> bool:
    t = spec.get("type")
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "string":
        return isinstance(v, str)
    if t == "object":
        return isinstance(v, dict)
    return True


def _check(schema: dict, args: dict, where: str = "") -> str | None:
    """A message for the first schema violation in `args`, or None."""
    props = schema.get("properties", {})
    unknown = sorted(set(args) - set(props))
    if unknown:
        return f"unknown {'override' if where else 'argument'}(s) {', '.join(unknown)}; allowed: {', '.join(props)}"
    missing = [r for r in schema.get("required", []) if r not in args]
    if missing:
        return f"missing required argument(s) {', '.join(missing)}"
    for k, v in args.items():
        spec = props[k]
        if not _type_ok(v, spec):
            return f"{where}{k} must be of type {spec['type']}"
        if "enum" in spec and v not in spec["enum"]:
            return f"{where}{k} must be one of {', '.join(map(str, spec['enum']))}"
        if "minimum" in spec and not spec["minimum"] <= v <= spec["maximum"]:
            return f"{where}{k} must be between {spec['minimum']} and {spec['maximum']}"
        if spec.get("type") == "object" and "properties" in spec:
            msg = _check(spec, v, where=f"{k}.")
            if msg:
                return msg
    return None


# ------------------------------------------------------------------------------- tools
def _patient(fixture: dict, pid: str):
    return fixture["patients"].get(pid)


def journey(pid: str, p: dict) -> dict:
    x = p["features"]
    ev = [
        {"date": p["dx_date"], "event": "diagnosis",
         "detail": f"rectal adenocarcinoma, {x['distance_cm']:g} cm from the anal verge"},
        {"date": p["dx_date"], "event": "baseline_mri",
         "findings": {"ct": x["ct"], "cn": x["cn"], "emvi": x["emvi"], "distance_cm": x["distance_cm"]}},
        {"date": p["dx_date"], "event": "lab", "test": "CEA", "value": p["cea_baseline"], "unit": "ng/mL"},
        {"date": p["tnt_start"], "event": "tnt_start", "regimen": p["regimen"]},
        {"date": p["tnt_end"], "event": "tnt_end"},
    ]
    if p.get("restaging_date"):
        ev += [
            {"date": p["restaging_date"], "event": "lab", "test": "CEA", "value": x["cea_post"], "unit": "ng/mL"},
            {"date": p["restaging_date"], "event": "restaging_mri", "mrtrg": x["mrtrg"]},
            {"date": p["restaging_date"], "event": "endoscopy", "response": x["endoscopy"]},
            {"date": p["restaging_date"], "event": "ctdna", "result": x["ctdna_post"]},
        ]
    else:
        ev += [
            {"date": p["tnt_end"], "event": "lab", "test": "CEA", "value": x["cea_post"], "unit": "ng/mL"},
            {"date": p["tnt_end"], "event": "ctdna", "result": x["ctdna_post"]},
            {"date": p["tnt_end"], "event": "note", "detail": "restaging MRI and endoscopy not yet performed"},
        ]
    return {"patient_id": pid, "age": x["age"], "sex": p["sex"], "events": ev}


def _inputs(pid: str, p: dict, overrides: dict):
    x = {**p["features"], **overrides}
    missing = [f for f in FEATURES if x.get(f) is None]
    if missing:
        return None, _err("insufficient_data", f"insufficient data for a prediction on {pid}",
                          missing=[{"feature": f, "label": FEATURES[f]["label"]} for f in missing],
                          hint="restaging MRI, endoscopy, CEA and ctDNA are due after TNT completion")
    return x, None


def _warnings(x: dict) -> list[str]:
    out = []
    for f, (lo, hi) in TRAINING_RANGE.items():
        if not lo <= x[f] <= hi:
            out.append(f"out_of_distribution: {f}={x[f]:g} is outside the training range {lo}-{hi}; "
                       "the estimate is less reliable")
    return out


def predict(pid: str, p: dict, h: int, overrides: dict, version: str) -> dict:
    x, err = _inputs(pid, p, overrides)
    if err:
        return err
    warn = _warnings(x)
    z = base_logit(h) + sum(shap_values(x).values())
    se = 0.30 + (0.15 if warn else 0.0)
    out = {"patient_id": pid, "model": "surgery-risk", "model_version": version, "horizon_months": h,
           "probability": round(_expit(z), 2),
           "ci95": [round(_expit(z - 1.96 * se), 2), round(_expit(z + 1.96 * se), 2)],
           "inputs": x, "warnings": warn}
    if overrides:
        out["overridden"] = sorted(overrides)
        out["warnings"] = warn + [f"what-if: {', '.join(sorted(overrides))} replaced by hypothetical values"]
    return out


def explain(pid: str, p: dict, h: int, version: str) -> dict:
    x, err = _inputs(pid, p, {})
    if err:
        return err
    sv = shap_values(x)
    contrib = sorted(({"feature": f, "label": FEATURES[f]["label"], "value": x[f], "shap": round(v, 2)}
                      for f, v in sv.items()), key=lambda c: -abs(c["shap"]))
    return {"patient_id": pid, "model_version": version, "horizon_months": h,
            "method": "SHAP, exact for this logistic model; log-odds scale",
            "base_probability": round(_expit(base_logit(h)), 2),
            "probability": round(probability(x, h), 2), "contributions": contrib}


def similar(pid: str, p: dict, k: int, fixture: dict) -> dict:
    nb = fixture["neighbours"][pid][:k]
    return {"patient_id": pid, "retriever_version": fixture["retriever_version"], "k": len(nb), "neighbours": nb,
            "summary": {"n_surgery": sum(n["outcome"] == "surgery" for n in nb),
                        "n_watch_and_wait": sum(n["outcome"] == "watch_and_wait" for n in nb)}}


def model_card(version: str) -> dict:
    return {"model": "surgery-risk", "version": version,
            "outcome": "TME surgery within 12 or 24 months of TNT completion (for regrowth, incomplete response "
                       "or patient choice)",
            "inputs": {f: s["label"] for f, s in FEATURES.items()},
            "training_cohort": {"n": 1184, "sites": 3, "period": "2015-2023"},
            "performance": {"auroc_24m": 0.81, "calibration_slope": 0.94},
            "intended_use": "decision support for MDT discussion after TNT; not a substitute for clinical judgement",
            "limitations": ["not validated outside the training ranges of age, distance and CEA",
                            "does not predict surgical complications, stoma or survival"]}


def call(name: str, args: dict, fixture: dict) -> dict:
    if name not in _SCHEMAS:
        return _err("unknown_tool", f"unknown tool {name!r}; available: {', '.join(_SCHEMAS)}")
    msg = _check(_SCHEMAS[name], args)
    if msg:
        return _err("invalid_arguments", msg)
    a = {**DEFAULTS.get(name, {}), **args}
    version = fixture["model_version"]
    if name == "model_card":
        return model_card(version)
    if name == "find_patients":
        return {"patients": [{"patient_id": pid, "age": p["features"]["age"], "sex": p["sex"],
                              "tnt_completed": p["tnt_end"]}
                             for pid, p in sorted(fixture["patients"].items())
                             if a.get("age") in (None, p["features"]["age"]) and a.get("sex") in (None, p["sex"])]}
    p = _patient(fixture, a["patient_id"])
    if p is None:
        return _err("not_found", f"patient {a['patient_id']} not found")
    if name == "get_patient_journey":
        return journey(a["patient_id"], p)
    if name == "predict_surgery":
        return predict(a["patient_id"], p, a["horizon_months"], a["overrides"], version)
    if name == "explain_prediction":
        return explain(a["patient_id"], p, a["horizon_months"], version)
    return similar(a["patient_id"], p, a["k"], fixture)
