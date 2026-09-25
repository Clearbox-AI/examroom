"""CRC pilot suite — suite-local checkers (frozen with the suite, hash-covered).

score_summary   — structured summarization: the reply must be a JSON object with a
                  fixed schema; fields are compared to the ground truth one by one.
                  Tiers: parse (valid JSON object), schema (all keys), values.
score_tool_call — MCP-style tool use: the reply must be a single
                  {"tool": "execute_python", "arguments": {"code": …}} call.
                  Tiers: call (shape + tool name), compiles (code is valid Python),
                  content (the code touches every required column/literal, taken
                  from the item's `requires:` tags — chosen so any correct
                  solution must contain them).
"""

from __future__ import annotations

import json
import re

STR_KEYS = ("experimental_arm", "control_arm")
NUM_KEYS = ("n_patients", "orr_experimental_pct", "orr_control_pct",
            "median_pfs_experimental_months", "median_pfs_control_months", "hazard_ratio")
ALL_KEYS = STR_KEYS + NUM_KEYS


def _first_json(text: str):
    i = text.find("{")
    if i < 0:
        raise ValueError("no JSON object in output")
    obj, _ = json.JSONDecoder().raw_decode(text[i:])
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def score_summary(item, output: str) -> dict:
    ref = json.loads(item.target)
    try:
        got = _first_json(output)
    except Exception as e:
        return {"score": 0.0, "tiers": {"parse": 0.0, "schema": 0.0, "values": 0.0},
                "failed": [f"parse: {e}"]}
    failed = []
    schema = float(all(k in got for k in ALL_KEYS))
    if not schema:
        failed.append("schema: missing " + ", ".join(k for k in ALL_KEYS if k not in got))
    ok = 0
    for k in NUM_KEYS:
        tol = 0.005 if k == "hazard_ratio" else 0.05
        try:
            good = abs(float(got.get(k)) - float(ref[k])) <= tol
        except (TypeError, ValueError):
            good = False
        ok += good
        if not good:
            failed.append(f"value.{k}")
    for k in STR_KEYS:
        good = bool(_norm(ref[k])) and _norm(ref[k]) in _norm(got.get(k, ""))
        ok += good
        if not good:
            failed.append(f"value.{k}")
    values = ok / len(ALL_KEYS)
    return {"score": round((1.0 + schema + values) / 3, 4),
            "tiers": {"parse": 1.0, "schema": schema, "values": round(values, 4)},
            "failed": failed}


def score_tool_call(item, output: str) -> dict:
    zero = {"call": 0.0, "compiles": 0.0, "content": 0.0}
    try:
        call = _first_json(output)
    except Exception as e:
        return {"score": 0.0, "tiers": zero, "failed": [f"call: {e}"]}
    args = call.get("arguments")
    shaped = (call.get("tool") == "execute_python" and isinstance(args, dict)
              and isinstance(args.get("code"), str) and args["code"].strip())
    if not shaped:
        return {"score": 0.0, "tiers": zero, "failed": ["call: not a single execute_python call"]}
    failed = []
    code = args["code"]
    try:
        compile(code, "<tool_call>", "exec")
        compiles = 1.0
    except SyntaxError as e:
        compiles = 0.0
        failed.append(f"compiles: {e}")
    required = [t.split(":", 1)[1] for t in item.tags if t.startswith("requires:")]
    missing = [r for r in required if r not in code]
    content = 1.0 - len(missing) / len(required) if required else 1.0
    failed += [f"content: {m!r} not referenced" for m in missing]
    return {"score": round((1.0 + compiles + content) / 3, 4),
            "tiers": {"call": 1.0, "compiles": compiles, "content": round(content, 4)},
            "failed": failed}
