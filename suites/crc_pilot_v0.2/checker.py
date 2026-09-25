"""CRC pilot v0.2 — suite-local checkers (frozen with the suite, hash-covered).

score_summary        — structured summarization to JSON. Generic: the expected keys
                       and values come from the item's target JSON, so base and hard
                       variants (with derived fields) share one scorer.
                       Tiers: parse / schema / values.
score_tool_call      — MCP-style execute_python call, static checks only:
                       call shape, code compiles, required tokens present (tags).
score_tool_call_hard — the emitted code is EXECUTED in a subprocess (10s timeout)
                       against the DataFrame from the item's CSV block; the printed
                       result must match the expected value(s) in `expect:` tags.
                       Tiers: call / runs / result.
                       NOTE: this runs model-generated Python on your machine
                       (subprocess + timeout, no other sandboxing).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


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
    schema = float(all(k in got for k in ref))
    if not schema:
        failed.append("schema: missing " + ", ".join(k for k in ref if k not in got))
    ok = 0
    for k, want in ref.items():
        if isinstance(want, (int, float)):
            tol = 0.005 if abs(want) < 3 else 0.15  # ratios exact-ish; percents/months allow rounding
            try:
                good = abs(float(got.get(k)) - want) <= tol
            except (TypeError, ValueError):
                good = False
        else:
            good = bool(_norm(want)) and _norm(want) in _norm(got.get(k, ""))
        ok += good
        if not good:
            failed.append(f"value.{k}")
    values = ok / len(ref)
    return {"score": round((1.0 + schema + values) / 3, 4),
            "tiers": {"parse": 1.0, "schema": schema, "values": round(values, 4)},
            "failed": failed}


def _parse_call(output: str):
    call = _first_json(output)
    args = call.get("arguments")
    if not (call.get("tool") == "execute_python" and isinstance(args, dict)
            and isinstance(args.get("code"), str) and args["code"].strip()):
        raise ValueError("not a single execute_python call")
    return args["code"]


def score_tool_call(item, output: str) -> dict:
    zero = {"call": 0.0, "compiles": 0.0, "content": 0.0}
    try:
        code = _parse_call(output)
    except Exception as e:
        return {"score": 0.0, "tiers": zero, "failed": [f"call: {e}"]}
    failed = []
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


def _csv_block(prompt: str) -> str:
    lines, keep = [], False
    for ln in prompt.split("\n"):
        if ln.startswith("patient_id,"):
            keep = True
        if keep:
            if not ln.strip():
                break
            lines.append(ln)
    if not lines:
        raise ValueError("no CSV block in prompt")
    return "\n".join(lines)


def score_tool_call_hard(item, output: str) -> dict:
    zero = {"call": 0.0, "runs": 0.0, "result": 0.0}
    try:
        code = _parse_call(output)
    except Exception as e:
        return {"score": 0.0, "tiers": zero, "failed": [f"call: {e}"]}
    runner = ("import io\nimport pandas as pd\n"
              f"df = pd.read_csv(io.StringIO({_csv_block(item.prompt)!r}))\n" + code)
    try:
        proc = subprocess.run([sys.executable, "-c", runner], capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return {"score": round(1 / 3, 4), "tiers": {"call": 1.0, "runs": 0.0, "result": 0.0},
                "failed": ["runs: timeout after 10s"]}
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()[-1:] or ["nonzero exit"]
        return {"score": round(1 / 3, 4), "tiers": {"call": 1.0, "runs": 0.0, "result": 0.0},
                "failed": [f"runs: {err[0]}"]}
    failed = []
    expects = [t.split(":", 1)[1] for t in item.tags if t.startswith("expect:")]
    out_nums = [float(x) for x in NUM_RE.findall(proc.stdout)]
    ok = 0
    for e in expects:
        try:
            want = float(e)
            good = any(abs(x - want) <= 0.05 for x in out_nums)
        except ValueError:
            good = e in proc.stdout
        ok += good
        if not good:
            failed.append(f"result: expected {e!r} not in output {proc.stdout.strip()!r:.80}")
    result = ok / len(expects) if expects else 1.0
    return {"score": round((1.0 + 1.0 + result) / 3, 4),
            "tiers": {"call": 1.0, "runs": 1.0, "result": round(result, 4)},
            "failed": failed}
