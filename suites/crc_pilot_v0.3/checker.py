"""CRC pilot v0.3 — suite-local checkers (frozen with the suite, hash-covered).

score_summary        — structured summarization to JSON. Generic: the expected keys
                       and values come from the item's target JSON, so base and hard
                       variants (with derived fields) share one scorer.
                       Tiers: parse / schema / values.
score_tool_call      — MCP-style execute_python call, static checks only:
                       call shape, code compiles, required tokens present (tags).
score_tool_call_hard — the emitted code is EXECUTED in a subprocess (10s timeout)
                       against the DataFrame from the item's CSV block. The LAST
                       non-empty line of stdout must carry the value from the item's
                       `expect:` tag and no other number, so dumping the frame, a
                       boolean mask or an indexed Series does not score (v0.2 matched
                       anywhere in stdout). Tiers: call / runs / result.
                       Needs pandas in the interpreter running benchsmith
                       (`uv sync --extra crc`). A missing pandas RAISES instead of
                       being scored as a model failure; the function carries a
                       `preflight` hook the runner calls before the first model call.
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


# ----------------------------------------------------------------- hard tool tier (executed)
_ENV_OK = False


def preflight_tool_hard() -> None:
    """Raise if the interpreter cannot import pandas: an environment gap, not a model failure."""
    global _ENV_OK
    if _ENV_OK:
        return
    proc = subprocess.run([sys.executable, "-c", "import pandas"], capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["nonzero exit"]
        raise RuntimeError(f"tool_hard checker: {sys.executable} cannot import pandas ({tail[0]}); "
                           "install it with `uv sync --extra crc`")
    _ENV_OK = True


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


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _matches(expect: str, line: str) -> bool:
    """`line` carries `expect` and nothing else numeric.

    Numeric expect: exactly one number on the line, within 0.05. String expect (a
    patient id): present as a whole token, and no other number on the line. Prose
    around the value is fine ("Difference: -2.03"); a frame dump, a boolean mask or
    a Series printed with its index is not.
    """
    try:
        want = float(expect)
    except ValueError:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(expect)}(?![A-Za-z0-9_])", line) is None:
            return False
        return not NUM_RE.findall(line.replace(expect, ""))
    nums = NUM_RE.findall(line)
    return len(nums) == 1 and abs(float(nums[0]) - want) <= 0.05


def score_tool_call_hard(item, output: str) -> dict:
    preflight_tool_hard()
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
    expects = [t.split(":", 1)[1] for t in item.tags if t.startswith("expect:")]
    line = _last_line(proc.stdout)
    failed = [f"result: expected {e!r}, last output line {line!r:.80}" for e in expects if not _matches(e, line)]
    result = 1.0 - len(failed) / len(expects) if expects else 1.0
    return {"score": round((1.0 + 1.0 + result) / 3, 4),
            "tiers": {"call": 1.0, "runs": 1.0, "result": round(result, 4)},
            "failed": failed}


score_tool_call_hard.preflight = preflight_tool_hard
