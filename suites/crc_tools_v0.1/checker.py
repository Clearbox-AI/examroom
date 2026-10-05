"""crc_tools v0.1 — episode checker (frozen with the suite, hash-covered).

`score_episode(item, episode)` scores one tool-use episode against the mock
surgery-risk service (mock_api.py).  The episode is what examroom's `agent` scorer
recorded: {"steps": [{"tool", "arguments", "result"}…], "final": str, "stop": str}.
The item's `target` is a JSON spec:

  calls     the reference trajectory [{"tool", "arguments"}…] (replayed by the
            conformance test, which must score 1.0)
  final     the reference answer
  patients  the patient(s) the request is about
  whatifs   the overrides the request asks for (each a {feature: value} dict)
  require   calls the model must make: {"tool" ("*": any), "args" (subset, after defaults),
            "error_ok" (the call may return an error: it is the point)}
  facts     what the answer must state: ["num", v | [v, …]] (any of the values; a
            probability is also found as a percentage), ["feature", code] (an input
            named in words), ["term", regex], ["near", regex, regex] (both in one
            sentence), ["knn"] (a surgery or watch-and-wait count, or the
            neighbours' ids, from a similar_patients result the model obtained)
  lang      en | it

Scoring:
  0  if there is no final answer, or the answer states a number that neither the
     request (in digits or words) nor an admissible tool result supports.  A result is admissible when it
     is about a patient the request names (or about no patient: search, model card)
     and, for a prediction with overrides, when those are a what-if the request asked
     for: another patient's data, or a what-if with substituted values (e.g. a CEA of
     32 run in place of a rejected 320), cannot back the answer.  Supported forms: the
     number as written, as a
     percentage or back, months as years or back, a complement, a difference of two
     returned probabilities, and a kNN count as a share of k.
  else 0.4 * share of required calls made + 0.6 * share of facts stated (only the
     facts share where no call is required), the facts part halved when the answer
     is not in the clinician's language.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

_spec = importlib.util.spec_from_file_location("crc_tools_v0_1_mock_api", Path(__file__).with_name("mock_api.py"))
api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(api)

# ------------------------------------------------------------------------ numbers in text
_NUM_TEXT = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?")
_NUM_ANY = re.compile(r"\d+(?:[.,]\d+)?")
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.M)
_CI_LEVEL = re.compile(r"95\s?%\s?(?:CI|IC|confidence|di confidenza)|(?:CI|IC)\s?(?:al\s)?95\s?%"
                       r"|confiden\w*\s(?:\w+\s){0,2}95\s?%", re.I)


def _numbers(text: str, decimal_comma: bool) -> list[tuple[float, int]]:
    out = []
    for m in _NUM_TEXT.finditer(text):
        s = m.group()
        if "," in s:
            if not decimal_comma:
                continue
            s = s.replace(",", ".")
        out.append((float(s), len(s.split(".")[1]) if "." in s else 0))
    return out


def _walk(obj, key=None):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)
    else:
        yield key, obj


def _derived(results: list[dict]) -> set[float]:
    """Numbers a narration may compute from tool results: differences between returned
    probabilities, and a kNN count as a share of k."""
    probs = {v for r in results for k, v in _walk(r) if k == "probability" and isinstance(v, float)}
    out = {round(abs(a - b), 4) for a in probs for b in probs if a != b}
    for r in results:
        s, k = r.get("summary"), r.get("k")
        if isinstance(s, dict) and k:
            out |= {round(n / k, 4) for n in s.values() if isinstance(n, int)}
    return out


def _allowed(context: str, extra: set[float]) -> set[float]:
    base = {float(m.group().replace(",", ".")) for m in _NUM_ANY.finditer(context)} | extra
    out = set()
    for v in base:
        out |= {v, v * 100, v / 100, v / 12, v * 12, 100 - v, 1 - v}
    return out


def unsupported_numbers(answer: str, context: str, results: list[dict], lang: str) -> list[str]:
    text = _CI_LEVEL.sub(" ", _LIST_MARKER.sub(" ", answer))
    allowed = _allowed(context, _derived(results))
    return [f"{v:g}" for v, d in _numbers(text, decimal_comma=(lang == "it"))
            if not any(abs(a - abs(v)) <= 0.5 * 10 ** -d + 1e-9 for a in allowed)]


NUMBER_WORDS = {
    "en": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"],
    "it": ["zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove", "dieci", "undici",
           "dodici"],
}


def _states(answer: str, value: float, lang: str) -> bool:
    """`value` appears in the answer as itself, as a percentage (a probability may be
    rounded to a whole percent) or, for a small count, as a word."""
    if value == int(value) and 0 <= value <= 12 and re.search(
            rf"\b{NUMBER_WORDS[lang][int(value)]}\b", answer, re.I):
        return True
    forms = {value} | ({value * 100} if 0 < value < 1 else set())
    for v, d in _numbers(answer, decimal_comma=(lang == "it")):
        if any(abs(f - abs(v)) <= 0.5 * 10 ** -d + 1e-9 for f in forms):
            return True
    return False


def _spelled(text: str) -> str:
    """Numbers the request spells out ("within two years", "entro un anno"), as digits."""
    found = [str(i) for lang in NUMBER_WORDS for i, w in enumerate(NUMBER_WORDS[lang])
             if re.search(rf"\b{w}\b", text, re.I)]
    if re.search(r"\b(?:a|one|un)\s+(?:year|anno)\b", text, re.I):
        found.append("1")
    return " ".join(found)


# ------------------------------------------------------------------------ words in text
FEATURE_TERMS = {
    "age": r"\bage\b|\baged\b|years? old|\betà\b|\banziano|\belderly|\bolder\b|\bgiovane|\byounger\b",
    "ct": r"\bc?t[234]\b|\bt[- ]?stage|\bt[- ]?category|tumou?r stage|stadio t\b|categoria t\b|stadio clinico t",
    "cn": r"\bc?n[012]\b|\bn[- ]?stage|nodal|lymph|linfonod",
    "distance_cm": r"distance|anal verge|margine anale|distanza|height|altezza",
    "emvi": r"emvi|extramural|extramurale",
    "cea_post": r"\bcea\b",
    "mrtrg": r"trg|tumou?r regression|regression grade|grado di regressione|regressione tumorale",
    "endoscopy": r"endoscop",
    "ctdna_post": r"ctdna|tumou?r dna|circulating (?:tumou?r )?dna|dna tumorale|dna circolante",
}

EN_WORDS = {"the", "and", "of", "is", "with", "for", "this", "that", "patient", "probability", "model",
            "risk", "are", "was", "at", "be", "in", "surgery", "similar"}
IT_WORDS = {"il", "la", "di", "che", "è", "per", "con", "del", "della", "paziente", "probabilità",
            "modello", "rischio", "una", "sono", "nel", "dei", "le", "gli", "alla", "chirurgia", "simili"}


def _language(text: str) -> str:
    words = re.findall(r"[a-zàèéìòù]+", text.lower())
    return "it" if sum(w in IT_WORDS for w in words) > sum(w in EN_WORDS for w in words) else "en"


def _fact_ok(fact: list, answer: str, lang: str, results: list[dict]) -> bool:
    kind, v = (fact + [None])[:2]
    if kind == "knn":   # the counts, or the neighbours listed one by one
        sims = [r for r in results if isinstance(r.get("summary"), dict)]
        counts = {n for r in sims for n in r["summary"].values()}
        return any(_states(answer, float(n), lang) for n in counts) or any(
            sum(n["patient_id"] in answer for n in r["neighbours"]) >= min(3, len(r["neighbours"])) for r in sims)
    if kind == "near":
        return any(re.search(v, s, re.I) and re.search(fact[2], s, re.I) for s in re.split(r"[.!?\n]", answer))
    if kind == "num":
        return any(_states(answer, float(x), lang) for x in (v if isinstance(v, list) else [v]))
    pattern = FEATURE_TERMS[v] if kind == "feature" else v
    return re.search(pattern, answer, re.I) is not None


# ------------------------------------------------------------------------ calls
def _same(want, got) -> bool:
    if isinstance(want, dict):
        return isinstance(got, dict) and set(want) == set(got) and all(_same(want[k], got[k]) for k in want)
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(want - got) <= 0.051
    return want == got


def _filled(step: dict, fixture: dict) -> dict:
    """Arguments with defaults filled in; an override equal to the patient's actual
    value changes nothing and is dropped."""
    args = {**api.DEFAULTS.get(step["tool"], {}), **(step["arguments"] or {})}
    p = fixture.get("patients", {}).get(args.get("patient_id"))
    if isinstance(args.get("overrides"), dict) and p:
        args["overrides"] = {f: v for f, v in args["overrides"].items() if p["features"].get(f) != v}
    return args


def _matches(req: dict, step: dict, fixture: dict) -> bool:
    if req["tool"] not in ("*", step["tool"]) or step["arguments"] is None:
        return False
    if "error" in step["result"] and not req.get("error_ok"):
        return False
    args = _filled(step, fixture)
    return all(k in args and _same(v, args[k]) for k, v in req.get("args", {}).items())


def _admissible(step: dict, spec: dict, fixture: dict) -> bool:
    if step["arguments"] is None:
        return False
    args = _filled(step, fixture)
    if args.get("patient_id") not in (None, *spec["patients"]):
        return False
    ov = args.get("overrides")
    return not ov or "error" in step["result"] or any(_same(w, ov) for w in spec.get("whatifs", []))


def score_episode(item, episode: dict) -> dict:
    spec = json.loads(item.target)
    lang, steps = spec["lang"], episode["steps"]
    final = episode.get("final") or ""
    zero = {"calls": 0.0, "facts": 0.0, "language": 0.0, "answered": 0.0, "faithful": 0.0}
    n_calls = len(steps)
    if episode.get("stop") != "final" or not final.strip():
        return {"score": 0.0, "tiers": zero, "n_calls": n_calls, "failed": [f"no final answer ({episode.get('stop')})"]}
    results = [s["result"] for s in steps if _admissible(s, spec, item.fixture or {})]
    context = "\n".join([item.prompt, _spelled(item.prompt)] + [json.dumps(r) for r in results])
    unsupported = unsupported_numbers(final, context, results, lang)
    if unsupported:
        return {"score": 0.0, "tiers": {**zero, "answered": 1.0}, "n_calls": n_calls,
                "failed": [f"unsupported number {u}" for u in unsupported]}
    req = spec.get("require", [])
    missed_calls = [r for r in req if not any(_matches(r, s, item.fixture or {}) for s in steps)]
    calls = 1 - len(missed_calls) / len(req) if req else 1.0
    facts = spec.get("facts", [])
    missed_facts = [f for f in facts if not _fact_ok(f, final, lang, results)]
    fact_share = 1 - len(missed_facts) / len(facts) if facts else 1.0
    language = float(_language(final) == lang)
    report = fact_share * (0.5 + 0.5 * language)
    score = 0.4 * calls + 0.6 * report if req else report
    failed = ([f"missing call {r['tool']}({json.dumps(r.get('args', {}))})" for r in missed_calls]
              + [f"missing {':'.join(map(str, f))}" for f in missed_facts] + ([] if language else [f"not written in {lang}"]))
    return {"score": round(score, 4), "n_calls": n_calls, "failed": failed,
            "tiers": {"calls": round(calls, 4), "facts": round(fact_share, 4), "language": language,
                      "answered": 1.0, "faithful": 1.0}}
