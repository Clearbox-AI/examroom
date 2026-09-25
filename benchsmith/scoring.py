"""Scorers, one per Kind. Each returns (score in [0,1], detail dict)."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import math
import re
import string
import sys
from pathlib import Path
from typing import Any

from benchsmith.backends.base import Backend
from benchsmith.items import Item, Kind, Task

LETTERS = "ABCDEFGH"


# --- checkers -----------------------------------------------------------------
_MODULE_CACHE: dict[str, Any] = {}


def checker_module(mod: str, root: Path | None = None):
    """Resolve the module half of a checker spec.

    Two forms:
      "package.module"   importable module (a checker shipped as installed code)
      "<path>.py"        file relative to the suite directory — the normal case,
                         so a frozen suite carries its own scoring code
    """
    if not mod.endswith(".py"):
        return importlib.import_module(mod)
    path = (Path(root) / mod if root is not None else Path(mod)).resolve()
    key = str(path)
    if key not in _MODULE_CACHE:
        if not path.is_file():
            raise FileNotFoundError(f"checker file {mod!r} not found under {root}")
        name = f"benchsmith_checker_{hashlib.sha1(key.encode()).hexdigest()[:12]}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULE_CACHE[key] = module
    return _MODULE_CACHE[key]


def load_checker(spec: str, root: Path | None = None):
    """Resolve "module:function". A checker function may carry a `preflight`
    attribute (a zero-argument callable); the runner calls it once per run before
    the first model call, so a checker that needs an external tool or library can
    fail loudly up front instead of scoring an environment failure as a model failure."""
    mod, fn = spec.split(":")
    return getattr(checker_module(mod, root), fn)


# --- likelihood ---------------------------------------------------------------
def _normalise(lp: float, ntok: int, text: str, mode: str) -> float:
    if mode == "sum":
        return lp
    if mode == "mean":            # per-token mean (the brief's default)
        return lp / max(ntok, 1)
    if mode == "byte":            # per-byte, tokenizer-independent alternative
        return lp / max(len(text.encode()), 1)
    raise ValueError(f"unknown normalize mode {mode!r}")


def score_choice_loglik(item: Item, task: Task, backend: Backend) -> tuple[float, dict]:
    mode = task.scoring.get("normalize", "mean")
    raw = backend.loglik(item.prompt, item.options)
    scores = [_normalise(lp, n, o, mode) for (lp, n), o in zip(raw, item.options)]
    best = max(range(len(scores)), key=scores.__getitem__)
    n_best = sum(1 for s in scores if s == scores[best])
    correct = best == item.answer_idx and n_best == 1
    return float(correct), {
        "pred_idx": best, "scores": [round(s, 4) for s in scores],
        "ntok": [n for _, n in raw], "tie": n_best > 1,
    }


def score_pair_loglik(item: Item, task: Task, backend: Backend) -> tuple[float, dict]:
    mode = task.scoring.get("normalize", "mean")
    (lg, ng), (lb, nb) = backend.loglik(item.prompt, [item.good, item.bad])
    g, b = _normalise(lg, ng, item.good, mode), _normalise(lb, nb, item.bad, mode)
    return float(g > b), {"good": round(g, 4), "bad": round(b, 4), "margin": round(g - b, 4)}


# --- generation ---------------------------------------------------------------
def mcq_prompt(item: Item, task: Task) -> str:
    instr = task.scoring.get(
        "instruction", "Answer with the single letter of the correct option and nothing else."
    )
    opts = "\n".join(f"{LETTERS[i]}. {o.strip()}" for i, o in enumerate(item.options))
    return f"{item.prompt.rstrip()}\n\n{opts}\n\n{instr}"


def parse_letter(text: str, n_options: int) -> int | None:
    m = re.search(rf"\b([{LETTERS[:n_options]}])\b", text.strip().upper())
    return LETTERS.index(m.group(1)) if m else None


def _sampled(item: Item, task: Task, backend: Backend, prompt: str, judge) -> tuple[float, dict]:
    """Generate `item.samples` completions and average the judge over them.

    A judge may return a float in [0, 1] or a dict {"score": float, ...}; extra
    dict keys (e.g. per-tier breakdowns) are kept in the result detail.
    """
    temperature = task.scoring.get("temperature", 0.0 if item.samples == 1 else 0.7)
    max_tokens = task.scoring.get("max_tokens", item.max_tokens)
    instruction = task.scoring.get("instruction")
    if instruction:  # run-time suffix for chat models that need "answer only" steering
        prompt = f"{prompt}\n\n{instruction}"
    outs = backend.generate(prompt, max_tokens, temperature, item.samples)
    verdicts = [judge(o) for o in outs]
    scores = [v["score"] if isinstance(v, dict) else float(v) for v in verdicts]
    detail: dict[str, Any] = {"outputs": outs, "per_sample": scores}
    extras = [v for v in verdicts if isinstance(v, dict)]
    if extras:
        detail["judge"] = extras if len(extras) > 1 else extras[0]
    return sum(scores) / len(scores), detail


def score_choice_generate(item: Item, task: Task, backend: Backend) -> tuple[float, dict]:
    n = len(item.options)
    return _sampled(item, task, backend, mcq_prompt(item, task),
                    lambda o: float(parse_letter(o, n) == item.answer_idx))


def normalise_text(s: str) -> str:
    s = s.strip().lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def score_exact(item: Item, task: Task, backend: Backend) -> tuple[float, dict]:
    tgt = normalise_text(item.target)
    return _sampled(item, task, backend, item.prompt, lambda o: float(normalise_text(o) == tgt))


def score_regex(item: Item, task: Task, backend: Backend) -> tuple[float, dict]:
    rx = re.compile(item.target, re.I | re.S)
    return _sampled(item, task, backend, item.prompt, lambda o: float(rx.search(o) is not None))


_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_NUM = re.compile(
    r"(?P<ten_s>[-+]?)10\s*(?:\^|\*\*)\s*\{?(?P<ten_e>[-+]?\d+)\}?"          # bare 10^n
    r"|(?P<mant>[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))"                        # 1,234.5 or .69
    r"(?:\s*(?:[×*·]|\\times|[xX])\s*10\s*(?:\^|\*\*)\s*\{?(?P<pexp>[-+]?\d+)\}?"  # a × 10^n
    r"|[eE](?P<eexp>[-+]?\d+))?"                                             # a e n
)


_DECIMAL_COMMA = re.compile(r"(\d),(\d{1,2})(?!\d)")  # 68,6 -> 68.6 ; 1,234 stays a thousands group


def _prep(text: str) -> str:
    text = text.replace("−", "-")
    return re.sub(  # 10⁴ -> 10^4 so scientific notation survives
        r"10\s*([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: "10^" + m.group(1).translate(_SUP), text
    )


def _value(m: re.Match) -> float:
    if m.group("ten_e") is not None:
        v = 10.0 ** int(m.group("ten_e"))
        return -v if m.group("ten_s") == "-" else v
    v = float(m.group("mant").replace(",", ""))
    exp = m.group("pexp") or m.group("eexp")
    return v * 10.0 ** int(exp) if exp else v


def first_number(text: str) -> float | None:
    m = _NUM.search(_prep(text))
    return _value(m) if m else None


def last_number(text: str) -> float | None:
    ms = list(_NUM.finditer(_prep(text)))
    return _value(ms[-1]) if ms else None


def score_numeric(item: Item, task: Task, backend: Backend) -> tuple[float, dict]:
    target = float(item.target)
    tol = item.tolerance if item.tolerance is not None else task.scoring.get("tolerance", 0.0)
    rel = task.scoring.get("relative", False)
    # optional scoring policies (task-level):
    #   abs: compare magnitudes — sign conventions the prompt doesn't pin down don't count
    #   min_rel_tol: tolerance floor as a fraction of |target|, so an answer correctly
    #   rounded to fewer digits than the reference still passes
    use_abs = task.scoring.get("abs", False)
    # answer_prefix: parse the number AFTER the last occurrence of this marker (a
    # string, or a list of accepted markers such as ["RISPOSTA:", "ANSWER:"]), so
    # chatty models can show working; outputs without any marker fall back to the
    # whole output, so terse models still pass
    prefixes = task.scoring.get("answer_prefix") or []
    prefixes = [prefixes] if isinstance(prefixes, str) else list(prefixes)
    # number: which number of the (remaining) text is the answer — "first" (default)
    # or "last"; "last" suits "answer with only the number" prompts, where a preamble
    # echoing the question ("In Study 2, ...") would otherwise be read as the answer
    which = task.scoring.get("number", "first")
    if which not in ("first", "last"):
        raise ValueError(f"scoring.number must be 'first' or 'last', got {which!r}")
    # decimal_comma: read "68,6" as 68.6 (Italian locale) instead of 686
    decimal_comma = task.scoring.get("decimal_comma", False)
    eff_tol = tol * abs(target) if rel else tol
    eff_tol = max(eff_tol, task.scoring.get("min_rel_tol", 0.0) * abs(target))

    def judge(o: str) -> float:
        text = o
        hits = [(o.lower().rfind(p.lower()), len(p)) for p in prefixes]
        if any(i >= 0 for i, _ in hits):
            i, n = max(h for h in hits if h[0] >= 0)
            text = o[i + n:]
        if decimal_comma:
            text = _DECIMAL_COMMA.sub(r"\1.\2", text)
        x = (last_number if which == "last" else first_number)(text)
        if x is None or math.isnan(x):
            return 0.0
        err = abs(abs(x) - abs(target)) if use_abs else abs(x - target)
        return float(err <= eff_tol)

    return _sampled(item, task, backend, item.prompt, judge)


def score_program(item: Item, task: Task, backend: Backend, root: Path | None = None) -> tuple[float, dict]:
    checker = load_checker(item.checker, root)
    return _sampled(item, task, backend, item.prompt, lambda o: checker(item, o))


SCORERS = {
    Kind.CHOICE_LOGLIK: score_choice_loglik,
    Kind.PAIR_LOGLIK: score_pair_loglik,
    Kind.CHOICE_GENERATE: score_choice_generate,
    Kind.EXACT: score_exact,
    Kind.REGEX: score_regex,
    Kind.NUMERIC: score_numeric,
    Kind.PROGRAM: score_program,
}


def resolve_protocol(kind: Kind, backend: Backend) -> Kind:
    """Pick the scoring protocol a backend can actually run for this item kind."""
    if kind == Kind.CHOICE_LOGLIK and not backend.supports_logprobs:
        return Kind.CHOICE_GENERATE
    if kind == Kind.PAIR_LOGLIK and not backend.supports_logprobs:
        raise RuntimeError("pair_loglik items need a backend with logprobs (hf:…)")
    if kind in (Kind.CHOICE_GENERATE, Kind.EXACT, Kind.REGEX, Kind.NUMERIC, Kind.PROGRAM) \
            and not backend.supports_generate:
        raise RuntimeError(f"{kind} items need a backend that can generate")
    return kind


def score_item(item: Item, task: Task, backend: Backend, root: Path | None = None) -> tuple[float, dict[str, Any], Kind]:
    """`root` is the suite directory, used to resolve suite-local checker files."""
    protocol = resolve_protocol(item.kind, backend)
    if protocol == Kind.PROGRAM:
        score, detail = score_program(item, task, backend, root)
    else:
        score, detail = SCORERS[protocol](item, task, backend)
    return score, detail, protocol
