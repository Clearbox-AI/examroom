"""Scorers, one per Kind. Each returns (score in [0,1], detail dict)."""

from __future__ import annotations

import importlib
import math
import re
import string
from typing import Any

from benchsmith.backends.base import Backend
from benchsmith.items import Item, Kind, Task

LETTERS = "ABCDEFGH"


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


_NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def first_number(text: str) -> float | None:
    m = _NUM.search(text)
    return float(m.group().replace(",", "")) if m else None


def score_numeric(item: Item, task: Task, backend: Backend) -> tuple[float, dict]:
    target = float(item.target)
    tol = item.tolerance if item.tolerance is not None else task.scoring.get("tolerance", 0.0)
    rel = task.scoring.get("relative", False)

    def judge(o: str) -> float:
        x = first_number(o)
        if x is None or math.isnan(x):
            return 0.0
        err = abs(x - target)
        return float(err <= (tol * abs(target) if rel else tol))

    return _sampled(item, task, backend, item.prompt, judge)


def score_program(item: Item, task: Task, backend: Backend) -> tuple[float, dict]:
    mod, fn = item.checker.split(":")
    checker = getattr(importlib.import_module(mod), fn)
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


def score_item(item: Item, task: Task, backend: Backend) -> tuple[float, dict[str, Any], Kind]:
    protocol = resolve_protocol(item.kind, backend)
    score, detail = SCORERS[protocol](item, task, backend)
    return score, detail, protocol
