"""Suite validation: schema, duplicates, answer balance, tokenisation conventions, leakage."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from benchsmith.items import Kind, Suite, Task

_WORD = re.compile(r"\w+")


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, dict] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def prior_accuracy(items) -> tuple[float, float, str | None]:
    """Expected accuracy of a frequency-prior strategy vs chance.

    The prior: among the options shown, pick the string that is most often the
    correct answer in the *other* items (leave-one-out, so a unique answer is not
    a giveaway); ties are scored as the expected value of a random pick among
    the tied options. A well-balanced set keeps this ≈ chance.
    Returns (prior_acc, chance, most_overrepresented_option).
    """
    freq = Counter(it.answer for it in items)
    hits = 0.0
    chance = 0.0
    for it in items:
        f = {o: freq[o] - (o == it.answer) for o in it.options}
        top = max(f.values())
        tied = [o for o, v in f.items() if v == top]
        hits += (it.answer in tied) / len(tied)
        chance += 1 / len(it.options)
    n = len(items)
    top_ans = freq.most_common(1)[0][0] if freq else None
    return hits / n, chance / n, top_ans


def conditional_rates(items, min_shown: int = 20) -> dict[str, tuple[float, int]]:
    """P(correct | option shown) for every option string shown >= min_shown times.

    This is the exploitable quantity: if ' summer' is correct 80% of the times it
    is offered, a model that merely likes the word 'summer' wins those items.
    """
    shown: Counter = Counter()
    right: Counter = Counter()
    for it in items:
        for o in it.options:
            shown[o] += 1
        right[it.answer] += 1
    return {o: (right[o] / shown[o], shown[o]) for o in shown if shown[o] >= min_shown}


def ngrams(text: str, n: int):
    w = [t.lower() for t in _WORD.findall(text)]
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def build_corpus_index(corpus_dir: Path, n: int, exts=(".txt", ".md", ".jsonl", ".json")) -> set[tuple]:
    idx: set[tuple] = set()
    for p in sorted(corpus_dir.rglob("*")):
        if p.suffix in exts and p.is_file():
            idx |= ngrams(p.read_text(errors="ignore"), n)
    return idx


def validate_suite(
    suite: Suite,
    min_items: int = 200,
    balance_tolerance: float = 0.10,
    corpus_dir: Path | None = None,
    ngram: int = 13,
) -> Report:
    rep = Report()
    seen_ids: Counter = Counter()
    seen_content: dict[str, str] = {}

    for task in suite.tasks:
        items = suite.items[task.name]
        m: dict = {"n": len(items)}
        if len(items) < min_items:
            rep.warnings.append(f"{task.name}: {len(items)} items < {min_items}")

        for it in items:
            seen_ids[it.id] += 1
            key = it.canonical_key() if hasattr(it, "canonical_key") else _content_key(it)
            if key in seen_content:
                rep.errors.append(f"{task.name}: {it.id} duplicates {seen_content[key]}")
            seen_content[key] = it.id
            _check_conventions(task, it, rep)

        if task.kind in (Kind.CHOICE_LOGLIK, Kind.CHOICE_GENERATE) and items:
            prior, chance, top = prior_accuracy(items)
            m.update(prior_accuracy=round(prior, 3), chance=round(chance, 3), top_answer=top)
            if prior > chance + balance_tolerance:
                rep.errors.append(
                    f"{task.name}: frequency prior scores {prior:.1%} vs chance {chance:.1%} "
                    f"(most common answer: {top!r}) — rebalance")
            rates = conditional_rates(items)
            if rates:
                worst = max(rates, key=lambda o: rates[o][0])
                m["max_conditional"] = {"option": worst, "p_correct_given_shown": round(rates[worst][0], 3),
                                        "shown": rates[worst][1]}
                if rates[worst][0] > chance + balance_tolerance + 0.10:
                    rep.errors.append(
                        f"{task.name}: option {worst!r} is correct {rates[worst][0]:.0%} of the "
                        f"{rates[worst][1]} times it is shown (chance {chance:.0%}) — rebalance or vary distractors")
                elif rates[worst][0] > chance + balance_tolerance:
                    rep.warnings.append(
                        f"{task.name}: option {worst!r} is correct {rates[worst][0]:.0%} of the "
                        f"{rates[worst][1]} times it is shown (chance {chance:.0%})")
            pos = Counter(it.answer_idx for it in items)
            m["answer_idx_dist"] = dict(sorted(pos.items()))
            if task.kind == Kind.CHOICE_GENERATE:
                top_pos = max(pos.values()) / len(items)
                if top_pos > chance + balance_tolerance:
                    rep.errors.append(f"{task.name}: answer position imbalance {top_pos:.1%}")
        if task.kind == Kind.PAIR_LOGLIK and items:
            diffs = [len(it.good.split()) - len(it.bad.split()) for it in items]
            jacc = []
            for it in items:
                a, b = set(it.good.lower().split()), set(it.bad.lower().split())
                jacc.append(len(a & b) / len(a | b))
            m.update(mean_len_diff=round(sum(diffs) / len(diffs), 2),
                     good_longer_frac=round(sum(d > 0 for d in diffs) / len(diffs), 3),
                     mean_vocab_jaccard=round(sum(jacc) / len(jacc), 3))
            if abs(m["good_longer_frac"] - 0.5) > 0.25 and abs(m["mean_len_diff"]) > 0.5:
                rep.warnings.append(f"{task.name}: good is systematically {'longer' if m['mean_len_diff'] > 0 else 'shorter'} "
                                    f"than bad (mean diff {m['mean_len_diff']} words) — a length prior could help")
        rep.metrics[task.name] = m

    for iid, c in seen_ids.items():
        if c > 1:
            rep.errors.append(f"duplicate id {iid} ({c}x)")
    if corpus_dir is not None:
        from benchsmith.leakage import check_leakage

        lk = check_leakage(suite, [corpus_dir], n=ngram)
        rep.metrics["leakage"] = {"n": ngram, "docs_scanned": lk.docs_scanned, "items_indexed": lk.items_indexed,
                                  "items_too_short": lk.items_too_short, "items_colliding": len(lk.collisions)}
        for iid, hits in lk.collisions.items():
            rep.errors.append(f"leakage: {iid} shares a {ngram}-gram with {hits[0][0]}: {hits[0][1]!r}")
    return rep


def _content_key(it) -> str:
    return "\x1f".join([it.prompt, *(it.options or []), it.good or "", it.bad or "", it.target or ""])


def _check_conventions(task: Task, it, rep: Report) -> None:
    conts = it.options if it.options else ([it.good, it.bad] if it.good else [])
    if task.kind in (Kind.CHOICE_LOGLIK, Kind.PAIR_LOGLIK) and conts:
        leading = {c[:1].isspace() for c in conts}
        if len(leading) > 1:
            rep.errors.append(f"{task.name}: {it.id} mixes leading-space conventions across options")
        if it.prompt.endswith((" ", "\t")) and True in leading:
            rep.errors.append(f"{task.name}: {it.id} prompt ends with space AND options start with space")
        if not it.prompt.endswith((" ", "\n")) and False in leading and not it.prompt.endswith(("'", '"', "(")):
            rep.warnings.append(f"{task.name}: {it.id} options lack a leading space after a word prompt")
        if task.kind == Kind.PAIR_LOGLIK and len(conts) == 2:
            g, b = conts
            lg, lb = len(g.split()), len(b.split())
            if abs(lg - lb) > max(3, 0.5 * min(lg, lb)):
                rep.warnings.append(f"{task.name}: {it.id} good/bad length mismatch ({lg} vs {lb} words)")
        if any(c != c.rstrip() for c in conts):
            rep.warnings.append(f"{task.name}: {it.id} option has trailing whitespace")
