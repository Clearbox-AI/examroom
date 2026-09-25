"""Programmatic authoring: templates × slot-fills with dedup and answer balancing.

Usage:
    b = Builder(task="calendar", seed=1000)
    b.add("The day after Monday is", answer=" Tuesday", distractors=[" Friday", " Sunday"], template="day_after")
    items = b.build()            # deduplicated, stable ids
    b.rebalance()                # downsample so each answer string is correct ~equally often
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict

from examroom.items import Item, Kind, Provenance


class Builder:
    def __init__(self, task: str, seed: int = 1000, kind: Kind = Kind.CHOICE_LOGLIK, author: str | None = None):
        self.task = task
        self.kind = kind
        self.seed = seed
        self.rng = random.Random(seed)
        self.author = author
        self._items: dict[str, Item] = {}

    def pick(self, pool: list[str], k: int, exclude: str | set[str]) -> list[str]:
        ex = {exclude} if isinstance(exclude, str) else set(exclude)
        cands = [p for p in pool if p not in ex]
        return self.rng.sample(cands, min(k, len(cands)))

    def add(self, prompt: str, answer: str, distractors: list[str], template: str,
            category: str | None = None, notes: str | None = None, tags: list[str] | None = None) -> Item | None:
        options = [answer] + [d for d in distractors if d != answer]
        # shuffle positions so answer_idx carries no signal (matters for choice_generate)
        self.rng.shuffle(options)
        key = prompt + "\x1f" + "\x1f".join(sorted(options))
        if key in self._items:
            return None
        iid = f"{self.task}-{hashlib.sha1(key.encode()).hexdigest()[:8]}"
        it = Item(
            id=iid, kind=self.kind, prompt=prompt, options=options, answer_idx=options.index(answer),
            category=category or self.task, tags=tags or [], notes=notes,
            provenance=Provenance(method="template", source=template, seed=self.seed, author=self.author),
        )
        self._items[key] = it
        return it

    def items(self) -> list[Item]:
        return list(self._items.values())

    def answer_counts(self) -> Counter:
        return Counter(it.answer for it in self._items.values())

    def rebalance(self, max_ratio: float = 1.5) -> int:
        """Downsample over-represented answers within each *option-set family*.

        Items are grouped by the sorted set of their options' vocabulary (e.g. all
        'day of week' items), and within a group no answer may be correct more than
        `max_ratio` × the least frequent answer. Returns number of items dropped.
        """
        groups: dict[frozenset, list[Item]] = defaultdict(list)
        vocab_of: dict[str, frozenset] = {}
        # family = connected component of options co-occurring in items
        parent: dict[str, str] = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for it in self._items.values():
            for o in it.options[1:]:
                parent[find(o)] = find(it.options[0])
        for it in self._items.values():
            groups[find(it.options[0])].append(it)

        dropped = 0
        for _, its in groups.items():
            counts = Counter(it.answer for it in its)
            if len(counts) < 2:
                continue
            cap = int(min(counts.values()) * max_ratio) + 1
            per_answer: Counter = Counter()
            shuffled = its[:]
            self.rng.shuffle(shuffled)
            for it in shuffled:
                per_answer[it.answer] += 1
                if per_answer[it.answer] > cap:
                    key = it.prompt + "\x1f" + "\x1f".join(sorted(it.options))
                    del self._items[key]
                    dropped += 1
        return dropped


class PairBuilder:
    """Authoring helper for `pair_loglik` items (context + good/bad continuation)."""

    def __init__(self, task: str, seed: int = 1000, author: str | None = None):
        self.task = task
        self.seed = seed
        self.rng = random.Random(seed)
        self.author = author
        self._items: dict[str, Item] = {}

    def add(self, context: str, good: str, bad: str, violation: str, template: str,
            notes: str | None = None, tags: list[str] | None = None) -> Item | None:
        if good == bad:
            return None
        key = context + "\x1f" + good + "\x1f" + bad
        if key in self._items:
            return None
        iid = f"{self.task}-{violation[:4]}-{hashlib.sha1(key.encode()).hexdigest()[:8]}"
        it = Item(
            id=iid, kind=Kind.PAIR_LOGLIK, prompt=context, good=good, bad=bad,
            category=violation, tags=tags or [], notes=notes,
            provenance=Provenance(method="template", source=template, seed=self.seed, author=self.author),
        )
        self._items[key] = it
        return it

    def items(self) -> list[Item]:
        return list(self._items.values())

    def counts(self) -> Counter:
        return Counter(it.category for it in self._items.values())
