"""Deterministic fake backend for tests and dry runs.

`dummy:uniform`  -> pseudo-random logprobs (accuracy ≈ chance), echo-ish generation
`dummy:oracle`   -> always prefers the shortest continuation (a *length* prior; useful
                    to check that length normalisation removes the bias)
"""

from __future__ import annotations

import hashlib

from examroom.backends.base import BackendInfo, Stats, Timer


class DummyBackend:
    supports_logprobs = True
    supports_generate = True
    supports_tools = True  # never calls a tool: answers "A" at once (the no-tool-use floor)

    def __init__(self, mode: str = "uniform", **_):
        self.mode = mode
        self.info = BackendInfo(backend="dummy", model=mode)
        self.stats = Stats()

    @staticmethod
    def _ntok(s: str) -> int:
        return max(1, len(s.split()))

    def loglik(self, prompt, continuations):
        out = []
        with Timer() as t:
            for c in continuations:
                n = self._ntok(c)
                if self.mode == "oracle":
                    lp = -0.5 * n
                else:
                    h = int(hashlib.md5((prompt + "\x00" + c).encode()).hexdigest(), 16)
                    lp = -(1.0 + (h % 1000) / 200.0) * n
                out.append((lp, n))
        self.stats.tick(t.seconds, prompt_tokens=len(continuations) * self._ntok(prompt))
        return out

    def generate(self, prompt, max_tokens, temperature, n):
        with Timer() as t:
            outs = ["A"] * n
        self.stats.tick(t.seconds, prompt_tokens=self._ntok(prompt), generated_tokens=n)
        return outs

    def chat(self, messages, tools, max_tokens, temperature):
        with Timer() as t:
            reply = {"content": "A", "tool_calls": []}
        self.stats.tick(t.seconds, prompt_tokens=sum(self._ntok(m.get("content") or "") for m in messages),
                        generated_tokens=1)
        return reply

    def close(self):
        pass
