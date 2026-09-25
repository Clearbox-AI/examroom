"""OpenAI-compatible chat backend (OpenAI, OpenRouter, vLLM, Ollama, llama.cpp server…).

Generation only. Likelihood items cannot be scored through chat APIs; the runner
falls back to `choice_generate` protocol for them and records that in the ledger.
"""

from __future__ import annotations

import os
import time

from benchsmith.backends.base import BackendInfo, Stats, Timer


class OpenAICompatBackend:
    supports_logprobs = False
    supports_generate = True

    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 reasoning: str | None = None, **_):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "none"))
        # reasoning: OpenRouter's unified knob for hybrid-thinking models
        # ("off" disables it; "low"/"medium"/"high" set effort). Only sent when given,
        # so plain OpenAI-compatible servers (llama.cpp, vLLM, …) are unaffected.
        if reasoning == "off":
            self.extra_body = {"reasoning": {"enabled": False}}
        elif reasoning in ("low", "medium", "high"):
            self.extra_body = {"reasoning": {"effort": reasoning}}
        elif reasoning is None:
            self.extra_body = None
        else:
            raise ValueError("reasoning must be off|low|medium|high")
        self.info = BackendInfo(backend="openai", model=model,
                                extra={"base_url": base_url, "reasoning": reasoning})
        self.stats = Stats()

    def loglik(self, prompt, continuations):
        raise NotImplementedError("chat APIs do not expose continuation logprobs")

    def generate(self, prompt, max_tokens, temperature, n):
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        with Timer() as t:
            for attempt in range(8):  # ride out 429/5xx from hosted routers (upstream pools can stall for minutes)
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model, max_tokens=max_tokens, temperature=temperature, n=n,
                        messages=[{"role": "user", "content": prompt}],
                        **({"extra_body": self.extra_body} if self.extra_body else {}),
                    )
                    break
                except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError):
                    if attempt == 7:
                        raise
                    time.sleep(min(2.0 * 2 ** attempt, 60.0))
        usage = resp.usage
        self.stats.tick(
            t.seconds,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            generated_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        return [(c.message.content or "") for c in resp.choices]

    def close(self):
        pass
