"""OpenAI-compatible chat backend (OpenAI, OpenRouter, vLLM, Ollama, llama.cpp server…).

Generation only. Likelihood items cannot be scored through chat APIs; the runner
falls back to `choice_generate` protocol for them and records that in the ledger.
"""

from __future__ import annotations

import os

from benchsmith.backends.base import BackendInfo, Stats, Timer


class OpenAICompatBackend:
    supports_logprobs = False
    supports_generate = True

    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY", **_):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "none"))
        self.info = BackendInfo(backend="openai", model=model, extra={"base_url": base_url})
        self.stats = Stats()

    def loglik(self, prompt, continuations):
        raise NotImplementedError("chat APIs do not expose continuation logprobs")

    def generate(self, prompt, max_tokens, temperature, n):
        with Timer() as t:
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=max_tokens, temperature=temperature, n=n,
                messages=[{"role": "user", "content": prompt}],
            )
        usage = resp.usage
        self.stats.tick(
            t.seconds,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            generated_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        return [(c.message.content or "") for c in resp.choices]

    def close(self):
        pass
