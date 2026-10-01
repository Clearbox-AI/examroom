"""OpenAI-compatible chat backend (OpenAI, OpenRouter, vLLM, Ollama, llama.cpp server…).

Generation only. Likelihood items cannot be scored through chat APIs; the runner
falls back to `choice_generate` protocol for them and records that in the ledger.
"""

from __future__ import annotations

import os
import time

from examroom.backends.base import BackendInfo, Stats, Timer, strip_reasoning


class OpenAICompatBackend:
    supports_logprobs = False
    supports_generate = True

    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 reasoning: str | None = None, **_):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "none"))
        # reasoning=off|on switches thinking for hybrid models.  OpenRouter takes its own
        # unified field (where low|medium|high also set effort); llama.cpp and vLLM servers
        # take the chat template's enable_thinking flag.  Nothing is sent when not given.
        on_openrouter = "openrouter.ai" in (base_url or "")
        if reasoning is None:
            self.extra_body = None
        elif on_openrouter and reasoning in ("off", "on"):
            self.extra_body = {"reasoning": {"enabled": reasoning == "on"}}
        elif on_openrouter and reasoning in ("low", "medium", "high"):
            self.extra_body = {"reasoning": {"effort": reasoning}}
        elif base_url and not on_openrouter and reasoning in ("off", "on"):
            self.extra_body = {"chat_template_kwargs": {"enable_thinking": reasoning == "on"}}
        else:
            raise ValueError(f"reasoning={reasoning!r} is not supported for base_url={base_url!r} "
                             "(off|on for local servers and OpenRouter; low|medium|high on OpenRouter)")
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
        return [strip_reasoning(c.message.content or "") for c in resp.choices]

    def close(self):
        pass
