"""Native Anthropic API backend (Claude models).

Generation only, like every chat API: `choice_loglik` falls back to the
`choice_generate` protocol, `pair_loglik` items refuse to run.

Options (`--opt k=v`):
    api_key_env=ANTHROPIC_API_KEY   env var holding the key; if unset the SDK's
                                    own resolution applies (ANTHROPIC_API_KEY,
                                    ANTHROPIC_AUTH_TOKEN, `ant auth login` profile)
    base_url=…                      alternative endpoint (gateways, proxies)
    thinking=off|adaptive|none      default "off": send {"type": "disabled"} so
                                    short-max_tokens items get answer text, not
                                    thinking budget.  "adaptive" turns it on;
                                    "none" omits the parameter entirely (needed
                                    for models that reject "disabled", e.g.
                                    claude-fable-5 — note omission means
                                    *adaptive* on the Claude 5 family)
    effort=low|…|max                sets output_config.effort when given

The runner's per-item `temperature` is NOT sent: current Claude models reject
sampling parameters (400).  Sampled items (samples>1) therefore draw from the
model's own decoding, not a benchsmith-chosen temperature.
"""

from __future__ import annotations

import os

from benchsmith.backends.base import BackendInfo, Stats, Timer

_THINKING = {"off": {"type": "disabled"}, "adaptive": {"type": "adaptive"}, "none": None}


class AnthropicBackend:
    supports_logprobs = False
    supports_generate = True

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        thinking: str = "off",
        effort: str | None = None,
        **_,
    ):
        from anthropic import Anthropic

        if thinking not in _THINKING:
            raise ValueError(f"thinking must be one of {sorted(_THINKING)}, got {thinking!r}")
        self.model = model
        self.thinking = _THINKING[thinking]
        self.effort = effort
        self.client = Anthropic(base_url=base_url, api_key=os.environ.get(api_key_env))
        self.info = BackendInfo(
            backend="anthropic",
            model=model,
            extra={"base_url": base_url, "thinking": thinking, "effort": effort,
                   "sampling": "model default (temperature not sent)"},
        )
        self.stats = Stats()

    def loglik(self, prompt, continuations):
        raise NotImplementedError("the Anthropic API does not expose continuation logprobs")

    def generate(self, prompt, max_tokens, temperature, n):
        kwargs: dict = {}
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking
        if self.effort is not None:
            kwargs["output_config"] = {"effort": self.effort}
        outs = []
        for _ in range(n):  # the API has no n parameter
            with Timer() as t:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
            text = "".join(b.text for b in resp.content if b.type == "text")
            if resp.stop_reason == "refusal":
                text = ""
            outs.append(text)
            u = resp.usage
            self.stats.tick(
                t.seconds,
                prompt_tokens=getattr(u, "input_tokens", 0) or 0,
                generated_tokens=getattr(u, "output_tokens", 0) or 0,
            )
        return outs

    def close(self):
        pass
