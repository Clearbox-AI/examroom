"""Native Anthropic API backend (Claude models).

Generation and native tool calling, like every chat API: `choice_loglik` falls
back to the `choice_generate` protocol, `pair_loglik` items refuse to run.

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
model's own decoding, not an examroom-chosen temperature.
"""

from __future__ import annotations

import json
import os

from examroom.backends.base import BackendInfo, Stats, Timer

_THINKING = {"off": {"type": "disabled"}, "adaptive": {"type": "adaptive"}, "none": None}


class AnthropicBackend:
    supports_logprobs = False
    supports_generate = True
    supports_tools = True

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

    def chat(self, messages, tools, max_tokens, temperature):
        """One assistant turn with native tool use.  Translates the OpenAI-format
        conversation the agent scorer keeps (system message, assistant tool_calls,
        role=tool results) into Messages API blocks, and the reply back."""
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        conv: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                continue
            if m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if conv and conv[-1]["role"] == "user" and isinstance(conv[-1]["content"], list):
                    conv[-1]["content"].append(block)  # parallel calls: all results in one user turn
                else:
                    conv.append({"role": "user", "content": [block]})
            elif m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                for c in m.get("tool_calls") or []:
                    try:
                        args = json.loads(c["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    blocks.append({"type": "tool_use", "id": c["id"], "name": c["function"]["name"], "input": args})
                conv.append({"role": "assistant", "content": blocks})
            else:
                conv.append({"role": m["role"], "content": m["content"]})
        kwargs: dict = {}
        if system:
            kwargs["system"] = system
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking
        if self.effort is not None:
            kwargs["output_config"] = {"effort": self.effort}
        with Timer() as t:
            resp = self.client.messages.create(
                model=self.model, max_tokens=max_tokens, messages=conv,
                tools=[{"name": x["name"], "description": x.get("description", ""), "input_schema": x["inputSchema"]}
                       for x in tools],
                **kwargs,
            )
        u = resp.usage
        self.stats.tick(t.seconds, prompt_tokens=getattr(u, "input_tokens", 0) or 0,
                        generated_tokens=getattr(u, "output_tokens", 0) or 0)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [{"id": b.id, "name": b.name, "arguments": json.dumps(b.input)}
                 for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason == "refusal":
            text, calls = "", []
        return {"content": text, "tool_calls": calls}

    def close(self):
        pass
