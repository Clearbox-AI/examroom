import pytest

from benchsmith.backends.anthropic_api import AnthropicBackend


class _Block:
    def __init__(self, type, text=""):
        self.type = type
        self.text = text


class _Usage:
    input_tokens = 10
    output_tokens = 5


class _Resp:
    def __init__(self, blocks, stop_reason="end_turn"):
        self.content = blocks
        self.stop_reason = stop_reason
        self.usage = _Usage()


class _Messages:
    def __init__(self, resp):
        self.resp = resp
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return self.resp


def _backend(monkeypatch, resp, **opts):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    b = AnthropicBackend("claude-sonnet-5", **opts)
    b.client.messages = _Messages(resp)
    return b


def test_generate_extracts_text_and_never_sends_temperature(monkeypatch):
    resp = _Resp([_Block("thinking"), _Block("text", " 42 m/s")])
    b = _backend(monkeypatch, resp)
    outs = b.generate("prompt", max_tokens=24, temperature=0.7, n=2)
    assert outs == [" 42 m/s", " 42 m/s"]
    for call in b.client.messages.calls:
        assert "temperature" not in call
        assert call["thinking"] == {"type": "disabled"}  # default thinking=off
        assert call["max_tokens"] == 24
    assert b.stats.generated_tokens == 10  # 5 per call, n=2


def test_thinking_and_effort_options(monkeypatch):
    resp = _Resp([_Block("text", "ok")])
    b = _backend(monkeypatch, resp, thinking="none", effort="low")
    b.generate("p", 8, 0.0, 1)
    (call,) = b.client.messages.calls
    assert "thinking" not in call
    assert call["output_config"] == {"effort": "low"}
    with pytest.raises(ValueError):
        _backend(monkeypatch, resp, thinking="bogus")


def test_refusal_yields_empty_output(monkeypatch):
    resp = _Resp([_Block("text", "partial")], stop_reason="refusal")
    b = _backend(monkeypatch, resp)
    assert b.generate("p", 8, 0.0, 1) == [""]


def test_loglik_unsupported(monkeypatch):
    b = _backend(monkeypatch, _Resp([]))
    assert b.supports_logprobs is False
    with pytest.raises(NotImplementedError):
        b.loglik("p", [" a", " b"])
