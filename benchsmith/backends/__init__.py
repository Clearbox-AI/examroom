"""Model backends. Each backend exposes `loglik` and/or `generate` plus resource stats."""

from __future__ import annotations

from benchsmith.backends.base import Backend, BackendInfo, Stats


def load_backend(spec: str, **kwargs) -> Backend:
    """Build a backend from a spec string `<type>:<model>`.

    Examples:
        hf:HuggingFaceTB/SmolLM2-135M
        hf:/path/to/checkpoint
        openai:gpt-4o-mini            (needs OPENAI_API_KEY / --base-url)
        dummy:uniform                 (for tests; deterministic pseudo-random scores)
    """
    if ":" not in spec:
        raise ValueError("backend spec must be '<type>:<model>', e.g. hf:HuggingFaceTB/SmolLM2-135M")
    kind, model = spec.split(":", 1)
    if kind == "hf":
        from benchsmith.backends.hf import HFBackend
        return HFBackend(model, **kwargs)
    if kind == "openai":
        from benchsmith.backends.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(model, **kwargs)
    if kind == "dummy":
        from benchsmith.backends.dummy import DummyBackend
        return DummyBackend(model, **kwargs)
    raise ValueError(f"unknown backend type {kind!r}")


__all__ = ["Backend", "BackendInfo", "Stats", "load_backend"]
