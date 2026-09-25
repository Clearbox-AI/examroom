from __future__ import annotations

import resource
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class BackendInfo:
    backend: str                 # hf | openai | dummy
    model: str                   # model id or path
    dtype: str | None = None
    quantization: str | None = None
    device: str | None = None
    n_params: int | None = None
    revision: str | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d.update(d.pop("extra"))
        return d


@dataclass
class Stats:
    """Resource accounting, accumulated by the runner through `tick`."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    wall_seconds: float = 0.0
    forward_calls: int = 0
    peak_rss_mb: float | None = None
    peak_device_mem_mb: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def tick(self, seconds: float, prompt_tokens: int = 0, generated_tokens: int = 0) -> None:
        # with parallel workers, wall_seconds is summed API time (not elapsed wall clock),
        # so tokens_per_second stays a per-request rate
        with self._lock:
            self.wall_seconds += seconds
            self.prompt_tokens += prompt_tokens
            self.generated_tokens += generated_tokens
            self.forward_calls += 1

    def finalize(self) -> dict:
        self.peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        tot = self.prompt_tokens + self.generated_tokens
        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "wall_seconds": round(self.wall_seconds, 3),
            "tokens_per_second": round(tot / self.wall_seconds, 2) if self.wall_seconds else None,
            "generated_tokens_per_second": (
                round(self.generated_tokens / self.wall_seconds, 2)
                if self.wall_seconds and self.generated_tokens else None
            ),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "peak_device_mem_mb": self.peak_device_mem_mb,
        }


@runtime_checkable
class Backend(Protocol):
    info: BackendInfo
    stats: Stats
    supports_logprobs: bool
    supports_generate: bool

    def loglik(self, prompt: str, continuations: list[str]) -> list[tuple[float, int]]:
        """Return (sum logprob, n_tokens) of each continuation given prompt.

        Continuations are appended verbatim: the caller controls the leading-space
        convention. Token counts are needed for length normalisation.
        """
        ...

    def generate(self, prompt: str, max_tokens: int, temperature: float, n: int) -> list[str]:
        ...

    def close(self) -> None: ...


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.seconds = time.perf_counter() - self.t0
