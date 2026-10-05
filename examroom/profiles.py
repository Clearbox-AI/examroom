"""Run profiles: the frozen run-time protocol under which runs are comparable.

A suite's content hash pins what is asked and how answers are scored; a profile
pins how the model is driven — reasoning on or off, the generation budget,
sampling.  The leaderboard only ranks runs that share a profile.  A run records
the full profile, and a profile is never edited: a change is a new version.

Anything a suite needs to be answerable by a chat model ("give only the final
number") belongs in the suite's task `scoring.instruction`, which is hashed —
not in a run-time `--instruction`, which a profiled run refuses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    version: int
    reasoning: str               # off | on
    temperature: float | None    # None: the suite's default (greedy for single samples)
    min_max_tokens: int | None   # floor on every generative item's budget; thinking needs room
    description: str

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"

    def overrides(self) -> dict:
        """Task scoring overrides the runner merges into every task for this run."""
        o = {}
        if self.temperature is not None:
            o["temperature"] = self.temperature
        if self.min_max_tokens is not None:
            o["min_max_tokens"] = self.min_max_tokens
        return o

    def backend_options(self, backend_type: str) -> dict:
        """Backend kwargs that switch reasoning, in each backend's own vocabulary."""
        if backend_type in ("hf", "openai"):
            return {"reasoning": self.reasoning}
        if backend_type == "anthropic":
            return {"thinking": "adaptive" if self.reasoning == "on" else "off"}
        if backend_type == "dummy":
            return {}
        raise ValueError(f"profile {self.id} does not know how to drive backend {backend_type!r}")

    def as_dict(self) -> dict:
        return {"id": self.id, **asdict(self)}


PROFILES = {p.name: p for p in (
    Profile("nothink", 1, reasoning="off", temperature=None, min_max_tokens=None,
            description="reasoning disabled; the suite's own budgets; greedy single samples"),
    Profile("think", 1, reasoning="on", temperature=0.6, min_max_tokens=16384,
            description="reasoning enabled; temperature 0.6; at least 16k tokens per answer"),
)}

# backend options a profile fixes; passing them with --profile would silently change the protocol
PROFILE_FIXED_OPTIONS = {"reasoning", "thinking", "effort", "chat_template"}


def get_profile(name: str) -> Profile:
    base = name.split("@")[0]
    if base not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; known: {', '.join(p.id for p in PROFILES.values())}")
    p = PROFILES[base]
    if "@" in name and name != p.id:
        raise KeyError(f"profile {name!r} is not the current version ({p.id})")
    return p
