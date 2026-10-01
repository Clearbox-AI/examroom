"""Model registry: size facts that belong to a model (and a quantisation of it), not to a run.

A run records what its backend could measure; an API backend can measure nothing.
The registry is the curated source the leaderboard reads at display time, so
correcting an entry re-tiers every past run of that model.  On disk (JSON):

    {"models": {
       "Qwen/Qwen3-30B-A3B": {
          "n_params": "30.5B", "active_params": "3.3B",
          "source": "config.json: 128 experts, 8 per token",
          "aliases": ["qwen/qwen3-30b-a3b", "qwen3-30b-a3b"],
          "quants": {"Q4_K_M": {"weights_mb": 17697, "file": "Qwen3-30B-A3B-Q4_K_M.gguf",
                                "sha256": "…"}}}}}

A run is matched by `model_id` (recorded with `--opt model_id=…`) or by its
backend model string against ids and aliases, case-insensitively.

Tiers are a property of a (model, precision) row: tiny and small by total
parameters, local by weight footprint at the precision the run used (the same
27B model is local at Q4 and at bf16, an 80B MoE only once quantised).  They are
not exclusive: a 7B bf16 model is small and local.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

SIZE_KEYS = ("n_params", "active_params", "weights_mb")

_COUNT = re.compile(r"^\s*(\d*\.?\d+(?:e[+-]?\d+)?)\s*([kmbt]?)\s*$", re.I)
_MULT = {"": 1, "k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def parse_count(s) -> int:
    """'4B' -> 4_000_000_000; also '600M', '30.5b', '1.2e9', '4000'."""
    m = _COUNT.match(str(s))
    if not m:
        raise ValueError(f"not a parameter count: {s!r} (use e.g. 4B, 600M, 1.2e9)")
    return int(round(float(m.group(1)) * _MULT[m.group(2).lower()]))


def parse_size(key: str, value) -> int | float:
    if key == "weights_mb":
        return float(value)
    if key in ("n_params", "active_params"):
        return parse_count(value)
    raise ValueError(f"unknown size key {key!r}; expected one of {', '.join(SIZE_KEYS)}")


def fmt_size(v, key: str = "n_params") -> str:
    if v is None:
        return "?"
    if key == "weights_mb":
        return f"{v / 1024:.1f}GB" if v >= 1024 else f"{v:.0f}MB"
    for unit, d in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if v >= d:
            return f"{v / d:.3g}{unit}"
    return str(v)


# (tier, size field, inclusive maximum); weights in MiB, matching weights_mb
TIERS = (
    ("tiny", "n_params", 1_000_000_000),
    ("small", "n_params", 14_000_000_000),
    ("local", "weights_mb", 128 * 1024),  # 128 GB unified memory: DGX Spark, Strix Halo
)

# approximate bits per weight including quantisation overhead, for estimating a
# footprint when the registry has no measured size for the precision a run used
BITS_PER_WEIGHT = {
    "fp32": 32, "bf16": 16, "fp16": 16, "fp8": 8, "8bit": 8, "int8": 8, "4bit": 4.5, "mxfp4": 4.25,
    "q8_0": 8.5, "q6_k": 6.56, "q5_k_m": 5.69, "q5_k_s": 5.54, "q5_0": 5.5,
    "q4_k_m": 4.85, "q4_k_s": 4.58, "q4_0": 4.5, "iq4_xs": 4.25, "q3_k_m": 3.91, "q2_k": 2.96,
}
# hosted APIs rarely state the precision they serve; fp8/bf16 are typical, so an
# unknown precision is estimated at 8 bits per weight and flagged as an estimate
UNKNOWN_PRECISION_BITS = 8
_DTYPES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}


def precision_key(name: str | None) -> str | None:
    return None if not name else _DTYPES.get(name.lower(), name.lower())


def run_precision(backend: dict) -> str | None:
    """Precision a recorded run used: its quantisation, else the hf dtype; None when unknown."""
    if backend.get("quantization"):
        return str(backend["quantization"])
    if backend.get("backend") == "hf" and backend.get("dtype"):
        return _DTYPES.get(backend["dtype"], backend["dtype"])
    return None


@dataclass
class ModelSize:
    model_id: str | None         # registry id, None when the model is not registered
    precision: str | None
    n_params: int | None
    active_params: int | None
    weights_mb: float | None
    weights_source: str | None   # registry | run | estimate
    tiers: list[str]


def tiers_for(n_params: int | None, weights_mb: float | None) -> list[str]:
    vals = {"n_params": n_params, "weights_mb": weights_mb}
    return [name for name, key, limit in TIERS if vals[key] is not None and vals[key] <= limit]


class Registry:
    def __init__(self, models: dict | None = None, path: Path | None = None):
        self.models: dict[str, dict] = models or {}
        self.path = path

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        path = Path(path)
        if not path.is_file():
            return cls({}, path)
        return cls(json.loads(path.read_text()).get("models", {}), path)

    def save(self, path: str | Path | None = None) -> None:
        path = Path(path or self.path)
        path.write_text(json.dumps({"models": dict(sorted(self.models.items()))}, indent=2,
                                   ensure_ascii=False) + "\n")

    def find(self, name: str | None) -> str | None:
        """Registry id for a model id or alias (case-insensitive), or None."""
        if not name:
            return None
        n = name.lower()
        for mid, e in self.models.items():
            if mid.lower() == n or n in (a.lower() for a in e.get("aliases", [])):
                return mid
        return None

    def add(self, model_id: str, n_params, active_params=None, aliases=(), source: str | None = None) -> dict:
        if (other := self.find(model_id)) and other != model_id:
            raise ValueError(f"{model_id!r} is already an alias of {other!r}")
        for a in aliases:
            if (other := self.find(a)) and other != model_id:
                raise ValueError(f"alias {a!r} already belongs to {other!r}")
        e = self.models.setdefault(model_id, {})
        e["n_params"] = parse_count(n_params)
        e["active_params"] = parse_count(active_params) if active_params is not None else e.get("active_params")
        e["aliases"] = sorted(set(e.get("aliases", [])) | set(aliases))
        if source:
            e["source"] = source
        e.setdefault("quants", {})
        return e

    def add_quant(self, model_id: str, precision: str, weights_mb: float | None = None,
                  file: str | Path | None = None) -> dict:
        mid = self.find(model_id)
        if mid is None:
            raise KeyError(f"{model_id!r} is not registered; `examroom models add` it first")
        q: dict = {}
        if file is not None:
            file = Path(file)
            h = hashlib.sha256()
            with file.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            q = {"weights_mb": round(file.stat().st_size / 2**20, 1), "file": file.name, "sha256": h.hexdigest()}
        if weights_mb is not None:
            q["weights_mb"] = float(weights_mb)
        if "weights_mb" not in q:
            raise ValueError("give weights_mb or a weights file")
        self.models[mid].setdefault("quants", {})[precision] = q
        return q

    def size(self, backend: dict) -> ModelSize:
        """Size and tiers of a recorded run (its backend_json): registry facts win over
        what the run recorded, and a missing footprint is estimated from the precision."""
        mid = self.find(backend.get("model_id")) or self.find(backend.get("model"))
        e = self.models.get(mid, {})
        prec = run_precision(backend)
        n_params = e.get("n_params") or backend.get("n_params")
        active = e.get("active_params") or backend.get("active_params")
        weights, src = None, None
        quants = {precision_key(k): v for k, v in e.get("quants", {}).items()}
        if prec and precision_key(prec) in quants:
            weights, src = quants[precision_key(prec)]["weights_mb"], "registry"
        elif backend.get("weights_mb") is not None:
            weights, src = backend["weights_mb"], "run"
        elif n_params:
            bits = BITS_PER_WEIGHT.get(precision_key(prec)) if prec else UNKNOWN_PRECISION_BITS
            if bits is not None:
                weights, src = round(n_params * bits / 8 / 2**20, 1), "estimate"
        return ModelSize(mid, prec, n_params, active, weights, src, tiers_for(n_params, weights))
