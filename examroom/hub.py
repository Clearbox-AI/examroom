"""Model facts from the Hugging Face Hub, for the registry.

Total parameters come from the repo's safetensors metadata (every stored tensor:
vision encoder, MTP heads and n-gram embeddings included — what has to be
loaded).  A precision's footprint is the summed size of its weight files: all
*.safetensors of a repo, or one GGUF.  Multimodal projector files (mmproj) are
left out, since the suites are text-only.  Active parameters of an MoE model are
not derivable from metadata reliably, so they are entered from the model card.

Needs `huggingface_hub` (installed with the `hf` extra).  Gated repos expose this
metadata without a token; set HF_TOKEN for higher rate limits.
"""

from __future__ import annotations

import fnmatch

_DTYPE_PRECISION = {"BF16": "bf16", "F16": "fp16", "F32": "fp32", "F8_E4M3": "fp8", "F8_E5M2": "fp8"}


def _info(repo: str):
    from huggingface_hub import HfApi

    return HfApi().model_info(repo, files_metadata=True)


def weight_files(siblings, match: str | None = None) -> list:
    """The files holding one precision's weights: a single GGUF (optionally chosen by
    glob), else every safetensors shard."""
    names = [s for s in siblings if s.rfilename.endswith((".gguf", ".safetensors"))
             and "mmproj" not in s.rfilename.lower()]
    if match:
        names = [s for s in names if fnmatch.fnmatch(s.rfilename, match)]
    ggufs = [s for s in names if s.rfilename.endswith(".gguf")]
    if ggufs:
        if len(ggufs) > 1:
            raise ValueError("several GGUF files: " + ", ".join(s.rfilename for s in ggufs)
                             + "; choose one with --match")
        return ggufs
    if not names:
        raise ValueError("no weight files (.safetensors / .gguf) found" + (f" matching {match!r}" if match else ""))
    return names


def quant_facts(repo: str, match: str | None = None) -> dict:
    info = _info(repo)
    files = weight_files(info.siblings, match)
    q = {"weights_mb": round(sum(s.size for s in files) / 2**20, 1), "repo": repo, "revision": info.sha}
    if len(files) == 1:
        q["file"] = files[0].rfilename
        if files[0].lfs:
            q["sha256"] = files[0].lfs.sha256
    else:
        q["files"] = len(files)
    return q


def model_facts(repo: str) -> dict:
    """Total parameters, native precision and its footprint for a safetensors repo."""
    info = _info(repo)
    st = info.safetensors
    if st is None:
        raise ValueError(f"{repo} has no safetensors metadata; register it by hand with `models add`")
    params = {k: v for k, v in st.parameters.items() if k in _DTYPE_PRECISION}
    native = _DTYPE_PRECISION[max(params, key=params.get)] if params else None
    files = weight_files(info.siblings, "*.safetensors")
    return {
        "n_params": st.total, "precision": native, "revision": info.sha,
        "quant": {"weights_mb": round(sum(s.size for s in files) / 2**20, 1), "repo": repo,
                  "revision": info.sha, "files": len(files)},
    }
