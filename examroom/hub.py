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
import json
from pathlib import Path

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


def parse_suite_ref(ref: str) -> tuple[str, str, str | None]:
    """hf://<org>/<repo>/<suite dir>[@<revision>] -> (repo id, suite dir, revision)."""
    body = ref.removeprefix("hf://")
    body, _, rev = body.partition("@")
    parts = body.strip("/").split("/")
    if len(parts) < 3:
        raise ValueError(f"suite ref must be hf://<org>/<repo>/<suite dir>[@<revision>], got {ref!r}")
    return "/".join(parts[:2]), "/".join(parts[2:]), rev or None


def fetch_suite(ref: str) -> Path:
    """Download one suite directory from a Hugging Face dataset repo (cached) and
    return its local path.  Suites carry checker code that runs when items are
    scored: load them only from repos you trust, pinned to a revision; the run
    still refuses content whose hash does not match its MANIFEST.  Private repos
    (held-out splits) need HF_TOKEN or `hf auth login`."""
    from huggingface_hub import snapshot_download

    repo, path, rev = parse_suite_ref(ref)
    local = snapshot_download(repo_id=repo, repo_type="dataset", revision=rev, allow_patterns=[f"{path}/*"])
    root = Path(local) / path
    if not (root / "suite.json").is_file():
        raise FileNotFoundError(f"{ref}: no suite.json at {path!r} in {repo}" + (f"@{rev}" if rev else ""))
    return root


# --- publishing -------------------------------------------------------------------
# Every publish refuses rather than guesses: the repo must already exist (creating
# public repos stays a deliberate step), a held-out suite only goes to a private
# repo, and a published suite directory is never overwritten with other content.
def _api():
    from huggingface_hub import HfApi

    return HfApi()


def _remote_manifest(api, repo: str, path: str) -> dict | None:
    from huggingface_hub import hf_hub_download

    if f"{path}/MANIFEST.json" not in set(api.list_repo_files(repo, repo_type="dataset")):
        return None
    return json.loads(Path(hf_hub_download(repo, f"{path}/MANIFEST.json", repo_type="dataset")).read_text())


def publish_suite(suite_dir: str | Path, repo: str) -> str:
    """Upload a frozen suite directory to <repo>/<dir name>; returns the commit URL
    (or a note when that exact content is already published)."""
    from examroom.items import load_suite
    from examroom.splits import is_heldout

    suite = load_suite(suite_dir)
    mf = Path(suite.root) / "MANIFEST.json"
    if not mf.is_file() or json.loads(mf.read_text()).get("content_hash") != suite.content_hash():
        raise ValueError(f"{suite.root} is not frozen (or changed since freeze): run `examroom freeze`")
    api = _api()
    info = api.repo_info(repo, repo_type="dataset")
    if is_heldout(suite.name) and not info.private:
        raise PermissionError(f"{suite.name} is a held-out split and {repo} is public: publish it to a private repo")
    path = Path(suite.root).name
    remote = _remote_manifest(api, repo, path)
    if remote and remote.get("content_hash") != suite.content_hash():
        raise FileExistsError(f"{repo}/{path} already holds {remote.get('suite')}@{remote.get('version')} "
                              f"hash {remote.get('content_hash')}: published suites are immutable, use a new directory")
    if remote:
        return f"{repo}/{path} already holds this content (hash {suite.content_hash()})"
    commit = api.upload_folder(folder_path=str(suite.root), path_in_repo=path, repo_id=repo, repo_type="dataset",
                               ignore_patterns=["__pycache__/*", "*.pyc"],
                               commit_message=f"{suite.name}@{suite.version} (hash {suite.content_hash()})")
    return commit.commit_url


def publish_results(results_dir: str | Path, repo: str, extra_files: dict[str, str | Path] | None = None) -> str:
    """Upload exported run records under runs/, plus e.g. models.json and board.json at the root."""
    from examroom.export import load_exports

    recs = load_exports(results_dir)
    leaky = [r["run_id"] for r in recs if r.get("heldout") and "results" in r]
    if leaky:
        raise ValueError(f"held-out run records carry per-item results: {leaky}")
    if not recs:
        raise ValueError(f"no run records in {results_dir}")
    api = _api()
    api.repo_info(repo, repo_type="dataset")
    commit = api.upload_folder(folder_path=str(results_dir), path_in_repo="runs", repo_id=repo, repo_type="dataset",
                               allow_patterns=["*.json"], commit_message=f"{len(recs)} run records")
    for name, path in (extra_files or {}).items():
        commit = api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=repo, repo_type="dataset",
                                 commit_message=f"update {name}")
    return commit.commit_url


def publish_page(page: str | Path, space: str) -> str:
    """Upload the leaderboard page as index.html of an existing static Space."""
    api = _api()
    api.repo_info(space, repo_type="space")
    commit = api.upload_file(path_or_fileobj=str(page), path_in_repo="index.html", repo_id=space,
                             repo_type="space", commit_message="update leaderboard")
    return commit.commit_url
