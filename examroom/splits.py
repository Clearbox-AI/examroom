"""Public / held-out splits of a suite.

A leaderboard ranks on items nobody can train on.  `split_suite` divides a frozen
suite into two frozen suites with the same tasks and scoring code:

    <name>@<version>          public: browsable, per-item results are published
    <name>-heldout@<version>  held-out: kept in a private repo; only per-task
                              aggregates of its runs are ever published

The split is stratified by (task, category) and deterministic in (seed, item id),
so it can be reproduced and audited.  The held-out MANIFEST (hash, counts, no item
text) is copied next to the public suite as HELDOUT_MANIFEST.json, so anyone can
check which content the leaderboard's held-out runs were scored on.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from examroom.items import Suite, SuiteMeta, load_suite, save_suite, write_manifest

HELDOUT_SUFFIX = "-heldout"


def is_heldout(suite_name: str) -> bool:
    return suite_name.endswith(HELDOUT_SUFFIX)


def _rank(seed: int, item_id: str) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def split_items(suite: Suite, heldout_frac: float, seed: int,
                group_id: str | None = None) -> tuple[dict, dict]:
    """(public, heldout) items per task.

    Items whose ids are equal once `group_id` (a regex) is removed form one group
    and land on the same side: translations of a held-out item must not be public
    (e.g. group_id="-(en|it)-" ties crc-mcq-en-000 to crc-mcq-it-000).  Groups are
    stratified by the (task, category) pairs they cover, and in each stratum the
    held-out share is rounded to the nearest group."""
    if not 0 < heldout_frac < 1:
        raise ValueError("heldout_frac must be in (0, 1)")
    rx = re.compile(group_id) if group_id else None
    groups: dict[str, list] = {}
    for t, it in suite.iter_items():
        groups.setdefault(rx.sub("", it.id) if rx else it.id, []).append((t.name, it))
    strata: dict[tuple, list[str]] = {}
    for key, members in groups.items():
        strata.setdefault(tuple(sorted({(t, it.category or "") for t, it in members})), []).append(key)
    hold_ids: set[str] = set()
    for keys in strata.values():
        ranked = sorted(keys, key=lambda k: _rank(seed, k))
        for k in ranked[:round(len(keys) * heldout_frac)]:
            hold_ids |= {it.id for _, it in groups[k]}
    public: dict[str, list] = {}
    heldout: dict[str, list] = {}
    for t in suite.tasks:  # keep the source order inside each split
        public[t.name] = [it for it in suite.items[t.name] if it.id not in hold_ids]
        heldout[t.name] = [it for it in suite.items[t.name] if it.id in hold_ids]
        if not public[t.name] or not heldout[t.name]:
            raise ValueError(f"task {t.name} ({len(suite.items[t.name])} items) is too small to split "
                             f"at {heldout_frac:.0%}")
    return public, heldout


def _copy_support_files(src: Path, dst: Path) -> None:
    """Everything but item files and manifests: checker code, reference stats, README."""
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".jsonl" and f.name not in ("suite.json", "MANIFEST.json",
                                                                    "HELDOUT_MANIFEST.json"):
            shutil.copy2(f, dst / f.name)


def quoted_heldout(public_dir: str | Path, heldout: Suite, min_chars: int = 40) -> list[tuple[str, str]]:
    """(file, item id) for held-out items whose prompt (or reference answer) appears
    in a public support file, e.g. a README showing sample items."""
    texts = {f.name: " ".join(f.read_text(errors="ignore").split())
             for f in Path(public_dir).iterdir() if f.is_file() and f.suffix in (".md", ".txt", ".py", ".json")
             and f.name not in ("MANIFEST.json", "HELDOUT_MANIFEST.json")}
    hits = []
    for _, it in heldout.iter_items():
        for field in (it.prompt, it.target or "", it.good or ""):
            frag = " ".join(field.split())
            if len(frag) < min_chars:
                continue
            frag = frag[:120]
            hits += [(name, it.id) for name, t in texts.items() if frag in t]
    return sorted(set(hits))


def split_suite(src: str | Path, public_dir: str | Path, heldout_dir: str | Path,
                heldout_frac: float = 0.5, seed: int = 0, group_id: str | None = None) -> tuple[Suite, Suite]:
    suite = load_suite(src)
    if is_heldout(suite.name):
        raise ValueError(f"{suite.name} is already a held-out split")
    mf = Path(suite.root) / "MANIFEST.json"
    if not mf.is_file() or json.loads(mf.read_text()).get("content_hash") != suite.content_hash():
        raise ValueError(f"{suite.root} is not frozen (or changed since): split only frozen suites")
    public_dir, heldout_dir = Path(public_dir), Path(heldout_dir)
    for d in (public_dir, heldout_dir):
        if d.exists() and any(d.iterdir()):
            raise FileExistsError(f"{d} is not empty")
    pub_items, hold_items = split_items(suite, heldout_frac, seed, group_id)
    origin = {"suite": suite.name, "version": suite.version, "content_hash": suite.content_hash(),
              "heldout_frac": heldout_frac, "seed": seed, "group_id": group_id}
    out = []
    for d, name, items, part in ((public_dir, suite.name, pub_items, "public"),
                                 (heldout_dir, suite.name + HELDOUT_SUFFIX, hold_items, "heldout")):
        meta = SuiteMeta(name=name, version=suite.version, description=suite.meta.description,
                         tasks=suite.tasks, meta={**suite.meta.meta, "split": part, "split_of": origin})
        save_suite(d, meta, items)
        _copy_support_files(Path(suite.root), d)
        s = load_suite(d)
        write_manifest(s)
        out.append(s)
    public, heldout = out
    shutil.copy2(heldout_dir / "MANIFEST.json", public_dir / "HELDOUT_MANIFEST.json")
    return public, heldout
