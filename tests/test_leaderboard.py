import json
import re

import pytest

from examroom.cli import main
from examroom.export import load_exports
from examroom.items import Item, Kind, SuiteMeta, Task, load_suite, save_suite, write_manifest
from examroom.leaderboard import build_board, load_board, render_board
from examroom.registry import Registry
from examroom.splits import quoted_heldout, split_items, split_suite


def bilingual_suite(tmp_path, n=20):
    def items(lang):
        return [Item(id=f"s-{lang}-{i:03d}", kind=Kind.NUMERIC, prompt=f"[{lang}] question number {i}", target=str(i),
                     category="odd" if i % 2 else "even") for i in range(n)]

    meta = SuiteMeta(name="bi", version="1.0",
                     tasks=[Task(name="q_en", file="q_en.jsonl", kind=Kind.NUMERIC),
                            Task(name="q_it", file="q_it.jsonl", kind=Kind.NUMERIC)])
    save_suite(tmp_path / "bi", meta, {"q_en": items("en"), "q_it": items("it")})
    (tmp_path / "bi" / "README.md").write_text("Example item: [en] question number 3\n")
    s = load_suite(tmp_path / "bi")
    write_manifest(s)
    return s


def test_split_is_stratified_grouped_and_deterministic(tmp_path):
    s = bilingual_suite(tmp_path)
    pub, held = split_items(s, 0.5, seed=1, group_id="-(en|it)-")
    key = lambda items: {re.sub("-(en|it)-", "-", it.id) for v in items.values() for it in v}
    assert not key(pub) & key(held)                                   # translations stay together
    assert {t: len(v) for t, v in held.items()} == {"q_en": 10, "q_it": 10}
    assert sum(it.category == "odd" for it in held["q_en"]) == 5     # stratified by category
    assert split_items(s, 0.5, seed=1, group_id="-(en|it)-")[1] == held
    assert split_items(s, 0.5, seed=2, group_id="-(en|it)-")[1] != held
    ungrouped = split_items(s, 0.5, seed=1)[1]
    assert key(ungrouped) & key(split_items(s, 0.5, seed=1)[0])       # without grouping they straddle
    with pytest.raises(ValueError, match="too small"):
        split_items(s, 0.01, seed=1)


def test_split_suite_writes_two_frozen_suites(tmp_path):
    s = bilingual_suite(tmp_path)
    pub, held = split_suite(s.root, tmp_path / "pub", tmp_path / "held", 0.5, seed=0, group_id="-(en|it)-")
    assert (pub.name, held.name) == ("bi", "bi-heldout") and len(pub) + len(held) == len(s)
    hm = json.loads((tmp_path / "pub" / "HELDOUT_MANIFEST.json").read_text())
    assert hm["content_hash"] == held.content_hash() and pub.meta.meta["split"] == "public"
    assert held.meta.meta["split_of"]["content_hash"] == s.content_hash()
    # the README quotes item 3; flagged only if item 3 went to the held-out side
    in_held = any(it.id == "s-en-003" for _, it in held.iter_items())
    assert bool(quoted_heldout(pub.root, held, min_chars=20)) == in_held
    with pytest.raises(FileExistsError):
        split_suite(s.root, tmp_path / "pub", tmp_path / "other")
    with pytest.raises(ValueError, match="already a held-out"):
        split_suite(held.root, tmp_path / "a", tmp_path / "b")


def test_cli_split_run_export_leaderboard(tmp_path, capsys):
    s = bilingual_suite(tmp_path)
    pub, held = tmp_path / "pub", tmp_path / "held"
    assert main(["split", str(s.root), "--public", str(pub), "--heldout", str(held),
                 "--group-id=-(en|it)-"]) in (0, 1)  # 1 only when the README quotes a held-out item
    db, res = str(tmp_path / "l.sqlite"), tmp_path / "results"
    for model, quant in (("org/small", "Q4_K_M"), ("org/tiny", "bf16")):
        assert main(["run", str(held), "dummy:uniform", "--profile", "nothink", "--opt", f"model_id={model}",
                     "--opt", f"quantization={quant}", "--ledger", db]) == 0
    assert main(["run", str(pub), "dummy:uniform", "--profile", "nothink", "--ledger", db]) == 0
    assert main(["run", str(held), "dummy:uniform", "--ledger", db]) == 0          # unprofiled: not exported
    assert main(["export", "--profiled", "--out", str(res), "--ledger", db]) == 0
    recs = load_exports(res)
    assert len(recs) == 3
    for r in recs:
        assert r["heldout"] == r["suite"].endswith("-heldout")
        assert ("results" in r) != r["heldout"]                                    # no per-item data for held-out

    reg = tmp_path / "models.json"
    main(["models", "--registry", str(reg), "add", "org/small", "--params", "8B", "--dense"])
    main(["models", "--registry", str(reg), "add", "org/tiny", "--params", "0.5B", "--dense"])
    board = tmp_path / "board.json"
    hh = load_suite(held).content_hash()
    board.write_text(json.dumps({"profiles": ["nothink@1", "think@1"],
                                 "suites": [{"name": "bi-heldout", "version": "1.0", "hash": hh, "label": "Bilingual"}]}))
    out = tmp_path / "lb.html"
    assert main(["leaderboard", str(res), "--board", str(board), "--registry", str(reg), "--out", str(out)]) == 0
    page = out.read_text()
    assert "org/small" in page and "org/tiny" in page and "Bilingual" in page
    assert "No ranked runs under this profile yet." in page                       # think@1 is empty


def test_board_latest_run_wins_and_overall_needs_every_suite(tmp_path):
    def rec(run_id, ts, suite, score, model="org/m"):
        return {"format": "examroom-run/1", "run_id": run_id, "ts": ts, "suite": suite, "suite_version": "1",
                "suite_hash": "h-" + suite, "backend": {"backend": "openai", "model": model, "profile": "nothink@1"},
                "tasks": {"a": {"score": score, "n": 10}, "b": {"score": score / 2, "n": 30}}}

    board = {"profiles": ["nothink@1"], "suites": [{"name": "x", "version": "1", "hash": "h-x"},
                                                   {"name": "y", "version": "1", "hash": "h-y"}]}
    recs = [rec("old", "2026-01-01", "x", 0.2), rec("new", "2026-02-01", "x", 0.8), rec("y1", "2026-01-01", "y", 0.4),
            rec("only-x", "2026-01-01", "x", 1.0, model="org/other")]
    view = build_board(recs, board, Registry())["views"][0]
    top = view["rows"][0]
    assert top["model"] == "openai:org/m" and top["suites"][0]["run_id"] == "new"
    assert top["suites"][0]["score"] == pytest.approx(0.6)          # task mean, not item-weighted
    assert top["overall"] == pytest.approx((0.6 + 0.3) / 2)
    assert view["rows"][1]["overall"] is None                       # missing suite y: unranked overall
    assert "openai:org/other" in render_board(build_board(recs, board, Registry()))
    bad = tmp_path / "board.json"
    bad.write_text(json.dumps({"suites": []}))
    with pytest.raises(ValueError, match="profiles"):
        load_board(bad)


def test_hf_suite_ref(monkeypatch, tmp_path):
    import examroom.hub as hub

    assert hub.parse_suite_ref("hf://clearbox-ai/examroom-suites/crc_v1@abc") == \
        ("clearbox-ai/examroom-suites", "crc_v1", "abc")
    assert hub.parse_suite_ref("hf://org/repo/a/b") == ("org/repo", "a/b", None)
    with pytest.raises(ValueError):
        hub.parse_suite_ref("hf://org/repo")
    s = bilingual_suite(tmp_path)
    calls = []

    def fake_snapshot(repo_id, repo_type, revision, allow_patterns):
        calls.append((repo_id, repo_type, revision, allow_patterns))
        return str(tmp_path)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    loaded = load_suite("hf://org/suites/bi@rev1")
    assert loaded.content_hash() == s.content_hash()
    assert calls == [("org/suites", "dataset", "rev1", ["bi/*"])]
    with pytest.raises(FileNotFoundError):
        load_suite("hf://org/suites/missing")


class FakeApi:
    def __init__(self, private=False, files=()):
        self.private, self.files, self.uploads = private, set(files), []

    def repo_info(self, repo, repo_type):
        return type("Info", (), {"private": self.private})()

    def list_repo_files(self, repo, repo_type):
        return sorted(self.files)

    def upload_folder(self, **kw):
        self.uploads.append(kw)
        return type("C", (), {"commit_url": "https://hf.co/commit/1"})()

    upload_file = upload_folder


def test_publish_guards(monkeypatch, tmp_path):
    import examroom.hub as hub

    s = bilingual_suite(tmp_path)
    pub, held = split_suite(s.root, tmp_path / "pub", tmp_path / "held", 0.5, group_id="-(en|it)-")
    api = FakeApi(private=False)
    monkeypatch.setattr(hub, "_api", lambda: api)
    with pytest.raises(PermissionError):
        hub.publish_suite(held.root, "org/public-suites")                  # held-out never goes public
    assert hub.publish_suite(pub.root, "org/public-suites").startswith("https://")
    assert api.uploads[-1]["path_in_repo"] == "pub"
    api.private = True
    assert hub.publish_suite(held.root, "org/private-suites").startswith("https://")

    # a published directory is immutable
    api.files = {"pub/MANIFEST.json"}
    monkeypatch.setattr(hub, "_remote_manifest", lambda a, r, p: {"suite": "bi", "version": "1.0", "content_hash": "other"})
    with pytest.raises(FileExistsError):
        hub.publish_suite(pub.root, "org/public-suites")
    monkeypatch.setattr(hub, "_remote_manifest", lambda a, r, p: {"content_hash": pub.content_hash()})
    n = len(api.uploads)
    assert "already holds" in hub.publish_suite(pub.root, "org/public-suites") and len(api.uploads) == n

    # results: held-out records must be aggregate-only
    res = tmp_path / "res" / "bi-heldout@1.0"
    res.mkdir(parents=True)
    (res / "r1.json").write_text(json.dumps({"format": "examroom-run/1", "run_id": "r1", "heldout": True,
                                             "results": [{"item_id": "s-en-001"}]}))
    with pytest.raises(ValueError, match="per-item"):
        hub.publish_results(tmp_path / "res", "org/results")
    (res / "r1.json").write_text(json.dumps({"format": "examroom-run/1", "run_id": "r1", "heldout": True}))
    reg = tmp_path / "models.json"
    reg.write_text("{}")
    hub.publish_results(tmp_path / "res", "org/results", {"models.json": reg})
    assert api.uploads[-1]["path_in_repo"] == "models.json"
