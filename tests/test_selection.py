import json

import pytest

from examroom.backends.hf import generation_text
from examroom.cli import main
from examroom.items import Item, Kind, SuiteMeta, Task, load_suite, save_suite
from examroom.ledger import Ledger
from examroom.runner import ItemResult, RunResult
from examroom.selection import fmt_size, max_lower_bound, minimum_for, parse_count, select


def make_suite(tmp_path, acceptance=None, n_easy=40, n_hard=12):
    def items(prefix, n):
        return [Item(id=f"{prefix}-{i:03d}", kind=Kind.NUMERIC, prompt=f"q{i}", target="1") for i in range(n)]

    meta = SuiteMeta(name="gate", version="1.0",
                     tasks=[Task(name="easy", file="easy.jsonl", kind=Kind.NUMERIC),
                            Task(name="hard_en", file="hard_en.jsonl", kind=Kind.NUMERIC)],
                     meta={"acceptance": acceptance} if acceptance else {})
    save_suite(tmp_path / "gate", meta, {"easy": items("easy", n_easy), "hard_en": items("hard", n_hard)})
    return load_suite(tmp_path / "gate")


def record(led, suite, run_id, model, n_params, correct: dict[str, int], suite_hash=None):
    results = []
    for task, k in correct.items():
        for i, it in enumerate(suite.items[task]):
            results.append(ItemResult(task, it.id, float(i < k), "numeric", {}, 0.0))
    led.record(RunResult(run_id=run_id, ts=f"2026-09-27T10:00:{len(led.runs()):02d}", suite=suite.name,
                         suite_version=suite.version, suite_hash=suite_hash or suite.content_hash(),
                         backend={"backend": "dummy", "model": model, "n_params": n_params},
                         label=None, git_commit=None, results=results))


def test_parse_and_format_sizes():
    assert parse_count("4B") == 4_000_000_000
    assert parse_count("30.5b") == 30_500_000_000
    assert parse_count("600M") == 600_000_000
    assert parse_count("1.2e9") == 1_200_000_000
    assert parse_count(135000000) == 135_000_000
    with pytest.raises(ValueError):
        parse_count("4GB")
    assert fmt_size(4_000_000_000) == "4B" and fmt_size(135_000_000) == "135M"
    assert fmt_size(2560.0, "weights_mb") == "2.5GB" and fmt_size(None) == "?"


def test_last_matching_pattern_wins():
    crit = [("*", 0.9), ("mcq_*", 0.7), ("mcq_hard_*", 0.6)]
    assert minimum_for("extract_en", crit) == 0.9
    assert minimum_for("mcq_en", crit) == 0.7
    assert minimum_for("mcq_hard_it", crit) == 0.6
    assert minimum_for("x", [("mcq_*", 0.7)]) is None


def test_max_lower_bound_matches_wilson():
    from examroom.diff import wilson
    for n in (12, 16, 100):
        assert max_lower_bound(n) == pytest.approx(wilson(1.0, n)[0])


def test_select_smallest_passing(tmp_path):
    s = make_suite(tmp_path)
    led = Ledger(tmp_path / "l.sqlite")
    record(led, s, "big", "m-27b", 27e9, {"easy": 40, "hard_en": 12})
    record(led, s, "small", "m-4b", 4e9, {"easy": 40, "hard_en": 7})     # fails hard_en
    record(led, s, "mid", "m-8b", 8e9, {"easy": 39, "hard_en": 12})
    record(led, s, "unsized", "m-x", None, {"easy": 40, "hard_en": 12})
    record(led, s, "other", "m-1b", 1e9, {"easy": 40, "hard_en": 12}, suite_hash="0" * 16)

    sel = select(led, s, [("*", 0.7)])
    assert [c.run_id for c in sel.candidates] == ["small", "mid", "big", "unsized"]  # by size, unknown last
    assert sel.skipped_runs == 1                                                     # different content
    assert [c.passed for c in sel.candidates] == [False, True, True, True]
    assert sel.best.run_id == "mid"
    assert not sel.unreachable

    # lower-bound gating: 12/12 demonstrates only 0.76, so a 0.9 minimum is unreachable
    sel = select(led, s, [("*", 0.9)])
    assert sel.unreachable and sel.unreachable[0][0] == "hard_en"
    assert sel.best is None
    # the raw score clears it
    assert select(led, s, [("*", 0.9)], gate="point").best.run_id == "mid"


def test_select_ungated_tasks_and_missing_results(tmp_path):
    s = make_suite(tmp_path)
    led = Ledger(tmp_path / "l.sqlite")
    record(led, s, "partial", "m-1b", 1e9, {"easy": 40})                    # never ran hard_en
    record(led, s, "full", "m-2b", 2e9, {"easy": 40, "hard_en": 0})
    sel = select(led, s, [("easy", 0.8)])                                   # hard_en ungated
    assert sel.minimums == {"easy": 0.8, "hard_en": None}
    assert sel.best.run_id == "partial"
    sel = select(led, s, [("*", 0.5)], gate="point")
    assert sel.candidates[0].cells["hard_en"].ok is False                   # missing = not demonstrated
    assert sel.best is None
    assert select(led, s, []).candidates[0].passed is None


def test_cli_run_records_size_then_select(tmp_path, capsys):
    s = make_suite(tmp_path, acceptance={"*": 0.0})
    db = str(tmp_path / "l.sqlite")
    assert main(["run", str(s.root), "dummy:uniform", "--allow-unfrozen", "--ledger", db,
                 "--opt", "n_params=4B", "--opt", "active_params=600M", "--opt", "quantization=Q4_K_M"]) == 0
    (row,) = Ledger(db).runs()
    assert row["n_params"] == 4_000_000_000 and row["quantization"] == "Q4_K_M"
    assert json.loads(row["backend_json"])["active_params"] == 600_000_000

    assert main(["annotate", row["run_id"][:6], "n_params=3B", "weights_mb=2048", "--ledger", db]) == 0
    row = Ledger(db).run(row["run_id"])
    assert row["n_params"] == 3_000_000_000 and json.loads(row["backend_json"])["weights_mb"] == 2048.0
    assert main(["annotate", row["run_id"], "colour=red", "--ledger", db]) == 1

    capsys.readouterr()
    assert main(["select", str(s.root), "--ledger", db]) == 0               # suite meta.acceptance: all pass
    out = capsys.readouterr().out
    assert "smallest passing: [1] uniform  n_params=3B" in out
    assert main(["select", str(s.root), "--ledger", db, "--min", "easy=1.0"]) == 1   # dummy can't pass
    assert main(["select", str(s.root), "--ledger", db, "--min", "nope=2"]) == 1


class FakeTok:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        assert not tokenize and add_generation_prompt
        think = kw.get("enable_thinking", True)
        return f"<bos><user>{messages[0]['content']}</user><assistant>{'' if think else '<nothink>'}"


def test_generation_text_uses_chat_template():
    assert generation_text(FakeTok(), "hi", use_chat=False) == ("hi", True)
    text, special = generation_text(FakeTok(), "hi", use_chat=True)
    assert text == "<bos><user>hi</user><assistant>" and special is False   # template owns BOS
    text, _ = generation_text(FakeTok(), "hi", True, {"enable_thinking": False})
    assert text.endswith("<nothink>")
