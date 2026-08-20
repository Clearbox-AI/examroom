import json
from pathlib import Path

import pytest

from benchsmith.authoring.templates import Builder
from benchsmith.backends.dummy import DummyBackend
from benchsmith.diff import diff_runs, mcnemar_exact_p, wilson
from benchsmith.items import Item, Kind, SuiteMeta, Task, load_suite, save_suite
from benchsmith.ledger import Ledger
from benchsmith.runner import run_suite
from benchsmith.scoring import first_number, parse_letter, score_item
from benchsmith.validate import prior_accuracy, validate_suite


def make_suite(tmp_path: Path, n=12):
    b = Builder("days", seed=1)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday"]
    for i, d in enumerate(days):
        for tpl in ["The day after {} is", "If today is {}, tomorrow is", "{} is followed by"]:
            nxt = days[(i + 1) % 4]
            b.add(tpl.format(d), " " + nxt, [" " + x for x in b.pick(days, 2, {nxt, d})], "t")
    items = {"days": b.items()}
    meta = SuiteMeta(name="toy", version="1.0", tasks=[Task(name="days", file="days.jsonl", kind=Kind.CHOICE_LOGLIK)])
    save_suite(tmp_path / "toy", meta, items)
    return load_suite(tmp_path / "toy")


def test_item_validation():
    with pytest.raises(ValueError):
        Item(id="x", kind=Kind.CHOICE_LOGLIK, prompt="p", options=[" a"], answer_idx=0)
    with pytest.raises(ValueError):
        Item(id="x", kind=Kind.CHOICE_LOGLIK, prompt="p", options=[" a", " b"], answer_idx=2)
    with pytest.raises(ValueError):
        Item(id="x", kind=Kind.PAIR_LOGLIK, prompt="p", good="a", bad="a")
    it = Item(id="x", kind=Kind.EXACT, prompt="2+2=", target="4")
    assert it.answer == "4"


def test_suite_roundtrip_and_hash(tmp_path):
    s = make_suite(tmp_path)
    assert len(s) == 12
    h = s.content_hash()
    assert load_suite(tmp_path / "toy").content_hash() == h
    # editing an item changes the hash
    p = tmp_path / "toy" / "days.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    rows[0]["prompt"] += "!"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert load_suite(tmp_path / "toy").content_hash() != h


def test_validate_balance_and_conventions(tmp_path):
    s = make_suite(tmp_path)
    rep = validate_suite(s, min_items=5)
    assert rep.ok, rep.errors
    # inject imbalance: every answer ' Monday'
    bad = [Item(id=f"b{i}", kind=Kind.CHOICE_LOGLIK, prompt=f"q{i}", options=[" Monday", " Tuesday"], answer_idx=0)
           for i in range(30)]
    s.items["days"] = bad
    rep = validate_suite(s, min_items=5)
    assert any("rebalance" in e for e in rep.errors)
    # mixed leading space
    s.items["days"] = [Item(id="m", kind=Kind.CHOICE_LOGLIK, prompt="q", options=[" a", "b"], answer_idx=0)]
    rep = validate_suite(s, min_items=0)
    assert any("leading-space" in e for e in rep.errors)


def test_prior_accuracy_loo_handles_unique_answers():
    items = [Item(id=str(i), kind=Kind.CHOICE_LOGLIK, prompt=f"q{i}", options=[f" a{i}", " z", " y"], answer_idx=0)
             for i in range(10)]
    prior, chance, _ = prior_accuracy(items)
    assert abs(prior - chance) < 1e-9  # unique answers are not a giveaway


def test_leakage_check(tmp_path):
    s = make_suite(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    item = s.items["days"][0]
    (corpus / "doc.txt").write_text("filler " * 5 + item.prompt + " " + " ".join(["x"] * 20))
    rep = validate_suite(s, min_items=0, corpus_dir=corpus, ngram=4)
    assert any("4-gram" in e for e in rep.errors)


def test_rebalance():
    b = Builder("t", seed=0)
    for i in range(40):
        b.add(f"q{i}", " A", [" B", " C"], "x")
    for i in range(10):
        b.add(f"r{i}", " B", [" A", " C"], "x")
    dropped = b.rebalance(max_ratio=1.5)
    c = b.answer_counts()
    assert dropped > 0 and c[" A"] <= int(10 * 1.5) + 1


def test_scoring_helpers():
    assert parse_letter("The answer is B.", 4) == 1
    assert parse_letter("b", 4) == 1
    assert parse_letter("Banana", 4) is None
    assert first_number("about 1,234.5 km") == 1234.5


def test_run_ledger_diff(tmp_path):
    s = make_suite(tmp_path)
    led = Ledger(tmp_path / "l.sqlite")
    r1 = run_suite(s, DummyBackend("uniform"), label="a")
    r2 = run_suite(s, DummyBackend("oracle"), label="b")
    assert len(r1.results) == 12 and 0 <= r1.score <= 1
    assert r1.stats["prompt_tokens"] > 0
    led.record(r1); led.record(r2)
    assert len(led.runs()) == 2
    d = diff_runs(led, r1.run_id[:6], r2.run_id)
    assert d.same_benchmark and d.tasks[0].n == 12
    assert len(d.tasks[0].fixed) + len(d.tasks[0].broken) <= 12
    hist = led.item_history(s.items["days"][0].id)
    assert len(hist) == 2


def test_length_normalisation_matters():
    # oracle backend prefers short continuations under 'sum', not under 'mean'
    it = Item(id="x", kind=Kind.CHOICE_LOGLIK, prompt="p", options=[" a b c", " d"], answer_idx=0)
    be = DummyBackend("oracle")
    _, det_sum, _ = score_item(it, Task(name="t", file="f", kind=Kind.CHOICE_LOGLIK, scoring={"normalize": "sum"}), be)
    _, det_mean, _ = score_item(it, Task(name="t", file="f", kind=Kind.CHOICE_LOGLIK, scoring={"normalize": "mean"}), be)
    assert det_sum["pred_idx"] == 1
    assert det_mean["tie"]  # equal per-token logprob -> tie, counted as wrong


def test_stats():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(10, 0) < 0.01
    lo, hi = wilson(0.5, 800)
    assert 0.46 < lo < 0.47 and 0.53 < hi < 0.54


def test_generative_fallback_for_backend_without_logprobs(tmp_path):
    s = make_suite(tmp_path)
    be = DummyBackend("uniform")
    be.supports_logprobs = False
    r = run_suite(s, be, limit=3)
    assert all(x.protocol == "choice_generate" for x in r.results)
