import json
from pathlib import Path

import pytest

from examroom.authoring.templates import Builder
from examroom.backends.base import BackendInfo, Stats
from examroom.backends.dummy import DummyBackend
from examroom.diff import diff_runs, mcnemar_exact_p, wilson
from examroom.items import Item, Kind, SuiteMeta, Task, load_suite, save_suite
from examroom.ledger import Ledger
from examroom.runner import run_suite
from examroom.scoring import first_number, last_number, load_checker, parse_letter, score_item
from examroom.validate import prior_accuracy, validate_suite


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
    assert first_number(".69") == 0.69 and first_number("x = .5e2") == 50.0
    assert last_number("In Study 2, the rate was 41.6%.") == 41.6
    assert last_number("no digits") is None


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


def test_numeric_answer_prefix():
    from examroom.scoring import score_numeric

    class Echo:
        def __init__(self, text): self.text = text
        def generate(self, prompt, max_tokens, temperature, n): return [self.text] * n

    task = Task(name="t", file="f", kind=Kind.NUMERIC, scoring={"answer_prefix": "ANSWER:"})
    it = Item(id="x", kind=Kind.NUMERIC, prompt="p", target="50.3", tolerance=0.05)
    assert score_numeric(it, task, Echo("144/228 = 63.2%, so...\nAnswer: 50.3"))[0] == 1.0
    assert score_numeric(it, task, Echo("50.3"))[0] == 1.0          # no marker: first-number fallback
    assert score_numeric(it, task, Echo("first 7 then ANSWER: 50.3"))[0] == 1.0
    assert score_numeric(it, task, Echo("144/228, ANSWER: 63.2"))[0] == 0.0


def test_numeric_policies_prefix_list_last_number_decimal_comma():
    from examroom.scoring import score_numeric

    class Echo:
        def __init__(self, text): self.text = text
        def generate(self, prompt, max_tokens, temperature, n): return [self.text] * n

    it = Item(id="x", kind=Kind.NUMERIC, prompt="p", target="68.6", tolerance=0.05)
    both = Task(name="t", file="f", kind=Kind.NUMERIC,
                scoring={"answer_prefix": ["RISPOSTA:", "ANSWER:"], "decimal_comma": True})
    assert score_numeric(it, both, Echo("calcolo 12,3 ...\nRISPOSTA: 68,6"))[0] == 1.0   # Italian marker + comma
    assert score_numeric(it, both, Echo("working 12.3\nANSWER: 68.6"))[0] == 1.0        # English marker accepted too
    assert score_numeric(it, both, Echo("ANSWER: 1\nRISPOSTA: 68.6"))[0] == 1.0         # the LAST marker of either wins
    assert score_numeric(it, both, Echo("1,234 patients; RISPOSTA: 68.6"))[0] == 1.0    # a thousands group is untouched
    last = Task(name="t", file="f", kind=Kind.NUMERIC, scoring={"number": "last"})
    it2 = Item(id="y", kind=Kind.NUMERIC, prompt="p", target="41.6", tolerance=0.001)
    assert score_numeric(it2, last, Echo("In Study 2, the rate was 41.6%"))[0] == 1.0    # echo no longer read as the answer
    assert score_numeric(it2, last, Echo('3{ "text": "41.6" }'))[0] == 1.0
    assert score_numeric(it2, Task(name="t", file="f", kind=Kind.NUMERIC), Echo("In Study 2, 41.6"))[0] == 0.0
    with pytest.raises(ValueError):
        score_numeric(it2, Task(name="t", file="f", kind=Kind.NUMERIC, scoring={"number": "middle"}), Echo("1"))


def test_suite_local_checker(tmp_path):
    root = tmp_path / "s"
    root.mkdir()
    (root / "checker.py").write_text(
        "def score(item, output):\n    return {'score': float('ok' in output), 'tiers': {'t1': 1.0}}\n")
    meta = SuiteMeta(name="s", version="1.0", tasks=[Task(name="t", file="t.jsonl", kind=Kind.PROGRAM)])
    items = {"t": [Item(id="p", kind=Kind.PROGRAM, prompt="q", checker="./checker.py:score")]}
    s = save_suite(root, meta, items)

    class Echo:
        supports_logprobs, supports_generate = False, True
        def generate(self, prompt, max_tokens, temperature, n):
            return ["ok"] * n

    score, detail, _ = score_item(items["t"][0], s.tasks[0], Echo(), root=s.root)
    assert score == 1.0 and detail["judge"]["tiers"] == {"t1": 1.0}
    # module-style specs still resolve; a missing checker file is a clear error
    assert load_checker("examroom.scoring:first_number")("3 m") == 3
    with pytest.raises(FileNotFoundError):
        load_checker("./nope.py:score", root)
    # the checker file is part of the content hash: editing it moves the benchmark
    h = s.content_hash()
    (root / "checker.py").write_text("def score(item, output):\n    return 1.0\n")
    assert load_suite(root).content_hash() != h


def test_checker_preflight_runs_before_any_model_call(tmp_path):
    root = tmp_path / "s"
    root.mkdir()
    (root / "checker.py").write_text(
        "def score(item, output):\n    return 1.0\n"
        "def _probe():\n    raise RuntimeError('pandas missing')\n"
        "score.preflight = _probe\n")
    meta = SuiteMeta(name="s", version="1.0", tasks=[Task(name="t", file="t.jsonl", kind=Kind.PROGRAM)])
    items = {"t": [Item(id="p", kind=Kind.PROGRAM, prompt="q", checker="./checker.py:score")]}
    s = save_suite(root, meta, items)

    class Counting:
        supports_logprobs, supports_generate = False, True
        info, stats, calls = BackendInfo(backend="dummy", model="x"), Stats(), 0

        def generate(self, prompt, max_tokens, temperature, n):
            self.calls += 1
            return ["x"] * n

    be = Counting()
    with pytest.raises(RuntimeError, match="pandas missing"):
        run_suite(s, be)
    assert be.calls == 0  # the environment gap surfaced before any model call


def test_stats():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(10, 0) < 0.01
    lo, hi = wilson(0.5, 800)
    assert 0.46 < lo < 0.47 and 0.53 < hi < 0.54


def test_parallel_workers_match_sequential(tmp_path):
    s = make_suite(tmp_path)
    r1 = run_suite(s, DummyBackend("uniform"))
    r4 = run_suite(s, DummyBackend("uniform"), workers=4)
    assert [(r.item_id, r.score) for r in r1.results] == [(r.item_id, r.score) for r in r4.results]
    assert r4.stats["prompt_tokens"] == r1.stats["prompt_tokens"]
    # a backend that opts out stays sequential (still one result per item, in order)
    be = DummyBackend("uniform")
    be.parallel_safe = False
    r = run_suite(s, be, workers=4)
    assert [x.item_id for x in r.results] == [x.item_id for x in r1.results]


def test_generative_fallback_for_backend_without_logprobs(tmp_path):
    s = make_suite(tmp_path)
    be = DummyBackend("uniform")
    be.supports_logprobs = False
    r = run_suite(s, be, limit=3)
    assert all(x.protocol == "choice_generate" for x in r.results)


def test_report_renders(tmp_path):
    import json
    from examroom.ledger import Ledger
    from examroom.report import render_report
    from examroom.runner import ItemResult, RunResult

    led = Ledger(tmp_path / "l.sqlite")
    for rid, score in [("run-a", 1.0), ("run-b", 0.0)]:
        led.record(RunResult(
            run_id=rid, ts=f"2026-08-28T10:00:0{score:.0f}", suite="s", suite_version="1.0",
            suite_hash="h" * 16, backend={"backend": "dummy", "model": "m"},
            stats={"tokens_per_second": 10.0}, label=rid, git_commit=None,
            results=[ItemResult("t", "item-1", score, "numeric",
                                {"outputs": ["x" * 5000], "per_sample": [score]}, 0.1)]))
    out = render_report(led, tmp_path / "r.html")
    html = out.read_text()
    assert "run-a" in html and "run-b" in html and "item-1" in html
    payload = html.split('type="application/json">')[1].split("</script>")[0]
    data = json.loads(payload.replace("<\\/", "</"))
    assert len(data["runs"]) == 2
    (item,) = data["runs"][0]["items"]
    assert "truncated" in item["outputs"][0] and len(item["outputs"][0]) < 2200
