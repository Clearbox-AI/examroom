import json
from pathlib import Path

import pytest

from benchsmith.checks import calibration
from benchsmith.checks.schedules import PARSERS, ParseError, check_t2, check_t3, score
from benchsmith.items import load_suite
from benchsmith.ledger import Ledger
from benchsmith.runner import run_suite

SUITE = Path(__file__).resolve().parents[1] / "suites" / "schedules_v1"
pytestmark = pytest.mark.skipif(not SUITE.exists(), reason="schedules suite not authored")


@pytest.fixture(scope="module")
def suite():
    return load_suite(SUITE)


def ref_out(it):
    return it.target[len(it.prompt):]


def test_references_score_near_perfect(suite):
    # the generator's own output must satisfy the rules derived from it, except the
    # documented shopping-trip jitter quirk (depart/return jittered independently)
    n = perfect = 0
    other = set()
    for t, it in suite.iter_items():
        r = score(it, ref_out(it))
        n += 1
        perfect += r["score"] == 1.0
        other |= {f for f in r["failed"] if f != "t3.start_before_end"}
    assert not other, other
    assert perfect / n > 0.99


def test_all_three_parsers_agree(suite):
    docs = {t.name: PARSERS[t.name.split("_")[1]](suite.items[t.name][0].target) for t in suite.tasks}
    # different docs, but same schema shape
    for d in docs.values():
        assert set(d) == {"profile", "week"} and len(d["week"]) == 7
        assert {"occupancy", "meals", "showers", "appliances", "ev_charging", "trips"} <= set(d["week"][0])


@pytest.mark.parametrize("fmt", ["xml", "json", "kv"])
def test_broken_outputs(suite, fmt):
    it = suite.items[f"schedule_{fmt}"][0]
    out = ref_out(it)
    assert score(it, "")["tiers"]["t1"] == 0.0
    if fmt == "kv":   # line grammar: a doc cut at a line boundary still parses; T2 catches the missing days
        cut = out.rfind("\n", 0, len(out) // 2)
        trunc = score(it, out[:cut])["tiers"]
        assert trunc["t1"] == 1.0 and trunc["t2"] < 1.0
        assert score(it, out[: len(out) // 2 + 3])["tiers"]["t1"] in (0.0, 1.0)  # mid-line cut may break the grammar
    else:
        assert score(it, out[: len(out) // 2])["tiers"]["t1"] == 0.0
    # trailing prose after the structure is tolerated (the training docs end with a summary)
    assert score(it, out + "\n\nThis home has 2 people.")["tiers"]["t1"] == 1.0


def test_semantic_violations_detected(suite):
    it = suite.items["schedule_kv"][0]
    out = ref_out(it)
    bad = out.replace("day: saturday", "day: caturday")
    r = score(it, bad)
    assert r["tiers"]["t1"] == 1.0 and "t2.seven_days_in_order" in r["failed"]
    bad = out.replace("meal: breakfast", "meal: lunch", 1)
    r = score(it, bad)
    assert "t3.breakfast_and_dinner_present" in r["failed"]


def test_kv_grammar_strict(suite):
    it = suite.items["schedule_kv"][0]
    with pytest.raises(ParseError):
        PARSERS["kv"](it.target + "\n  meal: brunch at=noon")


def test_calibration_on_references_is_near_zero(suite):
    ref = json.loads((SUITE / "reference_stats.json").read_text())
    docs = [PARSERS[it.category](it.target) for _, it in suite.iter_items()]
    model = calibration.collect(docs)
    kls = [calibration.kl(model[k], ref["fields"][k])[0] for k in ref["fields"] if k in model]
    assert max(kls) < 0.05, max(kls)   # 900 docs vs 50k reference: tiny sampling noise only


def test_program_kind_runs_through_runner_and_ledger(suite, tmp_path):
    class RefBackend:  # echoes the reference continuation for whatever prompt it sees
        supports_logprobs = False
        supports_generate = True

        def __init__(self):
            from benchsmith.backends.base import BackendInfo, Stats
            self.info = BackendInfo(backend="dummy", model="reference")
            self.stats = Stats()
            self.lookup = {it.prompt: ref_out(it) for _, it in suite.iter_items()}

        def generate(self, prompt, max_tokens, temperature, n):
            return [self.lookup[prompt]] * n

        def loglik(self, *a):
            raise NotImplementedError

        def close(self):
            pass

    run = run_suite(suite, RefBackend(), limit=4)
    assert len(run.results) == 12 and run.score > 0.99
    assert run.results[0].detail["judge"]["tiers"]["t1"] == 1.0
    led = Ledger(tmp_path / "l.sqlite")
    led.record(run)
    ref = json.loads((SUITE / "reference_stats.json").read_text())
    rep = calibration.calibrate({it.id: it for _, it in suite.iter_items()}, led.results(run.run_id), ref)
    assert rep["parsed"] == 12
    led.set_metric(run.run_id, "t4_mean_kl", rep["mean_kl"], rep)
    assert led.metrics(run.run_id)["t4_mean_kl"][0] == rep["mean_kl"]
