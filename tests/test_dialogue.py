from pathlib import Path

import pytest

from benchsmith.checks.dialogue import score
from benchsmith.items import Item, Kind, load_suite
from benchsmith.validate import validate_suite

SUITE = Path(__file__).resolve().parents[1] / "suites" / "dialogue_v1"
pytestmark = pytest.mark.skipif(not SUITE.exists(), reason="dialogue suite not authored")


def opener(text="Good morning, did you sleep well?"):
    return Item(id="o", kind=Kind.PROGRAM, prompt=f"A: {text}\nB:", checker="benchsmith.checks.dialogue:score")


def test_dialogue_checker_signals():
    good = score(opener(), " Yes, I slept very well, thanks.\nA: Great, breakfast is ready.\nB: Coming!")
    assert good["score"] == 1.0
    assert score(opener(), "")["score"] == 0.0
    off = score(opener(), " The capital of France is Paris.")
    assert off["tiers"]["on_topic"] == 0 and off["tiers"]["format"] == 1
    bad_alt = score(opener(), " Yes, thanks.\nB: And you?\nB: Fine.")
    assert bad_alt["tiers"]["alternation"] == 0
    prose = score(opener(), " Yes, thanks.\nThe next morning they went to the market.")
    assert prose["tiers"]["format"] == 0
    echo = score(opener(), " Good morning, did you sleep well?")
    assert echo["tiers"]["non_degenerate"] == 0


def test_suite_is_balanced_and_length_matched():
    s = load_suite(SUITE)
    rep = validate_suite(s, min_items=100)
    assert rep.ok, rep.errors
    for t in s.tasks:
        if t.kind == Kind.PAIR_LOGLIK:
            m = rep.metrics[t.name]
            assert abs(m["mean_len_diff"]) < 0.5, (t.name, m)
            assert all(it.prompt.endswith("B:") and it.good.startswith(" ") and it.bad.startswith(" ")
                       for it in s.items[t.name])
    assert sum(len(s.items[t.name]) for t in s.tasks if t.kind == Kind.PAIR_LOGLIK) >= 400
    assert len(s.items["openers"]) >= 100
