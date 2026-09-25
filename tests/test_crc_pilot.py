"""crc_pilot v0.4 suite-local checker: the executed hard tool tier."""

import json
import subprocess
from pathlib import Path

import pytest

from benchsmith.items import load_suite
from benchsmith.scoring import checker_module

SUITE = Path(__file__).resolve().parents[1] / "suites" / "crc_pilot_v0.4"
pytestmark = pytest.mark.skipif(not SUITE.exists(), reason="crc_pilot v0.4 not authored")


@pytest.fixture(scope="module")
def suite():
    return load_suite(SUITE)


@pytest.fixture(scope="module")
def ck(suite):
    return checker_module("./checker.py", suite.root)


def call(code: str) -> str:
    return json.dumps({"tool": "execute_python", "arguments": {"code": code}})


def expect_of(it) -> str:
    return next(t.split(":", 1)[1] for t in it.tags if t.startswith("expect:"))


def test_reference_code_scores_perfect(suite, ck):
    for it in suite.items["tool_hard_en"] + suite.items["tool_hard_it"]:
        assert ck.score_tool_call_hard(it, it.target)["score"] == 1.0, it.id


def test_dumping_the_frame_or_a_mask_does_not_score(suite, ck):
    for it in suite.items["tool_hard_en"]:
        r = ck.score_tool_call_hard(it, call("print(df)"))
        assert r["tiers"] == {"call": 1.0, "runs": 1.0, "result": 0.0}, (it.id, r)
    it = next(i for i in suite.items["tool_hard_en"] if expect_of(i).isdigit())
    r = ck.score_tool_call_hard(it, call("print(df['pfs_months'] > df['pfs_months'].mean())"))
    assert r["tiers"]["result"] == 0.0


def test_prose_around_the_value_is_fine(suite, ck):
    it = suite.items["tool_hard_en"][0]
    assert ck.score_tool_call_hard(it, call(f"print('Difference:', {expect_of(it)})"))["score"] == 1.0
    it = next(i for i in suite.items["tool_hard_en"] if expect_of(i).startswith("P"))
    assert ck.score_tool_call_hard(it, call(f"print('best responder:', {expect_of(it)!r})"))["score"] == 1.0
    # a value on an earlier line does not count; only the last line is read
    assert ck.score_tool_call_hard(it, call(f"print({expect_of(it)!r}); print(df)"))["tiers"]["result"] == 0.0


def test_count_answers_vary(suite):
    counts = {expect_of(i) for i in suite.items["tool_hard_en"] if expect_of(i).isdigit()}
    assert len(counts) > 1, counts  # v0.2's median-based count was always 5


def test_runtime_error_and_bad_call(suite, ck):
    it = suite.items["tool_hard_en"][0]
    assert ck.score_tool_call_hard(it, call("print(df['nope'])"))["tiers"] == {"call": 1.0, "runs": 0.0, "result": 0.0}
    assert ck.score_tool_call_hard(it, "not a call")["score"] == 0.0


def test_missing_pandas_raises_instead_of_scoring(suite, ck, monkeypatch):
    it = suite.items["tool_hard_en"][0]
    fake = subprocess.CompletedProcess(args=[], returncode=1, stdout="",
                                       stderr="ModuleNotFoundError: No module named 'pandas'")
    monkeypatch.setattr(ck, "_ENV_OK", False)
    monkeypatch.setattr(ck.subprocess, "run", lambda *a, **k: fake)
    with pytest.raises(RuntimeError, match="pandas"):
        ck.score_tool_call_hard(it, it.target)
    with pytest.raises(RuntimeError, match="pandas"):
        ck.score_tool_call_hard.preflight()


def test_direction_is_balanced(suite):
    for task in ("extract_en", "summary_en", "extract_hard_en", "summary_hard_en"):
        tags = [t for i in suite.items[task] for t in i.tags if t.startswith("direction:")]
        assert len(tags) == len(suite.items[task])
        frac = tags.count("direction:exp") / len(tags)
        assert 0.3 <= frac <= 0.7, (task, frac)
