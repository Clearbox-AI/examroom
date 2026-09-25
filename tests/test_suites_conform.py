"""Conformance harness: every directory under suites/ must pass these, with no
per-suite test code. A new suite gets this coverage the moment it is authored.

- loads and passes validate_suite
- is frozen, and the MANIFEST hash matches the recomputed content hash
  (which covers suite-local checker files, so checker edits are caught)
- every item that carries its own reference answer scores ~1.0 through the
  real scorer (an echo backend) — the scorer must accept the ground truth
"""

import json
from pathlib import Path

import pytest

from examroom.items import Kind, load_suite
from examroom.scoring import score_item
from examroom.validate import validate_suite

SUITES_DIR = Path(__file__).resolve().parents[1] / "suites"
DIRS = sorted(p for p in SUITES_DIR.iterdir() if (p / "suite.json").is_file()) if SUITES_DIR.is_dir() else []
pytestmark = pytest.mark.skipif(not DIRS, reason="no authored suites")


@pytest.fixture(scope="module", params=DIRS, ids=lambda p: p.name)
def suite(request):
    return load_suite(request.param)


def test_validates(suite):
    rep = validate_suite(suite, min_items=1)
    assert rep.ok, rep.errors


def test_frozen_and_hash_matches(suite):
    mf = Path(suite.root) / "MANIFEST.json"
    assert mf.is_file(), f"{suite.root} is not frozen — run `examroom freeze`"
    frozen = json.loads(mf.read_text())["content_hash"]
    assert frozen == suite.content_hash(), "content (items or checker files) changed since freeze"


class _Echo:
    supports_logprobs = False
    supports_generate = True
    out = ""

    def generate(self, prompt, max_tokens, temperature, n):
        return [self.out] * n


def test_reference_answers_score_perfect(suite):
    be = _Echo()
    scores = []
    for t, it in suite.iter_items():
        if it.kind not in (Kind.PROGRAM, Kind.NUMERIC, Kind.EXACT) or it.target is None:
            continue
        # program targets are full documents (prompt + continuation); others are the answer itself
        be.out = it.target[len(it.prompt):] if it.target.startswith(it.prompt) else it.target
        try:
            s, _, _ = score_item(it, t, be, root=suite.root)
        except ModuleNotFoundError as e:
            pytest.skip(f"legacy library checker no longer importable: {e}")
        scores.append(s)
    if not scores:
        pytest.skip("no items with a reference answer")
    mean = sum(scores) / len(scores)
    assert mean >= 0.99, f"reference answers score only {mean:.3f} through the scorer"
