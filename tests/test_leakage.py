import pyarrow as pa
import pyarrow.parquet as pq

from benchsmith.items import Item, Kind, Suite, SuiteMeta, Task
from benchsmith.leakage import check_leakage


def mk(items, meta=None, tmp_path=None):
    m = SuiteMeta(name="s", version="1", tasks=[Task(name="t", file="t.jsonl", kind=items[0].kind)], meta=meta or {})
    return Suite(tmp_path, m, {"t": items})


def test_ngram_streaming_parquet(tmp_path):
    long_prompt = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    it = Item(id="a", kind=Kind.CHOICE_LOGLIK, prompt=long_prompt, options=[" x", " y"], answer_idx=0)
    short = Item(id="b", kind=Kind.CHOICE_LOGLIK, prompt="too short", options=[" x", " y"], answer_idx=0)
    pq.write_table(pa.table({"text": ["filler " + long_prompt + " tail", "nothing here"], "source": ["w", "w"]}),
                   tmp_path / "shard_00000.parquet")
    rep = check_leakage(mk([it, short], tmp_path=tmp_path), [tmp_path], n=13)
    assert rep.docs_scanned == 2 and rep.items_too_short == 1
    assert set(rep.collisions) == {"a"} and rep.collisions["a"][0][0] == "shard_00000.parquet:0"


def test_exact_mode_prefix_and_identity(tmp_path):
    doc = "household: HH1\nprofile: x\nday: monday\n  meal: breakfast\n\nThis home has 2 people."
    it = Item(id="p", kind=Kind.PROGRAM, prompt="household: HH1\nprofile: x\n", target=doc.split("\n\n")[0],
              checker="m:f", category="kv")
    (tmp_path / "c.jsonl").write_text('{"text": %s}\n{"text": "household: HH2\\nprofile: y\\n"}\n' % __import__("json").dumps(doc))
    rep = check_leakage(mk([it], meta={"leakage_check": "exact"}, tmp_path=tmp_path), [tmp_path])
    kinds = {k for _, k in rep.collisions["p"]}
    assert "identical document" in kinds and any("prompt prefix" in k for k in kinds)
    # a different seed's doc with the same boilerplate does not collide
    it2 = it.model_copy(update={"id": "q", "prompt": "household: HH9\nprofile: z\n", "target": "household: HH9\nprofile: z\nday: monday"})
    rep = check_leakage(mk([it2], meta={"leakage_check": "exact"}, tmp_path=tmp_path), [tmp_path])
    assert rep.ok
