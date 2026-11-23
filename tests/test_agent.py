"""The agent scorer: a tool-calling loop against a suite-local mock API."""

import json

import pytest

from examroom.items import Item, Kind, Suite, SuiteMeta, Task
from examroom.scoring import resolve_protocol, score_item

ENV = '''
TOOLS = [{"name": "add", "description": "a + b", "inputSchema": {"type": "object",
          "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}]

def call(name, args, fixture):
    if name != "add":
        return {"error": "unknown tool"}
    return {"sum": args["a"] + args["b"] + fixture.get("bias", 0)}
'''
CHECKER = '''
def score(item, ep):
    sums = [s["result"].get("sum") for s in ep["steps"]]
    return {"score": float(ep["stop"] == "final" and str(sums[-1]) in ep["final"]) if sums else 0.0,
            "n": len(ep["steps"])}
'''


class Scripted:
    supports_logprobs = False
    supports_generate = True
    supports_tools = True

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    def chat(self, messages, tools, max_tokens, temperature):
        self.seen.append([dict(m) for m in messages])
        return self.turns.pop(0)


def _call(name, args, i=0):
    return {"content": "", "tool_calls": [{"id": f"c{i}", "name": name, "arguments": args}]}


@pytest.fixture
def suite_dir(tmp_path):
    (tmp_path / "env.py").write_text(ENV)
    (tmp_path / "chk.py").write_text(CHECKER)
    return tmp_path


def _item(**kw):
    return Item(id="a1", kind=Kind.AGENT, prompt="add 2 and 3", checker="./chk.py:score", env="./env.py",
                fixture={"bias": 10}, **kw)


def test_episode_runs_tools_and_scores(suite_dir):
    task = Task(name="t", file="t.jsonl", kind=Kind.AGENT, scoring={"system": "sys"})
    be = Scripted([_call("add", '{"a": 2, "b": 3}'), {"content": "it is 15", "tool_calls": []}])
    score, detail, proto = score_item(_item(), task, be, root=suite_dir)
    assert (score, proto) == (1.0, Kind.AGENT)
    ep = detail["episodes"][0]
    assert ep["stop"] == "final" and ep["steps"][0]["result"] == {"sum": 15}
    assert be.seen[0][0] == {"role": "system", "content": "sys"}
    tool_msg = be.seen[1][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"]) == {"sum": 15}
    assert "-> add(" in detail["outputs"][0] and "FINAL: it is 15" in detail["outputs"][0]


def test_bad_arguments_are_reported_to_the_model(suite_dir):
    task = Task(name="t", file="t.jsonl", kind=Kind.AGENT)
    be = Scripted([_call("add", "{not json"), _call("add", '[1, 2]', 1), {"content": "?", "tool_calls": []}])
    _, detail, _ = score_item(_item(), task, be, root=suite_dir)
    steps = detail["episodes"][0]["steps"]
    assert [s["arguments"] for s in steps] == [None, None]
    assert steps[0]["raw"] == "{not json" and "error" in steps[1]["result"]


def test_turn_budget_ends_the_episode(suite_dir):
    task = Task(name="t", file="t.jsonl", kind=Kind.AGENT, scoring={"max_turns": 2})
    be = Scripted([_call("add", '{"a": 1, "b": 1}', i) for i in range(5)])
    score, detail, _ = score_item(_item(), task, be, root=suite_dir)
    ep = detail["episodes"][0]
    assert score == 0.0 and ep["stop"] == "max_turns" and len(ep["steps"]) == 2


def test_agent_needs_tool_backend_and_env_is_hashed(suite_dir):
    class NoTools:
        supports_logprobs = False
        supports_generate = True
    with pytest.raises(RuntimeError):
        resolve_protocol(Kind.AGENT, NoTools())
    with pytest.raises(ValueError):
        Item(id="x", kind=Kind.AGENT, prompt="p", checker="./chk.py:score")   # no env
    meta = SuiteMeta(name="s", version="1", tasks=[Task(name="t", file="t.jsonl", kind=Kind.AGENT)])
    suite = Suite(suite_dir, meta, {"t": [_item()]})
    h = suite.content_hash()
    (suite_dir / "env.py").write_text(ENV + "\n# changed\n")
    assert suite.content_hash() != h
