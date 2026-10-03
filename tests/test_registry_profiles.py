import json

import pytest

from examroom.backends.base import strip_reasoning
from examroom.cli import main
from examroom.items import Item, Kind, Task
from examroom.ledger import Ledger
from examroom.profiles import PROFILES, get_profile
from examroom.registry import Registry, tiers_for
from examroom.scoring import score_item
from examroom.selection import select
from tests.test_selection import make_suite, record

GB = 1024  # weights_mb per GiB


def reg_with_models():
    reg = Registry()
    reg.add("Qwen/Qwen3-0.6B", "0.6B", aliases=["qwen/qwen3-0.6b"])
    reg.add("Qwen/Qwen3-14B", "14B")
    reg.add("Qwen/Qwen3.6-27B", "27B", aliases=["qwen3.6-27b"])
    reg.add("Qwen/Qwen3-Next-80B-A3B", "80B", active_params="3B", aliases=["qwen/qwen3-next-80b-a3b"])
    reg.add_quant("Qwen/Qwen3-Next-80B-A3B", "Q4_K_M", weights_mb=48 * GB)
    return reg


def test_tiers_are_inclusive_and_overlap():
    assert tiers_for(1_000_000_000, 2 * GB) == ["tiny", "small", "local"]
    assert tiers_for(14_800_000_000, 28 * GB) == ["small", "local"]   # a 14B-class model with extra heads
    assert tiers_for(15_000_000_000, None) == ["small"]
    assert tiers_for(15_100_000_000, None) == []
    assert tiers_for(80_000_000_000, 128 * GB) == ["local"]
    assert tiers_for(80_000_000_000, 128 * GB + 1) == []


def test_registry_lookup_by_id_alias_or_model_id():
    reg = reg_with_models()
    assert reg.find("QWEN/qwen3-next-80b-a3b") == "Qwen/Qwen3-Next-80B-A3B"
    assert reg.find("qwen3.6-27b") == "Qwen/Qwen3.6-27B"
    assert reg.find("unknown") is None
    ms = reg.size({"backend": "openai", "model": "my-local-name", "model_id": "Qwen/Qwen3.6-27B",
                   "quantization": "Q4_K_M"})
    assert ms.model_id == "Qwen/Qwen3.6-27B"
    with pytest.raises(ValueError, match="already belongs"):
        reg.add("Other/Model", "1B", aliases=["qwen3.6-27b"])


def test_size_precedence_and_tiers_by_precision():
    reg = reg_with_models()
    # registered quant footprint wins; MoE active params come from the registry
    ms = reg.size({"backend": "openai", "model": "qwen/qwen3-next-80b-a3b", "quantization": "q4_k_m"})
    assert (ms.weights_mb, ms.weights_source, ms.active_params) == (48 * GB, "registry", 3_000_000_000)
    assert ms.tiers == ["local"]
    # hosted API, precision unknown: estimated at 8 bits/weight -> 80B is ~75 GiB, still local
    ms = reg.size({"backend": "openai", "model": "qwen/qwen3-next-80b-a3b"})
    assert ms.weights_source == "estimate" and ms.precision is None and ms.tiers == ["local"]
    # bf16 of the same model does not fit 128 GB
    assert reg.size({"backend": "hf", "model": "Qwen/Qwen3-Next-80B-A3B", "dtype": "bfloat16"}).tiers == []
    # a measured run footprint is used when the registry has no entry for that precision
    ms = reg.size({"backend": "hf", "model": "Qwen/Qwen3-14B", "dtype": "bfloat16", "weights_mb": 28_000})
    assert (ms.precision, ms.weights_source, ms.tiers) == ("bf16", "run", ["small", "local"])
    # unregistered model: the run's own numbers, as before the registry existed
    ms = reg.size({"backend": "hf", "model": "local/ckpt", "n_params": 135_000_000, "weights_mb": 260})
    assert ms.model_id is None and ms.tiers == ["tiny", "small", "local"]


def test_registry_file_roundtrip_and_quant_from_file(tmp_path):
    path = tmp_path / "models.json"
    reg = Registry.load(path)
    reg.add("org/m", "4B", aliases=["m"])
    w = tmp_path / "m-Q4_K_M.gguf"
    w.write_bytes(b"x" * 2**20)
    q = reg.add_quant("m", "Q4_K_M", file=w)
    assert q["weights_mb"] == 1.0 and len(q["sha256"]) == 64 and q["file"] == w.name
    reg.save()
    again = Registry.load(path)
    assert again.models["org/m"]["quants"]["Q4_K_M"]["weights_mb"] == 1.0
    with pytest.raises(KeyError):
        again.add_quant("nope", "bf16", weights_mb=1)


def test_strip_reasoning():
    assert strip_reasoning("<think>so 42?</think>\n\n41") == "41"
    assert strip_reasoning("reasoning opened by the template</think>7") == "7"
    assert strip_reasoning("<think>out of budget, 42") == ""
    assert strip_reasoning("plain 3") == "plain 3"


def test_profiles():
    assert get_profile("think").id == "think@1" and get_profile("nothink@1") is PROFILES["nothink"]
    with pytest.raises(KeyError):
        get_profile("think@0")
    with pytest.raises(KeyError):
        get_profile("fast")
    think = PROFILES["think"]
    assert think.backend_options("anthropic") == {"thinking": "adaptive"}
    assert PROFILES["nothink"].backend_options("openai") == {"reasoning": "off"}
    assert PROFILES["nothink"].overrides() == {}
    assert think.overrides() == {"temperature": 0.6, "min_max_tokens": 16384}


def test_min_max_tokens_is_a_floor():
    seen = []

    class Rec:
        supports_logprobs, supports_generate = False, True

        def generate(self, prompt, max_tokens, temperature, n):
            seen.append(max_tokens)
            return ["1"] * n

    it = Item(id="a", kind=Kind.NUMERIC, prompt="q", target="1", max_tokens=32)
    for scoring in ({}, {"min_max_tokens": 1000}, {"max_tokens": 4000, "min_max_tokens": 1000}):
        score_item(it, Task(name="t", file="t.jsonl", kind=Kind.NUMERIC, scoring=scoring), Rec())
    assert seen == [32, 1000, 4000]


def test_openai_reasoning_switch_per_server():
    pytest.importorskip("openai")
    from examroom.backends.openai_compat import OpenAICompatBackend

    local = OpenAICompatBackend("qwen", base_url="http://localhost:8080/v1", reasoning="off")
    assert local.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    router = OpenAICompatBackend("qwen/qwen3", base_url="https://openrouter.ai/api/v1", reasoning="on")
    assert router.extra_body == {"reasoning": {"enabled": True}}
    assert OpenAICompatBackend("x", base_url="https://openrouter.ai/api/v1", reasoning="low").extra_body == \
        {"reasoning": {"effort": "low"}}
    assert OpenAICompatBackend("x", base_url="http://localhost:8080/v1").extra_body is None
    with pytest.raises(ValueError):
        OpenAICompatBackend("x", base_url="http://localhost:8080/v1", reasoning="high")


def test_select_filters_by_tier_and_profile(tmp_path):
    s = make_suite(tmp_path)
    led = Ledger(tmp_path / "l.sqlite")
    record(led, s, "tiny0001", "qwen/qwen3-0.6b", None, {"easy": 40, "hard_en": 12})
    record(led, s, "next0001", "qwen/qwen3-next-80b-a3b", None, {"easy": 40, "hard_en": 12})
    with led.con:  # tag profiles the way `run --profile` records them
        for rid, prof in (("tiny0001", "nothink@1"), ("next0001", "think@1")):
            b = json.loads(led.run(rid)["backend_json"]) | {"profile": prof}
            led.con.execute("UPDATE runs SET backend_json = ? WHERE run_id = ?", (json.dumps(b), rid))
    reg = reg_with_models()
    sel = select(led, s, [], registry=reg, size_key="weights_mb")
    assert [c.run_id for c in sel.candidates] == ["tiny0001", "next0001"]
    assert sel.candidates[0].size_source == "estimate" and sel.candidates[1].tiers == ["local"]
    assert [c.run_id for c in select(led, s, [], registry=reg, tier="small").candidates] == ["tiny0001"]
    sel = select(led, s, [], registry=reg, profile="think")
    assert [c.run_id for c in sel.candidates] == ["next0001"] and sel.filtered_runs == 1
    with pytest.raises(ValueError):
        select(led, s, [], registry=reg, tier="medium")


def test_cli_profiled_run_and_models(tmp_path, capsys):
    s = make_suite(tmp_path)
    db, reg = str(tmp_path / "l.sqlite"), str(tmp_path / "models.json")
    base = ["run", str(s.root), "dummy:uniform", "--allow-unfrozen", "--ledger", db]
    assert main(base + ["--profile", "think", "--max-tokens", "64"]) == 1        # profile fixes the budget
    assert main(base + ["--profile", "think", "--opt", "reasoning=off"]) == 1
    assert main(base + ["--profile", "fast"]) == 1
    assert main(base + ["--profile", "think", "--limit", "2"]) == 1               # ranked runs are complete
    assert main(base + ["--profile", "think", "--limit", "2", "--dry-run"]) == 0
    assert main(base + ["--profile", "think", "--opt", "model_id=org/tiny"]) == 0
    b = json.loads(Ledger(db).runs()[0]["backend_json"])
    assert b["profile"] == "think@1" and b["overrides"]["min_max_tokens"] == 16384 and b["model_id"] == "org/tiny"

    assert main(["models", "--registry", reg, "add", "org/tiny", "--params", "500M", "--alias", "uniform"]) == 0
    assert main(["models", "--registry", reg, "quant", "org/tiny", "bf16", "--weights-mb", "1000"]) == 0
    assert main(["models", "--registry", reg, "quant", "org/none", "bf16", "--weights-mb", "1"]) == 1
    capsys.readouterr()
    assert main(["models", "--registry", reg, "list"]) == 0
    assert "tiers=tiny,small,local" in capsys.readouterr().out
    assert main(["select", str(s.root), "--ledger", db, "--registry", reg, "--tier", "tiny",
                 "--profile", "think"]) == 0
    out = capsys.readouterr().out
    assert "tier=tiny" in out and "profile=think@1" in out and "tiers=tiny,small,local" in out


class _Sib:
    def __init__(self, name, size, sha=None):
        self.rfilename, self.size = name, size
        self.lfs = type("L", (), {"sha256": sha})() if sha else None


def test_hub_weight_files_and_facts(monkeypatch, tmp_path, capsys):
    import examroom.hub as hub

    gguf = [_Sib("README.md", 10), _Sib("m-mmproj.gguf", 5 * 2**20, "aa"), _Sib("m_q4_0.gguf", 300 * 2**20, "bb")]
    assert [s.rfilename for s in hub.weight_files(gguf)] == ["m_q4_0.gguf"]       # projector excluded
    two = gguf + [_Sib("m_q8_0.gguf", 2**20, "cc")]
    with pytest.raises(ValueError, match="--match"):
        hub.weight_files(two)
    assert [s.rfilename for s in hub.weight_files(two, "*q8_0*")] == ["m_q8_0.gguf"]

    shards = [_Sib("model-00001.safetensors", 2**30), _Sib("model-00002.safetensors", 2**30), _Sib("x.json", 1)]
    mistral = shards + [_Sib("consolidated.safetensors", 2 * 2**30)]                # same weights, other format
    assert [s.rfilename for s in hub.weight_files(mistral)] == ["model-00001.safetensors", "model-00002.safetensors"]
    assert [s.rfilename for s in hub.weight_files([mistral[-1]])] == ["consolidated.safetensors"]
    st = type("S", (), {"total": 1_100_000_000, "parameters": {"BF16": 1_099_999_990, "I64": 10}})()
    infos = {"org/m": type("I", (), {"sha": "abc123", "siblings": shards, "safetensors": st})(),
             "org/m-gguf": type("I", (), {"sha": "def456", "siblings": gguf, "safetensors": None})()}
    monkeypatch.setattr(hub, "_info", lambda repo: infos[repo])
    reg = str(tmp_path / "models.json")
    assert main(["models", "--registry", reg, "add-hub", "org/m", "--active", "1B", "--dense"]) == 1
    assert main(["models", "--registry", reg, "add-hub", "org/m", "--dense", "--alias", "m"]) == 0
    assert main(["models", "--registry", reg, "quant-hub", "m", "Q4_0", "org/m-gguf"]) == 0
    e = Registry.load(reg).models["org/m"]
    assert e["n_params"] == e["active_params"] == 1_100_000_000 and e["hub"]["revision"] == "abc123"
    assert e["quants"]["bf16"] == {"weights_mb": 2048.0, "repo": "org/m", "revision": "abc123", "files": 2}
    assert e["quants"]["Q4_0"]["sha256"] == "bb" and e["quants"]["Q4_0"]["weights_mb"] == 300.0
    capsys.readouterr()
    main(["models", "--registry", reg, "list"])
    assert "active=dense" in capsys.readouterr().out
