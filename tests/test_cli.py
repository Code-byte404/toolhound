import json

import pytest

from toolprobe import cli
from toolprobe.backend import GenResult


def test_registry_shape():
    assert len(cli.MODELS) >= 3
    for name, quants in cli.MODELS.items():
        assert "bf16" in quants and "q4" in quants
        assert quants["q4"].startswith("mlx-community/")


def _fake_run(repo, case, tools, **kw):
    return GenResult('{"name": "get_weather", "arguments": {"location": "Tokyo"}}',
                     0.1, 50.0, 100.0)


def test_run_default_baseline_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_free", _fake_run)
    rc = cli.main(["run", "--model", "qwen3.5-2b", "--quant", "q4",
                   "--cases", "cases/smoke.jsonl", "--out", str(tmp_path)])
    assert rc == 0
    data = json.loads((tmp_path / "run-qwen3.5-2b.json").read_text())
    q4 = data["models"]["qwen3.5-2b"]["quants"]["q4"]
    assert len(q4["baseline"]["cases"]) == 3           # method dimension
    assert (tmp_path / "run-qwen3.5-2b.md").exists()


def test_run_pa_tool_method(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_free", _fake_run)
    # fake candidate generator so no MLX: everything renames to itself (identity-ish)
    monkeypatch.setattr(cli, "_make_gen",
                        lambda repo: (lambda prompt, n, temp, seed: ["get_weather"] * n))
    rc = cli.main(["run", "--model", "qwen3.5-2b", "--quant", "q4",
                   "--method", "baseline,pa_tool", "--cases", "cases/smoke.jsonl",
                   "--out", str(tmp_path), "--cache-dir", str(tmp_path / "cache")])
    assert rc == 0
    data = json.loads((tmp_path / "run-qwen3.5-2b.json").read_text())
    q4 = data["models"]["qwen3.5-2b"]["quants"]["q4"]
    assert set(q4) == {"baseline", "pa_tool"}
    assert "adaptation" in q4["pa_tool"]                 # rename map recorded
    assert "name_map" in q4["pa_tool"]["adaptation"]     # tool renames
    assert "param_maps" in q4["pa_tool"]["adaptation"]   # param renames (spec §7)
    md = (tmp_path / "run-qwen3.5-2b.md").read_text()
    assert "pa_tool" in md


def _fake_constrained(repo, case, tools, **kw):
    # a well-formed, schema-valid call (as constrained decoding would guarantee)
    return GenResult('<tool_call>{"name": "get_weather", "arguments": '
                     '{"location": "Tokyo"}}</tool_call>', 0.1, 50.0, 100.0)


def test_constrained_rejected_for_current_lineup(tmp_path, monkeypatch):
    # the current registry is free-decoding only; requesting constrained must fail fast
    # (before loading any model) with a clear message, not silently mis-decode.
    monkeypatch.setattr(cli, "run_free", _fake_run)
    with pytest.raises(SystemExit):
        cli.main(["run", "--model", "qwen3.5-2b", "--decode", "constrained",
                  "--cases", "cases/smoke.jsonl", "--out", str(tmp_path)])


def test_run_decode_axis_nests_and_compares(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_free", _fake_run)
    monkeypatch.setattr(cli, "run_constrained", _fake_constrained)
    # no current model has a constrained wire family; bypass the fail-fast guard to
    # exercise the decode-nesting/report logic itself.
    monkeypatch.setattr("toolprobe.grammar.detect_family", lambda repo: "qwen")
    rc = cli.main(["run", "--model", "qwen3.5-2b", "--quant", "q4",
                   "--decode", "free,constrained", "--cases", "cases/smoke.jsonl",
                   "--out", str(tmp_path)])
    assert rc == 0
    data = json.loads((tmp_path / "run-qwen3.5-2b.json").read_text())
    assert data["decodes"] == ["free", "constrained"]
    q4 = data["models"]["qwen3.5-2b"]["quants"]["q4"]
    # >1 decode => nested by decode under the method
    assert set(q4["baseline"]) == {"free", "constrained"}
    assert len(q4["baseline"]["constrained"]["cases"]) == 3
    md = (tmp_path / "run-qwen3.5-2b.md").read_text()
    assert "Decode comparison" in md


def test_unknown_decode_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_free", _fake_run)
    with pytest.raises(SystemExit):
        cli.main(["run", "--model", "qwen3.5-2b", "--decode", "bogus",
                  "--cases", "cases/smoke.jsonl", "--out", str(tmp_path)])


def test_unknown_method_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_free", _fake_run)
    with pytest.raises(SystemExit):
        cli.main(["run", "--model", "qwen3.5-2b", "--method", "bogus",
                  "--cases", "cases/smoke.jsonl", "--out", str(tmp_path)])


def test_unknown_model_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_free", _fake_run)
    with pytest.raises(SystemExit):
        cli.main(["run", "--model", "gpt-4", "--cases", "cases/smoke.jsonl",
                  "--out", str(tmp_path)])
