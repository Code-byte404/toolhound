import json

from toolprobe import cli
from toolprobe.backend import GenResult


def test_registry_shape():
    assert len(cli.MODELS) >= 3  # §9: at least 3 models
    for name, quants in cli.MODELS.items():
        assert "bf16" in quants and "q4" in quants
        assert quants["q4"].startswith("mlx-community/")


def _fake_gen(repo, case, tools, **kw):
    return GenResult('{"name": "get_weather", "arguments": {"location": "Tokyo"}}',
                     0.1, 50.0, 100.0)


def test_run_command_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_free", _fake_gen)
    rc = cli.main(["run", "--model", "qwen2.5-0.5b", "--quant", "q4",
                   "--cases", "cases/smoke.jsonl", "--out", str(tmp_path)])
    assert rc == 0
    data = json.loads((tmp_path / "run-qwen2.5-0.5b.json").read_text())
    assert data["env"]["mlx_lm"]
    q4 = data["models"]["qwen2.5-0.5b"]["quants"]["q4"]
    assert len(q4["cases"]) == 3
    assert (tmp_path / "run-qwen2.5-0.5b.md").exists()


def test_run_multi_model_combined_report(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_free", _fake_gen)
    rc = cli.main(["run", "--model", "qwen2.5-0.5b,qwen2.5-1.5b", "--quant", "bf16,q4",
                   "--cases", "cases/smoke.jsonl", "--out", str(tmp_path)])
    assert rc == 0
    data = json.loads((tmp_path / "run-qwen2.5-0.5b+qwen2.5-1.5b.json").read_text())
    assert set(data["models"]) == {"qwen2.5-0.5b", "qwen2.5-1.5b"}
    assert set(data["models"]["qwen2.5-1.5b"]["quants"]) == {"bf16", "q4"}
    md = (tmp_path / "run-qwen2.5-0.5b+qwen2.5-1.5b.md").read_text()
    assert "qwen2.5-0.5b" in md and "qwen2.5-1.5b" in md


def test_unknown_model_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_free", _fake_gen)
    import pytest
    with pytest.raises(SystemExit):
        cli.main(["run", "--model", "gpt-4", "--cases", "cases/smoke.jsonl",
                  "--out", str(tmp_path)])
