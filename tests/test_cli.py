import json

from toolprobe import cli
from toolprobe.backend import GenResult


def test_registry_shape():
    for name, quants in cli.MODELS.items():
        assert "q4" in quants and quants["q4"].startswith("mlx-community/")


def test_run_command_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_free",
        lambda repo, case, tools, **kw: GenResult(
            '{"name": "get_weather", "arguments": {"location": "Tokyo"}}', 0.1, 50.0, 100.0))
    rc = cli.main(["run", "--model", "qwen2.5-0.5b", "--quant", "q4",
                   "--cases", "cases/smoke.jsonl", "--out", str(tmp_path)])
    assert rc == 0
    data = json.loads((tmp_path / "run-qwen2.5-0.5b.json").read_text())
    assert data["env"]["mlx_lm"]
    q4 = data["quants"]["q4"]
    assert len(q4["cases"]) == 3
    assert (tmp_path / "run-qwen2.5-0.5b.md").exists()
