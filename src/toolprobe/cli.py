"""CLI: toolprobe run / attribute. --model accepts a comma-separated list."""
import argparse
import json
from pathlib import Path

from rich.console import Console

from .attribution import build_attribution
from .models import load_cases, load_tools
from .parser import parse_framework, parse_rescue
from .report import (METRICS, attribution_table, bootstrap_ci, env_header,
                     reliability_table, to_markdown)
from .runner import run_free
from .scorer import score

MODELS = {
    "qwen2.5-0.5b": {"bf16": "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
                     "q4": "mlx-community/Qwen2.5-0.5B-Instruct-4bit"},
    "qwen2.5-1.5b": {"bf16": "mlx-community/Qwen2.5-1.5B-Instruct-bf16",
                     "q4": "mlx-community/Qwen2.5-1.5B-Instruct-4bit"},
    "llama-3.2-3b": {"bf16": "mlx-community/Llama-3.2-3B-Instruct-bf16",
                     "q4": "mlx-community/Llama-3.2-3B-Instruct-4bit"},
}
TOOLS_PATH = Path("cases/tools.yaml")
console = Console()


def _models_arg(s: str) -> list[str]:
    names = s.split(",")
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; choose from {', '.join(MODELS)}")
    return names


def _label(models: list[str]) -> str:
    return "+".join(models)


def _run_one(repo: str, cases, tools) -> dict:
    out = []
    for case in cases:
        raw = run_free(repo, case, tools).text
        call = parse_framework(raw) or parse_rescue(raw, "lenient")
        s = score(call, case.expected, case.arg_rules, tools)
        out.append({"id": case.id, "raw": raw, "score": {k: bool(v) for k, v in s.items()}})
    return {"cases": out}


def _aggregate(model: str, quant: str, result: dict) -> dict:
    row = {"model": model, "quant": quant}
    for m in METRICS:
        vals = [int(c["score"].get(m, False)) for c in result["cases"] if m in c["score"]]
        row[m] = bootstrap_ci(vals) if vals else (0.0, 0.0, 0.0)
    return row


def cmd_run(args) -> int:
    cases, tools = load_cases(args.cases), load_tools(TOOLS_PATH)
    models = _models_arg(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = env_header()
    data = {"env": env, "cases_file": args.cases, "models": {}}
    rows = []
    for model in models:
        data["models"][model] = {"quants": {}}
        for quant in args.quant.split(","):
            repo = MODELS[model][quant]
            result = _run_one(repo, cases, tools)
            result["repo"] = repo
            data["models"][model]["quants"][quant] = result
            rows.append(_aggregate(model, quant, result))
    console.print(reliability_table(rows))
    label = _label(models)
    (out_dir / f"run-{label}.json").write_text(json.dumps(data, indent=2))
    (out_dir / f"run-{label}.md").write_text(to_markdown(rows, {}, env))
    return 0


def _serialize_report(report: dict) -> dict:
    return {len_: {"per_quant": {q: {c.value: n for c, n in cnt.items()}
                                 for q, cnt in leaf["per_quant"].items()},
                   "quant_delta": {c.value: d for c, d in leaf["quant_delta"].items()},
                   "parser_gap_repros": [{"case_id": r.case_id, "raw": r.raw}
                                         for r in leaf["parser_gap_repros"]]}
            for len_, leaf in report.items()}


def cmd_attribute(args) -> int:
    cases, tools = load_cases(args.cases), load_tools(TOOLS_PATH)
    models = _models_arg(args.model)
    env = env_header()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        repos = MODELS[model]
        report = build_attribution(repos["bf16"], repos["q4"], cases, tools)
        console.print(f"[bold]{model}[/bold]")
        console.print(attribution_table(report))
        (out_dir / f"attribution-{model}.json").write_text(
            json.dumps({"env": env, "model": model, "report": _serialize_report(report)},
                       indent=2))
        (out_dir / f"attribution-{model}.md").write_text(to_markdown([], report, env))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="toolprobe")
    sub = p.add_subparsers(dest="cmd", required=True)
    models_help = f"comma-separated; choose from: {', '.join(MODELS)}"
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, help=models_help)
    r.add_argument("--quant", default="q4")
    r.add_argument("--cases", default="cases/default.jsonl")
    r.add_argument("--out", default="reports")
    r.set_defaults(fn=cmd_run)
    a = sub.add_parser("attribute")
    a.add_argument("--model", required=True, help=models_help)
    a.add_argument("--cases", default="cases/default.jsonl")
    a.add_argument("--out", default="reports")
    a.set_defaults(fn=cmd_attribute)
    args = p.parse_args(argv)
    return args.fn(args)
