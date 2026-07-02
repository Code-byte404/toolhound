"""CLI: toolprobe run / attribute."""
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
}
TOOLS_PATH = Path("cases/tools.yaml")
console = Console()


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
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {"env": env_header(), "model": args.model, "cases_file": args.cases, "quants": {}}
    rows = []
    for quant in args.quant.split(","):
        repo = MODELS[args.model][quant]
        result = _run_one(repo, cases, tools)
        result["repo"] = repo
        data["quants"][quant] = result
        rows.append(_aggregate(args.model, quant, result))
    console.print(reliability_table(rows))
    (out_dir / f"run-{args.model}.json").write_text(json.dumps(data, indent=2))
    (out_dir / f"run-{args.model}.md").write_text(to_markdown(rows, {}, data["env"]))
    return 0


def cmd_attribute(args) -> int:
    cases, tools = load_cases(args.cases), load_tools(TOOLS_PATH)
    repos = MODELS[args.model]
    report = build_attribution(repos["bf16"], repos["q4"], cases, tools)
    console.print(attribution_table(report))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    serial = {len_: {"per_quant": {q: {c.value: n for c, n in cnt.items()}
                                   for q, cnt in leaf["per_quant"].items()},
                     "quant_delta": {c.value: d for c, d in leaf["quant_delta"].items()},
                     "parser_gap_repros": [{"case_id": r.case_id, "raw": r.raw}
                                           for r in leaf["parser_gap_repros"]]}
              for len_, leaf in report.items()}
    env = env_header()
    (out_dir / f"attribution-{args.model}.json").write_text(
        json.dumps({"env": env, "report": serial}, indent=2))
    (out_dir / f"attribution-{args.model}.md").write_text(to_markdown([], report, env))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="toolprobe")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=MODELS)
    r.add_argument("--quant", default="q4")
    r.add_argument("--cases", default="cases/default.jsonl")
    r.add_argument("--out", default="reports")
    r.set_defaults(fn=cmd_run)
    a = sub.add_parser("attribute")
    a.add_argument("--model", required=True, choices=MODELS)
    a.add_argument("--cases", default="cases/default.jsonl")
    a.add_argument("--out", default="reports")
    a.set_defaults(fn=cmd_attribute)
    args = p.parse_args(argv)
    return args.fn(args)
