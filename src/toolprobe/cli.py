"""CLI: toolprobe run / attribute. --model accepts a comma-separated list."""
import argparse
import json
from pathlib import Path

from rich.console import Console

from .attribution import build_attribution
from .backend import generate as _backend_generate
from .methods import METHODS, get_method
from .models import load_cases, load_tools
from .parser import parse_framework, parse_rescue
from .report import (METRICS, attribution_table, bootstrap_ci, comparison_table,
                     decode_comparison_table, env_header, reliability_table, to_markdown)
from .runner import run_constrained, run_free
from .scorer import score

DECODES = ("free", "constrained")

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


def _methods_arg(s: str) -> list[str]:
    names = s.split(",")
    unknown = [m for m in names if m not in METHODS]
    if unknown:
        raise SystemExit(f"unknown method(s) {unknown}; choose from {', '.join(METHODS)}")
    return names


def _decodes_arg(s: str) -> list[str]:
    names = s.split(",")
    unknown = [d for d in names if d not in DECODES]
    if unknown:
        raise SystemExit(f"unknown decode(s) {unknown}; choose from {', '.join(DECODES)}")
    return names


def _label(models: list[str]) -> str:
    return "+".join(models)


def _make_gen(repo: str):
    """Real PA-Tool candidate generator (MLX-backed). Monkeypatched in tests."""
    def gen(prompt: str, n: int, temp: float, seed: int) -> list[str]:
        return [_backend_generate(repo, prompt, max_tokens=12, temp=temp, seed=seed + i).text
                for i in range(n)]
    return gen


def _run_one(repo: str, cases, tools, method_name: str, decode: str,
             cache_dir, base_seed: int) -> dict:
    method = get_method(method_name, cache_dir=cache_dir, base_seed=base_seed) \
        if method_name == "pa_tool" else get_method(method_name)
    mr = method.prepare(repo, tools, gen=_make_gen(repo))
    # constrained decoding masks the JSON body to the PRESENTED (mr.tools) schema;
    # scoring still canonicalizes back, so method + constrained compose correctly.
    run = run_constrained if decode == "constrained" else run_free
    out = []
    for case in cases:
        raw = run(repo, case, mr.tools).text
        call = mr.canonicalize(parse_framework(raw) or parse_rescue(raw, "lenient"))
        s = score(call, case.expected, case.arg_rules, tools)   # canonical tools + gold
        out.append({"id": case.id, "raw": raw, "score": {k: bool(v) for k, v in s.items()}})
    result = {"cases": out, "repo": repo}
    if method_name == "pa_tool" and mr.meta is not None:
        # full rename map (tool + param, both inverse maps) for reproducibility/audit (spec §7)
        result["adaptation"] = {"name_map": mr.meta["name_map"],
                                "param_maps": mr.meta["param_maps"]}
    return result


def _aggregate(model: str, quant: str, method: str, decode: str, result: dict) -> dict:
    row = {"model": model, "quant": quant, "method": method, "decode": decode}
    for m in METRICS:
        vals = [int(c["score"].get(m, False)) for c in result["cases"] if m in c["score"]]
        row[m] = bootstrap_ci(vals) if vals else (0.0, 0.0, 0.0)
    return row


def cmd_run(args) -> int:
    cases, tools = load_cases(args.cases), load_tools(TOOLS_PATH)
    models = _models_arg(args.model)
    methods = _methods_arg(args.method)
    decodes = _decodes_arg(args.decode)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = env_header()
    env["pa_seed"] = args.pa_seed
    data = {"env": env, "cases_file": args.cases, "methods": methods,
            "decodes": decodes, "models": {}}
    rows = []
    for model in models:
        data["models"][model] = {"quants": {}}
        for quant in args.quant.split(","):
            repo = MODELS[model][quant]
            per_method: dict = {}
            data["models"][model]["quants"][quant] = per_method
            for method in methods:
                for decode in decodes:
                    result = _run_one(repo, cases, tools, method, decode,
                                      args.cache_dir, args.pa_seed)
                    # backward-compatible: single decode keeps quants[q][method] = result;
                    # only nest by decode when more than one is requested.
                    if len(decodes) == 1:
                        per_method[method] = result
                    else:
                        per_method.setdefault(method, {})[decode] = result
                    rows.append(_aggregate(model, quant, method, decode, result))
    console.print(reliability_table(rows))
    if len(methods) > 1:
        console.print(comparison_table(rows))
    if len(decodes) > 1:
        console.print(decode_comparison_table(rows))
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
    r.add_argument("--method", default="baseline", help=f"comma-separated; {', '.join(METHODS)}")
    r.add_argument("--decode", default="free",
                   help=f"comma-separated; {', '.join(DECODES)}")
    r.add_argument("--pa-seed", type=int, default=0, dest="pa_seed",
                   help="PA-Tool base seed (reproducible adaptation)")
    r.add_argument("--cache-dir", default=".cache/pa_tool", dest="cache_dir",
                   help="PA-Tool adaptation cache dir")
    r.set_defaults(fn=cmd_run)
    a = sub.add_parser("attribute")
    a.add_argument("--model", required=True, help=models_help)
    a.add_argument("--cases", default="cases/default.jsonl")
    a.add_argument("--out", default="reports")
    a.set_defaults(fn=cmd_attribute)
    args = p.parse_args(argv)
    return args.fn(args)
