"""Reporting: point estimates + bootstrap 95% CIs (variance comes from the
case set, NOT seeds -- temp=0 is deterministic, design doc §4.3), rich tables,
markdown export with a reproducibility header."""
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from rich.table import Table

from .attribution import Cause


def bootstrap_ci(per_case: list[int], n: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    arr = np.asarray(per_case, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    means = arr[idx].mean(axis=1)
    return (float(arr.mean()),
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def env_header() -> dict:
    def v(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return "not installed"
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        chip = platform.machine()
    return {"chip": chip, "macos": platform.mac_ver()[0],
            "mlx": v("mlx"), "mlx_lm": v("mlx-lm"), "mlx_vlm": v("mlx-vlm"),
            "outlines_core": v("outlines-core"), "toolprobe": v("mlx-toolprobe")}


def _ci(cell: tuple[float, float, float]) -> str:
    p, lo, hi = cell
    return f"{p:.2f} [{lo:.2f}, {hi:.2f}]"


METRICS = ("parse_ok", "schema_valid", "tool_correct", "args_correct")


def reliability_table(rows: list[dict]) -> Table:
    t = Table(title="Tool-calling reliability (point [95% bootstrap CI])")
    for col in ("model", "quant", "method", "decode", *METRICS):
        t.add_column(col)
    for r in rows:
        t.add_row(r["model"], r["quant"], r.get("method", "baseline"), r.get("decode", "free"),
                  *[_ci(r[m]) for m in METRICS if m in r])
    return t


def ci_overlap(a: tuple, b: tuple) -> bool:
    _, alo, ahi = a
    _, blo, bhi = b
    return not (ahi < blo or bhi < alo)


def comparison_rows(rows: list[dict]) -> list[dict]:
    """Method vs baseline, compared WITHIN the same (model, quant, decode) so a
    method's delta isn't confounded by the decode mode."""
    by_key: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        key = (r["model"], r["quant"], r.get("decode", "free"))
        by_key.setdefault(key, {})[r.get("method", "baseline")] = r
    out: list[dict] = []
    for (model, quant, decode), by_method in by_key.items():
        base = by_method.get("baseline")
        if base is None:
            continue
        for method, r in by_method.items():
            if method == "baseline":
                continue
            for m in METRICS:
                if m in r and m in base:
                    bp, mp = base[m], r[m]
                    out.append({"model": model, "quant": quant, "decode": decode,
                                "method": method, "metric": m,
                                "baseline": bp[0], "method_point": mp[0],
                                "delta": mp[0] - bp[0],
                                "credible": (not ci_overlap(bp, mp)) and mp[0] > bp[0]})
    return out


def comparison_table(rows: list[dict]) -> Table:
    t = Table(title="Method vs baseline (delta; credible = CIs disjoint & higher)")
    for col in ("model", "quant", "method", "metric", "baseline", "method_val", "delta", "credible"):
        t.add_column(col)
    for c in comparison_rows(rows):
        t.add_row(c["model"], c["quant"], c["method"], c["metric"],
                  f"{c['baseline']:.2f}", f"{c['method_point']:.2f}",
                  f"{c['delta']:+.2f}", "yes" if c["credible"] else "no")
    return t


def decode_comparison_rows(rows: list[dict]) -> list[dict]:
    """Constrained vs free, compared WITHIN the same (model, quant, method). free
    is the reference. This is the constrained-decoding deliverable: expect
    parse_ok/schema_valid to jump (syntax layer) while the decision metrics move little."""
    by_key: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        key = (r["model"], r["quant"], r.get("method", "baseline"))
        by_key.setdefault(key, {})[r.get("decode", "free")] = r
    out: list[dict] = []
    for (model, quant, method), by_decode in by_key.items():
        base = by_decode.get("free")
        if base is None:
            continue
        for decode, r in by_decode.items():
            if decode == "free":
                continue
            for m in METRICS:
                if m in r and m in base:
                    bp, mp = base[m], r[m]
                    out.append({"model": model, "quant": quant, "method": method,
                                "decode": decode, "metric": m,
                                "free": bp[0], "decode_point": mp[0],
                                "delta": mp[0] - bp[0],
                                "credible": (not ci_overlap(bp, mp)) and mp[0] > bp[0]})
    return out


def decode_comparison_table(rows: list[dict]) -> Table:
    t = Table(title="Constrained vs free (delta; credible = CIs disjoint & higher)")
    for col in ("model", "quant", "method", "metric", "free", "constrained", "delta", "credible"):
        t.add_column(col)
    for c in decode_comparison_rows(rows):
        t.add_row(c["model"], c["quant"], c["method"], c["metric"],
                  f"{c['free']:.2f}", f"{c['decode_point']:.2f}",
                  f"{c['delta']:+.2f}", "yes" if c["credible"] else "no")
    return t


def attribution_table(report: dict) -> Table:
    t = Table(title="Failure attribution (counts per cause)")
    t.add_column("leniency")
    t.add_column("quant")
    for c in Cause:
        t.add_column(c.value)
    for leniency, leaf in report.items():
        for quant, counter in leaf["per_quant"].items():
            t.add_row(leniency, quant, *[str(counter.get(c, 0)) for c in Cause])
    return t


def to_markdown(rows: list[dict], report: dict, env: dict) -> str:
    """Render a report. Only sections with data are emitted, so `run`
    (rows only) and `attribute` (report only) each produce a clean file."""
    lines = ["# toolprobe report", "",
             "## Environment",
             *[f"- {k}: {v}" for k, v in env.items()]]
    if rows:
        lines += ["", "## Reliability", "",
                  "| model | quant | method | decode | " + " | ".join(METRICS) + " |",
                  "|" + "---|" * (4 + len(METRICS))]
        for r in rows:
            lines.append("| " + " | ".join(
                [r["model"], r["quant"], r.get("method", "baseline"), r.get("decode", "free"),
                 *[_ci(r[m]) for m in METRICS if m in r]]) + " |")
        comp = comparison_rows(rows)
        if comp:
            lines += ["", "## Method comparison", "",
                      "| model | quant | method | metric | baseline | method_val | delta | credible |",
                      "|---|---|---|---|---|---|---|---|"]
            for c in comp:
                lines.append("| " + " | ".join([
                    c["model"], c["quant"], c["method"], c["metric"],
                    f"{c['baseline']:.2f}", f"{c['method_point']:.2f}",
                    f"{c['delta']:+.2f}", "yes" if c["credible"] else "no"]) + " |")
        dcomp = decode_comparison_rows(rows)
        if dcomp:
            lines += ["", "## Decode comparison (constrained vs free)", "",
                      "| model | quant | method | metric | free | constrained | delta | credible |",
                      "|---|---|---|---|---|---|---|---|"]
            for c in dcomp:
                lines.append("| " + " | ".join([
                    c["model"], c["quant"], c["method"], c["metric"],
                    f"{c['free']:.2f}", f"{c['decode_point']:.2f}",
                    f"{c['delta']:+.2f}", "yes" if c["credible"] else "no"]) + " |")
    if report:
        lines += ["", "## Attribution", "",
                  "| leniency | quant | " + " | ".join(c.value for c in Cause) + " |",
                  "|" + "---|" * (2 + len(Cause))]
        for leniency, leaf in report.items():
            for quant, counter in leaf["per_quant"].items():
                lines.append("| " + " | ".join([leniency, quant,
                                                *[str(counter.get(c, 0)) for c in Cause]]) + " |")
    return "\n".join(lines) + "\n"
