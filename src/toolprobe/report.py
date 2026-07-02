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
            "mlx": v("mlx"), "mlx_lm": v("mlx-lm"), "toolprobe": v("mlx-toolprobe")}


def _ci(cell: tuple[float, float, float]) -> str:
    p, lo, hi = cell
    return f"{p:.2f} [{lo:.2f}, {hi:.2f}]"


METRICS = ("parse_ok", "schema_valid", "tool_correct", "args_correct")


def reliability_table(rows: list[dict]) -> Table:
    t = Table(title="Tool-calling reliability (point [95% bootstrap CI])")
    for col in ("model", "quant", *METRICS):
        t.add_column(col)
    for r in rows:
        t.add_row(r["model"], r["quant"], *[_ci(r[m]) for m in METRICS if m in r])
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
    lines = ["# toolprobe report", "",
             "## Environment",
             *[f"- {k}: {v}" for k, v in env.items()], "",
             "## Reliability", "",
             "| model | quant | " + " | ".join(METRICS) + " |",
             "|" + "---|" * (2 + len(METRICS))]
    for r in rows:
        lines.append("| " + " | ".join([r["model"], r["quant"],
                                        *[_ci(r[m]) for m in METRICS if m in r]]) + " |")
    lines += ["", "## Attribution", "",
              "| leniency | quant | " + " | ".join(c.value for c in Cause) + " |",
              "|" + "---|" * (2 + len(Cause))]
    for leniency, leaf in report.items():
        for quant, counter in leaf["per_quant"].items():
            lines.append("| " + " | ".join([leniency, quant,
                                            *[str(counter.get(c, 0)) for c in Cause]]) + " |")
    return "\n".join(lines) + "\n"
