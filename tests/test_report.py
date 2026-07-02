from collections import Counter

from toolprobe.attribution import Cause
from toolprobe.report import (attribution_table, bootstrap_ci, env_header,
                              reliability_table, to_markdown)


def test_bootstrap_ci_seeded_and_sane():
    data = [1] * 15 + [0] * 5
    p1, lo1, hi1 = bootstrap_ci(data, seed=0)
    p2, lo2, hi2 = bootstrap_ci(data, seed=0)
    assert (p1, lo1, hi1) == (p2, lo2, hi2)      # reproducible
    assert p1 == 0.75 and 0.5 <= lo1 <= p1 <= hi1 <= 1.0


def test_bootstrap_ci_degenerate():
    p, lo, hi = bootstrap_ci([1, 1, 1])
    assert p == lo == hi == 1.0


def _fake_attr_report():
    pq = {"bf16": Counter({Cause.PASS: 20, Cause.MODEL_DECISION_FAILURE: 2}),
          "q4": Counter({Cause.PASS: 15, Cause.MODEL_FORMAT_FAILURE: 5,
                         Cause.MODEL_DECISION_FAILURE: 2})}
    delta = {c: pq["q4"][c] - pq["bf16"][c] for c in Cause}
    leaf = {"per_quant": pq, "quant_delta": delta, "parser_gap_repros": []}
    return {"strict": leaf, "lenient": leaf}


def test_tables_and_markdown_render():
    rows = [{"model": "qwen2.5-0.5b", "quant": "q4",
             "parse_ok": (0.9, 0.8, 1.0), "schema_valid": (0.9, 0.8, 1.0),
             "tool_correct": (0.8, 0.6, 0.95), "args_correct": (0.7, 0.5, 0.9)}]
    assert reliability_table(rows) is not None
    assert attribution_table(_fake_attr_report()) is not None
    md = to_markdown(rows, _fake_attr_report(), env_header())
    assert "qwen2.5-0.5b" in md and "model_format_failure" in md
    assert "0.70" in md and "[0.50, 0.90]" in md


def test_env_header_records_versions():
    env = env_header()
    assert "mlx_lm" in env and "macos" in env and "chip" in env
