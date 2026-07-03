from toolprobe.report import (ci_overlap, comparison_rows, comparison_table,
                              reliability_table, to_markdown, env_header)


def _rows():
    base = {"model": "m", "quant": "q4", "method": "baseline",
            "parse_ok": (0.9, 0.85, 0.95), "schema_valid": (0.9, 0.85, 0.95),
            "tool_correct": (0.60, 0.50, 0.70), "args_correct": (0.50, 0.40, 0.60)}
    pat = {"model": "m", "quant": "q4", "method": "pa_tool",
           "parse_ok": (0.9, 0.85, 0.95), "schema_valid": (0.9, 0.85, 0.95),
           "tool_correct": (0.85, 0.78, 0.92), "args_correct": (0.55, 0.45, 0.65)}
    return [base, pat]


def test_ci_overlap():
    assert ci_overlap((0.6, 0.5, 0.7), (0.65, 0.55, 0.75)) is True
    assert ci_overlap((0.6, 0.5, 0.7), (0.85, 0.78, 0.92)) is False


def test_comparison_rows_flags_credible_gain():
    cr = {(r["metric"]): r for r in comparison_rows(_rows())}
    # tool_correct: CIs disjoint and pa_tool higher => credible
    assert cr["tool_correct"]["credible"] is True
    assert abs(cr["tool_correct"]["delta"] - 0.25) < 1e-9
    # args_correct: CIs overlap => not credible
    assert cr["args_correct"]["credible"] is False


def test_tables_and_markdown_include_method():
    rows = _rows()
    assert reliability_table(rows) is not None
    assert comparison_table(rows) is not None
    md = to_markdown(rows, {}, env_header())
    assert "pa_tool" in md and "Method comparison" in md


def test_reliability_backward_compatible_without_method_key():
    # v1 rows lacking a "method" key must still render
    row = {"model": "m", "quant": "q4",
           "parse_ok": (0.9, 0.8, 1.0), "schema_valid": (0.9, 0.8, 1.0),
           "tool_correct": (0.8, 0.6, 0.95), "args_correct": (0.7, 0.5, 0.9)}
    assert reliability_table([row]) is not None
    assert "## Reliability" in to_markdown([row], {}, env_header())
