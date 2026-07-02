"""One synthetic fixture per decision-tree branch (design doc 附三). No MLX."""
from pathlib import Path

from toolprobe import attribution
from toolprobe.attribution import Cause, attribute_case
from toolprobe.backend import GenResult
from toolprobe.models import Case, load_tools

FIX = Path(__file__).parent / "fixtures"
TOOLS = load_tools(Path(__file__).parent.parent / "cases" / "tools.yaml")

CASE = Case(id="c1", cat="C1", tools=["get_weather"],
            turns=[{"role": "user", "content": "Weather in Tokyo?"}],
            expected={"tool": "get_weather", "args": {"location": "Tokyo"}},
            arg_rules={"location": {"equiv": ["Tokyo", "tokyo"]}})


def fake_run(raw):
    return lambda repo, case, tools, **kw: GenResult(raw, 0.0, 0.0, 0.0)


def patch(monkeypatch, raw, sanity=True):
    monkeypatch.setattr(attribution, "run_free", fake_run(raw))
    monkeypatch.setattr(attribution, "template_sanity", lambda repo: sanity)


def test_branch_a_pass(monkeypatch):
    patch(monkeypatch, (FIX / "clean_toolcall.txt").read_text())
    assert attribute_case("m", CASE, TOOLS).cause == Cause.PASS


def test_branch_a_decision_failure_wrong_tool(monkeypatch):
    patch(monkeypatch, '{"name": "search_web", "arguments": {"query": "tokyo weather"}}')
    assert attribute_case("m", CASE, TOOLS).cause == Cause.MODEL_DECISION_FAILURE


def test_branch_b_template_bug(monkeypatch):
    patch(monkeypatch, (FIX / "garbage.txt").read_text(), sanity=False)
    assert attribute_case("m", CASE, TOOLS).cause == Cause.FRAMEWORK_TEMPLATE_BUG


def test_branch_c_parser_gap(monkeypatch):
    dirty = 'Sure! ```json\n{"name": "get_weather", "arguments": {"location": "Tokyo"}}\n```'
    patch(monkeypatch, dirty)
    rec = attribute_case("m", CASE, TOOLS)
    assert rec.cause == Cause.FRAMEWORK_PARSER_GAP
    assert "get_weather" in rec.raw  # minimal repro retained for upstream report


def test_branch_c_rescued_but_wrong_is_decision(monkeypatch):
    patch(monkeypatch, '```json\n{"name": "get_weather", "arguments": {"location": "Osaka"}}\n```')
    assert attribute_case("m", CASE, TOOLS).cause == Cause.MODEL_DECISION_FAILURE


def test_branch_d_format_failure(monkeypatch):
    patch(monkeypatch, (FIX / "garbage.txt").read_text(), sanity=True)
    assert attribute_case("m", CASE, TOOLS).cause == Cause.MODEL_FORMAT_FAILURE


def test_abstention_pass_and_false_trigger(monkeypatch):
    abstain = Case(id="c5", cat="C5", tools=["get_weather"],
                   turns=[{"role": "user", "content": "Thanks!"}], expected=None)
    patch(monkeypatch, "You're welcome!")
    assert attribute_case("m", abstain, TOOLS).cause == Cause.PASS
    patch(monkeypatch, '{"name": "get_weather", "arguments": {"location": "x"}}')
    assert attribute_case("m", abstain, TOOLS).cause == Cause.MODEL_DECISION_FAILURE


def test_build_attribution_counters_and_delta(monkeypatch):
    outputs = {"bf16": (FIX / "clean_toolcall.txt").read_text(),
               "q4": (FIX / "garbage.txt").read_text()}
    monkeypatch.setattr(attribution, "run_free",
                        lambda repo, case, tools, **kw: GenResult(outputs[repo], 0, 0, 0))
    monkeypatch.setattr(attribution, "template_sanity", lambda repo: True)
    monkeypatch.setattr(attribution, "assert_same_template", lambda a, b: None)
    rep = attribution.build_attribution("bf16", "q4", [CASE], TOOLS)
    len_rep = rep["lenient"]
    assert len_rep["per_quant"]["bf16"][Cause.PASS] == 1
    assert len_rep["per_quant"]["q4"][Cause.MODEL_FORMAT_FAILURE] == 1
    assert len_rep["quant_delta"][Cause.MODEL_FORMAT_FAILURE] == 1
    assert "strict" in rep  # dual-leniency robustness always produced
