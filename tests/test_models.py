from pathlib import Path

from toolprobe.models import ToolCall, load_cases, load_tools

CASES = Path(__file__).parent.parent / "cases"


def test_load_default_cases():
    cases = load_cases(CASES / "default.jsonl")
    assert len(cases) == 21
    byid = {c.id: c for c in cases}
    assert byid["c1_weather_01"].expected.tool == "get_weather"
    assert byid["c5_abstain_01"].expected is None
    assert byid["c6_multi_01"].turns[1].tool_call["tool"] == "read_calendar"
    assert byid["c7_scale_15_01"].n_tools == 15


def test_load_tools_openai_shape():
    tools = load_tools(CASES / "tools.yaml")
    assert len(tools) == 15
    gw = tools["get_weather"]
    assert gw["type"] == "function"
    assert gw["function"]["name"] == "get_weather"
    assert gw["function"]["parameters"]["required"] == ["location"]


def test_every_case_tool_exists_in_palette():
    tools = load_tools(CASES / "tools.yaml")
    for c in load_cases(CASES / "default.jsonl"):
        for t in c.tools:
            assert t in tools, f"{c.id} references unknown tool {t}"


def test_toolcall_model():
    tc = ToolCall(tool="get_time", args={"timezone": "Europe/London"})
    assert tc.args["timezone"] == "Europe/London"


def test_case_provenance_fields_optional_and_loaded():
    from toolprobe.models import Case
    # defaults when absent (backward-compatible with existing cases)
    c = Case(id="x", cat="C1", tools=["get_time"],
             turns=[{"role": "user", "content": "hi"}])
    assert c.split is None and c.family is None
    # populated when present
    c2 = Case(id="y", cat="C1", tools=["get_time"], split="dev", family="c1_time",
              turns=[{"role": "user", "content": "hi"}])
    assert c2.split == "dev" and c2.family == "c1_time"
