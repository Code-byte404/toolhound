from pathlib import Path

from toolprobe.models import ToolCall, load_cases, load_tools

CASES = Path(__file__).parent.parent / "cases"


def test_load_default_cases():
    dev = load_cases(CASES / "dev.jsonl")
    test = load_cases(CASES / "test.jsonl")
    default = load_cases(CASES / "default.jsonl")
    # default.jsonl is exactly the dev+test union
    assert len(default) == len(dev) + len(test)
    assert 280 <= len(default) <= 330
    # provenance survives round-trip through jsonl
    assert {c.split for c in default} == {"dev", "test"}
    assert all(c.family for c in default)


def test_load_tools_openai_shape():
    tools = load_tools(CASES / "tools.yaml")
    assert len(tools) == 32
    gw = tools["get_weather"]
    assert gw["type"] == "function"
    assert gw["function"]["name"] == "get_weather"
    assert gw["function"]["parameters"]["required"] == ["location"]


def test_palette_has_confusable_clusters():
    tools = load_tools(CASES / "tools.yaml")
    for name in ["send_email", "send_sms", "send_slack_message",
                 "create_event", "create_reminder", "set_alarm",
                 "play_music", "play_video", "set_volume"]:
        assert name in tools, f"missing clustered tool {name}"


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
