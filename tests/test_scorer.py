from pathlib import Path

from toolprobe.models import ExpectedCall, ToolCall, load_tools
from toolprobe.scorer import passed, score

TOOLS = load_tools(Path(__file__).parent.parent / "cases" / "tools.yaml")


def exp(tool, **args):
    return ExpectedCall(tool=tool, args=args)


def test_exact_pass():
    s = score(ToolCall(tool="get_weather", args={"location": "Tokyo"}),
              exp("get_weather", location="Tokyo"), {}, TOOLS)
    assert s["parse_ok"] and s["schema_valid"] and s["tool_correct"] and s["args_correct"]
    assert passed(s, exp("get_weather", location="Tokyo"))


def test_equiv_rule():
    rules = {"location": {"equiv": ["Tokyo", "Tokyo, Japan", "tokyo"]}}
    s = score(ToolCall(tool="get_weather", args={"location": "tokyo"}),
              exp("get_weather", location="Tokyo"), rules, TOOLS)
    assert s["args_correct"]


def test_set_rule_ignores_order():
    rules = {"attendees": {"match": "set"}}
    s = score(ToolCall(tool="create_event",
                       args={"title": "design review", "start": "2026-03-20T14:00:00",
                             "attendees": ["lee@corp.com", "sam@corp.com"]}),
              exp("create_event", title="design review", start="2026-03-20T14:00:00",
                  attendees=["sam@corp.com", "lee@corp.com"]),
              rules, TOOLS)
    assert s["args_correct"]


def test_iso8601_minute_normalization():
    rules = {"datetime": {"normalize": "iso8601_minute"}}
    s = score(ToolCall(tool="create_reminder",
                       args={"text": "call mom", "datetime": "2026-03-21T18:00:00.000+00:00"}),
              exp("create_reminder", text="call mom", datetime="2026-03-21T18:00"),
              rules, TOOLS)
    assert s["args_correct"]


def test_semantic_rule():
    rules = {"query": {"match": "semantic"}}
    s = score(ToolCall(tool="search_web", args={"query": "SpaceX latest launch date"}),
              exp("search_web", query="latest SpaceX launch date"), rules, TOOLS)
    assert s["args_correct"]


def test_wrong_tool():
    s = score(ToolCall(tool="create_event", args={"title": "x", "start": "2026-03-21T18:00"}),
              exp("create_reminder", text="call mom", datetime="2026-03-21T18:00"), {}, TOOLS)
    assert s["tool_correct"] is False
    assert not passed(s, exp("create_reminder", text="call mom", datetime="2026-03-21T18:00"))


def test_schema_invalid_missing_required_and_bad_enum():
    s = score(ToolCall(tool="get_weather", args={"unit": "kelvin"}),
              exp("get_weather", location="Paris"), {}, TOOLS)
    assert s["schema_valid"] is False


def test_abstention_correct_and_false_trigger():
    s = score(None, None, {}, TOOLS)
    assert s["abstention_correct"] is True and s["false_trigger"] is False
    assert "parse_ok" not in s  # layered metrics don't apply to abstention cases
    assert passed(s, None)
    s2 = score(ToolCall(tool="search_web", args={"query": "hi"}), None, {}, TOOLS)
    assert s2["abstention_correct"] is False and s2["false_trigger"] is True


def test_no_call_when_expected_fails_all_layers():
    s = score(None, exp("get_weather", location="Tokyo"), {}, TOOLS)
    # layered: an unparseable output fails every downstream metric too
    assert s == {"parse_ok": False, "schema_valid": False,
                 "tool_correct": False, "args_correct": False}
    assert not passed(s, exp("get_weather", location="Tokyo"))
