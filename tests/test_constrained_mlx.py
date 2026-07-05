"""End-to-end tests for constrained (grammar-guided) decoding. Apple Silicon only.
temp=0 makes these deterministic, so behavioral assertions are stable per model."""
import pytest

from toolprobe.models import Case, load_tools
from toolprobe.parser import parse_framework
from toolprobe.runner import run_constrained
from toolprobe.scorer import _validate_schema

TOOLS = load_tools("cases/tools.yaml")
QWEN = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
LLAMA = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def _case(tools, content, expected):
    return Case(id="t", cat="C1", tools=tools,
                turns=[{"role": "user", "content": content}], expected=expected)


@pytest.mark.mlx
def test_constrained_call_is_parseable_and_schema_valid():
    case = _case(["get_weather"], "What's the weather in Paris? Use a tool.",
                 {"tool": "get_weather", "args": {"location": "Paris"}})
    out = run_constrained(QWEN, case, TOOLS).text
    assert "<tool_call>" in out and "</tool_call>" in out       # native wrapper preserved
    call = parse_framework(out)
    assert call is not None and call.tool == "get_weather"
    assert _validate_schema(call, TOOLS)                        # syntax layer solved


@pytest.mark.mlx
def test_constrained_abstention_is_reachable():
    """Trigger-gating must leave a 'no call' exit: a chit-chat prompt should not be
    forced into a tool call even though a tool is offered (design doc §4.4)."""
    case = _case(["get_weather"], "Hi! How are you feeling today?", None)
    out = run_constrained(QWEN, case, TOOLS).text
    assert "<tool_call>" not in out          # model stayed free -> replied in NL
    assert parse_framework(out) is None      # scored as abstention, not a call


@pytest.mark.mlx
def test_constrained_is_deterministic():
    case = _case(["get_weather"], "Weather in Tokyo? Use a tool.",
                 {"tool": "get_weather", "args": {"location": "Tokyo"}})
    a = run_constrained(QWEN, case, TOOLS).text
    b = run_constrained(QWEN, case, TOOLS).text
    assert a == b


@pytest.mark.mlx
def test_constrained_llama_bare_json_is_schema_enforced():
    """Llama-3.2 usually emits its call as bare leading JSON (no <|python_tag|>).
    The bare-JSON trigger must still constrain it: create_event's `attendees` is an
    array the free model tends to fill with the string "[]" (schema-invalid)."""
    case = _case(["create_event"],
                 "Schedule 'budget review' tomorrow 10-11am on my calendar. Use a tool.",
                 {"tool": "create_event", "args": {"title": "budget review"}})
    out = run_constrained(LLAMA, case, TOOLS).text
    call = parse_framework(out)
    assert call is not None and call.tool == "create_event"
    assert _validate_schema(call, TOOLS)     # constrained => schema-valid despite bare JSON


@pytest.mark.mlx
def test_constrained_llama_emits_python_tag():
    """Guards the special-token risk: the Llama grammar must let the model emit its
    single-token opener <|python_tag|> and still produce a schema-valid call."""
    case = _case(["get_weather"], "What's the weather in Berlin? Use a tool.",
                 {"tool": "get_weather", "args": {"location": "Berlin"}})
    out = run_constrained(LLAMA, case, TOOLS).text
    assert "<|python_tag|>" in out
    call = parse_framework(out)
    assert call is not None and call.tool == "get_weather" and _validate_schema(call, TOOLS)
