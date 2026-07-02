from pathlib import Path

from toolprobe.models import Case, load_tools
from toolprobe.templates import FIXED_DATE, case_to_messages, render_with_tokenizer

CASE = Case(id="c1", cat="C1", tools=["get_weather"],
            turns=[{"role": "user", "content": "Weather in Tokyo tomorrow?"}],
            expected={"tool": "get_weather", "args": {"location": "Tokyo"}})

MULTI = Case(id="c6", cat="C6", tools=["read_calendar"],
             turns=[{"role": "user", "content": "Am I free Thursday?"},
                    {"role": "assistant", "tool_call": {"tool": "read_calendar",
                                                        "args": {"date": "2026-03-26"}}},
                    {"role": "tool", "content": "[]"},
                    {"role": "user", "content": "Book dentist at 3pm."}],
             expected=None)


def test_date_injected_as_system_message():
    msgs = case_to_messages(CASE)
    assert msgs[0]["role"] == "system" and FIXED_DATE in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "Weather in Tokyo tomorrow?"}


def test_multiturn_assistant_tool_call_becomes_tool_calls_message():
    msgs = case_to_messages(MULTI)
    a = msgs[2]
    assert a["role"] == "assistant"
    assert a["tool_calls"][0]["function"]["name"] == "read_calendar"
    assert msgs[3]["role"] == "tool"


class FakeTokenizer:
    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True,
                            tokenize=False):
        names = ",".join(t["function"]["name"] for t in (tools or []))
        return f"TOOLS[{names}]|" + "|".join(m["role"] for m in messages)


def test_render_passes_native_tools():
    tools = load_tools(Path(__file__).parent.parent / "cases" / "tools.yaml")
    out = render_with_tokenizer(FakeTokenizer(), CASE, tools)
    assert "TOOLS[get_weather]" in out and out.startswith("TOOLS")
