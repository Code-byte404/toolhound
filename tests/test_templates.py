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


class TemplateTokenizer:
    """Fake tokenizer whose template renders a fixed format-instruction example."""

    def __init__(self, example):
        self.example = example

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True,
                            tokenize=False):
        name = tools[0]["function"]["name"] if tools else ""
        return (f"system: use {name}\n"
                f"<tool_call>\n{self.example}\n</tool_call>\nuser: hi\n")

    def encode(self, text):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode()


def test_template_sanity_ok_for_wellformed_example(monkeypatch):
    from toolprobe import templates
    tok = TemplateTokenizer('{"name": <function-name>, "arguments": <args-json-object>}')
    monkeypatch.setattr(templates, "get_tokenizer", lambda repo: tok)
    assert templates.template_sanity("any") is True


def test_template_sanity_catches_doubled_brace_example(monkeypatch):
    # Jinja-escaping leak observed in Qwen2.5's template: the format example
    # renders as {{"name": ...}} and teaches small models malformed JSON.
    from toolprobe import templates
    tok = TemplateTokenizer('{{"name": <function-name>, "arguments": <args-json-object>}}')
    monkeypatch.setattr(templates, "get_tokenizer", lambda repo: tok)
    assert templates.template_sanity("any") is False
