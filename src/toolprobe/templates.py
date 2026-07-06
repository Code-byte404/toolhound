"""Fair-prompt rendering: each model gets its OWN chat template's tool format
(design doc §4.2 hard rule) via tokenizer.apply_chat_template(tools=...).
Fixed date injected so 'tomorrow/Friday' cases never drift (§4.3)."""
import json
import re
from functools import lru_cache

from .backend import get_tokenizer
from .models import Case

FIXED_DATE = "2026-03-20"  # a Friday; all gold datetimes in cases/ assume it


def _granite_tool_call_text(tool: str, args: dict) -> str:
    """Granite's native assistant tool-call wire format -- what it emits and what
    parser._parse_granite recognizes: <|tool_call|>[{"name":..,"arguments":..}]."""
    return "<|tool_call|>" + json.dumps([{"name": tool, "arguments": args}])


# Repos whose chat template has NO tool_calls branch: it silently DROPS an assistant
# tool call in multi-turn history, so the model would see a tool RESULT with no
# preceding CALL (broken context). For these we serialize the prior call into `content`
# using the model's OWN native format -- keeping multi-turn context faithful without
# hand-rolling a foreign format (fair-prompt). Granite 3.3's template only knows
# `available_tools` ('tool_calls' absent); surfaced by the harness 2026-07-06.
_NATIVE_TOOL_CALL_SERIALIZERS = {"granite": _granite_tool_call_text}


@lru_cache(maxsize=8)
def _tool_call_content_serializer(repo: str):
    """If `repo`'s template drops assistant tool_calls, return a (tool, args)->str
    native serializer to embed the prior call in `content`; else None (the template
    renders the OpenAI tool_calls field faithfully). The drop is DETECTED by probing
    the template with a unique marker, so a new template-lacking model is caught even
    if unregistered -- in which case we raise rather than silently break context."""
    tok = get_tokenizer(repo)
    marker = "ZZ9_TOOLCALL_MARKER"
    probe = [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "",
              "tool_calls": [{"type": "function",
                              "function": {"name": "probe", "arguments": {"x": marker}}}]},
             {"role": "tool", "content": "ok"},
             {"role": "user", "content": "again"}]
    tool = {"type": "function", "function": {
        "name": "probe", "description": "p",
        "parameters": {"type": "object", "properties": {"x": {"type": "string"}},
                       "required": ["x"]}}}
    try:
        rendered = tok.apply_chat_template(probe, tools=[tool],
                                           add_generation_prompt=True, tokenize=False)
    except Exception:
        rendered = ""
    if marker in rendered:
        return None                       # template renders tool_calls faithfully
    r = repo.lower()                      # template drops them -> need a native serializer
    for key, fn in _NATIVE_TOOL_CALL_SERIALIZERS.items():
        if key in r:
            return fn
    raise ValueError(
        f"{repo}'s chat template drops assistant tool_calls in multi-turn history and no "
        f"native serializer is registered; add one to _NATIVE_TOOL_CALL_SERIALIZERS so "
        f"prior tool calls render faithfully (otherwise multi-turn context is broken).")


def case_to_messages(case: Case, current_date: str = FIXED_DATE, *,
                     tool_call_serializer=None) -> list[dict]:
    msgs: list[dict] = [{"role": "system",
                         "content": f"Current date: {current_date} (Friday). "
                                    "Use the provided tools when appropriate."}]
    for turn in case.turns:
        if turn.role == "assistant" and turn.tool_call is not None:
            if tool_call_serializer is not None:
                # template can't render tool_calls -> embed the call in content, natively
                native = tool_call_serializer(turn.tool_call["tool"], turn.tool_call["args"])
                msgs.append({"role": "assistant", "content": (turn.content or "") + native})
            else:
                # arguments as a DICT, not a JSON string: Qwen3.5's template does
                # `arguments | items` (needs a mapping); Gemma accepts a dict too.
                msgs.append({"role": "assistant", "content": turn.content or "", "tool_calls": [{
                    "type": "function",
                    "function": {"name": turn.tool_call["tool"],
                                 "arguments": turn.tool_call["args"]},
                }]})
        else:
            msgs.append({"role": turn.role, "content": turn.content or ""})
    return msgs


def render_with_tokenizer(tokenizer, case: Case, tools: dict[str, dict],
                          current_date: str = FIXED_DATE, *,
                          tool_call_serializer=None) -> str:
    # enable_thinking=False keeps hybrid-reasoning models (Qwen3.5) from emitting a
    # <think> block before the tool call -- this is a tool-calling reliability eval,
    # not a reasoning one. It's forwarded as a template variable, so models whose
    # template ignores it (Granite, Gemma) are unaffected.
    return tokenizer.apply_chat_template(
        case_to_messages(case, current_date, tool_call_serializer=tool_call_serializer),
        tools=[tools[t] for t in case.tools],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )


def render(repo: str, case: Case, tools: dict[str, dict],
           current_date: str = FIXED_DATE) -> str:
    return render_with_tokenizer(get_tokenizer(repo), case, tools, current_date,
                                 tool_call_serializer=_tool_call_content_serializer(repo))


_EXAMPLE_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def template_sanity(repo: str) -> bool:
    """Gate separating 'framework/template bug' from 'model format failure' (附三).
    Checks: (1) the official template actually renders a passed tool into the
    prompt; (2) tokenize->detokenize round-trip preserves the tool name
    (special tokens not mangled/swallowed); (3) any tool-call format example
    the template shows is not itself malformed -- a Jinja-escaping leak like
    Qwen2.5's {{"name": ...}} teaches the model the wrong format."""
    tok = get_tokenizer(repo)
    probe = {"type": "function", "function": {
        "name": "probe_tool", "description": "sanity probe",
        "parameters": {"type": "object", "properties": {"x": {"type": "string"}},
                       "required": ["x"]}}}
    try:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": "call the tool with x=1"}],
            tools=[probe], add_generation_prompt=True, tokenize=False)
    except Exception:
        return False
    if "probe_tool" not in prompt:
        return False
    if any("{{" in ex for ex in _EXAMPLE_RE.findall(prompt)):
        return False
    ids = tok.encode(prompt)
    return "probe_tool" in tok.decode(ids)
