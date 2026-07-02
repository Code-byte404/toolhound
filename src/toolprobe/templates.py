"""Fair-prompt rendering: each model gets its OWN chat template's tool format
(design doc §4.2 hard rule) via tokenizer.apply_chat_template(tools=...).
Fixed date injected so 'tomorrow/Friday' cases never drift (§4.3)."""
import json
import re

from .backend import get_tokenizer
from .models import Case

FIXED_DATE = "2026-03-20"  # a Friday; all gold datetimes in cases/ assume it


def case_to_messages(case: Case, current_date: str = FIXED_DATE) -> list[dict]:
    msgs: list[dict] = [{"role": "system",
                         "content": f"Current date: {current_date} (Friday). "
                                    "Use the provided tools when appropriate."}]
    for turn in case.turns:
        if turn.role == "assistant" and turn.tool_call is not None:
            msgs.append({"role": "assistant", "content": turn.content or "", "tool_calls": [{
                "type": "function",
                "function": {"name": turn.tool_call["tool"],
                             "arguments": json.dumps(turn.tool_call["args"])},
            }]})
        else:
            msgs.append({"role": turn.role, "content": turn.content or ""})
    return msgs


def render_with_tokenizer(tokenizer, case: Case, tools: dict[str, dict],
                          current_date: str = FIXED_DATE) -> str:
    return tokenizer.apply_chat_template(
        case_to_messages(case, current_date),
        tools=[tools[t] for t in case.tools],
        add_generation_prompt=True, tokenize=False,
    )


def render(repo: str, case: Case, tools: dict[str, dict],
           current_date: str = FIXED_DATE) -> str:
    return render_with_tokenizer(get_tokenizer(repo), case, tools, current_date)


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
