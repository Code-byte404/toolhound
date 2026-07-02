"""Dual parser. parse_framework = what strict framework tooling would accept.
parse_rescue = progressively lenient salvage. Leniency is a free parameter that
swings attribution, so both tiers' behavior is defined here exactly (§4.2):

framework: a <tool_call>JSON</tool_call> block, a <|python_tag|>JSON tail
(Llama's canonical tool token), or the whole output being one JSON object --
with key name (str) + an args dict under EITHER canonical key "arguments"
(OpenAI/Qwen) or "parameters" (Llama). These are all well-formed native
formats, so recognizing them keeps attribution honest across model families.
strict rescue: + fenced code blocks, [TOOL_CALLS] prefix, tool/args and
nested {"function": {...}} key aliases, arguments given as a JSON string.
lenient rescue: + first balanced {...} anywhere in the text, single-quote
pseudo-JSON (ast.literal_eval), trailing-comma tolerance."""
import ast
import json
import re

from .models import ToolCall

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _args_field(obj: dict):
    """Args dict under either canonical key: 'arguments' (Qwen/OpenAI) or
    'parameters' (Llama). Returns None if neither is present."""
    if "arguments" in obj:
        return obj["arguments"]
    if "parameters" in obj:
        return obj["parameters"]
    return None


def _strict_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _leading_json(text: str):
    """Parse a leading JSON object, tolerating trailing special tokens
    (e.g. Llama's <|eom_id|>) after the <|python_tag|> call."""
    try:
        obj, _ = json.JSONDecoder().raw_decode(text.strip())
        return obj
    except json.JSONDecodeError:
        return None


def parse_framework(raw: str) -> ToolCall | None:
    m = _TOOL_CALL_RE.search(raw)
    if m:
        obj = _strict_json(m.group(1))
    elif "<|python_tag|>" in raw:
        obj = _leading_json(raw.split("<|python_tag|>", 1)[1])
    else:
        obj = _strict_json(raw.strip())
    if isinstance(obj, dict) and isinstance(obj.get("name"), str):
        args = _args_field(obj)
        if isinstance(args, dict):
            return ToolCall(tool=obj["name"], args=args)
    return None


def parse_rescue(raw: str, leniency: str = "lenient") -> ToolCall | None:
    for candidate in _candidates(raw, leniency):
        call = _normalize(_to_obj(candidate, leniency))
        if call is not None:
            return call
    return None


def _candidates(raw: str, leniency: str):
    base = []
    m = _TOOL_CALL_RE.search(raw)
    if m:
        base.append(m.group(1))
    stripped = raw.strip()
    if stripped.startswith("[TOOL_CALLS]"):
        base.append(stripped.removeprefix("[TOOL_CALLS]").strip())
    base.extend(_FENCE_RE.findall(raw))
    base.append(stripped)
    yield from base
    if leniency == "lenient":
        balanced = list(_balanced_objects(raw))
        yield from balanced
        # doubled-brace dedup: buggy chat templates (e.g. Qwen2.5) teach
        # {{"name": ...}} and small models copy it literally. Tried last so
        # well-formed candidates always win first.
        for c in base + balanced:
            if "{{" in c:
                yield c.replace("{{", "{").replace("}}", "}")


def _balanced_objects(raw: str):
    depth, start = 0, None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield raw[start:i + 1]


def _to_obj(text: str, leniency: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if leniency == "lenient":
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", text))
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            pass
    return None


def _normalize(obj) -> ToolCall | None:
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):
        obj = obj["function"]
    name = obj.get("name") or obj.get("tool")
    args = _args_field(obj)
    if args is None:
        args = obj.get("args")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if isinstance(name, str) and isinstance(args, dict):
        return ToolCall(tool=name, args=args)
    return None
