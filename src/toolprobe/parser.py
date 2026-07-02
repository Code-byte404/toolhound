"""Dual parser. parse_framework = what strict framework tooling would accept.
parse_rescue = progressively lenient salvage. Leniency is a free parameter that
swings attribution, so both tiers' behavior is defined here exactly (§4.2):

framework: single <tool_call>JSON</tool_call> block, or the whole output being
one JSON object, with keys name (str) + arguments (dict).
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


def parse_framework(raw: str) -> ToolCall | None:
    m = _TOOL_CALL_RE.search(raw)
    body = m.group(1) if m else raw.strip()
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return None
    if (isinstance(obj, dict) and isinstance(obj.get("name"), str)
            and isinstance(obj.get("arguments"), dict)):
        return ToolCall(tool=obj["name"], args=obj["arguments"])
    return None


def parse_rescue(raw: str, leniency: str = "lenient") -> ToolCall | None:
    for candidate in _candidates(raw, leniency):
        call = _normalize(_to_obj(candidate, leniency))
        if call is not None:
            return call
    return None


def _candidates(raw: str, leniency: str):
    m = _TOOL_CALL_RE.search(raw)
    if m:
        yield m.group(1)
    stripped = raw.strip()
    if stripped.startswith("[TOOL_CALLS]"):
        yield stripped.removeprefix("[TOOL_CALLS]").strip()
    yield from _FENCE_RE.findall(raw)
    yield stripped
    if leniency == "lenient":
        yield from _balanced_objects(raw)


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
    args = obj.get("arguments") if "arguments" in obj else obj.get("args")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if isinstance(name, str) and isinstance(args, dict):
        return ToolCall(tool=name, args=args)
    return None
