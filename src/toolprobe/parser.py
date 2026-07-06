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
# Qwen3.5: <tool_call><function=NAME><parameter=KEY>value</parameter>...</function></tool_call>
# (XML-ish, NOT JSON; param values are raw text)
_FUNCTION_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)
# Granite: <|tool_call|>[{"name": ..., "arguments": {...}}]
_GRANITE_TOKEN = "<|tool_call|>"
# Gemma-4: <|tool_call>call:NAME{key:<|"|>str<|"|>,key:[<|"|>a<|"|>,<|"|>b<|"|>],key:bare}<tool_call|>
# (bespoke, NOT JSON): string values wrapped in <|"|> quote tokens, arrays in [...],
# numbers/bools bare. Values can contain commas/colons (inside quotes) and lists, so it
# needs a small hand parser, not a per-pair regex. Opener "<|tool_call>" is distinct from
# Granite's "<|tool_call|>" (neither is a substring of the other) -> unambiguous dispatch.
_GEMMA_OPEN = "<|tool_call>"
_GEMMA_QUOTE = '<|"|>'
_GEMMA_NAME_RE = re.compile(r"<\|tool_call>\s*call:\s*([A-Za-z_]\w*)")  # name is an identifier


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


def _coerce(v: str):
    """Qwen3.5 emits param values as raw text between <parameter> tags. Coerce to a
    JSON type where possible (numbers/bools/arrays/objects), else keep the string --
    parser is lenient, the scorer decides correctness."""
    v = v.strip()
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return v


def _parse_function_xml(raw: str) -> ToolCall | None:
    """Qwen3.5 XML tool call: <function=NAME><parameter=KEY>value</parameter>...</function>."""
    m = _FUNCTION_RE.search(raw)
    if not m:
        return None
    name = m.group(1).strip()
    args = {k.strip(): _coerce(val) for k, val in _PARAMETER_RE.findall(m.group(2))}
    return ToolCall(tool=name, args=args) if name else None


def _parse_granite(raw: str) -> ToolCall | None:
    """Granite: <|tool_call|>[{"name": .., "arguments": {..}}] -- a JSON list of calls."""
    obj = _leading_json(raw.split(_GRANITE_TOKEN, 1)[1])
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if isinstance(obj, dict) and isinstance(obj.get("name"), str):
        args = _args_field(obj)
        if isinstance(args, dict):
            return ToolCall(tool=obj["name"], args=args)
    return None


def _gemma_value(s: str, i: int):
    """Parse one Gemma value at s[i:], return (value, next_index). A value is a
    <|"|>-quoted string (kept verbatim, may contain commas/colons), a [...] array of
    values, or a bare token (number/bool -> coerced)."""
    n = len(s)
    while i < n and s[i] in " \t\r\n":
        i += 1
    if s.startswith(_GEMMA_QUOTE, i):
        e = s.find(_GEMMA_QUOTE, i + len(_GEMMA_QUOTE))
        if e < 0:                                  # unterminated -> take the rest
            return s[i + len(_GEMMA_QUOTE):], n
        return s[i + len(_GEMMA_QUOTE):e], e + len(_GEMMA_QUOTE)
    if i < n and s[i] == "[":
        i += 1
        items = []
        while i < n:
            while i < n and s[i] in " \t\r\n,":
                i += 1
            if i >= n or s[i] == "]":
                i += 1
                break
            v, i = _gemma_value(s, i)
            items.append(v)
        return items, i
    j = i                                          # bare value up to a separator
    while j < n and s[j] not in ",}]":
        j += 1
    return _coerce(s[i:j]), j


def _gemma_body(body: str) -> dict:
    args, i, n = {}, 0, len(body)
    while i < n:
        while i < n and body[i] in " \t\r\n,":
            i += 1
        c = body.find(":", i)
        if c < 0:
            break
        key = body[i:c].strip()
        val, i = _gemma_value(body, c + 1)
        if key:
            args[key] = val
    return args


def _parse_gemma(raw: str) -> ToolCall | None:
    """Gemma-4 bespoke call: <|tool_call>call:NAME{key:value,...}<tool_call|>. NAME is a
    leading identifier (any stray token before `{`, e.g. a mis-emitted `<audio|>`, is
    ignored); the {...} body is walked by _gemma_body (handles quoted strings, arrays,
    and bare scalars). Braces balanced in case a string value ever contains one."""
    m = _GEMMA_NAME_RE.search(raw)
    if not m:
        return None
    name = m.group(1)
    b = raw.find("{", m.end())
    if b < 0:
        return ToolCall(tool=name, args={})
    depth, end = 0, -1
    for k in range(b, len(raw)):
        if raw[k] == "{":
            depth += 1
        elif raw[k] == "}":
            depth -= 1
            if depth == 0:
                end = k
                break
    body = raw[b + 1:end] if end > 0 else raw[b + 1:]
    return ToolCall(tool=name, args=_gemma_body(body))


def parse_framework(raw: str) -> ToolCall | None:
    if "<function=" in raw:            # Qwen3.5 XML tool call
        return _parse_function_xml(raw)
    if _GRANITE_TOKEN in raw:          # Granite tool-call list
        return _parse_granite(raw)
    if _GEMMA_OPEN in raw:             # Gemma-4 bespoke call format
        return _parse_gemma(raw)
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
    if "<function=" in raw:            # Qwen3.5 XML -- salvage same as framework
        call = _parse_function_xml(raw)
        if call is not None:
            return call
    if _GRANITE_TOKEN in raw:
        call = _parse_granite(raw)
        if call is not None:
            return call
    if _GEMMA_OPEN in raw:             # Gemma-4 bespoke call -- salvage same as framework
        call = _parse_gemma(raw)
        if call is not None:
            return call
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
