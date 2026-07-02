"""Layered scoring: parse_ok -> schema_valid -> tool_correct -> args_correct.
Parser is lenient; this module is the strict judge. Arg comparison always goes
through arg_rules -- never bare exact-match (design doc §4.1)."""
import re
from datetime import datetime
from typing import Any

from .models import ExpectedCall, ToolCall


def score(call: ToolCall | None, expected: ExpectedCall | None,
          arg_rules: dict[str, dict[str, Any]], tools: dict[str, dict]) -> dict:
    if expected is None:
        return {
            "parse_ok": call is not None,
            "abstention_correct": call is None,
            "false_trigger": call is not None,
        }
    if call is None:
        return {"parse_ok": False}
    s: dict[str, Any] = {"parse_ok": True}
    s["schema_valid"] = _validate_schema(call, tools)
    s["tool_correct"] = call.tool == expected.tool
    if s["tool_correct"]:
        s["args_correct"] = _args_match(call.args, expected.args, arg_rules)
    return s


def passed(s: dict, expected: ExpectedCall | None) -> bool:
    """Single definition of a fully-passing case (used by attribution + report)."""
    if expected is None:
        return bool(s.get("abstention_correct"))
    return all(s.get(k) for k in ("schema_valid", "tool_correct", "args_correct"))


def _validate_schema(call: ToolCall, tools: dict[str, dict]) -> bool:
    if call.tool not in tools:
        return False
    params = tools[call.tool]["function"]["parameters"]
    props, required = params.get("properties", {}), params.get("required", [])
    if any(k not in call.args for k in required):
        return False
    for k, v in call.args.items():
        if k not in props:
            return False
        spec = props[k]
        if "enum" in spec and v not in spec["enum"]:
            return False
        t = spec.get("type")
        if t == "string" and not isinstance(v, str):
            return False
        if t == "number" and not isinstance(v, (int, float)):
            return False
        if t == "array" and not isinstance(v, list):
            return False
    return True


def _args_match(got: dict, gold: dict, rules: dict) -> bool:
    return all(_one_arg_ok(got.get(k), gold_v, rules.get(k, {})) for k, gold_v in gold.items())


def _one_arg_ok(got: Any, gold: Any, rule: dict) -> bool:
    if got is None:
        return False
    if "equiv" in rule:
        return _norm(got) in {_norm(e) for e in rule["equiv"]} or _norm(got) == _norm(gold)
    if rule.get("match") == "set":
        return isinstance(got, list) and set(map(_norm, got)) == set(map(_norm, gold))
    if rule.get("match") == "semantic":
        return _jaccard(str(got), str(gold)) >= 0.5  # v1 approximation of embedding similarity
    if rule.get("normalize") == "iso8601_minute":
        a, b = _iso_minute(str(got)), _iso_minute(str(gold))
        return a is not None and a == b
    return _norm(got) == _norm(gold)


def _norm(v: Any) -> Any:
    return v.strip().lower() if isinstance(v, str) else v


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta | tb else 1.0


def _iso_minute(s: str) -> str | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).strftime("%Y-%m-%dT%H:%M")
    except ValueError:
        return None
