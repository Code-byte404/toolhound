"""Pure validation of the case set: schema, arg-rules, abstention, multi-turn,
id-uniqueness, slot leakage, and category distribution. No mlx."""
import json
from collections import Counter

from .models import Case

TARGETS = {"C1": 40, "C2": 45, "C3": 55, "C4": 35, "C5": 40, "C6": 30, "C7": 55}
CAT_TOL = 8
TOTAL_MIN, TOTAL_MAX = 280, 330


def validate(dev: list[Case], test: list[Case], tools: dict, slots: dict) -> list[str]:
    errors: list[str] = []
    all_cases = list(dev) + list(test)
    _check_ids(all_cases, errors)
    for c in all_cases:
        _check_case(c, tools, errors)
    _check_slot_leakage(slots, errors)
    _check_utterance_leakage(dev, test, errors)
    _check_distribution(all_cases, dev, test, errors)
    return errors


def _check_ids(cases: list[Case], errors: list[str]) -> None:
    dupes = [cid for cid, n in Counter(c.id for c in cases).items() if n > 1]
    for cid in dupes:
        errors.append(f"duplicate case id: {cid}")


def _check_case(c: Case, tools: dict, errors: list[str]) -> None:
    if c.cat == "C5" or c.expected is None:
        if c.expected is not None:
            errors.append(f"{c.id}: C5 case must have expected=null")
        if c.arg_rules:
            errors.append(f"{c.id}: abstention case must not carry arg_rules")
        _check_prior_turns(c, tools, errors)
        return
    exp = c.expected
    if exp.tool not in c.tools:
        errors.append(f"{c.id}: expected tool {exp.tool} not in case tools {c.tools}")
    if exp.tool not in tools:
        errors.append(f"{c.id}: expected tool {exp.tool} not defined in palette")
        return
    params = tools[exp.tool]["function"]["parameters"]
    props, required = params.get("properties", {}), params.get("required", [])
    for r in required:
        if r not in exp.args:
            errors.append(f"{c.id}: missing required arg '{r}' for {exp.tool}")
    for k, v in exp.args.items():
        if k not in props:
            errors.append(f"{c.id}: arg '{k}' not a property of {exp.tool}")
            continue
        spec = props[k]
        if "enum" in spec and v not in spec["enum"]:
            errors.append(f"{c.id}: arg '{k}'={v!r} not in enum {spec['enum']}")
        t = spec.get("type")
        if t == "string" and not isinstance(v, str):
            errors.append(f"{c.id}: arg '{k}' should be string, got {type(v).__name__}")
        if t == "number" and not isinstance(v, (int, float)):
            errors.append(f"{c.id}: arg '{k}' should be number, got {type(v).__name__}")
        if t == "array" and not isinstance(v, list):
            errors.append(f"{c.id}: arg '{k}' should be array, got {type(v).__name__}")
    for arg in c.arg_rules:
        if arg not in exp.args:
            errors.append(f"{c.id}: arg_rules key '{arg}' has no matching expected arg")
    if c.n_tools is not None and len(c.tools) != c.n_tools:
        errors.append(f"{c.id}: n_tools={c.n_tools} but {len(c.tools)} tools listed")
    _check_prior_turns(c, tools, errors)


def _check_prior_turns(c: Case, tools: dict, errors: list[str]) -> None:
    for turn in c.turns:
        if turn.tool_call is not None and turn.tool_call.get("tool") not in tools:
            errors.append(f"{c.id}: prior tool_call references unknown tool "
                          f"{turn.tool_call.get('tool')}")


def _check_slot_leakage(slots: dict, errors: list[str]) -> None:
    for stype, pools in slots.items():
        dev = {json.dumps(b, sort_keys=True) for b in pools.get("dev", [])}
        test = {json.dumps(b, sort_keys=True) for b in pools.get("test", [])}
        if dev & test:
            errors.append(f"slot '{stype}' leaks bindings between dev and test")


def _utterance_sig(c: Case) -> tuple:
    return tuple(t.content for t in c.turns if t.role == "user")


def _check_utterance_leakage(dev: list[Case], test: list[Case], errors: list[str]) -> None:
    dev_sigs = {_utterance_sig(c) for c in dev}
    for c in test:
        if _utterance_sig(c) in dev_sigs:
            errors.append(f"{c.id}: identical user utterance appears in both dev and test")


def _check_distribution(all_cases, dev, test, errors: list[str]) -> None:
    counts = Counter(c.cat for c in all_cases)
    total = sum(counts.values())
    if not (TOTAL_MIN <= total <= TOTAL_MAX):
        errors.append(f"total case count {total} outside [{TOTAL_MIN}, {TOTAL_MAX}]")
    for cat, want in TARGETS.items():
        got = counts.get(cat, 0)
        if abs(got - want) > CAT_TOL:
            errors.append(f"category {cat}: {got} cases, want {want}±{CAT_TOL}")
    dev_c, test_c = Counter(c.cat for c in dev), Counter(c.cat for c in test)
    for cat in TARGETS:
        d, t = dev_c.get(cat, 0), test_c.get(cat, 0)
        if abs(d - t) > max(3, int(0.25 * (d + t))):
            errors.append(f"category {cat}: dev/test imbalance {d} vs {t}")
