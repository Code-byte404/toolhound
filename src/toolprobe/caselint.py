"""Pure validation of the case set: schema, arg-rules, abstention, multi-turn,
id-uniqueness, slot leakage, and category distribution. No mlx."""
import json
from collections import Counter

from .models import Case

TARGETS = {"C1": 40, "C2": 45, "C3": 55, "C4": 35, "C5": 40, "C6": 30, "C7": 55}
CAT_TOL = 8
TOTAL_MIN, TOTAL_MAX = 280, 330
C7_SIZES = (3, 8, 16, 32)

# Arg keys whose string value is a concrete entity that must NOT cross the
# dev/test split. Free-text keys (subject/body/channel/name/room/door), numeric
# keys (amount/level/temperature/duration_minutes) and enum keys (unit/state)
# are deliberately excluded — they legitimately recur across splits.
ENTITY_KEYS = frozenset(
    {"location", "timezone", "title", "text", "query", "symbol", "word", "topic"}
)


def validate(dev: list[Case], test: list[Case], tools: dict, slots: dict) -> list[str]:
    errors: list[str] = []
    all_cases = list(dev) + list(test)
    _check_ids(all_cases, errors)
    for c in all_cases:
        _check_case(c, tools, errors)
    _check_slot_leakage(slots, errors)
    _check_utterance_leakage(dev, test, errors)
    _check_value_leakage(dev, test, errors)
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
    for arg, rule in c.arg_rules.items():
        if arg not in exp.args:
            errors.append(f"{c.id}: arg_rules key '{arg}' has no matching expected arg")
        if not _valid_arg_rule(rule):
            errors.append(f"{c.id}: arg_rules['{arg}'] has invalid shape {rule!r} "
                          f"(expected one of equiv/match:set/match:semantic/"
                          f"normalize:iso8601_minute)")
    if c.n_tools is not None and len(c.tools) != c.n_tools:
        errors.append(f"{c.id}: n_tools={c.n_tools} but {len(c.tools)} tools listed")
    _check_prior_turns(c, tools, errors)


def _check_prior_turns(c: Case, tools: dict, errors: list[str]) -> None:
    for turn in c.turns:
        if turn.tool_call is not None and turn.tool_call.get("tool") not in tools:
            errors.append(f"{c.id}: prior tool_call references unknown tool "
                          f"{turn.tool_call.get('tool')}")


def _valid_arg_rule(rule) -> bool:
    """An arg_rules entry must be exactly one of the four frozen scorer shapes.
    Anything else silently falls through the scorer to default equality."""
    if not isinstance(rule, dict) or len(rule) != 1:
        return False
    (key, val), = rule.items()
    if key == "equiv":
        return isinstance(val, list)
    if key == "match":
        return val in ("set", "semantic")
    if key == "normalize":
        return val == "iso8601_minute"
    return False


def _check_slot_leakage(slots: dict, errors: list[str]) -> None:
    for stype, pools in slots.items():
        dev = {json.dumps(b, sort_keys=True) for b in pools.get("dev", [])}
        test = {json.dumps(b, sort_keys=True) for b in pools.get("test", [])}
        if dev & test:
            errors.append(f"slot '{stype}' leaks bindings between dev and test")


def _iter_strings(v):
    """Yield every string leaf inside a value, recursing into lists."""
    if isinstance(v, str):
        yield v
    elif isinstance(v, list):
        for item in v:
            yield from _iter_strings(item)


def _iter_calls(c: Case):
    """(tool, args) for the expected call and every prior-turn tool_call, so
    handwritten single-turn and multi-turn cases are covered identically."""
    if c.expected is not None:
        yield c.expected.tool, c.expected.args
    for turn in c.turns:
        tc = turn.tool_call
        if tc:
            yield tc.get("tool"), tc.get("args") or {}


def _extract_values(cases: list[Case]) -> tuple[set, set, set]:
    """Concrete entity values in a case list: emails, ordered currency pairs,
    and (key, value) entity items. See ENTITY_KEYS for the checked keys."""
    emails: set[str] = set()
    pairs: set[tuple] = set()
    entities: set[tuple] = set()
    for c in cases:
        for tool, args in _iter_calls(c):
            if not isinstance(args, dict):
                continue
            for v in args.values():
                for s in _iter_strings(v):
                    if "@" in s:
                        emails.add(s)
            for k, v in args.items():
                if k in ENTITY_KEYS:
                    for s in _iter_strings(v):
                        entities.add((k, s))
            if tool == "convert_currency" and "from" in args and "to" in args:
                pairs.add((args["from"], args["to"]))
    return emails, pairs, entities


def _check_value_leakage(dev: list[Case], test: list[Case], errors: list[str]) -> None:
    """Enforce concrete-value disjointness between dev and test (design §9 check
    8). Whole-binding slot leakage (_check_slot_leakage) does not catch a single
    email/pair/entity reused across otherwise-distinct bindings; this does."""
    dev_e, dev_p, dev_ent = _extract_values(dev)
    test_e, test_p, test_ent = _extract_values(test)
    for email in sorted(dev_e & test_e):
        errors.append(f"value leak (email): {email!r} appears in both dev and test")
    for frm, to in sorted(dev_p & test_p):
        errors.append(f"value leak (currency pair): {frm}->{to} appears in both "
                      f"dev and test")
    for key, val in sorted(dev_ent & test_ent):
        errors.append(f"value leak (entity {key}): {val!r} appears in both dev and test")


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
    _check_c7_sizes(dev, test, errors)


def _check_c7_sizes(dev, test, errors: list[str]) -> None:
    """C7 catalog-scaling cases must cover every size in {3,8,16,32}, each
    present and dev/test-balanced. Correct by construction today (7/7 per size);
    guarded so a future template edit that drops or skews a size is caught."""
    dev_c7 = Counter(c.n_tools for c in dev if c.cat == "C7")
    test_c7 = Counter(c.n_tools for c in test if c.cat == "C7")
    for size in sorted(set(dev_c7) | set(test_c7)):
        if size not in C7_SIZES:
            errors.append(f"C7 unexpected catalog size {size} (allowed {list(C7_SIZES)})")
    for size in C7_SIZES:
        d, t = dev_c7.get(size, 0), test_c7.get(size, 0)
        if d == 0 or t == 0:
            errors.append(f"C7 size {size}: missing in dev or test (dev={d} test={t})")
        elif abs(d - t) > max(1, int(0.25 * (d + t))):
            errors.append(f"C7 size {size}: dev/test imbalance {d} vs {t}")
