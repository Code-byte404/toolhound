from toolprobe.caselint import validate
from toolprobe.models import Case

TOOLS = {
    "get_weather": {"type": "function", "function": {"name": "get_weather",
        "parameters": {"type": "object",
            "properties": {"location": {"type": "string"},
                           "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["location"]}}},
}


def _case(**kw):
    base = dict(id="c", cat="C1", tools=["get_weather"], split="dev", family="f",
                turns=[{"role": "user", "content": "weather in Tokyo?"}],
                expected={"tool": "get_weather", "args": {"location": "Tokyo"}},
                arg_rules={"location": {"equiv": ["Tokyo"]}})
    base.update(kw)
    return Case(**base)


def test_valid_case_passes():
    # A single well-formed synthetic case must not trip any per-case check
    # (schema/arg_rules/abstention/id/leakage). The corpus-level distribution
    # check (_check_distribution) inevitably fires for a 1-case input since it
    # compares against the real ~304-case targets (TARGETS/TOTAL_MIN/MAX) —
    # that check is exercised for real over cases/dev.jsonl+test.jsonl in
    # tests/test_cases_valid.py, not here. Filter its (deterministic) noise.
    errs = validate([_case()], [], TOOLS, {})
    non_distribution = [e for e in errs
                         if "category " not in e and "total case count" not in e]
    assert non_distribution == []


def test_unknown_tool_flagged():
    c = _case(expected={"tool": "nope", "args": {}}, arg_rules={})
    errs = validate([c], [], TOOLS, {})
    assert any("nope" in e for e in errs)


def test_missing_required_arg_flagged():
    c = _case(expected={"tool": "get_weather", "args": {"unit": "celsius"}},
              arg_rules={})
    errs = validate([c], [], TOOLS, {})
    assert any("required" in e.lower() for e in errs)


def test_bad_enum_flagged():
    c = _case(expected={"tool": "get_weather", "args": {"location": "Tokyo", "unit": "kelvin"}},
              arg_rules={})
    errs = validate([c], [], TOOLS, {})
    assert any("enum" in e.lower() for e in errs)


def test_argrule_for_unknown_arg_flagged():
    c = _case(arg_rules={"ghost": {"equiv": ["x"]}})
    errs = validate([c], [], TOOLS, {})
    assert any("ghost" in e for e in errs)


def test_abstention_with_argrules_flagged():
    c = _case(cat="C5", expected=None, arg_rules={"location": {"equiv": ["x"]}})
    errs = validate([c], [], TOOLS, {})
    assert any("abstention" in e.lower() or "arg_rules" in e for e in errs)


def test_duplicate_id_flagged():
    # Distinct user utterances so _check_utterance_leakage does NOT fire — this
    # isolates the assertion to _check_ids alone (both cases share id="dup").
    errs = validate(
        [_case(id="dup")],
        [_case(id="dup", turns=[{"role": "user", "content": "weather in Osaka?"}])],
        TOOLS, {})
    assert any(e.startswith("duplicate case id") and "dup" in e for e in errs)


def test_slot_leakage_flagged():
    slots = {"city": {"dev": [{"location": "Tokyo"}], "test": [{"location": "Tokyo"}]}}
    errs = validate([_case()], [], TOOLS, slots)
    assert any("leak" in e.lower() for e in errs)


def test_utterance_leakage_flagged():
    d = _case(id="a", split="dev")
    t = _case(id="b", split="test")  # identical user utterance
    errs = validate([d], [t], TOOLS, {})
    assert any("utterance" in e.lower() for e in errs)
