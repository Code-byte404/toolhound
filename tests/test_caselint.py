from toolprobe.caselint import validate
from toolprobe.models import Case

TOOLS = {
    "get_weather": {"type": "function", "function": {"name": "get_weather",
        "parameters": {"type": "object",
            "properties": {"location": {"type": "string"},
                           "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["location"]}}},
    "create_event": {"type": "function", "function": {"name": "create_event",
        "parameters": {"type": "object",
            "properties": {"title": {"type": "string"}, "start": {"type": "string"},
                           "attendees": {"type": "array", "items": {"type": "string"}}},
            "required": ["title", "start"]}}},
    "convert_currency": {"type": "function", "function": {"name": "convert_currency",
        "parameters": {"type": "object",
            "properties": {"amount": {"type": "number"}, "from": {"type": "string"},
                           "to": {"type": "string"}},
            "required": ["amount", "from", "to"]}}},
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
                         if "category " not in e and "total case count" not in e
                         and not e.startswith("C7 ")]
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


def _leak_case(cid, split, tool, args, content):
    return _case(id=cid, split=split, tools=[tool], family="f",
                 turns=[{"role": "user", "content": content}],
                 expected={"tool": tool, "args": args}, arg_rules={})


def test_value_leakage_email_flagged():
    # Same concrete email in a dev attendees list and a test attendees list —
    # whole-binding slot leakage would miss this; the value check must catch it.
    d = _leak_case("d", "dev", "create_event",
                   {"title": "dev sync", "start": "2026-03-20T10:00:00",
                    "attendees": ["shared@corp.com", "d2@corp.com"]}, "book dev sync")
    t = _leak_case("t", "test", "create_event",
                   {"title": "test sync", "start": "2026-03-21T10:00:00",
                    "attendees": ["shared@corp.com", "t2@corp.com"]}, "book test sync")
    errs = validate([d], [t], TOOLS, {})
    assert any("shared@corp.com" in e and "leak" in e.lower() for e in errs)


def test_value_leakage_currency_pair_flagged():
    # Identical ORDERED (from,to) pair in both splits (amounts differ) — leak.
    d = _leak_case("d", "dev", "convert_currency",
                   {"amount": 100, "from": "USD", "to": "EUR"}, "convert 100 usd to eur")
    t = _leak_case("t", "test", "convert_currency",
                   {"amount": 250, "from": "USD", "to": "EUR"}, "convert 250 usd to eur")
    errs = validate([d], [t], TOOLS, {})
    assert any("USD" in e and "EUR" in e and "leak" in e.lower() for e in errs)


def test_value_leakage_entity_title_flagged():
    d = _leak_case("d", "dev", "create_event",
                   {"title": "vet visit", "start": "2026-03-20T10:00:00"}, "book vet visit dev")
    t = _leak_case("t", "test", "create_event",
                   {"title": "vet visit", "start": "2026-03-21T10:00:00"}, "book vet visit test")
    errs = validate([d], [t], TOOLS, {})
    assert any("vet visit" in e and "leak" in e.lower() for e in errs)


def test_value_leakage_prior_turn_currency_pair_flagged():
    # The leaking pair lives only in a prior-turn tool_call, not in expected.
    d = _case(id="d", split="dev", tools=["convert_currency"],
              turns=[{"role": "user", "content": "convert 100 usd to cad"},
                     {"role": "assistant", "tool_call": {"tool": "convert_currency",
                        "args": {"amount": 100, "from": "USD", "to": "CAD"}}},
                     {"role": "tool", "content": "135.0"},
                     {"role": "user", "content": "now to mxn"}],
              expected={"tool": "convert_currency",
                        "args": {"amount": 100, "from": "USD", "to": "MXN"}}, arg_rules={})
    t = _case(id="t", split="test", tools=["convert_currency"],
              turns=[{"role": "user", "content": "convert 50 usd to cad"},
                     {"role": "assistant", "tool_call": {"tool": "convert_currency",
                        "args": {"amount": 50, "from": "USD", "to": "CAD"}}},
                     {"role": "tool", "content": "67.5"},
                     {"role": "user", "content": "now to jpy"}],
              expected={"tool": "convert_currency",
                        "args": {"amount": 50, "from": "USD", "to": "JPY"}}, arg_rules={})
    errs = validate([d], [t], TOOLS, {})
    assert any("USD" in e and "CAD" in e and "leak" in e.lower() for e in errs)


def test_value_leakage_clean_disjoint_ok():
    # Disjoint values; individual currency codes recurring (USD) is NOT a leak,
    # only the ORDERED pair is checked. No value-leak error may fire.
    d = _leak_case("d", "dev", "convert_currency",
                   {"amount": 100, "from": "USD", "to": "EUR"}, "convert 100 usd to eur")
    t = _leak_case("t", "test", "convert_currency",
                   {"amount": 250, "from": "USD", "to": "GBP"}, "convert 250 usd to gbp")
    errs = validate([d], [t], TOOLS, {})
    assert not any("leak" in e.lower() for e in errs)


def test_bad_arg_rule_value_flagged():
    # {"match": "st"} is not a recognized shape -> would silently fall through
    # the frozen scorer to default equality. Must be flagged.
    c = _case(arg_rules={"location": {"match": "st"}})
    errs = validate([c], [], TOOLS, {})
    assert any("location" in e and ("invalid" in e.lower() or "shape" in e.lower())
               for e in errs)


def test_bad_arg_rule_key_flagged():
    c = _case(arg_rules={"location": {"normallize": "iso8601_minute"}})  # typo key
    errs = validate([c], [], TOOLS, {})
    assert any("location" in e and ("invalid" in e.lower() or "shape" in e.lower())
               for e in errs)


def test_good_arg_rule_shapes_ok():
    for rule in ({"equiv": ["Tokyo"]}, {"match": "set"}, {"match": "semantic"},
                 {"normalize": "iso8601_minute"}):
        c = _case(arg_rules={"location": rule})
        errs = validate([c], [], TOOLS, {})
        assert not any("shape" in e.lower() or "invalid arg" in e.lower() for e in errs)


def test_c7_missing_size_flagged():
    def c7(cid, split, n):
        return _case(id=cid, cat="C7", split=split, n_tools=n, tools=["get_weather"],
                     turns=[{"role": "user", "content": f"weather {cid}"}],
                     expected={"tool": "get_weather", "args": {"location": "Tokyo"}},
                     arg_rules={})
    errs = validate([c7("d1", "dev", 3)], [c7("t1", "test", 3)], TOOLS, {})
    assert any("C7" in e and "size" in e.lower() for e in errs)
