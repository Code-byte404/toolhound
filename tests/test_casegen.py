from pathlib import Path

import pytest

from toolprobe.casegen import generate_all, load_slots, load_templates, subst, expand_template

CASES = Path(__file__).parent.parent / "cases"


def test_load_slots_shape_and_disjoint():
    slots = load_slots(CASES / "slots.yaml")
    assert "city" in slots
    assert set(slots["city"]) == {"dev", "test"}
    assert isinstance(slots["city"]["dev"], list)
    assert isinstance(slots["city"]["dev"][0], dict)
    # dev and test bindings must not overlap for any slot type (leakage firewall)
    import json
    for stype, pools in slots.items():
        dev = {json.dumps(b, sort_keys=True) for b in pools["dev"]}
        test = {json.dumps(b, sort_keys=True) for b in pools["test"]}
        assert dev.isdisjoint(test), f"slot {stype} leaks between dev and test"


def test_subst_preserves_type_for_exact_placeholder():
    b = {"amount": 100, "loc": "Tokyo", "xs": ["a", "b"]}
    assert subst("{amount}", b) == 100          # int preserved
    assert subst("{xs}", b) == ["a", "b"]        # list preserved
    assert subst("in {loc} now", b) == "in Tokyo now"  # embedded -> str
    assert subst({"k": "{amount}"}, b) == {"k": 100}
    assert subst(["{loc}", "x"], b) == ["Tokyo", "x"]


C1_TMPL = {
    "id": "c1_weather", "cat": "C1", "slot": "city", "count": 3,
    "tools": ["get_weather"],
    "phrasings": ["What's the weather in {location}?", "How's the weather in {location}?"],
    "expected": {"tool": "get_weather", "args": {"location": "{location}"}},
    "arg_rules": {"location": {"equiv_field": "location_equiv"}},
}
SLOTS = {"city": {
    "dev": [{"location": "Tokyo", "location_equiv": ["Tokyo", "tokyo"]},
            {"location": "Paris", "location_equiv": ["Paris"]},
            {"location": "Seattle", "location_equiv": ["Seattle"]}],
    "test": [{"location": "Berlin", "location_equiv": ["Berlin"]},
             {"location": "Madrid", "location_equiv": ["Madrid"]},
             {"location": "Toronto", "location_equiv": ["Toronto"]}]}}


def test_expand_c1_counts_split_and_argrules():
    dev = expand_template(C1_TMPL, SLOTS, "dev")
    assert len(dev) == 3
    assert all(c["split"] == "dev" and c["family"] == "c1_weather" for c in dev)
    assert all(c["cat"] == "C1" for c in dev)
    # arg_rules equiv_field resolved from the binding
    c0 = dev[0]
    assert c0["expected"] == {"tool": "get_weather", "args": {"location": "Tokyo"}}
    assert c0["arg_rules"] == {"location": {"equiv": ["Tokyo", "tokyo"]}}
    assert c0["turns"] == [{"role": "user", "content": "What's the weather in Tokyo?"}]
    # unique (binding, phrasing) pairs -> unique ids
    assert len({c["id"] for c in dev}) == 3
    # distinct bindings are actually used and cover the whole dev pool
    # (fails if pairing were broken, e.g. always reusing bindings[0])
    locs = [c["expected"]["args"]["location"] for c in dev]
    assert sorted(locs) == ["Paris", "Seattle", "Tokyo"]


def test_expand_raises_when_count_exceeds_unique_pairs():
    # 3 dev bindings * 1 phrasing = 3 unique pairs, but count=99 is impossible
    tmpl = {"id": "c1_over", "cat": "C1", "slot": "city", "count": 99,
            "tools": ["get_weather"],
            "phrasings": ["Weather in {location}?"],
            "expected": {"tool": "get_weather", "args": {"location": "{location}"}},
            "arg_rules": {"location": {"equiv_field": "location_equiv"}}}
    with pytest.raises(ValueError):
        expand_template(tmpl, SLOTS, "dev")


def test_expand_c5_abstention_has_null_expected_and_no_argrules():
    tmpl = {"id": "c5_chit", "cat": "C5", "slot": "chitchat", "count": 1,
            "tools": ["get_weather", "search_web"],
            "phrasings": ["{utterance}"], "expected": None}
    slots = {"chitchat": {"dev": [{"utterance": "Thanks, that helps!"}], "test": [{"utterance": "You rock!"}]}}
    dev = expand_template(tmpl, slots, "dev")
    assert dev[0]["expected"] is None
    assert "arg_rules" not in dev[0]


C7_TMPL = {
    "id": "c7_time", "cat": "C7", "slot": "timezone_city",
    "sizes": [3, 8], "count_per_size": 2, "target_tool": "get_time",
    "catalog": ["get_weather", "search_web", "convert_currency", "send_email",
                "create_event", "read_calendar", "translate", "get_stock"],
    "phrasings": ["What's the time in {city}?"],
    "expected": {"tool": "get_time", "args": {"timezone": "{timezone}"}},
    "arg_rules": {"timezone": {"equiv_field": "timezone_equiv"}},
}
TZ_SLOTS = {"timezone_city": {
    "dev": [{"city": "Sydney", "timezone": "Australia/Sydney", "timezone_equiv": ["Australia/Sydney", "Sydney"]},
            {"city": "London", "timezone": "Europe/London", "timezone_equiv": ["Europe/London", "London"]}],
    "test": [{"city": "New York", "timezone": "America/New_York", "timezone_equiv": ["America/New_York", "New York"]},
             {"city": "Mumbai", "timezone": "Asia/Kolkata", "timezone_equiv": ["Asia/Kolkata", "Mumbai"]}]}}


def test_expand_c7_sizes_and_catalog():
    dev = expand_template(C7_TMPL, TZ_SLOTS, "dev")
    assert len(dev) == 2 * 2  # sizes * count_per_size
    by_size = {}
    for c in dev:
        by_size.setdefault(c["n_tools"], []).append(c)
    assert set(by_size) == {3, 8}
    for size, group in by_size.items():
        for c in group:
            assert len(c["tools"]) == size
            assert c["tools"][0] == "get_time"       # target present & first
            assert c["expected"]["tool"] == "get_time"
    assert len({c["id"] for c in dev}) == 4          # unique ids


def test_expand_c6_multiturn_carryover():
    tmpl = {
        "id": "c6_curr", "cat": "C6", "slot": "currency_carry", "count": 1,
        "tools": ["convert_currency"],
        "phrasings": ["unused"],
        "turns": [
            {"role": "user", "content": "Convert {amount} {from} to {to}."},
            {"role": "assistant", "tool_call": {"tool": "convert_currency",
             "args": {"amount": "{amount}", "from": "{from}", "to": "{to}"}}},
            {"role": "tool", "content": "{result}"},
            {"role": "user", "content": "And the same amount to {to2}?"},
        ],
        "expected": {"tool": "convert_currency",
                     "args": {"amount": "{amount}", "from": "{from}", "to": "{to2}"}},
        "arg_rules": {},
    }
    slots = {"currency_carry": {
        "dev": [{"amount": 100, "from": "USD", "to": "EUR", "to2": "GBP", "result": "92.0"}],
        "test": [{"amount": 200, "from": "CAD", "to": "USD", "to2": "MXN", "result": "148.0"}]}}
    c = expand_template(tmpl, slots, "dev")[0]
    assert c["turns"][0]["content"] == "Convert 100 USD to EUR."
    assert c["turns"][1]["tool_call"]["args"]["amount"] == 100     # int preserved
    assert c["turns"][2]["content"] == "92.0"
    assert c["expected"]["args"] == {"amount": 100, "from": "USD", "to": "GBP"}


def test_generate_all_splits_and_families():
    templates = [C1_TMPL, C7_TMPL]
    slots = {**SLOTS, **TZ_SLOTS}
    dev, test = generate_all(templates, slots)
    assert len(dev) == 3 + 4 and len(test) == 3 + 4
    assert all(c["split"] == "dev" for c in dev)
    assert {c["family"] for c in dev} == {"c1_weather", "c7_time"}


def test_handwritten_present_and_split_tagged():
    from toolprobe.models import load_cases
    dev = load_cases(CASES / "dev.jsonl")
    test = load_cases(CASES / "test.jsonl")
    hand = [c for c in dev + test if c.family == "handwritten"]
    assert len(hand) >= 15
    assert all(c.split in ("dev", "test") for c in hand)
    # every handwritten case is C5 (abstain) or C6 (multi-turn)
    assert all(c.cat in ("C5", "C6") for c in hand)


def test_real_templates_hit_distribution():
    from collections import Counter
    templates = load_templates(CASES / "templates.yaml")
    slots = load_slots(CASES / "slots.yaml")
    dev, test = generate_all(templates, slots)
    total = Counter(c["cat"] for c in dev + test)
    targets = {"C1": 40, "C2": 45, "C3": 55, "C4": 35, "C5": 40, "C6": 30, "C7": 55}
    for cat, want in targets.items():
        # templated-only lower bound (handwritten in Task 7 tops up C5/C6)
        assert total[cat] >= want - 12, f"{cat}: got {total[cat]}, want ~{want}"
    assert 250 <= sum(total.values()) <= 320
