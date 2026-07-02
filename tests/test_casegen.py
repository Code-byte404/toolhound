from pathlib import Path

import pytest

from toolprobe.casegen import load_slots, subst, expand_template

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
