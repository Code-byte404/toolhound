from pathlib import Path

from toolprobe.casegen import load_slots

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
