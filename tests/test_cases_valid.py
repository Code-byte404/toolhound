from pathlib import Path

from toolprobe.caselint import validate
from toolprobe.casegen import load_slots
from toolprobe.models import load_cases, load_tools

CASES = Path(__file__).parent.parent / "cases"


def test_generated_case_set_is_valid():
    dev = load_cases(CASES / "dev.jsonl")
    test = load_cases(CASES / "test.jsonl")
    tools = load_tools(CASES / "tools.yaml")
    slots = load_slots(CASES / "slots.yaml")
    errors = validate(dev, test, tools, slots)
    assert errors == [], "case set invalid:\n" + "\n".join(errors)
