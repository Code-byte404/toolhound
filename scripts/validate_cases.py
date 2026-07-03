"""Validate cases/dev.jsonl + cases/test.jsonl. Exit 1 with a report on failure."""
import sys
from pathlib import Path

from toolprobe.caselint import validate
from toolprobe.casegen import load_slots
from toolprobe.models import load_cases, load_tools

CASES = Path("cases")


def main() -> int:
    dev = load_cases(CASES / "dev.jsonl")
    test = load_cases(CASES / "test.jsonl")
    tools = load_tools(CASES / "tools.yaml")
    slots = load_slots(CASES / "slots.yaml")
    errors = validate(dev, test, tools, slots)
    if errors:
        print(f"INVALID — {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"OK — dev={len(dev)} test={len(test)} total={len(dev) + len(test)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
