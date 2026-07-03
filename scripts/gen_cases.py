"""Generate cases/dev.jsonl, cases/test.jsonl, cases/default.jsonl from
templates.yaml + slots.yaml (+ handwritten/). Deterministic; run from repo root."""
import json
import sys
from pathlib import Path

from toolprobe.casegen import generate_all, load_slots, load_templates

CASES = Path("cases")


def _read_handwritten(name: str) -> list[dict]:
    p = CASES / "handwritten" / name
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main() -> int:
    slots = load_slots(CASES / "slots.yaml")
    templates = load_templates(CASES / "templates.yaml")
    dev, test = generate_all(templates, slots)
    dev += _read_handwritten("dev.jsonl")
    test += _read_handwritten("test.jsonl")
    _write(CASES / "dev.jsonl", dev)
    _write(CASES / "test.jsonl", test)
    _write(CASES / "default.jsonl", dev + test)
    print(f"dev={len(dev)} test={len(test)} total={len(dev) + len(test)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
