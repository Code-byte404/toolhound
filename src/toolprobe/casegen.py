"""Deterministic case generation from templates + split-tagged slot pools.
Pure logic (no mlx). Produces case dicts for cases/dev.jsonl and cases/test.jsonl."""
import re
from pathlib import Path

import yaml


def load_slots(path: str | Path) -> dict[str, dict[str, list[dict]]]:
    """slots.yaml -> {slot_type: {"dev": [binding,...], "test": [binding,...]}}."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    for stype, pools in raw.items():
        for split in ("dev", "test"):
            if split not in pools:
                raise ValueError(f"slot type {stype!r} missing split {split!r}")
    return raw
