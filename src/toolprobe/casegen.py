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


_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_EXACT = re.compile(r"^\{(\w+)\}$")


def load_templates(path: str | Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)


def subst(obj, binding: dict):
    if isinstance(obj, str):
        m = _EXACT.match(obj)
        if m:
            return binding[m.group(1)]  # preserve raw type
        return _PLACEHOLDER.sub(lambda x: str(binding[x.group(1)]), obj)
    if isinstance(obj, list):
        return [subst(v, binding) for v in obj]
    if isinstance(obj, dict):
        return {k: subst(v, binding) for k, v in obj.items()}
    return obj


def _resolve_rules(rules: dict, binding: dict) -> dict:
    out = {}
    for arg, rule in (rules or {}).items():
        if "equiv_field" in rule:
            out[arg] = {"equiv": binding[rule["equiv_field"]]}
        else:
            out[arg] = rule
    return out


def _build_turns(tmpl: dict, binding: dict, phrasing: str) -> list[dict]:
    if "turns" in tmpl:
        return [subst(t, binding) for t in tmpl["turns"]]
    return [{"role": "user", "content": subst(phrasing, binding)}]


def _make_case(tmpl, binding, phrasing, split, seq, cat, size=None) -> dict:
    cid = (f"{tmpl['id']}_{split}_{seq:03d}" if size is None
           else f"{tmpl['id']}_s{size}_{split}_{seq:02d}")
    tools = _catalog(tmpl, size) if size is not None else list(tmpl["tools"])
    case = {
        "id": cid, "cat": cat, "family": tmpl["id"], "split": split,
        "tools": tools,
        "turns": _build_turns(tmpl, binding, phrasing),
        "expected": subst(tmpl["expected"], binding) if tmpl.get("expected") is not None else None,
    }
    if size is not None:
        case["n_tools"] = size
    if case["expected"] is None:
        case.pop("arg_rules", None)
    else:
        case["arg_rules"] = _resolve_rules(tmpl.get("arg_rules", {}), binding)
    return case


def expand_template(tmpl: dict, slots: dict, split: str) -> list[dict]:
    if tmpl["cat"] == "C7":
        return _expand_c7(tmpl, slots, split)
    bindings = slots[tmpl["slot"]][split]
    phrasings = tmpl["phrasings"]
    s, p = len(bindings), len(phrasings)
    count = tmpl["count"]
    if count > s * p:
        raise ValueError(f"{tmpl['id']}/{split}: count {count} exceeds {s}*{p} unique pairs")
    return [_make_case(tmpl, bindings[i % s], phrasings[(i // s) % p],
                       split, i, tmpl["cat"])
            for i in range(count)]


def _catalog(tmpl: dict, size: int) -> list[str]:  # implemented in Task 5
    target = tmpl["target_tool"]
    distractors = [t for t in tmpl["catalog"] if t != target]
    if size - 1 > len(distractors):
        raise ValueError(f"{tmpl['id']}: size {size} needs {size-1} distractors, "
                         f"have {len(distractors)}")
    return [target] + distractors[:size - 1]


def _expand_c7(tmpl: dict, slots: dict, split: str) -> list[dict]:  # implemented in Task 5
    bindings = slots[tmpl["slot"]][split]
    phrasings = tmpl["phrasings"]
    s, p = len(bindings), len(phrasings)
    n = tmpl["count_per_size"]
    if n > s * p:
        raise ValueError(f"{tmpl['id']}/{split}: count_per_size {n} exceeds {s}*{p}")
    cases = []
    for size in tmpl["sizes"]:
        for i in range(n):
            cases.append(_make_case(tmpl, bindings[i % s], phrasings[(i // s) % p],
                                    split, i, "C7", size=size))
    return cases


def generate_all(templates: list[dict], slots: dict) -> tuple[list[dict], list[dict]]:
    dev, test = [], []
    for tmpl in templates:
        dev.extend(expand_template(tmpl, slots, "dev"))
        test.extend(expand_template(tmpl, slots, "test"))
    return dev, test
