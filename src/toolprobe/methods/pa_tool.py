"""PA-Tool (arXiv 2510.07248): adapt tool schemas to the model by renaming tool
and parameter names to high-"peakedness" candidates the model itself generates.
Faithful reconstruction from the paper (the public repo is a project page).
Pure logic here; the candidate generator is injected. No mlx."""
import re
from dataclasses import dataclass

from ..models import ToolCall

N_CANDIDATES = 32
CAND_TEMP = 0.4
ALPHA = 0.2
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _extract_name(text: str) -> str | None:
    m = _IDENT.search(text or "")
    return m.group(0).lower() if m else None


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def peakedness_scores(cands: list[str], alpha: float = ALPHA) -> list[int]:
    """Eq. 2: count candidates within edit distance tau = alpha * max_len."""
    if not cands:
        return []
    tau = alpha * max(len(c) for c in cands)
    return [sum(1 for j, sj in enumerate(cands) if j != i and edit_distance(si, sj) <= tau)
            for i, si in enumerate(cands)]


def select_name(cands: list[str], reference: str, alpha: float = ALPHA) -> str:
    """argmax peakedness; ties -> min edit distance to the greedy reference name."""
    scores = peakedness_scores(cands, alpha)
    best = max(scores)
    tied = [c for c, s in zip(cands, scores) if s == best]
    return min(tied, key=lambda c: edit_distance(c, reference))


TOOL_PROMPT = ("Choose a concise function name for this tool.\n"
               "Tool description: {desc}\n"
               "Reply with ONLY a short snake_case function name, nothing else.")
PARAM_PROMPT = ("Choose a concise name for a function parameter.\n"
                "Function description: {desc}\n"
                "The parameter (type: {type}) is currently named '{orig}'.\n"
                "Reply with ONLY a short snake_case parameter name, nothing else.")


@dataclass
class PAToolAdaptation:
    renamed_tools: dict[str, dict]
    name_map: dict[str, str]                 # renamed tool name -> canonical
    param_maps: dict[str, dict[str, str]]    # canonical tool -> {renamed param -> canonical}

    def to_dict(self) -> dict:
        return {"renamed_tools": self.renamed_tools,
                "name_map": self.name_map, "param_maps": self.param_maps}

    @classmethod
    def from_dict(cls, d: dict) -> "PAToolAdaptation":
        return cls(d["renamed_tools"], d["name_map"], d["param_maps"])


def _pick(gen, prompt, n, temp, alpha, seed) -> tuple[str | None, int]:
    """Sample n candidates + a greedy reference, return (selected_name, seed_used)."""
    cands = [c for c in (_extract_name(t) for t in gen(prompt, n, temp, seed)) if c]
    seed += n
    ref_raw = gen(prompt, 1, 0.0, seed)
    seed += 1
    ref = (_extract_name(ref_raw[0]) if ref_raw else None)
    if not cands:
        return None, seed
    return select_name(cands, ref or cands[0], alpha), seed


def pa_tool_adapt(tools: dict[str, dict], gen, *, n: int = N_CANDIDATES,
                  temp: float = CAND_TEMP, alpha: float = ALPHA,
                  base_seed: int = 0) -> PAToolAdaptation:
    renamed_tools: dict[str, dict] = {}
    name_map: dict[str, str] = {}
    param_maps: dict[str, dict[str, str]] = {}
    seed = base_seed
    used_tools: set[str] = set()
    for canon, fdict in tools.items():
        fn = fdict["function"]
        desc = fn.get("description", "")
        new_tool, seed = _pick(gen, TOOL_PROMPT.format(desc=desc), n, temp, alpha, seed)
        if not new_tool or new_tool in used_tools or new_tool in tools:
            new_tool = canon                     # collision / empty guard
        used_tools.add(new_tool)

        params = fn.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])
        new_props: dict[str, dict] = {}
        ren2canon: dict[str, str] = {}
        used_p: set[str] = set()
        for pname, pspec in props.items():
            prompt = PARAM_PROMPT.format(desc=desc, type=pspec.get("type", ""), orig=pname)
            new_p, seed = _pick(gen, prompt, n, temp, alpha, seed)
            if not new_p or new_p in used_p or new_p in props:
                new_p = pname
            used_p.add(new_p)
            new_props[new_p] = pspec
            ren2canon[new_p] = pname
        canon2ren = {v: k for k, v in ren2canon.items()}
        new_params = dict(params)
        new_params["properties"] = new_props
        if required:
            new_params["required"] = [canon2ren.get(r, r) for r in required]
        renamed_tools[canon] = {"type": "function",
                                "function": {"name": new_tool, "description": desc,
                                             "parameters": new_params}}
        name_map[new_tool] = canon
        param_maps[canon] = ren2canon

    for canon in tools:
        name_map.setdefault(canon, canon)        # identity fallbacks
        for p in tools[canon]["function"].get("parameters", {}).get("properties", {}):
            param_maps[canon].setdefault(p, p)
    return PAToolAdaptation(renamed_tools, name_map, param_maps)


def make_canonicalize(a: PAToolAdaptation):
    def canonicalize(call: ToolCall | None) -> ToolCall | None:
        if call is None:
            return None
        canon_tool = a.name_map.get(call.tool, call.tool)
        pmap = a.param_maps.get(canon_tool, {})
        return ToolCall(tool=canon_tool, args={pmap.get(k, k): v for k, v in call.args.items()})
    return canonicalize


class PATool:  # fully implemented in Task 4
    name = "pa_tool"

    def prepare(self, repo, tools, *, gen=None):
        raise NotImplementedError("PATool.prepare implemented in Task 4")
