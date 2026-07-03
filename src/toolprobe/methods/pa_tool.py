"""PA-Tool (arXiv 2510.07248): adapt tool schemas to the model by renaming tool
and parameter names to high-"peakedness" candidates the model itself generates.
Faithful reconstruction from the paper (public repo is a project page), with two
documented deviations: schema-sparsity param context and the _is_valid_name safeguard.
Pure logic here; the candidate generator is injected. No mlx."""
import hashlib
import json
import keyword
import re
from dataclasses import dataclass
from pathlib import Path

from ..models import ToolCall
from .base import MethodResult

N_CANDIDATES = 32
CAND_TEMP = 0.4
ALPHA = 0.2
_LOGIC_VERSION = 1  # bump when adaptation logic changes, so stale caches don't silently hit
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Prompt-echo words and articles/prepositions the weak candidate-generator emits as
# prose openers; _extract_name would otherwise pick these as "names".
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "with", "is", "are", "be",
    "this", "that", "it", "as", "by", "for", "from", "your", "you",
    "name", "function", "tool", "parameter", "param", "reply", "choose", "concise",
    "short", "snake", "case", "example", "current", "value", "string", "here",
})
_MIN_NAME_LEN = 3


def _is_valid_name(name: str) -> bool:
    """A candidate is a usable tool/param name only if it is a real identifier that
    is not a Python keyword, not a prose stopword/prompt-echo, and long enough to be
    a deliberate name. Junk fails here and the caller keeps the canonical name.
    This filter is a reconstruction SAFEGUARD, a SECOND deviation from the paper
    (besides the schema-sparsity param context): the paper assumes a candidate-capable
    model and has no such filter; a weak generator emits prose whose leading token
    would otherwise become a tool name."""
    return (bool(name) and name.isidentifier() and not keyword.iskeyword(name)
            and name not in _STOPWORDS and len(name) >= _MIN_NAME_LEN)


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
    """Sample n candidates + a greedy reference, keep only valid names, return
    (selected_name, seed_used). No valid candidate -> None (caller keeps canonical)."""
    cands = [c for c in (_extract_name(t) for t in gen(prompt, n, temp, seed))
             if c and _is_valid_name(c)]
    seed += n
    ref_raw = gen(prompt, 1, 0.0, seed)
    seed += 1
    ref = _extract_name(ref_raw[0]) if ref_raw else None
    if ref is not None and not _is_valid_name(ref):
        ref = None
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


class PATool:
    """PA-Tool method (arXiv 2510.07248). Stochastic adaptation is seeded and
    cached; a cache hit reproduces a prior run without model access."""
    name = "pa_tool"

    def __init__(self, cache_dir=None, n: int = N_CANDIDATES, temp: float = CAND_TEMP,
                 alpha: float = ALPHA, base_seed: int = 0):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.n, self.temp, self.alpha, self.base_seed = n, temp, alpha, base_seed

    def _key(self, repo: str, tools: dict) -> str:
        payload = json.dumps({"repo": repo, "tools": tools, "seed": self.base_seed,
                              "n": self.n, "temp": self.temp, "alpha": self.alpha,
                              "logic_version": _LOGIC_VERSION},
                             sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def prepare(self, repo: str, tools: dict[str, dict], *, gen=None) -> MethodResult:
        cache_file = None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"pa_tool-{self._key(repo, tools)}.json"
            if cache_file.exists():
                a = PAToolAdaptation.from_dict(json.loads(cache_file.read_text()))
                return MethodResult(tools=a.renamed_tools,
                                    canonicalize=make_canonicalize(a), meta=a.to_dict())
        if gen is None:
            raise ValueError("PATool.prepare needs a `gen` (no cache hit)")
        a = pa_tool_adapt(tools, gen, n=self.n, temp=self.temp,
                          alpha=self.alpha, base_seed=self.base_seed)
        if cache_file is not None:
            cache_file.write_text(json.dumps(a.to_dict(), indent=2))
        return MethodResult(tools=a.renamed_tools,
                            canonicalize=make_canonicalize(a), meta=a.to_dict())
