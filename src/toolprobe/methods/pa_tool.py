"""PA-Tool (arXiv 2510.07248): adapt tool schemas to the model by renaming tool
and parameter names to high-"peakedness" candidates the model itself generates.
Faithful reconstruction from the paper (the public repo is a project page).
Pure logic here; the candidate generator is injected. No mlx."""
import re

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


class PATool:  # fully implemented in Task 4
    name = "pa_tool"

    def prepare(self, repo, tools, *, gen=None):
        raise NotImplementedError("PATool.prepare implemented in Task 4")
