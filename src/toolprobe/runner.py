"""Free (unconstrained) generation path. Constrained path is a v1.1 seam."""
from .backend import GenResult, generate
from .models import Case
from .templates import FIXED_DATE, render


def run_free(repo: str, case: Case, tools: dict[str, dict],
             current_date: str = FIXED_DATE, max_tokens: int = 256) -> GenResult:
    return generate(repo, render(repo, case, tools, current_date), max_tokens=max_tokens)
