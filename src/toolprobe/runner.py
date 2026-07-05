"""Generation paths: free (unconstrained) and grammar-constrained. Both render the
SAME fair prompt (each model's own tool template); only decoding differs."""
from .backend import GenResult, generate
from .grammar import build_grammar, detect_family
from .models import Case
from .templates import FIXED_DATE, render


def run_free(repo: str, case: Case, tools: dict[str, dict],
             current_date: str = FIXED_DATE, max_tokens: int = 256) -> GenResult:
    return generate(repo, render(repo, case, tools, current_date), max_tokens=max_tokens)


def run_constrained(repo: str, case: Case, tools: dict[str, dict],
                    current_date: str = FIXED_DATE, max_tokens: int = 256,
                    family: str | None = None) -> GenResult:
    """Trigger-gated constrained decoding: the model's JSON tool-call body is masked
    to a discriminated union over `case.tools`, but only after it emits its family's
    opener -- so abstention stays reachable. `tools` is the presented catalog (already
    method-renamed by the caller if a method is active); grammar restricts to its keys
    for `case.tools`. `family` overrides repo-based wire-family detection."""
    spec = build_grammar(family or detect_family(repo), case.tools, tools)
    return generate(repo, render(repo, case, tools, current_date),
                    max_tokens=max_tokens, grammar=spec)
