"""Method abstraction: a zero-training fix transforms how the tool catalog is
presented to the model, and supplies the inverse map to score a call made in
the transformed space. Pure logic — no mlx."""
from dataclasses import dataclass
from typing import Callable, Protocol

from ..models import ToolCall


@dataclass
class MethodResult:
    tools: dict[str, dict]  # keyed by CANONICAL tool name; values may have renamed content
    canonicalize: Callable[[ToolCall | None], ToolCall | None]
    meta: dict | None = None  # method-specific record (PA-Tool's rename maps) for the report


class Method(Protocol):
    name: str

    def prepare(self, repo: str, tools: dict[str, dict], *, gen=None) -> MethodResult: ...


class Baseline:
    """Fair-prompt baseline: present the catalog unchanged, score as-is."""
    name = "baseline"

    def prepare(self, repo: str, tools: dict[str, dict], *, gen=None) -> MethodResult:
        return MethodResult(tools=tools, canonicalize=lambda c: c)
