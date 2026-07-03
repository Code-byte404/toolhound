"""Method registry. baseline = identity; pa_tool = PA-Tool (arXiv 2510.07248)."""
from .base import Baseline, Method, MethodResult
from .pa_tool import PATool

METHODS = ("baseline", "pa_tool")


def get_method(name: str, **kw) -> Method:
    if name == "baseline":
        return Baseline()
    if name == "pa_tool":
        return PATool(**kw)
    raise SystemExit(f"unknown method {name!r}; choose from {', '.join(METHODS)}")
