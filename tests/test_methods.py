import pytest

from toolprobe.methods import METHODS, get_method
from toolprobe.methods.base import Baseline, MethodResult
from toolprobe.models import ToolCall


def test_baseline_is_identity():
    tools = {"get_weather": {"type": "function", "function": {"name": "get_weather"}}}
    mr = Baseline().prepare("repo", tools)
    assert isinstance(mr, MethodResult)
    assert mr.tools is tools                       # unchanged catalog
    call = ToolCall(tool="get_weather", args={"location": "Tokyo"})
    assert mr.canonicalize(call) is call           # identity
    assert mr.canonicalize(None) is None
    assert mr.meta is None    # baseline carries no adaptation metadata


def test_registry():
    assert set(METHODS) == {"baseline", "pa_tool"}
    assert get_method("baseline").name == "baseline"
    assert get_method("pa_tool").name == "pa_tool"
    with pytest.raises(SystemExit):
        get_method("nope")
