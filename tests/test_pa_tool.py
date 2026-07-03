from toolprobe.methods.pa_tool import (edit_distance, peakedness_scores,
                                       select_name, _extract_name, PAToolAdaptation,
                                       make_canonicalize, pa_tool_adapt)
from toolprobe.models import ToolCall


TOOLS = {
    "get_weather": {"type": "function", "function": {
        "name": "get_weather", "description": "Get current weather for a location",
        "parameters": {"type": "object",
                       "properties": {"location": {"type": "string"}},
                       "required": ["location"]}}},
}


def _fake_gen(mapping):
    """Return a gen that yields a fixed candidate list based on a keyword in the prompt."""
    def gen(prompt, n, temp, seed):
        for key, names in mapping.items():
            if key in prompt:
                return list(names) if n > 1 else [names[0]]
        return ["fallback"] * (n if n > 1 else 1)
    return gen


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("same", "same") == 0


def test_extract_name():
    assert _extract_name("get_weather") == "get_weather"
    assert _extract_name("  The name is: FetchForecast. ") == "the"  # first identifier
    assert _extract_name("123 456") is None


def test_peakedness_prefers_the_cluster():
    # four near-duplicates cluster; one outlier is isolated
    cands = ["get_weather", "get_wether", "get_weathr", "get_weater", "zzzzzzzz"]
    scores = peakedness_scores(cands, alpha=0.2)
    assert scores[-1] == 0                       # outlier: no neighbours within tau
    assert max(scores[:4]) >= 2                   # cluster members have neighbours
    assert select_name(cands, reference="get_weather", alpha=0.2).startswith("get_weath")


def test_select_name_tiebreak_by_reference():
    # two singletons tie at peakedness 0; reference breaks the tie
    cands = ["alpha", "omega"]
    assert select_name(cands, reference="alphb", alpha=0.2) == "alpha"


def test_adapt_renames_tool_and_param_and_builds_inverse():
    # tool candidates cluster on "fetch_forecast"; param on "city"
    gen = _fake_gen({"parameter": ["city", "citi", "cty", "place"] * 8,
                     "weather": ["fetch_forecast", "fetch_forcast", "fetch_forecas",
                                 "get_weather"] * 8})
    a = pa_tool_adapt(TOOLS, gen, n=32, temp=0.4, alpha=0.2, base_seed=0)
    fn = a.renamed_tools["get_weather"]["function"]
    assert fn["name"] == "fetch_forecast"                      # tool renamed
    assert fn["description"] == "Get current weather for a location"  # desc preserved
    assert set(fn["parameters"]["properties"]) == {"city"}     # param renamed
    assert fn["parameters"]["required"] == ["city"]            # required remapped
    assert a.name_map["fetch_forecast"] == "get_weather"       # inverse tool map
    assert a.name_map["get_weather"] == "get_weather"          # identity fallback
    assert a.param_maps["get_weather"]["city"] == "location"   # inverse param map
    assert a.param_maps["get_weather"]["location"] == "location"  # identity fallback


def test_canonicalize_maps_renamed_call_back():
    gen = _fake_gen({"parameter": ["city"] * 32,
                     "weather": ["fetch_forecast"] * 32})
    a = pa_tool_adapt(TOOLS, gen, base_seed=0)
    canon = make_canonicalize(a)
    got = canon(ToolCall(tool="fetch_forecast", args={"city": "Tokyo"}))
    assert got.tool == "get_weather" and got.args == {"location": "Tokyo"}
    assert canon(None) is None
    # an unmapped name is left as-is (will fail scoring downstream)
    passthru = canon(ToolCall(tool="mystery", args={"zzz": 1}))
    assert passthru.tool == "mystery" and passthru.args == {"zzz": 1}


def test_adaptation_roundtrips_through_dict():
    gen = _fake_gen({"parameter": ["city"] * 32, "weather": ["fetch_forecast"] * 32})
    a = pa_tool_adapt(TOOLS, gen, base_seed=0)
    b = PAToolAdaptation.from_dict(a.to_dict())
    assert b.name_map == a.name_map and b.param_maps == a.param_maps
    assert b.renamed_tools == a.renamed_tools


def test_patool_prepare_returns_result(tmp_path):
    from toolprobe.methods.pa_tool import PATool
    gen = _fake_gen({"parameter": ["city"] * 32, "weather": ["fetch_forecast"] * 32})
    mr = PATool(cache_dir=tmp_path, base_seed=0).prepare("repo-x", TOOLS, gen=gen)
    assert mr.tools["get_weather"]["function"]["name"] == "fetch_forecast"
    from toolprobe.models import ToolCall
    assert mr.canonicalize(ToolCall(tool="fetch_forecast", args={"city": "Tokyo"})).tool == "get_weather"


def test_patool_cache_hit_skips_gen(tmp_path):
    from toolprobe.methods.pa_tool import PATool
    calls = {"n": 0}
    def counting_gen(prompt, n, temp, seed):
        calls["n"] += 1
        return ["fetch_forecast"] * (n if n > 1 else 1)
    pt = PATool(cache_dir=tmp_path, base_seed=0)
    pt.prepare("repo-x", TOOLS, gen=counting_gen)
    first = calls["n"]
    assert first > 0
    # second prepare with the same (repo, tools, seed) must hit cache: gen not called again
    def exploding_gen(*a, **k):
        raise AssertionError("gen should not be called on cache hit")
    mr = pt.prepare("repo-x", TOOLS, gen=exploding_gen)
    assert calls["n"] == first
    assert mr.tools["get_weather"]["function"]["name"] == "fetch_forecast"
