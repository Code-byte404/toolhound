"""Pure logic tests for the constrained-decoding grammar builder (no mlx)."""
import pytest

from toolprobe.grammar import (FAMILIES, GrammarSpec, WireFamily, build_grammar,
                               detect_family)
from toolprobe.models import load_tools

TOOLS = load_tools("cases/tools.yaml")


def test_qwen_family_wrapper_and_args_key():
    spec = build_grammar("qwen", ["get_weather", "convert_currency"], TOOLS)
    assert isinstance(spec, GrammarSpec)
    assert spec.family.key == "qwen"
    assert spec.family.call_open == "<tool_call>"
    assert spec.family.call_close == "</tool_call>"
    assert spec.family.args_key == "arguments"


def test_llama_family_wrapper_and_args_key():
    spec = build_grammar("llama", ["get_weather"], TOOLS)
    assert spec.family.key == "llama"
    assert "<|python_tag|>" in spec.family.call_open
    assert spec.family.call_close == ""
    assert spec.family.args_key == "parameters"


def test_schema_is_discriminated_union_over_case_tools_only():
    spec = build_grammar("qwen", ["get_weather", "convert_currency"], TOOLS)
    branches = spec.json_schema()["oneOf"]
    assert len(branches) == 2
    names = {b["properties"]["name"]["const"] for b in branches}
    assert names == {"get_weather", "convert_currency"}
    # a tool NOT in the case must be absent (name-restriction is the point)
    assert "send_email" not in names


def test_schema_preserves_required_enum_type_fidelity():
    spec = build_grammar("qwen", ["get_weather"], TOOLS)
    branch = spec.json_schema()["oneOf"][0]
    assert branch["required"] == ["name", "arguments"]
    args = branch["properties"]["arguments"]
    assert args["required"] == ["location"]
    assert args["properties"]["unit"]["enum"] == ["celsius", "fahrenheit"]
    assert args["properties"]["location"]["type"] == "string"


def test_llama_schema_uses_parameters_key():
    spec = build_grammar("llama", ["get_weather"], TOOLS)
    props = spec.json_schema()["oneOf"][0]["properties"]
    assert "parameters" in props and "arguments" not in props


def test_abstention_allowed_by_default():
    spec = build_grammar("qwen", ["get_weather"], TOOLS)
    assert spec.allow_abstention is True


def test_schema_uses_presented_name_for_renamed_catalog():
    """A method (e.g. PA-Tool) keeps the catalog keyed by canonical name but renames
    the presented function.name; the grammar must constrain to the PRESENTED name."""
    renamed = {"get_weather": {"type": "function", "function": {
        "name": "fetch_forecast", "description": "d",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
                       "required": ["location"]}}}}
    spec = build_grammar("qwen", ["get_weather"], renamed)
    assert spec.json_schema()["oneOf"][0]["properties"]["name"]["const"] == "fetch_forecast"


def test_detect_family_maps_registered_repos():
    assert detect_family("mlx-community/Qwen2.5-0.5B-Instruct-4bit") == "qwen"
    assert detect_family("mlx-community/Llama-3.2-3B-Instruct-bf16") == "llama"


def test_detect_family_rejects_unknown_repo():
    with pytest.raises(ValueError):
        detect_family("mlx-community/SomeOther-Model")


def test_detect_family_rejects_qwen3_not_mismapped_to_qwen2():
    # Qwen3.5 shares the "qwen" brand but emits XML, NOT the JSON <tool_call>{...}
    # this grammar describes -- it must NOT match the qwen2 family (would decode-mask
    # to the wrong format). The current lineup is free-decoding only, by design.
    for repo in ["mlx-community/Qwen3.5-2B-4bit",
                 "mlx-community/granite-3.3-2b-instruct-4bit",
                 "mlx-community/gemma-4-12B-it-4bit"]:
        with pytest.raises(ValueError):
            detect_family(repo)


def test_families_registry_shape():
    assert set(FAMILIES) == {"qwen", "llama"}
    assert all(isinstance(f, WireFamily) for f in FAMILIES.values())


def test_json_schema_compiles_to_regex_if_outlines_present():
    """The schema must be consumable by the constrained backend. Skips in
    environments without outlines_core; the mlx e2e tests cover real decoding."""
    oc = pytest.importorskip("outlines_core")
    import json
    spec = build_grammar("qwen", ["get_weather", "convert_currency"], TOOLS)
    regex = oc.json_schema.build_regex_from_schema(json.dumps(spec.json_schema()))
    assert isinstance(regex, str) and "get_weather" in regex and "convert_currency" in regex
