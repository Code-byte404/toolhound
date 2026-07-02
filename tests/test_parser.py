from pathlib import Path

import pytest

from toolprobe.parser import parse_framework, parse_rescue

FIX = Path(__file__).parent / "fixtures"


def fx(name):
    return (FIX / name).read_text()


def test_framework_accepts_clean_and_bare():
    assert parse_framework(fx("clean_toolcall.txt")).tool == "get_weather"
    assert parse_framework(fx("bare_json.txt")).args["timezone"] == "Europe/London"


def test_framework_accepts_llama_parameters_key():
    # Llama-3.2's canonical format uses "parameters" not "arguments" -- a valid
    # well-formed call the framework parser must recognize, else a capable model
    # scores 0 purely from a key-name mismatch (real finding, 2026-07-02).
    call = parse_framework(fx("llama_bare.txt"))
    assert call is not None and call.tool == "get_weather"
    assert call.args["location"] == "Tokyo"
    assert parse_framework(fx("llama_toolcall.txt")).args["timezone"] == "Europe/London"


def test_framework_accepts_llama_python_tag():
    # <|python_tag|> is Llama's canonical tool-call token (like Qwen's
    # <tool_call>). Without it, strict-tier attribution can't see Llama's
    # false-triggers on abstention cases (real finding, 2026-07-02).
    call = parse_framework(fx("llama_python_tag.txt"))
    assert call is not None and call.tool == "search_web"
    assert call.args["query"] == "weather in New York"


@pytest.mark.parametrize("name", ["codeblock.txt", "mistral_style.txt", "alias_keys.txt",
                                  "prose_wrapped.txt", "single_quotes.txt", "garbage.txt"])
def test_framework_rejects_dirty(name):
    assert parse_framework(fx(name)) is None


def test_strict_rescue_saves_codeblock_mistral_alias_stringified():
    assert parse_rescue(fx("codeblock.txt"), "strict").tool == "convert_currency"
    assert parse_rescue(fx("mistral_style.txt"), "strict").tool == "search_web"
    assert parse_rescue(fx("alias_keys.txt"), "strict").args["unit"] == "fahrenheit"
    assert parse_rescue(fx("stringified_args.txt"), "strict").args["to"] == "alex@corp.com"


def test_strict_rescue_rejects_prose_and_single_quotes():
    assert parse_rescue(fx("prose_wrapped.txt"), "strict") is None
    assert parse_rescue(fx("single_quotes.txt"), "strict") is None


def test_lenient_rescue_saves_prose_and_single_quotes():
    assert parse_rescue(fx("prose_wrapped.txt"), "lenient").args["location"] == "Seattle"
    assert parse_rescue(fx("single_quotes.txt"), "lenient").tool == "get_stock"


def test_nothing_parses_garbage():
    assert parse_rescue(fx("garbage.txt"), "lenient") is None


def test_lenient_rescue_dedupes_doubled_braces():
    # Qwen2.5's chat template teaches a doubled-brace example ({{"name": ...}})
    # and small models copy it literally. En route the set-literal parse raises
    # TypeError in ast.literal_eval (must not crash); the lenient tier's
    # brace-dedup candidate must then recover the intended call.
    call = parse_rescue(fx("double_brace.txt"), "lenient")
    assert call is not None and call.tool == "get_time"
    assert call.args["timezone"] == "UTC"
    assert parse_rescue(fx("double_brace.txt"), "strict") is None
