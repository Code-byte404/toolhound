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


def test_framework_accepts_qwen35_function_xml():
    # Qwen3.5's native tool call is XML, not JSON:
    # <function=NAME><parameter=KEY>value</parameter>. Values are raw text; the
    # parser coerces JSON-typed ones (real captured output, 2026-07-06).
    call = parse_framework(fx("qwen35_function_xml.txt"))
    assert call is not None and call.tool == "get_weather"
    assert call.args["location"] == "Paris" and call.args["unit"] == "celsius"


def test_qwen35_xml_coerces_typed_param_values():
    raw = ("<tool_call><function=convert_currency>"
           "<parameter=amount>80</parameter>"
           "<parameter=from>EUR</parameter><parameter=to>USD</parameter>"
           "</function></tool_call>")
    call = parse_framework(raw)
    assert call.args["amount"] == 80 and isinstance(call.args["amount"], int)
    assert call.args["from"] == "EUR"


def test_framework_accepts_granite_tool_call_list():
    # Granite: <|tool_call|>[{"name": .., "arguments": {..}}] -- take the first call.
    call = parse_framework(fx("granite_tool_call.txt"))
    assert call is not None and call.tool == "get_weather"
    assert call.args["location"] == "Paris"


def test_framework_accepts_gemma_bespoke_call():
    # Gemma-4's native call is neither JSON nor XML:
    # <|tool_call>call:NAME{key:<|"|>str<|"|>,...}<tool_call|>. String values are
    # wrapped in <|"|> quote tokens (real captured output, 2026-07-06). Its opener
    # "<|tool_call>" is distinct from Granite's "<|tool_call|>".
    call = parse_framework(fx("gemma_tool_call.txt"))
    assert call is not None and call.tool == "get_weather"
    assert call.args["location"] == "Paris" and call.args["unit"] == "celsius"


def test_gemma_keeps_quoted_strings_but_coerces_bare_values():
    # Quoted (<|"|>...<|"|>) values stay strings even if numeric-looking;
    # bare values are coerced to their JSON type.
    raw = ('<|tool_call>call:book_room{guests:2,city:<|"|>Paris<|"|>,'
           'code:<|"|>007<|"|>}<tool_call|>')
    call = parse_framework(raw)
    assert call.args["guests"] == 2 and isinstance(call.args["guests"], int)
    assert call.args["city"] == "Paris"
    assert call.args["code"] == "007" and isinstance(call.args["code"], str)


def test_gemma_string_value_may_contain_comma():
    # A comma inside a quoted value must not split the pair (the <|"|> close token,
    # not the comma, terminates a string value).
    raw = '<|tool_call>call:get_weather{location:<|"|>Paris, France<|"|>}<tool_call|>'
    call = parse_framework(raw)
    assert call.args["location"] == "Paris, France"


def test_gemma_parses_array_valued_argument():
    # Gemma emits list args as [<|"|>a<|"|>,<|"|>b<|"|>]; the parser must return a real
    # list, not truncate at the first comma (real captured create_event output).
    raw = ('<|tool_call>call:create_event{attendees:[<|"|>a@corp.com<|"|>,<|"|>b@corp.com<|"|>],'
           'start:<|"|>2026-03-23T09:00:00Z<|"|>}<tool_call|>')
    call = parse_framework(raw)
    assert call.args["attendees"] == ["a@corp.com", "b@corp.com"]
    assert call.args["start"] == "2026-03-23T09:00:00Z"       # colon-containing value intact


def test_gemma_name_is_leading_identifier_ignoring_stray_token():
    # a stray special token between the name and '{' (Gemma sometimes mis-emits one)
    # must not corrupt the tool name.
    raw = '<|tool_call>call:search_web<audio|>{query:<|"|>laptops<|"|>}<tool_call|>'
    call = parse_framework(raw)
    assert call.tool == "search_web" and call.args["query"] == "laptops"


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
