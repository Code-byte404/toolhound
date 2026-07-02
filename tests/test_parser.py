from pathlib import Path

import pytest

from toolprobe.parser import parse_framework, parse_rescue

FIX = Path(__file__).parent / "fixtures"


def fx(name):
    return (FIX / name).read_text()


def test_framework_accepts_clean_and_bare():
    assert parse_framework(fx("clean_toolcall.txt")).tool == "get_weather"
    assert parse_framework(fx("bare_json.txt")).args["timezone"] == "Europe/London"


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
