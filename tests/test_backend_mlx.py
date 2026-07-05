import json

import pytest

from toolprobe.backend import GenResult, generate, load_model
from toolprobe.grammar import build_grammar
from toolprobe.models import load_tools
from toolprobe.parser import parse_framework
from toolprobe.scorer import _validate_schema

REPO = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
TOOLS = load_tools("cases/tools.yaml")


@pytest.mark.mlx
def test_generate_deterministic():
    _, tok = load_model(REPO)
    prompt = tok.apply_chat_template([{"role": "user", "content": "Say hi in 3 words."}],
                                     add_generation_prompt=True, tokenize=False)
    r1, r2 = generate(REPO, prompt, max_tokens=16), generate(REPO, prompt, max_tokens=16)
    assert isinstance(r1, GenResult) and r1.text == r2.text  # temp=0 => deterministic


@pytest.mark.mlx
def test_constrained_produces_valid_wrapped_call():
    """The seam works: constrained decoding yields a native, schema-valid call."""
    _, tok = load_model(REPO)
    spec = build_grammar("qwen", ["get_weather"], TOOLS)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "What's the weather in Paris? Use a tool."}],
        tools=[TOOLS["get_weather"]], add_generation_prompt=True, tokenize=False)
    out = generate(REPO, prompt, max_tokens=128, grammar=spec).text
    assert "<tool_call>" in out and "</tool_call>" in out
    call = parse_framework(out)
    assert call is not None and call.tool == "get_weather"
    assert _validate_schema(call, TOOLS)
    # the constrained JSON body must itself be valid JSON
    body = out.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
    assert json.loads(body)["name"] == "get_weather"


@pytest.mark.mlx
def test_sampling_seed_is_reproducible():
    _, tok = load_model(REPO)
    prompt = tok.apply_chat_template([{"role": "user", "content": "Name a color."}],
                                     add_generation_prompt=True, tokenize=False)
    a = generate(REPO, prompt, max_tokens=8, temp=0.8, seed=7)
    b = generate(REPO, prompt, max_tokens=8, temp=0.8, seed=7)
    assert a.text == b.text                      # same seed+temp => identical
