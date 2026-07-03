import pytest

from toolprobe.backend import GenResult, generate, load_model

REPO = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


@pytest.mark.mlx
def test_generate_deterministic():
    _, tok = load_model(REPO)
    prompt = tok.apply_chat_template([{"role": "user", "content": "Say hi in 3 words."}],
                                     add_generation_prompt=True, tokenize=False)
    r1, r2 = generate(REPO, prompt, max_tokens=16), generate(REPO, prompt, max_tokens=16)
    assert isinstance(r1, GenResult) and r1.text == r2.text  # temp=0 => deterministic


@pytest.mark.mlx
def test_grammar_reserved():
    with pytest.raises(NotImplementedError):
        generate(REPO, "hi", grammar="{}")


@pytest.mark.mlx
def test_sampling_seed_is_reproducible():
    _, tok = load_model(REPO)
    prompt = tok.apply_chat_template([{"role": "user", "content": "Name a color."}],
                                     add_generation_prompt=True, tokenize=False)
    a = generate(REPO, prompt, max_tokens=8, temp=0.8, seed=7)
    b = generate(REPO, prompt, max_tokens=8, temp=0.8, seed=7)
    assert a.text == b.text                      # same seed+temp => identical
