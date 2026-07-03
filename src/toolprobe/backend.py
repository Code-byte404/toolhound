"""The ONLY module that may import mlx / mlx_lm. Isolates API drift (build guide §3.1).
Deterministic generation: temp=0 greedy (top_p et al. irrelevant under greedy).
Verified against mlx 0.31.2 / mlx-lm 0.31.3."""
import time
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class GenResult:
    text: str
    ttft_s: float
    tok_per_s: float
    peak_mem_mb: float


@lru_cache(maxsize=4)
def load_model(repo: str):
    from mlx_lm import load
    return load(repo)


def get_tokenizer(repo: str):
    return load_model(repo)[1]


def assert_same_template(bf16_repo: str, q4_repo: str) -> None:
    """Quantization-confound guard (§4.2): only weight precision may differ."""
    t1, t2 = get_tokenizer(bf16_repo), get_tokenizer(q4_repo)
    assert t1.chat_template == t2.chat_template, (
        f"chat templates differ between {bf16_repo} and {q4_repo}; "
        "quantization comparison would be confounded")


def generate(repo: str, prompt: str, max_tokens: int = 256,
             grammar: str | None = None, temp: float = 0.0,
             seed: int | None = None) -> GenResult:
    if grammar is not None:
        raise NotImplementedError("constrained decoding is a v1.1 seam (Outlines/XGrammar)")
    import mlx.core as mx
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load_model(repo)
    if seed is not None:
        mx.random.seed(seed)
    sampler = make_sampler(temp=temp)
    t0 = time.perf_counter()
    text = mlx_generate(model, tokenizer, prompt=prompt,
                        max_tokens=max_tokens, sampler=sampler)
    dt = time.perf_counter() - t0
    n_tok = len(tokenizer.encode(text))
    peak_mb = mx.get_peak_memory() / (1024 * 1024)
    return GenResult(text=text, ttft_s=dt, tok_per_s=(n_tok / dt if dt > 0 else 0.0),
                     peak_mem_mb=peak_mb)
