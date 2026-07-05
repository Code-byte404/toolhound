"""The ONLY module that may import mlx / mlx_lm / outlines (build guide §3.1).
Isolates API drift. Generation defaults to temp=0 greedy (deterministic eval);
generate(temp>0, seed=...) does seeded sampling for reproducible stochastic draws
(PA-Tool candidate generation). Constrained decoding (grammar=...) is trigger-gated:
free generation until the model emits its family's tool-call opener, then the JSON
body is masked to the grammar via an outlines_core Guide (so abstention stays
reachable -- design doc §4.4). Verified against mlx 0.31.2 / mlx-lm 0.31.3 /
outlines-core 0.2.14."""
import time
from dataclasses import dataclass
from functools import lru_cache

from .grammar import GrammarSpec


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


@lru_cache(maxsize=8)
def _vocabulary(repo: str):
    """outlines_core Vocabulary for a repo. ~3s to build, so cache per repo.
    Built from the same HF repo mlx-lm loads, so token ids align."""
    from outlines_core import Vocabulary
    return Vocabulary.from_pretrained(repo)


@lru_cache(maxsize=64)
def _index(repo: str, regex: str):
    """outlines_core Index (compiled FSM) for a regex over a repo's vocab.
    Cached by (repo, regex): cases sharing a tool subset share a regex."""
    from outlines_core import Index
    return Index(regex, _vocabulary(repo))


# how many leading tokens to inspect for a bare-JSON (no special opener) tool call
_BARE_JSON_LOOKAHEAD = 4


def _subseq_end(seq: list[int], sub: tuple[int, ...]) -> int | None:
    """Index just past the first contiguous occurrence of `sub` in `seq`, else None."""
    if not sub:
        return None
    for i in range(len(seq) - len(sub) + 1):
        if tuple(seq[i:i + len(sub)]) == sub:
            return i + len(sub)
    return None


def _build_constrained_processor(repo: str, tokenizer, spec: GrammarSpec):
    """A stateful mlx-lm logits_processor implementing trigger-gated constrained
    decoding. Returns a fresh closure per generation (the Guide is stateful).

    Free until the model emits its family's tool-call opener (a single special
    token for Qwen/Llama), then masks each step to the grammar Guide over
    (JSON body + closer) so the emitted call is schema-valid AND well-wrapped.
    Once the Guide finishes, generation is free again (EOS follows naturally).
    If the opener never appears, generation stays free => abstention is reachable."""
    import mlx.core as mx
    from outlines_core import Guide

    regex = _schema_regex(spec)
    guide = Guide(_index(repo, regex))
    opener = tuple(tokenizer.encode(spec.family.call_open, add_special_tokens=False))
    st = {"base": None, "mode": "free", "fed": 0}

    def processor(tokens, logits):
        # mlx-lm passes cumulative tokens (incl. prompt) each step; calibrate once.
        if st["base"] is None:
            st["base"] = tokens.shape[-1]
            return logits
        gen = tokens[st["base"]:].tolist()
        if st["mode"] == "free":
            end = _subseq_end(gen, opener)
            if end is not None:
                st["mode"], st["fed"] = "constrained", end  # JSON body starts next
            elif len(gen) <= _BARE_JSON_LOOKAHEAD and tokenizer.decode(gen).lstrip().startswith("{"):
                # some families (Llama-3.2) emit the call as bare leading JSON with no
                # special opener token; constrain from the start so it stays schema-valid
                st["mode"], st["fed"] = "constrained", 0
        if st["mode"] == "constrained":
            for tid in gen[st["fed"]:]:
                if not guide.is_finished():
                    guide.advance(int(tid), return_tokens=False)
                st["fed"] += 1
            if guide.is_finished():
                st["mode"] = "done"
        if st["mode"] != "constrained" or guide.is_finished():
            return logits  # free / done: unmasked
        allowed = guide.get_tokens()
        mask = mx.full((logits.shape[-1],), -1e9, dtype=logits.dtype)
        mask[mx.array(allowed)] = 0.0
        return logits + mask

    return processor


def _schema_regex(spec: GrammarSpec) -> str:
    """Regex for the constrained region: optional leading whitespace + the
    discriminated-union JSON body + the family's closer literal. The closer keeps
    the Guide from stopping early (so parse_framework recognizes the call); the
    leading whitespace lets the JSON start on a whitespace token boundary (e.g.
    Qwen's newline after <tool_call>, or a leading space before a bare-JSON call)."""
    import json

    from outlines_core import json_schema
    body = json_schema.build_regex_from_schema(json.dumps(spec.json_schema()))
    return r"[ \t\r\n]*" + body + spec.family.call_close


def generate(repo: str, prompt: str, max_tokens: int = 256,
             grammar: GrammarSpec | None = None, temp: float = 0.0,
             seed: int | None = None) -> GenResult:
    import mlx.core as mx
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load_model(repo)
    if seed is not None:
        mx.random.seed(seed)
    sampler = make_sampler(temp=temp)
    processors = ([_build_constrained_processor(repo, tokenizer, grammar)]
                  if grammar is not None else None)
    t0 = time.perf_counter()
    text = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                        sampler=sampler, logits_processors=processors)
    dt = time.perf_counter() - t0
    n_tok = len(tokenizer.encode(text))
    peak_mb = mx.get_peak_memory() / (1024 * 1024)
    return GenResult(text=text, ttft_s=dt, tok_per_s=(n_tok / dt if dt > 0 else 0.0),
                     peak_mem_mb=peak_mb)
