# Contributing to Toolhound 🐾🔍

First off — thank you. Toolhound is a young project with a sharp mission (honest,
reproducible measurement of tool-calling reliability for small models on Apple
Silicon) and a lot of **well-scoped, self-contained** ways to help. This guide gets
you from clone to merged PR.

By contributing, you agree your work is licensed under the project's
[Apache-2.0 License](LICENSE).

## Ways to help

| | Contribution | Difficulty |
|---|---|---|
| 🧪 | **Add test cases** — especially abstention *traps* (utterances that look like tool requests but aren't) | good first issue |
| 📖 | **Docs** — help port the methodology notes to English | good first issue |
| 🧩 | **Add a model** to the registry, and file the template/parser bugs Toolhound finds upstream | moderate |
| 📊 | **Add a method** — wire an existing zero-training tool-calling fix into `methods/` | advanced |
| 🔬 | **Implement the constrained-decoding seam** (`backend.generate(grammar=...)` is a reserved stub) | advanced |

👉 Browse the [**good first issues**](https://github.com/Code-byte404/toolhound/labels/good%20first%20issue) for concrete starting points.

## Dev setup

Toolhound's **logic layer runs anywhere**; the **end-to-end (MLX) layer needs an
Apple Silicon Mac** (M1+), macOS 14+, Python 3.11+. Your conda env must be **arm64**
(`conda info` → platform `osx-arm64`), not Rosetta.

```bash
git clone https://github.com/Code-byte404/toolhound.git
cd toolhound
conda create -n toolhound python=3.11 && conda activate toolhound
pip install -e ".[dev]"

pytest                 # logic tests — run anywhere, no MLX needed
pytest -m mlx          # end-to-end tests — Apple Silicon only
python scripts/smoke.py
ruff check .
```

## The bar (please keep it green)

`ruff check . && pytest` must pass — CI runs both on every PR (logic tests only; MLX
wheels aren't available on Linux runners).

- **Parser / scorer / attribution are pure functions.** Every change needs a fixture
  in `tests/fixtures/` plus a test. This is the layer we lock down hardest — one
  fixture per attribution branch.
- Anything touching a real model is marked `@pytest.mark.mlx` (excluded by default).
- **Any claimed improvement must show non-overlapping 95% bootstrap CIs vs. baseline**
  on the held-out `cases/test.jsonl`. A directional delta is *not* enough — the harness
  flags a result `credible` only when its CI is disjoint from baseline's.

## Hard rules (these keep the results valid)

Breaking any of these invalidates the project's conclusions, so PRs that touch them
get extra scrutiny:

- **`backend.py` isolation** — it is the *only* module allowed to import `mlx` /
  `mlx_lm` / `outlines` / `xgrammar`. A hygiene test (`tests/test_hygiene.py`) enforces
  this. Verify library call signatures against the installed version; never write them
  from memory.
- **Fair prompt** — every model is tested with **its own** chat template's tool-call
  format (`apply_chat_template(tools=...)`), never a hand-rolled one.
- **Parser is lenient ("宽进"), scorer is strict ("严判")** — keep them decoupled. A method
  that "fixes" the model by loosening the scorer is cheating the harness, not fixing the
  model. Renaming methods must *adapt-then-canonicalize* so the scorer stays frozen.
- **Args are never bare string exact-match** — use `arg_rules` (`equiv` / `match: set` /
  `match: semantic` / `normalize: iso8601_minute`).
- **bf16 vs. q4** comparisons must pass `assert_same_template` first, so quantization
  damage is never confounded with template differences.
- **Dev/test discipline** — tune/select methods on `cases/dev.jsonl`, report on the
  held-out `cases/test.jsonl`. Never select a method on test.
- **Determinism** — `temperature=0`, fixed seed, fixed injected date; CIs come from
  bootstrap resampling over the case set, never from seed variance.

## Adding things

- **A model:** add it to `cli.MODELS` in `src/toolprobe/cli.py` (needs: non-gated on HF,
  bf16 & q4 share a chat template, tokenizer supports `tools=`). If it's a new family,
  extend `parser.py` for its native wrapper/keys — **with a fixture + test**.
- **Cases:** edit `cases/templates.yaml` / `cases/slots.yaml` / `cases/handwritten/`,
  then regenerate with `python scripts/gen_cases.py` and validate with
  `python scripts/validate_cases.py`. Don't hand-edit the generated `*.jsonl`.
- **A method:** implement the `Method.prepare(...) -> MethodResult` seam in `methods/`.
  It may only transform how the tool catalog is *presented* — the scorer must stay frozen.

## Pull requests

- Branch from `main`; use conventional commits (`feat:` / `fix:` / `test:` / `docs:` / `chore:`).
- Keep PRs focused and include tests + fixtures for any logic change.
- If Toolhound surfaced an **upstream** bug (chat template / framework parser), open an
  issue on the upstream repo (`mlx-lm` or the model) and link it — that's one of the most
  valuable things you can do here.

## Questions

Open an issue, or email **frankfish1984@gmail.com**. Happy hunting. 🕵️
