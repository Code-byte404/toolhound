# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**toolprobe** (`mlx-toolprobe`) — an MLX-native, reproducible diagnostic harness for tool-calling reliability of small (≤8B) models on Apple Silicon. Given `(model, quantization)`, it produces a reliability table (`parse_ok → schema_valid → tool_correct → args_correct`, each with bootstrap 95% CIs) and attributes every failure to one of four causes: `framework_template_bug` / `framework_parser_gap` (upstream's fault, reportable) vs `model_format_failure` / `model_decision_failure` (the model's fault).

Design docs (source of truth for intent, metrics, acceptance criteria) live one directory up: `../toolprobe-开发文档-v2.md` (why) and `../toolprobe-技术实现文档.md` (how).

## Commands

```bash
conda activate toolprobe                 # Python 3.11 env (arm64)
pip install -e ".[dev]"

pytest                                   # logic tests only (no MLX needed; default -m "not mlx")
pytest -m mlx                            # end-to-end tests, Apple Silicon + downloaded models
pytest tests/test_scorer.py -v           # single file
ruff check .

python scripts/smoke.py                  # bare MLX generation sanity check

# --model takes a comma-separated list; run emits one combined table
toolprobe run --model qwen2.5-0.5b,qwen2.5-1.5b,llama-3.2-3b --quant bf16,q4 --cases cases/default.jsonl
toolprobe attribute --model qwen2.5-0.5b,qwen2.5-1.5b,llama-3.2-3b # four-cause table per model, strict + lenient tiers
```

Registered models (`cli.MODELS`, all non-gated, bf16+q4 templates verified matching): `qwen2.5-0.5b`, `qwen2.5-1.5b`, `llama-3.2-3b`. Adding a model requires: non-gated on HF, bf16/q4 share a chat template, and the tokenizer supports `tools=` (Phi-3.5 was rejected for lacking native tool-template support).

Reports land in `reports/` (`.md` committed, `.json` gitignored).

## Architecture

Data flow: `case → templates.render → runner.run_free → parser → scorer → attribution → report`.

- `src/toolprobe/backend.py` — **the ONLY module allowed to import mlx/mlx_lm** (enforced by `tests/test_hygiene.py`). Verify mlx-lm call signatures against the installed version before changing (`pip show mlx-lm`, `inspect.signature`); APIs drift.
- `models.py` — pydantic `Case`/`Turn`/`ExpectedCall`/`ToolCall`; `load_cases` (jsonl), `load_tools` (yaml → OpenAI-style function dicts).
- `templates.py` — fair-prompt rendering via each model's own `apply_chat_template(tools=...)`; injects `FIXED_DATE = 2026-03-20` (Friday — gold datetimes in cases assume it); `template_sanity` gates the template-bug attribution branch.
- `parser.py` — `parse_framework` (strict, models what framework tooling accepts) + `parse_rescue` with **exactly defined** `strict`/`lenient` tiers. Parser leniency swings attribution, so both tiers' behavior is documented in the module docstring and attribution always reports both.
- `scorer.py` — strict layered judge. Args never compare by bare string equality: `arg_rules` (`equiv`/`match: set`/`match: semantic` (Jaccard ≥0.5, v1 stand-in for embeddings)/`normalize: iso8601_minute`). Abstention cases score `abstention_correct`/`false_trigger` only, never the layered metrics.
- `attribution.py` — the four-cause decision tree (design doc 附三) + `build_attribution` (bf16 vs q4 with `assert_same_template` confound guard, quant deltas, parser-gap repros for upstream issues).
- `report.py` — seeded bootstrap CIs (variance comes from the case set; temp=0 is deterministic, never "multi-seed"), rich tables, markdown with env/version reproducibility header.
- `cli.py` — `run`/`attribute` subcommands + `MODELS` registry (HF repo pairs; both quants must share a chat template).

Core principle: **parser is lenient ("宽进"), scorer is strict ("严判")** — their decoupling is what makes attribution valid.

## Testing Rules

- Logic modules (parser/scorer/attribution/report/models/templates-rendering) are pure and tested with synthetic fixtures in `tests/fixtures/` — one per attribution branch. These run anywhere, no MLX.
- Anything touching a real model is `@pytest.mark.mlx` (excluded by default via pytest.ini).
- TDD: failing test first, minimal fix, rerun. Attribution branches especially must stay locked by tests.

## Case set (v2)

Canonical files in `cases/`:
- `dev.jsonl` (152 cases) — **tune here**. Select and iterate on fixes (grammar, prompt tweaks, PA-Tool/TSCG variants) against this split.
- `test.jsonl` (152 cases) — **report here**, held out. Only run a fix against test once, after it's been picked on dev, to report its gain.
- `default.jsonl` (304 cases) — the dev+test union. Exploratory/baseline runs only (e.g. the 3-model v1 table below) — **never** used for method selection or as the basis of a reported improvement, since it mixes dev and test.
- `smoke.jsonl` — small fixed set for pipeline smoke checks (not part of the dev/test discipline).

**Dev/test discipline:** dev and test are generated from disjoint slot pools (see `cases/slots.yaml`), so a case in dev and its counterpart in test share a template/family but never the exact same entity values. This means a fix that overfits to specific dev slot values (rather than the underlying failure mode) won't transfer to test — generalization to unseen slots is the bar. Standard protocol: tune/select on `dev.jsonl`, report the resulting delta on `test.jsonl` with bootstrap CIs; a claimed improvement needs non-overlapping CIs on test, not just on dev.

Regenerate + validate:
```bash
python scripts/gen_cases.py        # templates.yaml + slots.yaml (+ handwritten/) -> dev/test/default.jsonl
python scripts/validate_cases.py   # schema, arg-rules, leakage, distribution checks
pytest tests/test_cases_valid.py   # same checks, enforced in plain CI
```
`toolprobe.casegen` (generation) and `toolprobe.caselint` (validation) are pure Python — no `mlx` import, run anywhere. Generated case files are **committed artifacts**: don't hand-edit `dev.jsonl`/`test.jsonl`/`default.jsonl` directly — edit `cases/templates.yaml`/`cases/slots.yaml`/`cases/handwritten/` and regenerate.

## Known Findings

**Model/template quirks the harness surfaced (don't "fix" the first as if it were ours):**
- Qwen2.5's chat template (upstream + mlx-community copies) renders its tool-call format example with doubled braces `{{"name": ...}}` — a template bug small models copy literally. `template_sanity` intentionally returns **False** for these repos (the `FRAMEWORK_TEMPLATE_BUG` classification working), and the lenient rescue tier dedupes doubled braces. See `tests/test_templates_mlx.py`.
- Llama-3.2 uses `parameters` (not `arguments`) as its args key and wraps calls in `<|python_tag|>` (not `<tool_call>`). Both are Llama's **canonical** formats, so `parse_framework` recognizes them — otherwise a capable model scores 0 purely from a family-format mismatch. If you add a new model family, check its native key/wrapper and extend `parser.py` (with a fixture + test) rather than loosening the scorer.

**First 3-model result (Qwen2.5-0.5B / 1.5B, Llama-3.2-3B; deterministic, reproducible):** each family has a *different* dominant failure cause — 0.5B is mostly `framework_template_bug`, 1.5B mostly `model_decision_failure` (outgrows the template bug), Llama entirely `model_decision_failure` (wrong arg types + abstention false-triggers). This cross-family contrast is the point of running ≥3 models; see `reports/`.

## Hard Rules (from the design docs — violating these invalidates results)

- Fair prompt: each model is tested with its own chat template's tool format; never a hand-rolled one.
- bf16 vs q4 comparisons must pass `assert_same_template` (only weight precision may differ).
- `temperature=0`, fixed injected date; CIs from bootstrap over cases with a seeded RNG.
- Attribution is always reported under both `strict` and `lenient` parser tiers.
- v2 scope (PA-Tool/TSCG integration) requires dev/test case-set separation; constrained decoding is a reserved seam (`backend.generate(grammar=...)` raises `NotImplementedError`).
