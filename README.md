<p align="center">
  <img src="assets/logo.svg" alt="Toolhound" width="168" height="168">
</p>

<h1 align="center">Toolhound</h1>

<p align="center">
  <b>The tool-call detective for small models on Apple Silicon.</b><br>
  When a small model botches a tool call, Toolhound tells you <i>who did it</i> —
  the chat template, the framework parser, or the model itself.
</p>

<p align="center">
  <a href="#quickstart"><img src="https://img.shields.io/badge/Apple%20Silicon-MLX%20native-000000?logo=apple&logoColor=white" alt="MLX native"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-124%20passing-2ea44f" alt="tests">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="license">
  <img src="https://img.shields.io/badge/status-v2%20%C2%B7%20pre--release-orange" alt="status">
</p>

<p align="center">
  <img src="assets/result-card.png" width="840"
       alt="Toolhound attributes every tool-call failure to one of four causes — a chat-template bug, a framework parser gap, a model format failure, or a model decision failure — separating the model's fault from the framework's.">
</p>

---

## The 60-second pitch

Everyone benchmarks tool-calling with a single number: *"Model X gets 71% of function calls right."*
That number is a **lie of omission**. It can't tell you *why* the other 29% failed — and the *why* is
the only thing that tells you what to do next.

Three models can *look* like they fail tool-calling for the same reason — and be wrong three different ways.
Here's what the **2026-07 lineup** actually surfaced (real numbers below):

| Model | Its "failure" looks like… | …but the real culprit is | So the fix is… |
|---|---|---|---|
| **Granite-3.3-2B** | 🧩 can't do multi-turn tool calls | the **chat template** silently drops the prior call | **File upstream** / faithful rendering |
| **Gemma-4-12B** | 🔧 emits schema-invalid calls | the **harness parser** missed its array syntax | **Fix the parser** — not the model |
| **Qwen3.5-2B** | 🧠 picks the wrong arguments | genuine **model judgment** (args 0.76) | **Better model or prompt** — grammar can't fix this |

> Two of those three are *not the model's fault.* A plain accuracy score would have blamed the model for all three.
> **Toolhound doesn't** — it caught a template bug and a parser gap, and only pinned the third on the model.

Toolhound is a **reproducible diagnostic harness** that runs entirely on your Mac (via
[MLX](https://github.com/ml-explore/mlx)) and attributes **every single failure** to one of four causes —
with bootstrap confidence intervals on every metric.

---

## The four suspects 🔍

Every failed tool call gets pinned on exactly one culprit:

| Cause | Whose fault | Reportable? |
|---|---|---|
| `framework_template_bug` | The chat template / tokenizer mangled the tool tokens | ✅ Upstream bug |
| `framework_parser_gap` | The model emitted a rescuable call; the framework parser missed it | ✅ Upstream bug |
| `model_format_failure` | The model can't emit a parseable call at all | The model |
| `model_decision_failure` | Valid format, but wrong tool or wrong arguments | The model |

The trick that makes this attribution *valid*: **the parser is lenient ("宽进"), the scorer is strict ("严判").**
We decouple *"could any reasonable parser rescue this output?"* from *"is this the correct answer?"* — so a
format failure is never confused with a judgment failure, and an upstream parser gap is never blamed on the model.

---

## Why this exists (and what it is *not*)

Toolhound is a **measuring stick**, not another schema-adaptation method. Its value is *honest measurement*:

- ✅ It **finds** chat-template bugs and parser gaps — and gives you a minimal repro to file upstream.
- ✅ It **separates** "the model can't format" from "the model can't decide" — because grammar-constrained
  decoding fixes the first and can *never* fix the second.
- ✅ It **quantifies** quantization damage (bf16 vs. q4) *without* confounding it with template differences.
- ✅ In v2, it **benchmarks existing zero-training fixes** ([PA-Tool](#roadmap), grammar-constrained decoding)
  on a held-out test set — never claiming an improvement unless its confidence interval is disjoint from baseline.
  The measuring stick already earned its keep: it caught that PA-Tool *hurts* small models, and confirmed that
  constrained decoding credibly helps — but **only where the failure is syntactic** (see below).

We are **not** the first to notice that chat templates break tool tokens, and we don't claim to be. Toolhound's
contribution is making that failure *legible, attributable, and reproducible* on consumer Apple hardware.

---

## Quickstart

> **Requires an Apple Silicon Mac** (M1 or newer), macOS 14+, Python 3.11+. MLX runs *only* on Apple Silicon —
> your conda env must be **arm64**, not Rosetta (`conda info` → platform should read `osx-arm64`).

```bash
git clone https://github.com/Code-byte404/toolhound.git
cd toolhound

conda create -n toolhound python=3.11 && conda activate toolhound
pip install -e ".[dev]"

# Smoke test: MLX loads a tiny non-gated model and generates one call
python scripts/smoke.py
```

Then run the detective on a model:

```bash
# 1) Reliability report: how often does each model get tool calls right?
toolprobe run \
  --model qwen3.5-2b,granite-3.3-2b,gemma4-12b \
  --quant q4 \
  --cases cases/test.jsonl \
  --out reports/

# 2) Attribution: for every failure, name the culprit (run under strict + lenient parsers)
toolprobe attribute --model qwen3.5-2b

# 3) Compare a zero-training fix against baseline (v2)
toolprobe run --model qwen3.5-2b --cases cases/dev.jsonl --method baseline,pa_tool

# 4) Free vs. grammar-constrained decoding — adds a "Decode comparison" table (v2)
#    (validated on the reference JSON-family lineup; see the constrained-decoding section)
toolprobe run --model qwen2.5-0.5b --cases cases/dev.jsonl --decode free,constrained
```

Both commands write matching `*.json` (machine-readable) and `*.md` (human-readable) reports into `reports/`,
each stamped with a full reproducibility header: chip, RAM, macOS version, exact `mlx` / `mlx-lm` / `mlx-vlm` versions,
model repo + revision, and the injected date.

*(The bundled model keys — `qwen3.5-2b`, `granite-3.3-2b`, `gemma4-12b` (2026-07 refresh) — are registered in
`src/toolprobe/cli.py`; each family has a distinct native tool-call format. Add your own there.)*

---

## What a report looks like

**Reliability** — layered scoring, so you see exactly where each model drops off:

```
Model: qwen3.5-2b  (q4)                       95% bootstrap CI · Apple M2 Pro
─────────────────────────────────────────────────────────────────────────
parse_ok          ███████████████████░  0.99  [0.98, 1.00]
schema_valid      ███████████████████░  0.99  [0.98, 1.00]
tool_correct      ███████████████████░  0.99  [0.98, 1.00]
args_correct      ███████████████░░░░░  0.76  [0.69, 0.83]   ← syntax solved, judgment isn't
```

The **current (2026-07) lineup** on held-out `test.jsonl` (q4, free decoding) shows the layered story
cleanly — the newer models have *solved the syntax layer*, so the only differentiator left is judgment:

```
model            parse_ok   schema_valid  tool_correct  args_correct   (held-out test.jsonl, q4)
qwen3.5-2b         0.99         0.99          0.99          0.76        ← syntax solved; decision-bound
granite-3.3-2b     1.00         1.00          1.00          0.75        ← syntax solved; decision-bound
gemma4-12b         1.00         1.00          1.00          0.97        ← the 12B also nails the decisions
```

Three families, three *distinct* native formats (Qwen3.5 XML · Granite JSON-list · Gemma-4 VLM), and every
one drives `parse/schema/tool` to ≈1.0 — which is exactly *why* grammar-constrained decoding has no room to help
here (see below). The full report, with bootstrap CIs, is in
[`reports/`](reports/run-qwen3.5-2b+granite-3.3-2b+gemma4-12b.md).

**Attribution** — every failure pinned to a suspect, shown under *both* parser tiers so you can see the
conclusion doesn't flip when the parser gets more lenient:

```
Failure attribution (strict parser)           Failure attribution (lenient parser)
─────────────────────────────────             ─────────────────────────────────
framework_template_bug     4                   framework_template_bug     4
framework_parser_gap       6   ← rescuable!    framework_parser_gap       1
model_format_failure       9                   model_format_failure       8
model_decision_failure    22                   model_decision_failure    22
```

*(Reliability numbers above are from a real bundled 3-model run on an Apple M2 Pro; the attribution
counts show the two-tier layout — run `toolprobe attribute` for your own CI-backed figures.)*

### Does a proposed fix actually help? (v2)

When you benchmark a method against baseline, Toolhound doesn't just print a delta — it flags each metric
`credible` **only** when the method's bootstrap CI is *disjoint* from baseline's. In a bundled 3-model demo
run, [PA-Tool](#roadmap) (a real zero-training tool-renaming method) didn't clear that bar on any metric — and
on one model it measurably *hurt* argument accuracy:

```
Method comparison — qwen2.5-1.5b (q4) · pa_tool vs. baseline
────────────────────────────────────────────────────────────
metric          baseline   pa_tool    delta    credible
tool_correct      0.96       0.96      +0.00      no
args_correct      0.71       0.43      −0.28      no   ← caught, not rubber-stamped
```

That's the entire point of a measuring stick: **it tells you when a fix *doesn't* work, with the statistics to
back it up.** *(Demonstration on the exploratory `default.jsonl`; real method selection uses the held-out
`dev` / `test` split so a gain has to generalize to unseen slots.)*

### And when a fix *does* work — grammar-constrained decoding (v2)

Toolhound runs a second axis, `--decode free,constrained`, that masks generation to a model's own
tool-call grammar via [`outlines-core`](https://github.com/dottxt-ai/outlines-core) — trigger-gated, so a model
can still decline to call a tool (abstention survives). Selected on the `dev` split, then reported **once** on
held-out `test.jsonl`, it was the first integrated fix to clear the credible bar — on **2 of 3** models of the
**reference JSON-family lineup** (Qwen2.x / Llama-3.x):

```
Constrained vs free decoding — reference lineup, held-out test.jsonl (q4) · Apple M2 Pro
──────────────────────────────────────────────────────────────────────────
model          metric          free → constrained    credible
qwen2.5-0.5b   parse/schema    0.50 → 0.76  (+0.27)    yes    ← format-bound: grammar rescues it
qwen2.5-0.5b   args_correct    0.31 → 0.52  (+0.21)    yes
llama-3.2-3b   schema_valid    0.68 → 1.00  (+0.32)    yes    ← emits invalid arg types; grammar forbids them
qwen2.5-1.5b   (every metric)  no credible change      no     ← already well-formed; its errors are judgment
```

The **null result on the 1.5B is the whole point.** Constrained decoding fixes the *syntax* layer (is it a
parseable, schema-valid call?) and can **never** fix the *decision* layer (is it the right tool with the right
values?). So it credibly helps the two models whose failures are syntactic, and does nothing for the model that
already emits clean calls and just picks wrong. Toolhound is what lets you tell those two situations apart — and
it even surfaces the honest cost (constrained decoding can degenerate into repetition on unbounded string fields).

> **Why the current (2026-07) lineup is free-decoding only.** When we refreshed the registry to newer models
> (`qwen3.5-2b`, `granite-3.3-2b`, `gemma4-12b`), Toolhound *measured* that none of them has syntactic headroom for
> constrained decoding to fix: Qwen3.5's syntax layer is already saturated (its misses are all judgment), Granite's
> only failures were a **multi-turn template bug** (see below — the fix is rendering, not grammar), and Gemma is a
> VLM with no logits hook to constrain. So `--decode constrained` now *fails fast* on the current models by design —
> the honest finding is that **newer 2B+ models have outgrown the format failures constrained decoding was built for.**
> The capability stays implemented and tested for when a format-bound model is added back.

### Field notes: two bugs the 2026-07 refresh caught 🐛

Swapping in three brand-new models immediately flushed out two bugs — and the useful part is *neither was the
model's fault*. Each maps onto one of the four suspects, which is exactly what Toolhound exists to prove.

**① `framework_template_bug` — Granite's chat template drops multi-turn tool calls.** Granite 3.3 was *perfect* on
every single-turn category but cratered on multi-turn cases, all with the same mangled shape
`<|tool_call|>tool[{…args…}]`. The cause was not the model: Granite's chat template has no `tool_calls` branch, so
it silently **dropped the prior assistant tool-call** from the history — the model was fed a tool *result* with no
preceding *call*. The fix is faithful rendering (serialize the prior call in Granite's own native format), **not** a
smarter model or a grammar:

```
Granite 3.3 · 16 affected multi-turn cases (test.jsonl, q4)   before → after the rendering fix
parse_ok        0.38  →  1.00        (6/16 → 16/16)
tool_correct    0.38  →  1.00
args_correct    0.00  →  0.94        (0/16 → 15/16)   ← broken context made every call wrong
```

**② `framework_parser_gap` — Toolhound's *own* parser missed Gemma's array syntax.** Gemma-4 looked like it emitted
15% schema-invalid calls. On inspection the calls were *fine* — Gemma writes list arguments as
`attendees:[<|"|>a<|"|>,<|"|>b<|"|>]`, and Toolhound's Gemma parser was truncating the list at the first comma. That
is textbook `framework_parser_gap` (a rescuable call the parser fumbled) — the harness caught it *in itself*, and the
honest move was to fix the parser, not dock the model:

```
Gemma-4-12B · full test.jsonl, 131 non-abstention (q4)        before → after the parser fix
schema_valid    0.85  →  1.00        (112/131 → 131/131)
args_correct    0.85  →  0.97        (111/130 → 127/131)
```

**That is the whole value proposition, twice.** A plain accuracy score would have quietly blamed Granite and Gemma
for **17** failures that belonged to a chat template and a parser gap. Toolhound's layered metrics flagged the
discrepancy, and its four-cause discipline forced the honest attribution — *fix the harness / file it upstream*, not
*"the model is bad."*

---

## How the attribution works

```
                       ┌─────────────────────────┐
   raw model output →  │  template_sanity check   │  tokens survived round-trip?
                       └───────────┬──────────────┘
                          no │             │ yes
                             ▼             ▼
              framework_template_bug   ┌───────────────────────────┐
                                       │  parse_framework (strict) │  did the framework see a call?
                                       └───────┬───────────────────┘
                                          no   │           │ yes
                                               ▼           ▼
                              ┌────────────────────────┐   scorer (strict):
                              │  parse_rescue (lenient) │   right tool? right args?
                              └────┬───────────────┬───┘        │
                             rescued│          garbage│         ▼
                                    ▼               ▼    model_decision_failure
                          framework_parser_gap   model_format_failure
```

The pipeline is a clean, testable data flow — each stage is a pure function with one job:

```
case → templates → runner → parser → scorer → attribution → report
```

Only **one** module (`backend.py`) is allowed to import `mlx` / `mlx_lm` / `mlx_vlm`; a hygiene test enforces it.
That keeps the parser, scorer, attribution, and grammar-builder logic 100% pure and unit-testable on *any* machine
(no Mac required for the logic tier — 124 tests run in <1s in CI; the constrained-decoding e2e tests are the
Mac-only tier).

---

## Roadmap

- [x] **v1** — diagnostic harness + four-cause attribution + bootstrap CIs
- [x] **v2 (in progress)** — 304-case dev/test dataset with slot-disjoint splits; `PA-Tool` method integration
- [x] **Grammar-constrained decoding** — abstention-safe, trigger-gated `--decode free,constrained` via
  `outlines-core` (torch-free); first fix to clear the credible bar on held-out `test.jsonl` (2 of 3 models)
- [x] **Model refresh (2026-07)** — registry now spans three current families with three *distinct* native
  formats: Qwen3.5 (XML), Granite 3.3 (JSON list), Gemma-4 (VLM via `mlx-vlm`, bespoke format)
- [ ] **More methods** — TSCG (integrate & measure existing work — *not* invent new)
- [ ] **A constrained-decoding string guard** — bounded-length / repetition-penalized string fields, to remove the
  degeneration cost the harness surfaced on unbounded string args
- [ ] **PNG report export** for dropping straight into issues and blog posts

---

## Contributing — we want detectives 🕵️

This is a young project with a clear mission and a lot of **well-scoped, self-contained** ways to help. If any
of these sound fun, open an issue and say hi:

- **🧩 Add a model** to the registry and file the template/parser bugs Toolhound finds upstream.
- **🧪 Add test cases** — especially tricky abstention traps (utterances that *look* like tool requests but aren't).
- **🔬 Harden constrained decoding** — bound string-field length / add a repetition penalty so unbounded string
  args (e.g. `translate.target_lang`) can't degenerate into a repetition loop under the grammar.
- **📊 Add a method** — wire an existing zero-training tool-calling fix into the `methods/` framework and let the
  benchmark judge it fairly.
- **📖 Docs** — help port the methodology notes to English.

Every metric in this project has a confidence interval and every logic path has a test. Please keep it that way —
`ruff check . && pytest` is the bar, and any claimed improvement must show **non-overlapping CIs vs. baseline**.

**New here?** → **[CONTRIBUTING.md](CONTRIBUTING.md)** walks you from clone to merged PR (dev setup, the hard
rules, the bar), and the [good first issues](https://github.com/Code-byte404/toolhound/labels/good%20first%20issue)
are concrete places to start.

---

## Reproducibility, by design

- `temperature=0`, `top_p=1`, fixed seed — deterministic generation.
- Confidence intervals come from **bootstrap resampling over the case set**, not seed variance.
- bf16 vs. q4 comparisons **assert an identical tokenizer + chat template** first, so quantization damage is
  never confounded with template differences.
- Every model is tested with **its own** chat template's tool format — never a hand-rolled one.
- Dates like "today / Friday" are pinned to a fixed injected date so runs are reproducible forever.

The whole report table is re-runnable with a single command.

---

## Contact & collaboration

This project is actively looking for collaborators. Whether you want to add a model, contribute test cases,
port a method, or just compare notes on small-model tool-calling reliability — I'd love to hear from you.

- 🐛 **Issues & ideas:** open a [GitHub issue](https://github.com/Code-byte404/toolhound/issues)
- ✉️ **Reach the maintainer:** [frankfish1984@gmail.com](mailto:frankfish1984@gmail.com)

## License

Released under the **Apache License 2.0**. See [`LICENSE`](LICENSE).

## Acknowledgements

Built on [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple.
Method integrations credit their original authors (see each file in `src/toolprobe/methods/`).

<p align="center"><sub>Toolhound ships as the <code>mlx-toolprobe</code> package with the <code>toolprobe</code> CLI. 🐾</sub></p>
