# Toolhound methodology

Toolhound is a diagnostic harness for tool-calling reliability. Its core job is not just
to report whether a model produced the right tool call, but to attribute each failure to
the layer that can actually fix it: the chat template, the framework parser, or the
model.

This page summarizes the methodology behind that attribution. It is the English version
of the design rationale already reflected in the README, CONTRIBUTING guide, tests, and
core modules.

## What Toolhound Measures

Every case has a tool catalog, a rendered prompt, raw model output, and an expected
result. The expected result is either:

- a tool name plus arguments, for cases where the assistant should call a tool
- `null`, for abstention cases where the assistant should not call any tool

Reliability is reported as layered metrics:

- `parse_ok`: a tool call could be extracted
- `schema_valid`: the extracted tool call matches the declared tool schema
- `tool_correct`: the selected tool is the expected one
- `args_correct`: the arguments match the expected arguments under the case's
  `arg_rules`

For abstention cases, the question is different: no call is correct, and any emitted call
is a false trigger.

## The Four Failure Causes

Toolhound pins each failing case to one of four causes:

| Cause | Meaning | Owner |
|---|---|---|
| `framework_template_bug` | The official chat template or tokenizer mangles the tool prompt before the model gets a fair input. | Upstream template or tokenizer |
| `framework_parser_gap` | The model emitted a rescuable correct call, but the framework parser missed it. | Upstream parser |
| `model_format_failure` | The prompt is sane, but the model output cannot be rescued into a tool call. | Model |
| `model_decision_failure` | A valid or rescuable call exists, but it chooses the wrong tool or wrong arguments. | Model or prompting strategy |

The point is to keep fixable upstream failures from being counted as model judgment
failures. A template bug should become an upstream repro. A parser gap should become a
parser repro. A model format failure and a model decision failure require different
model-side fixes.

## Lenient Parser, Strict Scorer

The attribution is valid because parsing and scoring answer separate questions.

The parser asks: **could a reasonable tool parser recover a call from this output?**

Toolhound uses two parser tiers:

- `parse_framework` accepts native framework shapes only, such as `<tool_call>...</tool_call>`,
  Llama's `<|python_tag|>` JSON tail, or a whole-output JSON object with canonical
  `arguments` or `parameters`.
- `parse_rescue` tries progressively more tolerant recovery, including fenced JSON,
  `[TOOL_CALLS]` prefixes, common `function` / `tool` / `args` aliases, JSON strings
  inside the argument field, balanced JSON objects embedded in prose, single-quote
  pseudo-JSON, trailing commas, and doubled braces from bad template examples.

The scorer asks: **is the recovered call actually correct?**

The scorer stays strict:

- the tool must exist in the declared catalog
- required arguments must be present
- argument names and simple JSON types must match the schema
- the selected tool must match the expected tool
- arguments must match through explicit `arg_rules`

This split prevents two common mistakes:

- A parser gap is not blamed on the model if the rescue parser can recover a correct
  call that the framework parser missed.
- A bad decision is not forgiven just because a lenient parser can read the output.

## Attribution Flow

For a case that expects a tool call, Toolhound applies the same staged decision:

1. Run the model with its rendered prompt and collect raw output.
2. Try the framework parser. If it finds a call, score that call strictly.
3. If the framework parser finds no call, run `template_sanity` on the model's official
   template. If the template cannot render or round-trip a minimal tool call, attribute
   the failure to `framework_template_bug`.
4. If the template is sane, try the rescue parser. If the rescued call scores as fully
   correct, attribute the failure to `framework_parser_gap`.
5. If rescue finds a call but strict scoring says it is wrong, attribute the failure to
   `model_decision_failure`.
6. If no rescue parser can recover a call, attribute the failure to
   `model_format_failure`.

For abstention cases, any parsed or rescued call is a `model_decision_failure`; no call
is a pass.

Attribution is reported under both strict and lenient rescue tiers so the conclusion does
not depend on one hidden parser setting.

## Bootstrap Confidence Intervals

Toolhound uses deterministic generation for ordinary evaluation: `temperature=0`, fixed
seed where sampling is used, and a fixed injected date for relative-date prompts.

Confidence intervals come from the case set, not from repeated model seeds. For each
metric, Toolhound stores a per-case success/failure vector, resamples that vector with
replacement, and reports the point estimate plus a 95% bootstrap interval.

For method comparisons, a directional delta is not enough. A method is marked
`credible` only when its interval is disjoint from the baseline interval and its point
estimate is higher.

## Dev/Test Discipline

Method selection happens on `cases/dev.jsonl`. Final claims are reported on the held-out
`cases/test.jsonl`.

The case generator and validator enforce the split discipline:

- generated cases come from `cases/templates.yaml` and `cases/slots.yaml`
- dev and test slots must not leak whole bindings
- concrete entity values such as locations, time zones, symbols, topics, and emails must
  not cross from dev into test
- identical user utterances cannot appear in both splits
- category distributions and C7 catalog sizes must stay balanced enough to keep the
  benchmark meaningful
- `arg_rules` must use one of the frozen scorer shapes: `equiv`, `match: set`,
  `match: semantic`, or `normalize: iso8601_minute`

This is why a method can improve on dev but still fail to become a claim: the improvement
has to survive the held-out test split and the confidence interval rule.

## Fair Prompt Rule

Toolhound does not hand-roll a universal tool prompt. Each model is tested with its own
official chat template through `tokenizer.apply_chat_template(..., tools=...)`.

That rule matters because a model should not be blamed for failing a tool format it was
never trained to see. It also means template failures are measurable: `template_sanity`
checks that a minimal tool call is rendered, that tokenization and detokenization
preserve the tool name, and that template examples are not malformed.

The same fairness rule applies to quantization comparisons. bf16 and q4 runs must pass
`assert_same_template` first, so any observed delta is not confounded with a different
chat template.

## Implementation Boundaries

The logic layer is kept pure and testable:

- parser, scorer, attribution, reporting, case generation, and case validation do not
  need MLX
- `backend.py` is the only module allowed to import `mlx` or `mlx_lm`
- hygiene tests enforce that boundary
- model-facing tests are marked separately for Apple Silicon / MLX environments

That boundary lets contributors work on parser, scorer, reporting, docs, and case logic
on any machine while preserving the Apple Silicon end-to-end path for model runs.

## Claims Toolhound Allows

Toolhound can support claims like:

- this model failed because the chat template rendered a broken tool prompt
- this output is an upstream parser gap because a reasonable rescue parser recovered the
  correct call
- this model can format calls but often chooses wrong arguments
- this method improved a metric only if the held-out test interval is disjoint from the
  baseline interval

Toolhound does not support claims based on a single accuracy number, a dev-only gain, a
hand-rolled prompt format, or a scorer that was loosened to make a method look better.
