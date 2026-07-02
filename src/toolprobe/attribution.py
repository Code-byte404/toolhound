"""Four-cause failure attribution (design doc §4.2 / 附三). The soul of the
project: separates upstream's fault (template bug, parser gap) from the
model's fault (format, decision). Always produced under BOTH parser
leniencies to prove conclusions don't hinge on that free parameter."""
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from .backend import assert_same_template
from .models import Case
from .parser import parse_framework, parse_rescue
from .runner import run_free
from .scorer import passed, score
from .templates import template_sanity


class Cause(Enum):
    PASS = "pass"
    FRAMEWORK_TEMPLATE_BUG = "framework_template_bug"  # upstream's fault; report it
    FRAMEWORK_PARSER_GAP = "framework_parser_gap"      # upstream's fault; keep repro
    MODEL_FORMAT_FAILURE = "model_format_failure"
    MODEL_DECISION_FAILURE = "model_decision_failure"


@dataclass
class AttributionRecord:
    case_id: str
    cause: Cause
    raw: str


def attribute_case(repo: str, case: Case, tools: dict[str, dict],
                   leniency: str = "lenient") -> AttributionRecord:
    raw = run_free(repo, case, tools).text

    fw_call = parse_framework(raw)
    if case.expected is None:
        # abstention case: any parsed call (even rescued) is a false trigger
        call = fw_call or parse_rescue(raw, leniency)
        cause = Cause.PASS if call is None else Cause.MODEL_DECISION_FAILURE
        return AttributionRecord(case.id, cause, raw)

    if fw_call is not None:  # A: framework got structure
        s = score(fw_call, case.expected, case.arg_rules, tools)
        cause = Cause.PASS if passed(s, case.expected) else Cause.MODEL_DECISION_FAILURE
        return AttributionRecord(case.id, cause, raw)

    if not template_sanity(repo):  # B: even a minimal tool render is broken
        return AttributionRecord(case.id, Cause.FRAMEWORK_TEMPLATE_BUG, raw)

    rescue = parse_rescue(raw, leniency)  # C: can the lenient parser salvage it?
    if rescue is not None:
        s = score(rescue, case.expected, case.arg_rules, tools)
        cause = (Cause.FRAMEWORK_PARSER_GAP if passed(s, case.expected)
                 else Cause.MODEL_DECISION_FAILURE)
        return AttributionRecord(case.id, cause, raw)

    return AttributionRecord(case.id, Cause.MODEL_FORMAT_FAILURE, raw)  # D


def build_attribution(repo_bf16: str, repo_q4: str, cases: list[Case],
                      tools: dict[str, dict]) -> dict:
    assert_same_template(repo_bf16, repo_q4)  # quantization-confound guard
    out: dict = {}
    for leniency in ("strict", "lenient"):
        per_quant: dict[str, Counter] = {}
        gap_repros: list[AttributionRecord] = []
        for quant, repo in (("bf16", repo_bf16), ("q4", repo_q4)):
            records = [attribute_case(repo, c, tools, leniency) for c in cases]
            per_quant[quant] = Counter(r.cause for r in records)
            gap_repros += [r for r in records if r.cause == Cause.FRAMEWORK_PARSER_GAP]
        out[leniency] = {
            "per_quant": per_quant,
            "quant_delta": {c: per_quant["q4"][c] - per_quant["bf16"][c] for c in Cause},
            "parser_gap_repros": gap_repros,  # minimal repros for upstream issues
        }
    return out
