"""Constrained-decoding grammar builder. PURE -- no mlx/outlines/xgrammar import
(enforced by tests/test_hygiene.py). Returns a backend-agnostic GrammarSpec whose
`.json_schema()` is a discriminated union over the case's tools; backend.py compiles
that into an outlines_core logits processor.

Constrained decoding is *trigger-gated* (design doc §4.4, line 145): only the JSON
BODY is schema-constrained, and only after the model emits its family's tool-call
opener. Before the opener the model is free, so abstention ("no call / natural-language
reply") stays reachable -- otherwise C5 abstention would be forced to 0. The family
opener/closer literals live in WireFamily and drive that trigger detection in backend.py;
they mirror the native formats parser.py already recognizes."""
from dataclasses import dataclass


@dataclass(frozen=True)
class WireFamily:
    """A model family's native tool-call wire format (fair-prompt principle)."""
    key: str          # "qwen" | "llama"
    call_open: str    # opener literal the model emits to start a call
    call_close: str   # closer literal ("" when the family has none, e.g. Llama)
    args_key: str     # JSON key holding the args: "arguments" (Qwen) | "parameters" (Llama)


# Qwen2.5: <tool_call>\n{"name":..,"arguments":{..}}\n</tool_call>
# Llama-3.2: <|python_tag|>{"name":..,"parameters":{..}}  (single special token, no closer)
FAMILIES: dict[str, WireFamily] = {
    "qwen": WireFamily("qwen", "<tool_call>", "</tool_call>", "arguments"),
    "llama": WireFamily("llama", "<|python_tag|>", "", "parameters"),
}


@dataclass
class ToolChoice:
    name: str
    params_schema: dict  # tools[name]["function"]["parameters"], verbatim


@dataclass
class GrammarSpec:
    family: WireFamily
    tools: tuple[ToolChoice, ...]
    allow_abstention: bool = True

    def json_schema(self) -> dict:
        """Discriminated union: name is a const per tool, args under the family's
        key conform to that tool's parameter schema (required/enum/type preserved)."""
        return {"oneOf": [
            {"type": "object",
             "properties": {"name": {"const": tc.name},
                            self.family.args_key: tc.params_schema},
             "required": ["name", self.family.args_key]}
            for tc in self.tools]}


def detect_family(repo: str) -> str:
    """Map an HF repo id to a wire family. Substring match over the registered
    families; raises on anything unrecognized rather than guessing a format."""
    r = repo.lower()
    for key in FAMILIES:
        if key in r:
            return key
    raise ValueError(
        f"no wire family for repo {repo!r}; register it in grammar.FAMILIES "
        f"(known: {', '.join(FAMILIES)})")


def build_grammar(family_key: str, case_tools: list[str], tools: dict[str, dict],
                  *, allow_abstention: bool = True) -> GrammarSpec:
    """Build the grammar for one case: the family wrapper + a discriminated union
    restricted to `case_tools` (so the model can only name a tool the case offers).

    `tools` is keyed by CANONICAL name (matching `case_tools`) but its values carry
    the PRESENTED name/params (a method like PA-Tool may have renamed them). The
    grammar constrains to what the model actually emits -- the presented name and
    param schema -- so the caller's `canonicalize` maps it back before scoring.
    For the baseline, presented == canonical, so this is a no-op."""
    family = FAMILIES[family_key]
    choices = tuple(
        ToolChoice(name=tools[t]["function"]["name"],
                   params_schema=tools[t]["function"]["parameters"])
        for t in case_tools)
    return GrammarSpec(family=family, tools=choices, allow_abstention=allow_abstention)
