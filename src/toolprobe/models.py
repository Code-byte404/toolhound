"""Pydantic models for cases, tools, and parsed tool calls."""
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel


class Turn(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str | None = None
    tool_call: dict | None = None  # prior assistant call in multi-turn cases


class ExpectedCall(BaseModel):
    tool: str
    args: dict[str, Any]


class ToolCall(BaseModel):
    """A tool call extracted from raw model output by a parser."""
    tool: str
    args: dict[str, Any]


class Case(BaseModel):
    id: str
    cat: str
    tools: list[str]
    turns: list[Turn]
    expected: ExpectedCall | None = None  # None = model should abstain
    arg_rules: dict[str, dict[str, Any]] = {}
    n_tools: int | None = None
    note: str | None = None
    split: Literal["dev", "test"] | None = None
    family: str | None = None


def load_cases(path: str | Path) -> list[Case]:
    with open(path) as f:
        return [Case(**json.loads(line)) for line in f if line.strip()]


def load_tools(path: str | Path) -> dict[str, dict]:
    """tools.yaml -> {name: OpenAI-style function dict} for apply_chat_template(tools=...)."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {
        name: {
            "type": "function",
            "function": {
                "name": name,
                "description": spec.get("description", ""),
                "parameters": spec["parameters"],
            },
        }
        for name, spec in raw.items()
    }
