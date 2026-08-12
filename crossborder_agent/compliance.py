"""Versioned conservative content checks for model-authored material."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


RULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "rules"
    / "aliexpress_content_policy_v1.json"
)


@lru_cache(maxsize=1)
def load_rules() -> dict[str, Any]:
    with RULE_PATH.open("r", encoding="utf-8") as handle:
        rules = json.load(handle)
    if not isinstance(rules, dict) or not rules.get("version"):
        raise ValueError("内容合规规则包无效")
    return rules


def flatten_generated_text(value: Any) -> str:
    if isinstance(value, dict):
        return "\n".join(flatten_generated_text(item) for item in value.values())
    if isinstance(value, list):
        return "\n".join(flatten_generated_text(item) for item in value)
    return value if isinstance(value, str) else ""


def generated_copy_violations(language: str, payload: Any) -> list[str]:
    rules = load_rules()
    text = flatten_generated_text(payload).casefold()
    terms = rules.get("prohibited_claims", {}).get(language, [])
    return [str(term) for term in terms if str(term).casefold() in text]


def visual_prompt_violations(prompt: str) -> list[str]:
    rules = load_rules()
    text = prompt.casefold()
    return [
        str(term)
        for term in rules.get("visual_prompt_forbidden", [])
        if str(term).casefold() in text
    ]
