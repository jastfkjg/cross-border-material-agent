"""Versioned conservative content checks for model-authored material."""

from __future__ import annotations

import json
import re
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
    violations = [str(term) for term in terms if str(term).casefold() in text]
    original_text = flatten_generated_text(payload)
    for pattern in rules.get("generated_text_regex", []):
        try:
            if re.search(str(pattern), original_text):
                violations.append(f"regex:{pattern}")
        except re.error:
            continue
    return violations


def visual_prompt_violations(prompt: str) -> list[str]:
    rules = load_rules()
    text = prompt.casefold()
    return [
        str(term)
        for term in rules.get("visual_prompt_forbidden", [])
        if str(term).casefold() in text
    ]


def source_visual_risk_reasons(observation: dict[str, Any]) -> list[str]:
    """Return normalized machine-readable risk reasons for one source image."""

    rules = load_rules()
    reasons: list[str] = []
    for field in rules.get("source_visual_hard_risk_fields", []):
        if observation.get(str(field)) is True:
            reasons.append(str(field))
    supplied = observation.get("risk_reasons")
    if isinstance(supplied, list):
        for reason in supplied:
            cleaned = str(reason).strip()
            if cleaned and cleaned not in reasons:
                reasons.append(cleaned)
    return reasons


def normalize_source_image_observations(
    analysis: dict[str, Any], image_urls: list[str]
) -> list[dict[str, Any]]:
    """Bind model observations to the actual input order and fail conservatively.

    The model is allowed to omit an item, but it is never allowed to invent a URL or
    move an observation to another source image.
    """

    rules = load_rules()
    allowed_roles = set(rules.get("allowed_source_roles", []))
    raw_items = analysis.get("images") if isinstance(analysis, dict) else None
    by_index: dict[int, dict[str, Any]] = {}
    if isinstance(raw_items, list):
        for raw in raw_items:
            if not isinstance(raw, dict) or not isinstance(raw.get("index"), int):
                continue
            index = raw["index"]
            if 0 <= index < len(image_urls) and index not in by_index:
                by_index[index] = raw

    normalized: list[dict[str, Any]] = []
    for index, url in enumerate(image_urls):
        raw = by_index.get(index, {})
        role = str(raw.get("role") or "unknown").strip().lower()
        if role not in allowed_roles:
            role = "unknown"
        item: dict[str, Any] = {
            "index": index,
            "url": url,
            "role": role,
            "dominant_color": str(raw.get("dominant_color") or "").strip(),
            "product_coverage": str(raw.get("product_coverage") or "unknown").lower(),
            "sharpness": str(raw.get("sharpness") or "unknown").lower(),
            "inspection_complete": bool(raw),
        }
        for field in (
            *rules.get("source_visual_hard_risk_fields", []),
            *rules.get("source_visual_review_fields", []),
        ):
            item[str(field)] = raw.get(str(field)) is True
        item["risk_reasons"] = source_visual_risk_reasons({**raw, **item})
        item["has_overlay_text"] = bool(
            item.get("has_overlay_text")
            or (
                item.get("has_text")
                and not item.get("has_intrinsic_product_text")
            )
        )
        hard_risk_fields = tuple(
            str(field) for field in rules.get("source_visual_hard_risk_fields", [])
        )
        hard_safe = not any(item.get(field) is True for field in hard_risk_fields)
        reference_product_clear = bool(
            role
            in {"hero", "front", "back", "side", "detail", "variant", "lifestyle"}
            and not item.get("product_obscured")
            and not item.get("low_sharpness")
            and item["sharpness"] != "low"
            and item["product_coverage"] != "low"
        )
        # Direct-listing safety and generation-reference usefulness are different.
        # Soft scene contamination can be removed by image editing and must not
        # starve the generator of every usable product-identity reference.
        item["safe_for_generation_reference"] = bool(
            raw and hard_safe and reference_product_clear
        )
        item["reference_requires_cleanup"] = bool(
            item["safe_for_generation_reference"]
            and any(
                item.get(field) is True
                for field in (
                    "has_text",
                    "has_overlay_text",
                    "has_logo",
                    "has_third_party_brand",
                    "has_person",
                    "has_unrelated_props",
                    "multiple_products",
                )
            )
        )
        item["safe_for_listing_fallback"] = bool(
            raw
            and hard_safe
            and not item["has_overlay_text"]
            and not item["has_logo"]
            and not item["has_third_party_brand"]
            and not item["product_obscured"]
            and not item["low_sharpness"]
            and item["sharpness"] != "low"
            and not item["has_unrelated_props"]
            and not item["multiple_products"]
        )
        item["safe_for_main_image"] = bool(
            item["safe_for_listing_fallback"]
            and not item["has_person"]
            and not item["has_unrelated_props"]
            and not item["multiple_products"]
            and item["product_complete"]
            and item["clean_neutral_background"]
            and item["product_coverage"] in {"high", "medium"}
            and item["sharpness"] != "low"
        )
        if not raw:
            item["risk_reasons"] = ["inspection_incomplete"]
        normalized.append(item)
    return normalized
