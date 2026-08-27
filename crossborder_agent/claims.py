"""Evidence-aware claim ledger and conservative taxonomy provenance checks."""

from __future__ import annotations

import re
from typing import Any

from .models import ClaimEvidence, ProductFacts, TaxonomyResult


_PRIVATE_SOURCE_MARKERS = (
    "货号",
    "货源",
    "吊牌",
    "领标",
    "库存",
    "包装",
    "厂家",
    "跨境",
    "是否",
    "体型",
    "适用人群",
    "适合人群",
)


def buyer_safe_source_name(name: str) -> bool:
    """Whether a seller field is suitable for shopper-facing publication."""

    return not any(marker in str(name) for marker in _PRIVATE_SOURCE_MARKERS)


def _normalized(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value).casefold())


def _same_source(left_name: str, left_value: str, right_name: str, right_value: str) -> bool:
    return (
        _normalized(left_name) == _normalized(right_name)
        and _normalized(left_value) == _normalized(right_value)
    )


def filter_invalid_mapping_provenance(
    facts: ProductFacts,
    taxonomy: TaxonomyResult,
) -> list[str]:
    """Remove only mappings whose claimed source is demonstrably contradictory.

    Missing or unfamiliar pointer shapes are accepted when the exact source
    name/value exists elsewhere in the canonical input. This avoids reducing
    attribute coverage merely because one provider changes a JSON path.
    """

    exact_sources: list[tuple[str, str, str]] = []
    for item in facts.attributes:
        exact_sources.append((item.name, item.value, item.evidence_pointer))
    for sku in facts.skus:
        for item in sku.attributes:
            exact_sources.append((item.name, item.value, item.evidence_pointer))

    source_by_pointer = {
        pointer: (name, value)
        for name, value, pointer in exact_sources
        if pointer
    }
    kept = []
    warnings: list[str] = []
    for item in taxonomy.attributes:
        pointer = item.source_evidence_pointer
        exact_match = any(
            _same_source(item.source_name, item.source_value, name, value)
            for name, value, _ in exact_sources
        )
        valid = exact_match
        if pointer in source_by_pointer:
            source_name, source_value = source_by_pointer[pointer]
            valid = _same_source(
                item.source_name,
                item.source_value,
                source_name,
                source_value,
            )
        elif pointer and pointer == facts.source_title_evidence_pointer:
            valid = bool(
                _normalized(item.source_value)
                and _normalized(item.source_value) in _normalized(facts.source_title)
            )

        if valid:
            kept.append(item)
            continue
        warnings.append(
            f"属性映射来源不一致，已移除 {item.name}: "
            f"{item.source_name}={item.source_value} ({pointer or '无证据指针'})"
        )
        if item.required and item.name not in taxonomy.missing_required:
            taxonomy.missing_required.append(item.name)
    taxonomy.attributes = kept
    return warnings


def build_claim_ledger(
    facts: ProductFacts,
    taxonomy: TaxonomyResult,
    vision: dict[str, Any] | None = None,
) -> list[ClaimEvidence]:
    """Create a compact claim-to-evidence registry used by copy and diagnostics."""

    claims: list[ClaimEvidence] = []

    def add(
        concept: str,
        value: str,
        source_type: str,
        source_name: str,
        pointer: str,
        surfaces: list[str],
        confidence: float = 1.0,
    ) -> None:
        if not str(value).strip():
            return
        claims.append(
            ClaimEvidence(
                claim_id=f"C{len(claims) + 1:03d}",
                concept=str(concept),
                value=str(value),
                source_type=source_type,
                source_name=str(source_name),
                evidence_pointer=str(pointer),
                confidence=max(0.0, min(1.0, float(confidence))),
                allowed_surfaces=list(surfaces),
            )
        )

    add(
        "seller_title",
        facts.source_title,
        "seller_title",
        "source_title",
        facts.source_title_evidence_pointer,
        ["buyer_copy", "machine_appendix", "image_prompt"],
    )
    for item in facts.attributes:
        buyer_safe = buyer_safe_source_name(item.name)
        surfaces = ["machine_appendix"]
        if buyer_safe:
            surfaces = ["buyer_copy", "machine_appendix", "image_prompt"]
        add(
            item.name,
            item.value,
            "seller_attribute",
            item.name,
            item.evidence_pointer,
            surfaces,
        )
    for sku in facts.skus:
        for item in sku.attributes:
            add(
                item.name,
                item.value,
                "seller_sku",
                item.name,
                item.evidence_pointer,
                ["buyer_copy", "machine_appendix", "image_prompt"],
            )
    for item in facts.size_conversions:
        value = "/".join(part for part in (item.kilograms, item.pounds) if part)
        add(
            "size_guidance",
            f"{item.source_label}: {value}",
            "seller_size_guidance",
            item.source_label,
            item.evidence_pointer,
            ["buyer_copy", "machine_appendix"],
        )
    for item in facts.size_chart_rows:
        values = [
            item.bust_cm,
            item.length_cm,
            item.weight_kg,
            item.weight_lb,
        ]
        add(
            "size_chart_row",
            f"{item.size_label}: {'/'.join(value for value in values if value)}",
            "source_image_ocr",
            item.size_label,
            item.evidence_pointer,
            ["buyer_copy", "machine_appendix", "media_guide"],
        )
    for item in taxonomy.attributes:
        add(
            item.name,
            item.platform_value or item.source_value,
            "taxonomy_mapping",
            item.source_name,
            item.source_evidence_pointer,
            ["machine_appendix"],
        )

    observations = (vision or {}).get("design_features")
    if isinstance(observations, list):
        for index, item in enumerate(observations):
            if isinstance(item, str):
                value = item
                confidence = 0.7
            elif isinstance(item, dict):
                value = str(item.get("feature") or item.get("value") or "")
                confidence = float(item.get("confidence") or 0.7)
            else:
                continue
            add(
                "visible_design_feature",
                value,
                "source_image_observation",
                "design_features",
                f"vision.design_features[{index}]",
                ["image_prompt", "media_guide"],
                confidence,
            )
    return claims


def publishable_claims(claims: list[ClaimEvidence]) -> list[dict[str, Any]]:
    """Serialize only buyer-copy claims and keep prompts within a bounded size."""

    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in claims:
        if "buyer_copy" not in item.allowed_surfaces:
            continue
        key = (_normalized(item.concept), _normalized(item.value), item.source_type)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "claim_id": item.claim_id,
                "concept": item.concept,
                "value": item.value,
                "source_type": item.source_type,
                "source_name": item.source_name,
                "evidence_pointer": item.evidence_pointer,
                "confidence": item.confidence,
            }
        )
        if len(result) == 60:
            break
    return result
