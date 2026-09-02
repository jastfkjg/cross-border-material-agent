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
    canonical_claims = facts.reconciled_fact_ledger.get("canonical_visual_claims", [])
    for index, item in enumerate(canonical_claims if isinstance(canonical_claims, list) else []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("concept") or "visible_design_feature")
        value = str(item.get("value") or "")
        if value:
            exact_sources.append(
                (
                    name,
                    value,
                    f"reconciled_fact_ledger.canonical_visual_claims[{index}]",
                )
            )

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
            f"Removed mapped attribute {item.attr_id}/{item.value_id or 'no-value-id'} "
            f"because its source evidence is inconsistent ({pointer or 'no evidence pointer'})"
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
    reconciliation = (
        facts.reconciled_fact_ledger
        if isinstance(facts.reconciled_fact_ledger, dict)
        else {}
    )
    attribute_decisions = {
        item.get("attribute_index"): item
        for item in reconciliation.get("attribute_decisions", [])
        if isinstance(item, dict) and isinstance(item.get("attribute_index"), int)
    }
    title_decision = str(reconciliation.get("seller_title_decision") or "publish")

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
        (
            ["buyer_copy", "machine_appendix", "image_prompt"]
            if title_decision == "publish"
            else ["machine_appendix"]
        ),
    )
    for attribute_index, item in enumerate(facts.attributes):
        buyer_safe = buyer_safe_source_name(item.name)
        surfaces = ["machine_appendix"]
        decision_row = attribute_decisions.get(attribute_index, {})
        decision = str(decision_row.get("decision") or "publish")
        surface_decisions = decision_row.get("surface_decisions")
        surface_decisions = (
            surface_decisions if isinstance(surface_decisions, dict) else {}
        )
        # Surface-specific model decisions prevent pixel observability from
        # suppressing a source-grounded non-visual fact everywhere. The host
        # translates surfaces mechanically and never infers product semantics.
        buyer_decision = str(surface_decisions.get("buyer_copy") or decision)
        media_decision = str(surface_decisions.get("media_generation") or decision)
        if buyer_safe and buyer_decision == "publish":
            surfaces.append("buyer_copy")
        if media_decision == "publish":
            surfaces.append("image_prompt")
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
    for table in facts.evidence_tables:
        is_rendered = table.presentation.get("decision") == "render"
        for cell in table.cells:
            add(
                "source_table_cell",
                cell.text,
                "source_image_table",
                table.table_id,
                cell.evidence_pointer,
                (
                    ["buyer_copy", "machine_appendix", "media_guide"]
                    if is_rendered
                    else ["machine_appendix"]
                ),
                cell.confidence,
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

    canonical_claims = reconciliation.get("canonical_visual_claims")
    if isinstance(canonical_claims, list):
        for index, item in enumerate(canonical_claims):
            if not isinstance(item, dict):
                continue
            concept = str(item.get("concept") or "visible_design_feature").strip()
            value = str(item.get("value") or "").strip()
            evidence = item.get("evidence")
            evidence_text = "; ".join(
                str(part).strip() for part in evidence if str(part).strip()
            ) if isinstance(evidence, list) else str(evidence or "")
            add(
                concept,
                value,
                "reconciled_visual_evidence",
                concept,
                evidence_text or f"reconciled_fact_ledger.canonical_visual_claims[{index}]",
                ["buyer_copy", "image_prompt", "media_guide"],
                float(item.get("confidence") or 0.8),
            )

    observations = (vision or {}).get("design_features")
    if not isinstance(observations, list):
        observations = (vision or {}).get("visible_design_features")
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
