"""Versioned decision state shared by planning, production, review, and repair.

The model owns semantic decisions.  This module only turns those decisions into
stable, auditable state and enforces strategy-independent dependency contracts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .models import ClaimEvidence, ProductFacts, TaxonomyResult


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(slots=True)
class CanonicalClaim:
    claim_id: str
    concept: str
    value: str
    decision: str
    evidence_pointers: list[str] = field(default_factory=list)
    allowed_surfaces: list[str] = field(default_factory=list)
    confidence: float = 1.0
    rationale: str = ""
    source_value: str = ""


@dataclass(slots=True)
class CanonicalProductState:
    """One semantic product state synthesized from the model's evidence decisions."""

    version: str
    claims: list[CanonicalClaim]
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    unresolved_questions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def publishable_claims(self) -> list[CanonicalClaim]:
        return [item for item in self.claims if "buyer_copy" in item.allowed_surfaces]


@dataclass(slots=True)
class EvidenceSufficiency:
    """Availability report; it never invents an answer to a semantic question."""

    version: str
    inspected_image_indexes: list[int] = field(default_factory=list)
    generation_reference_indexes: list[int] = field(default_factory=list)
    listing_fallback_indexes: list[int] = field(default_factory=list)
    role_to_indexes: dict[str, list[int]] = field(default_factory=dict)
    unresolved_questions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExpectedDeliverySpec:
    """Frozen expectations against which all later artifacts are reviewed."""

    version: str
    canonical_state_version: str
    taxonomy_version: str
    category_id: str
    attribute_schema_category_id: str
    required_files: list[str]
    required_locales: list[str]
    publishable_claim_ids: list[str]
    publishable_claims: list[dict[str, Any]]
    required_mapping_sources: list[dict[str, Any]]
    visual_identity_claim_ids: list[str]
    visual_identity_claims: list[dict[str, Any]]
    source_variant_expectations: list[dict[str, Any]]
    evidence_reference_indexes: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def taxonomy_coverage_gaps(self, taxonomy: TaxonomyResult) -> list[dict[str, Any]]:
        actual = {
            (
                "sales" if item.sales_attribute else "product",
                item.source_name,
                item.source_value,
            )
            for item in taxonomy.attributes
        }
        return [
            item
            for item in self.required_mapping_sources
            if (item["scope"], item["source_name"], item["source_value"]) not in actual
        ]


@dataclass(slots=True)
class DependencyNode:
    version: str = ""
    inputs: dict[str, str] = field(default_factory=dict)
    stale_reason: str = ""


@dataclass(slots=True)
class DependencyState:
    """Small generic build graph used to prevent stale downstream acceptance."""

    nodes: dict[str, DependencyNode] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "DependencyState":
        nodes: dict[str, DependencyNode] = {}
        for name, item in (value or {}).items():
            if not isinstance(item, dict):
                continue
            nodes[str(name)] = DependencyNode(
                version=str(item.get("version") or ""),
                inputs={
                    str(key): str(version)
                    for key, version in (item.get("inputs") or {}).items()
                },
                stale_reason=str(item.get("stale_reason") or ""),
            )
        return cls(nodes=nodes)

    def record(self, name: str, value: Any, **inputs: str) -> str:
        version = _fingerprint(value)
        self.nodes[name] = DependencyNode(version=version, inputs=dict(inputs))
        return version

    def invalidate(self, changed: str, dependents: Iterable[str], reason: str) -> None:
        for name in dependents:
            node = self.nodes.setdefault(name, DependencyNode())
            node.stale_reason = f"{changed}: {reason}"

    def stale_nodes(self) -> list[str]:
        return sorted(name for name, node in self.nodes.items() if node.stale_reason)

    def to_dict(self) -> dict[str, Any]:
        return {name: asdict(node) for name, node in sorted(self.nodes.items())}


def build_canonical_product_state(
    facts: ProductFacts,
    reconciliation: dict[str, Any] | None,
) -> CanonicalProductState:
    """Materialize model decisions without adding host-side semantic repairs."""

    ledger = reconciliation if isinstance(reconciliation, dict) else {}
    decisions = {
        item.get("attribute_index"): item
        for item in ledger.get("attribute_decisions", [])
        if isinstance(item, dict) and isinstance(item.get("attribute_index"), int)
    }
    claims: list[CanonicalClaim] = []
    title_decision = str(ledger.get("seller_title_decision") or "publish")
    claims.append(
        CanonicalClaim(
            claim_id="source-title",
            concept="seller_title",
            value=facts.source_title,
            decision=title_decision,
            evidence_pointers=[facts.source_title_evidence_pointer],
            allowed_surfaces=(
                ["buyer_copy", "image_prompt", "machine_appendix"]
                if title_decision == "publish"
                else ["machine_appendix"]
            ),
            source_value=facts.source_title,
        )
    )
    for index, item in enumerate(facts.attributes):
        decision_row = decisions.get(index, {})
        decision = str(decision_row.get("decision") or "publish")
        canonical_value = str(decision_row.get("canonical_value") or item.value)
        claims.append(
            CanonicalClaim(
                claim_id=f"source-attribute-{index}",
                concept=item.name,
                value=canonical_value,
                decision=decision,
                evidence_pointers=[item.evidence_pointer],
                allowed_surfaces=(
                    ["buyer_copy", "image_prompt", "machine_appendix"]
                    if decision == "publish"
                    else ["machine_appendix"]
                ),
                confidence=float(decision_row.get("confidence") or 1.0),
                rationale=str(decision_row.get("reason") or ""),
                source_value=item.value,
            )
        )
    seen_sku: set[tuple[str, str]] = set()
    for sku in facts.skus:
        for item in sku.attributes:
            key = (item.name, item.value)
            if key in seen_sku:
                continue
            seen_sku.add(key)
            claims.append(
                CanonicalClaim(
                    claim_id=f"source-sku-{len(seen_sku)}",
                    concept=item.name,
                    value=item.value,
                    decision="publish",
                    evidence_pointers=[item.evidence_pointer],
                    allowed_surfaces=["buyer_copy", "image_prompt", "machine_appendix"],
                    source_value=item.value,
                )
            )
    for index, item in enumerate(ledger.get("canonical_visual_claims", [])):
        if not isinstance(item, dict) or not str(item.get("value") or "").strip():
            continue
        evidence = item.get("evidence")
        pointers = (
            [str(value) for value in evidence if str(value).strip()]
            if isinstance(evidence, list)
            else [str(evidence)] if str(evidence or "").strip() else []
        )
        claims.append(
            CanonicalClaim(
                claim_id=f"visual-claim-{index}",
                concept=str(item.get("concept") or "visible_design_feature"),
                value=str(item.get("value")),
                decision="publish",
                evidence_pointers=pointers,
                allowed_surfaces=["buyer_copy", "image_prompt", "media_guide"],
                confidence=float(item.get("confidence") or 0.8),
                rationale=str(item.get("reason") or ""),
                source_value=str(item.get("value")),
            )
        )
    conflicts = [item for item in ledger.get("conflicts", []) if isinstance(item, dict)]
    unresolved = [
        item
        for item in conflicts
        if not item.get("resolution") and not item.get("surface_resolutions")
    ]
    payload = {
        "claims": [asdict(item) for item in claims],
        "conflicts": conflicts,
        "unresolved_questions": unresolved,
    }
    return CanonicalProductState(
        version=_fingerprint(payload),
        claims=claims,
        conflicts=conflicts,
        unresolved_questions=unresolved,
    )


def assess_evidence_sufficiency(
    vision: dict[str, Any] | None,
    canonical: CanonicalProductState,
) -> EvidenceSufficiency:
    images = vision.get("source_images", []) if isinstance(vision, dict) else []
    inspected: list[int] = []
    generation: list[int] = []
    fallback: list[int] = []
    roles: dict[str, list[int]] = {}
    for item in images if isinstance(images, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("index"), int):
            continue
        index = int(item["index"])
        role = str(item.get("role") or "unknown")
        roles.setdefault(role, []).append(index)
        if item.get("inspection_complete") is True:
            inspected.append(index)
        if item.get("safe_for_generation_reference") is True:
            generation.append(index)
        if item.get("safe_for_listing_fallback") is True:
            fallback.append(index)
    unresolved = list(canonical.unresolved_questions)
    if images and not inspected:
        unresolved.append(
            {"question": "No source image inspection completed", "affected_surfaces": ["media"]}
        )
    payload = {
        "inspected": inspected,
        "generation": generation,
        "fallback": fallback,
        "roles": roles,
        "unresolved": unresolved,
    }
    return EvidenceSufficiency(
        version=_fingerprint(payload),
        inspected_image_indexes=inspected,
        generation_reference_indexes=generation,
        listing_fallback_indexes=fallback,
        role_to_indexes=roles,
        unresolved_questions=unresolved,
    )


def build_expected_delivery_spec(
    *,
    canonical: CanonicalProductState,
    taxonomy: TaxonomyResult,
    claim_ledger: list[ClaimEvidence],
    evidence: EvidenceSufficiency,
    required_files: Iterable[str],
    preserve_mapping_sources: Iterable[dict[str, Any]] = (),
) -> ExpectedDeliverySpec:
    taxonomy_payload = {
        "category": taxonomy.category.category_id,
        "schema": taxonomy.attribute_schema_category_id,
        "attributes": [
            {
                "scope": "sales" if item.sales_attribute else "product",
                "source_name": item.source_name,
                "source_value": item.source_value,
                "attr_id": item.attr_id,
                "value_id": item.value_id,
            }
            for item in taxonomy.attributes
        ],
    }
    taxonomy_version = _fingerprint(taxonomy_payload)
    mapping_sources = [
        {
            "scope": "sales" if item.sales_attribute else "product",
            "source_name": item.source_name,
            "source_value": item.source_value,
        }
        for item in taxonomy.attributes
    ]
    for item in preserve_mapping_sources:
        if not isinstance(item, dict):
            continue
        normalized = {
            "scope": str(item.get("scope") or ""),
            "source_name": str(item.get("source_name") or ""),
            "source_value": str(item.get("source_value") or ""),
        }
        if all(normalized.values()) and normalized not in mapping_sources:
            mapping_sources.append(normalized)
    publishable = [
        item.claim_id for item in claim_ledger if "buyer_copy" in item.allowed_surfaces
    ]
    visual = [
        item.claim_id
        for item in canonical.claims
        if "image_prompt" in item.allowed_surfaces
    ]
    publishable_rows = [
        {
            "claim_id": item.claim_id,
            "concept": item.concept,
            "value": item.value,
            "evidence_pointer": item.evidence_pointer,
        }
        for item in claim_ledger
        if "buyer_copy" in item.allowed_surfaces
    ]
    visual_rows = [
        {
            "claim_id": item.claim_id,
            "concept": item.concept,
            "value": item.value,
            "evidence_pointers": item.evidence_pointers,
        }
        for item in canonical.claims
        if "image_prompt" in item.allowed_surfaces
    ]
    source_variants = [
        {
            "claim_id": item.claim_id,
            "concept": item.concept,
            "value": item.value,
            "evidence_pointers": item.evidence_pointers,
        }
        for item in canonical.claims
        if item.claim_id.startswith("source-sku-")
    ]
    payload = {
        "canonical": canonical.version,
        "taxonomy": taxonomy_version,
        "files": sorted(required_files),
        "locales": ["en", "ko", "pt"],
        "publishable": publishable,
        "publishable_rows": publishable_rows,
        "mappings": mapping_sources,
        "visual": visual,
        "visual_rows": visual_rows,
        "source_variants": source_variants,
        "references": evidence.generation_reference_indexes,
    }
    return ExpectedDeliverySpec(
        version=_fingerprint(payload),
        canonical_state_version=canonical.version,
        taxonomy_version=taxonomy_version,
        category_id=taxonomy.category.category_id,
        attribute_schema_category_id=taxonomy.attribute_schema_category_id,
        required_files=sorted(required_files),
        required_locales=["en", "ko", "pt"],
        publishable_claim_ids=publishable,
        publishable_claims=publishable_rows,
        required_mapping_sources=mapping_sources,
        visual_identity_claim_ids=visual,
        visual_identity_claims=visual_rows,
        source_variant_expectations=source_variants,
        evidence_reference_indexes=list(evidence.generation_reference_indexes),
    )
