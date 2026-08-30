"""Typed, serializable state used throughout the bounded agent pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ProductAttribute:
    attribute_id: str
    name: str
    value: str
    value_translated: str = ""
    evidence_pointer: str = ""


@dataclass(slots=True)
class SkuAttribute:
    attribute_id: str
    name: str
    value: str
    image_url: str = ""
    evidence_pointer: str = ""


@dataclass(slots=True)
class Sku:
    sku_id: str
    spec_id: str
    attributes: list[SkuAttribute] = field(default_factory=list)
    evidence_pointer: str = ""


@dataclass(slots=True)
class SizeConversion:
    source_label: str
    kilograms: str = ""
    pounds: str = ""
    evidence_pointer: str = ""


@dataclass(slots=True)
class EvidenceCell:
    """One source-grounded cell in a model-observed two-dimensional region."""

    row: int
    column: int
    text: str
    confidence: float = 1.0
    evidence_pointer: str = ""
    row_span: int = 1
    column_span: int = 1


@dataclass(slots=True)
class EvidenceTable:
    """A domain-neutral table plus the model's optional presentation decision.

    Host code understands coordinates, provenance and resource limits only.  It
    deliberately has no vocabulary for apparel measurements or other product
    fields; column meaning and presentation stay in model-authored data.
    """

    table_id: str
    source_image_index: int
    cells: list[EvidenceCell] = field(default_factory=list)
    source_url: str = ""
    evidence_pointer: str = ""
    presentation: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProductFacts:
    platform: str
    source_url: str
    offer_id: str
    source_title: str
    source_category_id: str
    source_category_name: str
    attributes: list[ProductAttribute]
    skus: list[Sku]
    product_image_urls: list[str]
    sku_image_urls: list[str]
    description_image_urls: list[str]
    size_conversions: list[SizeConversion]
    input_file: str
    fingerprint: str
    evidence_tables: list[EvidenceTable] = field(default_factory=list)
    source_title_evidence_pointer: str = ""
    # Evidence decisions made after source-image inspection.  The structure is
    # intentionally product-agnostic: source attributes are addressed by their
    # input index, and downstream stages consume the decision instead of
    # embedding category- or feature-specific exceptions in code.
    reconciled_fact_ledger: dict[str, Any] = field(default_factory=dict)

    def all_image_urls(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for url in (
            self.product_image_urls + self.sku_image_urls + self.description_image_urls
        ):
            if url and url not in seen:
                seen.add(url)
                result.append(url)
        return result

    def compact_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "source_url": self.source_url,
            "offer_id": self.offer_id,
            "source_title": self.source_title,
            "source_category_id": self.source_category_id,
            "source_category_name": self.source_category_name,
            "attributes": [
                {"id": item.attribute_id, "name": item.name, "value": item.value}
                for item in self.attributes
            ],
            "skus": [
                {
                    "sku_id": sku.sku_id,
                    "attributes": [
                        {"name": attr.name, "value": attr.value}
                        for attr in sku.attributes
                    ],
                }
                for sku in self.skus
            ],
            "evidence_tables": [
                {
                    "table_id": table.table_id,
                    "source_image_index": table.source_image_index,
                    "source_url": table.source_url,
                    "evidence": table.evidence_pointer,
                    "cells": [
                        {
                            "row": cell.row,
                            "column": cell.column,
                            "text": cell.text,
                            "confidence": cell.confidence,
                            "evidence": cell.evidence_pointer,
                            "row_span": cell.row_span,
                            "column_span": cell.column_span,
                        }
                        for cell in table.cells
                    ],
                    "presentation": table.presentation,
                }
                for table in self.evidence_tables
            ],
            "reconciled_fact_ledger": self.reconciled_fact_ledger,
            "image_urls": self.all_image_urls()[:12],
        }


@dataclass(slots=True)
class CategoryChoice:
    category_id: str
    name: str
    path: str
    confidence: float
    method: str
    candidates: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class MappedAttribute:
    attr_id: str
    name: str
    source_name: str
    source_value: str
    source_evidence_pointer: str = ""
    value_id: str = ""
    platform_value: str = ""
    required: bool = False
    sales_attribute: bool = False


@dataclass(slots=True)
class ClaimEvidence:
    """One publishable or machine-only claim with its original evidence pointer."""

    claim_id: str
    concept: str
    value: str
    source_type: str
    source_name: str
    evidence_pointer: str
    confidence: float = 1.0
    allowed_surfaces: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TaxonomyResult:
    category: CategoryChoice
    attributes: list[MappedAttribute] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    attribute_schema_category_id: str = ""


@dataclass(slots=True)
class CreativePlan:
    visual_theme: str
    main_prompt: str
    detail_prompts: list[str]
    video_prompt: str
    market_angles: dict[str, str] = field(default_factory=dict)
    # Machine-readable jobs and evidence preferences chosen by the orchestrator.
    # The host validates their shape but deliberately does not assign product-
    # specific semantics to a particular slot.
    detail_roles: list[str] = field(default_factory=list)
    main_candidate_count: int = 3
    detail_candidate_counts: list[int] = field(default_factory=list)
    main_reference_roles: list[str] = field(default_factory=list)
    detail_reference_roles: list[list[str]] = field(default_factory=list)
    # Exact inspected source-image evidence chosen by the model. Role labels are
    # useful hints; these indexes make the production dependency auditable.
    main_reference_indexes: list[int] = field(default_factory=list)
    detail_reference_indexes: list[list[int]] = field(default_factory=list)


@dataclass(slots=True)
class AssetResult:
    name: str
    path: str
    source_url: str = ""
    model: str = ""
    generated: bool = False
    fallback_reason: str = ""
    description: str = ""


@dataclass(slots=True)
class AgentAction:
    """A bounded repair action selected by the repair planner."""

    tool: str
    target: str
    instruction: str
    reason: str = ""
    dimension: str = ""
    priority: int = 0
    supporting_models: list[str] = field(default_factory=list)
    votes: int = 0
    execution_tier: str = "unclassified"
    verification: dict[str, Any] = field(default_factory=dict)
    defect_id: str = ""
    acceptance_criteria: str = ""


@dataclass(slots=True)
class AgentEvaluation:
    """Evidence findings adjudicated across independent evaluator reports.

    Legacy score fields remain serializable for archive compatibility, but they
    are no longer used to accept or reject a delivery. Semantic review supplies
    repair evidence; deterministic artifact contracts own submission.
    """

    round_index: int
    ready_for_delivery: bool
    weighted_score: float
    dimension_scores: dict[str, float] = field(default_factory=dict)
    summary: str = ""
    issues: list[dict[str, Any]] = field(default_factory=list)
    repair_actions: list[AgentAction] = field(default_factory=list)
    evaluator_models: list[str] = field(default_factory=list)
    model_dimension_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    model_weighted_scores: dict[str, float] = field(default_factory=dict)
    artifact_fingerprint: str = ""
    rubric_version: str = "evidence-v1"
    disagreement: bool = False
    adjudication: dict[str, Any] = field(default_factory=dict)
    score_method: str = "advisory-findings-no-score-gate"


@dataclass(slots=True)
class AgentActionResult:
    """Observable result of executing one model-selected tool call."""

    round_index: int
    tool: str
    target: str
    status: str
    detail: str = ""
    defect_id: str = ""
    before_hash: str = ""
    after_hash: str = ""
    changed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunState:
    started_at: str
    input_dir: str
    output_dir: str
    facts: ProductFacts
    taxonomy: TaxonomyResult
    creative_plan: CreativePlan
    claim_ledger: list[ClaimEvidence] = field(default_factory=list)
    vision_observations: dict[str, Any] = field(default_factory=dict)
    assets: list[AssetResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    api_calls: list[dict[str, Any]] = field(default_factory=list)
    agent_plan: dict[str, Any] = field(default_factory=dict)
    visual_set_review: dict[str, Any] = field(default_factory=dict)
    agent_evaluations: list[AgentEvaluation] = field(default_factory=list)
    agent_actions: list[AgentActionResult] = field(default_factory=list)
    agent_snapshots: list[dict[str, Any]] = field(default_factory=list)
    # Strategy-independent problem state for the top-level orchestrator.  The
    # host records observations and attempts; it never maps a finding to a
    # particular semantic repair.
    defect_ledger: list[dict[str, Any]] = field(default_factory=list)
    accepted_artifact_fingerprint: str = ""
    canonical_product_state: dict[str, Any] = field(default_factory=dict)
    evidence_sufficiency: dict[str, Any] = field(default_factory=dict)
    expected_delivery_spec: dict[str, Any] = field(default_factory=dict)
    dependency_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
