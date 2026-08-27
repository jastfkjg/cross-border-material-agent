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
class SizeChartRow:
    size_label: str
    bust_cm: str = ""
    length_cm: str = ""
    weight_kg: str = ""
    weight_lb: str = ""
    evidence_pointer: str = ""


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
    size_chart_rows: list[SizeChartRow] = field(default_factory=list)
    source_title_evidence_pointer: str = ""

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
            "size_chart": [
                {
                    "size": item.size_label,
                    "bust_cm": item.bust_cm,
                    "length_cm": item.length_cm,
                    "weight_kg": item.weight_kg,
                    "weight_lb": item.weight_lb,
                    "evidence": item.evidence_pointer,
                }
                for item in self.size_chart_rows
            ],
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
    # Stable machine-readable jobs for the five detail slots.  Prompts may be
    # rewritten by a model, but downstream review and localized media captions
    # must continue to use these canonical roles as their contract.
    detail_roles: list[str] = field(default_factory=list)


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
    """A bounded repair action selected by the delivery evaluator."""

    tool: str
    target: str
    instruction: str
    reason: str = ""
    dimension: str = ""
    priority: int = 0


@dataclass(slots=True)
class AgentEvaluation:
    """Structured whole-delivery feedback produced by the evaluator model."""

    round_index: int
    ready_for_delivery: bool
    weighted_score: float
    dimension_scores: dict[str, float] = field(default_factory=dict)
    summary: str = ""
    issues: list[dict[str, Any]] = field(default_factory=list)
    repair_actions: list[AgentAction] = field(default_factory=list)


@dataclass(slots=True)
class AgentActionResult:
    """Observable result of executing one model-selected tool call."""

    round_index: int
    tool: str
    target: str
    status: str
    detail: str = ""


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
