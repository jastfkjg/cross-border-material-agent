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


@dataclass(slots=True)
class Sku:
    sku_id: str
    spec_id: str
    attributes: list[SkuAttribute] = field(default_factory=list)


@dataclass(slots=True)
class SizeConversion:
    source_label: str
    kilograms: str = ""
    pounds: str = ""
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
    value_id: str = ""
    platform_value: str = ""
    required: bool = False
    sales_attribute: bool = False


@dataclass(slots=True)
class TaxonomyResult:
    category: CategoryChoice
    attributes: list[MappedAttribute] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CreativePlan:
    visual_theme: str
    main_prompt: str
    detail_prompts: list[str]
    video_prompt: str
    market_angles: dict[str, str] = field(default_factory=dict)


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
class RunState:
    started_at: str
    input_dir: str
    output_dir: str
    facts: ProductFacts
    taxonomy: TaxonomyResult
    creative_plan: CreativePlan
    vision_observations: dict[str, Any] = field(default_factory=dict)
    assets: list[AssetResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    api_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
