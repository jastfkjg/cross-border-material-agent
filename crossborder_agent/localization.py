"""Localized copy generation with immutable canonical data sections."""

from __future__ import annotations

import json
import re
from typing import Any

from .api import ApiError, QwenClient
from .compliance import generated_copy_violations
from .models import CreativePlan, ProductFacts, TaxonomyResult


LANGUAGES: dict[str, dict[str, str]] = {
    "en": {
        "locale": "en-US",
        "description": "natural US English marketplace copy",
        "overview": "Product overview",
        "highlights": "Verified highlights",
        "canonical": "Canonical listing data",
        "attributes": "Source product attributes",
        "platform_attributes": "Mapped AliExpress attributes",
        "skus": "SKU matrix",
        "sizes": "Seller-provided size guidance and unit conversion",
        "media": "Media guide",
        "note": "Important factual note",
        "color_note": "Colors can appear different depending on lighting and display settings. Canonical IDs, SKU combinations and source-declared values above are preserved from the provided data.",
    },
    "ko": {
        "locale": "ko-KR",
        "description": "자연스럽고 신뢰감 있는 한국 온라인 쇼핑몰 문체",
        "overview": "상품 소개",
        "highlights": "확인된 주요 특징",
        "canonical": "표준 상품 데이터",
        "attributes": "원본 상품 속성",
        "platform_attributes": "AliExpress 매핑 속성",
        "skus": "SKU 구성표",
        "sizes": "판매자 제공 사이즈 안내 및 단위 환산",
        "media": "미디어 안내",
        "note": "상품 정보 안내",
        "color_note": "조명과 화면 설정에 따라 색상이 다르게 보일 수 있습니다. 위의 표준 ID, SKU 조합 및 원본 표시 값은 제공된 데이터를 그대로 유지했습니다.",
    },
    "pt": {
        "locale": "pt-BR",
        "description": "português brasileiro natural para comércio eletrônico",
        "overview": "Visão geral do produto",
        "highlights": "Destaques verificados",
        "canonical": "Dados canônicos do anúncio",
        "attributes": "Atributos do produto na fonte",
        "platform_attributes": "Atributos mapeados para o AliExpress",
        "skus": "Matriz de SKUs",
        "sizes": "Orientação de tamanho do vendedor e conversão de unidades",
        "media": "Guia de mídia",
        "note": "Observação factual importante",
        "color_note": "As cores podem variar conforme a iluminação e a configuração da tela. Os IDs canônicos, as combinações de SKU e os valores declarados na fonte foram preservados a partir dos dados fornecidos.",
    },
}


_FALLBACK_CONTENT: dict[str, dict[str, Any]] = {
    "en": {
        "overview": "A localized listing prepared directly from the seller-provided product record. Review the verified attributes and SKU matrix below to select the appropriate variant.",
        "highlights": [
            "Product details are limited to information present in the source record.",
            "Available colors, sizes and combinations are listed by SKU below.",
            "Please use the seller-provided size guidance rather than assuming regional size equivalence.",
        ],
        "fit_note": "Regional size labels are not inferred when body measurements are unavailable.",
    },
    "ko": {
        "overview": "판매자가 제공한 원본 상품 정보를 바탕으로 현지화한 상품 페이지입니다. 아래의 확인된 속성과 SKU 구성표에서 원하는 옵션을 확인해 주세요.",
        "highlights": [
            "원본 데이터에서 확인 가능한 정보만 사용했습니다.",
            "판매 가능한 색상과 사이즈 조합은 아래 SKU 구성표에서 확인할 수 있습니다.",
            "지역별 사이즈가 동일하다고 가정하지 말고 판매자 제공 사이즈 안내를 확인해 주세요.",
        ],
        "fit_note": "신체 치수 정보가 없는 경우 한국 사이즈로 임의 변환하지 않습니다.",
    },
    "pt": {
        "overview": "Anúncio localizado a partir dos dados fornecidos pelo vendedor. Consulte os atributos verificados e a matriz de SKUs abaixo para escolher a variação adequada.",
        "highlights": [
            "Foram usadas somente informações presentes no cadastro de origem.",
            "As combinações disponíveis de cor e tamanho estão detalhadas na matriz de SKUs.",
            "Consulte a orientação de tamanho do vendedor sem presumir equivalência com tamanhos regionais.",
        ],
        "fit_note": "Não convertemos automaticamente para P, M ou G quando faltam medidas corporais.",
    },
}


def _allowed_numbers(facts: ProductFacts, taxonomy: TaxonomyResult) -> set[str]:
    material = [
        facts.offer_id,
        facts.source_title,
        facts.source_category_id,
        taxonomy.category.category_id,
        *(item.value for item in facts.attributes),
        *(sku.sku_id for sku in facts.skus),
        *(item.kilograms for item in facts.size_conversions),
        *(item.pounds for item in facts.size_conversions),
    ]
    return set(re.findall(r"\d+(?:[.,]\d+)?", " ".join(material)))


def _contains_unverified_numbers(value: Any, allowed: set[str]) -> bool:
    if isinstance(value, dict):
        return any(
            _contains_unverified_numbers(item, allowed) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_unverified_numbers(item, allowed) for item in value)
    if not isinstance(value, str):
        return False
    numbers = re.findall(r"\d+(?:[.,]\d+)?", value)
    return any(number not in allowed for number in numbers)


_FALLBACK_TITLES: dict[str, dict[str, str]] = {
    "en": {
        "29073": "Women's Shirt",
        "28951": "Women's Knit Pullover",
        "29069": "Women's T-Shirt",
        "28976": "Women's Wool-Blend Coat",
        "39107": "Women's Dress",
        "30408": "Men's Utility Jacket",
        "30843": "Boys' T-Shirt",
        "29553": "Girls' T-Shirt",
        "39153": "Women's Skirt",
        "30471": "Men's Casual Shirt",
        "30341": "Men's Flat-Front Shorts",
        "30335": "Men's Trousers",
    },
    "ko": {
        "29073": "여성 셔츠",
        "28951": "여성 니트 풀오버",
        "29069": "여성 티셔츠",
        "28976": "여성 울 혼방 코트",
        "39107": "여성 원피스",
        "30408": "남성 유틸리티 재킷",
        "30843": "남아 티셔츠",
        "29553": "여아 티셔츠",
        "39153": "여성 스커트",
        "30471": "남성 캐주얼 셔츠",
        "30341": "남성 플랫 프런트 쇼츠",
        "30335": "남성 팬츠",
    },
    "pt": {
        "29073": "Camisa feminina",
        "28951": "Suéter feminino de malha",
        "29069": "Camiseta feminina",
        "28976": "Casaco feminino de mistura de lã",
        "39107": "Vestido feminino",
        "30408": "Jaqueta utilitária masculina",
        "30843": "Camiseta infantil masculina",
        "29553": "Camiseta infantil feminina",
        "39153": "Saia feminina",
        "30471": "Camisa casual masculina",
        "30341": "Bermuda masculina sem pregas",
        "30335": "Calça masculina",
    },
}


def _fallback_payload(
    language: str, facts: ProductFacts, taxonomy: TaxonomyResult
) -> dict[str, Any]:
    payload = dict(_FALLBACK_CONTENT[language])
    payload["title"] = _FALLBACK_TITLES.get(language, {}).get(
        taxonomy.category.category_id,
        taxonomy.category.name or facts.source_category_name,
    )
    payload["media_descriptions"] = {
        "main_image.jpeg": "Clean primary product presentation.",
        "detail_image_1.jpeg": "Overall product and styling view.",
        "detail_image_2.jpeg": "Construction and visible design details.",
        "detail_image_3.jpeg": "Verified product feature presentation.",
        "detail_image_4.jpeg": "Available color or variant presentation based on source images.",
        "detail_image_5.jpeg": "Size, fit or use-context presentation based on available facts.",
        "product_video.mp4": "Short product presentation video.",
    }
    if language == "ko":
        payload["media_descriptions"] = {
            "main_image.jpeg": "상품을 선명하게 보여 주는 대표 이미지입니다.",
            "detail_image_1.jpeg": "상품의 전체 실루엣과 스타일을 보여 줍니다.",
            "detail_image_2.jpeg": "봉제와 눈으로 확인 가능한 디자인 디테일을 소개합니다.",
            "detail_image_3.jpeg": "원본 데이터에서 확인된 상품 특징을 보여 줍니다.",
            "detail_image_4.jpeg": "원본 이미지에 근거한 색상 또는 옵션 안내입니다.",
            "detail_image_5.jpeg": "확인 가능한 정보를 바탕으로 한 사이즈·핏 또는 활용 안내입니다.",
            "product_video.mp4": "상품을 간결하게 소개하는 영상입니다.",
        }
    elif language == "pt":
        payload["media_descriptions"] = {
            "main_image.jpeg": "Imagem principal com apresentação clara do produto.",
            "detail_image_1.jpeg": "Visão geral da peça e de sua proposta de uso.",
            "detail_image_2.jpeg": "Detalhes visíveis de construção e design.",
            "detail_image_3.jpeg": "Apresentação de características verificadas do produto.",
            "detail_image_4.jpeg": "Cores ou variações disponíveis com base nas imagens de origem.",
            "detail_image_5.jpeg": "Orientação de tamanho, caimento ou uso baseada nos dados disponíveis.",
            "product_video.mp4": "Vídeo curto de apresentação do produto.",
        }
    return payload


def generate_copy_payload(
    language: str,
    facts: ProductFacts,
    taxonomy: TaxonomyResult,
    creative_plan: CreativePlan,
    client: QwenClient | None,
) -> tuple[dict[str, Any], str]:
    fallback = _fallback_payload(language, facts, taxonomy)
    if client is None:
        return fallback, "deterministic-fallback"

    locale = LANGUAGES[language]
    system = f"""
You are a native {locale["locale"]} e-commerce copywriter and a strict factual editor.
Write {locale["description"]}. Return JSON only.
Do not translate mechanically from Chinese. Do not invent performance claims, measurements,
care instructions, composition, certifications, stock, price, brand authorization, or regional size equivalents.
Do not add unsupported numbers. Avoid absolute superlatives and body-shaming language.
""".strip()
    prompt = f"""
Produce JSON with exactly these keys:
- title: localized product title
- overview: two concise localized paragraphs as one string
- highlights: array of 3 to 5 concise strings
- fit_note: one conservative localized sizing note
- media_descriptions: object with exactly these keys:
  main_image.jpeg, detail_image_1.jpeg, detail_image_2.jpeg,
  detail_image_3.jpeg, detail_image_4.jpeg, detail_image_5.jpeg, product_video.mp4

Verified product facts:
{json.dumps(facts.compact_dict(), ensure_ascii=False)}

Resolved AliExpress category:
{json.dumps({"id": taxonomy.category.category_id, "name": taxonomy.category.name, "path": taxonomy.category.path}, ensure_ascii=False)}

Creative media plan:
{json.dumps({"theme": creative_plan.visual_theme, "detail_prompts": creative_plan.detail_prompts, "video_prompt": creative_plan.video_prompt}, ensure_ascii=False)}

Only write claims supported by the verified facts. The canonical data tables will be inserted by code;
do not repeat all SKUs or attributes in the prose.
""".strip()
    try:
        payload = client.chat_json(system, prompt)
    except ApiError:
        return fallback, "deterministic-fallback"
    required = {"title", "overview", "highlights", "fit_note", "media_descriptions"}
    if not required.issubset(payload):
        return fallback, "invalid-model-payload"
    if not isinstance(payload.get("highlights"), list) or not isinstance(
        payload.get("media_descriptions"), dict
    ):
        return fallback, "invalid-model-payload"
    expected_media = set(fallback["media_descriptions"])
    if set(payload["media_descriptions"]) != expected_media:
        return fallback, "invalid-model-payload"
    if _contains_unverified_numbers(payload, _allowed_numbers(facts, taxonomy)):
        return fallback, "numeric-fact-guard"
    if generated_copy_violations(language, payload):
        return fallback, "content-compliance-guard"
    return payload, client.config.chat_model


def _escape_table(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _canonical_section(facts: ProductFacts, taxonomy: TaxonomyResult) -> list[str]:
    category = taxonomy.category
    return [
        f"- **Source Platform:** {_escape_table(facts.platform)}",
        f"- **Product ID:** {_escape_table(facts.offer_id)}",
        f"- **Product URL:** {facts.source_url}",
        f"- **Source Category ID:** {_escape_table(facts.source_category_id)}",
        f"- **Source Category Name:** {_escape_table(facts.source_category_name)}",
        f"- **Leaf Category ID:** {_escape_table(category.category_id)}",
        f"- **Leaf Category Name:** {_escape_table(category.name)}",
        f"- **Leaf Category Path:** {_escape_table(category.path)}",
    ]


def render_description(
    language: str,
    payload: dict[str, Any],
    facts: ProductFacts,
    taxonomy: TaxonomyResult,
) -> str:
    locale = LANGUAGES[language]
    lines = [
        f"# {str(payload['title']).strip()}",
        "",
        f"## {locale['overview']}",
        "",
        str(payload["overview"]).strip(),
        "",
        f"## {locale['highlights']}",
        "",
    ]
    lines.extend(
        f"- {str(item).strip()}" for item in payload["highlights"] if str(item).strip()
    )
    lines.extend(["", f"## {locale['canonical']}", ""])
    lines.extend(_canonical_section(facts, taxonomy))

    lines.extend(
        [
            "",
            f"## {locale['attributes']}",
            "",
            "| Source Attribute ID | Source Attribute | Source Value | Evidence |",
            "|---|---|---|---|",
        ]
    )
    for item in facts.attributes:
        lines.append(
            f"| {_escape_table(item.attribute_id)} | {_escape_table(item.name)} | "
            f"{_escape_table(item.value)} | `{_escape_table(item.evidence_pointer)}` |"
        )

    lines.extend(
        [
            "",
            f"## {locale['platform_attributes']}",
            "",
            "| Type | Attribute ID | Platform Attribute | Source Attribute | Source Value | Value ID | Platform Value |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for item in taxonomy.attributes:
        item_type = "sales" if item.sales_attribute else "product"
        lines.append(
            f"| {item_type} | {_escape_table(item.attr_id)} | {_escape_table(item.name)} | "
            f"{_escape_table(item.source_name)} | {_escape_table(item.source_value)} | "
            f"{_escape_table(item.value_id)} | {_escape_table(item.platform_value)} |"
        )
    if taxonomy.missing_required:
        lines.append("")
        lines.append(
            "Missing required platform fields were not fabricated: "
            + ", ".join(_escape_table(item) for item in taxonomy.missing_required)
        )

    all_sku_names: list[str] = []
    for sku in facts.skus:
        for item in sku.attributes:
            if item.name and item.name not in all_sku_names:
                all_sku_names.append(item.name)
    lines.extend(["", f"## {locale['skus']}", ""])
    lines.append(
        "| SKU ID | Spec ID | "
        + " | ".join(_escape_table(name) for name in all_sku_names)
        + " |"
    )
    lines.append("|---|---|" + "---|" * len(all_sku_names))
    for sku in facts.skus:
        values = {item.name: item.value for item in sku.attributes}
        row = [sku.sku_id, sku.spec_id] + [
            values.get(name, "") for name in all_sku_names
        ]
        lines.append("| " + " | ".join(_escape_table(value) for value in row) + " |")

    lines.extend(["", f"## {locale['sizes']}", ""])
    if facts.size_conversions:
        lines.extend(
            [
                "| Seller Label | Metric | Imperial | Evidence |",
                "|---|---|---|---|",
            ]
        )
        for size in facts.size_conversions:
            lines.append(
                f"| {_escape_table(size.source_label)} | {_escape_table(size.kilograms)} | "
                f"{_escape_table(size.pounds)} | `{_escape_table(size.evidence_pointer)}` |"
            )
    else:
        lines.append(str(payload["fit_note"]).strip())

    lines.extend(["", f"## {locale['media']}", ""])
    for filename, description in payload["media_descriptions"].items():
        lines.append(f"- **{filename}:** {str(description).strip()}")

    lines.extend(
        [
            "",
            f"## {locale['note']}",
            "",
            str(payload["fit_note"]).strip(),
            "",
            locale["color_note"],
            "",
        ]
    )
    return "\n".join(lines)
