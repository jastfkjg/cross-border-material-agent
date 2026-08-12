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


FIELD_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "source_platform": "Source Platform",
        "product_id": "Product ID",
        "product_url": "Product URL",
        "source_category_id": "Source Category ID",
        "source_category_name": "Source Category Name",
        "leaf_category_id": "Leaf Category ID",
        "leaf_category_name": "Leaf Category Name",
        "leaf_category_path": "Leaf Category Path",
        "raw_note": "Original source labels and values are preserved below for exact traceability.",
        "source_attribute": "Source Attribute",
        "source_value": "Source Value",
        "platform_attribute": "Platform Attribute",
        "platform_value": "Platform Value",
        "evidence": "Evidence",
        "type": "Type",
        "seller_label": "Seller Label",
        "metric": "Metric",
        "imperial": "Imperial",
        "missing": "Required platform fields not present in the source were not fabricated",
    },
    "ko": {
        "source_platform": "원본 플랫폼",
        "product_id": "상품 ID",
        "product_url": "상품 URL",
        "source_category_id": "원본 카테고리 ID",
        "source_category_name": "원본 카테고리명",
        "leaf_category_id": "최종 카테고리 ID",
        "leaf_category_name": "최종 카테고리명",
        "leaf_category_path": "최종 카테고리 경로",
        "raw_note": "정확한 추적을 위해 원본 필드명과 값을 아래에 그대로 보존했습니다.",
        "source_attribute": "원본 속성",
        "source_value": "원본 값",
        "platform_attribute": "플랫폼 속성",
        "platform_value": "플랫폼 값",
        "evidence": "근거 위치",
        "type": "유형",
        "seller_label": "판매자 표기",
        "metric": "미터법",
        "imperial": "야드파운드법",
        "missing": "원본에 없는 필수 플랫폼 필드는 임의로 생성하지 않았습니다",
    },
    "pt": {
        "source_platform": "Plataforma de origem",
        "product_id": "ID do produto",
        "product_url": "URL do produto",
        "source_category_id": "ID da categoria de origem",
        "source_category_name": "Categoria de origem",
        "leaf_category_id": "ID da categoria final",
        "leaf_category_name": "Categoria final",
        "leaf_category_path": "Caminho da categoria final",
        "raw_note": "Os rótulos e valores originais são preservados abaixo para rastreabilidade exata.",
        "source_attribute": "Atributo de origem",
        "source_value": "Valor de origem",
        "platform_attribute": "Atributo da plataforma",
        "platform_value": "Valor da plataforma",
        "evidence": "Evidência",
        "type": "Tipo",
        "seller_label": "Indicação do vendedor",
        "metric": "Métrico",
        "imperial": "Imperial",
        "missing": "Campos obrigatórios ausentes na origem não foram inventados",
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
    allowed = set(re.findall(r"\d+(?:[.,]\d+)?", " ".join(material)))
    allowed.add("8")  # fixed video duration from the delivery contract
    grouped_values: dict[str, set[str]] = {}
    for item in facts.attributes:
        grouped_values.setdefault(item.name, set()).add(item.value)
    allowed.update(
        str(len(values)) for values in grouped_values.values() if len(values) > 1
    )
    if facts.skus:
        allowed.add(str(len(facts.skus)))
    return allowed


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


def _payload_validation_error(
    language: str,
    payload: Any,
    facts: ProductFacts,
    taxonomy: TaxonomyResult,
    expected_media: set[str],
) -> str:
    required = {"title", "overview", "highlights", "fit_note", "media_descriptions"}
    if not isinstance(payload, dict) or set(payload) != required:
        return "invalid-model-payload"
    if not all(
        isinstance(payload.get(key), str) and payload[key].strip()
        for key in ("title", "overview", "fit_note")
    ):
        return "invalid-model-payload"
    highlights = payload.get("highlights")
    if (
        not isinstance(highlights, list)
        or not 3 <= len(highlights) <= 5
        or not all(isinstance(item, str) and item.strip() for item in highlights)
    ):
        return "invalid-model-payload"
    media = payload.get("media_descriptions")
    if (
        not isinstance(media, dict)
        or set(media) != expected_media
        or not all(isinstance(item, str) and item.strip() for item in media.values())
    ):
        return "invalid-model-payload"
    generated_text = json.dumps(payload, ensure_ascii=False)
    if re.search(r"[\u4e00-\u9fff]", generated_text):
        return "source-script-contamination-guard"
    if _contains_unverified_numbers(payload, _allowed_numbers(facts, taxonomy)):
        return "numeric-fact-guard"
    if generated_copy_violations(language, payload):
        return "content-compliance-guard"
    return ""


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
Avoid inferred benefits such as breathable, lightweight, comfortable, durable, shape-retaining,
flattering, premium quality, or easy-care unless the exact benefit is explicitly stated in the source facts.
Prefer concrete source-backed details such as product type, composition, pattern, collar, sleeve,
closure, fit, length, colors, and seller-declared option guidance.
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
        draft = client.chat_json(system, prompt)
    except ApiError:
        return fallback, "deterministic-fallback"

    audit_system = f"""
You are a native {locale["locale"]} factual copy auditor. Return JSON only.
Rewrite the candidate listing so every product claim is directly supported by the verified source facts.
Remove inferred benefits, marketing embellishment, unsupported measurements and regional size equivalence.
Keep natural localized language, but prefer precise field-level facts over generic filler.
The result must contain exactly the requested five top-level keys and exactly the seven required media keys.
Do not copy any Chinese characters into the localized payload; translate source concepts fully into the target locale.
""".strip()
    audit_prompt = f"""
Audit and, where needed, correct this candidate payload:
{json.dumps(draft, ensure_ascii=False)}

Verified source facts:
{json.dumps(facts.compact_dict(), ensure_ascii=False)}

Resolved AliExpress category:
{json.dumps({"id": taxonomy.category.category_id, "name": taxonomy.category.name, "path": taxonomy.category.path}, ensure_ascii=False)}

Required top-level keys: title, overview, highlights, fit_note, media_descriptions.
Required media keys: main_image.jpeg, detail_image_1.jpeg, detail_image_2.jpeg,
detail_image_3.jpeg, detail_image_4.jpeg, detail_image_5.jpeg, product_video.mp4.
The video description may state the fixed 8-second duration. Output the corrected JSON object only.
""".strip()
    audit_applied = True
    try:
        audited = client.chat_json(audit_system, audit_prompt)
    except ApiError:
        audited = draft
        audit_applied = False

    payload = audited
    expected_media = set(fallback["media_descriptions"])
    validation_error = _payload_validation_error(
        language, payload, facts, taxonomy, expected_media
    )
    repaired = False
    if validation_error:
        repair_prompt = f"""
Repair the candidate payload because it failed this deterministic check: {validation_error}.
Return the same exact schema in natural {locale["locale"]}. Translate all source concepts fully;
do not copy Chinese characters. Remove unsupported numbers and claims instead of replacing them with new claims.

Candidate:
{json.dumps(payload, ensure_ascii=False)}

Verified source facts:
{json.dumps(facts.compact_dict(), ensure_ascii=False)}
""".strip()
        try:
            payload = client.chat_json(audit_system, repair_prompt)
        except ApiError:
            return fallback, validation_error
        validation_error = _payload_validation_error(
            language, payload, facts, taxonomy, expected_media
        )
        if validation_error:
            return fallback, validation_error
        repaired = True

    source = client.config.chat_model
    if repaired:
        source += "-factual-repair"
    elif audit_applied:
        source += "-factual-audit"
    else:
        source += "-validated-draft"
    return payload, source


def _escape_table(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _canonical_section(
    language: str, facts: ProductFacts, taxonomy: TaxonomyResult
) -> list[str]:
    category = taxonomy.category
    labels = FIELD_LABELS[language]
    return [
        f"- **{labels['source_platform']}:** {_escape_table(facts.platform)}",
        f"- **{labels['product_id']}:** {_escape_table(facts.offer_id)}",
        f"- **{labels['product_url']}:** {facts.source_url}",
        f"- **{labels['source_category_id']}:** {_escape_table(facts.source_category_id)}",
        f"- **{labels['source_category_name']}:** {_escape_table(facts.source_category_name)}",
        f"- **{labels['leaf_category_id']}:** {_escape_table(category.category_id)}",
        f"- **{labels['leaf_category_name']}:** {_escape_table(category.name)}",
        f"- **{labels['leaf_category_path']}:** {_escape_table(category.path)}",
    ]


def render_description(
    language: str,
    payload: dict[str, Any],
    facts: ProductFacts,
    taxonomy: TaxonomyResult,
) -> str:
    locale = LANGUAGES[language]
    labels = FIELD_LABELS[language]
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
    lines.extend(_canonical_section(language, facts, taxonomy))

    lines.extend(
        [
            "",
            f"## {locale['attributes']}",
            "",
            labels["raw_note"],
            "",
            f"| ID | {labels['source_attribute']} | {labels['source_value']} | {labels['evidence']} |",
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
            f"| {labels['type']} | ID | {labels['platform_attribute']} | {labels['source_attribute']} | {labels['source_value']} | Value ID | {labels['platform_value']} |",
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
            labels["missing"] + ": "
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
                f"| {labels['seller_label']} | {labels['metric']} | {labels['imperial']} | {labels['evidence']} |",
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
