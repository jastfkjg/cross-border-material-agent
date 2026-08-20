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
        "source_title": "Source Product Title",
        "product_id": "Product ID",
        "product_url": "Product URL",
        "source_category_id": "Source Category ID",
        "source_category_name": "Source Category Name",
        "leaf_category_id": "Leaf Category ID",
        "leaf_category_name": "Leaf Category Name",
        "leaf_category_path": "Leaf Category Path",
        "raw_note": "Source IDs and evidence pointers are preserved for traceability; labels and values are localized for shoppers.",
        "source_attribute": "Source Attribute",
        "source_value": "Source Value",
        "platform_attribute": "Platform Attribute",
        "platform_value": "Platform Value",
        "evidence": "Evidence",
        "type": "Type",
        "seller_label": "Seller Label",
        "metric": "Metric",
        "imperial": "Imperial",
        "sku_metric": "Seller Guidance (Metric)",
        "sku_imperial": "Seller Guidance (Imperial)",
        "missing": "Required platform fields not present in the source were not fabricated",
        "canonical_raw": "Canonical source value",
        "localized_value": "Localized display value",
        "sku_evidence": "Source evidence",
    },
    "ko": {
        "source_platform": "원본 플랫폼",
        "source_title": "원본 상품명",
        "product_id": "상품 ID",
        "product_url": "상품 URL",
        "source_category_id": "원본 카테고리 ID",
        "source_category_name": "원본 카테고리명",
        "leaf_category_id": "최종 카테고리 ID",
        "leaf_category_name": "최종 카테고리명",
        "leaf_category_path": "최종 카테고리 경로",
        "raw_note": "추적성을 위해 원본 ID와 근거 위치를 유지하고, 필드명과 값은 구매자가 읽기 쉽게 현지화했습니다.",
        "source_attribute": "원본 속성",
        "source_value": "원본 값",
        "platform_attribute": "플랫폼 속성",
        "platform_value": "플랫폼 값",
        "evidence": "근거 위치",
        "type": "유형",
        "seller_label": "판매자 표기",
        "metric": "미터법",
        "imperial": "야드파운드법",
        "sku_metric": "판매자 안내 (미터법)",
        "sku_imperial": "판매자 안내 (야드파운드법)",
        "missing": "원본에 없는 필수 플랫폼 필드는 임의로 생성하지 않았습니다",
        "canonical_raw": "원본 표기값",
        "localized_value": "현지화 표시값",
        "sku_evidence": "원본 근거 위치",
    },
    "pt": {
        "source_platform": "Plataforma de origem",
        "source_title": "Título do produto na origem",
        "product_id": "ID do produto",
        "product_url": "URL do produto",
        "source_category_id": "ID da categoria de origem",
        "source_category_name": "Categoria de origem",
        "leaf_category_id": "ID da categoria final",
        "leaf_category_name": "Categoria final",
        "leaf_category_path": "Caminho da categoria final",
        "raw_note": "Os IDs e os indicadores de evidência foram preservados para rastreabilidade; rótulos e valores estão localizados para o comprador.",
        "source_attribute": "Atributo de origem",
        "source_value": "Valor de origem",
        "platform_attribute": "Atributo da plataforma",
        "platform_value": "Valor da plataforma",
        "evidence": "Evidência",
        "type": "Tipo",
        "seller_label": "Indicação do vendedor",
        "metric": "Métrico",
        "imperial": "Imperial",
        "sku_metric": "Orientação do vendedor (métrico)",
        "sku_imperial": "Orientação do vendedor (imperial)",
        "missing": "Campos obrigatórios ausentes na origem não foram inventados",
        "canonical_raw": "Valor canônico da origem",
        "localized_value": "Valor localizado",
        "sku_evidence": "Evidência na origem",
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


_TERM_TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "123批发网": "123 Wholesale",
        "女式衬衫": "Women's shirts",
        "女式针织衫": "Women's knitwear",
        "女式T恤": "Women's T-shirts",
        "女式毛呢外套": "Women's wool-blend coats",
        "男式夹克": "Men's jackets",
        "男式衬衫": "Men's shirts",
        "男式休闲裤": "Men's casual pants",
        "童T恤": "Children's T-shirts",
        "半身裙": "Skirts",
        "连衣裙": "Dresses",
        "面料名称": "Fabric",
        "面料": "Fabric",
        "主面料成分": "Main material",
        "主面料成分2": "Secondary material",
        "图案": "Pattern",
        "款式": "Style",
        "袖长": "Sleeve length",
        "袖型": "Sleeve type",
        "版型": "Fit",
        "衣长": "Garment length",
        "领型": "Collar",
        "门襟": "Closure",
        "衣门襟": "Closure",
        "颜色": "Color",
        "尺码": "Size",
        "产品类别": "Product type",
        "类别": "Product type",
        "裤型": "Pant shape",
        "裤长": "Pant length",
        "腰型": "Waist rise",
        "裙型": "Skirt shape",
        "裙长": "Skirt length",
        "裙类别": "Skirt type",
        "穿着方式": "Wearing style",
        "穿搭方式": "Styling method",
        "适用性别": "Gender",
        "适合人群": "Intended wearer",
        "化纤": "Synthetic fiber",
        "涤纶（聚酯纤维）": "Polyester",
        "聚酯纤维（涤纶）": "Polyester",
        "聚酯纤维": "Polyester",
        "棉": "Cotton",
        "氨纶": "Spandex",
        "粘胶纤维": "Viscose",
        "纯色": "Solid color",
        "素色": "Solid color",
        "条纹": "Striped",
        "印花": "Printed",
        "几何": "Geometric",
        "长袖": "Long sleeve",
        "短袖": "Short sleeve",
        "常规袖": "Regular sleeve",
        "宽松型": "Relaxed fit",
        "宽松": "Relaxed fit",
        "修身型": "Slim fit",
        "合体型": "Regular fit",
        "POLO领": "Polo collar",
        "马球领": "Polo collar",
        "圆领": "Crew neck",
        "V领": "V-neck",
        "翻领": "Turn-down collar",
        "立领": "Stand collar",
        "高领": "High neck",
        "单排扣": "Single-breasted button closure",
        "纽扣": "Button",
        "拉链": "Zipper",
        "套头": "Pullover",
        "开衫": "Button-front style",
        "紫色": "Purple",
        "紫罗兰": "Violet",
        "白色": "White",
        "黑色": "Black",
        "绿色": "Green",
        "蓝色": "Blue",
        "粉色": "Pink",
        "黄色": "Yellow",
        "灰色": "Gray",
        "深灰色": "Dark gray",
        "卡其色": "Khaki",
        "杏色": "Apricot",
        "薄荷绿": "Mint green",
        "女": "Women",
        "男": "Men",
        "成人": "Adults",
        "常规": "Regular",
        "中长款": "Mid-length",
        "长裙": "Long skirt",
        "A字裙": "A-line skirt",
        "百褶裙": "Pleated skirt",
        "直筒型": "Straight cut",
        "直筒": "Straight cut",
        "中腰": "Mid rise",
        "五分裤": "Knee-length shorts",
        "休闲": "Casual",
        "休闲风": "Casual style",
        "简约": "Minimalist",
    },
    "ko": {
        "123批发网": "123 도매",
        "女式衬衫": "여성 셔츠",
        "女式针织衫": "여성 니트",
        "女式T恤": "여성 티셔츠",
        "女式毛呢外套": "여성 울 혼방 코트",
        "男式夹克": "남성 재킷",
        "男式衬衫": "남성 셔츠",
        "男式休闲裤": "남성 캐주얼 팬츠",
        "童T恤": "아동 티셔츠",
        "半身裙": "스커트",
        "连衣裙": "원피스",
        "面料名称": "원단",
        "面料": "원단",
        "主面料成分": "주요 소재",
        "主面料成分2": "보조 소재",
        "图案": "패턴",
        "款式": "스타일",
        "袖长": "소매 길이",
        "袖型": "소매 형태",
        "版型": "핏",
        "衣长": "총장",
        "领型": "칼라",
        "门襟": "여밈",
        "衣门襟": "여밈",
        "颜色": "색상",
        "尺码": "사이즈",
        "产品类别": "상품 유형",
        "类别": "상품 유형",
        "裤型": "바지 핏",
        "裤长": "바지 길이",
        "腰型": "허리선",
        "裙型": "스커트 형태",
        "裙长": "스커트 길이",
        "裙类别": "스커트 유형",
        "适用性别": "성별",
        "化纤": "합성섬유",
        "涤纶（聚酯纤维）": "폴리에스터",
        "聚酯纤维（涤纶）": "폴리에스터",
        "聚酯纤维": "폴리에스터",
        "棉": "면",
        "氨纶": "스판덱스",
        "粘胶纤维": "비스코스",
        "纯色": "솔리드 컬러",
        "素色": "무지",
        "条纹": "스트라이프",
        "印花": "프린트",
        "几何": "기하학 패턴",
        "长袖": "긴소매",
        "短袖": "반소매",
        "常规袖": "기본 소매",
        "宽松型": "루즈핏",
        "宽松": "루즈핏",
        "修身型": "슬림핏",
        "合体型": "레귤러핏",
        "POLO领": "폴로 칼라",
        "马球领": "폴로 칼라",
        "圆领": "라운드넥",
        "V领": "브이넥",
        "翻领": "테일러드 칼라",
        "立领": "스탠드 칼라",
        "高领": "하이넥",
        "单排扣": "싱글 버튼 여밈",
        "纽扣": "버튼",
        "拉链": "지퍼",
        "套头": "풀오버",
        "开衫": "앞여밈 스타일",
        "紫色": "퍼플",
        "紫罗兰": "바이올렛",
        "白色": "화이트",
        "黑色": "블랙",
        "绿色": "그린",
        "蓝色": "블루",
        "粉色": "핑크",
        "黄色": "옐로",
        "灰色": "그레이",
        "深灰色": "다크 그레이",
        "卡其色": "카키",
        "杏色": "애프리콧",
        "薄荷绿": "민트 그린",
        "女": "여성",
        "男": "남성",
        "成人": "성인",
        "常规": "기본",
        "中长款": "미디 길이",
        "长裙": "롱 스커트",
        "A字裙": "A라인 스커트",
        "百褶裙": "플리츠 스커트",
        "直筒型": "스트레이트 핏",
        "直筒": "스트레이트 핏",
        "中腰": "미드라이즈",
        "五分裤": "무릎 길이 쇼츠",
        "休闲": "캐주얼",
        "休闲风": "캐주얼 스타일",
        "简约": "미니멀",
    },
    "pt": {
        "123批发网": "123 Atacado",
        "女式衬衫": "Camisas femininas",
        "女式针织衫": "Malhas femininas",
        "女式T恤": "Camisetas femininas",
        "女式毛呢外套": "Casacos femininos de mistura de lã",
        "男式夹克": "Jaquetas masculinas",
        "男式衬衫": "Camisas masculinas",
        "男式休闲裤": "Calças casuais masculinas",
        "童T恤": "Camisetas infantis",
        "半身裙": "Saias",
        "连衣裙": "Vestidos",
        "面料名称": "Tecido",
        "面料": "Tecido",
        "主面料成分": "Material principal",
        "主面料成分2": "Material secundário",
        "图案": "Estampa",
        "款式": "Estilo",
        "袖长": "Comprimento da manga",
        "袖型": "Tipo de manga",
        "版型": "Modelagem",
        "衣长": "Comprimento da peça",
        "领型": "Gola",
        "门襟": "Fechamento",
        "衣门襟": "Fechamento",
        "颜色": "Cor",
        "尺码": "Tamanho",
        "产品类别": "Tipo de produto",
        "类别": "Tipo de produto",
        "裤型": "Modelagem da calça",
        "裤长": "Comprimento da calça",
        "腰型": "Altura da cintura",
        "裙型": "Modelagem da saia",
        "裙长": "Comprimento da saia",
        "裙类别": "Tipo de saia",
        "适用性别": "Gênero",
        "化纤": "Fibra sintética",
        "涤纶（聚酯纤维）": "Poliéster",
        "聚酯纤维（涤纶）": "Poliéster",
        "聚酯纤维": "Poliéster",
        "棉": "Algodão",
        "氨纶": "Elastano",
        "粘胶纤维": "Viscose",
        "纯色": "Cor lisa",
        "素色": "Sem estampa",
        "条纹": "Listrado",
        "印花": "Estampado",
        "几何": "Geométrico",
        "长袖": "Manga longa",
        "短袖": "Manga curta",
        "常规袖": "Manga regular",
        "宽松型": "Modelagem solta",
        "宽松": "Modelagem solta",
        "修身型": "Modelagem ajustada",
        "合体型": "Modelagem regular",
        "POLO领": "Gola polo",
        "马球领": "Gola polo",
        "圆领": "Gola redonda",
        "V领": "Decote V",
        "翻领": "Gola dobrável",
        "立领": "Gola alta curta",
        "高领": "Gola alta",
        "单排扣": "Fechamento frontal com botões",
        "纽扣": "Botão",
        "拉链": "Zíper",
        "套头": "Pulôver",
        "开衫": "Abertura frontal",
        "紫色": "Roxo",
        "紫罗兰": "Violeta",
        "白色": "Branco",
        "黑色": "Preto",
        "绿色": "Verde",
        "蓝色": "Azul",
        "粉色": "Rosa",
        "黄色": "Amarelo",
        "灰色": "Cinza",
        "深灰色": "Cinza-escuro",
        "卡其色": "Cáqui",
        "杏色": "Damasco",
        "薄荷绿": "Verde-menta",
        "女": "Feminino",
        "男": "Masculino",
        "成人": "Adultos",
        "常规": "Regular",
        "中长款": "Comprimento médio",
        "长裙": "Saia longa",
        "A字裙": "Saia evasê",
        "百褶裙": "Saia plissada",
        "直筒型": "Corte reto",
        "直筒": "Corte reto",
        "中腰": "Cintura média",
        "五分裤": "Bermuda até o joelho",
        "休闲": "Casual",
        "休闲风": "Estilo casual",
        "简约": "Minimalista",
    },
}


_MARKETING_ATTRIBUTE_NAMES = (
    "主面料成分",
    "图案",
    "版型",
    "袖长",
    "领型",
    "门襟",
    "衣门襟",
    "衣长",
    "裤型",
    "裤长",
    "裙型",
    "裙长",
    "面料",
    "面料名称",
)


def _source_terms(facts: ProductFacts, taxonomy: TaxonomyResult) -> set[str]:
    terms = {
        facts.platform,
        facts.source_category_name,
        taxonomy.category.name,
        taxonomy.category.path,
    }
    for item in facts.attributes:
        terms.update((item.name, item.value))
    for item in taxonomy.attributes:
        terms.update(
            (item.name, item.source_name, item.source_value, item.platform_value)
        )
    for sku in facts.skus:
        for item in sku.attributes:
            terms.update((item.name, item.value))
    terms.update(item.source_label for item in facts.size_conversions)
    return {term.strip() for term in terms if term and re.search(r"[\u4e00-\u9fff]", term)}


def _static_localize_term(language: str, value: str) -> str:
    raw = str(value).strip()
    if not raw or not re.search(r"[\u4e00-\u9fff]", raw):
        return raw
    translations = _TERM_TRANSLATIONS[language]
    if raw in translations:
        return translations[raw]
    rendered = raw
    for source, target in sorted(
        translations.items(), key=lambda item: len(item[0]), reverse=True
    ):
        rendered = rendered.replace(source, target)
    rendered = re.sub(
        r"(?P<low>\d+(?:\.\d+)?)\s*[-~～至]\s*(?P<high>\d+(?:\.\d+)?)\s*斤",
        lambda match: (
            f"{match.group('low')}–{match.group('high')} jin"
            if language in {"en", "pt"}
            else f"{match.group('low')}–{match.group('high')} 중국 진"
        ),
        rendered,
    )
    if re.search(r"[\u4e00-\u9fff]", rendered):
        return {
            "en": "Seller-declared source value",
            "ko": "판매자 원본 표기값",
            "pt": "Valor informado pelo vendedor",
        }[language]
    return rendered


def _fallback_term_map(
    language: str, facts: ProductFacts, taxonomy: TaxonomyResult
) -> dict[str, str]:
    return {
        term: _static_localize_term(language, term)
        for term in sorted(_source_terms(facts, taxonomy))
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
    expected_terms: set[str],
) -> str:
    required = {
        "title",
        "overview",
        "highlights",
        "fit_note",
        "media_descriptions",
        "localized_terms",
    }
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
    localized_terms = payload.get("localized_terms")
    if (
        not isinstance(localized_terms, dict)
        or set(localized_terms) != expected_terms
        or not all(
            isinstance(item, str) and item.strip()
            for item in localized_terms.values()
        )
    ):
        return "invalid-localized-terms"
    generated_text = "\n".join(
        [
            str(payload.get("title") or ""),
            str(payload.get("overview") or ""),
            str(payload.get("fit_note") or ""),
            *[str(item) for item in payload.get("highlights", [])],
            *[str(item) for item in media.values()],
            *[str(item) for item in localized_terms.values()],
        ]
    )
    natural_text = "\n".join(
        [
            str(payload.get("title") or ""),
            str(payload.get("overview") or ""),
            str(payload.get("fit_note") or ""),
            *[str(item) for item in payload.get("highlights", [])],
        ]
    )
    if "\n" in str(payload.get("title") or "") or len(
        str(payload.get("title") or "").strip()
    ) > 160:
        return "localized-title-guard"
    if re.search(r"[\u4e00-\u9fff]", generated_text):
        return "source-script-contamination-guard"
    if language == "ko" and not re.search(r"[\uac00-\ud7a3]", natural_text):
        return "ko-KR-native-language-guard"
    if language in {"en", "pt"} and re.search(r"[\uac00-\ud7a3]", natural_text):
        return f"{language}-native-language-guard"
    if language == "pt" and re.search(
        r"(?i)\b(?:rapariga|telemóvel|ficheiro|autocarro|fato de banho)\b",
        natural_text,
    ):
        return "pt-BR-variant-guard"
    localized_feature_values = {
        str(localized_terms.get(item.value) or "").strip().casefold()
        for item in facts.attributes
        if item.name in _MARKETING_ATTRIBUTE_NAMES
    }
    localized_feature_values.discard("")
    localized_feature_values.difference_update(
        {
            "seller-declared source value",
            "판매자 원본 표기값",
            "valor informado pelo vendedor",
        }
    )
    natural_folded = natural_text.casefold()
    matched_features = sum(
        value in natural_folded for value in localized_feature_values
    )
    if matched_features < min(3, len(localized_feature_values)):
        return "insufficient-verified-details"
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
    term_map = _fallback_term_map(language, facts, taxonomy)
    base_title = _FALLBACK_TITLES.get(language, {}).get(
        taxonomy.category.category_id,
        _static_localize_term(
            language, taxonomy.category.name or facts.source_category_name
        ),
    )
    selected_features: list[tuple[str, str]] = []
    seen_values: set[str] = set()
    for attribute_name in _MARKETING_ATTRIBUTE_NAMES:
        item = next(
            (attribute for attribute in facts.attributes if attribute.name == attribute_name),
            None,
        )
        if item is None:
            continue
        localized_name = term_map.get(
            item.name, _static_localize_term(language, item.name)
        )
        localized_value = term_map.get(
            item.value, _static_localize_term(language, item.value)
        )
        if (
            localized_value in seen_values
            or localized_value
            in {
                "Seller-declared source value",
                "판매자 원본 표기값",
                "Valor informado pelo vendedor",
            }
        ):
            continue
        selected_features.append((localized_name, localized_value))
        seen_values.add(localized_value)
        if len(selected_features) == 5:
            break

    title_values = [value for _, value in selected_features[:4]]
    payload["title"] = (
        f"{base_title} — {', '.join(title_values)}" if title_values else base_title
    )
    if language == "ko" and title_values:
        payload["title"] = f"{base_title} · {' · '.join(title_values)}"

    feature_summary = ", ".join(value for _, value in selected_features[:3])
    if language == "en":
        payload["overview"] = (
            f"{base_title} with source-verified details including {feature_summary}. "
            "Choose the color and size combination from the SKU matrix below."
            if feature_summary
            else f"{base_title} offered in the seller-declared options shown below."
        )
    elif language == "ko":
        payload["overview"] = (
            f"{feature_summary} 사양이 원본 데이터에서 확인된 {base_title}입니다. "
            "아래 SKU 구성표에서 색상과 사이즈 조합을 확인해 주세요."
            if feature_summary
            else f"판매자 원본 옵션으로 구성된 {base_title}입니다."
        )
    else:
        payload["overview"] = (
            f"{base_title} com características confirmadas na fonte: {feature_summary}. "
            "Consulte a matriz de SKUs para escolher a combinação de cor e tamanho."
            if feature_summary
            else f"{base_title} disponível nas opções declaradas pelo vendedor."
        )
    payload["highlights"] = [
        f"{name}: {value}" for name, value in selected_features
    ]
    if len(payload["highlights"]) < 3:
        payload["highlights"] = list(_FALLBACK_CONTENT[language]["highlights"])
    payload["localized_terms"] = term_map
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
    source_terms = _source_terms(facts, taxonomy)
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
The title, overview and highlights together must name at least three concrete verified product
attributes. Do not use process-oriented filler such as "source-grounded details" or describe the
fact-checking workflow to the shopper.
Use product-first phrasing natural to {locale["locale"]}; avoid translated syntax, keyword stuffing,
generic filler and mixed-language fragments. Keep the title under 160 characters.
For en-US, use US spelling and concise marketplace phrasing. For ko-KR, use natural Korean retail
sentence endings and Korean option terminology. For pt-BR, use Brazilian vocabulary and forms such
as produto, tamanho, camiseta and consulte; avoid European Portuguese vocabulary.
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
- localized_terms: object whose keys exactly match every source term in the list below and whose values are concise native translations. Keep model numbers, IDs and size codes unchanged, but translate all Chinese words.

Source terms requiring localized display values:
{json.dumps(sorted(source_terms), ensure_ascii=False)}

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
The result must contain exactly the requested six top-level keys and exactly the seven required media keys.
Do not copy Chinese characters into any customer-facing value. The localized_terms keys are the sole
exception: preserve those exact source keys, while translating every localized_terms value.
Apply native {locale["locale"]} grammar and retail terminology, not literal source-language word order.
""".strip()
    audit_prompt = f"""
Audit and, where needed, correct this candidate payload:
{json.dumps(draft, ensure_ascii=False)}

Verified source facts:
{json.dumps(facts.compact_dict(), ensure_ascii=False)}

Resolved AliExpress category:
{json.dumps({"id": taxonomy.category.category_id, "name": taxonomy.category.name, "path": taxonomy.category.path}, ensure_ascii=False)}

Required top-level keys: title, overview, highlights, fit_note, media_descriptions, localized_terms.
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
    expected_terms = set(fallback["localized_terms"])
    validation_error = _payload_validation_error(
        language, payload, facts, taxonomy, expected_media, expected_terms
    )
    repaired = False
    if validation_error:
        repair_prompt = f"""
Repair the candidate payload because it failed this deterministic check: {validation_error}.
Return the same exact schema in natural {locale["locale"]}. Translate all source concepts fully;
do not copy Chinese characters into values. Preserve the exact Chinese localized_terms keys listed below.
Remove unsupported numbers and claims instead of replacing them with new claims.
The localized_terms object must contain exactly these source keys:
{json.dumps(sorted(expected_terms), ensure_ascii=False)}

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
            language, payload, facts, taxonomy, expected_media, expected_terms
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


def _localized_display(language: str, value: str, term_map: dict[str, Any]) -> str:
    raw = str(value).strip()
    translated = term_map.get(raw)
    if isinstance(translated, str) and translated.strip():
        return translated.strip()
    return _static_localize_term(language, raw)


def _canonical_section(
    language: str,
    facts: ProductFacts,
    taxonomy: TaxonomyResult,
    term_map: dict[str, Any],
) -> list[str]:
    category = taxonomy.category
    labels = FIELD_LABELS[language]
    localized_platform = _localized_display(language, facts.platform, term_map)
    localized_source_category = _localized_display(
        language, facts.source_category_name, term_map
    )
    localized_leaf_name = _localized_display(language, category.name, term_map)
    localized_leaf_path = _localized_display(language, category.path, term_map)
    return [
        f"- **{labels['source_platform']}:** {_escape_table(facts.platform)}"
        + (
            f" ({_escape_table(localized_platform)})"
            if localized_platform != facts.platform
            else ""
        ),
        f"- **{labels['product_id']}:** {_escape_table(facts.offer_id)}",
        f"- **{labels['product_url']}:** {facts.source_url}",
        f"- **{labels['source_title']}:** {_escape_table(facts.source_title)}",
        f"- **{labels['source_category_id']}:** {_escape_table(facts.source_category_id)}",
        f"- **{labels['source_category_name']}:** {_escape_table(facts.source_category_name)}"
        + (
            f" ({_escape_table(localized_source_category)})"
            if localized_source_category != facts.source_category_name
            else ""
        ),
        f"- **{labels['leaf_category_id']}:** {_escape_table(category.category_id)}",
        f"- **{labels['leaf_category_name']}:** {_escape_table(category.name)}"
        + (
            f" ({_escape_table(localized_leaf_name)})"
            if localized_leaf_name != category.name
            else ""
        ),
        f"- **{labels['leaf_category_path']}:** {_escape_table(category.path)}"
        + (
            f" ({_escape_table(localized_leaf_path)})"
            if localized_leaf_path != category.path
            else ""
        ),
    ]


def _sku_size_guidance(facts: ProductFacts, values: dict[str, str]) -> tuple[str, str]:
    """Return only deterministic unit conversions tied to this exact SKU label."""

    if not facts.size_conversions:
        return "", ""
    sku_values = {str(value).strip().casefold() for value in values.values() if value}
    for conversion in facts.size_conversions:
        source_label = conversion.source_label.strip().casefold()
        if source_label in sku_values:
            return conversion.kilograms, conversion.pounds
    return "", ""


def render_description(
    language: str,
    payload: dict[str, Any],
    facts: ProductFacts,
    taxonomy: TaxonomyResult,
) -> str:
    locale = LANGUAGES[language]
    labels = FIELD_LABELS[language]
    raw_term_map = payload.get("localized_terms")
    term_map = raw_term_map if isinstance(raw_term_map, dict) else {}
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
    lines.extend(_canonical_section(language, facts, taxonomy, term_map))

    lines.extend(
        [
            "",
            f"## {locale['attributes']}",
            "",
            labels["raw_note"],
            "",
            f"| ID | {labels['source_attribute']} | {labels['canonical_raw']} | {labels['localized_value']} | {labels['evidence']} |",
            "|---|---|---|---|---|",
        ]
    )
    for item in facts.attributes:
        lines.append(
            f"| {_escape_table(item.attribute_id)} | {_escape_table(item.name)} / "
            f"{_escape_table(_localized_display(language, item.name, term_map))} | {_escape_table(item.value)} | "
            f"{_escape_table(_localized_display(language, item.value, term_map))} | `{_escape_table(item.evidence_pointer)}` |"
        )

    lines.extend(
        [
            "",
            f"## {locale['platform_attributes']}",
            "",
            f"| {labels['type']} | ID | {labels['platform_attribute']} | {labels['source_attribute']} | {labels['canonical_raw']} | Value ID | {labels['platform_value']} |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for item in taxonomy.attributes:
        item_type = "sales" if item.sales_attribute else "product"
        lines.append(
            f"| {item_type} | {_escape_table(item.attr_id)} | {_escape_table(item.name)} / "
            f"{_escape_table(_localized_display(language, item.name, term_map))} | {_escape_table(item.source_name)} | "
            f"{_escape_table(item.source_value)} | {_escape_table(item.value_id)} | "
            f"{_escape_table(item.platform_value)} / {_escape_table(_localized_display(language, item.platform_value, term_map))} |"
        )
    if taxonomy.missing_required:
        lines.append("")
        lines.append(
            labels["missing"] + ": "
            + ", ".join(
                _escape_table(_localized_display(language, item, term_map))
                for item in taxonomy.missing_required
            )
        )

    all_sku_names: list[str] = []
    for sku in facts.skus:
        for item in sku.attributes:
            if item.name and item.name not in all_sku_names:
                all_sku_names.append(item.name)
    lines.extend(["", f"## {locale['skus']}", ""])
    lines.append(
        "| SKU ID | Spec ID | "
        + " | ".join(
            _escape_table(_localized_display(language, name, term_map))
            for name in all_sku_names
        )
        + (
            f" | {labels['sku_metric']} | {labels['sku_imperial']}"
            if facts.size_conversions
            else ""
        )
        + f" | {labels['sku_evidence']}"
        + " |"
    )
    lines.append(
        "|---|---|"
        + "---|" * len(all_sku_names)
        + ("---|---|" if facts.size_conversions else "")
        + "---|"
    )
    for sku in facts.skus:
        values = {item.name: item.value for item in sku.attributes}
        row = [sku.sku_id, sku.spec_id] + [
            (
                f"{values.get(name, '')} / {_localized_display(language, values.get(name, ''), term_map)}"
                if values.get(name, "")
                and _localized_display(language, values.get(name, ""), term_map)
                != values.get(name, "")
                else values.get(name, "")
            )
            for name in all_sku_names
        ]
        if facts.size_conversions:
            metric, imperial = _sku_size_guidance(facts, values)
            row.extend([metric, imperial])
        row.append(sku.evidence_pointer)
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
                f"| {_escape_table(_localized_display(language, size.source_label, term_map))} | {_escape_table(size.kilograms)} | "
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
