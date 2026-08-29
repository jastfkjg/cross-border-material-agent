"""Localized shopper copy plus compact, publishable listing fields."""

from __future__ import annotations

import json
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from .api import ApiError, QwenClient
from .claims import buyer_safe_source_name, publishable_claims
from .compliance import generated_copy_violations
from .models import (
    ClaimEvidence,
    CreativePlan,
    ProductAttribute,
    ProductFacts,
    TaxonomyResult,
)


LANGUAGES: dict[str, dict[str, str]] = {
    "en": {
        "locale": "en-US",
        "description": "natural US English marketplace copy",
        "overview": "Product description",
        "highlights": "Key features",
        "appendix": "Listing information",
        "canonical": "Source and category",
        "attributes": "Product specifications",
        "platform_attributes": "AliExpress listing attributes",
        "skus": "Available variants",
        "sku_components": "SKU components",
        "sizes": "Size and fit",
        "size_chart": "Garment measurements from the seller's size chart",
        "media": "Media guide",
        "note": "Important factual note",
        "color_note": "Colors may appear slightly different depending on lighting and display settings.",
    },
    "ko": {
        "locale": "ko-KR",
        "description": "자연스럽고 신뢰감 있는 한국 온라인 쇼핑몰 문체",
        "overview": "상품 설명",
        "highlights": "주요 특징",
        "appendix": "등록 정보",
        "canonical": "상품 출처 및 카테고리",
        "attributes": "상품 사양",
        "platform_attributes": "AliExpress 등록 속성",
        "skus": "구매 가능 옵션",
        "sku_components": "SKU 구성",
        "sizes": "사이즈 및 핏",
        "size_chart": "판매자 사이즈표의 실측 정보",
        "media": "미디어 안내",
        "note": "상품 정보 안내",
        "color_note": "조명과 화면 설정에 따라 실제 색상이 다르게 보일 수 있습니다.",
    },
    "pt": {
        "locale": "pt-BR",
        "description": "português brasileiro natural para comércio eletrônico",
        "overview": "Descrição do produto",
        "highlights": "Principais características",
        "appendix": "Informações do anúncio",
        "canonical": "Origem e categoria",
        "attributes": "Especificações do produto",
        "platform_attributes": "Atributos para cadastro no AliExpress",
        "skus": "Variações disponíveis",
        "sku_components": "Composição dos SKUs",
        "sizes": "Tamanho e caimento",
        "size_chart": "Medidas da peça informadas na tabela do vendedor",
        "media": "Guia de mídia",
        "note": "Observação factual importante",
        "color_note": "As cores podem variar ligeiramente conforme a iluminação e a configuração da tela.",
    },
}


FIELD_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "source_platform": "Source Platform",
        "product_id": "Product ID",
        "product_url": "Source URL",
        "source_category_name": "Source Category Name",
        "leaf_category_id": "Leaf Category ID",
        "leaf_category_name": "Leaf Category Name",
        "leaf_category_path": "Leaf Category Path",
        "seller_label": "Seller Label",
        "metric": "Metric",
        "imperial": "Imperial",
        "size": "Size",
        "bust": "Bust",
        "garment_length": "Garment length",
        "seller_weight": "Seller weight guidance",
    },
    "ko": {
        "source_platform": "원본 플랫폼",
        "product_id": "상품 ID",
        "product_url": "출처 URL",
        "source_category_name": "원본 카테고리명",
        "leaf_category_id": "최종 카테고리 ID",
        "leaf_category_name": "최종 카테고리명",
        "leaf_category_path": "최종 카테고리 경로",
        "seller_label": "판매자 표기",
        "metric": "미터법",
        "imperial": "야드파운드법",
        "size": "사이즈",
        "bust": "가슴둘레",
        "garment_length": "총장",
        "seller_weight": "판매자 권장 체중",
    },
    "pt": {
        "source_platform": "Plataforma de origem",
        "product_id": "ID do produto",
        "product_url": "URL de origem",
        "source_category_name": "Categoria de origem",
        "leaf_category_id": "ID da categoria final",
        "leaf_category_name": "Categoria final",
        "leaf_category_path": "Caminho da categoria final",
        "seller_label": "Indicação do vendedor",
        "metric": "Métrico",
        "imperial": "Imperial",
        "size": "Tamanho",
        "bust": "Busto",
        "garment_length": "Comprimento da peça",
        "seller_weight": "Peso indicado pelo vendedor",
    },
}


_FALLBACK_CONTENT: dict[str, dict[str, Any]] = {
    "en": {
        "overview": "Explore the product details and available options below.\n\nReview the seller's size guidance before ordering; regional size equivalence is not assumed.",
        "highlights": [
            "Product type and construction are listed in the specifications.",
            "Available colors and sizes are shown in the variant table.",
            "Seller-provided size guidance is retained without inventing regional equivalents.",
        ],
        "fit_note": "Body measurements are not provided; seller size labels are retained without inferred regional equivalents.",
    },
    "ko": {
        "overview": "상품의 주요 디테일과 구매 가능한 옵션을 아래에서 확인해 보세요.\n\n구매 전 판매자 사이즈 안내를 확인해 주세요. 지역별 사이즈로 임의 환산하지 않았습니다.",
        "highlights": [
            "상품 종류와 구조는 상품 사양에서 확인할 수 있습니다.",
            "구매 가능한 색상과 사이즈는 옵션 표에 정리했습니다.",
            "지역별 사이즈를 임의로 적용하지 않고 판매자 안내를 표시했습니다.",
        ],
        "fit_note": "신체 치수 정보가 제공되지 않아 판매자 사이즈 표기를 유지하고 한국 사이즈로 임의 환산하지 않았습니다.",
    },
    "pt": {
        "overview": "Confira os principais detalhes do produto e as opções disponíveis abaixo.\n\nConsulte a orientação de tamanho do vendedor antes da compra; não presumimos equivalência regional.",
        "highlights": [
            "O tipo e a construção do produto aparecem nas especificações.",
            "As cores e os tamanhos disponíveis estão na tabela de variações.",
            "A orientação do vendedor é mantida sem inventar equivalências regionais.",
        ],
        "fit_note": "As medidas corporais não foram informadas; mantivemos os tamanhos do vendedor sem conversão automática para P, M ou G.",
    },
}


_TERM_TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "123批发网": "123 Wholesale",
        "T恤": "T-shirt",
        "面料名称": "Fabric",
        "面料": "Fabric",
        "主面料成分": "Main material",
        "主面料成分2": "Secondary material",
        "材质": "Material",
        "主面料成分含量": "Main material content",
        "主面料成分2含量": "Secondary material content",
        "上市年份/季节": "Year and season",
        "上市年份季节": "Year and season",
        "适合季节": "Season",
        "图案": "Pattern",
        "款式": "Style",
        "袖长": "Sleeve length",
        "袖型": "Sleeve type",
        "版型": "Fit",
        "衣长": "Garment length",
        "领型": "Neckline",
        "门襟": "Closure",
        "衣门襟": "Closure",
        "颜色": "Color",
        "尺码": "Size",
        "产品类别": "Product type",
        "类别": "Product type",
        "裤型": "Pant shape",
        "裤长": "Pant length",
        "腰型": "Waist rise",
        "腰线": "Waist rise",
        "裙型": "Skirt shape",
        "裙长": "Skirt length",
        "裙类别": "Skirt type",
        "工艺": "Construction detail",
        "弹力": "Stretch",
        "风格类型": "Style type",
        "风格": "Style",
        "适用场景": "Applicable occasion",
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
        "粘胶": "Viscose",
        "针织面料": "Knit fabric",
        "冰丝": "Ice-silk fabric",
        "雪纺": "Chiffon",
        "纯色": "Solid color",
        "素色": "Solid color",
        "条纹": "Striped",
        "印花": "Printed",
        "百褶": "Pleating",
        "无弹": "No stretch",
        "日式": "Japanese",
        "韩语": "Korean",
        "几何": "Geometric",
        "3D效果": "3D-effect print",
        "3D/立体图案": "3D-effect pattern",
        "卡通人物": "Cartoon-character print",
        "图片色": "Color shown",
        "花色": "Multicolor print",
        "长袖": "Long sleeve",
        "短袖": "Short sleeve",
        "常规袖": "Regular sleeve",
        "宽松型": "Relaxed fit",
        "宽松": "Relaxed fit",
        "修身型": "Slim fit",
        "修身": "Slim fit",
        "合体型": "Regular fit",
        "POLO领": "Polo collar",
        "马球领": "Polo collar",
        "圆领": "Crew neck",
        "V领": "V-neck",
        "翻领": "Turn-down collar",
        "立领": "Stand collar",
        "高领": "High neck",
        "挂脖": "Halter neck",
        "露肩": "Cold-shoulder design",
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
        "军绿色": "Army green",
        "焦糖色": "Caramel",
        "燕麦色": "Oatmeal",
        "雾霾蓝": "Dusty blue",
        "女": "Women",
        "男": "Men",
        "成人": "Adults",
        "常规": "Regular",
        "大码": "Plus size",
        "均码": "One Size",
        "基本款": "Basic style",
        "针织": "Knit construction",
        "气质通勤": "Polished workwear",
        "通勤风": "Workwear style",
        "适中": "Medium",
        "30%以下": "Under 30%",
        "无领标": "No neck label",
        "无吊牌": "No hangtag",
        "否": "No",
        "毛衣": "Sweater",
        "针织衫": "Knit top",
        "其他品牌": "Other brand",
        "通用": "General",
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
        "优雅风": "Elegant",
        "运动风": "Sporty",
        "文艺复古": "Art-inspired retro",
        "气质优雅": "Elegant",
        "春秋季": "Spring and fall",
        "春季": "Spring",
        "夏季": "Summer",
        "夏": "Summer",
        "秋季": "Fall",
        "秋": "Fall",
        "冬季": "Winter",
        "四季": "All seasons",
        "码": " size",
        "高温定型": "Heat setting",
        "其他": "Unbranded / other",
        "中": "Regular length",
        "普通款(50cm<衣长≤65cm)": "Regular length (over 50 cm and up to 65 cm)",
        "日韩休闲": "Korean/Japanese casual",
        "舒适休闲": "Relaxed casual style",
        "是": "Yes",
        "衬衫": "Shirt",
        "中东": "Middle East",
        "东南亚": "Southeast Asia",
        "50%（含）-70%（不含）": "50% to under 70%",
        "30%（含）-50%（不含）": "30% to under 50%",
        "包容度": "Fit",
        "图案花纹": "Pattern",
        "衣领类型": "Collar type",
        "季节": "Season",
        "设计": "Design",
        "尺码类型": "Size type",
        "场合": "Occasion",
        "毛呢": "Woolen",
        "羊毛": "Wool",
        "混纺": "Blend",
        "兔毛": "Rabbit hair",
        "人造皮毛": "Faux fur",
        "人造毛": "Faux fur",
        "仿皮草": "Faux fur",
        "皮草": "Fur",
        "短款": "Short length",
    },
    "ko": {
        "123批发网": "123 도매",
        "T恤": "티셔츠",
        "面料名称": "원단",
        "面料": "원단",
        "主面料成分": "주요 소재",
        "主面料成分2": "보조 소재",
        "材质": "소재",
        "主面料成分含量": "주요 소재 함량",
        "主面料成分2含量": "보조 소재 함량",
        "上市年份/季节": "출시 연도/계절",
        "上市年份季节": "출시 연도/계절",
        "适合季节": "계절",
        "图案": "패턴",
        "款式": "스타일",
        "袖长": "소매 길이",
        "袖型": "소매 형태",
        "版型": "핏",
        "衣长": "총장",
        "领型": "네크라인",
        "门襟": "여밈",
        "衣门襟": "여밈",
        "颜色": "색상",
        "尺码": "사이즈",
        "产品类别": "상품 유형",
        "类别": "상품 유형",
        "裤型": "바지 핏",
        "裤长": "바지 길이",
        "腰型": "허리선",
        "腰线": "허리선",
        "裙型": "스커트 형태",
        "裙长": "스커트 길이",
        "裙类别": "스커트 유형",
        "工艺": "제작 디테일",
        "弹力": "신축성",
        "风格类型": "스타일 유형",
        "风格": "스타일",
        "适用场景": "적용 상황",
        "适用性别": "성별",
        "化纤": "합성섬유",
        "涤纶（聚酯纤维）": "폴리에스터",
        "聚酯纤维（涤纶）": "폴리에스터",
        "聚酯纤维": "폴리에스터",
        "棉": "면",
        "氨纶": "스판덱스",
        "粘胶纤维": "비스코스",
        "粘胶": "비스코스",
        "针织面料": "니트 원단",
        "冰丝": "아이스 실크 원단",
        "雪纺": "시폰",
        "纯色": "솔리드 컬러",
        "素色": "무지",
        "条纹": "스트라이프",
        "印花": "프린트",
        "百褶": "플리츠",
        "无弹": "신축성 없음",
        "日式": "일본풍",
        "韩语": "한국풍",
        "几何": "기하학 패턴",
        "3D效果": "3D 효과 프린트",
        "3D/立体图案": "입체 패턴",
        "卡通人物": "캐릭터 프린트",
        "图片色": "이미지 표시 색상",
        "花色": "멀티컬러 프린트",
        "长袖": "긴소매",
        "短袖": "반소매",
        "常规袖": "기본 소매",
        "宽松型": "루즈핏",
        "宽松": "루즈핏",
        "修身型": "슬림핏",
        "修身": "슬림핏",
        "合体型": "레귤러핏",
        "POLO领": "폴로 칼라",
        "马球领": "폴로 칼라",
        "圆领": "라운드넥",
        "V领": "브이넥",
        "翻领": "테일러드 칼라",
        "立领": "스탠드 칼라",
        "高领": "하이넥",
        "挂脖": "홀터넥",
        "露肩": "숄더 컷아웃 디자인",
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
        "军绿色": "아미 그린",
        "焦糖色": "캐러멜",
        "燕麦色": "오트밀",
        "雾霾蓝": "더스티 블루",
        "女": "여성",
        "男": "남성",
        "成人": "성인",
        "常规": "기본",
        "大码": "플러스 사이즈",
        "均码": "프리사이즈",
        "基本款": "베이직 스타일",
        "针织": "니트 짜임",
        "气质通勤": "세련된 오피스룩",
        "通勤风": "오피스룩 스타일",
        "适中": "보통",
        "30%以下": "30% 미만",
        "无领标": "넥 라벨 없음",
        "无吊牌": "행택 없음",
        "否": "아니요",
        "毛衣": "스웨터",
        "针织衫": "니트 톱",
        "其他品牌": "기타 브랜드",
        "通用": "공용",
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
        "优雅风": "우아한 스타일",
        "运动风": "스포티 스타일",
        "文艺复古": "감성적인 레트로 스타일",
        "气质优雅": "우아한 스타일",
        "春秋季": "봄·가을",
        "春季": "봄",
        "夏季": "여름",
        "夏": "여름",
        "秋季": "가을",
        "秋": "가을",
        "冬季": "겨울",
        "四季": "사계절",
        "码": " 사이즈",
        "高温定型": "고온 열세팅",
        "中": "기본 길이",
        "其他": "기타/무브랜드",
        "普通款(50cm<衣长≤65cm)": "기본 길이(50 cm 초과~65 cm 이하)",
        "日韩休闲": "한·일 캐주얼",
        "舒适休闲": "편안한 캐주얼 스타일",
        "是": "예",
        "衬衫": "셔츠",
        "中东": "중동",
        "东南亚": "동남아시아",
        "50%（含）-70%（不含）": "50% 이상 70% 미만",
        "30%（含）-50%（不含）": "30% 이상 50% 미만",
        "包容度": "핏",
        "图案花纹": "패턴",
        "衣领类型": "칼라 유형",
        "季节": "계절",
        "设计": "디자인",
        "尺码类型": "사이즈 유형",
        "场合": "착용 상황",
        "毛呢": "울 소재",
        "羊毛": "울",
        "混纺": "혼방",
        "兔毛": "토끼털",
        "人造皮毛": "인조 퍼",
        "人造毛": "인조 퍼",
        "仿皮草": "인조 퍼",
        "皮草": "퍼",
        "短款": "짧은 기장",
    },
    "pt": {
        "123批发网": "123 Atacado",
        "T恤": "Camiseta",
        "面料名称": "Tecido",
        "面料": "Tecido",
        "主面料成分": "Material principal",
        "主面料成分2": "Material secundário",
        "材质": "Material",
        "主面料成分含量": "Teor do material principal",
        "主面料成分2含量": "Teor do material secundário",
        "上市年份/季节": "Ano e estação",
        "上市年份季节": "Ano e estação",
        "适合季节": "Estação",
        "图案": "Estampa",
        "款式": "Estilo",
        "袖长": "Comprimento da manga",
        "袖型": "Tipo de manga",
        "版型": "Modelagem",
        "衣长": "Comprimento da peça",
        "领型": "Decote",
        "门襟": "Fechamento",
        "衣门襟": "Fechamento",
        "颜色": "Cor",
        "尺码": "Tamanho",
        "产品类别": "Tipo de produto",
        "类别": "Tipo de produto",
        "裤型": "Modelagem da calça",
        "裤长": "Comprimento da calça",
        "腰型": "Altura da cintura",
        "腰线": "Altura da cintura",
        "裙型": "Modelagem da saia",
        "裙长": "Comprimento da saia",
        "裙类别": "Tipo de saia",
        "工艺": "Detalhe de construção",
        "弹力": "Elasticidade",
        "风格类型": "Tipo de estilo",
        "风格": "Estilo",
        "适用场景": "Ocasião aplicável",
        "适用性别": "Gênero",
        "化纤": "Fibra sintética",
        "涤纶（聚酯纤维）": "Poliéster",
        "聚酯纤维（涤纶）": "Poliéster",
        "聚酯纤维": "Poliéster",
        "棉": "Algodão",
        "氨纶": "Elastano",
        "粘胶纤维": "Viscose",
        "粘胶": "Viscose",
        "针织面料": "Malha",
        "冰丝": "Tecido ice silk",
        "雪纺": "Chiffon",
        "纯色": "Cor lisa",
        "素色": "Sem estampa",
        "条纹": "Listrado",
        "印花": "Estampado",
        "百褶": "Plissado",
        "无弹": "Sem elasticidade",
        "日式": "Japonês",
        "韩语": "Coreano",
        "几何": "Geométrico",
        "3D效果": "Estampa com efeito 3D",
        "3D/立体图案": "Estampa com efeito 3D",
        "卡通人物": "Estampa de personagem",
        "图片色": "Cor mostrada",
        "花色": "Estampa multicolorida",
        "长袖": "Manga longa",
        "短袖": "Manga curta",
        "常规袖": "Manga regular",
        "宽松型": "Modelagem solta",
        "宽松": "Modelagem solta",
        "修身型": "Modelagem ajustada",
        "修身": "Modelagem ajustada",
        "合体型": "Modelagem regular",
        "POLO领": "Gola polo",
        "马球领": "Gola polo",
        "圆领": "Gola redonda",
        "V领": "Decote V",
        "翻领": "Gola dobrável",
        "立领": "Gola alta curta",
        "高领": "Gola alta",
        "挂脖": "Gola halter",
        "露肩": "Design com recorte nos ombros",
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
        "军绿色": "Verde militar",
        "焦糖色": "Caramelo",
        "燕麦色": "Aveia",
        "雾霾蓝": "Azul acinzentado",
        "女": "Feminino",
        "男": "Masculino",
        "成人": "Adultos",
        "常规": "Regular",
        "大码": "Tamanho plus size",
        "均码": "Tamanho único",
        "基本款": "Modelo básico",
        "针织": "Construção em malha",
        "气质通勤": "Visual de trabalho elegante",
        "通勤风": "Estilo para o trabalho",
        "适中": "Médio",
        "30%以下": "Menos de 30%",
        "无领标": "Sem etiqueta na gola",
        "无吊牌": "Sem tag",
        "否": "Não",
        "毛衣": "Suéter",
        "针织衫": "Blusa de malha",
        "其他品牌": "Outra marca",
        "通用": "Uso geral",
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
        "优雅风": "Estilo elegante",
        "运动风": "Estilo esportivo",
        "文艺复古": "Estilo retrô",
        "气质优雅": "Estilo elegante",
        "春秋季": "Primavera e outono",
        "春季": "Primavera",
        "夏季": "Verão",
        "夏": "Verão",
        "秋季": "Outono",
        "秋": "Outono",
        "冬季": "Inverno",
        "四季": "Todas as estações",
        "码": " tamanho",
        "高温定型": "Termofixação",
        "中": "Comprimento regular",
        "其他": "Sem marca/outro",
        "普通款(50cm<衣长≤65cm)": "Comprimento regular (acima de 50 cm e até 65 cm)",
        "日韩休闲": "Casual coreano/japonês",
        "舒适休闲": "Estilo casual descontraído",
        "是": "Sim",
        "衬衫": "Camisa",
        "中东": "Oriente Médio",
        "东南亚": "Sudeste Asiático",
        "50%（含）-70%（不含）": "De 50% a menos de 70%",
        "30%（含）-50%（不含）": "De 30% a menos de 50%",
        "包容度": "Modelagem",
        "图案花纹": "Estampa",
        "衣领类型": "Tipo de gola",
        "季节": "Estação",
        "设计": "Design",
        "尺码类型": "Tipo de tamanho",
        "场合": "Ocasião",
        "毛呢": "Lã",
        "羊毛": "Lã",
        "混纺": "Mistura",
        "兔毛": "Pelo de coelho",
        "人造皮毛": "Pele sintética",
        "人造毛": "Pele sintética",
        "仿皮草": "Pele sintética",
        "皮草": "Pele",
        "短款": "Comprimento curto",
    },
}


_MARKETING_ATTRIBUTE_NAMES = (
    "图案",
    "领型",
    "袖长",
    "版型",
    "腰型",
    "衣长",
    "裙型",
    "裙长",
    "裤型",
    "裤长",
    "门襟",
    "衣门襟",
    "主面料成分",
    "面料",
    "面料名称",
    "工艺",
    "弹力",
    "颜色",
    "风格",
    "风格类型",
    "适用场景",
)


# Keep the published specification table useful to a shopper or listing operator.
# Supply-chain flags, label/tag status and internal sourcing metadata stay in the
# private fact ledger rather than leaking into localized sales copy.
_PUBLIC_ATTRIBUTE_NAMES = (
    "品牌",
    "面料名称",
    "主面料成分",
    "图案",
    "厚薄",
    "款式",
    "版型",
    "袖型",
    "上市年份/季节",
    "领型",
    "袖长",
    "衣长",
    "工艺",
    "颜色",
    "尺码",
    "风格类型",
    "穿着方式",
    "门襟",
    "风格",
    "柔软度",
    "产品类别",
    "适用性别",
    "适合人群",
    "裤型",
    "裤长",
    "腰型",
    "裙型",
    "裙长",
    "裙类别",
)

_PROCESS_FILLER_PATTERNS: dict[str, tuple[str, ...]] = {
    "en": (
        "source-verified",
        "source-grounded",
        "verified source",
        "canonical data",
        "sku matrix below",
        "fact-check",
    ),
    "ko": (
        "원본 데이터에서 확인",
        "원본 정보를 바탕",
        "검증된 원본",
        "기계 판독",
    ),
    "pt": (
        "confirmadas na fonte",
        "com base nos dados de origem",
        "dados canônicos",
        "leitura automática",
    ),
}

_UNTRANSLATED_VALUES = {
    "Seller-declared source value",
    "판매자 원본 표기값",
    "Valor informado pelo vendedor",
}


# Reconciled appearance facts can carry an English value (the reconciler emits
# ``faux fur``, ``solid``, ``wide lapel``) while seller attributes remain in
# Chinese. These small, generic lexicons let the deterministic fallback and the
# fact-coverage gate localize that reconciled truth for all three markets
# without re-introducing a product-specific exception.
_RECONCILED_MATERIAL_PHRASES: dict[str, dict[str, str]] = {
    "faux fur": {"en": "Faux-fur", "ko": "인조 퍼", "pt": "pele sintética"},
    "artificial fur": {"en": "Faux-fur", "ko": "인조 퍼", "pt": "pele sintética"},
    "plush": {"en": "Plush", "ko": "플러시", "pt": "pelúcia"},
    "denim": {"en": "Denim", "ko": "데님", "pt": "jeans"},
    "leather": {"en": "Leather", "ko": "가죽", "pt": "couro"},
    "cotton": {"en": "Cotton", "ko": "면", "pt": "algodão"},
    "knit": {"en": "Knit", "ko": "니트", "pt": "malha"},
    "silk": {"en": "Silk", "ko": "실크", "pt": "seda"},
    "chiffon": {"en": "Chiffon", "ko": "시폰", "pt": "chiffon"},
    "velvet": {"en": "Velvet", "ko": "벨벳", "pt": "veludo"},
    "satin": {"en": "Satin", "ko": "새틴", "pt": "cetim"},
    "linen": {"en": "Linen", "ko": "린넨", "pt": "linho"},
    "wool": {"en": "Wool", "ko": "울", "pt": "lã"},
    "cashmere": {"en": "Cashmere", "ko": "캐시미어", "pt": "caxemira"},
    "fleece": {"en": "Fleece", "ko": "플리스", "pt": "moletom"},
    "lace": {"en": "Lace", "ko": "레이스", "pt": "renda"},
}

_GARMENT_TYPE_KEYWORDS: tuple[tuple[str, dict[str, str]], ...] = (
    ("羽绒服", {"en": "down jacket", "ko": "다운 재킷", "pt": "jaqueta acolchoada"}),
    ("连衣裙", {"en": "dress", "ko": "원피스", "pt": "vestido"}),
    ("半身裙", {"en": "skirt", "ko": "스커트", "pt": "saia"}),
    ("针织衫", {"en": "knit top", "ko": "니트 톱", "pt": "blusa de malha"}),
    ("卫衣", {"en": "hoodie", "ko": "후디", "pt": "moletom"}),
    ("牛仔裤", {"en": "jeans", "ko": "청바지", "pt": "jeans"}),
    ("T恤", {"en": "T-shirt", "ko": "티셔츠", "pt": "camiseta"}),
    ("衬衫", {"en": "shirt", "ko": "셔츠", "pt": "camisa"}),
    ("短裤", {"en": "shorts", "ko": "쇼츠", "pt": "bermuda"}),
    ("长裤", {"en": "pants", "ko": "바지", "pt": "calça"}),
    ("夹克", {"en": "jacket", "ko": "재킷", "pt": "jaqueta"}),
    ("棉衣", {"en": "padded coat", "ko": "패딩", "pt": "casaco acolchoado"}),
    ("外套", {"en": "coat", "ko": "코트", "pt": "casaco"}),
    ("大衣", {"en": "coat", "ko": "코트", "pt": "casaco"}),
    ("毛衣", {"en": "sweater", "ko": "스웨터", "pt": "suéter"}),
    ("背心", {"en": "vest", "ko": "조끼", "pt": "colete"}),
    ("马甲", {"en": "vest", "ko": "조끼", "pt": "colete"}),
    ("上衣", {"en": "top", "ko": "상의", "pt": "top"}),
    ("裙子", {"en": "skirt", "ko": "스커트", "pt": "saia"}),
    ("长裙", {"en": "skirt", "ko": "스커트", "pt": "saia"}),
    ("短裙", {"en": "skirt", "ko": "스커트", "pt": "saia"}),
    ("裤", {"en": "pants", "ko": "바지", "pt": "calça"}),
)


def _reconciled_decision_rows(facts: ProductFacts) -> dict[int, dict[str, str]]:
    ledger = facts.reconciled_fact_ledger
    if not isinstance(ledger, dict):
        return {}
    rows: dict[int, dict[str, str]] = {}
    for item in ledger.get("attribute_decisions", []):
        if not isinstance(item, dict) or not isinstance(item.get("attribute_index"), int):
            continue
        rows[item["attribute_index"]] = {
            "decision": str(item.get("decision") or "publish"),
            "canonical_value": str(item.get("canonical_value") or ""),
        }
    return rows


def _reconciled_material_phrase(language: str, facts: ProductFacts) -> str:
    """Localized material adjective from reconciled visual claims, or empty."""

    ledger = facts.reconciled_fact_ledger
    if not isinstance(ledger, dict):
        return ""
    claims = ledger.get("canonical_visual_claims")
    if not isinstance(claims, list):
        return ""
    combined = " ".join(
        str(claim.get("value") or "").casefold()
        for claim in claims
        if isinstance(claim, dict)
        and any(
            key in str(claim.get("concept") or "").casefold()
            for key in ("material", "texture", "surface", "composition")
        )
    )
    for keyword, phrases in _RECONCILED_MATERIAL_PHRASES.items():
        if keyword in combined:
            return phrases[language]
    return ""


def _category_type_noun(language: str, category_label: str) -> str:
    """Extract a generic garment noun from a category label, or empty."""

    label = str(category_label or "")
    for keyword, nouns in _GARMENT_TYPE_KEYWORDS:
        if keyword in label:
            return nouns[language]
    return ""


def _compose_reconciled_title(language: str, material: str, type_noun: str) -> str:
    if not material:
        return type_noun or {"en": "Product", "ko": "상품", "pt": "Produto"}[language]
    if not type_noun:
        return material
    if language == "pt":
        return f"{type_noun.capitalize()} de {material}"
    return f"{material} {type_noun}".strip()


# The reconcilers emit some short canonical values directly in English (``solid``,
# ``wide lapel``, ``hip-length``) while most remain Chinese. Map that small set so
# the deterministic path localizes them for every market.
_RECONCILED_APPEARANCE_VALUES: dict[str, dict[str, str]] = {
    "solid": {"en": "Solid color", "ko": "솔리드 컬러", "pt": "Cor lisa"},
    "striped": {"en": "Striped", "ko": "스트라이프", "pt": "Listrado"},
    "stripes": {"en": "Striped", "ko": "스트라이프", "pt": "Listrado"},
    "wide lapel": {"en": "Wide lapel", "ko": "와이드 라펠", "pt": "lapela larga"},
    "turn-down collar": {"en": "Turn-down collar", "ko": "테일러드 칼라", "pt": "gola dobrável"},
    "button": {"en": "Button", "ko": "버튼", "pt": "Botão"},
    "buttons": {"en": "Button", "ko": "버튼", "pt": "Botão"},
    "hip-length": {"en": "Hip-length", "ko": "힙 기장", "pt": "comprimento no quadril"},
    "short": {"en": "Short length", "ko": "짧은 기장", "pt": "Comprimento curto"},
}


def _localize_reconciled_value(
    language: str, value: str, term_map: dict[str, Any]
) -> str:
    """Localize a reconciled canonical value whether it is Chinese or English."""

    key = str(value).strip().casefold()
    if not re.search(r"[一-鿿]", str(value)):
        mapped = _RECONCILED_APPEARANCE_VALUES.get(key)
        if mapped is not None:
            return mapped[language]
        return str(value)
    return _localized_display(language, str(value), term_map)


def _reconciled_feature_values(
    language: str, facts: ProductFacts, term_map: dict[str, Any]
) -> list[str]:
    """Localized verified appearance concepts that buyer copy should substantiate.

    Published seller attributes keep their value; rejected attributes contribute
    their reconciled canonical value instead. This is the fact-coverage signal
    the copy gate checks against, so correct copy never has to repeat a seller
    claim the reconcilers overturned.
    """

    decision_rows = _reconciled_decision_rows(facts)
    values: list[str] = []
    for index, item in enumerate(facts.attributes):
        if item.name not in _MARKETING_ATTRIBUTE_NAMES:
            continue
        row = decision_rows.get(index, {})
        decision = row.get("decision", "publish")
        if decision == "publish":
            source_value = item.value
        elif decision == "reject" and row.get("canonical_value") not in {"", "N/A"}:
            source_value = row["canonical_value"]
        else:
            continue
        localized = _localize_reconciled_value(language, source_value, term_map)
        if (
            localized
            and localized != "—"
            and localized not in values
            and localized.casefold()
            not in {
                "seller-declared source value",
                "판매자 원본 표기값",
                "valor informado pelo vendedor",
            }
        ):
            values.append(localized)
    return values


def _source_terms(facts: ProductFacts, taxonomy: TaxonomyResult) -> set[str]:
    terms = {
        facts.platform,
        facts.source_category_name,
        taxonomy.category.name,
        taxonomy.category.path,
    }
    # Include every shopper-safe source field rather than a benchmark-derived
    # apparel whitelist. Unknown product types can therefore still render their
    # real attributes in the machine appendix.
    for item in facts.attributes:
        if buyer_safe_source_name(item.name):
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
    season_match = re.fullmatch(
        r"(?P<year>\d{4})年?(?P<season>春季|夏季|秋季|冬季)", raw
    )
    if season_match:
        season = translations.get(season_match.group("season"), "")
        if season:
            return f"{season_match.group('year')} {season}"
    rendered = raw
    for source, target in sorted(
        translations.items(), key=lambda item: len(item[0]), reverse=True
    ):
        rendered = rendered.replace(source, target)
    rendered = re.sub(
        r"(?P<low>\d+(?:\.\d+)?)\s*[-~～至]\s*(?P<high>\d+(?:\.\d+)?)\s*斤",
        lambda match: _localized_jin_range(language, match),
        rendered,
    )
    if re.search(r"[\u4e00-\u9fff]", rendered):
        return {
            "en": "Seller-declared source value",
            "ko": "판매자 원본 표기값",
            "pt": "Valor informado pelo vendedor",
        }[language]
    return rendered


def _decimal_measurement(value: Decimal) -> str:
    rounded = value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return format(rounded.normalize(), "f")


def _localized_jin_range(language: str, match: re.Match[str]) -> str:
    low_kg = Decimal(match.group("low")) / 2
    high_kg = Decimal(match.group("high")) / 2
    kilograms = f"{_decimal_measurement(low_kg)}–{_decimal_measurement(high_kg)} kg"
    if language != "en":
        return kilograms
    low_lb = low_kg * Decimal("2.2046226218")
    high_lb = high_kg * Decimal("2.2046226218")
    return (
        f"{kilograms} ({_decimal_measurement(low_lb)}–"
        f"{_decimal_measurement(high_lb)} lb)"
    )


def _seller_weight_display(language: str, raw: str) -> str:
    match = re.search(
        r"(?P<low>\d+(?:\.\d+)?)\s*[-~～至]\s*(?P<high>\d+(?:\.\d+)?)\s*斤",
        raw,
    )
    if not match:
        return ""
    prefix = raw[: match.start()].strip().rstrip("【[（(").strip()
    localized_prefix = _static_localize_term(language, prefix) if prefix else ""
    if localized_prefix in _UNTRANSLATED_VALUES:
        localized_prefix = ""
    converted = _localized_jin_range(language, match)
    return f"{localized_prefix} — {converted}" if localized_prefix else converted


def _append_us_length_conversion(raw: str, rendered: str) -> str:
    centimeters = re.findall(r"(?<!\d)(\d+(?:\.\d+)?)\s*cm\b", raw, re.IGNORECASE)
    if not centimeters:
        return rendered
    inches = [
        _decimal_measurement(Decimal(value) / Decimal("2.54"))
        for value in centimeters
    ]
    if len(inches) == 1:
        conversion = f"{inches[0]} in"
    else:
        conversion = f"{'–'.join(inches)} in"
    if conversion.casefold() in rendered.casefold():
        return rendered
    return f"{rendered} ({conversion})"


def _fallback_term_map(
    language: str, facts: ProductFacts, taxonomy: TaxonomyResult
) -> dict[str, str]:
    return {
        term: _static_localize_term(language, term)
        for term in sorted(_source_terms(facts, taxonomy))
    }


def _allowed_numbers(facts: ProductFacts, taxonomy: TaxonomyResult) -> set[str]:
    material = [
        facts.platform,
        facts.offer_id,
        facts.source_title,
        facts.source_category_id,
        taxonomy.category.category_id,
        *(item.value for item in facts.attributes),
        *(sku.sku_id for sku in facts.skus),
        *(item.kilograms for item in facts.size_conversions),
        *(item.pounds for item in facts.size_conversions),
        *(item.bust_cm for item in facts.size_chart_rows),
        *(item.length_cm for item in facts.size_chart_rows),
        *(item.weight_kg for item in facts.size_chart_rows),
        *(item.weight_lb for item in facts.size_chart_rows),
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
    normalized_allowed = {number.replace(",", ".") for number in allowed}
    return any(number.replace(",", ".") not in normalized_allowed for number in numbers)


def _repair_numeric_fields(
    payload: dict[str, Any], fallback: dict[str, Any], allowed: set[str]
) -> dict[str, Any]:
    """Replace only fields containing unsupported numbers instead of discarding the draft."""

    repaired = dict(payload)
    for key in ("title", "overview", "fit_note"):
        if _contains_unverified_numbers(repaired.get(key), allowed):
            repaired[key] = fallback[key]

    highlights = repaired.get("highlights")
    if isinstance(highlights, list):
        safe = [item for item in highlights if not _contains_unverified_numbers(item, allowed)]
        for item in fallback["highlights"]:
            if len(safe) >= 3:
                break
            if item not in safe:
                safe.append(item)
        repaired["highlights"] = safe[:5]

    for object_key in ("media_descriptions", "localized_terms"):
        values = repaired.get(object_key)
        fallback_values = fallback[object_key]
        if not isinstance(values, dict):
            repaired[object_key] = dict(fallback_values)
            continue
        fixed = dict(values)
        for key, value in fixed.items():
            if _contains_unverified_numbers(value, allowed):
                fixed[key] = fallback_values.get(key, value)
        repaired[object_key] = fixed
    return repaired


_BUYER_COPY_FIELDS = ("title", "overview", "highlights", "fit_note")


def _compose_copy_layers(
    candidate: Any, fallback: dict[str, Any]
) -> dict[str, Any]:
    """Combine model-authored buyer copy with a deterministic machine appendix.

    Category labels, localized source terms, media keys and the tables rendered from
    them are an exact delivery contract. Keeping that layer out of writer responses
    prevents a prose revision from deleting identifiers or mutating parseable data.
    """

    source = candidate if isinstance(candidate, dict) else {}
    payload = dict(fallback)
    for key in _BUYER_COPY_FIELDS:
        if key in source:
            payload[key] = source[key]
    claim_refs = source.get("claim_refs")
    if isinstance(claim_refs, dict):
        payload["claim_refs"] = claim_refs
    payload["media_descriptions"] = dict(fallback["media_descriptions"])
    localized_terms = dict(fallback["localized_terms"])
    candidate_terms = source.get("localized_terms")
    if isinstance(candidate_terms, dict):
        for term in localized_terms:
            value = candidate_terms.get(term)
            if (
                isinstance(value, str)
                and value.strip()
                and not re.search(r"[\u4e00-\u9fff]", value)
            ):
                localized_terms[term] = value.strip()
    payload["localized_terms"] = localized_terms
    return payload


def _verified_fit_note(language: str, facts: ProductFacts, fallback_note: str) -> str:
    has_size_evidence = bool(
        facts.size_chart_rows
        or facts.size_conversions
        or any(
            re.search(r"(?:尺码|适合身高|\bsize\b)", item.name, re.I)
            for item in facts.attributes
        )
        or any(
            re.search(r"(?:尺码|适合身高|\bsize\b)", item.name, re.I)
            for sku in facts.skus
            for item in sku.attributes
        )
    )
    if not has_size_evidence:
        return {
            "en": "Review the seller-listed specifications and option labels before ordering.",
            "ko": "구매 전 판매자가 제공한 상품 사양과 옵션 표기를 확인해 주세요.",
            "pt": "Consulte as especificações e opções informadas pelo vendedor antes da compra.",
        }[language]
    if not facts.size_chart_rows:
        return fallback_note
    return {
        "en": "Use the seller's garment measurements below; regional size equivalence is not inferred.",
        "ko": "아래 판매자 실측표를 확인해 주세요. 한국 사이즈로 임의 환산하지 않았습니다.",
        "pt": "Consulte as medidas da peça abaixo; não foi presumida equivalência automática com P, M ou G.",
    }[language]


def _salvage_copy_payload(
    language: str,
    payload: Any,
    fallback: dict[str, Any],
    facts: ProductFacts,
    taxonomy: TaxonomyResult,
    expected_media: set[str],
    expected_terms: set[str],
) -> tuple[dict[str, Any], str]:
    """Keep independently safe model fields instead of discarding the payload.

    The machine translation map is intentionally deterministic here. Buyer prose
    remains model-authored when it passes its own structural and factual guards.
    """

    candidate = payload if isinstance(payload, dict) else {}
    merged = dict(fallback)
    for key in ("title", "overview", "fit_note"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip() and not re.search(r"[\u4e00-\u9fff]", value):
            merged[key] = value.strip()
    highlights = candidate.get("highlights")
    if (
        isinstance(highlights, list)
        and 1 <= len(highlights) <= 8
        and all(
            isinstance(item, str)
            and item.strip()
            and not re.search(r"[\u4e00-\u9fff]", item)
            for item in highlights
        )
    ):
        merged["highlights"] = [item.strip() for item in highlights]
    # Media keys and localized terms are a separate machine layer. They never
    # come from a writer or repair response, including field-level salvage.
    machine_layers = _compose_copy_layers(candidate, fallback)
    merged["media_descriptions"] = dict(machine_layers["media_descriptions"])
    merged["localized_terms"] = dict(machine_layers["localized_terms"])

    for _ in range(5):
        error = _payload_validation_error(
            language, merged, facts, taxonomy, expected_media, expected_terms
        )
        if not error:
            return merged, "field-level-salvage"
        if error == "localized-title-guard":
            merged["title"] = fallback["title"]
        elif error == "numeric-fact-guard":
            merged = _repair_numeric_fields(
                merged, fallback, _allowed_numbers(facts, taxonomy)
            )
        elif error == "unsupported-fit-guidance-guard":
            unsafe = {
                "en": ("true to size", "usual size", "size up", "size down"),
                "ko": ("정사이즈", "평소 사이즈", "한 사이즈"),
                "pt": ("tamanho normal", "tamanho habitual", "tamanho maior", "tamanho menor"),
            }[language]
            for key in ("title", "overview", "fit_note"):
                if any(token in str(merged[key]).casefold() for token in unsafe):
                    merged[key] = fallback[key]
            if any(
                any(token in str(item).casefold() for token in unsafe)
                for item in merged["highlights"]
            ):
                merged["highlights"] = fallback["highlights"]
        elif error == "insufficient-verified-details":
            # Preserve the model title and overview, but inject fact-led fallback
            # bullets until the deterministic concept coverage gate is satisfied.
            combined = list(merged["highlights"])
            for item in fallback["highlights"]:
                if item not in combined:
                    combined.append(item)
                if len(combined) >= 5:
                    break
            merged["highlights"] = combined[:5]
            if combined == candidate.get("highlights"):
                break
        else:
            break
    return fallback, _payload_validation_error(
        language, merged, facts, taxonomy, expected_media, expected_terms
    ) or "unsafe-field-salvage"


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
    if not isinstance(payload, dict) or not required.issubset(payload):
        return "invalid-model-payload"
    if not all(
        isinstance(payload.get(key), str) and payload[key].strip()
        for key in ("title", "overview", "fit_note")
    ):
        return "invalid-model-payload"
    highlights = payload.get("highlights")
    if (
        not isinstance(highlights, list)
        or not 1 <= len(highlights) <= 8
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
    buyer_text = "\n".join(
        [
            str(payload.get("title") or ""),
            str(payload.get("overview") or ""),
            str(payload.get("fit_note") or ""),
            *[str(item) for item in payload.get("highlights", [])],
            *[str(item) for item in media.values()],
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
    ) > 128:
        return "localized-title-guard"
    if re.search(r"[\u4e00-\u9fff]", buyer_text):
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
    if any(
        phrase.casefold() in natural_text.casefold()
        for phrase in _PROCESS_FILLER_PATTERNS[language]
    ):
        return "process-language-in-shopper-copy"
    unsupported_fit_claims = {
        "en": (
            "true to size",
            "select your usual size",
            "choose your usual size",
            "size up",
            "size down",
            "standard size",
            "standard sizes",
            "universal size",
        ),
        "ko": (
            "정사이즈",
            "평소 사이즈",
            "한 사이즈 크게",
            "한 사이즈 작게",
            "표준 사이즈",
            "공용 사이즈",
        ),
        "pt": (
            "tamanho normal",
            "seu tamanho habitual",
            "um tamanho maior",
            "um tamanho menor",
            "tamanho padrão",
            "tamanhos padrão",
            "tamanho universal",
        ),
    }
    if any(phrase in natural_text.casefold() for phrase in unsupported_fit_claims[language]):
        return "unsupported-fit-guidance-guard"
    unavailable_measurement_references = {
        "en": (
            "garment measurements below",
            "measurements listed below",
            "specific garment measurements",
        ),
        "ko": ("아래 실측", "하단 실측", "구체적인 의류 실측"),
        "pt": (
            "medidas da peça abaixo",
            "medidas listadas abaixo",
            "medidas específicas da peça",
        ),
    }
    if not facts.size_chart_rows and any(
        phrase in natural_text.casefold()
        for phrase in unavailable_measurement_references[language]
    ):
        return "missing-size-chart-reference-guard"
    normalized_overview = re.sub(
        r"[^\w\uac00-\ud7a3]+", "", str(payload["overview"]).casefold()
    )
    normalized_highlights = [
        re.sub(r"[^\w\uac00-\ud7a3]+", "", str(item).casefold())
        for item in highlights
    ]
    if len(set(normalized_highlights)) != len(normalized_highlights) or any(
        # Short feature phrases naturally summarize prose; reject only a
        # sentence-like bullet copied back from the overview.
        len(item) >= 32 and item in normalized_overview
        for item in normalized_highlights
    ):
        return "repetitive-shopper-copy-guard"
    incomplete_main_material = any(
        item.name == "主面料成分含量" and item.value == "30%以下"
        for item in facts.attributes
    )
    predominant_material_terms = {
        "en": "main material",
        "ko": "주요 소재",
        "pt": "material principal",
    }
    if (
        incomplete_main_material
        and predominant_material_terms[language].casefold()
        in natural_text.casefold()
    ):
        return "incomplete-composition-marketing-guard"
    feature_values = _reconciled_feature_values(language, facts, localized_terms)
    matched_features = sum(
        _localized_concept_is_mentioned(language, value, natural_text)
        for value in feature_values
    )
    if matched_features < min(2, len(feature_values)):
        return "insufficient-verified-details"
    buyer_payload = {
        key: payload.get(key)
        for key in ("title", "overview", "highlights", "fit_note", "media_descriptions")
    }
    if _contains_unverified_numbers(
        buyer_payload, _allowed_numbers(facts, taxonomy)
    ):
        return "numeric-fact-guard"
    if generated_copy_violations(language, buyer_payload):
        return "content-compliance-guard"
    return ""


_GENERIC_FEATURE_TOKENS = {
    "en": {"style", "design", "product", "garment", "material", "fabric", "fit", "type"},
    "ko": {"스타일", "디자인", "상품", "제품", "소재", "원단", "핏"},
    "pt": {"estilo", "design", "produto", "peça", "material", "tecido", "corte", "tipo"},
}


def _localized_concept_is_mentioned(
    language: str, localized_value: str, natural_text: str
) -> bool:
    """Accept natural synonyms without weakening the factual concept gate.

    Model copy commonly turns canonical labels such as ``Straight Fit`` into
    ``straight-leg silhouette``. Exact-substring matching rejected that valid
    wording and discarded the entire draft. We first accept the full normalized
    phrase, then require one distinctive content token from the canonical value.
    """

    value = re.sub(r"[^\w\uac00-\ud7a3]+", " ", localized_value.casefold()).strip()
    text = re.sub(r"[^\w\uac00-\ud7a3]+", " ", natural_text.casefold()).strip()
    if not value:
        return False
    if value in text:
        return True
    tokens = {
        token
        for token in value.split()
        if token not in _GENERIC_FEATURE_TOKENS[language]
        and (
            # Korean and Chinese content words are often one or two syllables;
            # requiring three characters would drop every valid Korean concept.
            len(token) >= 3
            or (
                len(token) >= 2
                and bool(re.search(r"[가-힣一-鿿]", token))
            )
        )
    }
    return any(token in text for token in tokens)


def _natural_join(language: str, values: list[str]) -> str:
    clean = [value for value in dict.fromkeys(values) if value and value != "—"]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if language == "ko":
        return " · ".join(clean)
    conjunction = {"en": "and", "pt": "e"}[language]
    if len(clean) == 2:
        return f"{clean[0]} {conjunction} {clean[1]}"
    if language == "pt":
        return f"{', '.join(clean[:-1])} {conjunction} {clean[-1]}"
    return f"{', '.join(clean[:-1])}, {conjunction} {clean[-1]}"


def _localized_category_title(
    language: str,
    category_name: str,
) -> str:
    """Turn a resolved category label into a title noun without category IDs."""

    localized = _static_localize_term(language, category_name).strip()
    if localized in _UNTRANSLATED_VALUES:
        return {"en": "Product", "ko": "상품", "pt": "Produto"}[language]
    if language == "en":
        singular_suffixes = {
            "T-shirts": "T-Shirt",
            "shirts": "Shirt",
            "coats": "Coat",
            "jackets": "Jacket",
            "skirts": "Skirt",
            "dresses": "Dress",
        }
        for plural, singular in singular_suffixes.items():
            if localized.endswith(plural):
                return localized[: -len(plural)] + singular
        return localized
    if language == "pt":
        words = localized.split()
        noun_singular = {
            "Camisetas": "Camiseta",
            "Camisas": "Camisa",
            "Malhas": "Malha",
            "Casacos": "Casaco",
            "Jaquetas": "Jaqueta",
            "Calças": "Calça",
            "Saias": "Saia",
            "Vestidos": "Vestido",
        }
        adjective_singular = {
            "femininas": "feminina",
            "masculinas": "masculina",
            "femininos": "feminino",
            "masculinos": "masculino",
            "infantis": "infantil",
        }
        return " ".join(
            noun_singular.get(word, adjective_singular.get(word, word))
            for word in words
        )
    return localized


def _attribute_display_values(
    language: str,
    facts: ProductFacts,
    term_map: dict[str, Any],
    names: tuple[str, ...],
) -> list[str]:
    decision_rows = _reconciled_decision_rows(facts)
    values: list[str] = []
    for index, item in enumerate(facts.attributes):
        row = decision_rows.get(index, {})
        decision = row.get("decision", "publish")
        if item.name not in names:
            continue
        if decision == "publish":
            source_value = item.value
        elif decision == "reject" and row.get("canonical_value") not in {"", "N/A"}:
            source_value = row["canonical_value"]
        else:
            continue
        localized = _localize_reconciled_value(language, source_value, term_map)
        if localized != "—" and localized not in values:
            values.append(localized)
    return values


def _compose_localized_title(
    language: str,
    base_title: str,
    feature_by_source: dict[str, str],
) -> str:
    """Build a native title from verified concepts, never from a product ID.

    Attribute priority is shared by all apparel products. Locale-specific grammar
    controls where the product noun and modifiers appear; unknown attributes stay
    in the specification table instead of being forced into a keyword-heavy title.
    """

    ordered_names = (
        "设计",
        "图案",
        "领型",
        "袖长",
        "版型",
        "腰型",
        "裙型",
        "裙长",
        "裤型",
        "裤长",
    )
    features = [
        (name, feature_by_source[name])
        for name in ordered_names
        if feature_by_source.get(name)
    ]
    features = list(dict.fromkeys(features))[:4]
    if not features:
        return base_title
    if language == "en":
        phrase_overrides = {
            "Cold-shoulder": "cold-shoulder cutouts",
            "V-neck": "a V-neck",
            "Halter neck": "a halter neckline",
            "Long sleeve": "long sleeves",
            "Short sleeve": "short sleeves",
            "Relaxed fit": "a relaxed fit",
            "Slim fit": "a slim fit",
            "Vintage floral print": "a vintage floral print",
            "Floral print": "a floral print",
            "3D-effect print": "a 3D-effect print",
            "Striped": "a striped pattern",
            "Solid color": "a solid-color design",
        }
        phrases: list[str] = []
        for source_name, value in features:
            phrase = phrase_overrides.get(value)
            if phrase is None:
                lowered = value.lower()
                phrase = (
                    f"a {lowered}"
                    if source_name in {"图案", "领型", "版型"}
                    else lowered
                )
            phrases.append(phrase)
        return f"{base_title} with {_natural_join(language, phrases)}"
    if language == "ko":
        return f"{' · '.join(value for _, value in features)} {base_title}"
    pt_overrides = {
        "Listrado": "estampa listrada",
        "Cor lisa": "cor lisa",
        "Estampa multicolorida": "estampa multicolorida",
    }
    phrases = [
        pt_overrides.get(value, value[:1].lower() + value[1:])
        for _, value in features
    ]
    return f"{base_title} com {_natural_join(language, phrases)}"


def _benefit_led_highlight(
    language: str,
    source_name: str,
    localized_name: str,
    localized_value: str,
) -> str:
    """Turn one verified feature into a conservative shopper-facing benefit."""

    if language == "en":
        if source_name == "领型":
            return (
                "V-neckline creates a clean, open shape"
                if localized_value == "V-neck"
                else f"{localized_value} defines the neckline shape"
            )
        if source_name == "袖长":
            if localized_value == "Long sleeve":
                return "Long sleeves provide full-arm coverage"
            if localized_value == "Short sleeve":
                return "Short sleeves leave the forearms uncovered"
            return f"{localized_value} defines the sleeve coverage"
        if source_name == "版型":
            if localized_value == "Relaxed fit":
                return "Relaxed fit leaves room through the body"
            if localized_value == "Slim fit":
                return "Slim fit follows a closer silhouette"
            return f"{localized_value} shapes the overall silhouette"
        if source_name == "图案":
            return f"{localized_value} defines the visual style"
        if source_name == "工艺":
            return f"{localized_value} defines the visible construction detail"
        if source_name == "腰型":
            return f"Seller-listed {localized_value.lower()} waist profile"
        if source_name == "弹力":
            return f"Seller-listed stretch level: {localized_value.lower()}"
        if source_name == "颜色":
            return f"Offered in {localized_value}"
        if source_name in {"裙型", "裤型"}:
            return f"{localized_value} defines the product silhouette"
        if source_name in {"面料", "面料名称", "主面料成分"}:
            return f"Seller lists {localized_value} for the fabric"
        if source_name in {"衣长", "裤长", "裙长"}:
            return f"Seller-listed {localized_value.lower()} proportion"
    elif language == "ko":
        def ko_with_particle(value: str) -> str:
            if not value:
                return value
            final = ord(value[-1]) - 0xAC00
            has_batchim = 0 <= final <= 11171 and final % 28 != 0
            return value + ("으로" if has_batchim else "로")

        if source_name == "设计":
            final = ord(localized_value[-1]) - 0xAC00 if localized_value else -1
            particle = "을" if 0 <= final <= 11171 and final % 28 != 0 else "를"
            return f"{localized_value}{particle} 주요 디자인 디테일로 적용했습니다"
        if source_name == "领型":
            return (
                "브이넥으로 열린 목선 형태를 보여 줍니다"
                if localized_value == "브이넥"
                else f"{ko_with_particle(localized_value)} 네크라인 형태를 보여 줍니다"
            )
        if source_name == "袖长":
            return (
                "긴소매로 팔 전체를 덮습니다"
                if localized_value == "긴소매"
                else f"{localized_value} 소매 구성을 적용했습니다"
            )
        if source_name == "版型":
            if localized_value == "루즈핏":
                return "루즈핏으로 몸통에 여유가 있습니다"
            if localized_value == "슬림핏":
                return "슬림핏으로 몸에 가까운 실루엣을 만듭니다"
            return f"{localized_value}으로 전체 실루엣을 구성했습니다"
        if source_name == "图案":
            return f"{localized_value}가 디자인의 시각적 포인트입니다"
        if source_name == "工艺":
            return f"{localized_value} 디테일이 구조적 포인트를 이룹니다"
        if source_name == "腰型":
            return f"판매자가 표기한 허리선은 {localized_value}입니다"
        if source_name == "弹力":
            return f"판매자가 표기한 신축성은 {localized_value}입니다"
        if source_name == "颜色":
            return f"{localized_value} 색상으로 제공됩니다"
        if source_name in {"裙型", "裤型"}:
            return f"{localized_value} 형태로 전체 실루엣을 구성했습니다"
        if source_name in {"面料", "面料名称", "主面料成分"}:
            return f"판매자가 표기한 원단은 {localized_value}입니다"
        if source_name in {"衣长", "裤长", "裙长"}:
            return f"판매자가 표기한 {localized_value} 비율"
    else:
        if source_name == "设计":
            return f"{localized_value} como detalhe visual principal"
        if source_name == "领型":
            return (
                "Decote V com desenho aberto"
                if localized_value == "Decote V"
                else f"{localized_value} define o formato do decote"
            )
        if source_name == "袖长":
            return (
                "Mangas longas oferecem cobertura dos braços"
                if localized_value == "Manga longa"
                else f"{localized_value} define a cobertura das mangas"
            )
        if source_name == "版型":
            if localized_value == "Modelagem solta":
                return "Modelagem solta oferece mais espaço no corpo"
            if localized_value == "Modelagem ajustada":
                return "Modelagem ajustada acompanha uma silhueta mais próxima ao corpo"
            return f"{localized_value} define a silhueta geral"
        if source_name == "图案":
            return f"{localized_value} define o estilo visual"
        if source_name == "工艺":
            return f"{localized_value} define o detalhe de construção visível"
        if source_name == "腰型":
            return f"Altura da cintura informada pelo vendedor: {localized_value.lower()}"
        if source_name == "弹力":
            return f"Elasticidade informada pelo vendedor: {localized_value.lower()}"
        if source_name == "颜色":
            return f"Disponível em {localized_value.lower()}"
        if source_name in {"裙型", "裤型"}:
            return f"{localized_value} define a silhueta da peça"
        if source_name in {"面料", "面料名称", "主面料成分"}:
            return f"O vendedor informa {localized_value} para o tecido"
        if source_name in {"衣长", "裤长", "裙长"}:
            return f"Proporção de {localized_value.lower()} informada pelo vendedor"
    return {
        "en": f"Seller-listed {localized_name.lower()}: {localized_value}",
        "ko": f"판매자 표기 {localized_name}: {localized_value}",
        "pt": f"{localized_name} informado pelo vendedor: {localized_value}",
    }[language]


def _fallback_payload(
    language: str, facts: ProductFacts, taxonomy: TaxonomyResult
) -> dict[str, Any]:
    payload = dict(_FALLBACK_CONTENT[language])
    payload["fit_note"] = _verified_fit_note(
        language, facts, str(payload.get("fit_note") or "")
    )
    term_map = _fallback_term_map(language, facts, taxonomy)
    # When the online model selected the leaf, its category is authoritative.
    # In explicit offline mode the lexical resolver is only a weak fallback, so
    # the seller's own category label is safer copy evidence than its guessed leaf.
    category_label = (
        taxonomy.category.name
        if taxonomy.category.method == "model-constrained-all-leaves"
        else facts.source_category_name or taxonomy.category.name
    )
    title_is_publishable = (
        not facts.reconciled_fact_ledger
        or facts.reconciled_fact_ledger.get("seller_title_decision", "publish")
        == "publish"
    )
    reconciled_material = _reconciled_material_phrase(language, facts)
    if reconciled_material and not title_is_publishable:
        # The seller title/category embeds a material that visual reconciliation
        # refuted (e.g. a "wool" category on a faux-fur garment). Lead with the
        # reconciled material over a generic type noun instead of repeating the
        # refuted material in the shopper-facing title.
        base_title = _compose_reconciled_title(
            language,
            reconciled_material,
            _category_type_noun(language, category_label),
        )
    else:
        base_title = _localized_category_title(language, category_label)
    selected_features: list[tuple[str, str]] = []
    feature_by_source: dict[str, str] = {}
    seen_values: set[str] = set()
    decision_rows = _reconciled_decision_rows(facts)
    # Published attributes stay as-is. A rejected attribute contributes its
    # reconciled canonical value instead of the refuted seller value, so the
    # deterministic fallback never reasserts a fact the reconcilers overturned.
    publishable_attributes: list[tuple[int, ProductAttribute, str]] = []
    for index, item in enumerate(facts.attributes):
        decision = decision_rows.get(index, {}).get("decision", "publish")
        if decision == "publish":
            publishable_attributes.append((index, item, item.value))
        elif decision == "reject":
            canonical = decision_rows[index].get("canonical_value", "")
            if canonical and canonical not in {"N/A", ""}:
                publishable_attributes.append((index, item, canonical))
    buyer_title_source = facts.source_title if title_is_publishable else ""
    mapped_product_source_names = [
        item.source_name
        for item in taxonomy.attributes
        if item.source_name and not item.sales_attribute
    ]
    mapped_sales_source_names = [
        item.source_name
        for item in taxonomy.attributes
        if item.source_name and item.sales_attribute
    ]
    attribute_priority = list(
        dict.fromkeys(
            [
                *mapped_product_source_names,
                *_MARKETING_ATTRIBUTE_NAMES,
                *[
                    item.name
                    for _, item, _ in publishable_attributes
                    if buyer_safe_source_name(item.name)
                ],
                *mapped_sales_source_names,
            ]
        )
    )
    for attribute_name in attribute_priority:
        found = next(
            (
                (item, source_value)
                for _, item, source_value in publishable_attributes
                if item.name == attribute_name
            ),
            None,
        )
        if found is None:
            continue
        item, source_value = found
        localized_name = term_map.get(
            item.name, _static_localize_term(language, item.name)
        )
        localized_value = _localize_reconciled_value(language, source_value, term_map)
        if item.name == "图案" and any(
            token in buyer_title_source for token in ("花卉", "花朵", "花印")
        ):
            localized_value = {
                "en": "Vintage floral print"
                if "复古" in buyer_title_source
                else "Floral print",
                "ko": "빈티지 플로럴 프린트"
                if "复古" in buyer_title_source
                else "플로럴 프린트",
                "pt": "Estampa floral retrô"
                if "复古" in buyer_title_source
                else "Estampa floral",
            }[language]
        if (
            localized_value in seen_values
            or re.fullmatch(r"\d{6,}", str(localized_value).strip()) is not None
            or (item.name == "图案" and source_value in {"图片色", "图色"})
            or localized_value
            in {
                "Seller-declared source value",
                "판매자 원본 표기값",
                "Valor informado pelo vendedor",
            }
        ):
            continue
        selected_features.append((localized_name, localized_value))
        feature_by_source.setdefault(item.name, localized_value)
        seen_values.add(localized_value)
        if len(selected_features) == 5:
            break

    payload["title"] = _compose_localized_title(
        language, base_title, feature_by_source
    )
    feature_summary = _natural_join(
        language, [value for _, value in selected_features[:4]]
    )
    colors = _attribute_display_values(language, facts, term_map, ("颜色",))
    category_words = [
        word for word in re.findall(r"[\w\uac00-\ud7a3]+", base_title) if len(word) >= 3
    ]
    cleaned_colors: list[str] = []
    for color in colors:
        cleaned = color
        for word in category_words:
            cleaned = re.sub(
                rf"{re.escape(word)}s?\b", "", cleaned, flags=re.IGNORECASE
            )
        cleaned = cleaned.strip(" -/·")
        if cleaned and cleaned not in cleaned_colors:
            cleaned_colors.append(cleaned)
    colors = cleaned_colors
    sizes = _attribute_display_values(language, facts, term_map, ("尺码",))
    # Composite seller codes remain in the exact SKU table, but are not suitable
    # as shopper-facing color/size prose. Very large option sets are likewise
    # summarized by the table instead of becoming unreadable marketing bullets.
    if len(colors) > 6 or any("#" in value for value in colors):
        colors = []
    if len(sizes) > 10 or any("#" in value for value in sizes):
        sizes = []
    patterns = _attribute_display_values(language, facts, term_map, ("图案",))
    closures = _attribute_display_values(
        language, facts, term_map, ("门襟", "衣门襟", "穿着方式")
    )
    size_label = (
        _localized_display(language, facts.size_conversions[0].source_label, term_map)
        if facts.size_conversions
        else ""
    )
    option_summary = _natural_join(language, colors)
    if language == "en":
        feature_phrases = [value.lower() for _, value in selected_features[:4]]
        phrase_overrides = {
            "v-neck": "a V-neck",
            "long sleeve": "long sleeves",
            "relaxed fit": "a relaxed fit",
            "slim fit": "a slim fit",
            "vintage floral print": "a vintage floral print",
            "floral print": "a floral print",
            "3d-effect print": "a 3D-effect print",
        }
        feature_phrases = [
            phrase_overrides.get(phrase, phrase) for phrase in feature_phrases
        ]
        plural_bottom = bool(
            re.search(r"\b(?:shorts|pants|trousers)\b", base_title, re.IGNORECASE)
        )
        subject = (
            f"This pair of {base_title.lower()}"
            if plural_bottom
            else f"This {base_title.lower()}"
        )
        first = (
            f"{subject} combines "
            f"{_natural_join('en', feature_phrases)}."
            if feature_phrases
            else f"A {base_title.lower()} with a clean, product-focused presentation."
        )
        if patterns and closures:
            first += (
                f" The {patterns[0].lower()} finish is paired with a "
                f"{closures[0].lower()} construction."
            )
        second_parts = []
        if len(colors) == 1:
            second_parts.append(f"Available in {colors[0]}.")
        elif option_summary:
            second_parts.append(f"Available colors are {option_summary}.")
        if size_label:
            second_parts.append(f"This style is offered in {size_label}.")
        elif sizes:
            second_parts.append(
                f"Seller size labels: {_natural_join(language, sizes)}."
            )
        second_parts.append(
            "Review the seller's size guidance before ordering; regional size equivalence is not assumed."
        )
        payload["overview"] = first + "\n\n" + " ".join(second_parts)
    elif language == "ko":
        first = (
            f"{feature_summary} 디테일을 갖춘 {base_title}입니다."
            if feature_summary
            else f"상품 특징을 명확하게 정리한 {base_title}입니다."
        )
        if patterns and closures:
            first += f" {patterns[0]} 디자인에 {closures[0]} 구조를 적용했습니다."
        second_parts = []
        if len(colors) == 1:
            second_parts.append(f"{colors[0]} 색상으로 제공됩니다.")
        elif option_summary:
            second_parts.append(f"색상은 {option_summary} 중에서 선택할 수 있습니다.")
        if size_label:
            second_parts.append(f"사이즈는 {size_label}으로 제공됩니다.")
        elif sizes:
            second_parts.append(
                f"판매자 사이즈 표기는 {_natural_join(language, sizes)}입니다."
            )
        second_parts.append("구매 전 판매자 사이즈 안내를 확인해 주세요.")
        payload["overview"] = first + "\n\n" + " ".join(second_parts)
    else:
        pt_feature_summary = _natural_join(
            language,
            [value[:1].lower() + value[1:] for _, value in selected_features[:4]],
        )
        first = (
            f"Esta peça apresenta {pt_feature_summary}."
            if pt_feature_summary
            else f"Este {base_title.lower()} apresenta as características verificadas do produto."
        )
        if patterns and closures:
            pattern = patterns[0][:1].lower() + patterns[0][1:]
            closure = closures[0][:1].lower() + closures[0][1:]
            first += f" O acabamento {pattern} combina com a construção {closure}."
        second_parts = []
        if len(colors) == 1:
            second_parts.append(f"Disponível na cor {colors[0].lower()}.")
        elif option_summary:
            second_parts.append(f"As cores disponíveis são {option_summary}.")
        if size_label:
            second_parts.append(f"O modelo está disponível em {size_label}.")
        elif sizes:
            second_parts.append(
                f"Os tamanhos informados pelo vendedor são {_natural_join(language, sizes)}."
            )
        second_parts.append(
            "Consulte a orientação de tamanho do vendedor antes da compra; não presumimos equivalência regional."
        )
        payload["overview"] = first + "\n\n" + " ".join(second_parts)

    source_by_value = {value: source for source, value in feature_by_source.items()}
    payload["highlights"] = [
        _benefit_led_highlight(
            language,
            source_by_value.get(value, ""),
            name,
            value,
        )
        for name, value in selected_features[:4]
    ]
    if size_label:
        payload["highlights"].append(
            {
                "en": f"Seller-listed size guidance begins with {size_label}",
                "ko": f"판매자 사이즈 안내는 {size_label}부터 확인할 수 있습니다",
                "pt": f"A orientação de tamanho do vendedor começa em {size_label}",
            }[language]
        )
    elif sizes:
        payload["highlights"].append(
            {
                "en": f"Seller-listed sizes: {_natural_join(language, sizes)}",
                "ko": f"판매자 표기 사이즈: {_natural_join(language, sizes)}",
                "pt": f"Tamanhos informados pelo vendedor: {_natural_join(language, sizes)}",
            }[language]
        )
    if colors and len(payload["highlights"]) < 5:
        color_label = _static_localize_term(language, "颜色")
        payload["highlights"].append(
            f"{color_label}: {_natural_join(language, colors)}"
        )
    payload["highlights"] = payload["highlights"][:5]
    if len(payload["highlights"]) < 3:
        for fallback_highlight in _FALLBACK_CONTENT[language]["highlights"]:
            if fallback_highlight not in payload["highlights"]:
                payload["highlights"].append(fallback_highlight)
            if len(payload["highlights"]) == 3:
                break
    payload["localized_terms"] = term_map
    media_feature_summary = feature_summary
    payload["media_descriptions"] = {
        "main_image.jpeg": f"Primary image showing the complete {base_title.lower()}.",
        "detail_image_1.jpeg": f"Full product view highlighting {media_feature_summary or 'the overall product form'}.",
        "detail_image_2.jpeg": "Closer view of the construction and visible design details.",
        "detail_image_3.jpeg": "Alternate product view showing another verified visible feature.",
        "detail_image_4.jpeg": (
            f"Color comparison featuring {option_summary}."
            if option_summary
            else "Alternate product view or verified option comparison."
        ),
        "detail_image_5.jpeg": (
            f"Size and fit reference for {size_label}."
            if size_label
            else "Practical product presentation based on verified source evidence."
        ),
        "product_video.mp4": "Eight-second product presentation showing the complete form and key visible details.",
    }
    if language == "ko":
        feature_object = feature_summary
        if feature_object:
            final = ord(feature_object[-1]) - 0xAC00
            feature_object += "을" if 0 <= final <= 11171 and final % 28 != 0 else "를"
        payload["media_descriptions"] = {
            "main_image.jpeg": f"{base_title}의 전체 형태를 보여 주는 대표 이미지입니다.",
            "detail_image_1.jpeg": f"{feature_object or '전체 실루엣을'} 강조한 상품 전체 이미지입니다.",
            "detail_image_2.jpeg": "짜임과 눈에 보이는 디자인 디테일을 가까이 보여 줍니다.",
            "detail_image_3.jpeg": "다른 각도에서 확인 가능한 상품 특징을 보여 줍니다.",
            "detail_image_4.jpeg": (
                f"{option_summary} 색상을 비교해 보여 줍니다."
                if option_summary
                else "다른 상품 각도 또는 확인된 옵션을 보여 줍니다."
            ),
            "detail_image_5.jpeg": (
                f"{size_label}의 사이즈와 핏을 안내합니다."
                if size_label
                else "확인된 원본 정보를 바탕으로 상품을 실용적으로 보여 줍니다."
            ),
            "product_video.mp4": "상품의 전체 형태와 주요 특징을 보여 주는 8초 영상입니다.",
        }
    elif language == "pt":
        payload["media_descriptions"] = {
            "main_image.jpeg": f"Imagem principal mostrando o {base_title.lower()} por inteiro.",
            "detail_image_1.jpeg": f"Vista completa destacando {feature_summary or 'a forma geral do produto'}.",
            "detail_image_2.jpeg": "Vista aproximada da construção e dos detalhes visíveis do design.",
            "detail_image_3.jpeg": "Outro ângulo mostrando uma característica visível confirmada.",
            "detail_image_4.jpeg": (
                f"Comparação das cores {option_summary}."
                if option_summary
                else "Outro ângulo do produto ou comparação de opção confirmada."
            ),
            "detail_image_5.jpeg": (
                f"Referência de tamanho e caimento para {size_label}."
                if size_label
                else "Apresentação prática baseada nas informações verificadas do produto."
            ),
            "product_video.mp4": "Vídeo de 8 segundos mostrando a forma completa e os principais detalhes visíveis.",
        }
    return payload


def generate_copy_payload(
    language: str,
    facts: ProductFacts,
    taxonomy: TaxonomyResult,
    creative_plan: CreativePlan,
    client: QwenClient | None,
    *,
    claim_ledger: list[ClaimEvidence] | None = None,
    agent_guidance: str = "",
    revision_feedback: str = "",
    skill_instructions: str = "",
    audit_valid_draft: bool = True,
) -> tuple[dict[str, Any], str]:
    fallback = _fallback_payload(language, facts, taxonomy)
    if client is None:
        return fallback, "deterministic-fallback"

    locale = LANGUAGES[language]
    trace = getattr(client, "trace", None)
    claim_context = (
        json.dumps(publishable_claims(claim_ledger), ensure_ascii=False)
        if claim_ledger
        else "No separate ledger supplied; use only the verified product facts below."
    )
    system = f"""
You are a native {locale["locale"]} e-commerce copywriter and a strict factual editor.
Write {locale["description"]}. Return JSON only.
Do not translate mechanically from Chinese. Do not invent performance claims, measurements,
care instructions, composition, certifications, stock, price, brand authorization, or regional size equivalents.
Do not add unsupported numbers. Avoid absolute superlatives and body-shaming language.
Do not add ratings, review or sales counts, urgency, promotions, shipping or return promises,
endorsements, contact details, social handles, or external links.
Avoid inferred benefits such as breathable, lightweight, comfortable, durable, shape-retaining,
flattering, premium quality, or easy-care unless the exact benefit is explicitly stated in the source facts.
When a material is labeled as the main material but its declared content is below 30% and the complete
fiber composition is unavailable, do not present it as the predominant material. Omit it from shopper
prose or qualify it as a seller-listed material with incomplete composition information.
Prefer concrete source-backed, category-relevant details from the supplied facts. Do not assume
apparel parts, materials, functions or usage when this product belongs to another category.
The title, overview and highlights together must name at least three concrete verified product
attributes. Do not use process-oriented filler such as "source-grounded details" or describe the
fact-checking workflow to the shopper.
Treat the source title as useful product evidence: retain distinctive, concrete construction,
appearance, compatibility or form-factor details only when they are explicitly named there.
Use product-first phrasing natural to {locale["locale"]}; avoid translated syntax, keyword stuffing,
generic filler and mixed-language fragments. Keep the title under 128 characters.
Build the title from the localized product category plus the strongest supported search concepts such
as construction, silhouette, length, pattern, color or material. Prefer a readable title over listing
every available field, and do not reduce a compound seller value to only one of its stated components.
Write from this product's most distinctive verified construction detail rather than a stock opening.
Avoid reusable templates such as "designed for everyday style", "a must-have addition", "perfect for
any occasion", "elevate your wardrobe", "clean and polished look", or their translated equivalents.
Every highlight must communicate a different buyer decision: visible design, silhouette/construction,
available option, or conservative fit guidance. Do not turn highlights into raw "Field: Value" rows.
Do not repeat the overview as bullets. Use the overview for the overall product proposition, then make
each highlight add one new source-backed decision point.
The shopper prose is a presentation layer: never include category IDs, attribute IDs, SKU IDs, source
labels or audit terminology there. Code will place those exact machine fields in a separate appendix.
Use normal grammar rather than concatenating localized field values or copying Title Case attribute
labels into a sentence.
Write a concise, substantive overview as two short natural paragraphs. The first should explain the
distinctive construction and silhouette; the second should add supported options, seller-listed use or
style context, and conservative sizing information. Do not pad either paragraph when those facts are
absent. Do not refer shoppers to an audit, evidence ledger, canonical data, source verification process
or SKU matrix.
Call source size labels "seller-listed sizes", never standard, universal or true-to-size. Refer to
garment measurements below only when verified size_chart rows actually exist.
For en-US, use US spelling and concise marketplace phrasing. For ko-KR, use natural Korean retail
sentence endings and Korean option terminology. For pt-BR, use Brazilian vocabulary and forms such
as produto, tamanho, camiseta and consulte; avoid European Portuguese vocabulary.
{skill_instructions}
""".strip()
    prompt = f"""
Produce a JSON object containing at least these required fields; extra explanatory fields are ignored:
- title: localized product title
- overview: concise substantive localized prose using natural paragraphing
- highlights: preferably 3 to 5 distinct natural shopper-facing feature phrases, not raw field labels
- fit_note: one conservative localized sizing note
- claim_refs: optional object mapping title, overview, highlights and fit_note to supporting claim_id values
  from the ledger. Use only listed IDs; this metadata is for the delivery audit and is not shopper copy.
- localized_terms: object with exactly the source-term keys listed below. Translate each value naturally
  into {locale["locale"]}; preserve model numbers, IDs and established brand spellings. Do not leave Chinese
  characters in the translated values.

Do not generate category tables, attribute tables, SKU rows or media filenames. Code builds that machine
appendix independently from verified structured data after the buyer copy passes review.

Source terms requiring localization (keys must remain exact):
{json.dumps(sorted(fallback["localized_terms"]), ensure_ascii=False)}

Verified product facts:
{json.dumps(facts.compact_dict(), ensure_ascii=False)}

Resolved AliExpress category:
{json.dumps({"id": taxonomy.category.category_id, "name": taxonomy.category.name, "path": taxonomy.category.path}, ensure_ascii=False)}

Bounded manager localization priority:
{agent_guidance or "Use the locale rules and verified facts above."}

Independent evaluator revision feedback:
{revision_feedback or "No prior defect; produce the initial localized payload."}

Only write claims supported by the verified facts. Compact localized listing tables will be inserted
by code; do not repeat all SKUs or attributes in the prose. Internal evidence pointers and Chinese
source values will not be published.

Publishable claim ledger (use only these claims in shopper prose; source-image observations that are
not present here may guide media but must not be promoted into buyer claims):
{claim_context}
The seller_title row is evidence context, not permission to repeat promotional or body-effect wording;
extract only concrete product type, construction, silhouette, pattern, color and option facts from it.
""".strip()
    try:
        draft = client.chat_json(system, prompt)
    except ApiError as exc:
        if trace:
            trace.emit(
                "copy.generation_failed",
                language=language,
                category=exc.category,
                retryable=exc.retryable,
                error=str(exc),
            )
        return fallback, "deterministic-fallback"
    draft_localized_terms = (
        draft.get("localized_terms") if isinstance(draft, dict) else None
    )
    if trace:
        trace.emit("copy.draft", language=language, payload=draft)

    if not audit_valid_draft:
        fast_payload = _compose_copy_layers(draft, fallback)
        fast_error = _payload_validation_error(
            language,
            fast_payload,
            facts,
            taxonomy,
            set(fallback["media_descriptions"]),
            set(fallback["localized_terms"]),
        )
        if trace:
            trace.emit(
                "copy.validation",
                language=language,
                stage="fast-draft",
                validation_error=fast_error,
            )
        if not fast_error:
            return fast_payload, f"{client.config.chat_model}-validated-draft-fast"
        if fast_error in {
            "missing-size-chart-reference-guard",
            "repetitive-shopper-copy-guard",
        }:
            # These are presentation defects, not missing machine facts. In the
            # latency-sensitive profile, prefer the already validated factual
            # fallback instead of spending a second model call on prose polish.
            return fallback, fast_error

    audit_system = f"""
You are a native {locale["locale"]} factual copy auditor. Return JSON only.
Rewrite the candidate listing so every product claim is directly supported by the verified source facts.
Remove inferred benefits, marketing embellishment, unsupported measurements and regional size equivalence.
Do not describe a seller-listed material as predominant when its declared content is below 30% and
the full fiber composition is unavailable; omit or explicitly qualify that material statement.
Remove ratings, sales counts, urgency, promotions, shipping or return promises, endorsements,
contact details, social handles and external links.
Keep natural localized language, but prefer precise field-level facts over generic filler.
Replace stock marketing phrases with product-specific syntax and vary sentence structure. Keep buyer
copy free of IDs and raw field labels because code renders machine-oriented data in a separate appendix.
Make each highlight serve a distinct purchase decision instead of repeating the overview.
The result must contain the four requested buyer-copy fields; ignore harmless extra top-level metadata
instead of rewriting good prose. Do not copy Chinese characters into buyer-facing prose. Do not add
machine appendix fields: code supplies category, attribute, SKU and media data independently.
When the candidate contains valid claim_refs, preserve only ledger IDs that still support the rewritten field.
Apply native {locale["locale"]} grammar and retail terminology, not literal source-language word order.
Use natural paragraphing for the overview. Remove process language, evidence-led phrasing and
instructions to inspect a SKU matrix.
{skill_instructions}
""".strip()
    audit_prompt = f"""
Audit and, where needed, correct this candidate buyer-copy payload:
{json.dumps({key: draft.get(key) for key in (*_BUYER_COPY_FIELDS, "claim_refs") if key in draft}, ensure_ascii=False)}

Verified source facts:
{json.dumps(facts.compact_dict(), ensure_ascii=False)}

Resolved AliExpress category:
{json.dumps({"id": taxonomy.category.category_id, "name": taxonomy.category.name, "path": taxonomy.category.path}, ensure_ascii=False)}

Required top-level keys: title, overview, highlights, fit_note.
Do not output media_descriptions or localized_terms. Output the corrected buyer-copy JSON object only.
""".strip()
    audit_applied = True
    try:
        audited = client.chat_json(audit_system, audit_prompt)
    except ApiError as exc:
        audited = draft
        audit_applied = False
        if trace:
            trace.emit(
                "copy.audit_failed",
                language=language,
                category=exc.category,
                retryable=exc.retryable,
                error=str(exc),
            )
    if isinstance(audited, dict) and isinstance(draft_localized_terms, dict):
        audited = dict(audited)
        audited["localized_terms"] = draft_localized_terms
    if trace:
        trace.emit(
            "copy.audit",
            language=language,
            audit_applied=audit_applied,
            payload=audited,
        )

    payload = _compose_copy_layers(audited, fallback)
    expected_media = set(fallback["media_descriptions"])
    expected_terms = set(fallback["localized_terms"])
    validation_error = _payload_validation_error(
        language, payload, facts, taxonomy, expected_media, expected_terms
    )
    if trace:
        trace.emit(
            "copy.validation",
            language=language,
            stage="audited",
            validation_error=validation_error,
        )
    repaired = False
    if validation_error == "numeric-fact-guard":
        payload = _repair_numeric_fields(
            payload, fallback, _allowed_numbers(facts, taxonomy)
        )
        validation_error = _payload_validation_error(
            language, payload, facts, taxonomy, expected_media, expected_terms
        )
        repaired = not validation_error
    if validation_error:
        repair_prompt = f"""
Repair the candidate payload because it failed this deterministic check: {validation_error}.
Return all required schema fields in natural {locale["locale"]}; harmless extra top-level fields are
allowed and ignored. Remove Chinese fragments from buyer-facing prose. Do not generate machine appendix
fields; code restores the exact deterministic appendix after this repair.
Remove unsupported numbers and claims instead of replacing them with new claims.

Candidate buyer copy:
{json.dumps({key: payload.get(key) for key in (*_BUYER_COPY_FIELDS, "claim_refs") if key in payload}, ensure_ascii=False)}

Verified source facts:
{json.dumps(facts.compact_dict(), ensure_ascii=False)}
""".strip()
        try:
            payload = client.chat_json(audit_system, repair_prompt)
        except ApiError as exc:
            if trace:
                trace.emit(
                    "copy.repair_failed",
                    language=language,
                    validation_error=validation_error,
                    category=exc.category,
                    retryable=exc.retryable,
                    error=str(exc),
                )
            salvaged, salvage_source = _salvage_copy_payload(
                language,
                payload,
                fallback,
                facts,
                taxonomy,
                expected_media,
                expected_terms,
            )
            if salvaged is not fallback:
                return salvaged, f"{client.config.chat_model}-{salvage_source}"
            return fallback, validation_error
        if isinstance(payload, dict) and isinstance(draft_localized_terms, dict):
            payload = dict(payload)
            payload["localized_terms"] = draft_localized_terms
        payload = _compose_copy_layers(payload, fallback)
        validation_error = _payload_validation_error(
            language, payload, facts, taxonomy, expected_media, expected_terms
        )
        if validation_error == "numeric-fact-guard":
            payload = _repair_numeric_fields(
                payload, fallback, _allowed_numbers(facts, taxonomy)
            )
            validation_error = _payload_validation_error(
                language, payload, facts, taxonomy, expected_media, expected_terms
            )
        if validation_error:
            if trace:
                trace.emit(
                    "copy.validation",
                    language=language,
                    stage="repair",
                    validation_error=validation_error,
                    payload=payload,
                )
            salvaged, salvage_source = _salvage_copy_payload(
                language,
                payload,
                fallback,
                facts,
                taxonomy,
                expected_media,
                expected_terms,
            )
            if salvaged is not fallback:
                return salvaged, f"{client.config.chat_model}-{salvage_source}"
            return fallback, validation_error
        repaired = True

    source = client.config.chat_model
    if repaired:
        source += "-factual-repair"
    elif audit_applied:
        source += "-factual-audit"
    else:
        source += "-validated-draft"
    if trace:
        trace.emit(
            "copy.complete",
            language=language,
            source=source,
            repaired=repaired,
            audit_applied=audit_applied,
            payload=payload,
        )
    return payload, source


def _escape_table(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _localized_display(language: str, value: str, term_map: dict[str, Any]) -> str:
    raw = str(value).strip()
    translated = term_map.get(raw)
    if isinstance(translated, str) and translated.strip():
        rendered = translated.strip()
    else:
        rendered = _static_localize_term(language, raw)
    if rendered in _UNTRANSLATED_VALUES:
        if "/" in raw:
            localized_parts = [
                _static_localize_term(language, part.strip())
                for part in raw.split("/")
                if part.strip()
            ]
            localized_parts = [
                part for part in localized_parts if part not in _UNTRANSLATED_VALUES
            ]
            if localized_parts:
                return "/".join(localized_parts)
            return {
                "en": "Marketplace category",
                "ko": "마켓플레이스 카테고리",
                "pt": "Categoria do marketplace",
            }[language]
        return "—"
    seller_weight = _seller_weight_display(language, raw)
    if seller_weight:
        rendered = seller_weight
    if language == "en":
        rendered = _append_us_length_conversion(raw, rendered)
    return rendered


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
    verified_note = _verified_fit_note(
        language, facts, str(payload.get("fit_note") or "").strip()
    )
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
    category = taxonomy.category
    localized_platform = _localized_display(language, facts.platform, term_map)
    localized_source_category = _localized_display(
        language, facts.source_category_name, term_map
    )
    localized_leaf_name = _localized_display(language, category.name, term_map)
    localized_leaf_path = _localized_display(language, category.path, term_map)
    lines.extend(["", f"## {locale['appendix']}", ""])
    lines.extend(
        [
            f"- **{labels['source_platform']}:** {localized_platform}",
            f"- **{labels['product_id']}:** {_escape_table(facts.offer_id)}",
            f"- **{labels['product_url']}:** {facts.source_url}",
            f"- **{labels['source_category_name']}:** {localized_source_category}",
            f"- **{labels['leaf_category_id']}:** {_escape_table(category.category_id)}",
            f"- **{labels['leaf_category_name']}:** {localized_leaf_name}",
            f"- **{labels['leaf_category_path']}:** {localized_leaf_path}",
        ]
    )

    main_material_content = next(
        (
            item
            for item in facts.attributes
            if item.name == "主面料成分含量" and item.value
        ),
        None,
    )
    qualified_material_labels = {
        "en": "Listed material",
        "ko": "표기 소재",
        "pt": "Material informado",
    }
    incomplete_material_notes = {
        "en": "seller-listed content: {content}; full fiber composition not provided",
        "ko": "판매자 표기 함량: {content}; 전체 혼용률 정보 없음",
        "pt": "teor informado pelo vendedor: {content}; composição têxtil completa não informada",
    }
    public_rows: dict[str, dict[str, list[str]]] = {}
    natural_buyer_text = "\n".join(
        [
            str(payload.get("title") or ""),
            str(payload.get("overview") or ""),
            str(payload.get("fit_note") or ""),
            *[str(item) for item in payload.get("highlights", [])],
        ]
    )
    mapped_source_names = {
        item.source_name for item in taxonomy.attributes if item.source_name
    }
    for item in facts.attributes:
        localized_value = _localized_display(language, item.value, term_map)
        mentioned_in_buyer_copy = (
            buyer_safe_source_name(item.name)
            and
            localized_value != "—"
            and len(re.sub(r"\W+", "", localized_value)) >= 3
            and _localized_concept_is_mentioned(
                language, localized_value, natural_buyer_text
            )
        )
        if (
            not buyer_safe_source_name(item.name)
            and item.name not in mapped_source_names
            and not mentioned_in_buyer_copy
        ):
            continue
        name = _localized_display(language, item.name, term_map)
        value = localized_value
        if (
            item.name == "主面料成分"
            and main_material_content is not None
            and main_material_content.value == "30%以下"
        ):
            name = qualified_material_labels[language]
            content = _localized_display(
                language, main_material_content.value, term_map
            )
            value = (
                f"{value} ("
                + incomplete_material_notes[language].format(content=content)
                + ")"
            )
        if name == "—" or value == "—":
            continue
        row = public_rows.setdefault(name, {"ids": [], "values": []})
        if item.attribute_id not in row["ids"]:
            row["ids"].append(item.attribute_id)
        if value not in row["values"]:
            row["values"].append(value)
    if public_rows:
        attribute_header = {
            "en": ("Attribute ID", "Attribute", "Value", "Source"),
            "ko": ("속성 ID", "속성", "값", "출처"),
            "pt": ("ID do atributo", "Atributo", "Valor", "Fonte"),
        }[language]
        seller_attribute_source = {
            "en": "Seller product attributes",
            "ko": "판매자 상품 속성",
            "pt": "Atributos informados pelo vendedor",
        }[language]
        lines.extend(
            [
                "",
                f"## {locale['attributes']}",
                "",
                f"| {attribute_header[0]} | {attribute_header[1]} | {attribute_header[2]} | {attribute_header[3]} |",
                "|---|---|---|---|",
            ]
        )
        for name, row in public_rows.items():
            lines.append(
                "| "
                + " | ".join(
                    (
                        _escape_table(", ".join(row["ids"])),
                        _escape_table(name),
                        _escape_table(_natural_join(language, row["values"])),
                        _escape_table(seller_attribute_source),
                    )
                )
                + " |"
            )

    platform_headers = {
        "en": ("Type", "Attribute ID", "Attribute", "Value ID", "Value", "Source"),
        "ko": ("유형", "속성 ID", "속성", "값 ID", "값", "출처"),
        "pt": ("Tipo", "ID do atributo", "Atributo", "ID do valor", "Valor", "Fonte"),
    }[language]
    platform_types = {
        "en": ("Product", "Sales"),
        "ko": ("상품", "판매"),
        "pt": ("Produto", "Venda"),
    }[language]
    lines.extend(
        [
            "",
            f"## {locale['platform_attributes']}",
            "",
            f"| {' | '.join(platform_headers)} |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in taxonomy.attributes:
        item_type = platform_types[1] if item.sales_attribute else platform_types[0]
        name = _localized_display(language, item.name, term_map)
        value = _localized_display(language, item.platform_value, term_map)
        source_name = _localized_display(language, item.source_name, term_map)
        evidence_pointer = str(item.source_evidence_pointer or "")
        if evidence_pointer == facts.source_title_evidence_pointer:
            source_label = {
                "en": "Mapped from seller product title",
                "ko": "판매자 상품명에서 매핑",
                "pt": "Mapeado do título informado pelo vendedor",
            }[language]
        elif "/productSkuInfos/" in evidence_pointer:
            source_label = {
                "en": f"Mapped from seller SKU attribute: {source_name}",
                "ko": f"판매자 SKU 속성에서 매핑: {source_name}",
                "pt": f"Mapeado do atributo de SKU do vendedor: {source_name}",
            }[language]
        elif source_name == "—":
            source_label = {
                "en": "Mapped from seller listing facts",
                "ko": "판매자 상품 정보에서 매핑",
                "pt": "Mapeado das informações do anúncio do vendedor",
            }[language]
        else:
            source_label = {
                "en": f"Mapped from seller attribute: {source_name}",
                "ko": f"판매자 속성에서 매핑: {source_name}",
                "pt": f"Mapeado do atributo do vendedor: {source_name}",
            }[language]
        lines.append(
            f"| {item_type} | {_escape_table(item.attr_id)} | {_escape_table(name)} | "
            f"{_escape_table(item.value_id)} | {_escape_table(value)} | {_escape_table(source_label)} |"
        )

    all_sku_names: list[str] = []
    for sku in facts.skus:
        for item in sku.attributes:
            if item.name and item.name not in all_sku_names:
                all_sku_names.append(item.name)
    sku_source_note = {
        "en": "Source: seller SKU data.",
        "ko": "출처: 판매자 SKU 데이터.",
        "pt": "Fonte: dados de SKU informados pelo vendedor.",
    }[language]
    lines.extend(["", f"## {locale['skus']}", "", sku_source_note, ""])
    lines.append(
        "| SKU ID | Spec ID | "
        + " | ".join(
            _escape_table(
                f"{_localized_display(language, name, term_map)} "
                f"(ID {next((attr.attribute_id for sku in facts.skus for attr in sku.attributes if attr.name == name), '')})"
            )
            for name in all_sku_names
        )
        + " |"
    )
    lines.append("|---|---|" + "---|" * len(all_sku_names))
    for sku in facts.skus:
        values = {item.name: item.value for item in sku.attributes}
        row = [sku.sku_id, sku.spec_id] + [
            _localized_display(language, values.get(name, ""), term_map)
            for name in all_sku_names
        ]
        lines.append("| " + " | ".join(_escape_table(value) for value in row) + " |")

    if facts.size_chart_rows:
        lines.extend(
            [
                "",
                f"## {locale['size_chart']}",
                "",
                f"| {labels['size']} | {labels['bust']} | {labels['garment_length']} | {labels['seller_weight']} |",
                "|---|---|---|---|",
            ]
        )
        for item in facts.size_chart_rows:
            bust = f"{item.bust_cm} cm" if item.bust_cm else ""
            length = f"{item.length_cm} cm" if item.length_cm else ""
            weight = item.weight_kg
            if language == "en":
                if item.bust_cm:
                    bust += f" ({_decimal_measurement(Decimal(item.bust_cm) / Decimal('2.54'))} in)"
                if item.length_cm:
                    length += f" ({_decimal_measurement(Decimal(item.length_cm) / Decimal('2.54'))} in)"
                if item.weight_lb:
                    weight = f"{item.weight_kg} ({item.weight_lb})"
            lines.append(
                f"| {_escape_table(_localized_display(language, item.size_label, term_map))} | {_escape_table(bust)} | "
                f"{_escape_table(length)} | {_escape_table(weight)} |"
            )

    include_imperial = language == "en"
    lines.extend(["", f"## {locale['sizes']}", ""])
    if facts.size_conversions:
        if include_imperial:
            lines.extend(
                [
                    f"| {labels['seller_label']} | {labels['metric']} | {labels['imperial']} |",
                    "|---|---|---|",
                ]
            )
        else:
            lines.extend(
                [
                    f"| {labels['seller_label']} | {labels['metric']} |",
                    "|---|---|",
                ]
            )
        for size in facts.size_conversions:
            row = [
                _localized_display(language, size.source_label, term_map),
                size.kilograms,
            ]
            if include_imperial:
                row.append(size.pounds)
            lines.append("| " + " | ".join(_escape_table(value) for value in row) + " |")
    else:
        lines.append(verified_note)

    lines.extend(["", f"## {locale['media']}", ""])
    for filename, description in payload["media_descriptions"].items():
        lines.append(f"- **{filename}:** {str(description).strip()}")

    lines.extend(["", f"## {locale['note']}", ""])
    if facts.size_conversions or facts.size_chart_rows:
        lines.extend([verified_note, ""])
    lines.extend([locale["color_note"], ""])
    return "\n".join(lines)
