"""Bounded creative planning for the fixed delivery contract."""

from __future__ import annotations

import json
from typing import Any

from .api import ApiError, QwenClient
from .compliance import visual_prompt_violations
from .models import CreativePlan, ProductFacts, TaxonomyResult


_PRESERVATION = (
    "Preserve the exact product identity, silhouette, construction, pattern, trims and visible color from the "
    "reference images. Do not add or remove pockets, buttons, prints, logos, fasteners, sleeves or accessories. "
    "Do not invent text, measurements, materials, certifications or brand marks. No watermark."
)

_MAIN_PRESENTATION = (
    "Show exactly one physical product in exactly one verified colorway. The complete product must be visible, "
    "including collar or waistband, both sleeves or legs, cuffs and hem. Use a product-only studio presentation "
    "such as a clean flat lay, hanger, or invisible mannequin; no person, mannequin body, colorway lineup, duplicate "
    "garment, inset, split screen, montage or collage."
)

_BASE_DETAIL_SLOT_DIRECTIVES = (
    "This is detail slot 1: show a complete front three-quarter product presentation on a hanger or invisible form "
    "with the uncluttered, practical editorial styling familiar to US marketplace shoppers. Use a warm neutral wall "
    "and soft daylight; it must be visibly different from the square white-background hero.",
    "This is detail slot 2: make a tight but readable upper-garment close-up centered on the verified collar, "
    "front opening and visible fastening construction. Use restrained Korean commerce styling: pale neutral tones, "
    "precise spacing and soft diffused light. Do not add Hangul or show an isolated generic fabric swatch.",
    "This is detail slot 3: make a distinct close-up of the verified sleeve, cuff, hem and natural drape while "
    "keeping enough of the product visible to identify it. Use warm natural daylight and a subtle contemporary "
    "Brazilian marketplace mood without flags, landmarks, stereotypes or written Portuguese.",
    "This is detail slot 4: create a clean catalog lineup using only color variants visibly present in the supplied "
    "references. Show two or three complete products with identical construction and no labels or swatches.",
    "This is detail slot 5: show one adult wearer in a restrained everyday styling context, with the product fully "
    "visible and unobstructed from collar through hem. Use culturally neutral, inclusive styling that reads naturally "
    "across the US, South Korea and Brazil. Keep anatomy natural and do not add accessories to the product.",
)


def _product_family(taxonomy: TaxonomyResult) -> str:
    category_id = taxonomy.category.category_id
    if category_id in {"30341", "30335", "39153"}:
        return "bottom"
    if category_id in {"30843", "29553"}:
        return "children"
    if category_id == "39107":
        return "dress"
    return "top"


def _detail_slot_directives(taxonomy: TaxonomyResult) -> tuple[str, ...]:
    family = _product_family(taxonomy)
    category_id = taxonomy.category.category_id
    wearer = (
        "one adult man"
        if category_id in {"30341", "30335", "30408", "30471"}
        else "one adult woman"
    )
    slot_1 = _BASE_DETAIL_SLOT_DIRECTIVES[0]
    slot_4 = _BASE_DETAIL_SLOT_DIRECTIVES[3]
    if family == "bottom":
        return (
            slot_1,
            "This is detail slot 2: make a tight but readable close-up centered on the verified waistband, "
            "front closure and pocket construction. Use restrained Korean commerce styling with pale neutral tones, "
            "precise spacing and diffused light. Do not invent a fly, drawstring, belt loop or pocket.",
            "This is detail slot 3: make a distinct close-up of the verified leg shape, side seam and hem, "
            "while keeping enough of the product visible to identify it. Use warm natural daylight and a subtle "
            "contemporary Brazilian marketplace mood without flags, landmarks, stereotypes or written text.",
            slot_4,
            f"This is detail slot 5: show {wearer} in a restrained everyday styling context, with the waistband, "
            "both legs and hem visible and unobstructed. Use inclusive cross-market styling appropriate to the US, "
            "South Korea and Brazil. Keep anatomy, body proportions and garment length natural; do not reshape the body.",
        )
    if family == "children":
        return (
            slot_1,
            _BASE_DETAIL_SLOT_DIRECTIVES[1],
            _BASE_DETAIL_SLOT_DIRECTIVES[2],
            slot_4,
            "This is detail slot 5: create a product-only, age-appropriate outfit flat lay. Keep the item complete "
            "and unobstructed, use only neutral unbranded props with inclusive cross-market styling, and do not depict "
            "a child or adult wearer.",
        )
    if family == "dress":
        return (
            slot_1,
            "This is detail slot 2: make a tight but readable close-up centered on the verified neckline, bodice "
            "and closure construction. Use restrained Korean commerce styling with pale neutral tones, precise spacing "
            "and diffused light. Do not invent buttons, a zipper, belt or trim.",
            "This is detail slot 3: show the verified waist transition, skirt drape and hem while keeping enough "
            "of the dress visible to identify its silhouette and length. Use warm natural daylight and a subtle "
            "contemporary Brazilian marketplace mood without flags, landmarks, stereotypes or written text.",
            slot_4,
            _BASE_DETAIL_SLOT_DIRECTIVES[4].replace("one adult wearer", wearer),
        )
    return (
        *_BASE_DETAIL_SLOT_DIRECTIVES[:4],
        _BASE_DETAIL_SLOT_DIRECTIVES[4].replace("one adult wearer", wearer),
    )


def _video_guard(taxonomy: TaxonomyResult) -> str:
    family = _product_family(taxonomy)
    focus = {
        "bottom": "Keep the waistband, closure, pockets, both legs and hem stable and fully visible.",
        "children": "Use a product-only presentation; do not add a child, adult, hands, toys or character graphics.",
        "dress": "Keep the neckline, bodice, waist transition, skirt silhouette and hem stable.",
        "top": "Keep the collar or neckline, front opening, pockets, both sleeves, cuffs and hem stable.",
    }[family]
    return (
        "Use one continuous slow 10-to-15-degree camera arc with a subtle push-in; no scene cuts. "
        + focus
        + " Preserve exact product construction, pattern and color in every frame. No morphing, duplicate product, "
        "new accessories, hands covering the product, text, logo animation, camera shake, flicker, speech or music."
    )


_VISUAL_ATTRIBUTE_MARKERS = (
    "产品类别",
    "类别",
    "款式",
    "图案",
    "版型",
    "领型",
    "袖长",
    "袖型",
    "衣长",
    "门襟",
    "裤型",
    "裤长",
    "腰型",
    "裙型",
    "裙长",
    "颜色",
)


def _visual_fact_summary(facts: ProductFacts) -> str:
    """Keep visual prompts focused on appearance rather than invisible claims."""

    selected: list[str] = []
    for item in facts.attributes:
        if any(marker in item.name for marker in _VISUAL_ATTRIBUTE_MARKERS):
            fact = f"{item.name}: {item.value}"
            if fact not in selected:
                selected.append(fact)
        if len(selected) == 12:
            break
    return ", ".join(selected)


def fallback_creative_plan(
    facts: ProductFacts, taxonomy: TaxonomyResult
) -> CreativePlan:
    verified = _visual_fact_summary(facts)
    identity = f"Product type: {facts.source_category_name}."
    if verified:
        identity += f" Verified visible source attributes: {verified}."
    identity += " When text facts and reference pixels appear to conflict, preserve the reference product pixels."
    slot_directives = _detail_slot_directives(taxonomy)
    return CreativePlan(
        visual_theme="Clean international marketplace editorial with neutral, culturally inclusive styling",
        main_prompt=(
            f"Create a square premium e-commerce hero image using the reference product. {identity} "
            "Use a clean white-to-very-light-gray studio background, soft realistic shadow, even lighting, "
            "accurate color, centered composition, and generous margins. Product occupies about 80 percent of frame. "
            "No promotional text, badges, borders, collage or extra props. "
            + _MAIN_PRESENTATION
            + " "
            + _PRESERVATION
        ),
        detail_prompts=[
            (
                f"Create a vertical 4:5 full-product editorial view. {identity} Use a restrained neutral lifestyle "
                "setting appropriate for a global marketplace, with the product unobstructed and clearly readable. "
                + slot_directives[0]
                + " "
                + _PRESERVATION
            ),
            (
                f"Create a vertical 4:5 detail-focused commerce image showing visible construction and design details. "
                f"{identity} Use realistic close-up photography while keeping the full product identity consistent. "
                + slot_directives[1]
                + " "
                + _PRESERVATION
            ),
            (
                f"Create a vertical 4:5 premium product feature photograph. {identity} Use one coherent full-frame "
                "composition with no dividers, inset panels, split screen, generated words, numbers, icons or labels. "
                + slot_directives[2]
                + " "
                + _PRESERVATION
            ),
            (
                f"Create a vertical 4:5 product-variant presentation based only on colors and variants visible in the "
                f"reference images. {identity} Arrange the variants in a clean, consistent catalog composition without text. "
                + slot_directives[3]
                + " "
                + _PRESERVATION
            ),
            (
                f"Create a vertical 4:5 final commerce image showing a practical styling or use context suitable for "
                f"the product. {identity} Keep the composition inclusive and culturally neutral. No written size claims. "
                + slot_directives[4]
                + " "
                + _PRESERVATION
            ),
        ],
        video_prompt=(
            "An 8-second premium e-commerce product video based on the first frame. Clean neutral commercial lighting. "
            + _video_guard(taxonomy)
        ),
        market_angles={
            "en": "clear specifications and versatile styling",
            "ko": "accurate options and restrained marketplace presentation",
            "pt": "clear variation guidance and practical styling",
        },
    )


def _valid_plan_payload(payload: dict[str, Any]) -> bool:
    required = {
        "visual_theme",
        "main_prompt",
        "detail_prompts",
        "video_prompt",
        "market_angles",
    }
    if not required.issubset(payload):
        return False
    if (
        not isinstance(payload.get("detail_prompts"), list)
        or len(payload["detail_prompts"]) != 5
    ):
        return False
    if not all(
        isinstance(item, str) and 80 <= len(item) <= 5000
        for item in payload["detail_prompts"]
    ):
        return False
    if not isinstance(payload.get("market_angles"), dict):
        return False
    return all(
        isinstance(payload.get(key), str) and payload[key].strip()
        for key in ("visual_theme", "main_prompt", "video_prompt")
    )


def create_creative_plan(
    facts: ProductFacts,
    taxonomy: TaxonomyResult,
    vision: dict[str, Any],
    client: QwenClient | None,
) -> tuple[CreativePlan, str]:
    fallback = fallback_creative_plan(facts, taxonomy)
    if client is None:
        return fallback, "deterministic-fallback"

    system = (
        "You are an expert cross-border fashion e-commerce creative director. Return JSON only. "
        "Your plan must prioritize product identity preservation, factual accuracy, AliExpress-ready composition, "
        "cultural neutrality, and a coherent visual set. Generated images must contain no written text."
    )
    prompt = f"""
Plan exactly one main image, five detail images and one short product video.
Return JSON with exactly these keys:
- visual_theme: string
- main_prompt: detailed English image-edit prompt
- detail_prompts: array of exactly five detailed English image-edit prompts
- video_prompt: detailed English image-to-video prompt for 8 seconds
- market_angles: object with en, ko, pt strings

Hard constraints for every image prompt:
{_PRESERVATION}

Main image: square, clean near-white studio background, centered, no text. {_MAIN_PRESENTATION}
Details: vertical 4:5; cover overall view, construction/detail, verified features, variants, and practical context.
Do not request measurements, care instructions, material performance, certification, price, discount or brand claims.

Verified facts:
{json.dumps(facts.compact_dict(), ensure_ascii=False)}

Resolved category:
{json.dumps({"id": taxonomy.category.category_id, "name": taxonomy.category.name, "path": taxonomy.category.path}, ensure_ascii=False)}

Conservative source-image observations:
{json.dumps(vision, ensure_ascii=False)}
""".strip()
    try:
        payload = client.chat_json(system, prompt)
    except ApiError:
        return fallback, "deterministic-fallback"
    if not _valid_plan_payload(payload):
        return fallback, "invalid-model-plan"
    all_prompts = [
        payload["main_prompt"],
        payload["video_prompt"],
        *payload["detail_prompts"],
    ]
    if any(visual_prompt_violations(prompt) for prompt in all_prompts):
        return fallback, "content-compliance-guard"
    slot_directives = _detail_slot_directives(taxonomy)
    plan = CreativePlan(
        visual_theme=payload["visual_theme"].strip(),
        main_prompt=(
            payload["main_prompt"].strip()
            + " "
            + _MAIN_PRESENTATION
            + " "
            + _PRESERVATION
        ),
        detail_prompts=[
            item.strip()
            + " "
            + slot_directives[index]
            + " "
            + _PRESERVATION
            for index, item in enumerate(payload["detail_prompts"])
        ],
        video_prompt=payload["video_prompt"].strip() + " " + _video_guard(taxonomy),
        market_angles={
            key: str(value) for key, value in payload["market_angles"].items()
        },
    )
    return plan, client.config.chat_model
