"""Bounded creative planning for the fixed delivery contract."""

from __future__ import annotations

import json
from typing import Any

from .api import ApiError, QwenClient
from .compliance import visual_prompt_violations
from .models import CreativePlan, ProductFacts, TaxonomyResult


_PRESERVATION = (
    "Preserve the exact product identity, silhouette, construction, pattern, trims and visible color from the "
    "reference images. Do not add, remove or alter any visible product component, construction detail, fastening, "
    "surface design or included accessory. "
    "Do not invent text, measurements, materials, certifications or brand marks. No watermark."
)


def _is_children_product(facts: ProductFacts, taxonomy: TaxonomyResult) -> bool:
    """Use category meaning, not benchmark category IDs, for the wearer safety rule."""

    category_text = " ".join(
        (
            facts.source_category_name,
            taxonomy.category.name,
            taxonomy.category.path,
        )
    ).casefold()
    return any(
        marker in category_text
        for marker in ("child", "kid", "boy", "girl", "baby", "infant", "童", "婴")
    )


def _source_supports_wearer(vision: dict[str, Any] | None) -> bool:
    images = vision.get("source_images") if isinstance(vision, dict) else None
    return bool(
        isinstance(images, list)
        and any(isinstance(item, dict) and item.get("has_person") is True for item in images)
    )


def _verified_variants(
    facts: ProductFacts | None, vision: dict[str, Any] | None
) -> bool:
    """Require seller options and pixel evidence before asking for a variant lineup."""

    seller_colors = {
        item.value.strip().casefold()
        for item in (facts.attributes if facts is not None else [])
        if "颜色" in item.name or "color" in item.name.casefold()
        if item.value.strip()
    }
    if len(seller_colors) < 2 or not isinstance(vision, dict):
        return False
    observed = {
        str(value).strip().casefold()
        for value in vision.get("visible_colors", [])
        if str(value).strip()
    }
    images = vision.get("source_images")
    if isinstance(images, list):
        observed.update(
            str(item.get("dominant_color") or "").strip().casefold()
            for item in images
            if isinstance(item, dict)
            and item.get("role") in {"variant", "hero", "front", "back", "side"}
            and str(item.get("dominant_color") or "").strip()
        )
    return len(observed) >= 2


def _observed_feature_context(vision: dict[str, Any] | None) -> str:
    """Expose only concise visual evidence; never manufacture category-specific parts."""

    if not isinstance(vision, dict):
        return ""
    features: list[str] = []
    for key in ("visible_design_features", "preservation_constraints"):
        values = vision.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            clean = " ".join(str(value).split())[:180]
            if clean and clean not in features:
                features.append(clean)
            if len(features) >= 8:
                break
    return "; ".join(features)


def _main_presentation(
    facts: ProductFacts, taxonomy: TaxonomyResult, vision: dict[str, Any] | None
) -> str:
    common = (
        "Show one sellable product in one verified colorway as the single dominant subject. The complete garment "
        "must be visible from its highest to lowest product edge, with every source-visible section unobstructed. "
        "No colorway lineup, "
        "duplicate garment, inset, split screen, montage or collage. "
    )
    if _is_children_product(facts, taxonomy):
        return common + (
            "Use a product-only studio presentation such as a clean flat lay, hanger or invisible form; do not "
            "add a child, adult, hands, toys or character props."
        )
    if _source_supports_wearer(vision):
        return common + (
            "Prefer a product-only flat lay, hanger or invisible-form presentation. A single naturally proportioned, "
            "fully clothed adult wearer is optional because a trusted source already shows a wearer; keep the product "
            "unobstructed and the background clean."
        )
    return common + (
        "Use a product-only flat lay, hanger or invisible-form presentation; do not introduce a wearer that is absent "
        "from the inspected source references."
    )


def _detail_slot_specs(
    taxonomy: TaxonomyResult,
    facts: ProductFacts | None = None,
    vision: dict[str, Any] | None = None,
) -> tuple[tuple[str, str], ...]:
    evidence = _observed_feature_context(vision)
    evidence_clause = (
        f" Inspected source evidence includes: {evidence}."
        if evidence
        else " Use only construction and surface details directly visible in the inspected references."
    )
    variants = _verified_variants(facts, vision)
    # The taxonomy value is intentionally not used to choose construction details.  It is
    # still consulted only for the safety-sensitive wearer decision below.
    children = bool(facts and _is_children_product(facts, taxonomy))
    wearer_supported = _source_supports_wearer(vision) and not children
    slot_4_role = "verified_variants" if variants else "verified_alternate_view"
    slot_4 = (
        "This is detail slot 4: create a clean catalog comparison using only distinct variants directly visible in "
        "the inspected references. Keep construction identical and do not add labels or swatches."
        if variants
        else "This is detail slot 4: show a source-supported alternate viewpoint or a third, non-redundant visible "
        "detail. Do not assume a back view, color variant, print continuation or hidden construction."
    )
    context_role = "verified_use_context" if wearer_supported else "product_only_context"
    context = (
        "This is detail slot 5: show one naturally proportioned, fully clothed adult in the kind of restrained use "
        "context already supported by a trusted source. Keep the complete product unobstructed and do not alter fit."
        if wearer_supported
        else "This is detail slot 5: create a product-only practical context using a clean flat lay, hanger or "
        "invisible form. Do not introduce a wearer, body, hand or product accessory absent from trusted references."
    )
    return (
        (
            "complete_product",
            "This is detail slot 1: show one complete three-quarter product presentation, visibly different from "
            "the square hero, with every source-visible section readable and unobstructed.",
        ),
        (
            "primary_verified_detail",
            "This is detail slot 2: make a tight but readable close-up of the most category-defining construction or "
            "surface feature that is directly visible in the inspected references. Do not name or add an unseen part."
            + evidence_clause,
        ),
        (
            "secondary_verified_detail",
            "This is detail slot 3: show a different source-visible construction, edge, drape, texture or surface-design "
            "feature while retaining enough product context for identification. It must not repeat slot 2."
            + evidence_clause,
        ),
        (slot_4_role, slot_4),
        (context_role, context),
    )


def _detail_slot_directives(
    taxonomy: TaxonomyResult,
    facts: ProductFacts | None = None,
    vision: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    return tuple(
        directive for _, directive in _detail_slot_specs(taxonomy, facts, vision)
    )


def _video_guard(
    facts: ProductFacts, taxonomy: TaxonomyResult, vision: dict[str, Any] | None
) -> str:
    evidence = _observed_feature_context(vision)
    focus = (
        "Use a product-only presentation; do not add a child, adult, hands, toys or character graphics."
        if _is_children_product(facts, taxonomy)
        else "Keep every source-visible product section, edge and construction detail stable and unobstructed."
    )
    if evidence:
        focus += f" Preserve these inspected visual anchors: {evidence}."
    return (
        "Use one continuous slow 10-to-15-degree camera arc with a subtle push-in; no scene cuts. "
        + focus
        + " Preserve exact product construction, pattern and color in every frame. No morphing, duplicate product, "
        "new accessories, hands covering the product, text, logo animation, camera shake, flicker, speech or music."
    )


def _planning_vision_context(vision: dict[str, Any]) -> dict[str, Any]:
    """Keep full-scan evidence useful without sending every per-image flag to planning."""

    compact = {
        key: vision.get(key)
        for key in (
            "product_type",
            "visible_colors",
            "visible_design_features",
            "image_quality_notes",
            "prohibited_or_risky_visuals",
            "preservation_constraints",
        )
        if vision.get(key)
    }
    images = vision.get("source_images")
    if isinstance(images, list):
        roles: dict[str, int] = {}
        for item in images:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "unknown")
            roles[role] = roles.get(role, 0) + 1
        compact["scan_summary"] = {
            "total_images": len(images),
            "inspected_images": sum(
                isinstance(item, dict) and item.get("inspection_complete") is True
                for item in images
            ),
            "roles": roles,
            "wearer_reference_present": _source_supports_wearer(vision),
            "legible_size_rows": len(vision.get("size_chart_rows") or []),
        }
    return compact


def fallback_creative_plan(
    facts: ProductFacts,
    taxonomy: TaxonomyResult,
    vision: dict[str, Any] | None = None,
) -> CreativePlan:
    identity = f"Product type: {facts.source_category_name}."
    observed = _observed_feature_context(vision)
    if observed:
        identity += f" Inspected visible source evidence: {observed}."
    identity += " When text facts and reference pixels appear to conflict, preserve the reference product pixels."
    slot_specs = _detail_slot_specs(taxonomy, facts, vision)
    slot_directives = tuple(item[1] for item in slot_specs)
    return CreativePlan(
        visual_theme=(
            "Campaign Style Lock: restrained styling coordinated with every seller-verified product color; realistic "
            "fabric rendering and role-appropriate variation in background, lighting and framing; coherence comes "
            "from product identity rather than identical treatment; no typography, invented logos, or style drift"
        ),
        main_prompt=(
            f"Create a square premium e-commerce hero image using the reference product. {identity} "
            "Use a clean white-to-very-light-gray studio background, soft realistic shadow, even lighting, "
            "accurate color, centered composition, and safe margins with the product visually dominant. "
            "No promotional text, badges, borders, collage or extra props. "
            + _main_presentation(facts, taxonomy, vision)
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
                f"Create a vertical 4:5 evidence-led catalog photograph for its assigned commercial job. {identity} "
                "Use one clean, consistent full-frame composition without text, labels, swatches or invented variants. "
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
            + _video_guard(facts, taxonomy, vision)
        ),
        market_angles={
            "en": "clear specifications and versatile styling",
            "ko": "accurate options and restrained marketplace presentation",
            "pt": "clear variation guidance and practical styling",
        },
        detail_roles=[item[0] for item in slot_specs],
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
    *,
    agent_guidance: dict[str, Any] | None = None,
    skill_instructions: str = "",
) -> tuple[CreativePlan, str]:
    fallback = fallback_creative_plan(facts, taxonomy, vision)
    if client is None:
        return fallback, "deterministic-fallback"

    system = (
        "You are an expert cross-border fashion e-commerce creative director. Return JSON only. "
        "Your plan must prioritize product identity preservation, factual accuracy, AliExpress-ready composition, "
        "cultural neutrality, and a coherent visual set. Generated images must contain no written text."
        + ("\n\n" + skill_instructions if skill_instructions else "")
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

Main image: square, clean near-white studio background, centered, no text. {_main_presentation(facts, taxonomy, vision)}
The visual_theme must define a restrained campaign direction with explicit product-identity no-drift
rules. It may accommodate every seller-verified product color. Palette, background, lighting, framing
and product coverage may vary by commercial role and target-market context; do not impose one global
color count, background system or coverage percentage.
Details: vertical 4:5; assign exactly one primary commercial job to each slot. Cover the complete product,
two distinct source-visible details, one verified alternate view or genuinely verified variant comparison,
and a source-supported practical context. Never name a garment part merely because it is common for the category.
A perceptually different pose is not sufficient if it repeats another slot's job.
Do not request measurements, care instructions, material performance, certification, price, discount or brand claims.

The five slot jobs are fixed by code and cannot be reordered or redefined. Your detail_prompts are optional
styling proposals only; they will not replace the canonical storyboard contract:
{json.dumps([{"slot": index + 1, "role": role, "directive": directive} for index, (role, directive) in enumerate(_detail_slot_specs(taxonomy, facts, vision))], ensure_ascii=False)}

Verified facts:
{json.dumps(facts.compact_dict(), ensure_ascii=False)}

Resolved category:
{json.dumps({"id": taxonomy.category.category_id, "name": taxonomy.category.name, "path": taxonomy.category.path}, ensure_ascii=False)}

Conservative source-image observations:
{json.dumps(_planning_vision_context(vision), ensure_ascii=False)}

Bounded manager guidance:
{json.dumps(agent_guidance or {}, ensure_ascii=False)}
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
    # A previous implementation concatenated the model's independently planned
    # storyboard with a second fixed storyboard.  Contradictory commands (for
    # example, close-up + complete product) predictably produced split screens.
    # The deterministic storyboard is now the single structural source of truth;
    # the model contributes the campaign theme, hero/video treatment and market
    # angles, but cannot silently reassign a detail slot.
    canonical = fallback.detail_prompts
    plan = CreativePlan(
        visual_theme=payload["visual_theme"].strip(),
        main_prompt=(
            payload["main_prompt"].strip()
            + " "
            + _main_presentation(facts, taxonomy, vision)
            + " "
            + _PRESERVATION
        ),
        detail_prompts=[
            item
            + " Campaign styling lock: "
            + payload["visual_theme"].strip()
            + " The assigned slot has exactly one commercial job; do not add panels, insets, grids, or a second view."
            for item in canonical
        ],
        video_prompt=(
            payload["video_prompt"].strip()
            + " "
            + _video_guard(facts, taxonomy, vision)
        ),
        market_angles={
            key: str(value) for key, value in payload["market_angles"].items()
        },
        detail_roles=list(fallback.detail_roles),
    )
    return plan, client.config.chat_model
