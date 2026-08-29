"""Creative-plan validation and emergency fallback for the agent orchestrator."""

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
        "Show one sellable product in one verified variant as the single dominant subject. The complete product "
        "must be visible from its highest to lowest product edge, with every source-visible section unobstructed. "
        "No unsupported variant lineup, duplicate product, inset, split screen, montage or collage. "
    )
    if _is_children_product(facts, taxonomy):
        return common + (
            "Use a clean product-only studio presentation on an appropriate neutral support or surface; do not "
            "add a child, adult, hands, toys or character props."
        )
    if _source_supports_wearer(vision):
        return common + (
            "Prefer a product-only presentation on an appropriate neutral support or surface. A single naturally "
            "proportioned, fully clothed adult user is optional because a trusted source already shows a person; keep the product "
            "unobstructed and the background clean."
        )
    return common + (
        "Use a product-only presentation on an appropriate neutral support or surface; do not introduce a person that is absent "
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
        else "This is detail slot 5: create a product-only practical context using an appropriate clean surface, stand "
        "or support. Do not introduce a person, body, hand or accessory absent from trusted references."
    )
    return (
        (
            "complete_product",
            "This is detail slot 1: show the complete product from a clearly different viewing angle than the "
            "square hero, with every source-visible section readable and unobstructed.",
        ),
        (
            "primary_verified_detail",
            "This is detail slot 2: make a tight but readable close-up of the most category-defining construction or "
            "surface feature that is directly visible in the inspected references. Do not name or add an unseen part."
            + evidence_clause,
        ),
        (
            "secondary_verified_detail",
            "This is detail slot 3: show a different source-visible construction, edge, geometry, texture or surface-design "
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
            "Campaign Style Lock: restrained styling coordinated with every seller-verified product variant; realistic "
            "material and surface rendering with role-appropriate variation in background, lighting and framing; coherence comes "
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
                f"Create a vertical 4:5 detail-focused commerce image showing source-visible construction and design details. "
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
        main_candidate_count=3,
        detail_candidate_counts=[2] * 5,
        main_reference_roles=["hero", "front", "variant", "detail"],
        detail_reference_roles=[
            ["front", "hero", "lifestyle"],
            ["detail", "front", "side"],
            ["detail", "side", "back"],
            ["variant", "front", "hero"],
            ["lifestyle", "front", "hero"],
        ],
    )


def _candidate_count(value: Any, default: int) -> int:
    """Candidate counts are a resource boundary, not a creative policy."""

    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(parsed, 4))


def _role_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(
        dict.fromkeys(
            clean
            for item in value[:8]
            if (clean := " ".join(str(item).split())[:80])
        )
    )


def _normalize_creative_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Accept the native orchestrator shape and the legacy flat JSON shape."""

    main = payload.get("main")
    details = payload.get("details")
    video = payload.get("video")
    if isinstance(main, dict) and isinstance(details, list) and isinstance(video, dict):
        if len(details) != 5 or not all(isinstance(item, dict) for item in details):
            return None
        normalized = {
            "visual_theme": payload.get("visual_theme"),
            "main_prompt": main.get("prompt"),
            "main_candidate_count": main.get("candidate_count"),
            "main_reference_roles": main.get("reference_roles"),
            "detail_prompts": [item.get("prompt") for item in details],
            "detail_roles": [item.get("role") for item in details],
            "detail_candidate_counts": [item.get("candidate_count") for item in details],
            "detail_reference_roles": [item.get("reference_roles") for item in details],
            "video_prompt": video.get("prompt"),
            "market_angles": payload.get("market_angles"),
        }
    else:
        normalized = dict(payload)

    prompts = normalized.get("detail_prompts")
    roles = normalized.get("detail_roles")
    market_angles = normalized.get("market_angles")
    if not isinstance(prompts, list) or len(prompts) != 5:
        return None
    if not all(isinstance(item, str) and 40 <= len(item.strip()) <= 5000 for item in prompts):
        return None
    if roles is not None and (
        not isinstance(roles, list)
        or len(roles) != 5
        or not all(isinstance(item, str) and item.strip() for item in roles)
    ):
        return None
    if not isinstance(market_angles, dict):
        return None
    if not all(
        isinstance(normalized.get(key), str) and normalized[key].strip()
        for key in ("visual_theme", "main_prompt", "video_prompt")
    ):
        return None
    return normalized


def validate_creative_plan_payload(
    payload: dict[str, Any],
) -> tuple[CreativePlan | None, str]:
    """Validate and materialize the exact plan accepted by the orchestrator.

    Returning a correction message lets a native tool loop expose rejection as
    an observation instead of silently swapping in a different plan afterward.
    """

    normalized = _normalize_creative_payload(payload)
    if normalized is None:
        return None, (
            "creative_plan schema is invalid: provide a non-empty visual_theme, main/video prompts, "
            "exactly five detailed prompts and roles, and en/ko/pt market angles"
        )
    if not 40 <= len(normalized["main_prompt"].strip()) <= 5000:
        return None, "creative_plan.main.prompt must contain 40 to 5000 characters"
    if not 40 <= len(normalized["video_prompt"].strip()) <= 5000:
        return None, "creative_plan.video.prompt must contain 40 to 5000 characters"
    market_angles = normalized["market_angles"]
    if not all(
        isinstance(market_angles.get(key), str) and market_angles[key].strip()
        for key in ("en", "ko", "pt")
    ):
        return None, "creative_plan.market_angles must contain non-empty en, ko and pt strings"
    main_count = normalized.get("main_candidate_count")
    if main_count is not None and (
        isinstance(main_count, bool)
        or not isinstance(main_count, int)
        or not 1 <= main_count <= 4
    ):
        return None, "creative_plan.main.candidate_count must be an integer from 1 to 4"
    detail_counts = normalized.get("detail_candidate_counts")
    if detail_counts is not None and (
        not isinstance(detail_counts, list)
        or len(detail_counts) != 5
        or any(
            isinstance(item, bool) or not isinstance(item, int) or not 1 <= item <= 4
            for item in detail_counts
        )
    ):
        return None, "every creative_plan.details[].candidate_count must be an integer from 1 to 4"
    prompt_rows = [
        ("main", normalized["main_prompt"]),
        ("video", normalized["video_prompt"]),
        *[
            (f"detail_{index}", prompt)
            for index, prompt in enumerate(normalized["detail_prompts"], start=1)
        ],
    ]
    rejected = {
        name: visual_prompt_violations(prompt)
        for name, prompt in prompt_rows
        if visual_prompt_violations(prompt)
    }
    if rejected:
        return None, (
            "creative_plan contains forbidden visual-prompt terms; rewrite the affected prompts without "
            f"requesting or naming them: {json.dumps(rejected, ensure_ascii=False)}"
        )
    theme = normalized["visual_theme"].strip()
    roles = normalized.get("detail_roles") or [f"detail_{index}" for index in range(1, 6)]
    counts = normalized.get("detail_candidate_counts")
    counts = counts if isinstance(counts, list) and len(counts) == 5 else [2] * 5
    reference_roles = normalized.get("detail_reference_roles")
    reference_roles = (
        reference_roles
        if isinstance(reference_roles, list) and len(reference_roles) == 5
        else [[] for _ in range(5)]
    )
    return CreativePlan(
        visual_theme=theme,
        main_prompt=normalized["main_prompt"].strip() + " " + _PRESERVATION,
        detail_prompts=[
            item.strip()
            + " Campaign styling: "
            + theme
            + " Keep one coherent full-frame composition for the assigned job. "
            + _PRESERVATION
            for item in normalized["detail_prompts"]
        ],
        video_prompt=(
            normalized["video_prompt"].strip()
            + " Preserve exact reference-product identity, construction, pattern and visible color in every frame."
            " Do not add text, claims, marks, accessories, people or product components absent from trusted evidence."
        ),
        market_angles={
            key: str(value).strip()[:1000]
            for key, value in normalized["market_angles"].items()
            if key in {"en", "ko", "pt"} and str(value).strip()
        },
        detail_roles=[" ".join(str(item).split())[:100] for item in roles],
        main_candidate_count=_candidate_count(normalized.get("main_candidate_count"), 3),
        detail_candidate_counts=[_candidate_count(item, 2) for item in counts],
        main_reference_roles=_role_list(normalized.get("main_reference_roles")),
        detail_reference_roles=[_role_list(item) for item in reference_roles],
    ), ""


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
    orchestrated = (
        agent_guidance.get("creative_plan")
        if isinstance(agent_guidance, dict)
        else None
    )
    if isinstance(orchestrated, dict):
        plan, _ = validate_creative_plan_payload(orchestrated)
        if plan is not None:
            return plan, "agent-orchestrator"
    if client is None:
        return fallback, "deterministic-fallback"

    system = (
        "You are an expert cross-border marketplace creative director. Return JSON only. "
        "Your plan must prioritize product identity preservation, factual accuracy, AliExpress-ready composition, "
        "cultural neutrality, and a coherent visual set. Generated images must contain no written text."
        + ("\n\n" + skill_instructions if skill_instructions else "")
    )
    prompt = f"""
Plan exactly one main image, five complementary detail images and one short product video.
Return JSON with exactly these keys:
- visual_theme: string
- main_prompt: detailed English image-edit prompt
- detail_prompts: array of exactly five detailed English image-edit prompts
- detail_roles: array of exactly five concise machine-readable commercial jobs
- main_candidate_count: integer from 1 to 4
- detail_candidate_counts: array of five integers from 1 to 4
- main_reference_roles: source-image roles to prioritize
- detail_reference_roles: array of five source-image role arrays
- video_prompt: detailed English image-to-video prompt for 8 seconds
- market_angles: object with en, ko, pt strings

Hard constraints for every image prompt:
{_PRESERVATION}

Main image: square, clean near-white studio background, centered, no text. {_main_presentation(facts, taxonomy, vision)}
The visual_theme must define a restrained campaign direction with explicit product-identity no-drift
rules. It may accommodate every seller-verified product color. Palette, background, lighting, framing
and product coverage may vary by commercial role and target-market context; do not impose one global
color count, background system or coverage percentage.
Details: vertical 4:5. Choose the five commercial jobs and their order from the verified evidence.
Make the set useful and non-redundant. Never name a component merely because it is common for the category.
Do not request measurements, care instructions, material performance, certification, price, discount or brand claims.

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
    plan, _ = validate_creative_plan_payload(payload)
    if plan is None:
        return fallback, "invalid-model-plan"
    return plan, client.config.chat_model
