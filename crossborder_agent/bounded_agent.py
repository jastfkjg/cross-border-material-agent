"""LLM-led planning and whole-delivery evaluation with a bounded tool surface."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .agent_tools import BoundedToolRegistry
from .api import ApiError, QwenClient
from .media import MediaError, inspect_image, inspect_image_quality, inspect_video
from .models import (
    AgentAction,
    AgentEvaluation,
    AssetResult,
    CreativePlan,
    ProductFacts,
    TaxonomyResult,
)
from .skill_runtime import SkillLibrary


_RUBRIC_WEIGHTS = {
    "A1": 25,
    "A2": 20,
    "A3": 18,
    "A4": 15,
    "A5": 10,
    "A6": 7,
    "A7": 5,
}


class BoundedDeliveryAgent:
    """Let the model decide what needs work while code enforces hard boundaries."""

    def __init__(
        self,
        client: QwenClient | None,
        logger: logging.Logger,
        skills: SkillLibrary | None = None,
    ):
        self.client = client
        self.logger = logger
        self.skills = skills or SkillLibrary()

    def plan_delivery(
        self,
        facts: ProductFacts,
        taxonomy: TaxonomyResult,
        vision: dict[str, Any],
        tools: BoundedToolRegistry,
    ) -> dict[str, Any]:
        default = {
            "creative_direction": "Source-faithful international marketplace presentation",
            "localization_priorities": {
                "en": "natural en-US commerce copy",
                "ko": "natural ko-KR commerce copy",
                "pt": "natural pt-BR commerce copy",
            },
            "risk_priorities": ["A1", "A5", "A6", "A7"],
            "visual_sequence": [
                "hero",
                "overall",
                "construction",
                "verified feature",
                "variants",
                "fit or size guidance",
            ],
            "video_strategy": "stable product-led short presentation",
            "max_repair_rounds": 1,
            "max_actions_per_round": 3,
            "minimum_weighted_score": 82,
        }
        if self.client is None:
            return default
        system = (
            "You are the manager of a bounded cross-border commerce material agent. "
            "Return JSON only. Plan outcomes and priorities; do not invent product facts. "
            "The executor exposes only the listed tools and a deterministic final specification gate.\n\n"
            + self.skills.combine(
                "delivery-planning",
                "product-grounding",
                "aliexpress-taxonomy",
                "aliexpress-content-compliance",
                "rubric-evaluation",
            )
        )
        prompt = f"""
Create an execution policy for one AliExpress-ready delivery under a 30-minute total limit.
The initial production skeleton is fixed for reliability, but your plan controls creative emphasis,
localization priorities, evaluation threshold and bounded repair budget.

Return exactly these keys:
- creative_direction: concise English direction that can be passed to creative models
- localization_priorities: object with en, ko, pt strings
- risk_priorities: ordered array containing only A1 through A7
- visual_sequence: exactly six concise slot objectives, main image first
- video_strategy: concise English shot and motion strategy
- max_repair_rounds: integer 1 or 2
- max_actions_per_round: integer 1 through 4
- minimum_weighted_score: integer 70 through 95

Verified product facts:
{json.dumps(facts.compact_dict(), ensure_ascii=False)}

Resolved platform taxonomy:
{json.dumps({"category": taxonomy.category.name, "category_id": taxonomy.category.category_id, "path": taxonomy.category.path, "missing_required": taxonomy.missing_required}, ensure_ascii=False)}

Source visual observations:
{json.dumps(vision, ensure_ascii=False)}

Available repair tools for later rounds:
{json.dumps(tools.catalog(), ensure_ascii=False)}
""".strip()
        try:
            payload = self.client.chat_json(system, prompt)
        except ApiError as exc:
            self.logger.warning("LLM 交付规划不可用，采用保守有界策略: %s", exc)
            return default
        return self._normalize_plan(payload, default)

    @staticmethod
    def _normalize_plan(payload: dict[str, Any], default: dict[str, Any]) -> dict[str, Any]:
        result = dict(default)
        for key in ("creative_direction", "video_strategy"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                result[key] = value.strip()[:3000]
        locales = payload.get("localization_priorities")
        if isinstance(locales, dict) and all(
            isinstance(locales.get(key), str) and locales[key].strip()
            for key in ("en", "ko", "pt")
        ):
            result["localization_priorities"] = {
                key: locales[key].strip()[:1000] for key in ("en", "ko", "pt")
            }
        risks = payload.get("risk_priorities")
        if isinstance(risks, list):
            clean = [item for item in risks if item in _RUBRIC_WEIGHTS]
            if clean:
                result["risk_priorities"] = list(dict.fromkeys(clean))
        sequence = payload.get("visual_sequence")
        if (
            isinstance(sequence, list)
            and len(sequence) == 6
            and all(isinstance(item, str) and item.strip() for item in sequence)
        ):
            result["visual_sequence"] = [item.strip()[:500] for item in sequence]
        for key, minimum, maximum in (
            ("max_repair_rounds", 1, 2),
            ("max_actions_per_round", 1, 4),
            ("minimum_weighted_score", 70, 95),
        ):
            value = payload.get(key)
            if isinstance(value, int):
                result[key] = min(maximum, max(minimum, value))
        return result

    def evaluate_delivery(
        self,
        *,
        round_index: int,
        facts: ProductFacts,
        taxonomy: TaxonomyResult,
        creative_plan: CreativePlan,
        agent_plan: dict[str, Any],
        assets: list[AssetResult],
        localization_payloads: dict[str, dict[str, Any]],
        localization_sources: dict[str, str],
        visual_set_review: dict[str, Any],
        work_dir: Path,
        tools: BoundedToolRegistry,
    ) -> AgentEvaluation | None:
        if self.client is None:
            return None
        manifest, image_urls, video_urls = self._artifact_evidence(
            facts, assets, work_dir
        )
        copy_artifacts = self._copy_artifact_evidence(work_dir)
        system = (
            "You are an independent multimodal acceptance evaluator, not the producer. Return JSON only. "
            "Judge the complete delivery against A1-A7 and select only high-value repairs. Do not replace a "
            "weak generated asset with a source fallback: ask the matching tool to revise or regenerate it. "
            "Do not reward strategy claims that are not evidenced by the artifact manifest or supplied media.\n\n"
            + self.skills.combine(
                "rubric-evaluation",
                "product-grounding",
                "aliexpress-taxonomy",
                "aliexpress-content-compliance",
                "marketplace-localization",
                "commerce-visuals",
                "commerce-video",
            )
        )
        prompt = f"""
Evaluate this whole delivery. Source-reference images are listed first in the visual input map;
delivery image URLs follow. The video input, when present, is the actual generated video URL.
Local deterministic artifacts cannot be uploaded to the model and must be judged from their
physical inspection and provenance in the manifest.

Return exactly these keys:
- ready_for_delivery: boolean
- weighted_score: number from 0 to 100
- dimension_scores: object with numeric A1,A2,A3,A4,A5,A6,A7 scores from 0 to 100
- summary: concise diagnosis
- issues: array of objects with dimension, severity (blocker/major/minor), target, evidence, expected
- repair_actions: ordered array of at most {agent_plan.get('max_actions_per_round', 3)} objects with:
  tool, target, instruction, reason, dimension, priority (1 highest to 4 lowest)

Only choose tool/target combinations from the catalog. Each instruction must be an actionable,
artifact-specific correction prompt grounded in the evidence. Prefer revising the weakest high-weight
dimension. Do not request cosmetic changes that risk factual identity. Set ready_for_delivery true only
when the weighted score is at least {agent_plan.get('minimum_weighted_score', 82)}, there are no blocker
issues, and A1/A2/A5 have no major issue.
Treat a deterministic or validation-error copy source as a quality degradation: inspect its rendered
shopper preview and request revise_localized_copy when it is generic, process-oriented or misses a
distinctive source-title design detail. Do not reward raw evidence volume. Localized deliverables must
not contain Chinese source values, JSON pointers, canonical/evidence labels or duplicated audit tables.
Treat the six-image set review below as independent evidence about semantic duplication and missing
commercial roles. A set-level repair target is higher priority than a cosmetic per-image preference.
It may predate repairs in later rounds, so corroborate it against the current manifest and media.

Rubric weights:
{json.dumps(_RUBRIC_WEIGHTS)}

Verified fact ledger:
{json.dumps(facts.compact_dict(), ensure_ascii=False)}

Resolved platform result:
{json.dumps({"category_id": taxonomy.category.category_id, "category": taxonomy.category.name, "path": taxonomy.category.path, "attributes": [{"id": item.attr_id, "name": item.name, "value_id": item.value_id, "value": item.platform_value} for item in taxonomy.attributes], "missing_required": taxonomy.missing_required}, ensure_ascii=False)}

Manager plan:
{json.dumps(agent_plan, ensure_ascii=False)}

Creative plan actually used:
{json.dumps({"theme": creative_plan.visual_theme, "main": creative_plan.main_prompt, "details": creative_plan.detail_prompts, "video": creative_plan.video_prompt}, ensure_ascii=False)}

Localized copy payloads:
{json.dumps(localization_payloads, ensure_ascii=False)}

Copy generation sources:
{json.dumps(localization_sources, ensure_ascii=False)}

Six-image set review:
{json.dumps(visual_set_review, ensure_ascii=False)}

Rendered localized-copy evidence:
{json.dumps(copy_artifacts, ensure_ascii=False)}

Artifact manifest and local physical inspection:
{json.dumps(manifest, ensure_ascii=False)}

Visual input map:
{json.dumps([{"input_index": index, "url": url} for index, url in enumerate(image_urls)], ensure_ascii=False)}

Available repair tools:
{json.dumps(tools.catalog(), ensure_ascii=False)}
""".strip()
        try:
            payload = self.client.chat_json(
                system,
                prompt,
                images=image_urls,
                videos=video_urls,
                model=self.client.config.review_model,
            )
        except ApiError as exc:
            self.logger.warning("全局交付评估不可用，保留当前已校验版本: %s", exc)
            return None
        return self._parse_evaluation(
            payload,
            round_index=round_index,
            tools=tools,
            action_limit=int(agent_plan.get("max_actions_per_round", 3)),
        )

    @staticmethod
    def _copy_artifact_evidence(work_dir: Path) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for language in ("en", "ko", "pt"):
            path = work_dir / f"product_description_{language}.md"
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                evidence.append({"language": language, "readable": False})
                continue
            headings = [
                line.removeprefix("## ").strip()
                for line in text.splitlines()
                if line.startswith("## ")
            ]
            first = text.find("\n## ")
            second = text.find("\n## ", first + 1) if first >= 0 else -1
            third = text.find("\n## ", second + 1) if second >= 0 else -1
            shopper_preview = text[: third if third > 0 else 2500]
            localized_surface = re.sub(r"https?://[^\s)>]+", "", text)
            evidence.append(
                {
                    "language": language,
                    "readable": True,
                    "characters": len(text),
                    "headings": headings,
                    "chinese_character_count": len(
                        re.findall(r"[\u4e00-\u9fff]", localized_surface)
                    ),
                    "json_pointer_count": text.count("/ret/result/result"),
                    "shopper_preview": shopper_preview[:2500],
                }
            )
        return evidence

    def _artifact_evidence(
        self, facts: ProductFacts, assets: list[AssetResult], work_dir: Path
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        manifest: list[dict[str, Any]] = []
        source_references = list(
            dict.fromkeys(
                facts.product_image_urls[:3]
                + facts.sku_image_urls[:1]
                + facts.description_image_urls[:1]
            )
        )
        delivery_urls = [
            asset.source_url
            for asset in assets
            if asset.name.endswith(".jpeg") and asset.source_url
        ]
        image_urls = list(dict.fromkeys(source_references + delivery_urls))
        video_urls = [
            asset.source_url
            for asset in assets
            if asset.name == "product_video.mp4" and asset.generated and asset.source_url
        ][:1]
        for asset in assets:
            path = work_dir / asset.name
            item: dict[str, Any] = {
                "name": asset.name,
                "generated": asset.generated,
                "model": asset.model,
                "source_url": asset.source_url,
                "description": asset.description,
                "fallback_reason": asset.fallback_reason,
                "visual_input_index": (
                    image_urls.index(asset.source_url)
                    if asset.source_url in image_urls
                    else None
                ),
            }
            try:
                item["bytes"] = path.stat().st_size
                if asset.name.endswith(".jpeg"):
                    info = inspect_image(path)
                    quality = inspect_image_quality(path)
                    item["physical"] = {
                        "format": info.format,
                        "width": info.width,
                        "height": info.height,
                        "entropy": quality.entropy if quality else None,
                        "luminance_stddev": quality.luminance_stddev if quality else None,
                        "difference_hash": quality.difference_hash if quality else None,
                    }
                elif asset.name == "product_video.mp4":
                    item["physical"] = inspect_video(path)
            except (OSError, MediaError) as exc:
                item["physical_error"] = str(exc)
            manifest.append(item)
        return manifest, image_urls, video_urls

    @staticmethod
    def _parse_evaluation(
        payload: dict[str, Any],
        *,
        round_index: int,
        tools: BoundedToolRegistry,
        action_limit: int,
    ) -> AgentEvaluation:
        raw_scores = payload.get("dimension_scores")
        scores: dict[str, float] = {}
        if isinstance(raw_scores, dict):
            for dimension in _RUBRIC_WEIGHTS:
                value = raw_scores.get(dimension)
                if isinstance(value, (int, float)):
                    scores[dimension] = max(0.0, min(100.0, float(value)))
        weighted = payload.get("weighted_score")
        if not isinstance(weighted, (int, float)):
            weighted = sum(
                scores.get(key, 0.0) * weight / 100
                for key, weight in _RUBRIC_WEIGHTS.items()
            )
        issues = payload.get("issues")
        clean_issues = [item for item in issues if isinstance(item, dict)] if isinstance(issues, list) else []
        actions: list[AgentAction] = []
        seen_targets: set[tuple[str, str]] = set()
        raw_actions = payload.get("repair_actions")
        if isinstance(raw_actions, list):
            for item in raw_actions:
                if not isinstance(item, dict):
                    continue
                tool = str(item.get("tool") or "").strip()
                target = str(item.get("target") or "").strip()
                instruction = str(item.get("instruction") or "").strip()
                if not instruction or not tools.accepts(tool, target):
                    continue
                key = (tool, target)
                if key in seen_targets:
                    continue
                seen_targets.add(key)
                priority = item.get("priority")
                actions.append(
                    AgentAction(
                        tool=tool,
                        target=target,
                        instruction=instruction[:4000],
                        reason=str(item.get("reason") or "")[:1000],
                        dimension=str(item.get("dimension") or "")[:8],
                        priority=priority if isinstance(priority, int) else 4,
                    )
                )
        actions.sort(key=lambda item: item.priority)
        return AgentEvaluation(
            round_index=round_index,
            ready_for_delivery=payload.get("ready_for_delivery") is True,
            weighted_score=max(0.0, min(100.0, float(weighted))),
            dimension_scores=scores,
            summary=str(payload.get("summary") or "")[:3000],
            issues=clean_issues[:20],
            repair_actions=actions[: max(1, min(4, action_limit))],
        )
