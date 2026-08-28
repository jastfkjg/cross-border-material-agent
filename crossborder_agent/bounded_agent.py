"""LLM-led planning and whole-delivery evaluation with a bounded tool surface."""

from __future__ import annotations

import concurrent.futures
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
from .qa import _description_language_surfaces
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

_RUBRIC_DEFINITIONS = {
    "A1": {
        "name": "Content compliance",
        "scope": "All buyer-facing text, readable text in images/video, and visual elements.",
        "judge": "Apply only the supplied AliExpress material-content rules. Do not perform independent legal or intellectual-property review. Product-fact disagreement belongs to A5, not A1.",
        "weight": 25,
    },
    "A2": {
        "name": "Artifact specification compliance",
        "scope": "Completeness and physical file specifications only; do not judge semantic correctness here.",
        "judge": "Each text file must exist and be below 1 MB; hero must be jpeg/png and at least 800x800; every detail image must be jpeg/png, both dimensions above 260 px, and at most 5 MB; video must exist, be playable mp4/mov, and below 200 MB.",
        "weight": 20,
    },
    "A3": {
        "name": "Category and attribute accuracy",
        "scope": "Leaf category, product attribute key/value enumerations, and sales/SKU attribute values.",
        "judge": "Compare exact supplied platform identifiers and enumerations. Do not substitute a plausible free-text category for the supplied platform result.",
        "weight": 18,
    },
    "A4": {
        "name": "Localization adaptation",
        "scope": "Visual context, native wording and spelling, sizing, measurement units, cultural fit, gender/body presentation, and marketplace-channel adaptation for en-US, ko-KR, and pt-BR.",
        "judge": "Score native-market suitability without inventing regional size equivalence or unsupported cultural claims.",
        "weight": 15,
    },
    "A5": {
        "name": "Product fact consistency",
        "scope": "Every verifiable copy/media claim and its source match, including consistency across text, images, video, structured seller data, and reconciled visual evidence.",
        "judge": "Unsupported, unlabelled, source-conflicting, or cross-asset contradictory claims lose A5. The reconciled fact ledger is authoritative when structured appearance text conflicts with trusted source pixels.",
        "weight": 10,
    },
    "A6": {
        "name": "Image usability rate",
        "scope": "All images generated in the run.",
        "judge": "An image is usable only when it meets platform specifications and has no major quality, identity, construction, anatomy, text, or composition defect. At least 80% usable is the pass threshold.",
        "weight": 7,
    },
    "A7": {
        "name": "Video usability",
        "scope": "All generated/delivered videos.",
        "judge": "Judge playability, required physical specification, and intolerable visual defects. Do not score a video defect under A2 unless it is specifically a physical-format failure, or under A5 unless it makes a false product claim.",
        "weight": 5,
    },
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
        *,
        use_model: bool = True,
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
            "minimum_weighted_score": 90,
        }
        if self.client is None or not use_model:
            return default
        system = (
            "You are the manager of a bounded cross-border commerce material agent. "
            "Return JSON only. Plan outcomes and priorities; do not invent product facts. "
            "The executor exposes only the listed tools and a deterministic final specification gate.\n\n"
            + self.skills.compile(
                "manager",
                "delivery-quality",
                "product-grounding",
                "aliexpress-taxonomy",
            )
        )
        prompt = f"""
Create an execution policy for one AliExpress-ready delivery under a 30-minute total limit.
The initial production skeleton is fixed for reliability, but your plan controls creative emphasis,
localization priorities and evaluation threshold. The executor independently
performs up to three whole-delivery evaluations and uses the intervals between
them for bounded repairs when the remaining time allows.

Return exactly these keys:
- creative_direction: concise English direction that can be passed to creative models
- localization_priorities: object with en, ko, pt strings
- risk_priorities: ordered array containing only A1 through A7
- visual_sequence: exactly six concise slot objectives, main image first
- video_strategy: concise English shot and motion strategy
- minimum_weighted_score: integer 90 through 95

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
            ("minimum_weighted_score", 90, 95),
        ):
            value = payload.get(key)
            if isinstance(value, int):
                result[key] = min(maximum, max(minimum, value))
        return result

    def _evaluation_models(self) -> tuple[str, ...]:
        if self.client is None:
            return ()
        configured = getattr(self.client, "evaluation_models", ())
        if callable(configured):
            configured = configured()
        models = [str(item).strip() for item in configured if str(item).strip()]
        if len(dict.fromkeys(models)) < 2:
            config = self.client.config
            models.extend(
                [
                    str(getattr(config, "review_model", "") or ""),
                    str(getattr(config, "review_fallback_model", "") or ""),
                    "qwen3.8-max",
                    "qwen3.7-plus",
                ]
            )
        return tuple(dict.fromkeys(item for item in models if item))[:3]

    @staticmethod
    def _compact_visual_evidence(vision: dict[str, Any]) -> dict[str, Any]:
        source_images = vision.get("source_images")
        role_counts: dict[str, int] = {}
        inspected_count = 0
        if isinstance(source_images, list):
            for item in source_images:
                if not isinstance(item, dict):
                    continue
                if item.get("inspection_complete") is True:
                    inspected_count += 1
                role = str(item.get("role") or "unknown")
                role_counts[role] = role_counts.get(role, 0) + 1
        return {
            "product_type": vision.get("product_type"),
            "visible_colors": vision.get("visible_colors"),
            "visible_design_features": vision.get("visible_design_features")
            or vision.get("design_features"),
            "preservation_constraints": vision.get("preservation_constraints"),
            "image_quality_notes": vision.get("image_quality_notes"),
            "inspected_source_image_count": inspected_count,
            "source_image_role_counts": role_counts,
        }

    def reconcile_facts(
        self, facts: ProductFacts, vision: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve structured/visual evidence conflicts without product-specific rules."""

        if self.client is None or not vision:
            return {}
        attributes = [
            {
                "attribute_index": index,
                "id": item.attribute_id,
                "name": item.name,
                "value": item.value,
                "evidence_pointer": item.evidence_pointer,
            }
            for index, item in enumerate(facts.attributes)
        ]
        system = (
            "You are a conservative multimodal evidence reconciler. Return JSON only. "
            "Resolve generic product-appearance conflicts; never rely on product-specific exception lists. "
            "Consistent direct observations from multiple trusted source images take precedence over a conflicting "
            "seller appearance label. Structured identifiers, measurements and non-visual business facts remain "
            "structured facts unless directly disproved by valid evidence."
        )
        prompt = f"""
Reconcile the structured seller facts with the compact observations from all inspected source images.
Do not edit source data. Instead decide which appearance claims may be used in buyer copy and generation.

Return exactly:
- seller_title_decision: publish or machine_only
- attribute_decisions: array of objects with attribute_index, decision (publish, reject, or machine_only),
  canonical_value, reason, and visual_evidence (string array). Include every structured attribute whose
  buyer-facing appearance meaning is confirmed, contradicted, ambiguous, or superseded by source pixels.
- canonical_visual_claims: array of objects with concept, value, confidence (0-1), and evidence (string array).
  Include only directly and consistently visible claims useful to downstream copy or media.
- conflicts: array of concise objects with concept, structured_value, visual_value, resolution, and reason.

Rules:
- Use attribute_index exactly as supplied; never invent an index.
- publish means the structured value is safe for buyer-facing appearance claims.
- reject means trusted pixels directly contradict it.
- machine_only means evidence is ambiguous or the value is unsuitable for buyer-facing appearance claims.
- When direct source pixels conflict with structured appearance text, select the pixels for canonical_value.
- Absence is evidence only when the relevant structure is clearly visible in multiple independent source views.
- Do not infer materials, performance, measurements, brand, care, sizing equivalence, or unseen construction.

Seller title:
{json.dumps(facts.source_title, ensure_ascii=False)}

Indexed structured attributes:
{json.dumps(attributes, ensure_ascii=False)}

Compact source-image evidence:
{json.dumps(self._compact_visual_evidence(vision), ensure_ascii=False)}
""".strip()

        def call(model: str) -> tuple[str, dict[str, Any] | None, str]:
            try:
                return (
                    model,
                    self.client.chat_json(
                        system,
                        prompt,
                        model=model,
                        fallback_model=model,
                    ),
                    "",
                )
            except ApiError as exc:
                return model, None, str(exc)

        results: list[tuple[str, dict[str, Any] | None, str]] = []
        models = self._evaluation_models()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(models), thread_name_prefix="fact-reconcile"
        ) as executor:
            futures = [executor.submit(call, model) for model in models]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        normalized: dict[str, dict[str, Any]] = {}
        for model, payload, error in results:
            if payload is None:
                self.logger.warning("事实裁决模型 %s 不可用: %s", model, error)
                continue
            parsed = self._normalize_reconciliation(payload, len(attributes))
            if parsed:
                normalized[model] = parsed
            else:
                self.logger.warning("事实裁决模型 %s 返回结构无效", model)
        if not normalized:
            return {}
        return self._aggregate_reconciliations(normalized)

    @staticmethod
    def _normalize_reconciliation(
        payload: dict[str, Any], attribute_count: int
    ) -> dict[str, Any]:
        decisions: list[dict[str, Any]] = []
        seen: set[int] = set()
        raw_decisions = payload.get("attribute_decisions")
        if isinstance(raw_decisions, list):
            for item in raw_decisions:
                if not isinstance(item, dict):
                    continue
                index = item.get("attribute_index")
                decision = str(item.get("decision") or "")
                if (
                    not isinstance(index, int)
                    or not 0 <= index < attribute_count
                    or index in seen
                    or decision not in {"publish", "reject", "machine_only"}
                ):
                    continue
                seen.add(index)
                evidence = item.get("visual_evidence")
                decisions.append(
                    {
                        "attribute_index": index,
                        "decision": decision,
                        "canonical_value": str(item.get("canonical_value") or "")[:500],
                        "reason": str(item.get("reason") or "")[:1000],
                        "visual_evidence": [
                            str(value)[:500]
                            for value in evidence
                            if str(value).strip()
                        ][:12]
                        if isinstance(evidence, list)
                        else [],
                    }
                )
        canonical: list[dict[str, Any]] = []
        raw_canonical = payload.get("canonical_visual_claims")
        if isinstance(raw_canonical, list):
            for item in raw_canonical:
                if not isinstance(item, dict):
                    continue
                concept = str(item.get("concept") or "").strip()
                value = str(item.get("value") or "").strip()
                confidence = item.get("confidence")
                if not concept or not value or not isinstance(confidence, (int, float)):
                    continue
                evidence = item.get("evidence")
                canonical.append(
                    {
                        "concept": concept[:200],
                        "value": value[:500],
                        "confidence": max(0.0, min(1.0, float(confidence))),
                        "evidence": [
                            str(part)[:500]
                            for part in evidence
                            if str(part).strip()
                        ][:12]
                        if isinstance(evidence, list)
                        else [],
                    }
                )
        conflicts = [
            {str(key): str(value)[:1000] for key, value in item.items()}
            for item in payload.get("conflicts", [])
            if isinstance(item, dict)
        ][:30] if isinstance(payload.get("conflicts"), list) else []
        title = str(payload.get("seller_title_decision") or "machine_only")
        if title not in {"publish", "machine_only"}:
            title = "machine_only"
        return {
            "seller_title_decision": title,
            "attribute_decisions": decisions,
            "canonical_visual_claims": canonical,
            "conflicts": conflicts,
        }

    @staticmethod
    def _aggregate_reconciliations(
        results: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        all_indices = sorted(
            {
                item["attribute_index"]
                for result in results.values()
                for item in result["attribute_decisions"]
            }
        )
        decisions: list[dict[str, Any]] = []
        for index in all_indices:
            rows = [
                (model, item)
                for model, result in results.items()
                for item in result["attribute_decisions"]
                if item["attribute_index"] == index
            ]
            votes = [item["decision"] for _, item in rows]
            reject_count = votes.count("reject")
            publish_count = votes.count("publish")
            if reject_count and reject_count >= publish_count:
                decision = "reject" if reject_count > publish_count else "machine_only"
            elif "machine_only" in votes or reject_count:
                decision = "machine_only"
            else:
                decision = "publish"
            canonical_values = [
                item["canonical_value"] for _, item in rows if item["canonical_value"]
            ]
            decisions.append(
                {
                    "attribute_index": index,
                    "decision": decision,
                    "canonical_value": canonical_values[0] if canonical_values else "",
                    "reason": " | ".join(
                        f"{model}: {item['reason']}" for model, item in rows if item["reason"]
                    )[:2000],
                    "visual_evidence": list(
                        dict.fromkeys(
                            evidence
                            for _, item in rows
                            for evidence in item["visual_evidence"]
                        )
                    )[:20],
                    "model_votes": {model: item["decision"] for model, item in rows},
                }
            )

        canonical_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for model, result in results.items():
            for item in result["canonical_visual_claims"]:
                key = (
                    re.sub(r"\W+", "", item["concept"].casefold()),
                    re.sub(r"\W+", "", item["value"].casefold()),
                )
                if not all(key):
                    continue
                row = canonical_by_key.setdefault(
                    key,
                    {
                        "concept": item["concept"],
                        "value": item["value"],
                        "confidence_values": [],
                        "evidence": [],
                        "models": [],
                    },
                )
                row["confidence_values"].append(item["confidence"])
                row["evidence"].extend(item["evidence"])
                row["models"].append(model)
        canonical = []
        for row in canonical_by_key.values():
            canonical.append(
                {
                    "concept": row["concept"],
                    "value": row["value"],
                    "confidence": round(
                        sum(row["confidence_values"]) / len(row["confidence_values"]), 3
                    ),
                    "evidence": list(dict.fromkeys(row["evidence"]))[:20],
                    "models": list(dict.fromkeys(row["models"])),
                }
            )
        title_votes = {
            model: result["seller_title_decision"] for model, result in results.items()
        }
        title_decision = (
            "publish"
            if title_votes and all(value == "publish" for value in title_votes.values())
            else "machine_only"
        )
        conflicts = [
            {**item, "model": model}
            for model, result in results.items()
            for item in result["conflicts"]
        ][:50]
        return {
            "version": 1,
            "models": list(results),
            "seller_title_decision": title_decision,
            "seller_title_votes": title_votes,
            "attribute_decisions": decisions,
            "canonical_visual_claims": canonical,
            "conflicts": conflicts,
        }

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
            + self.skills.compile(
                "final-review",
                "delivery-quality",
                "product-grounding",
                "aliexpress-taxonomy",
                "marketplace-materials",
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
- repair_actions: ordered array of all evidence-backed repair targets worth attempting, with:
  tool, target, instruction, reason, dimension, priority (1 highest to 4 lowest)

Only choose tool/target combinations from the catalog. Each instruction must be an actionable,
artifact-specific correction prompt grounded in the evidence. Prefer revising the weakest high-weight
dimension. Do not request cosmetic changes that risk factual identity. Set ready_for_delivery true only
when the weighted score is at least {agent_plan.get('minimum_weighted_score', 90)}, there are no blocker
issues, and A1/A2/A5 have no major issue.
Return a repair action for every reparable issue that has an allowed tool/target; there is no arbitrary
per-round target-count cap. Do not omit a supported target merely because three other targets were listed.
Treat a deterministic or validation-error copy source as a quality degradation: inspect its rendered
shopper preview and request revise_localized_copy when it is generic, process-oriented or misses a
distinctive source-title design detail. Do not reward raw evidence volume. Buyer-facing prose and media
descriptions must use the requested locale and must not contain stray Chinese fragments. Do not treat
an exact source URL, identifier, model code or necessary source label inside the clearly separated
machine appendix as buyer-copy contamination. JSON pointers, canonical/evidence labels and duplicated
audit tables remain defects.
Treat the six-image set review below as independent evidence about semantic duplication and missing
commercial roles. A set-level repair target is higher priority than a cosmetic per-image preference.
It may predate repairs in later rounds, so corroborate it against the current manifest and media.

Complete rubric definitions and weights (use these exact dimension meanings):
{json.dumps(_RUBRIC_DEFINITIONS, ensure_ascii=False)}

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
        minimum_score = float(agent_plan.get("minimum_weighted_score", 90))

        def evaluate_with(model: str) -> tuple[str, AgentEvaluation | None, str]:
            try:
                payload = self.client.chat_json(
                    system,
                    prompt,
                    images=image_urls,
                    videos=video_urls,
                    model=model,
                    fallback_model=model,
                )
                return (
                    model,
                    self._parse_evaluation(
                        payload,
                        round_index=round_index,
                        tools=tools,
                        minimum_weighted_score=minimum_score,
                        evaluator_model=model,
                    ),
                    "",
                )
            except (ApiError, ValueError, TypeError) as exc:
                return model, None, str(exc)

        models = self._evaluation_models()
        results: list[tuple[str, AgentEvaluation | None, str]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(models), thread_name_prefix="delivery-evaluator"
        ) as executor:
            futures = [executor.submit(evaluate_with, model) for model in models]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        evaluations: dict[str, AgentEvaluation] = {}
        for model, evaluation, error in results:
            if evaluation is None:
                self.logger.warning("全局评估模型 %s 不可用: %s", model, error)
            else:
                evaluations[model] = evaluation
        if len(evaluations) < 2:
            self.logger.warning(
                "全局交付评估仅获得 %d 个有效模型结果，至少需要 2 个",
                len(evaluations),
            )
            return None
        return self._aggregate_evaluations(
            evaluations,
            round_index=round_index,
            minimum_weighted_score=minimum_score,
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
            shopper_preview, machine_appendix = _description_language_surfaces(
                text, language
            )
            localized_surface = re.sub(r"https?://[^\s)>]+", "", shopper_preview)
            machine_surface = re.sub(r"https?://[^\s)>]+", "", machine_appendix)
            evidence.append(
                {
                    "language": language,
                    "readable": True,
                    "characters": len(text),
                    "headings": headings,
                    "buyer_chinese_character_count": len(
                        re.findall(r"[\u4e00-\u9fff]", localized_surface)
                    ),
                    "machine_appendix_chinese_character_count": len(
                        re.findall(r"[\u4e00-\u9fff]", machine_surface)
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
        minimum_weighted_score: float = 90,
        evaluator_model: str = "",
    ) -> AgentEvaluation:
        raw_scores = payload.get("dimension_scores")
        scores: dict[str, float] = {}
        if isinstance(raw_scores, dict):
            for dimension in _RUBRIC_WEIGHTS:
                value = raw_scores.get(dimension)
                if isinstance(value, (int, float)):
                    scores[dimension] = max(0.0, min(100.0, float(value)))
        complete_dimension_scores = len(scores) == len(_RUBRIC_WEIGHTS)
        if not complete_dimension_scores:
            missing = [key for key in _RUBRIC_WEIGHTS if key not in scores]
            raise ValueError(
                "evaluation response is missing numeric rubric dimensions: "
                + ", ".join(missing)
            )
        weighted = sum(
            scores[key] * weight / 100 for key, weight in _RUBRIC_WEIGHTS.items()
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
        hard_issue = any(
            str(item.get("severity") or "") == "blocker"
            or (
                str(item.get("severity") or "") == "major"
                and str(item.get("dimension") or "") in {"A1", "A2", "A5"}
            )
            for item in clean_issues
        )
        ready = bool(
            payload.get("ready_for_delivery") is True
            and complete_dimension_scores
            and float(weighted) >= minimum_weighted_score
            and not hard_issue
        )
        return AgentEvaluation(
            round_index=round_index,
            ready_for_delivery=ready,
            weighted_score=max(0.0, min(100.0, float(weighted))),
            dimension_scores=scores,
            summary=str(payload.get("summary") or "")[:3000],
            issues=clean_issues[:20],
            repair_actions=actions,
            evaluator_models=[evaluator_model] if evaluator_model else [],
            model_dimension_scores=(
                {evaluator_model: dict(scores)} if evaluator_model else {}
            ),
            model_weighted_scores=(
                {evaluator_model: max(0.0, min(100.0, float(weighted)))}
                if evaluator_model
                else {}
            ),
        )

    @staticmethod
    def _aggregate_evaluations(
        evaluations: dict[str, AgentEvaluation],
        *,
        round_index: int,
        minimum_weighted_score: float,
    ) -> AgentEvaluation:
        models = list(evaluations)
        scores = {
            dimension: round(
                sum(item.dimension_scores[dimension] for item in evaluations.values())
                / len(evaluations),
                3,
            )
            for dimension in _RUBRIC_WEIGHTS
        }
        weighted = sum(
            scores[dimension] * weight / 100
            for dimension, weight in _RUBRIC_WEIGHTS.items()
        )

        issue_groups: dict[tuple[str, str, str], dict[str, Any]] = {}
        for model, evaluation in evaluations.items():
            for issue in evaluation.issues:
                dimension = str(issue.get("dimension") or "")[:8]
                severity = str(issue.get("severity") or "minor")
                target = str(issue.get("target") or "delivery")[:300]
                key = (dimension, target, severity)
                row = issue_groups.setdefault(
                    key,
                    {
                        "dimension": dimension,
                        "severity": severity,
                        "target": target,
                        "evidence_parts": [],
                        "expected_parts": [],
                        "models": [],
                    },
                )
                evidence = str(issue.get("evidence") or "").strip()
                expected = str(issue.get("expected") or "").strip()
                if evidence:
                    row["evidence_parts"].append(f"{model}: {evidence}")
                if expected:
                    row["expected_parts"].append(f"{model}: {expected}")
                row["models"].append(model)
        issues = [
            {
                "dimension": row["dimension"],
                "severity": row["severity"],
                "target": row["target"],
                "evidence": " | ".join(dict.fromkeys(row["evidence_parts"]))[:3000],
                "expected": " | ".join(dict.fromkeys(row["expected_parts"]))[:2000],
                "models": list(dict.fromkeys(row["models"])),
                "votes": len(set(row["models"])),
            }
            for row in issue_groups.values()
        ]

        action_groups: dict[tuple[str, str], dict[str, Any]] = {}
        for model, evaluation in evaluations.items():
            for action in evaluation.repair_actions:
                key = (action.tool, action.target)
                row = action_groups.setdefault(
                    key,
                    {
                        "instructions": [],
                        "reasons": [],
                        "dimensions": [],
                        "priorities": [],
                        "models": [],
                    },
                )
                row["instructions"].append(f"[{model}] {action.instruction}")
                if action.reason:
                    row["reasons"].append(f"[{model}] {action.reason}")
                if action.dimension:
                    row["dimensions"].append(action.dimension)
                row["priorities"].append(action.priority)
                row["models"].append(model)
        actions: list[AgentAction] = []
        for (tool, target), row in action_groups.items():
            dimension = max(
                row["dimensions"],
                key=lambda item: _RUBRIC_WEIGHTS.get(item, 0),
                default="",
            )
            models_text = ", ".join(dict.fromkeys(row["models"]))
            actions.append(
                AgentAction(
                    tool=tool,
                    target=target,
                    instruction=(
                        f"Combined evaluator feedback ({models_text}):\n- "
                        + "\n- ".join(dict.fromkeys(row["instructions"]))
                    )[:4000],
                    reason=" | ".join(dict.fromkeys(row["reasons"]))[:1000],
                    dimension=dimension,
                    priority=min(row["priorities"] or [4]),
                )
            )
        actions.sort(
            key=lambda item: (
                item.priority,
                -_RUBRIC_WEIGHTS.get(item.dimension, 0),
                item.target,
            )
        )
        hard_issue = any(
            str(item.get("severity") or "") == "blocker"
            or (
                str(item.get("severity") or "") == "major"
                and str(item.get("dimension") or "") in {"A1", "A2", "A5"}
            )
            for item in issues
        )
        ready = bool(weighted >= minimum_weighted_score and not hard_issue)
        summaries = [
            f"{model}: {evaluation.summary}"
            for model, evaluation in evaluations.items()
            if evaluation.summary
        ]
        return AgentEvaluation(
            round_index=round_index,
            ready_for_delivery=ready,
            weighted_score=round(weighted, 3),
            dimension_scores=scores,
            summary="\n".join(summaries)[:6000],
            issues=issues,
            repair_actions=actions,
            evaluator_models=models,
            model_dimension_scores={
                model: dict(evaluation.dimension_scores)
                for model, evaluation in evaluations.items()
            },
            model_weighted_scores={
                model: round(evaluation.weighted_score, 3)
                for model, evaluation in evaluations.items()
            },
        )
