"""LLM-led planning and whole-delivery evaluation with a bounded tool surface."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from .agent_loop import AgentLoopTool, AgentToolOutcome, NativeToolAgentLoop
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
from .planning import validate_creative_plan_payload
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

_INTERNAL_MINIMUM_WEIGHTED_SCORE = 95.0

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
    """Host isolated evaluator, adjudicator, planner, and verifier role calls."""

    def __init__(
        self,
        client: QwenClient | None,
        logger: logging.Logger,
        skills: SkillLibrary | None = None,
    ):
        self.client = client
        self.logger = logger
        self.skills = skills or SkillLibrary()
        self.orchestrator_messages: list[dict[str, Any]] = []
        self.orchestrator_system_prompt = ""

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
            "execution_order": ["hero", "details", "copy", "video"],
            "video_strategy": "stable product-led short presentation",
        }
        if self.client is None or not use_model:
            return default
        self.orchestrator_system_prompt = (
            "You are the top-level autonomous orchestrator for one cross-border commerce delivery. "
            "You own semantic decisions: what evidence to inspect, the creative storyboard and its order, "
            "reference selection, candidate breadth, localization emphasis, and later repair priorities. "
            "Use tools to inspect runtime evidence before deciding. Never invent a product fact, unseen visual "
            "feature, platform identifier, measurement, material, certification, or brand. The host owns only "
            "resource limits, tool authorization, product-identity/content safety, file integrity, and the final "
            "delivery schema. Call submit_delivery_plan only when its entire plan is grounded.\n\n"
            + self.skills.compile(
                "manager",
                "delivery-quality",
                "product-grounding",
                "aliexpress-taxonomy",
                "marketplace-materials",
            )
            + "\n\n"
            + self.skills.compile(
                "creative-plan",
                "product-grounding",
                "marketplace-materials",
            )
        )
        submitted: dict[str, Any] = {}

        def inspect_product(_: dict[str, Any]) -> dict[str, Any]:
            return {"facts": facts.compact_dict()}

        def inspect_taxonomy(_: dict[str, Any]) -> dict[str, Any]:
            return {
                "category": {
                    "id": taxonomy.category.category_id,
                    "name": taxonomy.category.name,
                    "path": taxonomy.category.path,
                },
                "mapped_attributes": [
                    {
                        "attribute": item.name,
                        "source_name": item.source_name,
                        "source_value": item.source_value,
                        "platform_value": item.platform_value,
                    }
                    for item in taxonomy.attributes
                ],
                "missing_required": taxonomy.missing_required,
            }

        def inspect_visual(_: dict[str, Any]) -> dict[str, Any]:
            return self._compact_visual_evidence(vision)

        def inspect_capabilities(_: dict[str, Any]) -> dict[str, Any]:
            return {"repair_tools": tools.catalog(), "candidate_count_range": [1, 4]}

        def submit(arguments: dict[str, Any]) -> AgentToolOutcome:
            order = arguments.get("execution_order")
            if (
                not isinstance(order, list)
                or len(order) != 4
                or set(order) != {"hero", "details", "copy", "video"}
                or order[0] != "hero"
            ):
                return AgentToolOutcome(
                    {
                        "ok": False,
                        "error": "execution_order must contain every stage once and place the reference hero first",
                    }
                )
            creative = arguments.get("creative_plan")
            if not isinstance(creative, dict):
                return AgentToolOutcome({"ok": False, "error": "creative_plan must be an object"})
            validated_plan, validation_error = validate_creative_plan_payload(creative)
            if validated_plan is None:
                return AgentToolOutcome(
                    {
                        "ok": False,
                        "error": validation_error,
                        "correction_required": True,
                    }
                )
            submitted.update(arguments)
            return AgentToolOutcome(
                {
                    "accepted": True,
                    "detail_roles": validated_plan.detail_roles,
                    "candidate_counts": {
                        "main": validated_plan.main_candidate_count,
                        "details": validated_plan.detail_candidate_counts,
                    },
                },
                terminate=True,
            )

        empty_schema = {"type": "object", "properties": {}, "additionalProperties": False}
        reference_roles = {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["hero", "front", "back", "side", "detail", "variant", "lifestyle"],
            },
            "maxItems": 5,
        }
        image_plan_schema = {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "minLength": 40, "maxLength": 5000},
                "candidate_count": {"type": "integer", "minimum": 1, "maximum": 4},
                "reference_roles": reference_roles,
            },
            "required": ["prompt", "candidate_count", "reference_roles"],
            "additionalProperties": False,
        }
        detail_plan_schema = {
            "type": "object",
            "properties": {
                "role": {"type": "string", "minLength": 2, "maxLength": 100},
                **image_plan_schema["properties"],
            },
            "required": ["role", *image_plan_schema["required"]],
            "additionalProperties": False,
        }
        submit_schema = {
            "type": "object",
            "properties": {
                "creative_direction": {"type": "string", "minLength": 10, "maxLength": 3000},
                "localization_priorities": {
                    "type": "object",
                    "properties": {key: {"type": "string", "minLength": 5, "maxLength": 1000} for key in ("en", "ko", "pt")},
                    "required": ["en", "ko", "pt"],
                    "additionalProperties": False,
                },
                "risk_priorities": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(_RUBRIC_WEIGHTS)},
                    "minItems": 1,
                    "uniqueItems": True,
                },
                "execution_order": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["hero", "details", "copy", "video"]},
                    "minItems": 4,
                    "maxItems": 4,
                    "uniqueItems": True,
                },
                "video_strategy": {"type": "string", "minLength": 10, "maxLength": 2000},
                "creative_plan": {
                    "type": "object",
                    "properties": {
                        "visual_theme": {"type": "string", "minLength": 10, "maxLength": 3000},
                        "main": image_plan_schema,
                        "details": {
                            "type": "array",
                            "items": detail_plan_schema,
                            "minItems": 5,
                            "maxItems": 5,
                        },
                        "video": {
                            "type": "object",
                            "properties": {"prompt": {"type": "string", "minLength": 40, "maxLength": 5000}},
                            "required": ["prompt"],
                            "additionalProperties": False,
                        },
                        "market_angles": {
                            "type": "object",
                            "properties": {key: {"type": "string", "minLength": 5, "maxLength": 1000} for key in ("en", "ko", "pt")},
                            "required": ["en", "ko", "pt"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["visual_theme", "main", "details", "video", "market_angles"],
                    "additionalProperties": False,
                },
            },
            "required": ["creative_direction", "localization_priorities", "risk_priorities", "execution_order", "video_strategy", "creative_plan"],
            "additionalProperties": False,
        }
        agent_tools = [
            AgentLoopTool("inspect_product", "Read seller-supplied product facts and source evidence pointers.", empty_schema, inspect_product),
            AgentLoopTool("inspect_taxonomy", "Read resolved platform category and grounded attribute mappings.", empty_schema, inspect_taxonomy),
            AgentLoopTool("inspect_visual_evidence", "Read compact observations from all inspected source images.", empty_schema, inspect_visual),
            AgentLoopTool("inspect_delivery_capabilities", "Read available repair capabilities and resource bounds.", empty_schema, inspect_capabilities),
            AgentLoopTool(
                "submit_delivery_plan",
                "Submit the complete grounded execution and creative plan. This ends planning and must be called alone.",
                submit_schema,
                submit,
                terminal=True,
            ),
        ]
        prompt = (
            "Plan this delivery. Inspect whatever evidence you need, then submit one complete plan. "
            "The output contract requires one square hero, exactly five vertical detail images, one short video, "
            "and localized en-US, ko-KR, and pt-BR copy. Detail jobs and their order are yours to choose. "
            "Also choose the production launch order after placing the reference hero first."
        )
        try:
            loop = NativeToolAgentLoop(
                self.client,
                system_prompt=self.orchestrator_system_prompt,
            )
            deadline = float(getattr(self.client.http, "deadline", time.monotonic() + 600))
            result = loop.run(
                prompt,
                agent_tools,
                max_turns=8,
                deadline=deadline,
                reserve_seconds=120,
            )
            self.orchestrator_messages = result.messages
        except ApiError as exc:
            self.logger.warning("LLM 编排规划不可用，采用保守有界策略: %s", exc)
            return default
        if not submitted:
            self.logger.warning("编排器未在预算内提交有效计划，采用保守计划")
            return default
        return self._normalize_plan(submitted, default)

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
        order = payload.get("execution_order")
        if (
            isinstance(order, list)
            and len(order) == 4
            and set(order) == {"hero", "details", "copy", "video"}
        ):
            result["execution_order"] = list(order)
        sequence = payload.get("visual_sequence")
        if (
            isinstance(sequence, list)
            and len(sequence) == 6
            and all(isinstance(item, str) and item.strip() for item in sequence)
        ):
            result["visual_sequence"] = [item.strip()[:500] for item in sequence]
        creative = payload.get("creative_plan")
        if isinstance(creative, dict):
            result["creative_plan"] = creative
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
        self,
        facts: ProductFacts,
        vision: dict[str, Any],
        *,
        decision_context: str = "",
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
- conflicts: array of objects with conflict_id, source_attribute_index, concept,
  structured_value, visual_value, evidence_refs (string array), affected_surfaces
  (any of buyer_copy, media_generation, marketplace_mapping), surface_resolutions
  (an object whose keys are affected surfaces and whose values are concise proposed
  evidence decisions), and reason. This is a data record for later agents, not a
  request to invoke any particular repair tool.

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

Current reconciliation ledger (empty on the initial pass):
{json.dumps(facts.reconciled_fact_ledger, ensure_ascii=False)}

Top-level orchestrator reconsideration context (may be empty; treat it as a
question to investigate, never as evidence by itself):
{json.dumps(decision_context[:3000], ensure_ascii=False)}
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
        conflicts: list[dict[str, Any]] = []
        raw_conflicts = payload.get("conflicts")
        if isinstance(raw_conflicts, list):
            allowed_surfaces = {
                "buyer_copy",
                "media_generation",
                "marketplace_mapping",
            }
            for position, item in enumerate(raw_conflicts[:30]):
                if not isinstance(item, dict):
                    continue
                source_index = item.get("source_attribute_index")
                if (
                    not isinstance(source_index, int)
                    or not 0 <= source_index < attribute_count
                ):
                    continue
                surfaces = item.get("affected_surfaces")
                normalized_surfaces = (
                    list(
                        dict.fromkeys(
                            str(value)
                            for value in surfaces
                            if str(value) in allowed_surfaces
                        )
                    )
                    if isinstance(surfaces, list)
                    else []
                )
                raw_resolutions = item.get("surface_resolutions")
                resolutions = (
                    {
                        str(key): str(value)[:1000]
                        for key, value in raw_resolutions.items()
                        if str(key) in normalized_surfaces and str(value).strip()
                    }
                    if isinstance(raw_resolutions, dict)
                    else {}
                )
                evidence = item.get("evidence_refs")
                conflicts.append(
                    {
                        "conflict_id": str(
                            item.get("conflict_id")
                            or f"conflict-{source_index}-{position}"
                        )[:200],
                        "source_attribute_index": source_index,
                        "concept": str(item.get("concept") or "")[:300],
                        "structured_value": str(item.get("structured_value") or "")[:500],
                        "visual_value": str(item.get("visual_value") or "")[:500],
                        "evidence_refs": (
                            [
                                str(value)[:500]
                                for value in evidence
                                if str(value).strip()
                            ][:20]
                            if isinstance(evidence, list)
                            else []
                        ),
                        "affected_surfaces": normalized_surfaces,
                        "surface_resolutions": resolutions,
                        "reason": str(item.get("reason") or "")[:1000],
                    }
                )
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
        artifact_fingerprint: str = "",
    ) -> AgentEvaluation | None:
        """Run independent evidence finders, then adjudicate their disagreements.

        Evaluators never choose tools and their scalar scores are ignored.  Code
        derives scores from accepted findings after a separate adjudication role.
        """
        if self.client is None:
            return None
        manifest, image_urls, video_urls = self._artifact_evidence(
            facts, assets, work_dir
        )
        copy_artifacts = self._copy_artifact_evidence(work_dir)
        system = (
            "You are an independent multimodal defect finder, not a scorer, producer, repair planner, "
            "or final decision maker. Return JSON only. Report only defects supported by concrete supplied "
            "artifact evidence. Never choose tools and never decide whether your own finding was repaired.\n\n"
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
directly reviewable delivery image URLs follow. The video input, when present, is the actual generated
video URL. A provenance_source_url is never the final artifact and must not be used to claim that the
final artifact contains the source image's text, people, background, layout, or other pixels.
Local deterministic artifacts cannot be uploaded to the model. Judge them only from final_artifact
physical/hash/rendered-text evidence in the manifest. If that evidence cannot establish a semantic
defect, do not invent one and do not request a repair for it.

Return exactly these keys:
- summary: concise evidence-grounded diagnosis
- findings: array of objects with dimension, criterion, severity (blocker/major/minor),
  target, evidence, and expected

Do not return scores, readiness, repair actions, tools, or speculative improvements. An empty findings
array means that you found no evidence-backed defect.
Treat a deterministic or validation-error copy source as a quality degradation: inspect its rendered
shopper preview and report the exact localized-copy target when it is generic, process-oriented or misses a
distinctive source-title design detail. Do not reward raw evidence volume. Buyer-facing prose and media
descriptions must use the requested locale and must not contain stray Chinese fragments. Do not treat
an exact source URL, identifier, model code or necessary source label inside the clearly separated
machine appendix as buyer-copy contamination. JSON pointers, canonical/evidence labels and duplicated
audit tables remain defects.
Treat the six-image set review below as independent evidence about semantic duplication and missing
commercial roles. A set-level defect is more important than a cosmetic per-image preference.
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

Allowed artifact target names (use these exact names when applicable):
{json.dumps(sorted({target for item in tools.catalog() for target in item.get('allowed_targets', [])}), ensure_ascii=False)}
""".strip()
        minimum_score = max(
            _INTERNAL_MINIMUM_WEIGHTED_SCORE,
            float(
                agent_plan.get(
                    "minimum_weighted_score", _INTERNAL_MINIMUM_WEIGHTED_SCORE
                )
            ),
        )

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
                    self._parse_evaluator_report(
                        payload,
                        round_index=round_index,
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
        return self._adjudicate_evaluations(
            evaluations,
            round_index=round_index,
            minimum_weighted_score=minimum_score,
            artifact_fingerprint=artifact_fingerprint,
            adjudication_context={
                "verified_facts": facts.compact_dict(),
                "taxonomy": {
                    "category_id": taxonomy.category.category_id,
                    "category": taxonomy.category.name,
                    "attributes": [
                        {
                            "id": item.attr_id,
                            "value_id": item.value_id,
                            "value": item.platform_value,
                        }
                        for item in taxonomy.attributes
                    ],
                },
                "localized_copy": copy_artifacts,
                "artifact_manifest": manifest,
                "visual_set_review": visual_set_review,
            },
            image_urls=image_urls,
            video_urls=video_urls,
        )

    def plan_repairs(
        self,
        evaluation: AgentEvaluation,
        *,
        tools: BoundedToolRegistry,
        previous_attempts: list[dict[str, Any]] | None = None,
    ) -> list[AgentAction]:
        """Translate adjudicated defects into bounded tool calls."""

        findings = [item for item in evaluation.issues if isinstance(item, dict)]
        if not findings:
            return []
        catalog = tools.catalog()
        payload: dict[str, Any] = {}
        if self.client is not None:
            system = (
                "You are a repair planner. You did not evaluate the delivery and you do not execute or "
                "verify repairs. Return JSON only. Choose the smallest safe action for adjudicated defects."
            )
            prompt = f"""
Return exactly one key, repair_actions, containing an ordered array. Every action must contain
defect_id, tool, target, instruction, acceptance_criteria, reason, dimension, and priority (1-4).
Use only the supplied defect IDs and exact tool/target pairs. Do not retry an action whose previous
observation says it cannot change the artifact unless you materially change its instruction.

Adjudicated findings:
{json.dumps(findings, ensure_ascii=False)}

Available tools:
{json.dumps(catalog, ensure_ascii=False)}

Previous observations:
{json.dumps(previous_attempts or [], ensure_ascii=False)}
""".strip()
            try:
                planner_model = str(
                    getattr(self.client.config, "chat_model", "")
                    or self.client.config.review_model
                )
                planner_fallback = str(
                    getattr(self.client.config, "chat_fallback_model", "")
                    or self.client.config.review_fallback_model
                )
                payload = self.client.chat_json(
                    system,
                    prompt,
                    model=planner_model,
                    fallback_model=planner_fallback,
                )
            except (ApiError, ValueError, TypeError) as exc:
                self.logger.warning("返修规划器不可用，使用确定性目标映射: %s", exc)

        by_defect = {str(item.get("defect_id") or ""): item for item in findings}
        actions: list[AgentAction] = []
        seen: set[tuple[str, str, str]] = set()
        rows = payload.get("repair_actions") if isinstance(payload, dict) else None
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                defect_id = str(row.get("defect_id") or "").strip()
                tool = str(row.get("tool") or "").strip()
                target = str(row.get("target") or "").strip()
                instruction = str(row.get("instruction") or "").strip()
                finding = by_defect.get(defect_id)
                key = (defect_id, tool, target)
                if (
                    finding is None
                    or not instruction
                    or not tools.accepts(tool, target)
                    or key in seen
                ):
                    continue
                seen.add(key)
                priority = row.get("priority")
                actions.append(
                    AgentAction(
                        defect_id=defect_id,
                        tool=tool,
                        target=target,
                        instruction=instruction[:4000],
                        acceptance_criteria=str(row.get("acceptance_criteria") or finding.get("expected") or "")[:2000],
                        reason=str(row.get("reason") or finding.get("evidence") or "")[:1000],
                        dimension=str(finding.get("dimension") or "")[:8],
                        priority=priority if isinstance(priority, int) else 4,
                        supporting_models=list(finding.get("models") or []),
                        votes=int(finding.get("votes") or 0),
                        execution_tier="planned",
                    )
                )
        if not actions:
            actions = self._fallback_repair_plan(findings, tools)
        actions.sort(key=lambda item: (item.priority, -_RUBRIC_WEIGHTS.get(item.dimension, 0), item.target))
        return actions

    @staticmethod
    def _fallback_repair_plan(
        findings: list[dict[str, Any]], tools: BoundedToolRegistry
    ) -> list[AgentAction]:
        target_tools = {
            "main_image.jpeg": "regenerate_main_image",
            "product_video.mp4": "regenerate_video",
        }
        actions: list[AgentAction] = []
        for finding in findings:
            target = str(finding.get("target") or "")
            if target.startswith("detail_image_") and target.endswith(".jpeg"):
                tool = "regenerate_detail_image"
            elif target.startswith("product_description_") and target.endswith(".md"):
                tool = "revise_localized_copy"
            else:
                tool = target_tools.get(target, "")
            if not tool or not tools.accepts(tool, target):
                continue
            severity = str(finding.get("severity") or "minor")
            actions.append(
                AgentAction(
                    defect_id=str(finding.get("defect_id") or ""),
                    tool=tool,
                    target=target,
                    instruction=(
                        f"Correct this adjudicated defect: {finding.get('evidence', '')}. "
                        f"Required result: {finding.get('expected', '')}. Preserve unrelated verified facts."
                    )[:4000],
                    acceptance_criteria=str(finding.get("expected") or "")[:2000],
                    reason=str(finding.get("evidence") or "")[:1000],
                    dimension=str(finding.get("dimension") or "")[:8],
                    priority={"blocker": 1, "major": 2, "minor": 3}.get(severity, 4),
                    supporting_models=list(finding.get("models") or []),
                    votes=int(finding.get("votes") or 0),
                    execution_tier="planned-fallback",
                )
            )
        return actions

    def verify_repair_outcome(
        self,
        actions: list[AgentAction],
        *,
        facts: ProductFacts,
        taxonomy: TaxonomyResult,
        assets: list[AssetResult],
        visual_set_review: dict[str, Any],
        work_dir: Path,
        before_hashes: dict[str, str],
        after_hashes: dict[str, str],
    ) -> dict[str, Any]:
        """Verify only targeted defects and critical regressions; never rescore."""

        changed = [
            action for action in actions
            if before_hashes.get(action.target) != after_hashes.get(action.target)
        ]
        if not changed:
            return {"accepted": False, "status": "no_change", "fixed_defect_ids": []}
        if self.client is None:
            return {"accepted": True, "status": "local-verification-only", "fixed_defect_ids": [item.defect_id for item in changed]}
        manifest, image_urls, video_urls = self._artifact_evidence(facts, assets, work_dir)
        system = (
            "You are an independent repair outcome verifier. You did not find the defects, plan the repair, "
            "or execute it. Return JSON only. Check only whether the named defects are now fixed and whether "
            "the changed artifacts introduced a blocker or major A1/A2/A5 regression. Never assign scores."
        )
        prompt = f"""
Return exactly: accepted (boolean), fixed_defect_ids (array), regressions (array of objects with
dimension, severity, target, evidence), and evidence (concise string).

Repair intents and acceptance criteria:
{json.dumps([{"defect_id": item.defect_id, "target": item.target, "dimension": item.dimension, "instruction": item.instruction, "acceptance_criteria": item.acceptance_criteria} for item in changed], ensure_ascii=False)}

Before/after content hashes:
{json.dumps({item.target: {"before": before_hashes.get(item.target), "after": after_hashes.get(item.target)} for item in changed}, ensure_ascii=False)}

Current artifact manifest:
{json.dumps(manifest, ensure_ascii=False)}

Current rendered copy evidence:
{json.dumps(self._copy_artifact_evidence(work_dir), ensure_ascii=False)}

Verified facts and reconciled ledger:
{json.dumps(facts.compact_dict(), ensure_ascii=False)}

Resolved category and attributes:
{json.dumps({"category_id": taxonomy.category.category_id, "category": taxonomy.category.name, "attributes": [{"id": item.attr_id, "value_id": item.value_id, "value": item.platform_value} for item in taxonomy.attributes]}, ensure_ascii=False)}

Current six-image set review:
{json.dumps(visual_set_review, ensure_ascii=False)}
""".strip()
        try:
            planner_model = str(
                getattr(self.client.config, "chat_model", "")
                or self.client.config.review_model
            )
            verifier_model = str(
                self.client.config.review_fallback_model
                if self.client.config.review_fallback_model != planner_model
                else self.client.config.review_model
            )
            payload = self.client.chat_json(
                system,
                prompt,
                images=image_urls,
                videos=video_urls,
                model=verifier_model,
                fallback_model=verifier_model,
            )
        except (ApiError, ValueError, TypeError) as exc:
            return {
                "accepted": False,
                "status": "verifier-unavailable",
                "evidence": str(exc)[:1000],
            }
        regressions = payload.get("regressions")
        clean_regressions = [item for item in regressions if isinstance(item, dict)] if isinstance(regressions, list) else []
        critical_regression = any(
            str(item.get("severity") or "") in {"blocker", "major"}
            and str(item.get("dimension") or "") in {"A1", "A2", "A5"}
            for item in clean_regressions
        )
        fixed = payload.get("fixed_defect_ids")
        fixed_ids = {str(item) for item in fixed} if isinstance(fixed, list) else set()
        expected_ids = {item.defect_id for item in changed}
        accepted = payload.get("accepted") is True and expected_ids.issubset(fixed_ids) and not critical_regression
        return {
            "accepted": accepted,
            "status": "verified" if accepted else "postcondition-failed",
            "fixed_defect_ids": sorted(fixed_ids),
            "regressions": clean_regressions,
            "evidence": str(payload.get("evidence") or "")[:2000],
        }

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
            if asset.name.endswith(".jpeg")
            and asset.generated
            and asset.source_url
        ]
        image_urls = list(dict.fromkeys(source_references + delivery_urls))
        video_urls = [
            asset.source_url
            for asset in assets
            if asset.name == "product_video.mp4" and asset.generated and asset.source_url
        ][:1]
        for asset in assets:
            path = work_dir / asset.name
            delivery_visual_url = (
                asset.source_url
                if asset.generated and asset.name.endswith(".jpeg")
                else ""
            )
            item: dict[str, Any] = {
                "name": asset.name,
                "generated": asset.generated,
                "model": asset.model,
                "evidence_mode": (
                    "remote-final-output"
                    if delivery_visual_url
                    else "local-final-inspection"
                ),
                "delivery_visual_url": delivery_visual_url,
                "provenance_source_url": asset.source_url,
                "description": asset.description,
                "fallback_reason": asset.fallback_reason,
                "visual_input_index": (
                    image_urls.index(delivery_visual_url)
                    if delivery_visual_url in image_urls
                    else None
                ),
            }
            try:
                item["bytes"] = path.stat().st_size
                item["final_artifact_sha256"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
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
                    if asset.model == "deterministic-size-chart":
                        rendered_rows = [
                            {
                                "size": str(row.size_label),
                                "bust": f"{row.bust_cm} cm" if row.bust_cm else "—",
                                "length": (
                                    f"{row.length_cm} cm" if row.length_cm else "—"
                                ),
                                "weight": (
                                    f"{row.weight_kg} / {row.weight_lb}"
                                    if row.weight_kg and row.weight_lb
                                    else row.weight_kg or "—"
                                ),
                            }
                            for row in facts.size_chart_rows
                        ]
                        item["deterministic_render"] = {
                            "text_language": "en",
                            "contains_cjk": False,
                            "title": "SIZE GUIDE",
                            "subtitle": "GARMENT MEASUREMENTS • SELLER-PROVIDED DATA",
                            "columns": ["SIZE", "BUST", "LENGTH", "WEIGHT GUIDE"],
                            "rows": rendered_rows,
                            "notes": [
                                "Measurements are transcribed from the seller's source size chart.",
                                "Check garment measurements; regional size equivalence is not assumed.",
                            ],
                        }
                elif asset.name == "product_video.mp4":
                    item["physical"] = inspect_video(path)
            except (OSError, MediaError) as exc:
                item["physical_error"] = str(exc)
            manifest.append(item)
        return manifest, image_urls, video_urls

    @staticmethod
    def _canonical_finding(item: dict[str, Any], evaluator_model: str) -> dict[str, Any] | None:
        dimension = str(item.get("dimension") or "").strip().upper()
        severity = str(item.get("severity") or "minor").strip().lower()
        target = Path(str(item.get("target") or "delivery").strip()).name or "delivery"
        criterion = re.sub(
            r"[^a-z0-9_-]+",
            "-",
            str(item.get("criterion") or "general").strip().casefold(),
        ).strip("-") or "general"
        evidence = str(item.get("evidence") or "").strip()
        if dimension not in _RUBRIC_WEIGHTS or severity not in {"blocker", "major", "minor"} or not evidence:
            return None
        defect_id = f"{dimension}:{target}:{criterion}"
        return {
            "defect_id": defect_id,
            "dimension": dimension,
            "criterion": criterion,
            "severity": severity,
            "target": target[:300],
            "evidence": evidence[:3000],
            "expected": str(item.get("expected") or "").strip()[:2000],
            "models": [evaluator_model] if evaluator_model else [],
            "votes": 1 if evaluator_model else 0,
        }

    @staticmethod
    def _scores_from_findings(findings: list[dict[str, Any]]) -> tuple[dict[str, float], float]:
        penalties = {"blocker": 100.0, "major": 30.0, "minor": 8.0}
        scores = {dimension: 100.0 for dimension in _RUBRIC_WEIGHTS}
        for item in findings:
            dimension = str(item.get("dimension") or "")
            if dimension in scores:
                scores[dimension] = max(
                    0.0,
                    scores[dimension] - penalties.get(str(item.get("severity") or "minor"), 8.0),
                )
        weighted = sum(scores[key] * weight / 100 for key, weight in _RUBRIC_WEIGHTS.items())
        return scores, round(weighted, 3)

    @classmethod
    def _parse_evaluator_report(
        cls,
        payload: dict[str, Any],
        *,
        round_index: int,
        evaluator_model: str = "",
    ) -> AgentEvaluation:
        raw_findings = payload.get("findings")
        if raw_findings is None:
            raw_findings = payload.get("issues")
        if not isinstance(raw_findings, list):
            raise ValueError("evaluator response is missing findings array")
        findings = [
            clean
            for item in raw_findings
            if isinstance(item, dict)
            for clean in [cls._canonical_finding(item, evaluator_model)]
            if clean is not None
        ][:30]
        scores, weighted = cls._scores_from_findings(findings)
        return AgentEvaluation(
            round_index=round_index,
            ready_for_delivery=False,
            weighted_score=weighted,
            dimension_scores=scores,
            summary=str(payload.get("summary") or "")[:3000],
            issues=findings,
            repair_actions=[],
            evaluator_models=[evaluator_model] if evaluator_model else [],
            model_dimension_scores={evaluator_model: dict(scores)} if evaluator_model else {},
            model_weighted_scores={evaluator_model: weighted} if evaluator_model else {},
        )

    def _adjudicate_evaluations(
        self,
        evaluations: dict[str, AgentEvaluation],
        *,
        round_index: int,
        minimum_weighted_score: float,
        artifact_fingerprint: str = "",
        adjudication_context: dict[str, Any] | None = None,
        image_urls: list[str] | None = None,
        video_urls: list[str] | None = None,
    ) -> AgentEvaluation:
        models = sorted(evaluations)
        grouped: dict[str, dict[str, Any]] = {}
        severity_rank = {"minor": 0, "major": 1, "blocker": 2}
        for model, report in evaluations.items():
            for finding in report.issues:
                defect_id = str(finding.get("defect_id") or "")
                row = grouped.setdefault(
                    defect_id,
                    {
                        **finding,
                        "evidence_parts": [],
                        "expected_parts": [],
                        "models": [],
                    },
                )
                if severity_rank.get(str(finding.get("severity")), 0) > severity_rank.get(str(row.get("severity")), 0):
                    row["severity"] = finding.get("severity")
                row["evidence_parts"].append(f"{model}: {finding.get('evidence', '')}")
                expected = str(finding.get("expected") or "").strip()
                if expected:
                    row["expected_parts"].append(f"{model}: {expected}")
                row["models"].append(model)

        consensus_ids = {
            defect_id for defect_id, row in grouped.items() if len(set(row["models"])) >= 2
        }
        disputed = [
            {
                "defect_id": defect_id,
                "dimension": row["dimension"],
                "criterion": row["criterion"],
                "severity": row["severity"],
                "target": row["target"],
                "evidence": " | ".join(dict.fromkeys(row["evidence_parts"]))[:3000],
                "expected": " | ".join(dict.fromkeys(row["expected_parts"]))[:2000],
                "models": sorted(set(row["models"])),
            }
            for defect_id, row in grouped.items()
            if defect_id not in consensus_ids
        ]
        decisions: dict[str, dict[str, Any]] = {}
        adjudicator_status = "not-needed"
        if disputed and self.client is not None:
            system = (
                "You are an evidence adjudicator, not an evaluator, repair planner, or scorer. Return JSON only. "
                "For each disputed finding decide whether the quoted artifact evidence supports that exact defect."
            )
            prompt = f"""
Return exactly one key, decisions, containing one object per disputed defect with defect_id,
supported (boolean), and reason. Do not introduce new defects, tools, actions, or scores.

Rubric:
{json.dumps(_RUBRIC_DEFINITIONS, ensure_ascii=False)}

Disputed findings:
{json.dumps(disputed, ensure_ascii=False)}

Independent artifact evidence:
{json.dumps(adjudication_context or {}, ensure_ascii=False)}
""".strip()
            try:
                payload = self.client.chat_json(
                    system,
                    prompt,
                    images=image_urls or [],
                    videos=video_urls or [],
                    model=self.client.config.review_model,
                    fallback_model=self.client.config.review_fallback_model,
                )
                rows = payload.get("decisions")
                if isinstance(rows, list):
                    decisions = {
                        str(item.get("defect_id") or ""): item
                        for item in rows
                        if isinstance(item, dict)
                    }
                adjudicator_status = "completed"
            except (ApiError, ValueError, TypeError) as exc:
                adjudicator_status = f"unavailable: {str(exc)[:300]}"

        accepted_ids = set(consensus_ids)
        for item in disputed:
            defect_id = item["defect_id"]
            decision = decisions.get(defect_id, {})
            if decision.get("supported") is True:
                accepted_ids.add(defect_id)
            elif not decisions and (
                item["severity"] == "blocker"
                or (item["severity"] == "major" and item["dimension"] in {"A1", "A2", "A5"})
            ):
                # A missing adjudicator must not silently clear a potentially unsafe delivery.
                accepted_ids.add(defect_id)

        issues: list[dict[str, Any]] = []
        for defect_id in sorted(accepted_ids):
            row = grouped[defect_id]
            issues.append(
                {
                    "defect_id": defect_id,
                    "dimension": row["dimension"],
                    "criterion": row["criterion"],
                    "severity": row["severity"],
                    "target": row["target"],
                    "evidence": " | ".join(dict.fromkeys(row["evidence_parts"]))[:3000],
                    "expected": " | ".join(dict.fromkeys(row["expected_parts"]))[:2000],
                    "models": sorted(set(row["models"])),
                    "votes": len(set(row["models"])),
                    "adjudication": (
                        "consensus" if defect_id in consensus_ids else str(decisions.get(defect_id, {}).get("reason") or "conservative-unresolved")[:1000]
                    ),
                }
            )
        scores, weighted = self._scores_from_findings(issues)
        hard_issue = any(
            item["severity"] == "blocker"
            or (item["severity"] == "major" and item["dimension"] in {"A1", "A2", "A5"})
            for item in issues
        )
        summaries = [
            f"{model}: {evaluation.summary}"
            for model, evaluation in evaluations.items()
            if evaluation.summary
        ]
        return AgentEvaluation(
            round_index=round_index,
            ready_for_delivery=bool(weighted >= minimum_weighted_score and not hard_issue),
            weighted_score=weighted,
            dimension_scores=scores,
            summary="\n".join(summaries)[:6000],
            issues=issues,
            repair_actions=[],
            evaluator_models=models,
            model_dimension_scores={model: dict(report.dimension_scores) for model, report in evaluations.items()},
            model_weighted_scores={model: report.weighted_score for model, report in evaluations.items()},
            artifact_fingerprint=artifact_fingerprint,
            disagreement=bool(disputed),
            adjudication={
                "status": adjudicator_status,
                "consensus_defect_ids": sorted(consensus_ids),
                "disputed_defect_ids": sorted(item["defect_id"] for item in disputed),
                "accepted_defect_ids": sorted(accepted_ids),
                "decisions": decisions,
            },
        )
