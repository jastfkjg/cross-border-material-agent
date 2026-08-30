"""Evidence responsibilities for the delivery pipeline."""

from __future__ import annotations

import concurrent.futures
import json
import re
from typing import Any

from ..agent_tools import ToolExecution
from ..api import ApiError
from ..claims import build_claim_ledger
from ..compliance import normalize_source_image_observations
from ..decision_state import (
    DependencyState,
    assess_evidence_sufficiency,
    build_canonical_product_state,
    build_expected_delivery_spec,
)
from ..models import (
    ProductFacts,
    RunState,
    SizeChartRow,
    TaxonomyResult,
)
from ..qa import EXPECTED_FILES
from .common import (
    unique as _unique,
)


class EvidencePipelineMixin:
    def _repair_fact_ledger(
        self,
        instruction: str,
        *,
        facts: ProductFacts,
        taxonomy: TaxonomyResult,
        vision: dict[str, Any],
        state: RunState,
    ) -> ToolExecution:
        before = json.dumps(
            facts.reconciled_fact_ledger,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        revised = self.agent.reconcile_facts(
            facts,
            vision,
            decision_context=instruction,
        )
        if not revised:
            return ToolExecution(
                "failed", "fact reconcilers returned no grounded ledger"
            )
        after = json.dumps(revised, ensure_ascii=False, sort_keys=True, default=str)
        if after == before:
            return ToolExecution(
                "completed",
                "fact ledger was reconsidered but the evidence decision did not change",
                {"changed": False},
            )
        facts.reconciled_fact_ledger = revised
        state.claim_ledger = build_claim_ledger(facts, taxonomy, vision)
        canonical_state = build_canonical_product_state(facts, revised)
        evidence_sufficiency = assess_evidence_sufficiency(vision, canonical_state)
        expected_delivery_spec = build_expected_delivery_spec(
            canonical=canonical_state,
            taxonomy=taxonomy,
            claim_ledger=state.claim_ledger,
            evidence=evidence_sufficiency,
            required_files=state.expected_delivery_spec.get(
                "required_files", EXPECTED_FILES
            ),
            preserve_mapping_sources=state.expected_delivery_spec.get(
                "required_mapping_sources", []
            ),
        )
        state.canonical_product_state = canonical_state.to_dict()
        state.evidence_sufficiency = evidence_sufficiency.to_dict()
        state.expected_delivery_spec = expected_delivery_spec.to_dict()
        dependency_state = DependencyState.from_dict(state.dependency_state)
        dependency_state.record("canonical", canonical_state.to_dict())
        dependency_state.record(
            "delivery_spec",
            expected_delivery_spec.to_dict(),
            canonical=canonical_state.version,
            taxonomy=expected_delivery_spec.taxonomy_version,
        )
        dependency_state.invalidate(
            "canonical", ["review"], "fact reconciliation changed"
        )
        state.dependency_state = dependency_state.to_dict()
        return ToolExecution(
            "completed",
            "canonical fact state, claim projection, and expected delivery specification were rebuilt",
            {
                "changed": True,
                "conflict_count": len(revised.get("conflicts", [])),
                "decision_count": len(revised.get("attribute_decisions", [])),
            },
        )

    def _analyze_source_images(self, facts: ProductFacts) -> dict[str, Any]:
        if self.client is None:
            self.warnings.append(
                "显式离线模式：跳过源图片视觉理解"
                if self.offline
                else "模型配置不可用，跳过源图片视觉理解"
            )
            return {}
        self._ensure_time(10 * 60)
        # The visual endpoint has a per-call image limit.  Preserve every distinct
        # source URL and batch it instead of uniformly sampling a handful; size
        # charts and construction details commonly sit near the end of descriptions.
        urls = _unique(
            facts.product_image_urls
            + facts.sku_image_urls
            + facts.description_image_urls
        )
        if not urls:
            return {}
        batches = [urls[index : index + 12] for index in range(0, len(urls), 12)]
        facts_json = json.dumps(facts.compact_dict(), ensure_ascii=False)
        skill_instructions = self.skills.compile(
            "source-vision",
            "product-grounding",
            "marketplace-materials",
        )

        def inspect_batch(
            batch_index: int, batch_urls: list[str]
        ) -> tuple[int, dict[str, Any], str]:
            try:
                payload = self.client.analyze_product_images(
                    facts_json,
                    batch_urls,
                    skill_instructions=skill_instructions,
                )
                return batch_index, payload, ""
            except (ApiError, ValueError) as exc:
                return batch_index, {}, str(exc)

        completed: list[tuple[list[str], dict[str, Any]]] = [
            (batch, {}) for batch in batches
        ]
        errors: list[str] = []
        worker_count = min(3, len(batches))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="source-vision",
        ) as executor:
            futures = [
                executor.submit(inspect_batch, index, batch)
                for index, batch in enumerate(batches)
            ]
            for future in concurrent.futures.as_completed(futures):
                batch_index, payload, error = future.result()
                completed[batch_index] = (batches[batch_index], payload)
                if error:
                    errors.append(f"batch {batch_index + 1}: {error}")

        result = _merge_source_vision_batches(completed, urls)
        source_images = result["source_images"]
        self._source_image_observations = {
            str(item["url"]): item for item in source_images
        }
        inspected = sum(
            item.get("inspection_complete") is True for item in source_images
        )
        rejected = sum(
            item.get("inspection_complete") is True
            and not item.get("safe_for_generation_reference", False)
            for item in source_images
        )
        self.logger.info(
            "完成 %d/%d 张源图片的分批视觉理解（%d 批）",
            inspected,
            len(urls),
            len(batches),
        )
        if errors:
            self.logger.warning("%d 个源图扫描批次失败，其余批次继续使用", len(errors))
            self.warnings.append(
                f"源图片分批扫描有 {len(errors)}/{len(batches)} 批失败: "
                + "; ".join(errors[:3])[:500]
            )
        if rejected:
            self.logger.info("源图片风控筛出 %d 张不宜作为生成参考图", rejected)
        third_party_count = sum(
            item.get("has_third_party_brand") is True for item in source_images
        )
        if third_party_count:
            self.warnings.append(
                f"{third_party_count} 张源图疑似含第三方品牌或角色；不可直接发布，"
                "仅可作为需清理的商品身份参考"
            )
        global_risks = result.get("prohibited_or_risky_visuals")
        if isinstance(global_risks, list) and global_risks:
            risk_summary = "; ".join(
                str(item).strip() for item in global_risks[:3] if str(item).strip()
            )
            if risk_summary:
                self.warnings.append(f"源图视觉风险需人工复核: {risk_summary[:500]}")
        return result

    def _apply_size_chart_observations(
        self, facts: ProductFacts, vision: dict[str, Any]
    ) -> None:
        """Promote only clearly structured, SKU-aligned visual measurements to facts."""

        raw_rows = vision.get("size_chart_rows") if isinstance(vision, dict) else None
        source_images = (
            vision.get("source_images") if isinstance(vision, dict) else None
        )
        if not isinstance(raw_rows, list) or not isinstance(source_images, list):
            return

        def size_code(value: Any) -> str:
            raw = re.split(r"[\(（\[【]", str(value or "").strip(), maxsplit=1)[0]
            compact = re.sub(r"[^A-Za-z0-9]+", "", raw).upper()
            repeated_x = re.fullmatch(r"(X+)L", compact)
            if repeated_x and len(repeated_x.group(1)) >= 2:
                return f"{len(repeated_x.group(1))}XL"
            return compact

        known_codes: list[str] = []
        for sku in facts.skus:
            for item in sku.attributes:
                code = size_code(item.value)
                if code and len(code) <= 24 and code not in known_codes:
                    known_codes.append(code)
        image_by_index = {
            item.get("index"): item
            for item in source_images
            if isinstance(item, dict) and isinstance(item.get("index"), int)
        }

        def measurement(value: Any) -> str:
            match = re.fullmatch(
                r"\s*(\d{1,4}(?:\.\d+)?)\s*(?:cm)?\s*", str(value or ""), re.I
            )
            if not match:
                return ""
            numeric = float(match.group(1))
            return match.group(1) if 0 < numeric <= 1000 else ""

        conversions: dict[str, tuple[str, str]] = {}
        for item in facts.size_conversions:
            code = size_code(item.source_label)
            if code:
                conversions[code] = (item.kilograms, item.pounds)

        candidate_rows = [
            raw
            for raw in raw_rows
            if isinstance(raw, dict) and size_code(raw.get("size_label"))
        ]
        matched_codes = {
            size_code(raw.get("size_label"))
            for raw in candidate_rows
            if size_code(raw.get("size_label")) in known_codes
        }
        require_sku_alignment = len(matched_codes) >= 2

        rows: list[SizeChartRow] = []
        seen: set[str] = set()
        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            code = size_code(raw.get("size_label"))
            bust = measurement(raw.get("bust_cm"))
            length = measurement(raw.get("length_cm"))
            source_index = raw.get("source_image_index")
            source_item = image_by_index.get(source_index)
            if (
                not code
                or (require_sku_alignment and code not in known_codes)
                or code in seen
                or not source_item
                or str(source_item.get("role") or "") != "size_chart"
                or not (bust or length)
            ):
                continue
            kilograms, pounds = conversions.get(code, ("", ""))
            rows.append(
                SizeChartRow(
                    size_label=code,
                    bust_cm=bust,
                    length_cm=length,
                    weight_kg=kilograms,
                    weight_lb=pounds,
                    evidence_pointer=f"source-image:{source_index}",
                )
            )
            if not self._size_chart_source_url:
                self._size_chart_source_url = str(source_item.get("url") or "")
            seen.add(code)
        if require_sku_alignment:
            rows.sort(key=lambda item: known_codes.index(item.size_label))
        if len(rows) < 2:
            return
        facts.size_chart_rows = rows
        self.logger.info("从源详情图提取并核验 %d 行尺码表", len(rows))

    def _ordered_source_urls(
        self,
        facts: ProductFacts,
        vision: dict[str, Any],
        *,
        preferred_roles: list[str] | tuple[str, ...] = (),
        preferred_indexes: list[int] | tuple[int, ...] = (),
    ) -> list[str]:
        ordered = _unique(
            facts.product_image_urls
            + facts.sku_image_urls
            + facts.description_image_urls
        )
        best_url = ""
        source_images = (
            vision.get("source_images") if isinstance(vision, dict) else None
        )
        best = vision.get("best_hero_image_index") if isinstance(vision, dict) else None
        if isinstance(source_images, list) and isinstance(best, int):
            best_item = next(
                (
                    item
                    for item in source_images
                    if isinstance(item, dict) and item.get("index") == best
                ),
                None,
            )
            if isinstance(best_item, dict):
                best_url = str(best_item.get("url") or "")
        if best_url in ordered:
            ordered.insert(0, ordered.pop(ordered.index(best_url)))
        explicit = [
            str(item.get("url") or "")
            for item in (source_images if isinstance(source_images, list) else [])
            if isinstance(item, dict)
            and item.get("index") in preferred_indexes
            and item.get("safe_for_generation_reference") is True
        ]
        ranked = self._source_urls_for_use(
            ordered,
            use="reference",
            preferred_roles=tuple(preferred_roles)
            or ("hero", "front", "variant", "detail"),
        )
        if explicit:
            ranked = _unique(explicit + ranked)
        product_roles = {
            "hero",
            "front",
            "back",
            "side",
            "detail",
            "variant",
            "lifestyle",
        }
        inspected_product = [
            url
            for url in ranked
            if self._source_image_observations.get(url, {}).get("role") in product_roles
        ]
        return inspected_product or [
            url
            for url in ranked
            if url in facts.product_image_urls or url in facts.sku_image_urls
        ]

    def _source_urls_for_use(
        self,
        urls: list[str],
        *,
        use: str,
        preferred_roles: tuple[str, ...] = (),
    ) -> list[str]:
        """Rank safe source images first and isolate known hard-risk material."""

        unique_urls = _unique(urls)
        role_rank = {role: index for index, role in enumerate(preferred_roles)}

        def rank(url: str) -> tuple[int, int]:
            observation = self._source_image_observations.get(url)
            if not observation or not observation.get("inspection_complete"):
                safety = (
                    3 if use == "reference" and self._source_image_observations else 2
                )
                return safety, len(role_rank)
            safe_key = (
                "safe_for_listing_fallback"
                if use == "fallback"
                else "safe_for_generation_reference"
            )
            if observation.get(safe_key) is True:
                safety = 0
            elif use == "fallback" and not self._terminal_fallback_risks(observation):
                safety = 1
            else:
                safety = 3
            return safety, role_rank.get(
                str(observation.get("role") or "unknown"), len(role_rank)
            )

        ranked = sorted(
            enumerate(unique_urls), key=lambda pair: (*rank(pair[1]), pair[0])
        )
        non_hard_risk = [url for _, url in ranked if rank(url)[0] < 3]
        if non_hard_risk:
            return non_hard_risk

        warning = f"所有可用源图均触发视觉风险信号，{use} 阶段仅作最后兜底"
        if warning not in self._source_selection_warnings:
            self._source_selection_warnings.add(warning)
            self.warnings.append(warning)
            self.logger.warning(warning)
        if use == "reference":
            return []
        return [url for _, url in ranked]

    @staticmethod
    def _terminal_fallback_risks(observation: dict[str, Any]) -> list[str]:
        terminal_fields = {
            "has_watermark",
            "has_contact_info",
            "has_qr_code",
            "has_price_or_discount",
            "has_review_graphic",
            "has_certification_seal",
            "has_platform_mark",
            "has_before_after",
            "adult_or_sensitive_visual",
            "has_hate_or_extremism",
            "has_violence_or_weapon",
            "has_drugs_tobacco_or_alcohol",
            "has_third_party_brand",
            "has_logo",
            "has_overlay_text",
            "has_unrelated_props",
            "multiple_products",
        }
        reasons = [
            str(reason).casefold()
            for reason in observation.get("risk_reasons", [])
            if str(reason) != "inspection_incomplete"
        ]
        explicit = [
            field for field in terminal_fields if observation.get(field) is True
        ]
        keywords = (
            "contact",
            "phone",
            "email",
            "qr",
            "watermark",
            "price",
            "discount",
            "review",
            "certification",
            "platform mark",
            "before and after",
            "adult",
            "hate",
            "extrem",
            "violence",
            "weapon",
            "drug",
            "tobacco",
            "alcohol",
            "third-party",
            "third party",
            "brand",
            "logo",
            "unrelated prop",
        )
        return explicit + [
            reason for reason in reasons if any(key in reason for key in keywords)
        ]

    def _fallback_source_urls(
        self, facts: ProductFacts, *, asset_name: str
    ) -> list[str]:
        primary = _unique(facts.product_image_urls + facts.sku_image_urls)
        if asset_name == "main_image.jpeg":
            all_sources = _unique(primary + facts.description_image_urls)
            inspected_hero_sources = [
                url
                for url in all_sources
                if self._source_image_observations.get(url, {}).get(
                    "safe_for_main_image"
                )
                is True
            ]
            if inspected_hero_sources:
                return self._source_urls_for_use(
                    inspected_hero_sources,
                    use="fallback",
                    preferred_roles=("hero", "front", "variant"),
                )
            if self._source_image_observations:
                warning = (
                    "未发现同时满足单品、完整展示、无人物道具和干净背景的源主图；"
                    "主图进入质量降级兜底"
                )
                if warning not in self.warnings:
                    self.warnings.append(warning)
                    self.logger.warning(warning)
        preferred = (
            ("hero", "front", "variant", "side", "back", "detail", "lifestyle")
            if asset_name == "main_image.jpeg"
            else ("detail", "front", "side", "back", "variant", "lifestyle", "hero")
        )
        detail_emergency = asset_name != "main_image.jpeg"
        ranked_primary = self._source_urls_for_use(
            primary,
            use="reference" if detail_emergency else "fallback",
            preferred_roles=preferred,
        )
        usable_primary = [
            url
            for url in ranked_primary
            if self._source_image_observations.get(url, {}).get("role")
            not in {"size_chart", "packaging"}
            and not self._source_image_observations.get(url, {}).get(
                "has_overlay_text", False
            )
            and (
                not self._source_image_observations.get(url)
                or (
                    self._source_image_observations[url].get(
                        "safe_for_generation_reference"
                    )
                    is True
                    if detail_emergency
                    else not self._terminal_fallback_risks(
                        self._source_image_observations[url]
                    )
                )
            )
        ]
        if usable_primary:
            return usable_primary
        ranked_description = self._source_urls_for_use(
            facts.description_image_urls,
            use="reference" if detail_emergency else "fallback",
            preferred_roles=preferred,
        )
        usable_description = [
            url
            for url in ranked_description
            if self._source_image_observations.get(url, {}).get("role")
            not in {"size_chart", "packaging"}
            and not self._source_image_observations.get(url, {}).get(
                "has_overlay_text", False
            )
            and (
                not self._source_image_observations.get(url)
                or (
                    self._source_image_observations[url].get(
                        "safe_for_generation_reference"
                    )
                    is True
                    if detail_emergency
                    else not self._terminal_fallback_risks(
                        self._source_image_observations[url]
                    )
                )
            )
        ]
        return usable_description or ranked_primary or ranked_description

    def _detail_fallback_plan(
        self,
        facts: ProductFacts,
        *,
        index: int,
        main_reference_url: str,
    ) -> tuple[list[str], str]:
        """Assign one deterministic, non-overlapping purpose to each detail slot.

        Use at most three alternate full views, then reserve the final two slots
        for complementary upper/lower construction crops. This avoids filling a
        five-image detail set with near-identical model poses merely because their
        URLs differ. A verified back/side view is preferred before another front.
        """

        sources = self._fallback_source_urls(
            facts, asset_name=f"detail_image_{index}.jpeg"
        )
        if not sources:
            return [], ""
        ordered = [url for url in sources if url != main_reference_url]
        role_priority = {
            "back": 0,
            "side": 1,
            "detail": 2,
            "variant": 3,
            "lifestyle": 4,
            "front": 5,
            "hero": 6,
            "unknown": 7,
        }
        # Python's stable sort preserves seller order when observations are absent.
        ordered = sorted(
            enumerate(ordered),
            key=lambda pair: (
                role_priority.get(
                    str(
                        self._source_image_observations.get(pair[1], {}).get(
                            "role", "unknown"
                        )
                    ),
                    len(role_priority),
                ),
                pair[0],
            ),
        )
        ordered = [url for _, url in ordered]
        if main_reference_url in sources:
            ordered.append(main_reference_url)
        if not ordered:
            ordered = list(sources)

        full_view_limit = min(len(ordered), 3)
        if index <= full_view_limit:
            selected = ordered[index - 1]
            return [selected] + [url for url in ordered if url != selected], ""

        crop_sequence = ("upper", "lower", "left", "right", "center")
        crop_index = index - full_view_limit - 1
        focus_crop = crop_sequence[crop_index % len(crop_sequence)]
        selected = (
            main_reference_url
            if main_reference_url in sources
            else ordered[crop_index % len(ordered)]
        )
        return [selected] + [url for url in ordered if url != selected], focus_crop

    def _safe_generation_reference(self, url: str) -> bool:
        observation = self._source_image_observations.get(url)
        return (
            not observation or observation.get("safe_for_generation_reference") is True
        )


def _merge_source_vision_batches(
    batches: list[tuple[list[str], dict[str, Any]]], all_urls: list[str]
) -> dict[str, Any]:
    """Merge batch-local model indexes into one global, source-bound ledger."""

    global_index = {url: index for index, url in enumerate(all_urls)}
    source_images: list[dict[str, Any]] = []
    hero_candidates: list[int] = []
    size_rows: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, ...]] = set()
    merged_lists: dict[str, list[str]] = {
        key: []
        for key in (
            "visible_colors",
            "visible_design_features",
            "image_quality_notes",
            "prohibited_or_risky_visuals",
            "preservation_constraints",
        )
    }
    product_type = ""

    for batch_urls, payload in batches:
        observations = normalize_source_image_observations(payload, batch_urls)
        local_to_global = {
            local_index: global_index[url]
            for local_index, url in enumerate(batch_urls)
            if url in global_index
        }
        for item in observations:
            local_index = item.get("index")
            if not isinstance(local_index, int) or local_index not in local_to_global:
                continue
            bound = dict(item)
            bound["index"] = local_to_global[local_index]
            source_images.append(bound)

        if not product_type:
            product_type = str(payload.get("product_type") or "").strip()
        for key, target in merged_lists.items():
            values = payload.get(key)
            if not isinstance(values, list):
                continue
            for value in values:
                clean = " ".join(str(value).split())
                if clean and clean not in target:
                    target.append(clean)

        local_best = payload.get("best_hero_image_index")
        if isinstance(local_best, int) and local_best in local_to_global:
            hero_candidates.append(local_to_global[local_best])

        raw_rows = payload.get("size_chart_rows")
        if not isinstance(raw_rows, list):
            continue
        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            local_source = raw.get("source_image_index")
            if not isinstance(local_source, int) or local_source not in local_to_global:
                continue
            row = dict(raw)
            row["source_image_index"] = local_to_global[local_source]
            signature = tuple(
                str(row.get(key) or "").strip().casefold()
                for key in (
                    "size_label",
                    "bust_cm",
                    "length_cm",
                    "weight_guidance",
                    "source_image_index",
                )
            )
            if signature in seen_rows:
                continue
            seen_rows.add(signature)
            size_rows.append(row)

    source_images.sort(key=lambda item: int(item["index"]))
    by_index = {int(item["index"]): item for item in source_images}
    best_index = next(
        (
            index
            for index in hero_candidates
            if by_index.get(index, {}).get("safe_for_main_image") is True
        ),
        -1,
    )
    if best_index < 0:
        best_index = next(
            (
                index
                for index in hero_candidates
                if by_index.get(index, {}).get("safe_for_generation_reference") is True
            ),
            -1,
        )
    if best_index < 0:
        best_index = next(
            (
                int(item["index"])
                for item in source_images
                if item.get("safe_for_generation_reference") is True
                and item.get("role") in {"hero", "front"}
            ),
            -1,
        )

    return {
        "product_type": product_type,
        **merged_lists,
        "best_hero_image_index": best_index,
        "size_chart_rows": size_rows,
        "source_images": source_images,
        "requested_image_count": len(all_urls),
        "inspected_image_count": sum(
            item.get("inspection_complete") is True for item in source_images
        ),
        "batch_count": len(batches),
    }
