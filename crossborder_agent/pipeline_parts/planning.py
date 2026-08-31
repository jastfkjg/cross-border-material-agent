"""Planning responsibilities for the delivery pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..agent_tools import BoundedToolRegistry, ToolSpec
from ..models import (
    CreativePlan,
    ProductFacts,
    RunState,
    TaxonomyResult,
)
from ..table_evidence import select_render_table


class PlanningPipelineMixin:
    @staticmethod
    def _create_tool_registry(
        *, protected_detail_indexes: set[int] | None = None
    ) -> BoundedToolRegistry:
        protected = protected_detail_indexes or set()
        registry = BoundedToolRegistry()
        registry.add_spec(
            ToolSpec(
                name="regenerate_main_image",
                description="Regenerate and reselect the hero from trusted references; preserve the current hero unless the revision succeeds.",
                targets=("main_image.jpeg",),
                estimated_seconds=150,
                side_effects="replaces main_image.jpeg only after a candidate is downloaded and validated",
                repair_stage="artifact",
                invalidates=("video", "media_descriptions", "review"),
            )
        )
        registry.add_spec(
            ToolSpec(
                name="regenerate_detail_image",
                description="Regenerate one detail storyboard slot with evaluator-specific corrections; preserve the current slot on failure.",
                targets=tuple(
                    f"detail_image_{index}.jpeg"
                    for index in range(1, 6)
                    if index not in protected
                ),
                estimated_seconds=120,
                side_effects="replaces only the named detail image after candidate acceptance",
                repair_stage="artifact",
                invalidates=("video", "media_descriptions", "review"),
            )
        )
        registry.add_spec(
            ToolSpec(
                name="revise_localized_copy",
                description="Rewrite and re-audit one locale payload using precise evaluator feedback while preserving compact platform listing tables.",
                targets=tuple(
                    f"product_description_{language}.md"
                    for language in ("en", "ko", "pt")
                ),
                estimated_seconds=75,
                side_effects="replaces only the named localized description after schema and factual validation",
                repair_stage="projection",
                invalidates=("review",),
            )
        )
        registry.add_spec(
            ToolSpec(
                name="regenerate_video",
                description="Regenerate the product video with a targeted temporal correction; preserve the current playable video on failure.",
                targets=("product_video.mp4",),
                estimated_seconds=210,
                side_effects="replaces product_video.mp4 only after download, audio stripping, and playback validation",
                repair_stage="artifact",
                invalidates=("review",),
            )
        )
        registry.add_spec(
            ToolSpec(
                name="reconcile_fact_ledger",
                description="Ask independent evidence reconcilers to reconsider the structured-versus-visual fact ledger using the supplied investigation context.",
                targets=("fact_ledger",),
                estimated_seconds=90,
                side_effects="rebuilds canonical facts, claims and expected delivery state when the evidence decision changes",
                repair_stage="evidence",
                invalidates=(
                    "taxonomy",
                    "localization",
                    "visual_plan",
                    "artifacts",
                    "review",
                ),
            )
        )
        registry.add_spec(
            ToolSpec(
                name="reconsider_taxonomy",
                description="Reopen generic taxonomy/schema exploration with current source evidence and the reconciled fact ledger; exact returned IDs are validated against the supplied snapshots.",
                targets=("taxonomy",),
                estimated_seconds=210,
                side_effects="atomically replaces grounded category/mappings, localized projections and delivery specification after successful exploration",
                repair_stage="decision",
                invalidates=("localization", "delivery_spec", "review"),
            )
        )
        registry.add_spec(
            ToolSpec(
                name="reselect_detail_set",
                description="Give the complete retained candidate state, current selection, and local-review evidence to the image-set editor for a new joint selection.",
                targets=("detail_image_set",),
                estimated_seconds=90,
                side_effects="atomically installs only a source-grounded, hard-safe combination selected for the existing slots",
                repair_stage="artifact_set",
                invalidates=("video", "media_descriptions", "review"),
            )
        )
        return registry

    def _bind_repair_tools(
        self,
        registry: BoundedToolRegistry,
        *,
        facts: ProductFacts,
        taxonomy: TaxonomyResult,
        vision: dict[str, Any],
        creative_plan: CreativePlan,
        agent_plan: dict[str, Any],
        state: RunState,
        localization_payloads: dict[str, dict[str, Any]],
        localization_sources: dict[str, str],
        work_dir: Path,
        downloads_dir: Path,
        category_tree: dict[str, Any],
        attribute_data: dict[str, Any],
    ) -> None:
        registry.bind(
            "regenerate_main_image",
            lambda target, instruction: self._repair_main_image(
                target,
                instruction,
                facts,
                creative_plan,
                vision,
                state.assets,
                work_dir,
                downloads_dir,
            ),
        )
        registry.bind(
            "regenerate_detail_image",
            lambda target, instruction: self._repair_detail_image(
                target,
                instruction,
                facts,
                creative_plan,
                state.assets,
                work_dir,
                downloads_dir,
            ),
        )
        registry.bind(
            "revise_localized_copy",
            lambda target, instruction: self._repair_localized_copy(
                target,
                instruction,
                facts,
                taxonomy,
                creative_plan,
                agent_plan,
                localization_payloads,
                localization_sources,
                work_dir,
            ),
        )
        registry.bind(
            "regenerate_video",
            lambda target, instruction: self._repair_video(
                target,
                instruction,
                facts,
                creative_plan,
                state.assets,
                work_dir,
                downloads_dir,
            ),
        )
        registry.bind(
            "reconcile_fact_ledger",
            lambda target, instruction: self._repair_fact_ledger(
                instruction,
                facts=facts,
                taxonomy=taxonomy,
                vision=vision,
                state=state,
            ),
        )
        registry.bind(
            "reconsider_taxonomy",
            lambda target, instruction: self._repair_taxonomy(
                instruction,
                facts=facts,
                taxonomy=taxonomy,
                category_tree=category_tree,
                attribute_data=attribute_data,
                creative_plan=creative_plan,
                state=state,
                localization_payloads=localization_payloads,
                localization_sources=localization_sources,
                work_dir=work_dir,
            ),
        )
        registry.bind(
            "reselect_detail_set",
            lambda target, instruction: self._repair_detail_set_selection(
                instruction,
                facts=facts,
                creative_plan=creative_plan,
                state=state,
                localization_payloads=localization_payloads,
                taxonomy=taxonomy,
                work_dir=work_dir,
                downloads_dir=downloads_dir,
            ),
        )

        def operation_available(operation: str) -> tuple[bool, str]:
            if self.client is None:
                return False, f"{operation} model client unavailable"
            probe = getattr(self.client, "operation_available", None)
            if callable(probe) and not probe(operation):
                return False, f"{operation} model capability is disabled for this run"
            return True, "available"

        def main_precondition(target: str) -> tuple[bool, str]:
            available, reason = operation_available("image")
            if not available:
                return available, reason
            references = self._ordered_source_urls(
                facts,
                vision,
                preferred_roles=creative_plan.main_reference_roles,
                preferred_indexes=creative_plan.main_reference_indexes,
            )
            if not references:
                return False, "no trusted hero generation reference"
            return True, "trusted hero reference available"

        def detail_precondition(target: str) -> tuple[bool, str]:
            available, reason = operation_available("image")
            if not available:
                return available, reason
            try:
                index = int(target.removeprefix("detail_image_").split(".", 1)[0])
            except ValueError:
                return False, "invalid detail target"
            selected_table = select_render_table(facts.evidence_tables)
            if (
                selected_table is not None
                and index
                == int(selected_table.presentation.get("target_detail_index") or 0)
            ):
                return False, "model-selected deterministic evidence table is protected"
            main_asset = self._find_asset(state.assets, "main_image.jpeg")
            references = self._detail_reference_selection(
                index,
                facts,
                main_asset.source_url,
                preferred_roles=(
                    creative_plan.detail_reference_roles[index - 1]
                    if index <= len(creative_plan.detail_reference_roles)
                    else ()
                ),
                preferred_indexes=(
                    creative_plan.detail_reference_indexes[index - 1]
                    if index <= len(creative_plan.detail_reference_indexes)
                    else ()
                ),
            )
            if not references:
                return False, f"no trusted detail reference for slot {index}"
            return True, f"{len(references)} trusted detail references available"

        def copy_precondition(target: str) -> tuple[bool, str]:
            available, reason = operation_available("chat")
            if not available:
                return available, reason
            language = target.removeprefix("product_description_").split(".", 1)[0]
            if language not in localization_payloads:
                return False, f"localized payload unavailable: {language}"
            return True, "localized payload available"

        def video_precondition(target: str) -> tuple[bool, str]:
            available, reason = operation_available("video")
            if not available:
                return available, reason
            main_asset = self._find_asset(state.assets, "main_image.jpeg")
            first_frame = main_asset.source_url
            if not first_frame or not self._safe_generation_reference(first_frame):
                candidates = self._source_urls_for_use(
                    self._fallback_source_urls(facts, asset_name="main_image.jpeg"),
                    use="reference",
                    preferred_roles=("hero", "front"),
                )
                first_frame = candidates[0] if candidates else ""
            if not first_frame:
                return False, "no safe video first frame"
            return True, "video model and safe first frame available"

        def evidence_precondition(target: str) -> tuple[bool, str]:
            available, reason = operation_available("chat")
            if not available:
                return available, reason
            if not vision:
                return False, "no inspected source-image evidence is available"
            return True, "structured and visual evidence are available"

        def taxonomy_precondition(target: str) -> tuple[bool, str]:
            available, reason = operation_available("chat")
            if not available:
                return available, reason
            if not category_tree or not attribute_data:
                return False, "taxonomy snapshots are unavailable"
            return True, "taxonomy snapshots and current evidence are available"

        def detail_set_precondition(target: str) -> tuple[bool, str]:
            available, reason = operation_available("review")
            if not available:
                return available, reason
            with self._detail_candidate_pool_lock:
                pool_count = sum(
                    1 for value in self._detail_candidate_pools.values() if value
                )
            if pool_count < 2:
                return False, "fewer than two detail slots retain candidate state"
            return True, f"candidate state retained for {pool_count} detail slots"

        registry.bind_precondition("regenerate_main_image", main_precondition)
        registry.bind_precondition("regenerate_detail_image", detail_precondition)
        registry.bind_precondition("revise_localized_copy", copy_precondition)
        registry.bind_precondition("regenerate_video", video_precondition)
        registry.bind_precondition("reconcile_fact_ledger", evidence_precondition)
        registry.bind_precondition("reconsider_taxonomy", taxonomy_precondition)
        registry.bind_precondition("reselect_detail_set", detail_set_precondition)
