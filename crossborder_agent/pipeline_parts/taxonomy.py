"""Taxonomy responsibilities for the delivery pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..agent_tools import ToolExecution
from ..api import ApiError
from ..claims import build_claim_ledger, filter_invalid_mapping_provenance
from ..decision_state import (
    DependencyState,
    assess_evidence_sufficiency,
    build_canonical_product_state,
    build_expected_delivery_spec,
)
from ..localization import generate_copy_payload
from ..models import (
    CreativePlan,
    ProductFacts,
    RunState,
    TaxonomyResult,
)
from ..qa import EXPECTED_FILES
from ..taxonomy import resolve_taxonomy
from ..taxonomy_agent import TaxonomyReActAgent


class TaxonomyPipelineMixin:
    def _repair_taxonomy(
        self,
        instruction: str,
        *,
        facts: ProductFacts,
        taxonomy: TaxonomyResult,
        category_tree: dict[str, Any],
        attribute_data: dict[str, Any],
        creative_plan: CreativePlan,
        state: RunState,
        localization_payloads: dict[str, dict[str, Any]],
        localization_sources: dict[str, str],
        work_dir: Path,
    ) -> ToolExecution:
        before = json.dumps(
            {
                "category": taxonomy.category.category_id,
                "schema": taxonomy.attribute_schema_category_id,
                "attributes": [
                    (item.attr_id, item.value_id, item.source_name, item.source_value)
                    for item in taxonomy.attributes
                ],
                "missing_required": taxonomy.missing_required,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        explorer = TaxonomyReActAgent(
            self.client,
            category_tree,
            attribute_data,
            skill_instructions=self.skills.compile(
                "taxonomy", "product-grounding", "aliexpress-taxonomy"
            ),
            trace=self.trace,
            max_turns=20,
        )
        try:
            revised_category = explorer.resolve_category(
                facts, decision_context=instruction
            )
        except ApiError as exc:
            explorer.close()
            return ToolExecution("failed", f"category reconsideration failed: {exc}")
        category_anchored_fallback = resolve_taxonomy(
            facts,
            category_tree,
            attribute_data,
            preferred_category_id=revised_category.category_id,
        )
        try:
            revised = explorer.resolve_attributes(
                facts,
                revised_category,
                decision_context=instruction,
            )
        except ApiError as exc:
            revised = category_anchored_fallback
            self.trace.emit(
                "taxonomy.attribute_transaction_failed",
                selected_category_id=revised_category.category_id,
                fallback_schema_id=revised.attribute_schema_category_id,
                error=str(exc),
            )
        finally:
            explorer.close()
        provenance_warnings = filter_invalid_mapping_provenance(facts, revised)
        after = json.dumps(
            {
                "category": revised.category.category_id,
                "schema": revised.attribute_schema_category_id,
                "attributes": [
                    (item.attr_id, item.value_id, item.source_name, item.source_value)
                    for item in revised.attributes
                ],
                "missing_required": revised.missing_required,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if after == before:
            return ToolExecution(
                "completed",
                "taxonomy was re-explored but the grounded result did not change",
                {"changed": False, "warnings": provenance_warnings},
            )
        required_sources = state.expected_delivery_spec.get(
            "required_mapping_sources", []
        )
        revised_sources = {
            (
                "sales" if item.sales_attribute else "product",
                item.source_name,
                item.source_value,
            )
            for item in revised.attributes
        }
        lost_sources = [
            item
            for item in required_sources
            if isinstance(item, dict)
            and (
                str(item.get("scope") or ""),
                str(item.get("source_name") or ""),
                str(item.get("source_value") or ""),
            )
            not in revised_sources
        ]
        if lost_sources:
            return ToolExecution(
                "rejected",
                "taxonomy proposal would drop frozen source coverage; reconsider category/schema and mappings together",
                {"missing_mapping_sources": lost_sources},
            )
        revised_claim_ledger = build_claim_ledger(
            facts, revised, state.vision_observations
        )
        # Localized term maps and copy are projections of taxonomy. Rebuild them
        # from the model instead of re-rendering stale pre-repair payloads.
        if self.client is not None:
            regenerated: dict[str, dict[str, Any]] = {}
            regenerated_sources: dict[str, str] = {}
            try:
                for language in ("en", "ko", "pt"):
                    payload, source = generate_copy_payload(
                        language,
                        facts,
                        revised,
                        creative_plan,
                        self.client,
                        claim_ledger=revised_claim_ledger,
                        agent_guidance=str(
                            state.agent_plan.get("localization_priorities", {}).get(
                                language, ""
                            )
                        ),
                        revision_feedback=(
                            "Upstream taxonomy changed. Rebuild the complete locale projection "
                            "from the current canonical evidence and taxonomy; do not reuse old terms."
                        ),
                        skill_instructions=self.skills.compile(
                            "copy", "product-grounding", "marketplace-materials"
                        ),
                    )
                    regenerated[language] = payload
                    regenerated_sources[language] = source
            except (ApiError, ValueError, TypeError) as exc:
                return ToolExecution(
                    "failed",
                    f"taxonomy changed but dependent localization rebuild failed: {exc}",
                )
            localization_payloads.update(regenerated)
            localization_sources.update(regenerated_sources)
        taxonomy.category = revised.category
        taxonomy.attributes = revised.attributes
        taxonomy.missing_required = revised.missing_required
        taxonomy.attribute_schema_category_id = revised.attribute_schema_category_id
        state.claim_ledger = revised_claim_ledger
        canonical_state = build_canonical_product_state(
            facts, facts.reconciled_fact_ledger
        )
        evidence_sufficiency = assess_evidence_sufficiency(
            state.vision_observations, canonical_state
        )
        expected_delivery_spec = build_expected_delivery_spec(
            canonical=canonical_state,
            taxonomy=taxonomy,
            claim_ledger=state.claim_ledger,
            evidence=evidence_sufficiency,
            required_files=state.expected_delivery_spec.get(
                "required_files", EXPECTED_FILES
            ),
            preserve_mapping_sources=required_sources,
        )
        state.expected_delivery_spec = expected_delivery_spec.to_dict()
        dependency_state = DependencyState.from_dict(state.dependency_state)
        taxonomy_version = dependency_state.record(
            "taxonomy",
            {
                "category": taxonomy.category.category_id,
                "schema": taxonomy.attribute_schema_category_id,
                "attributes": [
                    (item.attr_id, item.value_id, item.source_name, item.source_value)
                    for item in taxonomy.attributes
                ],
            },
            canonical=canonical_state.version,
        )
        dependency_state.record(
            "localization", localization_payloads, taxonomy=taxonomy_version
        )
        dependency_state.record(
            "delivery_spec",
            expected_delivery_spec.to_dict(),
            canonical=canonical_state.version,
            taxonomy=expected_delivery_spec.taxonomy_version,
        )
        dependency_state.invalidate(
            "taxonomy", ["review"], "taxonomy decision and projections changed"
        )
        state.dependency_state = dependency_state.to_dict()
        self._write_localized_descriptions(
            facts,
            taxonomy,
            creative_plan,
            localization_payloads,
            state.assets,
            work_dir,
            state.visual_set_review,
        )
        self.logger.info(
            "Taxonomy repair was committed atomically: category=%s schema=%s mappings=%d",
            taxonomy.category.category_id,
            taxonomy.attribute_schema_category_id,
            len(taxonomy.attributes),
        )
        return ToolExecution(
            "completed",
            "taxonomy and all dependent localized projections were rebuilt from grounded exploration",
            {
                "changed": True,
                "category_id": taxonomy.category.category_id,
                "schema_id": taxonomy.attribute_schema_category_id,
                "mapping_count": len(taxonomy.attributes),
                "warnings": provenance_warnings,
            },
        )

    def _adjudicate_taxonomy(
        self,
        facts: ProductFacts,
        taxonomy: TaxonomyResult,
        category_tree: dict[str, Any],
        attribute_data: dict[str, Any],
    ) -> TaxonomyResult:
        if self.client is None:
            return taxonomy
        explorer = TaxonomyReActAgent(
            self.client,
            category_tree,
            attribute_data,
            skill_instructions=self.skills.compile(
                "taxonomy", "product-grounding", "aliexpress-taxonomy"
            ),
            trace=self.trace,
        )
        try:
            resolved_category = explorer.resolve_category(facts)
        except ApiError as exc:
            explorer.close()
            self.logger.warning(
                "Category ReAct exploration failed; keeping the offline fallback result: %s",
                exc,
            )
            self.trace.emit(
                "taxonomy.category_transaction_failed",
                category=taxonomy.category.category_id,
                confidence=taxonomy.category.confidence,
                error=str(exc),
            )
            return taxonomy
        category_anchored_fallback = resolve_taxonomy(
            facts,
            category_tree,
            attribute_data,
            preferred_category_id=resolved_category.category_id,
        )
        self.trace.emit(
            "taxonomy.category_transaction_committed",
            local_fallback_category_id=taxonomy.category.category_id,
            selected_category_id=resolved_category.category_id,
            selected_category_path=resolved_category.path,
        )
        try:
            resolved = explorer.resolve_attributes(facts, resolved_category)
        except ApiError as exc:
            resolved = category_anchored_fallback
            self.logger.warning(
                "Attribute ReAct exploration failed; keeping the committed category and falling back only for schema/mapping: %s",
                exc,
            )
            self.trace.emit(
                "taxonomy.attribute_transaction_failed",
                selected_category_id=resolved_category.category_id,
                selected_category_path=resolved_category.path,
                fallback_schema_id=resolved.attribute_schema_category_id,
                fallback_mapping_count=len(resolved.attributes),
                error=str(exc),
            )
            return resolved
        finally:
            explorer.close()
        self.trace.emit(
            "taxonomy.react_resolved",
            local_fallback_category_id=taxonomy.category.category_id,
            selected_category_id=resolved.category.category_id,
            schema_category_id=resolved.attribute_schema_category_id,
            accepted_model_mapping_count=len(resolved.attributes),
            missing_required=resolved.missing_required,
        )
        return resolved
