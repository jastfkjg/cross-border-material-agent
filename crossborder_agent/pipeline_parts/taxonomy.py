"""Taxonomy responsibilities for the delivery pipeline."""

from __future__ import annotations

import json
import time
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
    def _fresh_taxonomy_agent(
        self,
        category_tree: dict[str, Any],
        attribute_data: dict[str, Any],
        *,
        max_turns: int,
    ) -> TaxonomyReActAgent:
        """Create a clean semantic recovery context over the same grounded data."""

        return TaxonomyReActAgent(
            self.client,
            category_tree,
            attribute_data,
            skill_instructions=self.skills.compile(
                "taxonomy", "product-grounding", "aliexpress-taxonomy"
            ),
            trace=self.trace,
            max_turns=max_turns,
        )

    def _recover_category_with_model(
        self,
        facts: ProductFacts,
        category_tree: dict[str, Any],
        attribute_data: dict[str, Any],
        *,
        context: str,
        max_turns: int = 14,
    ) -> CategoryChoice | None:
        """Let a fresh model context recover a semantic category transaction."""

        if self.client is None or self.deadline - time.monotonic() < 540:
            return None
        recovery = self._fresh_taxonomy_agent(
            category_tree, attribute_data, max_turns=max_turns
        )
        try:
            return recovery.resolve_category(facts, decision_context=context)
        except Exception as exc:
            self.trace.emit(
                "taxonomy.category_model_recovery_failed",
                error=str(exc),
                context=context[:1000],
            )
            return None
        finally:
            recovery.close()

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
        except Exception as exc:
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
        except Exception as exc:
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
        explorer = self._fresh_taxonomy_agent(
            category_tree, attribute_data, max_turns=50
        )
        try:
            resolved_category = explorer.resolve_category(facts)
        except Exception as exc:
            explorer.close()
            resolved_category = self._recover_category_with_model(
                facts,
                category_tree,
                attribute_data,
                context=(
                    "A previous category transaction ended without a validated result. "
                    f"Failure type={getattr(exc, 'category', type(exc).__name__)}; "
                    f"diagnostic={str(exc)[:1200]}. "
                    "Start from the source evidence and taxonomy workspace in this fresh context, "
                    "change the exploration strategy, and commit the best grounded leaf. Do not "
                    "delegate the semantic choice to a lexical fallback."
                ),
            )
            if resolved_category is None:
                self.logger.warning(
                    "Category ReAct exploration and fresh model recovery failed; keeping the availability fallback result: %s",
                    exc,
                )
                self.trace.emit(
                    "taxonomy.category_transaction_failed",
                    category=taxonomy.category.category_id,
                    confidence=taxonomy.category.confidence,
                    error=str(exc),
                )
                return taxonomy
            self.trace.emit(
                "taxonomy.category_model_recovered",
                fallback_category_id=taxonomy.category.category_id,
                selected_category_id=resolved_category.category_id,
                trigger=str(exc),
            )
            explorer = self._fresh_taxonomy_agent(
                category_tree, attribute_data, max_turns=32
            )

        # A disagreement with the structurally valid availability candidate is
        # an uncertainty signal, not a vote for either answer. Give a fresh
        # challenger the same complete evidence. If it disagrees too, a third
        # fresh context adjudicates the two grounded proposals. No product term,
        # category ID, or benchmark-specific preference is encoded here.
        if (
            resolved_category.category_id != taxonomy.category.category_id
            and self.deadline - time.monotonic() >= 780
        ):
            challenger = self._recover_category_with_model(
                facts,
                category_tree,
                attribute_data,
                context=(
                    "Independently challenge a prior category proposal. Do not assume either the "
                    "model proposal or the availability candidate is correct. Inspect source evidence, "
                    "leaf ancestry, sibling specialization and schema relationships before committing. "
                    f"Prior model proposal: {resolved_category.category_id} | "
                    f"{resolved_category.path}. Availability candidate: "
                    f"{taxonomy.category.category_id} | {taxonomy.category.path}."
                ),
                max_turns=12,
            )
            if (
                challenger is not None
                and challenger.category_id != resolved_category.category_id
                and self.deadline - time.monotonic() >= 660
            ):
                adjudicated = self._recover_category_with_model(
                    facts,
                    category_tree,
                    attribute_data,
                    context=(
                        "Two independent grounded category explorations disagree. Reinspect the exact "
                        "product evidence and relevant taxonomy records, explicitly test which leaf's "
                        "qualifiers are supported, and commit the best answer. Proposal A: "
                        f"{resolved_category.category_id} | {resolved_category.path}. Proposal B: "
                        f"{challenger.category_id} | {challenger.path}."
                    ),
                    max_turns=12,
                )
                if adjudicated is not None:
                    self.trace.emit(
                        "taxonomy.category_disagreement_adjudicated",
                        proposal_a=resolved_category.category_id,
                        proposal_b=challenger.category_id,
                        selected=adjudicated.category_id,
                    )
                    resolved_category = adjudicated
            elif challenger is not None:
                self.trace.emit(
                    "taxonomy.category_challenger_agreed",
                    selected_category_id=resolved_category.category_id,
                )
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
        except Exception as exc:
            recovered: TaxonomyResult | None = None
            if self.deadline - time.monotonic() >= 480:
                recovery = self._fresh_taxonomy_agent(
                    category_tree, attribute_data, max_turns=16
                )
                try:
                    recovered = recovery.resolve_attributes(
                        facts,
                        resolved_category,
                        decision_context=(
                            "A prior attribute transaction failed after the leaf category was committed. "
                            f"Failure type={getattr(exc, 'category', type(exc).__name__)}; "
                            f"diagnostic={str(exc)[:1600]}. "
                            "Use the fresh workspace to inspect the committed category's schema, source refs, "
                            "accepted mappings and unresolved evidence. Change strategy and submit the best "
                            "grounded mapping ledger; never change the committed category."
                        ),
                    )
                except Exception as recovery_exc:
                    self.trace.emit(
                        "taxonomy.attribute_model_recovery_failed",
                        selected_category_id=resolved_category.category_id,
                        error=str(recovery_exc),
                    )
                finally:
                    recovery.close()
            resolved = recovered or category_anchored_fallback
            self.logger.warning(
                (
                    "Attribute ReAct exploration failed; a fresh model recovery was used: %s"
                    if recovered is not None
                    else "Attribute ReAct exploration and fresh model recovery failed; keeping the committed category and falling back only for schema/mapping: %s"
                ),
                exc,
            )
            self.trace.emit(
                "taxonomy.attribute_transaction_failed",
                selected_category_id=resolved_category.category_id,
                selected_category_path=resolved_category.path,
                fallback_schema_id=resolved.attribute_schema_category_id,
                fallback_mapping_count=len(resolved.attributes),
                model_recovered=recovered is not None,
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
