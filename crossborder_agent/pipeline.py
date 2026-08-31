"""End-to-end bounded agent orchestration facade."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .api import ApiConfig, HttpJsonClient, QwenClient
from .bounded_agent import BoundedDeliveryAgent
from .claims import build_claim_ledger, filter_invalid_mapping_provenance
from .debug_trace import DebugTrace
from .decision_state import (
    DependencyState,
    assess_evidence_sufficiency,
    build_canonical_product_state,
    build_expected_delivery_spec,
)
from .input_loader import discover_input_files, load_json, load_product_facts
from .localization import generate_copy_payload
from .models import (
    AssetResult,
    CategoryChoice,
    CreativePlan,
    ProductFacts,
    RunState,
    TaxonomyResult,
)
from .pipeline_parts import (
    EvidencePipelineMixin,
    PlanningPipelineMixin,
    ProductionPipelineMixin,
    ReviewPipelineMixin,
    TaxonomyPipelineMixin,
    TransactionPipelineMixin,
)
from .pipeline_parts.evidence import (
    _merge_source_vision_batches as _merge_source_vision_batches,
)
from .pipeline_parts.common import (
    PipelineError,
    SemanticRejection as SemanticRejection,
)
from .planning import create_creative_plan, fallback_creative_plan
from .qa import EXPECTED_FILES, validate_delivery
from .skill_runtime import SkillLibrary
from .table_evidence import select_render_table
from .taxonomy import resolve_taxonomy


class Pipeline(
    PlanningPipelineMixin,
    EvidencePipelineMixin,
    TaxonomyPipelineMixin,
    ProductionPipelineMixin,
    ReviewPipelineMixin,
    TransactionPipelineMixin,
):
    def __init__(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        logger: logging.Logger,
        product_id: str = "",
        timeout_seconds: int = 29 * 60,
        offline: bool = False,
        debug: bool = False,
        run_profile: str = "full",
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.logger = logger
        self.product_id = product_id
        self.offline = offline
        self.debug = debug
        if run_profile not in {"full", "fast"}:
            raise ValueError(f"unsupported run profile: {run_profile}")
        self.run_profile = run_profile
        self.fast_mode = run_profile == "fast"
        self.started_monotonic = time.monotonic()
        self.deadline = self.started_monotonic + timeout_seconds
        self.trace = DebugTrace(logger, enabled=debug)
        self.api_config = None if offline else ApiConfig.from_environment()
        if not offline and self.api_config is None:
            required = (
                "DASHSCOPE_API_KEY",
                "DASHSCOPE_BASE_URL",
                "OPENAI_BASE_URL",
            )
            missing = [
                name for name in required if not os.environ.get(name, "").strip()
            ]
            self.offline = True
            self.logger.warning(
                "Model configuration is incomplete; switching to deterministic fallback: %s",
                ", ".join(missing),
            )
        self.client = (
            QwenClient(self.api_config, logger, self.deadline, self.trace)
            if self.api_config
            else None
        )
        self.downloader = (
            self.client.http
            if self.client
            else HttpJsonClient(logger, self.deadline, self.trace)
        )
        self.skills = SkillLibrary()
        self.agent = BoundedDeliveryAgent(self.client, logger, self.skills)
        self.warnings: list[str] = []
        self._raw_counter = 0
        self._raw_counter_lock = threading.Lock()
        self._detail_candidate_pool_lock = threading.Lock()
        self._detail_candidate_pools: dict[int, dict[str, Any]] = {}
        self._detail_candidate_reviews: dict[int, dict[str, Any]] = {}
        self._source_image_observations: dict[str, dict[str, Any]] = {}
        self._source_selection_warnings: set[str] = set()

    def _ensure_time(self, reserve_seconds: float = 0) -> None:
        remaining = self.deadline - time.monotonic()
        if remaining <= reserve_seconds:
            raise PipelineError(
                f"Insufficient runtime remains; {reserve_seconds:.0f} seconds must be reserved"
            )

    def run(self) -> RunState:
        self.trace.emit(
            "run.start",
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            product_id=self.product_id,
            offline=self.offline,
            debug=self.debug,
            run_profile=self.run_profile,
            models=self.client.model_summary if self.client else {"mode": "offline"},
            deadline_seconds=round(self.deadline - self.started_monotonic, 1),
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        work_dir = self.output_dir / f".agent-work-{uuid.uuid4().hex}"
        downloads_dir = work_dir / "_downloads"
        work_dir.mkdir(parents=True, exist_ok=False)
        downloads_dir.mkdir(parents=True, exist_ok=False)

        facts: ProductFacts | None = None
        taxonomy: TaxonomyResult | None = None
        creative_plan: CreativePlan | None = None
        state: RunState | None = None
        category_tree: Any = None
        attribute_data: Any = None
        localization_sources: dict[str, str] = {}
        localization_payloads: dict[str, dict[str, Any]] = {}
        plan_model = "deterministic-availability-fallback"

        try:
            product_path, categories_path, attributes_path = discover_input_files(
                self.input_dir, self.product_id
            )
            facts = load_product_facts(product_path)
            self.trace.emit(
                "input.loaded",
                offer_id=facts.offer_id,
                fingerprint=facts.fingerprint,
                source_category=facts.source_category_name,
                attribute_count=len(facts.attributes),
                sku_count=len(facts.skus),
                image_count=len(facts.all_image_urls()),
            )
            category_tree = load_json(categories_path)
            attribute_data = load_json(attributes_path)
            # Establish visual evidence and one canonical product decision before
            # any downstream projection is allowed to choose taxonomy or claims.
            vision = self._analyze_source_images(facts)
            self.trace.emit("vision.source_review", result=vision)
            facts.reconciled_fact_ledger = self.agent.reconcile_facts(facts, vision)
            canonical_state = build_canonical_product_state(
                facts, facts.reconciled_fact_ledger
            )
            evidence_sufficiency = assess_evidence_sufficiency(vision, canonical_state)
            self.trace.emit(
                "facts.reconciled",
                ledger=facts.reconciled_fact_ledger,
                canonical_state=canonical_state.to_dict(),
                evidence_sufficiency=evidence_sufficiency.to_dict(),
                models=facts.reconciled_fact_ledger.get("models", []),
                conflict_count=len(facts.reconciled_fact_ledger.get("conflicts", [])),
            )
            self.logger.info(
                "Fact-evidence adjudication completed: models=%s conflicts=%d attribute_decisions=%d",
                ",".join(facts.reconciled_fact_ledger.get("models", [])) or "none",
                len(facts.reconciled_fact_ledger.get("conflicts", [])),
                len(facts.reconciled_fact_ledger.get("attribute_decisions", [])),
            )
            self._apply_evidence_table_observations(facts, vision)
            taxonomy = resolve_taxonomy(facts, category_tree, attribute_data)
            if not self.fast_mode:
                taxonomy = self._adjudicate_taxonomy(
                    facts, taxonomy, category_tree, attribute_data
                )
            else:
                self.trace.emit("taxonomy.adjudication_skipped", reason="fast-profile")
            provenance_warnings = filter_invalid_mapping_provenance(facts, taxonomy)
            self.warnings.extend(provenance_warnings)
            for warning in provenance_warnings:
                self.logger.warning(warning)
            self.trace.emit(
                "taxonomy.resolved",
                category=taxonomy.category.__dict__
                if hasattr(taxonomy.category, "__dict__")
                else {
                    "category_id": taxonomy.category.category_id,
                    "name": taxonomy.category.name,
                    "path": taxonomy.category.path,
                    "confidence": taxonomy.category.confidence,
                    "method": taxonomy.category.method,
                    "candidates": taxonomy.category.candidates,
                },
                mapped_attributes=[
                    {
                        "attr_id": item.attr_id,
                        "name": item.name,
                        "source_name": item.source_name,
                        "source_value": item.source_value,
                        "value_id": item.value_id,
                        "platform_value": item.platform_value,
                        "sales_attribute": item.sales_attribute,
                    }
                    for item in taxonomy.attributes
                ],
                missing_required=taxonomy.missing_required,
            )
            self.logger.info(
                "Product %s: category_id=%s (confidence=%.2f, method=%s)",
                facts.offer_id,
                taxonomy.category.category_id,
                taxonomy.category.confidence,
                taxonomy.category.method,
            )

            claim_ledger = build_claim_ledger(facts, taxonomy, vision)
            self.trace.emit(
                "claims.ledger",
                count=len(claim_ledger),
                publishable_count=sum(
                    "buyer_copy" in item.allowed_surfaces for item in claim_ledger
                ),
            )
            selected_table = select_render_table(facts.evidence_tables)
            protected_detail_indexes = (
                {int(selected_table.presentation["target_detail_index"])}
                if selected_table is not None
                else set()
            )
            tool_registry = self._create_tool_registry(
                protected_detail_indexes=protected_detail_indexes
            )
            agent_plan = self.agent.plan_delivery(
                facts,
                taxonomy,
                vision,
                tool_registry,
                use_model=not self.fast_mode,
            )
            self.trace.emit("agent.plan", plan=agent_plan)
            creative_plan, plan_model = create_creative_plan(
                facts,
                taxonomy,
                vision,
                self.client,
                agent_guidance=agent_plan,
                skill_instructions=self.skills.compile(
                    "creative-plan",
                    "product-grounding",
                    "marketplace-materials",
                ),
            )
            self.logger.info("Creative-plan source: %s", plan_model)
            self.trace.emit(
                "creative.plan",
                source=plan_model,
                plan={
                    "visual_theme": creative_plan.visual_theme,
                    "main_prompt": creative_plan.main_prompt,
                    "detail_prompts": creative_plan.detail_prompts,
                    "video_prompt": creative_plan.video_prompt,
                    "market_angles": creative_plan.market_angles,
                },
            )

            expected_delivery_spec = build_expected_delivery_spec(
                canonical=canonical_state,
                taxonomy=taxonomy,
                claim_ledger=claim_ledger,
                evidence=evidence_sufficiency,
                required_files=EXPECTED_FILES,
            )
            dependencies = DependencyState()
            dependencies.record("evidence", evidence_sufficiency.to_dict())
            dependencies.record(
                "canonical",
                canonical_state.to_dict(),
                evidence=evidence_sufficiency.version,
            )
            dependencies.record(
                "taxonomy",
                {
                    "category": taxonomy.category.category_id,
                    "schema": taxonomy.attribute_schema_category_id,
                    "attributes": [
                        (
                            item.attr_id,
                            item.value_id,
                            item.source_name,
                            item.source_value,
                        )
                        for item in taxonomy.attributes
                    ],
                },
                canonical=canonical_state.version,
            )
            dependencies.record(
                "delivery_spec",
                expected_delivery_spec.to_dict(),
                canonical=canonical_state.version,
                taxonomy=expected_delivery_spec.taxonomy_version,
            )

            state = RunState(
                started_at=datetime.now(timezone.utc).isoformat(),
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir),
                facts=facts,
                taxonomy=taxonomy,
                creative_plan=creative_plan,
                claim_ledger=claim_ledger,
                vision_observations=vision,
                warnings=self.warnings,
                agent_plan=agent_plan,
                canonical_product_state=canonical_state.to_dict(),
                evidence_sufficiency=evidence_sufficiency.to_dict(),
                expected_delivery_spec=expected_delivery_spec.to_dict(),
                dependency_state=dependencies.to_dict(),
            )

            main_asset, main_reference_url = self._build_main_image(
                facts, creative_plan, vision, work_dir, downloads_dir
            )
            state.assets.append(main_asset)

            detail_assets: dict[int, AssetResult] = {}
            video_result: AssetResult | None = None

            with concurrent.futures.ThreadPoolExecutor(
                # Keep enough parallelism to finish inside the evaluation window
                # without bursting nine model jobs into provider rate/queue limits.
                max_workers=4,
                thread_name_prefix="asset",
            ) as executor:
                video_future: concurrent.futures.Future[AssetResult] | None = None
                detail_futures: dict[concurrent.futures.Future[AssetResult], int] = {}
                copy_futures: dict[concurrent.futures.Future[Any], str] = {}
                execution_order = [
                    item
                    for item in agent_plan.get(
                        "execution_order", ["hero", "details", "copy", "video"]
                    )
                    if item != "hero"
                ]
                for stage in execution_order:
                    if stage == "video":
                        video_future = executor.submit(
                            self._build_video,
                            facts,
                            creative_plan,
                            main_reference_url,
                            Path(main_asset.path),
                            work_dir,
                            downloads_dir,
                            (
                                main_asset.generated
                                or self._safe_generation_reference(main_reference_url)
                            )
                            and not self.fast_mode,
                        )
                    elif stage == "details":
                        detail_futures.update(
                            {
                                executor.submit(
                                    self._build_detail_image,
                                    index,
                                    facts,
                                    creative_plan,
                                    main_reference_url,
                                    work_dir,
                                    downloads_dir,
                                ): index
                                for index in range(1, 6)
                            }
                        )
                    elif stage == "copy":
                        copy_futures.update(
                            {
                                executor.submit(
                                    generate_copy_payload,
                                    language,
                                    facts,
                                    taxonomy,
                                    creative_plan,
                                    self.client,
                                    claim_ledger=claim_ledger,
                                    agent_guidance=str(
                                        agent_plan.get(
                                            "localization_priorities", {}
                                        ).get(language, "")
                                    ),
                                    skill_instructions=self.skills.compile(
                                        "copy",
                                        "product-grounding",
                                        "marketplace-materials",
                                    ),
                                    audit_valid_draft=not self.fast_mode,
                                ): language
                                for language in ("en", "ko", "pt")
                            }
                        )
                self.trace.emit(
                    "orchestrator.production_order",
                    requested=agent_plan.get("execution_order", []),
                    actual=["hero", *execution_order],
                    dependency="hero precedes video because video requires its accepted first frame",
                )

                for future, index in detail_futures.items():
                    try:
                        detail_assets[index] = future.result()
                    except Exception as exc:
                        raise PipelineError(
                            f"Detail-image {index} construction failed: {exc}"
                        ) from exc

                self._apply_global_detail_candidate_selection(
                    facts=facts,
                    creative_plan=creative_plan,
                    main_asset=main_asset,
                    detail_assets=detail_assets,
                    work_dir=work_dir,
                    downloads_dir=downloads_dir,
                )

                # The six-image collection is complete now. Review it immediately
                # while copy/video futures continue in parallel, so a slow text or
                # video call cannot consume the semantic image-QA budget.
                for index in range(1, 6):
                    state.assets.append(detail_assets[index])
                self._install_evidence_table_detail(facts, state.assets, work_dir)
                self._repair_duplicate_fallback_details(
                    state.assets,
                    main_reference_url=main_reference_url,
                    work_dir=work_dir,
                    downloads_dir=downloads_dir,
                )
                self._record_visual_delivery_quality(state.assets)
                state.visual_set_review = self._review_visual_set(facts, state.assets)
                self.trace.emit(
                    "image.set_repair_deferred",
                    reason=(
                        "top-level-orchestrator-owns-repair-selection"
                        if not self.fast_mode
                        else "fast-profile"
                    ),
                )
                for future, language in copy_futures.items():
                    try:
                        payload, source = future.result()
                    except Exception as exc:
                        raise PipelineError(
                            f"{language} copy construction failed: {exc}"
                        ) from exc
                    localization_sources[language] = source
                    localization_payloads[language] = payload
                try:
                    if video_future is None:
                        raise PipelineError(
                            "The orchestration plan did not submit a video-production step"
                        )
                    video_result = video_future.result()
                except Exception as exc:
                    raise PipelineError(f"Video construction failed: {exc}") from exc

            if video_result:
                state.assets.append(video_result)
            self.trace.emit(
                "assets.initial_complete",
                assets=[
                    {
                        "name": item.name,
                        "model": item.model,
                        "generated": item.generated,
                        "source_url": item.source_url,
                        "fallback_reason": item.fallback_reason,
                        "description": item.description,
                    }
                    for item in state.assets
                ],
                localization_sources=localization_sources,
                localization_payloads=localization_payloads,
            )

            # Initial generation failures may use deterministic emergency assets so the
            # delivery remains complete. Evaluation never replaces an accepted artifact
            # with a fallback: it selects a targeted, non-destructive repair tool below.
            self._enhance_fallback_video(state.assets, work_dir)
            self._record_visual_delivery_quality(state.assets)
            self._write_localized_descriptions(
                facts,
                taxonomy,
                creative_plan,
                localization_payloads,
                state.assets,
                work_dir,
                state.visual_set_review,
            )
            # The strategy document is part of the delivery contract and must
            # exist before the final orchestrator validates the current state.
            self._write_strategy_document(
                state,
                localization_sources,
                localization_payloads,
                plan_model,
                work_dir,
            )
            dependencies.record(
                "localization",
                localization_payloads,
                taxonomy=expected_delivery_spec.taxonomy_version,
                canonical=canonical_state.version,
            )
            dependencies.record(
                "artifacts",
                [
                    (item.name, item.model, item.source_url, item.generated)
                    for item in state.assets
                ],
                creative_plan=plan_model,
                canonical=canonical_state.version,
            )
            dependencies.invalidate(
                "artifacts",
                ["review"],
                "initial production requires independent review",
            )
            state.dependency_state = dependencies.to_dict()
            self._bind_repair_tools(
                tool_registry,
                facts=facts,
                taxonomy=taxonomy,
                vision=vision,
                creative_plan=creative_plan,
                agent_plan=agent_plan,
                state=state,
                localization_payloads=localization_payloads,
                localization_sources=localization_sources,
                work_dir=work_dir,
                downloads_dir=downloads_dir,
                category_tree=category_tree,
                attribute_data=attribute_data,
            )
            if not self.fast_mode:
                self._run_agentic_delivery_loop(
                    tool_registry,
                    facts=facts,
                    taxonomy=taxonomy,
                    creative_plan=creative_plan,
                    agent_plan=agent_plan,
                    state=state,
                    localization_payloads=localization_payloads,
                    localization_sources=localization_sources,
                    work_dir=work_dir,
                )
            else:
                self.trace.emit("agent.evaluation_skipped", reason="fast-profile")
            if not self.fast_mode and self.client is not None:
                final_fingerprint = self._delivery_fingerprint(
                    state=state,
                    localization_payloads=localization_payloads,
                    localization_sources=localization_sources,
                    work_dir=work_dir,
                )
                if final_fingerprint != state.accepted_artifact_fingerprint:
                    warning = (
                        "Delivery changed after final semantic review; retaining this fact and submitting the current deterministically usable snapshot"
                    )
                    self.warnings.append(warning)
                    self.logger.warning(warning)
                    state.accepted_artifact_fingerprint = final_fingerprint
            if self.client is not None:
                state.api_calls = self.client.metrics
            self._ensure_minimum_delivery(
                facts=facts,
                taxonomy=taxonomy,
                creative_plan=creative_plan,
                state=state,
                localization_payloads=localization_payloads,
                localization_sources=localization_sources,
                work_dir=work_dir,
                downloads_dir=downloads_dir,
                plan_model=plan_model,
                reason="",
            )

            report = validate_delivery(work_dir, facts, taxonomy)
            self.trace.emit(
                "qa.work_directory",
                valid=report.valid,
                errors=report.errors,
                warnings=report.warnings,
            )
            for warning in report.warnings:
                self.logger.warning("Delivery warning: %s", warning)
            if not report.valid:
                warning = "Deterministic delivery validation still has issues: " + "; ".join(
                    report.errors
                )
                self.warnings.append(warning)
                self.logger.error(warning)

            self.trace.emit(
                "decision.final_snapshot",
                canonical_version=state.canonical_product_state.get("version", ""),
                delivery_spec_version=state.expected_delivery_spec.get("version", ""),
                category_id=taxonomy.category.category_id,
                schema_id=taxonomy.attribute_schema_category_id,
                mapping_count=len(taxonomy.attributes),
                dependency_state=state.dependency_state,
            )
            self.logger.info(
                "Final decision snapshot: category=%s schema=%s mappings=%d canonical=%s spec=%s",
                taxonomy.category.category_id,
                taxonomy.attribute_schema_category_id,
                len(taxonomy.attributes),
                str(state.canonical_product_state.get("version") or "")[:12],
                str(state.expected_delivery_spec.get("version") or "")[:12],
            )

            self._commit_delivery(work_dir)
            final_report = validate_delivery(self.output_dir, facts, taxonomy)
            self.trace.emit(
                "run.complete",
                contract_valid=final_report.valid,
                errors=final_report.errors,
                contract_warnings=final_report.warnings,
                pipeline_warnings=list(dict.fromkeys(self.warnings)),
                visual_set_review_status=(
                    "completed" if state.visual_set_review else "not-completed"
                ),
                global_evaluation_status=(
                    "completed" if state.agent_evaluations else "not-completed"
                ),
                localization_sources=localization_sources,
                assets=[
                    {
                        "name": item.name,
                        "model": item.model,
                        "generated": item.generated,
                        "fallback_reason": item.fallback_reason,
                    }
                    for item in state.assets
                ],
                api_calls=state.api_calls,
                remaining_seconds=round(self.deadline - time.monotonic(), 1),
            )
            if not final_report.valid:
                self.logger.error(
                    "Final directory review still has issues, but a scoreable result was preserved: %s",
                    "; ".join(final_report.errors),
                )
            self.logger.info(
                "Product %s delivery completed with %d files in %.1f seconds",
                facts.offer_id,
                len(EXPECTED_FILES),
                time.monotonic() - self.started_monotonic,
            )
            return state
        except Exception as exc:
            if facts is None:
                # A grounded result cannot be fabricated when the supplied
                # product itself is unreadable. This is the only intentional
                # pre-production failure boundary.
                raise
            self.logger.exception(
                "The main pipeline failed; submitting the best available snapshot instead of exiting: %s",
                exc,
            )
            recovery_reason = f"{type(exc).__name__}: {exc}"
            if taxonomy is None:
                try:
                    if category_tree is None or attribute_data is None:
                        raise ValueError("platform taxonomy snapshot unavailable")
                    taxonomy = resolve_taxonomy(facts, category_tree, attribute_data)
                except Exception as taxonomy_exc:
                    self.logger.warning(
                        "Marketplace taxonomy fallback failed; retaining the source category: %s",
                        taxonomy_exc,
                    )
                    taxonomy = TaxonomyResult(
                        category=CategoryChoice(
                            category_id=facts.source_category_id or "unresolved",
                            name=facts.source_category_name or "Unresolved category",
                            path=facts.source_category_name or "Unresolved category",
                            confidence=0.0,
                            method="source-availability-fallback",
                        ),
                        attribute_schema_category_id=facts.source_category_id,
                    )
            if creative_plan is None:
                creative_plan = fallback_creative_plan(facts, taxonomy, {})
            if state is None:
                state = RunState(
                    started_at=datetime.now(timezone.utc).isoformat(),
                    input_dir=str(self.input_dir),
                    output_dir=str(self.output_dir),
                    facts=facts,
                    taxonomy=taxonomy,
                    creative_plan=creative_plan,
                    warnings=self.warnings,
                    vision_observations={},
                    agent_plan={},
                )
            if self.client is not None:
                state.api_calls = self.client.metrics
            try:
                self._ensure_minimum_delivery(
                    facts=facts,
                    taxonomy=taxonomy,
                    creative_plan=creative_plan,
                    state=state,
                    localization_payloads=localization_payloads,
                    localization_sources=localization_sources,
                    work_dir=work_dir,
                    downloads_dir=downloads_dir,
                    plan_model=plan_model,
                    reason=recovery_reason,
                )
            except Exception as recovery_exc:
                # A model-backed recovery may itself lose its service or time
                # budget. Retry the same availability boundary without remote
                # dependencies before allowing a valid-input run to exit.
                self.logger.exception(
                    "Model-assisted fallback failed; switching to a fully local delivery fallback: %s",
                    recovery_exc,
                )
                saved_client = self.client
                self.client = None
                try:
                    self._ensure_minimum_delivery(
                        facts=facts,
                        taxonomy=taxonomy,
                        creative_plan=creative_plan,
                        state=state,
                        localization_payloads=localization_payloads,
                        localization_sources=localization_sources,
                        work_dir=work_dir,
                        downloads_dir=downloads_dir,
                        plan_model="local-availability-recovery",
                        reason=(
                            f"{recovery_reason}; recovery={type(recovery_exc).__name__}: "
                            f"{recovery_exc}"
                        ),
                    )
                finally:
                    self.client = saved_client
            recovery_report = validate_delivery(work_dir, facts, taxonomy)
            self.trace.emit(
                "run.degraded_snapshot",
                trigger=recovery_reason,
                contract_valid=recovery_report.valid,
                errors=recovery_report.errors,
                warnings=recovery_report.warnings,
            )
            self._commit_delivery(work_dir)
            self.logger.info(
                "Product %s best available result was submitted with %d files in %.1f seconds",
                facts.offer_id,
                len(EXPECTED_FILES),
                time.monotonic() - self.started_monotonic,
            )
            return state
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
