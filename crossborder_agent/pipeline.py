"""End-to-end bounded agent orchestration."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_tools import BoundedToolRegistry, ToolExecution, ToolSpec
from .api import ApiConfig, ApiError, HttpJsonClient, QwenClient
from .bounded_agent import BoundedDeliveryAgent
from .claims import build_claim_ledger, filter_invalid_mapping_provenance
from .compliance import normalize_source_image_observations
from .debug_trace import DebugTrace
from .input_loader import discover_input_files, load_json, load_product_facts
from .localization import generate_copy_payload, render_description
from .media import (
    MediaError,
    create_catalog_video,
    create_size_chart_image,
    create_slideshow_video,
    hash_distance,
    inspect_image_quality,
    inspect_video,
    normalize_image,
    strip_video_audio,
)
from .models import (
    AgentActionResult,
    AssetResult,
    CreativePlan,
    ProductFacts,
    RunState,
    SizeChartRow,
    TaxonomyResult,
)
from .planning import create_creative_plan
from .qa import EXPECTED_FILES, validate_delivery
from .skill_runtime import SkillLibrary
from .taxonomy import resolve_taxonomy


class PipelineError(RuntimeError):
    """Raised when the agent cannot produce a complete, validated delivery."""


class SemanticRejection(ApiError):
    """All candidates contain a concrete product-identity or compliance defect."""

    def __init__(self, message: str, *, feedback: str = ""):
        super().__init__(message, retryable=True, category="semantic_rejection")
        self.feedback = feedback or message


_IMAGE_NEGATIVE_PROMPT = (
    "written text, letters, numbers, watermark, logo, brand mark, price tag, promotional badge, "
    "unreadable typography, distorted anatomy, extra limbs, malformed hands, product deformation, "
    "changed buttons, changed fasteners, changed pattern, changed color, blur, low resolution"
)

_SINGLE_COMPOSITION_NEGATIVE_PROMPT = (
    ", split screen, inset panel, repeated panel, mixed close-up and full-product composition"
)

_MAIN_NEGATIVE_PROMPT = (
    _IMAGE_NEGATIVE_PROMPT
    + ", collage, montage, split screen, inset, duplicate garment, multiple products, multiple colorways, "
    "cropped product, person, mannequin body"
)

_VIDEO_NEGATIVE_PROMPT = (
    "product morphing, changed product construction, changed color, changed pattern, added or removed components, "
    "changed fastenings or trims, duplicate product, extra product, warped material, flicker, scene cut, camera shake, "
    "hands covering product, text, subtitles, watermark, logo animation, speech, music"
)


def _reviewed_media_description(
    review: dict[str, Any] | None,
    name: str,
    language: str,
    stale_names: set[str],
) -> str:
    """Use a caption only when it describes the current, accepted final asset."""

    if not review or name in stale_names:
        return ""
    rows = review.get("assets")
    if not isinstance(rows, list):
        return ""
    row = next(
        (
            item
            for item in rows
            if isinstance(item, dict) and item.get("name") == name
        ),
        None,
    )
    if not isinstance(row, dict) or row.get("usable") is not True:
        return ""
    if any(
        row.get(field) is False
        for field in ("identity_consistent", "construction_consistent", "color_consistent")
    ):
        return ""
    if row.get("description_confidence") not in {"high", "medium"}:
        return ""
    descriptions = row.get("media_descriptions")
    value = descriptions.get(language) if isinstance(descriptions, dict) else ""
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if not 12 <= len(value) <= 300 or re.search(r"[\u4e00-\u9fff]", value):
        return ""
    if language == "ko" and not re.search(r"[\uac00-\ud7a3]", value):
        return ""
    if language in {"en", "pt"} and re.search(r"[\uac00-\ud7a3]", value):
        return ""
    return value


class Pipeline:
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
            missing = [name for name in required if not os.environ.get(name, "").strip()]
            raise PipelineError(
                "非离线运行缺少模型配置: "
                + ", ".join(missing)
                + "；开发降级测试请显式传入 --offline"
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
        self._source_image_observations: dict[str, dict[str, Any]] = {}
        self._source_selection_warnings: set[str] = set()
        self._size_chart_source_url = ""

    def _ensure_time(self, reserve_seconds: float = 0) -> None:
        remaining = self.deadline - time.monotonic()
        if remaining <= reserve_seconds:
            raise PipelineError(f"剩余运行时间不足，需要保留 {reserve_seconds:.0f} 秒")

    @staticmethod
    def _create_tool_registry(
        *, protect_size_chart: bool = False
    ) -> BoundedToolRegistry:
        registry = BoundedToolRegistry()
        registry.add_spec(
            ToolSpec(
                name="regenerate_main_image",
                description="Regenerate and reselect the hero from trusted references; preserve the current hero unless the revision succeeds.",
                targets=("main_image.jpeg",),
                estimated_seconds=150,
                side_effects="replaces main_image.jpeg only after a candidate is downloaded and validated",
            )
        )
        registry.add_spec(
            ToolSpec(
                name="regenerate_detail_image",
                description="Regenerate one detail storyboard slot with evaluator-specific corrections; preserve the current slot on failure.",
                targets=tuple(
                    f"detail_image_{index}.jpeg"
                    for index in range(1, 5 if protect_size_chart else 6)
                ),
                estimated_seconds=120,
                side_effects="replaces only the named detail image after candidate acceptance",
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
            )
        )
        registry.add_spec(
            ToolSpec(
                name="regenerate_video",
                description="Regenerate the product video with a targeted temporal correction; preserve the current playable video on failure.",
                targets=("product_video.mp4",),
                estimated_seconds=210,
                side_effects="replaces product_video.mp4 only after download, audio stripping, and playback validation",
            )
        )
        return registry

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
            taxonomy = resolve_taxonomy(facts, category_tree, attribute_data)
            if not self.fast_mode:
                taxonomy = self._adjudicate_taxonomy(
                    facts, taxonomy, category_tree, attribute_data
                )
            else:
                self.trace.emit(
                    "taxonomy.adjudication_skipped", reason="fast-profile"
                )
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
                "商品 %s: 类目 %s %s (%.2f, %s)",
                facts.offer_id,
                taxonomy.category.category_id,
                taxonomy.category.name,
                taxonomy.category.confidence,
                taxonomy.category.method,
            )

            vision = self._analyze_source_images(facts)
            self.trace.emit("vision.source_review", result=vision)
            self._apply_size_chart_observations(facts, vision)
            claim_ledger = build_claim_ledger(facts, taxonomy, vision)
            self.trace.emit(
                "claims.ledger",
                count=len(claim_ledger),
                publishable_count=sum(
                    "buyer_copy" in item.allowed_surfaces for item in claim_ledger
                ),
            )
            tool_registry = self._create_tool_registry(
                protect_size_chart=bool(facts.size_chart_rows)
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
            self.logger.info("创意计划来源: %s", plan_model)
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
            )

            main_asset, main_reference_url = self._build_main_image(
                facts, creative_plan, vision, work_dir, downloads_dir
            )
            state.assets.append(main_asset)

            localization_sources: dict[str, str] = {}
            localization_payloads: dict[str, dict[str, Any]] = {}
            detail_assets: dict[int, AssetResult] = {}
            video_result: AssetResult | None = None

            with concurrent.futures.ThreadPoolExecutor(
                # Keep enough parallelism to finish inside the evaluation window
                # without bursting nine model jobs into provider rate/queue limits.
                max_workers=4, thread_name_prefix="asset"
            ) as executor:
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
                detail_futures = {
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
                copy_futures = {
                    executor.submit(
                        generate_copy_payload,
                        language,
                        facts,
                        taxonomy,
                        creative_plan,
                        self.client,
                        claim_ledger=claim_ledger,
                        agent_guidance=str(
                            agent_plan.get("localization_priorities", {}).get(
                                language, ""
                            )
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

                for future, index in detail_futures.items():
                    try:
                        detail_assets[index] = future.result()
                    except Exception as exc:
                        raise PipelineError(f"详情图 {index} 构建失败: {exc}") from exc

                # The six-image collection is complete now. Review it immediately
                # while copy/video futures continue in parallel, so a slow text or
                # video call cannot consume the semantic image-QA budget.
                for index in range(1, 6):
                    state.assets.append(detail_assets[index])
                self._install_size_chart_detail(facts, state.assets, work_dir)
                self._repair_duplicate_fallback_details(
                    state.assets,
                    main_reference_url=main_reference_url,
                    work_dir=work_dir,
                    downloads_dir=downloads_dir,
                )
                self._record_visual_delivery_quality(state.assets)
                state.visual_set_review = self._review_visual_set(
                    facts, state.assets
                )
                if not self.fast_mode:
                    state.visual_set_review = self._repair_visual_set_once(
                        facts=facts,
                        creative_plan=creative_plan,
                        state=state,
                        review=state.visual_set_review,
                        work_dir=work_dir,
                        downloads_dir=downloads_dir,
                    )
                else:
                    self.trace.emit(
                        "image.set_repair_skipped", reason="fast-profile"
                    )
                for future, language in copy_futures.items():
                    try:
                        payload, source = future.result()
                    except Exception as exc:
                        raise PipelineError(f"{language} 文案构建失败: {exc}") from exc
                    localization_sources[language] = source
                    localization_payloads[language] = payload
                try:
                    video_result = video_future.result()
                except Exception as exc:
                    raise PipelineError(f"视频构建失败: {exc}") from exc

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
            )
            post_review_action_start = len(state.agent_actions)
            if not self.fast_mode:
                self._run_bounded_agent_loop(
                    tool_registry,
                    facts=facts,
                    taxonomy=taxonomy,
                    creative_plan=creative_plan,
                    agent_plan=agent_plan,
                    state=state,
                    localization_payloads=localization_payloads,
                    localization_sources=localization_sources,
                    visual_set_review=state.visual_set_review,
                    work_dir=work_dir,
                )
            else:
                self.trace.emit(
                    "agent.evaluation_skipped", reason="fast-profile"
                )
            # A catalog video is rebuilt from the final image set only when video
            # generation was unavailable from the outset; this is not an evaluator fallback.
            self._enhance_fallback_video(state.assets, work_dir)
            self._record_visual_delivery_quality(state.assets)
            stale_review_assets = {
                item.target
                for item in state.agent_actions[post_review_action_start:]
                if item.status == "completed"
                and item.tool in {"regenerate_main_image", "regenerate_detail_image"}
            }
            self._write_localized_descriptions(
                facts,
                taxonomy,
                creative_plan,
                localization_payloads,
                state.assets,
                work_dir,
                state.visual_set_review,
                stale_review_assets,
            )
            if self.client is not None:
                state.api_calls = self.client.metrics
            self._write_strategy_document(
                state,
                localization_sources,
                localization_payloads,
                plan_model,
                work_dir,
            )

            report = validate_delivery(work_dir, facts, taxonomy)
            self.trace.emit(
                "qa.work_directory",
                valid=report.valid,
                errors=report.errors,
                warnings=report.warnings,
            )
            for warning in report.warnings:
                self.logger.warning("交付告警: %s", warning)
            if not report.valid:
                raise PipelineError("交付校验失败: " + "; ".join(report.errors))

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
                raise PipelineError(
                    "最终目录复核失败: " + "; ".join(final_report.errors)
                )
            self.logger.info(
                "商品 %s 交付完成，共 %d 个文件，用时 %.1f 秒",
                facts.offer_id,
                len(EXPECTED_FILES),
                time.monotonic() - self.started_monotonic,
            )
            return state
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _write_localized_descriptions(
        self,
        facts: ProductFacts,
        taxonomy: TaxonomyResult,
        creative_plan: CreativePlan,
        payloads: dict[str, dict[str, Any]],
        assets: list[AssetResult],
        work_dir: Path,
        visual_set_review: dict[str, Any] | None = None,
        stale_review_assets: set[str] | None = None,
    ) -> None:
        asset_by_name = {asset.name: asset for asset in assets}
        fallback_templates = {
            "en": {
                "main": "Seller-source hero image normalized to a square listing format.",
                "details": [
                    "Alternate seller-source view showing the complete product.",
                    "Seller-source view showing a directly visible product detail.",
                    "Seller-source view showing a different directly visible product detail.",
                    "Seller-source alternate view supported by the supplied photography.",
                    "Detail crop derived from the seller's product photography.",
                ],
                "crops": {
                    "upper": "Seller-source close-up of source-visible details in the upper image area.",
                    "lower": "Seller-source close-up of source-visible details in the lower image area.",
                    "left": "Seller-source close-up of source-visible details on the left side.",
                    "right": "Seller-source close-up of source-visible details on the right side.",
                    "center": "Seller-source close-up of source-visible details in the center.",
                },
                "video": "Eight-second silent catalog video assembled from the final distinct product images.",
                "single_video": "Eight-second product presentation with restrained camera motion.",
                "size_chart": "Size chart showing the seller-provided garment measurements and weight guidance.",
            },
            "ko": {
                "main": "판매자 원본을 정사각형 등록 규격에 맞춰 정리한 대표 이미지입니다.",
                "details": [
                    "상품 전체를 보여 주는 판매자 원본의 다른 이미지입니다.",
                    "원본에서 직접 확인되는 상품 디테일을 보여 주는 이미지입니다.",
                    "원본에서 직접 확인되는 다른 상품 디테일을 보여 주는 이미지입니다.",
                    "판매자 사진으로 확인된 다른 시점의 상품 이미지입니다.",
                    "판매자 상품 사진에서 잘라낸 디테일 이미지입니다.",
                ],
                "crops": {
                    "upper": "원본 이미지 상단에서 직접 확인되는 디테일을 확대한 이미지입니다.",
                    "lower": "원본 이미지 하단에서 직접 확인되는 디테일을 확대한 이미지입니다.",
                    "left": "원본 이미지 왼쪽에서 직접 확인되는 디테일을 확대한 이미지입니다.",
                    "right": "원본 이미지 오른쪽에서 직접 확인되는 디테일을 확대한 이미지입니다.",
                    "center": "원본 이미지 중앙에서 직접 확인되는 디테일을 확대한 이미지입니다.",
                },
                "video": "서로 다른 최종 상품 이미지로 구성한 8초 무음 카탈로그 영상입니다.",
                "single_video": "절제된 카메라 움직임을 적용한 8초 단일 이미지 상품 영상입니다.",
                "size_chart": "판매자가 제공한 의류 실측과 권장 체중을 보여 주는 사이즈표입니다.",
            },
            "pt": {
                "main": "Imagem principal da fonte do vendedor adaptada ao formato quadrado do anúncio.",
                "details": [
                    "Outra foto da fonte do vendedor mostrando o produto por inteiro.",
                    "Foto da fonte mostrando um detalhe diretamente visível do produto.",
                    "Foto da fonte mostrando outro detalhe diretamente visível do produto.",
                    "Vista alternativa confirmada pelas fotos fornecidas pelo vendedor.",
                    "Recorte de detalhe derivado das fotos de produto do vendedor.",
                ],
                "crops": {
                    "upper": "Close de detalhes visíveis na área superior da foto do vendedor.",
                    "lower": "Close de detalhes visíveis na área inferior da foto do vendedor.",
                    "left": "Close de detalhes visíveis no lado esquerdo da foto do vendedor.",
                    "right": "Close de detalhes visíveis no lado direito da foto do vendedor.",
                    "center": "Close de detalhes visíveis no centro da foto do vendedor.",
                },
                "video": "Vídeo de catálogo silencioso de 8 segundos montado com as imagens finais distintas do produto.",
                "single_video": "Apresentação de 8 segundos com uma única imagem e movimento de câmera discreto.",
                "size_chart": "Tabela com as medidas da peça e o peso indicados pelo vendedor.",
            },
        }
        generated_templates = {
            "en": {
                "main": "Clean studio hero showing one complete product.",
                "video": "Eight-second product presentation based on the final hero image.",
                "roles": {
                    "complete_product": "Complete three-quarter view showing the full product.",
                    "primary_verified_detail": "Close view of the primary detail verified in the source images.",
                    "secondary_verified_detail": "Close view of a different detail verified in the source images.",
                    "verified_variants": "Catalog view comparing only seller-verified color variants.",
                    "verified_alternate_view": "Alternate view using only source-supported product information.",
                    "verified_use_context": "Source-supported practical use view with the complete product visible.",
                    "product_only_context": "Product-only view showing a practical, neutral context.",
                },
            },
            "ko": {
                "main": "상품 한 개의 전체 형태를 보여 주는 깔끔한 스튜디오 대표 이미지입니다.",
                "video": "최종 대표 이미지를 바탕으로 제작한 8초 상품 영상입니다.",
                "roles": {
                    "complete_product": "상품 전체를 보여 주는 완전한 3/4 각도 이미지입니다.",
                    "primary_verified_detail": "원본에서 확인된 핵심 디테일을 가까이 보여 줍니다.",
                    "secondary_verified_detail": "원본에서 확인된 다른 디테일을 가까이 보여 줍니다.",
                    "verified_variants": "판매자 원본에서 확인된 색상 옵션만 비교한 카탈로그 이미지입니다.",
                    "verified_alternate_view": "원본 근거만 사용한 다른 시점의 상품 이미지입니다.",
                    "verified_use_context": "상품 전체가 보이는 원본 근거 기반의 실용적 사용 이미지입니다.",
                    "product_only_context": "실용적이고 중립적인 맥락의 상품 전용 이미지입니다.",
                },
            },
            "pt": {
                "main": "Imagem principal de estúdio mostrando uma única peça por inteiro.",
                "video": "Apresentação de 8 segundos baseada na imagem principal final.",
                "roles": {
                    "complete_product": "Vista completa em três quartos mostrando todo o produto.",
                    "primary_verified_detail": "Close do principal detalhe confirmado nas imagens de origem.",
                    "secondary_verified_detail": "Close de outro detalhe confirmado nas imagens de origem.",
                    "verified_variants": "Vista de catálogo comparando apenas cores confirmadas pelo vendedor.",
                    "verified_alternate_view": "Vista alternativa usando apenas informações confirmadas na origem.",
                    "verified_use_context": "Vista de uso confirmada na origem com o produto inteiro visível.",
                    "product_only_context": "Composição sem modelo em um contexto prático e neutro.",
                },
            },
        }
        for language, payload in payloads.items():
            stale_names = stale_review_assets or set()
            media = payload.get("media_descriptions")
            if not isinstance(media, dict):
                media = {}
                payload["media_descriptions"] = media
            for name in (
                "main_image.jpeg",
                "detail_image_1.jpeg",
                "detail_image_2.jpeg",
                "detail_image_3.jpeg",
                "detail_image_4.jpeg",
                "detail_image_5.jpeg",
                "product_video.mp4",
            ):
                asset = asset_by_name.get(name)
                if asset is None:
                    continue
                reviewed_description = (
                    ""
                    if name == "product_video.mp4"
                    or asset.model == "deterministic-size-chart"
                    else _reviewed_media_description(
                        visual_set_review,
                        name,
                        language,
                        stale_names,
                    )
                )
                if reviewed_description:
                    media[name] = reviewed_description
                    continue
                if asset.generated:
                    if name == "main_image.jpeg":
                        media[name] = generated_templates[language]["main"]
                    elif name == "product_video.mp4":
                        media[name] = generated_templates[language]["video"]
                    else:
                        try:
                            detail_index = int(
                                name.removeprefix("detail_image_").split(".", 1)[0]
                            )
                        except ValueError:
                            detail_index = 1
                        role = (
                            creative_plan.detail_roles[detail_index - 1]
                            if detail_index <= len(creative_plan.detail_roles)
                            else "complete_product"
                        )
                        media[name] = generated_templates[language]["roles"].get(
                            role,
                            generated_templates[language]["roles"]["complete_product"],
                        )
                    continue
                if asset.model == "deterministic-size-chart":
                    kind = "size_chart"
                elif name == "product_video.mp4":
                    kind = (
                        "video"
                        if asset.model == "ffmpeg-catalog-fallback"
                        else "single_video"
                    )
                elif name == "main_image.jpeg":
                    kind = "main"
                else:
                    try:
                        detail_index = int(
                            name.removeprefix("detail_image_").split(".", 1)[0]
                        )
                    except ValueError:
                        detail_index = 1
                    crop_kind = next(
                        (
                            kind
                            for kind in ("upper", "lower", "left", "right", "center")
                            if kind in asset.description.casefold()
                        ),
                        "",
                    )
                    media[name] = (
                        fallback_templates[language]["crops"][crop_kind]
                        if crop_kind
                        else fallback_templates[language]["details"][
                            max(0, min(4, detail_index - 1))
                        ]
                    )
                    continue
                media[name] = fallback_templates[language][kind]
            description = render_description(language, payload, facts, taxonomy)
            (work_dir / f"product_description_{language}.md").write_text(
                description, encoding="utf-8"
            )

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

    def _run_bounded_agent_loop(
        self,
        registry: BoundedToolRegistry,
        *,
        facts: ProductFacts,
        taxonomy: TaxonomyResult,
        creative_plan: CreativePlan,
        agent_plan: dict[str, Any],
        state: RunState,
        localization_payloads: dict[str, dict[str, Any]],
        localization_sources: dict[str, str],
        visual_set_review: dict[str, Any],
        work_dir: Path,
    ) -> None:
        if self.client is None:
            self.warnings.append(
                "显式离线模式：跳过有界 Agent 全局评估循环"
                if self.offline
                else "模型配置不可用，跳过有界 Agent 全局评估循环"
            )
            self.trace.emit(
                "agent.evaluation_skipped",
                reason="offline" if self.offline else "no-model-client",
            )
            return
        max_repairs = int(agent_plan.get("max_repair_rounds", 1))
        repairs_used = 0
        round_index = 0
        while True:
            if self.deadline - time.monotonic() <= 4 * 60:
                self.warnings.append("剩余时间不足，停止新的 Agent 评估轮次并保留当前版本")
                self.trace.emit(
                    "agent.evaluation_skipped",
                    reason="insufficient-stage-budget",
                    round_index=round_index,
                    remaining_seconds=round(self.deadline - time.monotonic(), 1),
                )
                break
            evaluation = self.agent.evaluate_delivery(
                round_index=round_index,
                facts=facts,
                taxonomy=taxonomy,
                creative_plan=creative_plan,
                agent_plan=agent_plan,
                assets=state.assets,
                localization_payloads=localization_payloads,
                localization_sources=localization_sources,
                visual_set_review=visual_set_review,
                work_dir=work_dir,
                tools=registry,
            )
            if evaluation is None:
                self.warnings.append("LLM 全局评估未完成；不触发回退，保留当前已校验素材")
                self.trace.emit(
                    "agent.evaluation_failed",
                    round_index=round_index,
                    reason="reviewer-unavailable-or-invalid-response",
                )
                break
            state.agent_evaluations.append(evaluation)
            self.trace.emit(
                "agent.evaluation",
                round_index=round_index,
                ready=evaluation.ready_for_delivery,
                weighted_score=evaluation.weighted_score,
                dimension_scores=evaluation.dimension_scores,
                summary=evaluation.summary,
                issues=evaluation.issues,
                repair_actions=[
                    {
                        "tool": item.tool,
                        "target": item.target,
                        "instruction": item.instruction,
                        "reason": item.reason,
                        "dimension": item.dimension,
                        "priority": item.priority,
                    }
                    for item in evaluation.repair_actions
                ],
            )
            self.logger.info(
                "Agent 全局评估轮次 %d: score=%.1f ready=%s actions=%d",
                round_index,
                evaluation.weighted_score,
                evaluation.ready_for_delivery,
                len(evaluation.repair_actions),
            )
            if evaluation.ready_for_delivery or not evaluation.repair_actions:
                break
            if repairs_used >= max_repairs:
                self.warnings.append("已达到有界 Agent 修复轮次上限，保留最后成功版本")
                break

            completed = 0
            for action in evaluation.repair_actions:
                required = registry.estimated_seconds(action.tool) + 90
                if self.deadline - time.monotonic() <= required:
                    self.warnings.append(
                        f"剩余时间不足以安全执行 {action.tool} 并保留最终校验预算"
                    )
                    break
                result = registry.execute(
                    action.tool, action.target, action.instruction
                )
                state.agent_actions.append(
                    AgentActionResult(
                        round_index=round_index,
                        tool=action.tool,
                        target=action.target,
                        status=result.status,
                        detail=result.detail,
                    )
                )
                self.trace.emit(
                    "agent.repair_result",
                    round_index=round_index,
                    tool=action.tool,
                    target=action.target,
                    dimension=action.dimension,
                    instruction=action.instruction,
                    status=result.status,
                    detail=result.detail,
                    metadata=result.metadata,
                )
                if result.status == "completed":
                    completed += 1
                    self.logger.info(
                        "Agent 修复完成: %s/%s", action.tool, action.target
                    )
                else:
                    self.warnings.append(
                        f"Agent 修复未替换现有素材 {action.target}: {result.detail[:300]}"
                    )
            repairs_used += 1
            round_index += 1
            if completed == 0:
                break

    @staticmethod
    def _find_asset(assets: list[AssetResult], name: str) -> AssetResult:
        asset = next((item for item in assets if item.name == name), None)
        if asset is None:
            raise PipelineError(f"修复目标不存在: {name}")
        return asset

    def _repair_main_image(
        self,
        target: str,
        instruction: str,
        facts: ProductFacts,
        plan: CreativePlan,
        vision: dict[str, Any],
        assets: list[AssetResult],
        work_dir: Path,
        downloads_dir: Path,
    ) -> ToolExecution:
        if self.client is None:
            return ToolExecution("failed", "image model unavailable")
        source_urls = self._ordered_source_urls(facts, vision)
        if not source_urls:
            return ToolExecution("failed", "no trusted hero reference")
        asset = self._find_asset(assets, target)
        staged = work_dir / f".repair-main-{uuid.uuid4().hex}.jpeg"
        prompt = (
            plan.main_prompt
            + "\nIndependent evaluator correction for this revision: "
            + instruction
            + "\nCorrect only the identified defect and preserve all verified product features."
        )
        try:
            selected, model = self._generate_main_with_semantic_retry(
                facts,
                prompt,
                generation_references=source_urls[:1],
                review_references=source_urls[:3],
                incumbent_url=asset.source_url,
                minimum_improvement=6.0,
            )
            if asset.source_url and selected == asset.source_url:
                return ToolExecution(
                    "skipped", "hero revision did not beat the current asset by 6 points"
                )
            self._download_and_normalize(
                selected,
                staged,
                downloads_dir,
                canvas=(1600, 1600),
                white_background=True,
            )
            os.replace(staged, Path(asset.path))
            asset.source_url = selected
            asset.model = f"{model}-agent-repair"
            asset.generated = True
            asset.fallback_reason = ""
            asset.description = f"Agent-repaired hero: {instruction[:240]}"
            return ToolExecution("completed", "hero revision accepted")
        except (ApiError, MediaError, OSError, PipelineError) as exc:
            return ToolExecution("failed", f"hero revision rejected; prior hero preserved: {exc}")
        finally:
            staged.unlink(missing_ok=True)

    def _repair_detail_image(
        self,
        target: str,
        instruction: str,
        facts: ProductFacts,
        plan: CreativePlan,
        assets: list[AssetResult],
        work_dir: Path,
        downloads_dir: Path,
    ) -> ToolExecution:
        if self.client is None:
            return ToolExecution("failed", "image model unavailable")
        try:
            index = int(target.removeprefix("detail_image_").split(".", 1)[0])
        except ValueError:
            return ToolExecution("rejected", "invalid detail target")
        if index == 5 and facts.size_chart_rows:
            return ToolExecution(
                "skipped",
                "verified seller size chart is intentionally protected from generative replacement",
            )
        main_asset = self._find_asset(assets, "main_image.jpeg")
        references = self._detail_reference_selection(
            index, facts, main_asset.source_url
        )
        if not references:
            return ToolExecution("failed", "no trusted detail reference")
        asset = self._find_asset(assets, target)
        staged = work_dir / f".repair-detail-{index}-{uuid.uuid4().hex}.jpeg"
        prompt = (
            plan.detail_prompts[index - 1]
            + "\nIndependent evaluator correction for this revision: "
            + instruction
            + "\nCorrect only the identified defect; keep the intended slot and exact product identity."
        )
        try:
            selected, model = self._generate_detail_with_semantic_retry(
                index,
                facts,
                prompt,
                references=references[:3],
                incumbent_url=asset.source_url,
                minimum_improvement=6.0,
            )
            if asset.source_url and selected == asset.source_url:
                return ToolExecution(
                    "skipped",
                    f"detail slot {index} revision did not beat the current asset by 6 points",
                )
            self._download_and_normalize(
                selected,
                staged,
                downloads_dir,
                canvas=(1200, 1500),
                white_background=False,
            )
            os.replace(staged, Path(asset.path))
            asset.source_url = selected
            asset.model = f"{model}-agent-repair"
            asset.generated = True
            asset.fallback_reason = ""
            asset.description = (
                "Canonical detail role: "
                f"{plan.detail_roles[index - 1] if index <= len(plan.detail_roles) else f'slot_{index}'}"
            )
            return ToolExecution("completed", f"detail slot {index} revision accepted")
        except (ApiError, MediaError, OSError, PipelineError) as exc:
            return ToolExecution(
                "failed", f"detail revision rejected; prior slot preserved: {exc}"
            )
        finally:
            staged.unlink(missing_ok=True)

    def _repair_localized_copy(
        self,
        target: str,
        instruction: str,
        facts: ProductFacts,
        taxonomy: TaxonomyResult,
        plan: CreativePlan,
        agent_plan: dict[str, Any],
        payloads: dict[str, dict[str, Any]],
        sources: dict[str, str],
        work_dir: Path,
    ) -> ToolExecution:
        if self.client is None:
            return ToolExecution("failed", "chat model unavailable")
        language = target.removeprefix("product_description_").split(".", 1)[0]
        if language not in {"en", "ko", "pt"}:
            return ToolExecution("rejected", "unsupported locale target")
        incumbent = payloads.get(language)
        if not isinstance(incumbent, dict):
            return ToolExecution("failed", "current localized payload unavailable")
        candidate, source = generate_copy_payload(
            language,
            facts,
            taxonomy,
            plan,
            self.client,
            agent_guidance=str(
                agent_plan.get("localization_priorities", {}).get(language, "")
            ),
            revision_feedback=instruction,
            skill_instructions=self.skills.compile(
                "copy",
                "product-grounding",
                "marketplace-materials",
            ),
        )
        if not source.startswith(self.client.config.chat_model):
            return ToolExecution(
                "failed",
                f"localized revision did not pass model/schema audit ({source}); prior copy preserved",
            )
        if not self._copy_revision_improves(
            language, facts, incumbent, candidate, minimum_improvement=6.0
        ):
            return ToolExecution(
                "skipped",
                f"{language} revision did not beat current copy by 6 points; prior copy preserved",
            )
        try:
            rendered = render_description(language, candidate, facts, taxonomy)
            staged = work_dir / f".{target}.{uuid.uuid4().hex}.tmp"
            staged.write_text(rendered, encoding="utf-8")
            os.replace(staged, work_dir / target)
        except OSError as exc:
            return ToolExecution("failed", f"localized revision could not be installed: {exc}")
        payloads[language] = candidate
        sources[language] = f"{source}-agent-repair"
        return ToolExecution("completed", f"{language} copy revision accepted")

    def _copy_revision_improves(
        self,
        language: str,
        facts: ProductFacts,
        incumbent: dict[str, Any],
        candidate: dict[str, Any],
        *,
        minimum_improvement: float,
    ) -> bool:
        if self.client is None:
            return False
        system = (
            "You are a conservative cross-border listing A/B judge. Return JSON only. "
            "Compare factual completeness, source support, shopper-facing fluency, title quality, "
            "conversion usefulness and source-script contamination. Never reward invented claims."
        )
        prompt = f"""
Language: {language}
Candidate 0 is the current accepted copy. Candidate 1 is a proposed repair.
Return selected_index plus exactly two candidates containing index, score (0-100),
facts_supported, complete, native_and_natural, has_source_script_contamination, and reason.
Select candidate 1 only when it improves by at least {minimum_improvement:.0f} points and all its facts are supported.

Verified facts:
{json.dumps(facts.compact_dict(), ensure_ascii=False)}

Candidate 0:
{json.dumps(incumbent, ensure_ascii=False)}

Candidate 1:
{json.dumps(candidate, ensure_ascii=False)}
""".strip()
        try:
            review = self.client.chat_json(
                system, prompt, model=self.client.config.review_model
            )
        except ApiError as exc:
            self.logger.warning("%s 文案 A/B 评审不可用，保留旧文案: %s", language, exc)
            return False
        rows = review.get("candidates")
        if not isinstance(rows, list):
            return False
        by_index = {
            row.get("index"): row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("index"), int)
        }
        old = by_index.get(0)
        new = by_index.get(1)
        if not isinstance(old, dict) or not isinstance(new, dict):
            return False
        old_score = old.get("score")
        new_score = new.get("score")
        return bool(
            review.get("selected_index") == 1
            and isinstance(old_score, (int, float))
            and isinstance(new_score, (int, float))
            and float(new_score) >= float(old_score) + minimum_improvement
            and new.get("facts_supported") is True
            and new.get("complete") is not False
            and new.get("native_and_natural") is not False
            and new.get("has_source_script_contamination") is not True
        )

    def _repair_video(
        self,
        target: str,
        instruction: str,
        facts: ProductFacts,
        plan: CreativePlan,
        assets: list[AssetResult],
        work_dir: Path,
        downloads_dir: Path,
    ) -> ToolExecution:
        if self.client is None:
            return ToolExecution("failed", "video model unavailable")
        main_asset = self._find_asset(assets, "main_image.jpeg")
        first_frame_url = main_asset.source_url
        if not first_frame_url or not self._safe_generation_reference(first_frame_url):
            candidates = self._source_urls_for_use(
                self._fallback_source_urls(facts, asset_name="main_image.jpeg"),
                use="reference",
                preferred_roles=("hero", "front"),
            )
            first_frame_url = candidates[0] if candidates else ""
        if not first_frame_url:
            return ToolExecution("failed", "no safe video first frame")
        asset = self._find_asset(assets, target)
        raw_video = self._next_raw_path(downloads_dir, ".mp4")
        staged = work_dir / f".repair-video-{uuid.uuid4().hex}.mp4"
        prompt = (
            plan.video_prompt
            + "\nIndependent evaluator correction for this revision: "
            + instruction
            + "\nCorrect the temporal defect while preserving exact product identity in every frame."
        )
        try:
            video_url, model = self.client.generate_video(
                prompt,
                first_frame_url,
                negative_prompt=_VIDEO_NEGATIVE_PROMPT,
            )
            self.downloader.download(
                video_url, raw_video, max_bytes=199 * 1024 * 1024, timeout=300
            )
            if os.environ.get("AGENT_KEEP_VIDEO_AUDIO", "").strip() == "1":
                shutil.copyfile(raw_video, staged)
            else:
                strip_video_audio(raw_video, staged)
            inspect_video(staged)
            review_sources = self._source_urls_for_use(
                self._fallback_source_urls(facts, asset_name="main_image.jpeg"),
                use="reference",
                preferred_roles=("hero", "front"),
            )[:3]
            review = self.client.review_generated_video(
                json.dumps(facts.compact_dict(), ensure_ascii=False),
                review_sources,
                video_url,
                current_video_url=(
                    asset.source_url if asset.generated and asset.source_url else ""
                ),
            )
            if not self._video_revision_improves(review, has_incumbent=asset.generated):
                return ToolExecution(
                    "skipped",
                    "video revision did not pass semantic A/B improvement gate; prior video preserved",
                )
            os.replace(staged, Path(asset.path))
            asset.source_url = video_url
            asset.model = f"{model}-agent-repair"
            asset.generated = True
            asset.fallback_reason = ""
            asset.description = f"Agent-repaired product video: {instruction[:240]}"
            return ToolExecution("completed", "video revision accepted")
        except (ApiError, MediaError, OSError, PipelineError) as exc:
            return ToolExecution(
                "failed", f"video revision rejected; prior playable video preserved: {exc}"
            )
        finally:
            raw_video.unlink(missing_ok=True)
            staged.unlink(missing_ok=True)

    @staticmethod
    def _video_revision_improves(
        review: dict[str, Any], *, has_incumbent: bool
    ) -> bool:
        if has_incumbent:
            rows = review.get("candidates")
            if not isinstance(rows, list):
                return False
            by_index = {
                row.get("index"): row
                for row in rows
                if isinstance(row, dict) and isinstance(row.get("index"), int)
            }
            old = by_index.get(0)
            new = by_index.get(1)
            if not isinstance(old, dict) or not isinstance(new, dict):
                return False
            old_score = old.get("score")
            new_score = new.get("score")
            if (
                review.get("selected_index") != 1
                or not isinstance(old_score, (int, float))
                or not isinstance(new_score, (int, float))
                or float(new_score) < float(old_score) + 6.0
            ):
                return False
            candidate = new
        else:
            candidate = review
        return bool(
            candidate.get("usable") is True
            and candidate.get("identity_consistent") is not False
            and candidate.get("construction_consistent") is not False
            and candidate.get("color_and_pattern_consistent") is not False
            and candidate.get("motion_stable") is not False
            and candidate.get("unwanted_text") is not True
            and candidate.get("prohibited_visual") is not True
            and candidate.get("major_artifacts") is not True
        )

    def _adjudicate_taxonomy(
        self,
        facts: ProductFacts,
        taxonomy: TaxonomyResult,
        category_tree: dict[str, Any],
        attribute_data: dict[str, Any],
    ) -> TaxonomyResult:
        candidates = taxonomy.category.candidates[:12]
        top_score = (
            float(candidates[0].get("score") or 0) if candidates else 0.0
        )
        second_score = (
            float(candidates[1].get("score") or 0)
            if len(candidates) > 1
            else top_score
        )
        score_margin = max(0.0, top_score - second_score)
        # Curated aliases are deterministic and should not be second-guessed. A
        # locally ranked hidden category is adjudicated whenever the ranking is
        # weak or close, even if its absolute lexical score happened to be high.
        needs_model_adjudication = (
            taxonomy.category.method == "local-lexical-ranking"
            and (taxonomy.category.confidence < 0.91 or score_margin < 18.0)
        )
        if self.client is None or not needs_model_adjudication:
            return taxonomy
        allowed_ids = {str(item.get("category_id")) for item in candidates}
        if not allowed_ids:
            return taxonomy
        system = (
            "You are a conservative AliExpress apparel taxonomy classifier. Return JSON only. "
            "You must choose exactly one supplied leaf category ID. Do not invent an ID.\n\n"
            + self.skills.compile(
                "taxonomy", "product-grounding", "aliexpress-taxonomy"
            )
        )
        prompt = f"""
Choose the most accurate leaf category for the verified product.
Return JSON with selected_category_id and concise evidence.

Product:
{json.dumps(facts.compact_dict(), ensure_ascii=False)}

Allowed leaf candidates:
{json.dumps(candidates, ensure_ascii=False)}
""".strip()
        try:
            response = self.client.chat_json(system, prompt)
        except ApiError as exc:
            self.logger.warning("类目候选裁决失败，保留本地结果: %s", exc)
            self.trace.emit(
                "taxonomy.adjudication_failed",
                category=taxonomy.category.category_id,
                confidence=taxonomy.category.confidence,
                score_margin=score_margin,
                error=str(exc),
            )
            return taxonomy
        selected_id = str(response.get("selected_category_id") or "")
        if selected_id not in allowed_ids:
            self.logger.warning("类目裁决返回候选外 ID，已忽略: %s", selected_id)
            self.trace.emit(
                "taxonomy.adjudication_rejected",
                selected_category_id=selected_id,
                allowed_category_ids=sorted(allowed_ids),
            )
            return taxonomy
        self.trace.emit(
            "taxonomy.adjudication",
            local_category_id=taxonomy.category.category_id,
            selected_category_id=selected_id,
            confidence=taxonomy.category.confidence,
            score_margin=score_margin,
            evidence=response.get("evidence"),
        )
        return resolve_taxonomy(
            facts,
            category_tree,
            attribute_data,
            preferred_category_id=selected_id,
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
        inspected = sum(item.get("inspection_complete") is True for item in source_images)
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
        source_images = vision.get("source_images") if isinstance(vision, dict) else None
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
        self, facts: ProductFacts, vision: dict[str, Any]
    ) -> list[str]:
        ordered = _unique(
            facts.product_image_urls
            + facts.sku_image_urls
            + facts.description_image_urls
        )
        best_url = ""
        source_images = vision.get("source_images") if isinstance(vision, dict) else None
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
        ranked = self._source_urls_for_use(
            ordered,
            use="reference",
            preferred_roles=("hero", "front", "variant", "detail"),
        )
        product_roles = {"hero", "front", "back", "side", "detail", "variant", "lifestyle"}
        inspected_product = [
            url
            for url in ranked
            if self._source_image_observations.get(url, {}).get("role") in product_roles
        ]
        return inspected_product or [
            url for url in ranked if url in facts.product_image_urls or url in facts.sku_image_urls
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
                safety = 3 if use == "reference" and self._source_image_observations else 2
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

        ranked = sorted(enumerate(unique_urls), key=lambda pair: (*rank(pair[1]), pair[0]))
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
        explicit = [field for field in terminal_fields if observation.get(field) is True]
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
        return explicit + [reason for reason in reasons if any(key in reason for key in keywords)]

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
        return not observation or observation.get("safe_for_generation_reference") is True

    def _next_raw_path(self, downloads_dir: Path, suffix: str) -> Path:
        with self._raw_counter_lock:
            self._raw_counter += 1
            counter = self._raw_counter
        return downloads_dir / f"raw-{counter:03d}{suffix}"

    def _download_and_normalize(
        self,
        url: str,
        destination: Path,
        downloads_dir: Path,
        *,
        canvas: tuple[int, int],
        white_background: bool,
        focus_crop: str = "",
    ) -> None:
        raw_path = self._next_raw_path(downloads_dir, ".asset")
        self.downloader.download(url, raw_path, max_bytes=30 * 1024 * 1024, timeout=180)
        normalize_image(
            raw_path,
            destination,
            canvas=canvas,
            max_bytes=5 * 1024 * 1024,
            white_background=white_background,
            focus_crop=focus_crop,
        )

    def _fallback_image(
        self,
        source_urls: list[str],
        destination: Path,
        downloads_dir: Path,
        *,
        canvas: tuple[int, int],
        white_background: bool,
        avoid_hashes: list[int] | None = None,
        focus_crop: str = "",
    ) -> str:
        errors: list[str] = []
        for url in source_urls:
            candidate_destination = destination
            if avoid_hashes:
                candidate_destination = destination.with_name(
                    f".{destination.stem}-{uuid.uuid4().hex}.candidate.jpeg"
                )
            try:
                self._download_and_normalize(
                    url,
                    candidate_destination,
                    downloads_dir,
                    canvas=canvas,
                    white_background=white_background,
                    focus_crop=focus_crop,
                )
                if avoid_hashes:
                    quality = inspect_image_quality(candidate_destination)
                    if quality is not None and any(
                        hash_distance(quality.difference_hash, seen_hash) <= 10
                        for seen_hash in avoid_hashes
                    ):
                        errors.append(f"候选源图与已用详情图近重复: {url}")
                        candidate_destination.unlink(missing_ok=True)
                        continue
                    os.replace(candidate_destination, destination)
                return url
            except (ApiError, MediaError) as exc:
                errors.append(str(exc))
                candidate_destination.unlink(missing_ok=True)
        raise PipelineError("所有源图片回退均失败: " + "; ".join(errors[-3:]))

    @staticmethod
    def _candidate_soft_score(
        item: dict[str, Any], *, selected_index: Any
    ) -> float:
        raw_score = item.get("score")
        score = float(raw_score) if isinstance(raw_score, (int, float)) else 70.0
        if item.get("index") == selected_index:
            score += 2.0
        if item.get("usable") is False:
            score -= 8.0
        # A close-up may omit the full category outline while still delivering
        # its assigned local feature. Identity and slot mismatch remain hard.
        if item.get("critical_structure_unambiguous") is False:
            score -= 6.0
        coverage = str(item.get("product_coverage") or "").casefold()
        if coverage == "low":
            score -= 10.0
        elif coverage == "medium":
            score -= 3.0
        return max(0.0, min(100.0, score))

    def _choose_monotonic_candidate(
        self,
        label: str,
        candidate_urls: list[str],
        ranked: list[tuple[float, int, dict[str, Any]]],
        *,
        incumbent_index: int | None,
        minimum_improvement: float,
        hard_reasons: list[str],
    ) -> str:
        incumbent_valid = (
            isinstance(incumbent_index, int)
            and 0 <= incumbent_index < len(candidate_urls)
        )
        if not ranked:
            if incumbent_valid:
                self.logger.warning(
                    "%s 新候选均有语义硬伤，保留当前资产", label
                )
                return candidate_urls[incumbent_index]
            feedback = "; ".join(hard_reasons[:4]) or "评审未返回可比较候选"
            raise SemanticRejection(
                f"{label} 候选均未通过语义质检（存在硬伤）", feedback=feedback
            )

        best_score, best_index, _ = max(ranked, key=lambda row: (row[0], -row[1]))
        if incumbent_valid:
            incumbent_row = next(
                (row for row in ranked if row[1] == incumbent_index), None
            )
            # If the judge omitted or hard-rejected the incumbent, the old artifact
            # is still safer than an unproven replacement unless a new candidate is
            # clearly acceptable at a high absolute score.
            incumbent_score = incumbent_row[0] if incumbent_row else 80.0
            if best_index == incumbent_index:
                return candidate_urls[incumbent_index]
            required = incumbent_score + max(0.0, minimum_improvement)
            if best_score < required:
                self.logger.info(
                    "%s 修复候选提升不足: old=%.1f new=%.1f required=%.1f，保留旧资产",
                    label,
                    incumbent_score,
                    best_score,
                    required,
                )
                return candidate_urls[incumbent_index]
        self.logger.info(
            "%s 候选选优: 选择 %d/%d，软评分 %.1f",
            label,
            best_index + 1,
            len(candidate_urls),
            best_score,
        )
        return candidate_urls[best_index]

    def _build_main_image(
        self,
        facts: ProductFacts,
        plan: CreativePlan,
        vision: dict[str, Any],
        work_dir: Path,
        downloads_dir: Path,
    ) -> tuple[AssetResult, str]:
        destination = work_dir / "main_image.jpeg"
        source_urls = self._ordered_source_urls(facts, vision)
        generation_failure = (
            "image model unavailable"
            if self.client is None
            else "no eligible product reference for image editing"
        )
        if self.client is not None and source_urls:
            try:
                generated_url, model = self._generate_main_with_semantic_retry(
                    facts,
                    plan.main_prompt,
                    generation_references=source_urls[:1],
                    review_references=source_urls[:3],
                )
                try:
                    self._download_and_normalize(
                        generated_url,
                        destination,
                        downloads_dir,
                        canvas=(1600, 1600),
                        white_background=True,
                    )
                except (ApiError, MediaError) as download_error:
                    if self.deadline - time.monotonic() < 360:
                        raise
                    self.logger.warning(
                        "主图候选下载或物理校验失败，重新生成一次而非直接回退: %s",
                        download_error,
                    )
                    generated_url, model = self._generate_main_with_semantic_retry(
                        facts,
                        plan.main_prompt
                        + "\nThe previous output URL or file failed physical validation. Produce a fresh clean asset.",
                        generation_references=source_urls[:1],
                        review_references=source_urls[:3],
                    )
                    self._download_and_normalize(
                        generated_url,
                        destination,
                        downloads_dir,
                        canvas=(1600, 1600),
                        white_background=True,
                    )
                return (
                    AssetResult(
                        name="main_image.jpeg",
                        path=str(destination),
                        source_url=generated_url,
                        model=model,
                        generated=True,
                        description="Clean square hero image",
                    ),
                    generated_url,
                )
            except (ApiError, MediaError) as exc:
                generation_failure = str(exc)
                self.logger.warning("主图生成失败，使用源图回退: %s", exc)
                self.warnings.append(f"主图生成回退: {exc}")
        fallback_url = self._fallback_image(
            self._fallback_source_urls(facts, asset_name="main_image.jpeg"),
            destination,
            downloads_dir,
            canvas=(1600, 1600),
            white_background=True,
        )
        return (
            AssetResult(
                name="main_image.jpeg",
                path=str(destination),
                source_url=fallback_url,
                model="deterministic-source-fallback",
                generated=False,
                fallback_reason=generation_failure,
                description="Source-faithful square hero image",
            ),
            fallback_url,
        )

    def _generate_main_with_semantic_retry(
        self,
        facts: ProductFacts,
        prompt: str,
        *,
        generation_references: list[str],
        review_references: list[str],
        incumbent_url: str = "",
        minimum_improvement: float = 0.0,
    ) -> tuple[str, str]:
        if self.client is None:
            raise ApiError("image model unavailable")
        active_prompt = prompt
        last_rejection: SemanticRejection | None = None
        for semantic_attempt in range(2):
            candidate_urls, model = self.client.generate_image_candidates(
                active_prompt,
                generation_references,
                size="1600*1600",
                negative_prompt=_MAIN_NEGATIVE_PROMPT,
                count=2 if self.fast_mode else 3,
            )
            reviewed_urls = (
                [incumbent_url, *candidate_urls] if incumbent_url else candidate_urls
            )
            try:
                selected = self._select_main_candidate(
                    facts,
                    review_references,
                    reviewed_urls,
                    incumbent_index=0 if incumbent_url else None,
                    minimum_improvement=minimum_improvement,
                )
                return selected, model
            except SemanticRejection as exc:
                last_rejection = exc
                if semantic_attempt > 0 or self.deadline - time.monotonic() < 420:
                    raise
                self.logger.warning(
                    "主图存在明确语义硬伤，携带质检反馈重新生成一次: %s",
                    exc.feedback[:500],
                )
                active_prompt = (
                    prompt
                    + "\nMandatory correction after semantic rejection: "
                    + exc.feedback[:1200]
                    + "\nPreserve exact product identity and correct only these hard defects."
                )
        raise last_rejection or SemanticRejection("主图语义纠错失败")

    def _select_main_candidate(
        self,
        facts: ProductFacts,
        source_urls: list[str],
        candidate_urls: list[str],
        *,
        incumbent_index: int | None = None,
        minimum_improvement: float = 0.0,
    ) -> str:
        if not candidate_urls:
            raise ApiError("主图模型未返回候选")
        # A single generated candidate still needs semantic acceptance. Skipping
        # review here lets structure drift pass merely because no alternative
        # candidate was requested for this storyboard slot.
        if self.client is None:
            return candidate_urls[0]
        try:
            review = self.client.select_best_generated_image(
                json.dumps(facts.compact_dict(), ensure_ascii=False),
                source_urls,
                candidate_urls,
            )
        except ApiError as exc:
            keep = (
                incumbent_index
                if isinstance(incumbent_index, int)
                and 0 <= incumbent_index < len(candidate_urls)
                else 0
            )
            self.logger.warning(
                "主图语义评审不可用，保留确定性安全候选 %d，避免误回退: %s",
                keep,
                exc,
            )
            self.warnings.append(f"主图候选评审不可用，保留候选: {exc}")
            return candidate_urls[keep]

        candidates = review.get("candidates")
        selected = review.get("selected_index")
        self.trace.emit(
            "image.hero_review",
            source_urls=source_urls,
            candidate_urls=candidate_urls,
            selected_index=selected,
            review=review,
            incumbent_index=incumbent_index,
            minimum_improvement=minimum_improvement,
        )
        if not isinstance(candidates, list):
            keep = incumbent_index if incumbent_index is not None else 0
            self.warnings.append("主图语义评审结构不完整，采用确定性候选")
            return candidate_urls[keep]
        wearer_supported = self._hero_wearer_supported(facts, source_urls)
        ranked: list[tuple[float, int, dict[str, Any]]] = []
        hard_reasons: list[str] = []
        for item in candidates:
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                continue
            index = item["index"]
            if not 0 <= index < len(candidate_urls):
                continue
            hard_fields = {
                "identity_consistent": False,
                "construction_consistent": False,
                "correct_color": False,
                "single_product": False,
                "product_complete": False,
                "unwanted_text": True,
                "unwanted_brand_or_logo": True,
                "major_artifacts": True,
            }
            failed = [key for key, value in hard_fields.items() if item.get(key) is value]
            if item.get("has_person") is True and not wearer_supported:
                failed.append("unsupported_wearer")
            if (
                item.get("has_person") is True
                and item.get("anatomy_natural") is False
            ):
                failed.append("anatomy_natural")
            if failed:
                hard_reasons.append(
                    f"candidate {index}: {','.join(failed)}; {item.get('reason', '')}"
                )
                continue
            score = self._candidate_soft_score(item, selected_index=selected)
            if item.get("clean_neutral_background") is False:
                score -= 8
            if item.get("has_unrelated_props") is True:
                score -= 5
            if item.get("has_person") is True:
                score -= 2
            ranked.append((score, index, item))
        return self._choose_monotonic_candidate(
            "主图",
            candidate_urls,
            ranked,
            incumbent_index=incumbent_index,
            minimum_improvement=minimum_improvement,
            hard_reasons=hard_reasons,
        )

    @staticmethod
    def _is_children_product(facts: ProductFacts) -> bool:
        source_text = " ".join(
            [
                facts.source_title,
                facts.source_category_name,
                *[f"{item.name} {item.value}" for item in facts.attributes],
            ]
        ).casefold()
        return bool(
            re.search(r"[男女]?童|儿童|婴儿|婴幼儿", source_text)
            or re.search(
                r"\b(?:boy|boys|girl|girls|kid|kids|child|children|baby|toddler)\b",
                source_text,
            )
        )

    def _hero_wearer_supported(
        self, facts: ProductFacts, source_urls: list[str]
    ) -> bool:
        if self._is_children_product(facts):
            return False
        return any(
            observation.get("has_person") is True
            and observation.get("safe_for_generation_reference") is True
            for url in source_urls
            if (observation := self._source_image_observations.get(url))
        )

    def _build_detail_image(
        self,
        index: int,
        facts: ProductFacts,
        plan: CreativePlan,
        main_reference_url: str,
        work_dir: Path,
        downloads_dir: Path,
    ) -> AssetResult:
        destination = work_dir / f"detail_image_{index}.jpeg"
        if index == 5 and facts.size_chart_rows:
            create_size_chart_image(facts.size_chart_rows, destination)
            return AssetResult(
                name=destination.name,
                path=str(destination),
                model="deterministic-size-chart",
                generated=False,
                fallback_reason="source size chart transcribed and deterministically rendered",
                description="Verified seller garment measurements and weight guidance",
            )
        fallback_urls, focus_crop = self._detail_fallback_plan(
            facts, index=index, main_reference_url=main_reference_url
        )
        reference_selection = self._detail_reference_selection(
            index, facts, main_reference_url
        )
        generation_failure = (
            "image model unavailable"
            if self.client is None
            else "no eligible product reference for this detail slot"
        )
        if self.client is not None and reference_selection:
            try:
                generated_url, model = self._generate_detail_with_semantic_retry(
                    index,
                    facts,
                    plan.detail_prompts[index - 1],
                    references=reference_selection[:3],
                )
                try:
                    self._download_and_normalize(
                        generated_url,
                        destination,
                        downloads_dir,
                        canvas=(1200, 1500),
                        white_background=False,
                    )
                except (ApiError, MediaError) as download_error:
                    if self.deadline - time.monotonic() < 300:
                        raise
                    self.logger.warning(
                        "详情图 %d 候选下载或物理校验失败，重新生成一次: %s",
                        index,
                        download_error,
                    )
                    generated_url, model = self._generate_detail_with_semantic_retry(
                        index,
                        facts,
                        plan.detail_prompts[index - 1]
                        + "\nThe previous output URL or file failed physical validation. Produce a fresh asset.",
                        references=reference_selection[:3],
                    )
                    self._download_and_normalize(
                        generated_url,
                        destination,
                        downloads_dir,
                        canvas=(1200, 1500),
                        white_background=False,
                    )
                return AssetResult(
                    name=destination.name,
                    path=str(destination),
                    source_url=generated_url,
                    model=model,
                    generated=True,
                    description=(
                        f"Canonical detail role: "
                        f"{plan.detail_roles[index - 1] if index <= len(plan.detail_roles) else f'slot_{index}'}"
                    ),
                )
            except (ApiError, MediaError) as exc:
                generation_failure = str(exc)
                self.logger.warning("详情图 %d 生成失败，使用源图回退: %s", index, exc)
                self.warnings.append(f"详情图 {index} 生成回退: {exc}")

        fallback_url = self._fallback_image(
            fallback_urls,
            destination,
            downloads_dir,
            canvas=(1200, 1500),
            white_background=False,
            focus_crop=focus_crop,
        )
        fallback_description = (
            f"Seller-source {focus_crop} close-up for detail slot {index}"
            if focus_crop
            else f"Distinct seller-source full view for detail slot {index}"
        )
        return AssetResult(
            name=destination.name,
            path=str(destination),
            source_url=fallback_url,
            model="deterministic-source-fallback",
            generated=False,
            fallback_reason=generation_failure,
            description=fallback_description,
        )

    def _generate_detail_with_semantic_retry(
        self,
        index: int,
        facts: ProductFacts,
        prompt: str,
        *,
        references: list[str],
        incumbent_url: str = "",
        minimum_improvement: float = 0.0,
    ) -> tuple[str, str]:
        if self.client is None:
            raise ApiError("image model unavailable")
        active_prompt = prompt
        last_rejection: SemanticRejection | None = None
        semantic_attempts = 1 if self.fast_mode else 2
        for semantic_attempt in range(semantic_attempts):
            candidate_urls, model = self.client.generate_image_candidates(
                active_prompt,
                references,
                size="1200*1500",
                negative_prompt=(
                    _IMAGE_NEGATIVE_PROMPT
                    + _SINGLE_COMPOSITION_NEGATIVE_PROMPT
                    + (
                        ""
                        if index == 4
                        else ", collage, montage, grid, duplicate product, multiple views"
                    )
                ),
                count=1 if self.fast_mode else 2,
            )
            if self.fast_mode:
                self.trace.emit(
                    "image.detail_review_skipped",
                    asset=f"detail_image_{index}.jpeg",
                    reason="fast-profile",
                )
                return candidate_urls[0], model
            reviewed_urls = (
                [incumbent_url, *candidate_urls] if incumbent_url else candidate_urls
            )
            try:
                selected = self._select_detail_candidate(
                    index,
                    facts,
                    references,
                    reviewed_urls,
                    active_prompt,
                    incumbent_index=0 if incumbent_url else None,
                    minimum_improvement=minimum_improvement,
                )
                return selected, model
            except SemanticRejection as exc:
                last_rejection = exc
                if semantic_attempt + 1 >= semantic_attempts or self.deadline - time.monotonic() < 360:
                    raise
                self.logger.warning(
                    "详情图 %d 存在明确语义硬伤，携带质检反馈重新生成一次: %s",
                    index,
                    exc.feedback[:500],
                )
                active_prompt = (
                    prompt
                    + "\nMandatory correction after semantic rejection: "
                    + exc.feedback[:1200]
                    + "\nPreserve exact product identity and storyboard purpose."
                )
        raise last_rejection or SemanticRejection(f"详情图 {index} 语义纠错失败")

    def _select_detail_candidate(
        self,
        index: int,
        facts: ProductFacts,
        source_urls: list[str],
        candidate_urls: list[str],
        purpose: str,
        *,
        incumbent_index: int | None = None,
        minimum_improvement: float = 0.0,
    ) -> str:
        if not candidate_urls:
            raise ApiError(f"详情图 {index} 模型未返回候选")
        # A single generated detail candidate still needs semantic acceptance;
        # otherwise structure drift can pass just because there is no alternative.
        if self.client is None:
            return candidate_urls[0]
        try:
            review = self.client.select_best_detail_image(
                json.dumps(facts.compact_dict(), ensure_ascii=False),
                source_urls,
                candidate_urls,
                asset_name=f"detail_image_{index}.jpeg",
                purpose=purpose,
            )
        except ApiError as exc:
            keep = (
                incumbent_index
                if isinstance(incumbent_index, int)
                and 0 <= incumbent_index < len(candidate_urls)
                else 0
            )
            self.logger.warning(
                "详情图 %d 语义评审不可用，保留确定性安全候选 %d: %s",
                index,
                keep,
                exc,
            )
            self.warnings.append(f"详情图 {index} 评审不可用，保留候选: {exc}")
            return candidate_urls[keep]
        candidates = review.get("candidates")
        selected = review.get("selected_index")
        self.trace.emit(
            "image.detail_review",
            asset=f"detail_image_{index}.jpeg",
            purpose=purpose,
            source_urls=source_urls,
            candidate_urls=candidate_urls,
            selected_index=selected,
            review=review,
            incumbent_index=incumbent_index,
            minimum_improvement=minimum_improvement,
        )
        if not isinstance(candidates, list):
            keep = incumbent_index if incumbent_index is not None else 0
            self.warnings.append(f"详情图 {index} 评审结构不完整，采用确定性候选")
            return candidate_urls[keep]
        ranked: list[tuple[float, int, dict[str, Any]]] = []
        hard_reasons: list[str] = []
        for item in candidates:
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                continue
            candidate_index = item["index"]
            if not 0 <= candidate_index < len(candidate_urls):
                continue
            hard_fields = {
                "identity_consistent": False,
                "construction_consistent": False,
                "color_consistent": False,
                "pattern_consistent": False,
                "unwanted_text": True,
                "unwanted_brand_or_logo": True,
                "prohibited_visual": True,
                "major_artifacts": True,
                "unexpected_collage": True,
                "single_composition": False,
            }
            purpose_text = purpose.casefold()
            if any(
                token in purpose_text
                for token in ("wearer", "person", "adult", "man", "woman", "body")
            ):
                hard_fields["anatomy_natural"] = False
            failed = [key for key, value in hard_fields.items() if item.get(key) is value]
            if failed:
                hard_reasons.append(
                    f"candidate {candidate_index}: {','.join(failed)}; {item.get('reason', '')}"
                )
                continue
            if item.get("slot_match") is False:
                hard_reasons.append(
                    f"candidate {candidate_index}: slot_match; {item.get('reason', '')}"
                )
                continue
            score = self._candidate_soft_score(item, selected_index=selected)
            ranked.append((score, candidate_index, item))
        return self._choose_monotonic_candidate(
            f"详情图 {index}",
            candidate_urls,
            ranked,
            incumbent_index=incumbent_index,
            minimum_improvement=minimum_improvement,
            hard_reasons=hard_reasons,
        )

    def _detail_reference_selection(
        self, index: int, facts: ProductFacts, main_reference_url: str
    ) -> list[str]:
        product = facts.product_image_urls
        sku = facts.sku_image_urls
        description = facts.description_image_urls
        if index == 4 and sku:
            inspected_variants = [
                url for url in sku if url in self._source_image_observations
            ]
            variant_references = inspected_variants or _even_sample(sku, 3)
            return self._source_urls_for_use(
                _unique(_even_sample(variant_references, 3) + product[:1]),
                use="reference",
                preferred_roles=("variant", "front", "hero"),
            )[:3]
        role_preferences = {
            1: ("front", "hero", "lifestyle"),
            2: ("detail", "front", "side"),
            3: ("detail", "side", "back"),
            4: ("variant", "front", "hero"),
            5: ("lifestyle", "front", "hero"),
        }
        # Search the whole inspected source set for the role needed by each slot.
        # Description images are valuable for close-ups and lifestyle composition,
        # but the safety rank below excludes charts, promotional overlays and marks.
        candidate_pool = _unique(
            [main_reference_url] + product + description + _even_sample(sku, 3)
        )
        ranked = self._source_urls_for_use(
            candidate_pool,
            use="reference",
            preferred_roles=role_preferences.get(index, ()),
        )
        excluded_roles = {"size_chart", "packaging", "unknown"}
        role_safe = [
            url
            for url in ranked
            if self._source_image_observations.get(url, {}).get("role")
            not in excluded_roles
        ]
        return (role_safe or ranked)[:3]

    def _build_video(
        self,
        facts: ProductFacts,
        plan: CreativePlan,
        first_frame_url: str,
        main_image_path: Path,
        work_dir: Path,
        downloads_dir: Path,
        allow_generation: bool,
    ) -> AssetResult:
        destination = work_dir / "product_video.mp4"
        generation_failure = "video model configuration or safe first frame unavailable"
        if self.client is not None and allow_generation:
            try:
                video_url, model = self.client.generate_video(
                    plan.video_prompt,
                    first_frame_url,
                    negative_prompt=_VIDEO_NEGATIVE_PROMPT,
                )
                raw_video = self._next_raw_path(downloads_dir, ".mp4")
                self.downloader.download(
                    video_url, raw_video, max_bytes=199 * 1024 * 1024, timeout=300
                )
                if os.environ.get("AGENT_KEEP_VIDEO_AUDIO", "").strip() == "1":
                    shutil.copyfile(raw_video, destination)
                    inspect_video(destination)
                else:
                    strip_video_audio(raw_video, destination)
                return AssetResult(
                    name=destination.name,
                    path=str(destination),
                    source_url=video_url,
                    model=model,
                    generated=True,
                    description="Short source-guided product video",
                )
            except (ApiError, MediaError, OSError) as exc:
                generation_failure = str(exc)
                self.logger.warning("视频模型失败，创建确定性视频回退: %s", exc)
                self.warnings.append(f"视频生成回退: {exc}")
        elif self.client is not None:
            if self.fast_mode:
                generation_failure = "fast profile skips video-model generation"
                self.trace.emit("video.generation_skipped", reason="fast-profile")
            else:
                self.warnings.append("首帧源图触发知识产权或视觉风险，跳过衍生视频生成")
        create_slideshow_video(main_image_path, destination, duration=8)
        return AssetResult(
            name=destination.name,
            path=str(destination),
            model="ffmpeg-slideshow-fallback",
            generated=False,
            fallback_reason=generation_failure,
            description="Playable H.264 product presentation fallback",
        )

    def _review_generated_assets(
        self,
        facts: ProductFacts,
        assets: list[AssetResult],
        work_dir: Path,
        downloads_dir: Path,
    ) -> None:
        # Kept as a compatibility hook for callers outside Pipeline.run. Semantic
        # feedback is handled by _run_bounded_agent_loop and never triggers fallback.
        del downloads_dir
        self._install_size_chart_detail(facts, assets, work_dir)
        self._enhance_fallback_video(assets, work_dir)

    def _install_size_chart_detail(
        self, facts: ProductFacts, assets: list[AssetResult], work_dir: Path
    ) -> None:
        if not facts.size_chart_rows:
            return
        asset = next(
            (item for item in assets if item.name == "detail_image_5.jpeg"), None
        )
        if asset is None:
            return
        destination = work_dir / "detail_image_5.jpeg"
        try:
            create_size_chart_image(facts.size_chart_rows, destination)
        except MediaError as exc:
            self.logger.warning("本地化尺码表生成失败，保留原详情图: %s", exc)
            self.warnings.append(f"尺码表详情图生成失败: {exc}")
            return
        asset.path = str(destination)
        asset.source_url = self._size_chart_source_url or asset.source_url
        asset.model = "deterministic-size-chart"
        asset.generated = False
        asset.fallback_reason = "source size chart transcribed and deterministically rendered"
        asset.description = "Verified seller garment measurements and weight guidance"

    def _enhance_fallback_video(
        self, assets: list[AssetResult], work_dir: Path
    ) -> None:
        video_asset = next(
            (asset for asset in assets if asset.name == "product_video.mp4"), None
        )
        if not video_asset or video_asset.generated:
            return
        image_paths = [work_dir / "main_image.jpeg"] + [
            work_dir / f"detail_image_{index}.jpeg" for index in range(1, 6)
        ]
        candidate = work_dir / ".product_video_catalog.mp4"
        try:
            create_catalog_video(image_paths, candidate, duration=8)
            os.replace(candidate, Path(video_asset.path))
        except (MediaError, OSError) as exc:
            candidate.unlink(missing_ok=True)
            self.logger.warning("多镜头视频回退不可用，保留稳定单图视频: %s", exc)
            self.warnings.append(f"多镜头视频回退不可用: {exc}")
            return
        video_asset.model = "ffmpeg-catalog-fallback"
        rebuild_reason = "rebuilt from the final available image set"
        if rebuild_reason not in video_asset.fallback_reason:
            video_asset.fallback_reason = (
                (video_asset.fallback_reason + "; ")
                if video_asset.fallback_reason
                else ""
            ) + rebuild_reason
        video_asset.description = (
            "Eight-second multi-shot catalog fallback assembled from perceptually distinct available images"
        )

    def _repair_duplicate_fallback_details(
        self,
        assets: list[AssetResult],
        *,
        main_reference_url: str,
        work_dir: Path,
        downloads_dir: Path,
    ) -> None:
        """Turn deterministic duplicate warnings into bounded local repairs."""

        ordered = sorted(
            (asset for asset in assets if asset.name.endswith(".jpeg")),
            key=lambda asset: (
                0 if asset.name == "main_image.jpeg" else 1,
                asset.name,
            ),
        )
        accepted_hashes: list[int] = []
        # Match the delivery QA threshold, but protect verified back/side views:
        # apparel with the same print can hash similarly even when that view adds
        # genuinely different construction evidence. Repeated front/hero poses do
        # not receive that exemption and are converted into useful detail crops.
        automatic_repair_threshold = 10
        crop_sequence = ("upper", "lower", "left", "right", "center")
        for asset in ordered:
            try:
                quality = inspect_image_quality(Path(asset.path))
            except MediaError:
                quality = None
            if quality is None:
                continue
            if all(
                hash_distance(quality.difference_hash, seen)
                > automatic_repair_threshold
                for seen in accepted_hashes
            ):
                accepted_hashes.append(quality.difference_hash)
                continue
            observation = self._source_image_observations.get(asset.source_url, {})
            if (
                str(observation.get("role") or "") in {"back", "side"}
                and asset.source_url != main_reference_url
            ):
                accepted_hashes.append(quality.difference_hash)
                continue
            if (
                asset.generated
                or asset.model == "deterministic-size-chart"
                or not asset.name.startswith("detail_image_")
            ):
                continue

            source_url = main_reference_url or asset.source_url
            if not source_url:
                continue
            repaired = False
            for focus_crop in crop_sequence:
                staged = work_dir / f".{asset.name}.{focus_crop}.repair.jpeg"
                try:
                    self._fallback_image(
                        [source_url],
                        staged,
                        downloads_dir,
                        canvas=(1200, 1500),
                        white_background=False,
                        focus_crop=focus_crop,
                    )
                    candidate = inspect_image_quality(staged)
                    if candidate is None or any(
                        hash_distance(candidate.difference_hash, seen) <= 10
                        for seen in accepted_hashes
                    ):
                        staged.unlink(missing_ok=True)
                        continue
                    os.replace(staged, Path(asset.path))
                    accepted_hashes.append(candidate.difference_hash)
                    asset.source_url = source_url
                    asset.model = "deterministic-source-detail-crop"
                    asset.description = (
                        f"Seller-source {focus_crop} close-up repaired from a duplicate slot"
                    )
                    note = f"自动修复近重复详情图: {asset.name} -> {focus_crop} close-up"
                    if note not in self.warnings:
                        self.warnings.append(note)
                    repaired = True
                    break
                except (ApiError, MediaError, OSError):
                    staged.unlink(missing_ok=True)
            if not repaired:
                # The normal QA report retains the unresolved duplicate warning.
                continue

    def _record_visual_delivery_quality(self, assets: list[AssetResult]) -> None:
        """Expose rubric-level fallback risks that physical file QA cannot see."""

        image_assets = [asset for asset in assets if asset.name.endswith(".jpeg")]
        hashes: list[tuple[str, int]] = []
        for asset in image_assets:
            try:
                quality = inspect_image_quality(Path(asset.path))
            except MediaError:
                quality = None
            if quality is not None:
                hashes.append((asset.name, quality.difference_hash))
        distinct_names: set[str] = set()
        distinct_hashes: list[int] = []
        for name, image_hash in hashes:
            if all(hash_distance(image_hash, seen) > 10 for seen in distinct_hashes):
                distinct_names.add(name)
                distinct_hashes.append(image_hash)
        for left_index, (left_name, left_hash) in enumerate(hashes):
            for right_name, right_hash in hashes[left_index + 1 :]:
                if hash_distance(left_hash, right_hash) <= 10:
                    warning = f"最终商品图近重复: {left_name}, {right_name}"
                    if warning not in self.warnings:
                        self.warnings.append(warning)

        usable = 0
        risky_names: list[str] = []
        for asset in image_assets:
            if hashes and asset.name not in distinct_names:
                continue
            if asset.generated or asset.model == "deterministic-size-chart":
                usable += 1
                continue
            observation = self._source_image_observations.get(asset.source_url, {})
            safe_key = (
                "safe_for_main_image"
                if asset.name == "main_image.jpeg"
                else "safe_for_listing_fallback"
            )
            if observation.get(safe_key) is True:
                usable += 1
            elif observation:
                risky_names.append(asset.name)
            else:
                # Explicit offline mode cannot run a semantic listing-readiness
                # review. Count only physically valid, perceptually distinct
                # fallbacks and label the estimate accordingly below.
                usable += 1

        if risky_names:
            warning = (
                "最终视觉兜底未达到直接发布门禁: " + ", ".join(risky_names)
            )
            if warning not in self.warnings:
                self.warnings.append(warning)
        if image_assets:
            usable_rate = usable / len(image_assets)
            estimate_basis = (
                "视觉门禁与感知差异"
                if self._source_image_observations
                else "物理规格与感知差异（未执行模型语义门禁）"
            )
            if usable_rate < 0.8:
                warning = (
                    f"按{estimate_basis}估算的出图可用率为 {usable_rate:.0%}，低于 A6 的 80% 阈值"
                )
                if warning not in self.warnings:
                    self.warnings.append(warning)
            elif not self._source_image_observations:
                warning = (
                    f"离线图片物理规格与感知差异检查通过 {usable}/{len(image_assets)}；"
                    "语义可用率未评估，正式交付仍需确认背景、人物道具、分镜任务与商品身份"
                )
                if warning not in self.warnings:
                    self.warnings.append(warning)

        video = next(
            (asset for asset in assets if asset.name == "product_video.mp4"), None
        )
        if video and not video.generated and risky_names:
            warning = "回退视频继承了未通过直接发布门禁的静态图片"
            if warning not in self.warnings:
                self.warnings.append(warning)

    def _review_visual_set(
        self,
        facts: ProductFacts,
        assets: list[AssetResult],
    ) -> dict[str, Any]:
        """Judge hero + five details as one set, without mutating accepted files."""

        if self.client is None:
            self.trace.emit("image.set_review_skipped", reason="offline-or-no-review-client")
            return {}
        remaining = self.deadline - time.monotonic()
        if remaining <= 3 * 60:
            self.warnings.append("剩余时间不足，跳过六图集合语义评审")
            self.trace.emit(
                "image.set_review_skipped",
                reason="insufficient-stage-budget",
                remaining_seconds=round(remaining, 1),
            )
            return {}
        ordered_names = ["main_image.jpeg"] + [
            f"detail_image_{index}.jpeg" for index in range(1, 6)
        ]
        by_name = {asset.name: asset for asset in assets}
        if any(name not in by_name for name in ordered_names):
            self.warnings.append("六图集合不完整，无法执行集合级语义评审")
            self.trace.emit("image.set_review_skipped", reason="incomplete-image-set")
            return {}
        review_inputs = [by_name[name].source_url for name in ordered_names]
        if any(not url for url in review_inputs):
            self.warnings.append("六图集合缺少平台允许的远程评审 URL，跳过集合级模型评审")
            self.trace.emit("image.set_review_skipped", reason="missing-review-url")
            return {}

        if self.fast_mode:
            source_references = _unique(
                facts.product_image_urls[:2]
                + _even_sample(facts.sku_image_urls, 1)
            )
        else:
            source_references = _unique(
                facts.product_image_urls[:3]
                + _even_sample(facts.sku_image_urls, 1)
                + _even_sample(facts.description_image_urls, 1)
            )
        expected_assets = [
            {
                "name": name,
                "purpose": by_name[name].description or name,
                "representation": (
                    "source proxy for deterministic local rendering"
                    if by_name[name].model == "deterministic-size-chart"
                    else "final model output before lossless delivery normalization"
                    if by_name[name].generated
                    else "source-backed fallback"
                ),
            }
            for name in ordered_names
        ]
        try:
            review = self.client.review_generated_images(
                json.dumps(facts.compact_dict(), ensure_ascii=False),
                source_references,
                review_inputs,
                expected_assets,
            )
        except ApiError as exc:
            # A judge outage is not evidence that an already accepted image is bad.
            self.logger.warning("六图集合语义评审不可用，保留当前素材: %s", exc)
            self.warnings.append(f"六图集合语义评审未完成: {exc}")
            self.trace.emit(
                "image.set_review_failed",
                category=exc.category,
                retryable=exc.retryable,
                status_code=exc.status_code,
                error=str(exc),
            )
            return {}

        rows = review.get("assets")
        if not isinstance(rows, list) or len(rows) != len(ordered_names):
            self.warnings.append("六图集合评审返回结构异常，忽略该评审而不判定素材失败")
            self.trace.emit("image.set_review_invalid", review=review)
            return {}
        normalized_rows: list[dict[str, Any]] = []
        for index, name in enumerate(ordered_names):
            item = next(
                (
                    row
                    for row in rows
                    if isinstance(row, dict) and row.get("index") == index
                ),
                {},
            )
            normalized_rows.append({"name": name, **item})
        review["assets"] = normalized_rows
        review["reviewed_names"] = ordered_names
        review["review_model"] = getattr(
            self.client,
            "visual_review_model",
            self.client.config.review_model,
        )

        duplicate_pairs = review.get("near_duplicate_pairs")
        if isinstance(duplicate_pairs, list) and duplicate_pairs:
            readable_pairs: list[str] = []
            for pair in duplicate_pairs[:6]:
                if (
                    isinstance(pair, list)
                    and len(pair) == 2
                    and all(isinstance(item, int) for item in pair)
                    and all(0 <= item < len(ordered_names) for item in pair)
                ):
                    readable_pairs.append(
                        f"{ordered_names[pair[0]]}/{ordered_names[pair[1]]}"
                    )
            if readable_pairs:
                self.warnings.append(
                    "六图集合存在语义重复: " + ", ".join(readable_pairs)
                )
        missing_roles = review.get("missing_roles")
        if isinstance(missing_roles, list) and missing_roles:
            self.warnings.append(
                "六图集合商业任务覆盖不足: "
                + ", ".join(str(item)[:80] for item in missing_roles[:5])
            )
        self.trace.emit("image.set_review", review=review)
        return review

    def _repair_visual_set_once(
        self,
        *,
        facts: ProductFacts,
        creative_plan: CreativePlan,
        state: RunState,
        review: dict[str, Any],
        work_dir: Path,
        downloads_dir: Path,
    ) -> dict[str, Any]:
        """Repair at most one detail whose replacement improves the whole set."""

        if not review or review.get("set_usable") is True:
            return review
        if self.deadline - time.monotonic() <= 6 * 60:
            self.trace.emit(
                "image.set_repair_skipped",
                reason="insufficient-stage-budget",
                remaining_seconds=round(self.deadline - time.monotonic(), 1),
            )
            return review
        targets = review.get("repair_targets")
        if not isinstance(targets, list):
            return review
        target_index = next(
            (
                item
                for item in targets
                if isinstance(item, int)
                and 1 <= item <= 5
                and not (item == 5 and facts.size_chart_rows)
            ),
            None,
        )
        if target_index is None:
            return review
        rows = review.get("assets") if isinstance(review.get("assets"), list) else []
        row = next(
            (
                item
                for item in rows
                if isinstance(item, dict) and item.get("index") == target_index
            ),
            {},
        )
        instruction = "; ".join(
            item
            for item in (
                str(row.get("reason") or "").strip(),
                str(review.get("summary") or "").strip(),
                "Restore the canonical slot role and remove semantic duplication or split panels; use framing and product coverage appropriate to that role.",
            )
            if item
        )[:1400]
        target = f"detail_image_{target_index}.jpeg"
        result = self._repair_detail_image(
            target,
            instruction,
            facts,
            creative_plan,
            state.assets,
            work_dir,
            downloads_dir,
        )
        state.agent_actions.append(
            AgentActionResult(
                round_index=-1,
                tool="regenerate_detail_image",
                target=target,
                status=result.status,
                detail=result.detail,
            )
        )
        self.trace.emit(
            "image.set_repair_result",
            target=target,
            instruction=instruction,
            status=result.status,
            detail=result.detail,
        )
        if result.status != "completed":
            return review
        refreshed = self._review_visual_set(facts, state.assets)
        return refreshed or review

    def _write_strategy_document(
        self,
        state: RunState,
        localization_sources: dict[str, str],
        localization_payloads: dict[str, dict[str, Any]],
        plan_model: str,
        work_dir: Path,
    ) -> None:
        facts, taxonomy = state.facts, state.taxonomy
        generated_count = sum(1 for asset in state.assets if asset.generated)
        fallback_assets = [asset for asset in state.assets if not asset.generated]
        model_summary = (
            self.client.model_summary
            if self.client
            else {"mode": "explicit offline deterministic fallback"}
        )
        schema_id = taxonomy.attribute_schema_category_id or taxonomy.category.category_id
        schema_note = (
            f"叶子类目缺少独立属性元数据，属性映射使用同一平台快照中的上级/通用 schema {schema_id}；"
            f"上架叶子类目仍保持 {taxonomy.category.category_id}。"
            if schema_id != taxonomy.category.category_id
            else f"属性映射使用叶子类目 schema {schema_id}。"
        )
        failed_calls = [
            item for item in state.api_calls if str(item.get("status") or "") != "ok"
        ]
        execution_mode = "在线模型生成与评估" if self.client else "显式离线确定性降级"
        semantic_gate_note = (
            f"本次完成 {len(state.agent_evaluations)} 轮全局多模态评估。"
            if state.agent_evaluations
            else "本次未执行全局多模态语义评估；仅通过确定性事实、规格与感知差异门禁。"
        )
        raw_agent_plan = state.agent_plan if isinstance(state.agent_plan, dict) else {}
        agent_plan_controls = {
            "risk_priorities": [
                value
                for value in raw_agent_plan.get("risk_priorities", [])
                if value in {f"A{index}" for index in range(1, 8)}
            ],
            "max_repair_rounds": raw_agent_plan.get("max_repair_rounds"),
            "max_actions_per_round": raw_agent_plan.get("max_actions_per_round"),
            "minimum_weighted_score": raw_agent_plan.get("minimum_weighted_score"),
        }
        agent_plan_controls = {
            key: value for key, value in agent_plan_controls.items() if value is not None
        }

        def brief(value: str) -> str:
            cleaned = re.sub(r"https?://\S+", "[url]", value.replace("\n", " "))
            return cleaned[:260]

        known_claim_ids = {item.claim_id for item in state.claim_ledger}
        copy_claim_reference_lines: list[str] = []
        for language, payload in localization_payloads.items():
            refs = payload.get("claim_refs") if isinstance(payload, dict) else None
            referenced = sorted(
                {
                    value
                    for value in re.findall(r"\bC\d{3}\b", json.dumps(refs or {}))
                    if value in known_claim_ids
                }
            )
            if referenced:
                copy_claim_reference_lines.append(
                    f"- {language}: {', '.join(referenced)}"
                )

        lines = [
            "# 商品本地化素材生成策略说明",
            "",
            "## 1. 本次商品与目标",
            "",
            f"- 商品 ID：{facts.offer_id}",
            f"- 数据来源：{facts.platform}",
            f"- 源商品 URL：{facts.source_url}",
            "- 交付目标：英文、韩文、巴西葡萄牙文文案，1 张主图、5 张详情图、1 个商品视频。",
            "",
            "## 2. 事实一致性策略",
            "",
            "Agent 首先把商品 JSON 归一化为内部事实账本。标题、属性、SKU、图片 URL、商品 ID 和来源均在内部保留证据位置；"
            "只有源 JSON、源图片直接观察或确定性单位换算得到的信息可以进入文案和素材提示词。"
            "所有模型文案均经过结构、数值、事实和平台内容规则的确定性复核。",
            "三份文案只发布目标语言的买家文案和本地化显示值，不暴露中文原值或原始 JSON Pointer；"
            "本地化 Source 列标明商品事实、平台映射或卖家声明。平台类目 ID、属性 ID/Value ID、"
            "SKU ID/Spec ID 仍保留在精简表格中，兼顾上架解析与阅读体验。",
            "",
            f"本次共读取 {len(facts.attributes)} 条商品属性、{len(facts.skus)} 个 SKU、"
            f"{len(facts.all_image_urls())} 个不重复源图片 URL。",
            f"从源详情图核验并结构化 {len(facts.size_chart_rows)} 行服装尺码数据。",
            f"内部 Claim Ledger 共 {len(state.claim_ledger)} 条；每条记录来源类型、原始字段和证据指针，"
            "买家文案只开放 allowed_surfaces 包含 buyer_copy 的声明。",
            "",
            "### Claim Ledger（可发布声明与证据）",
            "",
            "| Claim ID | 声明概念 | 原始值 | 来源类型 | 来源字段 | 证据指针 |",
            "|---|---|---|---|---|---|",
            *[
                "| "
                + " | ".join(
                    (
                        item.claim_id,
                        brief(item.concept).replace("|", "\\|"),
                        (
                            "[卖家标题原文已在内部账本保留]"
                            if item.source_type == "seller_title"
                            else brief(item.value).replace("|", "\\|")
                        ),
                        item.source_type,
                        brief(item.source_name).replace("|", "\\|"),
                        brief(item.evidence_pointer).replace("|", "\\|"),
                    )
                )
                + " |"
                for item in state.claim_ledger
                if "buyer_copy" in item.allowed_surfaces
            ],
            *(
                ["", "模型返回的买家文案声明引用：", *copy_claim_reference_lines]
                if copy_claim_reference_lines
                else []
            ),
            "",
            "## 3. AliExpress 类目与属性策略",
            "",
            f"- 叶子类目 ID：{taxonomy.category.category_id}",
            f"- 叶子类目名称：{taxonomy.category.name}",
            f"- 类目路径：{taxonomy.category.path}",
            f"- 决策方式：{taxonomy.category.method}",
            f"- 置信度：{taxonomy.category.confidence:.2f}",
            f"- 命中的平台商品/销售属性数：{len(taxonomy.attributes)}",
            "",
            "类目采用“源类目同义词精确映射 → 性别/年龄/品类规则过滤 → 本地叶子节点排序”的确定性优先流程，"
            "父级属性元数据不进入可选类目集合。属性值只从对应类目允许的枚举中映射；多季节值按平台多选枚举逐项"
            "展开，缺失的必填值会明确保留为空。",
            schema_note,
            "",
            "## 4. 本地化策略",
            "",
            "- 英文按 en-US 电商语气编写，涉及体重时同时给出 lb。",
            "- 英文中的厘米规格同时给出确定性英寸换算；韩文与巴西葡萄牙文保留当地常用公制。",
            "- 韩文按 ko-KR 自然购物语气编写，避免机械直译和未经证实的韩国尺码映射。",
            "- 葡萄牙文按 pt-BR 编写，避免欧洲葡语表达和未经证实的 P/M/G 映射。",
            "- 三份文案共享同一个不可变商品 ID、URL、叶子类目、属性和完整 SKU 表。",
            "- 买家文案按 Feature → 可见结构优势 → 保守购买价值组织；任一步缺少事实或像素证据时停止延伸。",
            "- 买家段落与机器字段分层：前半部分负责自然销售表达，后半部分保留精确 ID、SKU 与来源类型。",
            "- 卖家提供的体重范围只按完全匹配的尺码标签写入对应 SKU，并同时展示 kg/lb，不推导地区尺码。",
            "- 可辨识的源尺码表先由视觉模型转录，再按SKU尺码代码、来源图角色和数值范围进行确定性校验；"
            "英文显示 cm/in 与 kg/lb，韩文和巴西葡萄牙文保留 cm/kg。",
            f"- 文案生成来源：{json.dumps(localization_sources, ensure_ascii=False)}",
            "",
            "## 5. 图片与视频生成策略",
            "",
            f"- 本次执行模式：{execution_mode}。",
            f"- Campaign Style Lock：{state.creative_plan.visual_theme}",
            f"- 创意计划来源：{plan_model}",
            f"- 模型配置：{json.dumps(model_summary, ensure_ascii=False)}",
            "- 主图采用方形浅色棚拍构图；优先生成三个候选，再按身份、结构、颜色、完整度、干净背景、单品覆盖和瑕疵自动选优。",
            "- 主图回退源图须优先满足无人物、无关道具、单一完整商品和干净中性背景；没有合格源图时明确记录质量降级。",
            "- 五张详情图各承担唯一商业任务：整体轮廓、领口/门襟、袖口/垂感、背面/下摆或真实变体、使用情境；"
            "感知哈希不同但任务重复的图片仍视为分镜失败。若源详情图存在可核验尺码表，"
            "第5张改为确定性重绘的干净尺码图。",
            "- 高风险的变体图与穿着场景各生成两个候选，并按商品身份、颜色、图案、结构、人体与分镜匹配自动选优。",
            "- 视频以最终主图或其源 URL 为首帧，按上装、下装、连衣裙或童装使用不同结构保护镜头；默认移除未审核音轨。",
            f"- 本次模型直接生成并通过校验的素材数：{generated_count}。",
            "",
            "## 6. 有界 Agent 规划、评估与定向修复",
            "",
            "交付管理器先依据事实账本、A1–A7 权重、源图观察和可用工具生成本次执行策略。初稿完成后，"
            "独立多模态评估器读取全套文案 payload、类目属性、素材清单、图片/视频及本地物理检查，输出逐维度"
            "评分、证据化问题和白名单工具调用。控制循环最多执行两轮，只修改被点名的单个素材。",
            "所有修复均先写入临时文件，完成候选语义选优、文案事实/schema 校验或视频播放校验后才原子替换；"
            "修复失败会保留上一版，不会因为评估意见自动降级为源图或幻灯片。",
            "- LLM 自由文本计划仅用于内部生成提示，不作为商品事实写入交付；策略文档只披露经过白名单筛选的控制参数。",
            f"- Agent 控制参数：{json.dumps(agent_plan_controls, ensure_ascii=False)}",
            f"- 已完成全局评估轮次：{len(state.agent_evaluations)}。",
            f"- 已执行定向修复工具调用：{len(state.agent_actions)}。",
            f"- {semantic_gate_note}",
            "",
            "## 7. 合规与质检",
            "",
            "生成提示词和最终文案均通过平台内容规则门禁。图片下载后统一解码为 RGB JPEG，并校验尺寸、"
            "文件大小、空白图和近重复图；全局评估时模型生成图与可信源图共同输入独立视觉评估，检查商品身份、"
            "具体结构、分镜覆盖、意外文字、水印和重大瑕疵。视频除容器和 200MB 上限外，还须完成全视频流解码，"
            "生成视频同时进入源图对照的时序语义评估。"
            "所有输出在写入最终目录前进行一次完整交付质检，写入后再次复核。",
            "源图检查区分商品本身的固有设计与背景营销元素；不适合发布的视觉内容不会进入生成参考或优先回退素材。"
            "视频语义质检缺失、超时或字段不完整时按失败处理。",
            "主图与全部详情图共同执行感知哈希去重；详情图等比保留商品主体时使用低对比度模糊延展背景，"
            "避免大块纯色填边。回退视频只使用感知上不同的最终图片，每个镜头被显式裁成有限时长后再拼接。",
            "",
            "## 8. 降级与稳定性",
            "",
            "API 请求对限流和暂时性错误执行指数退避；图片优先走同步多模态生成，视频异步任务保存 task_id 并轮询。"
            "只有初次图片模型不可用、没有可接受候选或下载失败时，才用经规格归一化的安全商品源图保证完整交付；"
            "只有初次视频生成不可用时，才使用最终图片集生成多镜头 H.264 商品展示视频。全局评估不会触发回退，"
            "而是调用定向重做/修改工具；重做失败时保留原成品。",
            f"- 本次 API 调用记录数：{len(state.api_calls)}；每次调用均记录模型、耗时、状态及调用后的剩余时间。",
            f"- 失败 API 调用数：{len(failed_calls)}。",
        ]
        lines.extend(["", "本次实际媒体结果：", ""])
        lines.extend(
            f"- {asset.name}：{asset.model}；{asset.description or '未提供说明'}。"
            for asset in state.assets
        )
        if state.agent_evaluations:
            lines.extend(["", "全局评估轨迹：", ""])
            lines.extend(
                f"- 第 {item.round_index + 1} 轮：加权分 {item.weighted_score:.1f}；"
                f"ready={item.ready_for_delivery}；{brief(item.summary)}"
                for item in state.agent_evaluations
            )
        if state.agent_actions:
            lines.extend(["", "定向修复轨迹：", ""])
            lines.extend(
                f"- 第 {item.round_index + 1} 轮 {item.tool}/{item.target}："
                f"{item.status}；{brief(item.detail)}"
                for item in state.agent_actions
            )
        if fallback_assets:
            lines.extend(["", "本次发生的素材回退：", ""])
            lines.extend(
                f"- {asset.name}：{asset.model}；原因：{brief(asset.fallback_reason or '模型产物不可用')}。"
                for asset in fallback_assets
            )
        if failed_calls:
            lines.extend(["", "API 失败摘要：", ""])
            lines.extend(
                f"- {item.get('operation', 'unknown')}/{item.get('model', 'unknown')}："
                f"{brief(str(item.get('error') or item.get('status') or 'error'))}"
                for item in failed_calls[:12]
            )
        if state.warnings:
            lines.extend(
                [
                    "",
                    "运行质检记录：",
                    "",
                    f"- 共记录 {len(state.warnings)} 项内部质检事件。",
                ]
            )
            lines.extend(f"- {brief(item)}" for item in state.warnings[:16])
        lines.append("")
        (work_dir / "strategy_document.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def _commit_delivery(self, work_dir: Path) -> None:
        for filename in sorted(EXPECTED_FILES):
            source = work_dir / filename
            destination = self.output_dir / filename
            os.replace(source, destination)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


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
                if by_index.get(index, {}).get("safe_for_generation_reference")
                is True
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


def _even_sample(values: list[str], count: int) -> list[str]:
    if len(values) <= count:
        return list(values)
    if count <= 1:
        return [values[0]]
    indices = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return [values[index] for index in indices]
