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
from .compliance import normalize_source_image_observations
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


_IMAGE_NEGATIVE_PROMPT = (
    "written text, letters, numbers, watermark, logo, brand mark, price tag, promotional badge, "
    "unreadable typography, distorted anatomy, extra limbs, malformed hands, product deformation, "
    "changed buttons, changed fasteners, changed pattern, changed color, blur, low resolution"
)

_MAIN_NEGATIVE_PROMPT = (
    _IMAGE_NEGATIVE_PROMPT
    + ", collage, montage, split screen, inset, duplicate garment, multiple products, multiple colorways, "
    "cropped product, person, mannequin body"
)

_VIDEO_NEGATIVE_PROMPT = (
    "product morphing, changed garment construction, changed color, changed pattern, added pockets, "
    "added buttons, duplicate product, extra garment, warped fabric, flicker, scene cut, camera shake, "
    "hands covering product, text, subtitles, watermark, logo animation, speech, music"
)


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
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.logger = logger
        self.product_id = product_id
        self.offline = offline
        self.started_monotonic = time.monotonic()
        self.deadline = self.started_monotonic + timeout_seconds
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
            QwenClient(self.api_config, logger, self.deadline)
            if self.api_config
            else None
        )
        self.downloader = (
            self.client.http if self.client else HttpJsonClient(logger, self.deadline)
        )
        self.skills = SkillLibrary()
        self.agent = BoundedDeliveryAgent(self.client, logger, self.skills)
        self.warnings: list[str] = []
        self._raw_counter = 0
        self._raw_counter_lock = threading.Lock()
        self._source_image_observations: dict[str, dict[str, Any]] = {}
        self._source_selection_warnings: set[str] = set()

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
            category_tree = load_json(categories_path)
            attribute_data = load_json(attributes_path)
            taxonomy = resolve_taxonomy(facts, category_tree, attribute_data)
            taxonomy = self._adjudicate_taxonomy(
                facts, taxonomy, category_tree, attribute_data
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
            self._apply_size_chart_observations(facts, vision)
            tool_registry = self._create_tool_registry(
                protect_size_chart=bool(facts.size_chart_rows)
            )
            agent_plan = self.agent.plan_delivery(
                facts, taxonomy, vision, tool_registry
            )
            creative_plan, plan_model = create_creative_plan(
                facts,
                taxonomy,
                vision,
                self.client,
                agent_guidance=agent_plan,
                skill_instructions=self.skills.combine(
                    "product-grounding",
                    "aliexpress-content-compliance",
                    "commerce-visuals",
                    "commerce-video",
                ),
            )
            self.logger.info("创意计划来源: %s", plan_model)

            state = RunState(
                started_at=datetime.now(timezone.utc).isoformat(),
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir),
                facts=facts,
                taxonomy=taxonomy,
                creative_plan=creative_plan,
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
                max_workers=7, thread_name_prefix="asset"
            ) as executor:
                video_future = executor.submit(
                    self._build_video,
                    facts,
                    creative_plan,
                    main_reference_url,
                    Path(main_asset.path),
                    work_dir,
                    downloads_dir,
                    main_asset.generated
                    or self._safe_generation_reference(main_reference_url),
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
                        agent_guidance=str(
                            agent_plan.get("localization_priorities", {}).get(
                                language, ""
                            )
                        ),
                        skill_instructions=self.skills.combine(
                            "product-grounding",
                            "aliexpress-content-compliance",
                            "marketplace-localization",
                        ),
                    ): language
                    for language in ("en", "ko", "pt")
                }

                for future, index in detail_futures.items():
                    try:
                        detail_assets[index] = future.result()
                    except Exception as exc:
                        raise PipelineError(f"详情图 {index} 构建失败: {exc}") from exc
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

            for index in range(1, 6):
                state.assets.append(detail_assets[index])
            if video_result:
                state.assets.append(video_result)

            # Initial generation failures may use deterministic emergency assets so the
            # delivery remains complete. Evaluation never replaces an accepted artifact
            # with a fallback: it selects a targeted, non-destructive repair tool below.
            self._install_size_chart_detail(facts, state.assets, work_dir)
            self._repair_duplicate_fallback_details(
                state.assets,
                main_reference_url=main_reference_url,
                work_dir=work_dir,
                downloads_dir=downloads_dir,
            )
            self._enhance_fallback_video(state.assets, work_dir)
            self._record_visual_delivery_quality(state.assets)
            self._write_localized_descriptions(
                facts, taxonomy, localization_payloads, state.assets, work_dir
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
            self._run_bounded_agent_loop(
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
            # A catalog video is rebuilt from the final image set only when video
            # generation was unavailable from the outset; this is not an evaluator fallback.
            self._enhance_fallback_video(state.assets, work_dir)
            self._record_visual_delivery_quality(state.assets)
            self._write_localized_descriptions(
                facts, taxonomy, localization_payloads, state.assets, work_dir
            )
            if self.client is not None:
                state.api_calls = self.client.metrics
            self._write_strategy_document(
                state, localization_sources, plan_model, work_dir
            )

            report = validate_delivery(work_dir, facts, taxonomy)
            for warning in report.warnings:
                self.logger.warning("交付告警: %s", warning)
            if not report.valid:
                raise PipelineError("交付校验失败: " + "; ".join(report.errors))

            self._commit_delivery(work_dir)
            final_report = validate_delivery(self.output_dir, facts, taxonomy)
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
        payloads: dict[str, dict[str, Any]],
        assets: list[AssetResult],
        work_dir: Path,
    ) -> None:
        asset_by_name = {asset.name: asset for asset in assets}
        fallback_templates = {
            "en": {
                "main": "Seller-source hero image normalized to a square listing format.",
                "details": [
                    "Alternate seller-source view showing the complete product.",
                    "Additional seller-source view showing the construction from another angle.",
                    "Additional seller-source view showing the silhouette and hem.",
                    "Full product reference showing the fit and proportions.",
                    "Detail crop derived from the seller's product photography.",
                ],
                "crops": {
                    "upper": "Seller-source close-up showing the neckline, print and upper construction.",
                    "lower": "Seller-source close-up showing the hem, print and lower proportions.",
                    "left": "Seller-source close-up showing the left-side sleeve and print details.",
                    "right": "Seller-source close-up showing the right-side sleeve and print details.",
                    "center": "Seller-source close-up showing the central print and garment construction.",
                },
                "video": "Eight-second silent catalog video assembled from the final distinct product images.",
                "single_video": "Eight-second product presentation with restrained camera motion.",
                "size_chart": "Size chart showing the seller-provided garment measurements and weight guidance.",
            },
            "ko": {
                "main": "판매자 원본을 정사각형 등록 규격에 맞춰 정리한 대표 이미지입니다.",
                "details": [
                    "상품 전체를 보여 주는 판매자 원본의 다른 이미지입니다.",
                    "다른 각도에서 상품 구조를 보여 주는 판매자 원본 이미지입니다.",
                    "실루엣과 밑단을 보여 주는 추가 판매자 원본 이미지입니다.",
                    "핏과 비율을 확인할 수 있는 상품 전체 이미지입니다.",
                    "판매자 상품 사진에서 잘라낸 디테일 이미지입니다.",
                ],
                "crops": {
                    "upper": "네크라인과 프린트, 상단 구조를 보여 주는 판매자 원본 클로즈업입니다.",
                    "lower": "밑단과 프린트, 하단 비율을 보여 주는 판매자 원본 클로즈업입니다.",
                    "left": "왼쪽 소매와 프린트 디테일을 보여 주는 판매자 원본 클로즈업입니다.",
                    "right": "오른쪽 소매와 프린트 디테일을 보여 주는 판매자 원본 클로즈업입니다.",
                    "center": "중앙 프린트와 의류 구조를 보여 주는 판매자 원본 클로즈업입니다.",
                },
                "video": "서로 다른 최종 상품 이미지로 구성한 8초 무음 카탈로그 영상입니다.",
                "single_video": "절제된 카메라 움직임을 적용한 8초 단일 이미지 상품 영상입니다.",
                "size_chart": "판매자가 제공한 의류 실측과 권장 체중을 보여 주는 사이즈표입니다.",
            },
            "pt": {
                "main": "Imagem principal da fonte do vendedor adaptada ao formato quadrado do anúncio.",
                "details": [
                    "Outra foto da fonte do vendedor mostrando o produto por inteiro.",
                    "Foto adicional da fonte do vendedor mostrando a construção em outro ângulo.",
                    "Foto adicional mostrando a silhueta e a barra da peça.",
                    "Referência do produto por inteiro mostrando caimento e proporções.",
                    "Recorte de detalhe derivado das fotos de produto do vendedor.",
                ],
                "crops": {
                    "upper": "Close da fonte do vendedor mostrando decote, estampa e construção superior.",
                    "lower": "Close da fonte do vendedor mostrando barra, estampa e proporções inferiores.",
                    "left": "Close da fonte do vendedor mostrando a manga esquerda e detalhes da estampa.",
                    "right": "Close da fonte do vendedor mostrando a manga direita e detalhes da estampa.",
                    "center": "Close da fonte do vendedor mostrando a estampa central e a construção da peça.",
                },
                "video": "Vídeo de catálogo silencioso de 8 segundos montado com as imagens finais distintas do produto.",
                "single_video": "Apresentação de 8 segundos com uma única imagem e movimento de câmera discreto.",
                "size_chart": "Tabela com as medidas da peça e o peso indicados pelo vendedor.",
            },
        }
        for language, payload in payloads.items():
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
                if asset is None or asset.generated:
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
        work_dir: Path,
    ) -> None:
        if self.client is None:
            self.warnings.append(
                "显式离线模式：跳过有界 Agent 全局评估循环"
                if self.offline
                else "模型配置不可用，跳过有界 Agent 全局评估循环"
            )
            return
        max_repairs = int(agent_plan.get("max_repair_rounds", 1))
        repairs_used = 0
        round_index = 0
        while True:
            if self.deadline - time.monotonic() <= 4 * 60:
                self.warnings.append("剩余时间不足，停止新的 Agent 评估轮次并保留当前版本")
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
                work_dir=work_dir,
                tools=registry,
            )
            if evaluation is None:
                self.warnings.append("LLM 全局评估未完成；不触发回退，保留当前已校验素材")
                break
            state.agent_evaluations.append(evaluation)
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
        staged = work_dir / f".repair-main-{uuid.uuid4().hex}.jpeg"
        prompt = (
            plan.main_prompt
            + "\nIndependent evaluator correction for this revision: "
            + instruction
            + "\nCorrect only the identified defect and preserve all verified product features."
        )
        try:
            candidate_urls, model = self.client.generate_image_candidates(
                prompt,
                source_urls[:1],
                size="1600*1600",
                negative_prompt=_MAIN_NEGATIVE_PROMPT,
                count=3,
            )
            selected = self._select_main_candidate(
                facts, source_urls[:3], candidate_urls
            )
            self._download_and_normalize(
                selected,
                staged,
                downloads_dir,
                canvas=(1600, 1600),
                white_background=True,
            )
            asset = self._find_asset(assets, target)
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
        staged = work_dir / f".repair-detail-{index}-{uuid.uuid4().hex}.jpeg"
        prompt = (
            plan.detail_prompts[index - 1]
            + "\nIndependent evaluator correction for this revision: "
            + instruction
            + "\nCorrect only the identified defect; keep the intended slot and exact product identity."
        )
        try:
            candidate_urls, model = self.client.generate_image_candidates(
                prompt,
                references[:3],
                size="1200*1500",
                negative_prompt=_IMAGE_NEGATIVE_PROMPT,
                count=2,
            )
            selected = self._select_detail_candidate(
                index, facts, references[:3], candidate_urls, prompt
            )
            self._download_and_normalize(
                selected,
                staged,
                downloads_dir,
                canvas=(1200, 1500),
                white_background=False,
            )
            asset = self._find_asset(assets, target)
            os.replace(staged, Path(asset.path))
            asset.source_url = selected
            asset.model = f"{model}-agent-repair"
            asset.generated = True
            asset.fallback_reason = ""
            asset.description = (
                f"Agent-repaired detail storyboard slot {index}: {instruction[:240]}"
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
            skill_instructions=self.skills.combine(
                "product-grounding",
                "aliexpress-content-compliance",
                "marketplace-localization",
            ),
        )
        if not source.startswith(self.client.config.chat_model):
            return ToolExecution(
                "failed",
                f"localized revision did not pass model/schema audit ({source}); prior copy preserved",
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
            asset = self._find_asset(assets, target)
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

    def _adjudicate_taxonomy(
        self,
        facts: ProductFacts,
        taxonomy: TaxonomyResult,
        category_tree: dict[str, Any],
        attribute_data: dict[str, Any],
    ) -> TaxonomyResult:
        if self.client is None or taxonomy.category.confidence >= 0.85:
            return taxonomy
        candidates = taxonomy.category.candidates[:12]
        allowed_ids = {str(item.get("category_id")) for item in candidates}
        if not allowed_ids:
            return taxonomy
        system = (
            "You are a conservative AliExpress apparel taxonomy classifier. Return JSON only. "
            "You must choose exactly one supplied leaf category ID. Do not invent an ID.\n\n"
            + self.skills.combine("product-grounding", "aliexpress-taxonomy")
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
            return taxonomy
        selected_id = str(response.get("selected_category_id") or "")
        if selected_id not in allowed_ids:
            self.logger.warning("类目裁决返回候选外 ID，已忽略: %s", selected_id)
            return taxonomy
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
        preferred = facts.product_image_urls[:5]
        sku_sample = _even_sample(facts.sku_image_urls, 4)
        description_sample = _even_sample(facts.description_image_urls, 3)
        urls = _unique(preferred + sku_sample + description_sample)[:12]
        try:
            result = self.client.analyze_product_images(
                json.dumps(facts.compact_dict(), ensure_ascii=False),
                urls,
                skill_instructions=self.skills.combine(
                    "product-grounding",
                    "aliexpress-content-compliance",
                    "commerce-visuals",
                ),
            )
            source_images = normalize_source_image_observations(result, urls)
            result["source_images"] = source_images
            self._source_image_observations = {
                str(item["url"]): item for item in source_images
            }
            rejected = sum(
                not item.get("safe_for_generation_reference", False)
                for item in source_images
            )
            self.logger.info("完成 %d 张源图片的视觉理解", len(urls))
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
        except ApiError as exc:
            self.logger.warning("源图片视觉理解失败，使用结构化事实继续: %s", exc)
            self.warnings.append(f"源图片视觉理解失败: {exc}")
            return {}

    def _apply_size_chart_observations(
        self, facts: ProductFacts, vision: dict[str, Any]
    ) -> None:
        """Promote only clearly structured, SKU-aligned visual measurements to facts."""

        raw_rows = vision.get("size_chart_rows") if isinstance(vision, dict) else None
        source_images = vision.get("source_images") if isinstance(vision, dict) else None
        if not isinstance(raw_rows, list) or not isinstance(source_images, list):
            return

        known_codes: list[str] = []
        for sku in facts.skus:
            for item in sku.attributes:
                if "尺码" not in item.name and "size" not in item.name.casefold():
                    continue
                match = re.match(r"\s*([A-Za-z0-9]+)", item.value)
                if match and match.group(1).upper() not in known_codes:
                    known_codes.append(match.group(1).upper())
        if not known_codes:
            return

        aliases = {
            "XXL": "2XL",
            "XXXL": "3XL",
            "XXXXL": "4XL",
        }
        image_by_index = {
            item.get("index"): item
            for item in source_images
            if isinstance(item, dict) and isinstance(item.get("index"), int)
        }

        def measurement(value: Any) -> str:
            match = re.fullmatch(
                r"\s*(\d{1,3}(?:\.\d+)?)\s*(?:cm)?\s*", str(value or ""), re.I
            )
            if not match:
                return ""
            numeric = float(match.group(1))
            return match.group(1) if 20 <= numeric <= 300 else ""

        conversions: dict[str, tuple[str, str]] = {}
        for item in facts.size_conversions:
            match = re.match(r"\s*([A-Za-z0-9]+)", item.source_label)
            if match:
                conversions[match.group(1).upper()] = (item.kilograms, item.pounds)

        rows: list[SizeChartRow] = []
        seen: set[str] = set()
        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            raw_code = str(raw.get("size_label") or "").strip().upper()
            code = aliases.get(raw_code, raw_code)
            bust = measurement(raw.get("bust_cm"))
            length = measurement(raw.get("length_cm"))
            source_index = raw.get("source_image_index")
            source_item = image_by_index.get(source_index)
            if (
                code not in known_codes
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
            seen.add(code)
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
        ranked_primary = self._source_urls_for_use(
            primary, use="fallback", preferred_roles=preferred
        )
        usable_primary = [
            url
            for url in ranked_primary
            if self._source_image_observations.get(url, {}).get("role")
            not in {"size_chart", "packaging"}
            and not self._source_image_observations.get(url, {}).get(
                "has_overlay_text", False
            )
            and not self._terminal_fallback_risks(
                self._source_image_observations.get(url, {})
            )
        ]
        if usable_primary:
            return usable_primary
        ranked_description = self._source_urls_for_use(
            facts.description_image_urls,
            use="fallback",
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
            and not self._terminal_fallback_risks(
                self._source_image_observations.get(url, {})
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

        Full source views are exhausted before bounded crops are introduced. The
        hero reference is deliberately placed after alternate source views so the
        first details do not immediately repeat the main image.
        """

        sources = self._fallback_source_urls(
            facts, asset_name=f"detail_image_{index}.jpeg"
        )
        if not sources:
            return [], ""
        ordered = [url for url in sources if url != main_reference_url]
        if main_reference_url in sources:
            ordered.append(main_reference_url)
        if not ordered:
            ordered = list(sources)

        if index <= len(ordered):
            selected = ordered[index - 1]
            return [selected] + [url for url in ordered if url != selected], ""

        crop_sequence = ("upper", "lower", "left", "right", "center")
        crop_index = index - len(ordered) - 1
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
                candidate_urls, model = self.client.generate_image_candidates(
                    plan.main_prompt,
                    source_urls[:1],
                    size="1600*1600",
                    negative_prompt=_MAIN_NEGATIVE_PROMPT,
                    count=3,
                )
                generated_url = self._select_main_candidate(
                    facts, source_urls[:3], candidate_urls
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

    def _select_main_candidate(
        self,
        facts: ProductFacts,
        source_urls: list[str],
        candidate_urls: list[str],
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
            self.logger.warning("主图候选自动选优不可用，拒绝未经语义验收的候选: %s", exc)
            self.warnings.append(f"主图候选自动选优不可用: {exc}")
            raise ApiError("主图候选无法完成语义验收") from exc

        candidates = review.get("candidates")
        selected = review.get("selected_index")
        if isinstance(candidates, list):
            usable_indices = {
                item.get("index")
                for item in candidates
                if isinstance(item, dict)
                and isinstance(item.get("index"), int)
                and 0 <= item["index"] < len(candidate_urls)
                and item.get("usable") is True
                and item.get("identity_consistent") is True
                and item.get("construction_consistent") is True
                and item.get("correct_color") is True
                and item.get("single_product") is True
                and item.get("product_complete") is True
                and item.get("clean_neutral_background") is True
                and item.get("has_person") is not True
                and item.get("has_unrelated_props") is not True
                and item.get("unwanted_text") is not True
                and item.get("unwanted_brand_or_logo") is not True
                and item.get("major_artifacts") is not True
            }
        else:
            usable_indices = set()
        if isinstance(selected, int) and selected in usable_indices:
            if 0 <= selected < len(candidate_urls):
                self.logger.info(
                    "主图候选自动选优: 选择 %d/%d", selected + 1, len(candidate_urls)
                )
                return candidate_urls[selected]
        if usable_indices:
            fallback_index = min(usable_indices)
            return candidate_urls[fallback_index]
        raise ApiError("主图候选均未通过商品身份与结构一致性质检")

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
                # Every generated slot gets model-ranked alternatives. This moves
                # semantic acceptance to the LLM instead of accepting a lone output.
                candidate_count = 2
                candidate_urls, model = self.client.generate_image_candidates(
                    plan.detail_prompts[index - 1],
                    reference_selection[:3],
                    size="1200*1500",
                    negative_prompt=_IMAGE_NEGATIVE_PROMPT,
                    count=candidate_count,
                )
                generated_url = self._select_detail_candidate(
                    index,
                    facts,
                    reference_selection[:3],
                    candidate_urls,
                    plan.detail_prompts[index - 1],
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
                        f"Detail storyboard slot {index}: "
                        f"{plan.detail_prompts[index - 1][:240]}"
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

    def _select_detail_candidate(
        self,
        index: int,
        facts: ProductFacts,
        source_urls: list[str],
        candidate_urls: list[str],
        purpose: str,
    ) -> str:
        if not candidate_urls:
            raise ApiError(f"详情图 {index} 模型未返回候选")
        # A single generated detail candidate still needs semantic acceptance;
        # otherwise structure drift can pass just because there is no alternative.
        if self.client is None:
            return candidate_urls[0]
        review = self.client.select_best_detail_image(
            json.dumps(facts.compact_dict(), ensure_ascii=False),
            source_urls,
            candidate_urls,
            asset_name=f"detail_image_{index}.jpeg",
            purpose=purpose,
        )
        candidates = review.get("candidates")
        selected = review.get("selected_index")
        usable: set[int] = set()
        if isinstance(candidates, list):
            for item in candidates:
                if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                    continue
                candidate_index = item["index"]
                if not 0 <= candidate_index < len(candidate_urls):
                    continue
                if (
                    item.get("usable") is True
                    and item.get("identity_consistent") is not False
                    and item.get("construction_consistent") is not False
                    and item.get("color_consistent") is not False
                    and item.get("pattern_consistent") is not False
                    and item.get("slot_match") is True
                    and item.get("critical_structure_unambiguous") is not False
                    and item.get("anatomy_natural") is not False
                    and item.get("unwanted_text") is not True
                    and item.get("unwanted_brand_or_logo") is not True
                    and item.get("prohibited_visual") is not True
                    and item.get("major_artifacts") is not True
                    and str(item.get("product_coverage") or "").lower() != "low"
                ):
                    usable.add(candidate_index)
        if isinstance(selected, int) and selected in usable:
            return candidate_urls[selected]
        if usable:
            return candidate_urls[min(usable)]
        raise ApiError(f"详情图 {index} 候选均未通过语义质检")

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
        asset.source_url = ""
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
        crop_sequence = ("upper", "lower", "left", "right", "center")
        for asset in ordered:
            try:
                quality = inspect_image_quality(Path(asset.path))
            except MediaError:
                quality = None
            if quality is None:
                continue
            if all(
                hash_distance(quality.difference_hash, seen) > 10
                for seen in accepted_hashes
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
                    f"离线出图可用率暂按{estimate_basis}估算为 {usable_rate:.0%}；"
                    "正式交付仍需模型语义门禁确认背景、人物道具与商品身份"
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

    def _write_strategy_document(
        self,
        state: RunState,
        localization_sources: dict[str, str],
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
            "- 卖家提供的体重范围只按完全匹配的尺码标签写入对应 SKU，并同时展示 kg/lb，不推导地区尺码。",
            "- 可辨识的源尺码表先由视觉模型转录，再按SKU尺码代码、来源图角色和数值范围进行确定性校验；"
            "英文显示 cm/in 与 kg/lb，韩文和巴西葡萄牙文保留 cm/kg。",
            f"- 文案生成来源：{json.dumps(localization_sources, ensure_ascii=False)}",
            "",
            "## 5. 图片与视频生成策略",
            "",
            "- 视觉主题：中性背景、商品优先、跨市场一致的电商摄影。",
            f"- 创意计划来源：{plan_model}",
            f"- 模型配置：{json.dumps(model_summary, ensure_ascii=False)}",
            "- 主图采用方形浅色棚拍构图；优先生成三个候选，再按身份、结构、颜色、完整度、干净背景、单品覆盖和瑕疵自动选优。",
            "- 主图回退源图须优先满足无人物、无关道具、单一完整商品和干净中性背景；没有合格源图时明确记录质量降级。",
            "- 五张详情图优先覆盖整体展示、领口/门襟、袖口/垂感、真实变体和使用情境；若源详情图存在可核验尺码表，"
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


def _even_sample(values: list[str], count: int) -> list[str]:
    if len(values) <= count:
        return list(values)
    if count <= 1:
        return [values[0]]
    indices = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return [values[index] for index in indices]
