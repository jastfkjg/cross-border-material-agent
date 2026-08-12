"""End-to-end bounded agent orchestration."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .api import ApiConfig, ApiError, HttpJsonClient, QwenClient
from .input_loader import discover_input_files, load_json, load_product_facts
from .localization import generate_copy_payload, render_description
from .media import MediaError, create_slideshow_video, inspect_video, normalize_image
from .models import AssetResult, CreativePlan, ProductFacts, RunState, TaxonomyResult
from .planning import create_creative_plan
from .qa import EXPECTED_FILES, validate_delivery
from .taxonomy import resolve_taxonomy


class PipelineError(RuntimeError):
    """Raised when the agent cannot produce a complete, validated delivery."""


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
        self.started_monotonic = time.monotonic()
        self.deadline = self.started_monotonic + timeout_seconds
        self.api_config = None if offline else ApiConfig.from_environment()
        self.client = (
            QwenClient(self.api_config, logger, self.deadline)
            if self.api_config
            else None
        )
        self.downloader = (
            self.client.http if self.client else HttpJsonClient(logger, self.deadline)
        )
        self.warnings: list[str] = []
        self._raw_counter = 0
        self._raw_counter_lock = threading.Lock()

    def _ensure_time(self, reserve_seconds: float = 0) -> None:
        remaining = self.deadline - time.monotonic()
        if remaining <= reserve_seconds:
            raise PipelineError(f"剩余运行时间不足，需要保留 {reserve_seconds:.0f} 秒")

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
            creative_plan, plan_model = create_creative_plan(
                facts, taxonomy, vision, self.client
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
            )

            main_asset, main_reference_url = self._build_main_image(
                facts, creative_plan, vision, work_dir, downloads_dir
            )
            state.assets.append(main_asset)

            localization_sources: dict[str, str] = {}
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
                    description = render_description(language, payload, facts, taxonomy)
                    (work_dir / f"product_description_{language}.md").write_text(
                        description, encoding="utf-8"
                    )
                try:
                    video_result = video_future.result()
                except Exception as exc:
                    raise PipelineError(f"视频构建失败: {exc}") from exc

            for index in range(1, 6):
                state.assets.append(detail_assets[index])
            if video_result:
                state.assets.append(video_result)

            self._review_generated_assets(facts, state.assets, work_dir, downloads_dir)
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
            "You must choose exactly one supplied leaf category ID. Do not invent an ID."
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
            self.warnings.append("模型配置不可用，跳过源图片视觉理解")
            return {}
        self._ensure_time(10 * 60)
        preferred = facts.product_image_urls[:5]
        description_sample = _even_sample(facts.description_image_urls, 5)
        urls = _unique(preferred + facts.sku_image_urls[:2] + description_sample)
        try:
            result = self.client.analyze_product_images(
                json.dumps(facts.compact_dict(), ensure_ascii=False), urls
            )
            self.logger.info("完成 %d 张源图片的视觉理解", len(urls))
            return result
        except ApiError as exc:
            self.logger.warning("源图片视觉理解失败，使用结构化事实继续: %s", exc)
            self.warnings.append(f"源图片视觉理解失败: {exc}")
            return {}

    def _ordered_source_urls(
        self, facts: ProductFacts, vision: dict[str, Any]
    ) -> list[str]:
        ordered = list(facts.product_image_urls)
        best = vision.get("best_hero_image_index") if isinstance(vision, dict) else None
        if isinstance(best, int) and 0 <= best < len(ordered):
            ordered.insert(0, ordered.pop(best))
        return _unique(ordered + facts.sku_image_urls + facts.description_image_urls)

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
    ) -> None:
        raw_path = self._next_raw_path(downloads_dir, ".asset")
        self.downloader.download(url, raw_path, max_bytes=30 * 1024 * 1024, timeout=180)
        normalize_image(
            raw_path,
            destination,
            canvas=canvas,
            max_bytes=5 * 1024 * 1024,
            white_background=white_background,
        )

    def _fallback_image(
        self,
        source_urls: list[str],
        destination: Path,
        downloads_dir: Path,
        *,
        canvas: tuple[int, int],
        white_background: bool,
    ) -> str:
        errors: list[str] = []
        for url in source_urls:
            try:
                self._download_and_normalize(
                    url,
                    destination,
                    downloads_dir,
                    canvas=canvas,
                    white_background=white_background,
                )
                return url
            except (ApiError, MediaError) as exc:
                errors.append(str(exc))
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
        if self.client is not None:
            try:
                generated_url, model = self.client.generate_image(
                    plan.main_prompt,
                    source_urls[:3],
                    size="1600*1600",
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
                self.logger.warning("主图生成失败，使用源图回退: %s", exc)
                self.warnings.append(f"主图生成回退: {exc}")
        fallback_url = self._fallback_image(
            source_urls,
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
                fallback_reason="image generation unavailable or rejected",
                description="Source-faithful square hero image",
            ),
            fallback_url,
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
        source_urls = _unique(
            [main_reference_url]
            + facts.product_image_urls
            + facts.sku_image_urls
            + facts.description_image_urls
        )
        if self.client is not None:
            reference_selection = _unique(
                [main_reference_url]
                + facts.product_image_urls[:2]
                + _even_sample(facts.sku_image_urls, 2)
            )
            try:
                generated_url, model = self.client.generate_image(
                    plan.detail_prompts[index - 1],
                    reference_selection[:3],
                    size="1200*1500",
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
                    description=f"Detail storyboard slot {index}",
                )
            except (ApiError, MediaError) as exc:
                self.logger.warning("详情图 %d 生成失败，使用源图回退: %s", index, exc)
                self.warnings.append(f"详情图 {index} 生成回退: {exc}")

        rotated_sources = source_urls[index - 1 :] + source_urls[: index - 1]
        fallback_url = self._fallback_image(
            rotated_sources,
            destination,
            downloads_dir,
            canvas=(1200, 1500),
            white_background=False,
        )
        return AssetResult(
            name=destination.name,
            path=str(destination),
            source_url=fallback_url,
            model="deterministic-source-fallback",
            generated=False,
            fallback_reason="image generation unavailable or rejected",
            description=f"Source-faithful detail image {index}",
        )

    def _build_video(
        self,
        facts: ProductFacts,
        plan: CreativePlan,
        first_frame_url: str,
        main_image_path: Path,
        work_dir: Path,
        downloads_dir: Path,
    ) -> AssetResult:
        destination = work_dir / "product_video.mp4"
        if self.client is not None:
            try:
                video_url, model = self.client.generate_video(
                    plan.video_prompt, first_frame_url
                )
                raw_video = self._next_raw_path(downloads_dir, ".mp4")
                self.downloader.download(
                    video_url, raw_video, max_bytes=199 * 1024 * 1024, timeout=300
                )
                shutil.copyfile(raw_video, destination)
                inspect_video(destination)
                return AssetResult(
                    name=destination.name,
                    path=str(destination),
                    source_url=video_url,
                    model=model,
                    generated=True,
                    description="Short source-guided product video",
                )
            except (ApiError, MediaError, OSError) as exc:
                self.logger.warning("视频模型失败，创建确定性视频回退: %s", exc)
                self.warnings.append(f"视频生成回退: {exc}")
        create_slideshow_video(main_image_path, destination, duration=8)
        return AssetResult(
            name=destination.name,
            path=str(destination),
            model="ffmpeg-slideshow-fallback",
            generated=False,
            fallback_reason="video generation unavailable or invalid",
            description="Playable H.264 product presentation fallback",
        )

    def _review_generated_assets(
        self,
        facts: ProductFacts,
        assets: list[AssetResult],
        work_dir: Path,
        downloads_dir: Path,
    ) -> None:
        if self.client is None:
            return
        image_assets = [
            asset
            for asset in assets
            if asset.name.endswith(".jpeg") and asset.generated
        ]
        if not image_assets:
            return
        self._ensure_time(3 * 60)
        try:
            review = self.client.review_generated_images(
                json.dumps(facts.compact_dict(), ensure_ascii=False),
                [asset.source_url for asset in image_assets],
            )
        except ApiError as exc:
            self.logger.warning(
                "生成图片语义质检失败，保留已通过物理校验的图片: %s", exc
            )
            self.warnings.append(f"生成图片语义质检不可用: {exc}")
            return
        reviews = review.get("assets")
        if not isinstance(reviews, list):
            return
        source_urls = _unique(
            facts.product_image_urls
            + facts.sku_image_urls
            + facts.description_image_urls
        )
        main_was_rejected = False
        for item in reviews:
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                continue
            index = item["index"]
            if not 0 <= index < len(image_assets):
                continue
            rejected = (
                item.get("usable") is False
                or item.get("identity_consistent") is False
                or item.get("unwanted_text") is True
                or item.get("major_artifacts") is True
            )
            if not rejected:
                continue
            asset = image_assets[index]
            destination = Path(asset.path)
            is_main = asset.name == "main_image.jpeg"
            try:
                fallback_url = self._fallback_image(
                    source_urls,
                    destination,
                    downloads_dir,
                    canvas=(1600, 1600) if is_main else (1200, 1500),
                    white_background=is_main,
                )
            except PipelineError as exc:
                self.logger.warning(
                    "语义质检拒绝 %s，但源图回退失败: %s", asset.name, exc
                )
                continue
            asset.source_url = fallback_url
            asset.model = "deterministic-source-fallback"
            asset.generated = False
            asset.fallback_reason = (
                f"semantic QA rejected generated image: {item.get('reason', '')}"
            )
            self.warnings.append(f"{asset.name} 因语义质检回退到源图")
            if is_main:
                main_was_rejected = True
        if main_was_rejected:
            video_asset = next(
                (asset for asset in assets if asset.name == "product_video.mp4"), None
            )
            if video_asset and video_asset.generated:
                video_path = Path(video_asset.path)
                try:
                    create_slideshow_video(
                        work_dir / "main_image.jpeg", video_path, duration=8
                    )
                    video_asset.source_url = ""
                    video_asset.model = "ffmpeg-slideshow-fallback"
                    video_asset.generated = False
                    video_asset.fallback_reason = (
                        "main image semantic QA rejection invalidated generated video"
                    )
                    self.warnings.append("主图语义质检回退后，视频同步回退为源图展示")
                except MediaError as exc:
                    self.logger.warning("主图被拒后无法重建视频: %s", exc)

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
            else {"mode": "deterministic fallback"}
        )
        lines = [
            "# 商品本地化素材生成策略说明",
            "",
            "## 1. 本次商品与目标",
            "",
            f"- 商品 ID：{facts.offer_id}",
            f"- 数据来源：{facts.platform}",
            f"- 源商品标题：{facts.source_title}",
            f"- 源商品 URL：{facts.source_url}",
            "- 交付目标：英文、韩文、巴西葡萄牙文文案，1 张主图、5 张详情图、1 个商品视频。",
            "",
            "## 2. 事实一致性策略",
            "",
            "Agent 首先把商品 JSON 归一化为事实账本。标题、属性、SKU、图片 URL、商品 ID 和来源均保留证据位置；"
            "只有源 JSON、源图片直接观察或确定性单位换算得到的信息可以进入文案和素材提示词。"
            "模型不得补全面料功能、洗护方法、认证、品牌授权、价格、库存或地区尺码映射。",
            "",
            f"本次共读取 {len(facts.attributes)} 条商品属性、{len(facts.skus)} 个 SKU、"
            f"{len(facts.all_image_urls())} 个不重复源图片 URL。",
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
            "类目采用“源类目同义词精确映射 → 性别/年龄/品类规则过滤 → 本地叶子节点排序”的确定性优先流程。"
            "属性值只从对应类目允许的枚举中映射；缺失的必填值会明确保留为空，不进行事实编造。",
            "",
            "## 4. 本地化策略",
            "",
            "- 英文按 en-US 电商语气编写，涉及体重时同时给出 lb。",
            "- 韩文按 ko-KR 自然购物语气编写，避免机械直译和未经证实的韩国尺码映射。",
            "- 葡萄牙文按 pt-BR 编写，避免欧洲葡语表达和未经证实的 P/M/G 映射。",
            "- 三份文案共享同一个不可变商品 ID、URL、叶子类目、属性和完整 SKU 表。",
            f"- 文案生成来源：{json.dumps(localization_sources, ensure_ascii=False)}",
            "",
            "## 5. 图片与视频生成策略",
            "",
            f"- 视觉主题：{state.creative_plan.visual_theme}",
            f"- 创意计划来源：{plan_model}",
            f"- 模型配置：{json.dumps(model_summary, ensure_ascii=False)}",
            "- 主图采用方形浅色棚拍构图，禁止促销文字、边框、水印及未经证实的视觉声明。",
            "- 五张详情图依次覆盖整体展示、设计细节、已验证特征、真实变体和实际使用情境。",
            "- 视频以最终主图或其源 URL 为首帧，使用慢速稳定镜头，禁止服装变形、换色、字幕和复杂手部交互。",
            f"- 本次模型直接生成并通过校验的素材数：{generated_count}。",
            "",
            "## 6. 合规与质检",
            "",
            "生成提示词统一禁止虚假功能、绝对化宣传、额外商标、价格折扣、认证和测量值。"
            "图片下载后统一解码为 RGB JPEG，并校验尺寸和文件大小；模型生成图还会接受商品身份一致性、"
            "意外文字、水印和重大瑕疵检查。视频必须为可识别的 MP4/MOV 容器且小于 200MB。"
            "所有输出在写入最终目录前进行一次完整交付质检，写入后再次复核。",
            "",
            "## 7. 降级与稳定性",
            "",
            "API 请求对限流和暂时性错误执行指数退避；异步图像/视频任务保存 task_id 并轮询。"
            "图片模型失败或语义质检不通过时，回退到经规格归一化的商品源图；视频模型失败时，"
            "使用已验证主图生成可播放的 H.264 商品展示视频。所有回退都优先保证商品事实一致性和文件可用性。",
        ]
        if fallback_assets:
            lines.extend(["", "本次发生的素材回退：", ""])
            lines.extend(
                f"- {asset.name}：{asset.model}；原因：{asset.fallback_reason or '确定性安全回退'}"
                for asset in fallback_assets
            )
        if state.warnings:
            lines.extend(["", "运行质检记录：", ""])
            lines.extend(f"- {warning}" for warning in state.warnings)
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
