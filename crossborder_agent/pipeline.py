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
from .compliance import normalize_source_image_observations
from .input_loader import discover_input_files, load_json, load_product_facts
from .localization import generate_copy_payload, render_description
from .media import (
    MediaError,
    create_catalog_video,
    create_slideshow_video,
    hash_distance,
    inspect_image_quality,
    inspect_video,
    normalize_image,
    strip_video_audio,
)
from .models import AssetResult, CreativePlan, ProductFacts, RunState, TaxonomyResult
from .planning import create_creative_plan
from .qa import EXPECTED_FILES, validate_delivery
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
        self._source_image_observations: dict[str, dict[str, Any]] = {}
        self._source_selection_warnings: set[str] = set()

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

            self._review_generated_assets(facts, state.assets, work_dir, downloads_dir)
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
                "image": "Source-derived product view normalized to the required listing format to preserve visual facts.",
                "video": "Eight-second multi-shot catalog video assembled from the final validated product images.",
            },
            "ko": {
                "image": "상품의 시각적 사실을 유지하기 위해 원본 이미지를 필수 등록 규격에 맞춰 정리한 이미지입니다.",
                "video": "최종 검수된 상품 이미지로 구성한 8초 멀티컷 카탈로그 영상입니다.",
            },
            "pt": {
                "image": "Imagem derivada da fonte e ajustada ao formato exigido para preservar os dados visuais do produto.",
                "video": "Vídeo de catálogo de 8 segundos, em vários planos, montado com as imagens finais validadas do produto.",
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
                kind = "video" if name == "product_video.mp4" else "image"
                media[name] = fallback_templates[language][kind]
            description = render_description(language, payload, facts, taxonomy)
            (work_dir / f"product_description_{language}.md").write_text(
                description, encoding="utf-8"
            )

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
        sku_sample = _even_sample(facts.sku_image_urls, 4)
        description_sample = _even_sample(facts.description_image_urls, 3)
        urls = _unique(preferred + sku_sample + description_sample)[:12]
        try:
            result = self.client.analyze_product_images(
                json.dumps(facts.compact_dict(), ensure_ascii=False), urls
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
                    f"{third_party_count} 张源图疑似含第三方品牌或角色；发布前须核验授权，且不用于衍生生成"
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
        if asset_name.startswith("detail_image_") and primary:
            try:
                index = int(asset_name.removeprefix("detail_image_").split(".", 1)[0])
            except ValueError:
                index = 1
            offset = (index - 1) % len(primary)
            primary = primary[offset:] + primary[:offset]
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
        avoid_hashes: list[int] | None = None,
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
                )
                if avoid_hashes:
                    quality = inspect_image_quality(candidate_destination)
                    if quality is not None and any(
                        hash_distance(quality.difference_hash, seen_hash) <= 2
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
                fallback_reason="image generation unavailable or rejected",
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
        if len(candidate_urls) == 1 or self.client is None:
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
        source_urls = _unique(
            [main_reference_url]
            + facts.product_image_urls
            + facts.sku_image_urls
            + facts.description_image_urls
        )
        source_urls = self._fallback_source_urls(
            facts, asset_name=f"detail_image_{index}.jpeg"
        )
        reference_selection = self._detail_reference_selection(
            index, facts, main_reference_url
        )
        if self.client is not None and reference_selection:
            try:
                candidate_count = 2 if index in {4, 5} else 1
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
        if len(candidate_urls) == 1 or self.client is None:
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
                    and item.get("anatomy_natural") is not False
                    and item.get("unwanted_text") is not True
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
        self._replace_near_duplicate_details(facts, assets, downloads_dir)
        image_assets = [
            asset
            for asset in assets
            if asset.name.endswith(".jpeg") and asset.generated
        ]
        main_was_rejected = False
        if image_assets and self.deadline - time.monotonic() > 3 * 60:
            source_review_urls = _unique(
                facts.product_image_urls[:3]
                + _even_sample(facts.sku_image_urls, 1)
                + _even_sample(facts.description_image_urls, 1)
            )[:5]
            source_review_urls = self._source_urls_for_use(
                source_review_urls, use="reference"
            )[:5]
            try:
                review = self.client.review_generated_images(
                    json.dumps(facts.compact_dict(), ensure_ascii=False),
                    source_review_urls,
                    [asset.source_url for asset in image_assets],
                    [
                        {"name": asset.name, "purpose": asset.description}
                        for asset in image_assets
                    ],
                )
            except ApiError as exc:
                self.logger.warning(
                    "生成图片语义质检失败，保留已通过物理校验的图片: %s", exc
                )
                self.warnings.append(f"生成图片语义质检不可用: {exc}")
            else:
                main_was_rejected = self._apply_image_review(
                    facts, image_assets, review, downloads_dir
                )
        elif image_assets:
            self.warnings.append("剩余时间不足，跳过生成图片语义质检")

        if main_was_rejected:
            video_asset = next(
                (asset for asset in assets if asset.name == "product_video.mp4"), None
            )
            if video_asset and video_asset.generated:
                replaced = self._fallback_video(
                    video_asset,
                    work_dir / "main_image.jpeg",
                    "main image semantic QA rejection invalidated generated video",
                )
                if replaced:
                    self.warnings.append("主图语义质检回退后，视频同步回退为源图展示")

        self._replace_near_duplicate_details(
            facts, assets, downloads_dir, include_fallback=True
        )
        self._review_generated_video(facts, assets, work_dir)
        self._enhance_fallback_video(assets, work_dir)

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
        video_asset.fallback_reason = (
            (video_asset.fallback_reason + "; ") if video_asset.fallback_reason else ""
        ) + "rebuilt from the final validated image set"
        video_asset.description = (
            "Eight-second multi-shot catalog video assembled from the final validated images"
        )

    def _apply_image_review(
        self,
        facts: ProductFacts,
        image_assets: list[AssetResult],
        review: dict[str, Any],
        downloads_dir: Path,
    ) -> bool:
        reviews = review.get("assets")
        if not isinstance(reviews, list):
            self.warnings.append("生成图片语义质检返回结构无效，保留物理校验结果")
            return False
        main_was_rejected = False
        for item in reviews:
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                continue
            index = item["index"]
            if not 0 <= index < len(image_assets):
                continue
            asset = image_assets[index]
            rejected = (
                item.get("usable") is not True
                or item.get("identity_consistent") is not True
                or item.get("construction_consistent") is not True
                or item.get("color_consistent") is not True
                or item.get("pattern_consistent") is not True
                or item.get("slot_match") is not True
                or item.get("unwanted_text") is not False
                or item.get("prohibited_visual") is not False
                or item.get("major_artifacts") is not False
                or (
                    item.get("unexpected_collage") is True
                    and asset.name != "detail_image_4.jpeg"
                )
                or str(item.get("product_coverage") or "").lower()
                not in {"high", "medium"}
            )
            if not rejected:
                continue
            destination = Path(asset.path)
            is_main = asset.name == "main_image.jpeg"
            try:
                fallback_url = self._fallback_image(
                    self._source_urls_for_use(
                        self._fallback_source_urls(facts, asset_name=asset.name),
                        use="fallback",
                    ),
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
        return main_was_rejected

    def _replace_near_duplicate_details(
        self,
        facts: ProductFacts,
        assets: list[AssetResult],
        downloads_dir: Path,
        *,
        include_fallback: bool = False,
    ) -> None:
        seen: list[tuple[str, int]] = []
        for asset in assets:
            if not asset.name.startswith("detail_image_") or (
                not include_fallback and not asset.generated
            ):
                continue
            try:
                quality = inspect_image_quality(Path(asset.path))
            except MediaError as exc:
                self.logger.warning("详情图质量信号读取失败 %s: %s", asset.name, exc)
                continue
            if quality is None:
                return
            duplicate_of = next(
                (
                    name
                    for name, difference_hash in seen
                    if hash_distance(quality.difference_hash, difference_hash) <= 2
                ),
                "",
            )
            if not duplicate_of:
                seen.append((asset.name, quality.difference_hash))
                continue
            try:
                fallback_url = self._fallback_image(
                    self._source_urls_for_use(
                        self._fallback_source_urls(facts, asset_name=asset.name),
                        use="fallback",
                    ),
                    Path(asset.path),
                    downloads_dir,
                    canvas=(1200, 1500),
                    white_background=False,
                    avoid_hashes=[difference_hash for _, difference_hash in seen],
                )
            except PipelineError as exc:
                self.logger.warning("重复详情图回退失败 %s: %s", asset.name, exc)
                warning = f"{asset.name} 与 {duplicate_of} 近重复，未找到不同的安全源图"
                if warning not in self.warnings:
                    self.warnings.append(warning)
                continue
            asset.source_url = fallback_url
            asset.model = "deterministic-source-fallback"
            asset.generated = False
            asset.fallback_reason = f"near-duplicate of {duplicate_of}"
            self.warnings.append(f"{asset.name} 与 {duplicate_of} 近重复，已回退到源图")
            try:
                replacement_quality = inspect_image_quality(Path(asset.path))
            except MediaError:
                replacement_quality = None
            if replacement_quality is not None:
                seen.append((asset.name, replacement_quality.difference_hash))

    def _fallback_sources_for_asset(
        self, facts: ProductFacts, asset_name: str
    ) -> list[str]:
        source_urls = _unique(
            facts.product_image_urls
            + facts.sku_image_urls
            + facts.description_image_urls
        )
        if not source_urls or not asset_name.startswith("detail_image_"):
            return source_urls
        try:
            index = int(asset_name.removeprefix("detail_image_").split(".", 1)[0])
        except ValueError:
            return source_urls
        offset = (index - 1) % len(source_urls)
        return source_urls[offset:] + source_urls[:offset]

    def _review_generated_video(
        self, facts: ProductFacts, assets: list[AssetResult], work_dir: Path
    ) -> None:
        video_asset = next(
            (asset for asset in assets if asset.name == "product_video.mp4"), None
        )
        if not video_asset or not video_asset.generated or not video_asset.source_url:
            return
        if self.deadline - time.monotonic() <= 2 * 60:
            replaced = self._fallback_video(
                video_asset,
                work_dir / "main_image.jpeg",
                "insufficient time for generated-video semantic QA",
            )
            if replaced:
                self.warnings.append("剩余时间不足，生成视频已安全回退")
            return
        source_review_urls = _unique(
            facts.product_image_urls[:3] + _even_sample(facts.sku_image_urls, 1)
        )[:4]
        source_review_urls = self._source_urls_for_use(
            source_review_urls, use="reference"
        )[:4]
        try:
            review = self.client.review_generated_video(
                json.dumps(facts.compact_dict(), ensure_ascii=False),
                source_review_urls,
                video_asset.source_url,
            )
        except ApiError as exc:
            self.logger.warning("生成视频语义质检失败，执行安全回退: %s", exc)
            replaced = self._fallback_video(
                video_asset,
                work_dir / "main_image.jpeg",
                f"generated-video semantic QA unavailable: {exc}",
            )
            if replaced:
                self.warnings.append("生成视频语义质检不可用，已安全回退")
            return
        rejected = (
            review.get("usable") is not True
            or review.get("identity_consistent") is not True
            or review.get("construction_consistent") is not True
            or review.get("color_and_pattern_consistent") is not True
            or review.get("motion_stable") is not True
            or review.get("unwanted_text") is not False
            or review.get("prohibited_visual") is not False
            or review.get("major_artifacts") is not False
        )
        if not rejected:
            return
        replaced = self._fallback_video(
            video_asset,
            work_dir / "main_image.jpeg",
            f"semantic QA rejected generated video: {review.get('reason', '')}",
        )
        if replaced:
            self.warnings.append("product_video.mp4 因语义质检回退到稳定主图视频")

    def _fallback_video(
        self, video_asset: AssetResult, main_image_path: Path, reason: str
    ) -> bool:
        video_path = Path(video_asset.path)
        try:
            create_slideshow_video(main_image_path, video_path, duration=8)
        except MediaError as exc:
            self.logger.warning("无法重建视频回退: %s", exc)
            return False
        video_asset.source_url = ""
        video_asset.model = "ffmpeg-slideshow-fallback"
        video_asset.generated = False
        video_asset.fallback_reason = reason
        return True

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
            "三份文案均并列保留面向买家的本地化显示值与源数据原值；平台类目 ID、属性 ID/Value ID、"
            "SKU ID/Spec ID 及 JSON Pointer 证据位置可直接机器解析，翻译不会覆盖标准答案字段。",
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
            "当童装 T 恤叶子在平台快照中缺失属性元数据时，只复用同快照中通用 T 恤的稳定属性 ID/枚举 schema；"
            "男童/女童叶子类目 ID 保持不变，颜色与身高销售规格仍逐项来自源 SKU。",
            "",
            "## 4. 本地化策略",
            "",
            "- 英文按 en-US 电商语气编写，涉及体重时同时给出 lb。",
            "- 英文中的厘米规格同时给出确定性英寸换算；韩文与巴西葡萄牙文保留当地常用公制。",
            "- 韩文按 ko-KR 自然购物语气编写，避免机械直译和未经证实的韩国尺码映射。",
            "- 葡萄牙文按 pt-BR 编写，避免欧洲葡语表达和未经证实的 P/M/G 映射。",
            "- 三份文案共享同一个不可变商品 ID、URL、叶子类目、属性和完整 SKU 表。",
            "- 卖家提供的体重范围只按完全匹配的尺码标签写入对应 SKU，并同时展示 kg/lb，不推导地区尺码。",
            f"- 文案生成来源：{json.dumps(localization_sources, ensure_ascii=False)}",
            "",
            "## 5. 图片与视频生成策略",
            "",
            f"- 视觉主题：{state.creative_plan.visual_theme}",
            f"- 创意计划来源：{plan_model}",
            f"- 模型配置：{json.dumps(model_summary, ensure_ascii=False)}",
            "- 主图采用方形浅色棚拍构图；优先生成三个候选，再按身份、结构、颜色、完整度、干净背景、单品覆盖和瑕疵自动选优。",
            "- 主图回退源图须优先满足无人物、无关道具、单一完整商品和干净中性背景；没有合格源图时明确记录质量降级。",
            "- 五张详情图依次覆盖美区整体展示、韩区精细商品摄影、巴西区自然光细节、真实变体和跨市场使用情境；"
            "不使用国旗、地标、文化刻板印象或生成文字。",
            "- 高风险的变体图与穿着场景各生成两个候选，并按商品身份、颜色、图案、结构、人体与分镜匹配自动选优。",
            "- 视频以最终主图或其源 URL 为首帧，按上装、下装、连衣裙或童装使用不同结构保护镜头；默认移除未审核音轨。",
            f"- 本次模型直接生成并通过校验的素材数：{generated_count}。",
            "",
            "## 6. 合规与质检",
            "",
            "生成提示词统一禁止虚假功能、绝对化宣传、额外商标、价格折扣、认证和测量值。"
            "图片下载后统一解码为 RGB JPEG，并校验尺寸、文件大小、空白图和近重复图；模型生成图还会与"
            "可信源图共同输入视觉质检，检查商品身份与具体结构、分镜覆盖、意外文字、水印和重大瑕疵。视频除容器和"
            "200MB 上限外，还须完成全视频流解码，并通过源图对照的时序语义质检。"
            "所有输出在写入最终目录前进行一次完整交付质检，写入后再次复核。",
            "源图检查区分商品本身的印花文字与背景营销叠字；后者以及价格、联系方式、二维码、水印、平台标识和"
            "敏感视觉元素不会进入发布回退素材。视频语义质检缺失、超时或字段不完整时按失败处理。",
            "",
            "## 7. 降级与稳定性",
            "",
            "API 请求对限流和暂时性错误执行指数退避；图片优先走同步多模态生成，视频异步任务保存 task_id 并轮询。"
            "图片模型失败或语义质检不通过时，回退到经规格归一化的商品源图；视频模型失败时，"
            "使用最终质检后的主图与详情图生成多镜头 H.264 商品展示视频。所有回退都优先保证商品事实一致性和文件可用性。",
            f"- 本次 API 调用记录数：{len(state.api_calls)}；每次调用均记录模型、耗时、状态及调用后的剩余时间。",
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
