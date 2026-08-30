"""Production responsibilities for the delivery pipeline."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from ..agent_tools import ToolExecution
from ..api import ApiError
from ..localization import generate_copy_payload, render_description
from ..media import (
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
from ..models import (
    AssetResult,
    CreativePlan,
    ProductFacts,
    RunState,
    TaxonomyResult,
)
from .common import (
    IMAGE_NEGATIVE_PROMPT as _IMAGE_NEGATIVE_PROMPT,
    MAIN_NEGATIVE_PROMPT as _MAIN_NEGATIVE_PROMPT,
    SINGLE_COMPOSITION_NEGATIVE_PROMPT as _SINGLE_COMPOSITION_NEGATIVE_PROMPT,
    VIDEO_NEGATIVE_PROMPT as _VIDEO_NEGATIVE_PROMPT,
    PipelineError,
    SemanticRejection,
    even_sample as _even_sample,
    reviewed_media_description as _reviewed_media_description,
    unique as _unique,
)


class ProductionPipelineMixin:
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
                    "complete_product": "Complete alternate-angle view showing the full product.",
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
                    "complete_product": "상품 전체를 보여 주는 다른 각도의 전체 이미지입니다.",
                    "primary_verified_detail": "원본에서 확인된 핵심 디테일을 가까이 보여 줍니다.",
                    "secondary_verified_detail": "원본에서 확인된 다른 디테일을 가까이 보여 줍니다.",
                    "verified_variants": "판매자 원본에서 확인된 색상 옵션만 비교한 카탈로그 이미지입니다.",
                    "verified_alternate_view": "판매자 이미지에서 확인된 다른 시점의 상품 이미지입니다.",
                    "verified_use_context": "상품 전체가 보이는 판매자 이미지 기반의 실용적 사용 장면입니다.",
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
                        "video" if asset.model.startswith("ffmpeg-") else "single_video"
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

    def _repair_detail_set_selection(
        self,
        instruction: str,
        *,
        facts: ProductFacts,
        taxonomy: TaxonomyResult,
        creative_plan: CreativePlan,
        state: RunState,
        localization_payloads: dict[str, dict[str, Any]],
        work_dir: Path,
        downloads_dir: Path,
    ) -> ToolExecution:
        detail_assets = {
            index: self._find_asset(state.assets, f"detail_image_{index}.jpeg")
            for index in range(1, 6)
        }
        changed = self._apply_global_detail_candidate_selection(
            facts=facts,
            creative_plan=creative_plan,
            main_asset=self._find_asset(state.assets, "main_image.jpeg"),
            detail_assets=detail_assets,
            work_dir=work_dir,
            downloads_dir=downloads_dir,
            editorial_context=instruction,
        )
        if not changed:
            return ToolExecution(
                "completed",
                "the set editor reconsidered the complete retained pool and kept the current combination",
                {"changed": False},
            )
        state.visual_set_review = self._review_visual_set(facts, state.assets) or {}
        self._write_localized_descriptions(
            facts,
            taxonomy,
            creative_plan,
            localization_payloads,
            state.assets,
            work_dir,
            state.visual_set_review,
        )
        return ToolExecution(
            "completed",
            "the set editor installed and reviewed a different candidate combination",
            {"changed": True},
        )

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
        source_urls = self._ordered_source_urls(
            facts,
            vision,
            preferred_roles=plan.main_reference_roles,
            preferred_indexes=plan.main_reference_indexes,
        )
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
                minimum_improvement=0.0,
                candidate_count=plan.main_candidate_count,
            )
            if asset.source_url and selected == asset.source_url:
                return ToolExecution(
                    "skipped",
                    "hero revision did not score higher than the current asset",
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
            return ToolExecution(
                "failed", f"hero revision rejected; prior hero preserved: {exc}"
            )
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
            index,
            facts,
            main_asset.source_url,
            preferred_roles=(
                plan.detail_reference_roles[index - 1]
                if index <= len(plan.detail_reference_roles)
                else ()
            ),
            preferred_indexes=(
                plan.detail_reference_indexes[index - 1]
                if index <= len(plan.detail_reference_indexes)
                else ()
            ),
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
                minimum_improvement=0.0,
                candidate_count=(
                    plan.detail_candidate_counts[index - 1]
                    if index <= len(plan.detail_candidate_counts)
                    else None
                ),
            )
            if asset.source_url and selected == asset.source_url:
                return ToolExecution(
                    "skipped",
                    f"detail slot {index} revision did not score higher than the current asset",
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
                "Orchestrator-assigned detail role: "
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
        if not self._copy_revision_is_safe(language, facts, incumbent, candidate):
            return ToolExecution(
                "skipped",
                f"{language} revision did not pass factual/safety comparison; prior copy preserved",
            )
        try:
            rendered = render_description(language, candidate, facts, taxonomy)
            staged = work_dir / f".{target}.{uuid.uuid4().hex}.tmp"
            staged.write_text(rendered, encoding="utf-8")
            os.replace(staged, work_dir / target)
        except OSError as exc:
            return ToolExecution(
                "failed", f"localized revision could not be installed: {exc}"
            )
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
            review_sources = _unique(
                [first_frame_url]
                + self._source_urls_for_use(
                    self._fallback_source_urls(facts, asset_name="main_image.jpeg"),
                    use="reference",
                    preferred_roles=("hero", "front"),
                )
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
                "failed",
                f"video revision rejected; prior playable video preserved: {exc}",
            )
        finally:
            raw_video.unlink(missing_ok=True)
            staged.unlink(missing_ok=True)

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
        source_urls = self._ordered_source_urls(
            facts,
            vision,
            preferred_roles=plan.main_reference_roles,
            preferred_indexes=plan.main_reference_indexes,
        )
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
                    candidate_count=(
                        None if self.fast_mode else plan.main_candidate_count
                    ),
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
                        candidate_count=(
                            None if self.fast_mode else plan.main_candidate_count
                        ),
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
        candidate_count: int | None = None,
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
                count=(
                    max(1, min(int(candidate_count), 4))
                    if candidate_count is not None
                    else (2 if self.fast_mode else 3)
                ),
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
            index,
            facts,
            main_reference_url,
            preferred_roles=(
                plan.detail_reference_roles[index - 1]
                if index <= len(plan.detail_reference_roles)
                else ()
            ),
            preferred_indexes=(
                plan.detail_reference_indexes[index - 1]
                if index <= len(plan.detail_reference_indexes)
                else ()
            ),
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
                    record_pool=True,
                    candidate_count=(
                        None
                        if self.fast_mode
                        else (
                            plan.detail_candidate_counts[index - 1]
                            if index <= len(plan.detail_candidate_counts)
                            else None
                        )
                    ),
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
                        record_pool=True,
                        candidate_count=(
                            None
                            if self.fast_mode
                            else (
                                plan.detail_candidate_counts[index - 1]
                                if index <= len(plan.detail_candidate_counts)
                                else None
                            )
                        ),
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
                        f"Orchestrator-assigned detail role: "
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

    def _apply_global_detail_candidate_selection(
        self,
        *,
        facts: ProductFacts,
        creative_plan: CreativePlan,
        main_asset: AssetResult,
        detail_assets: dict[int, AssetResult],
        work_dir: Path,
        downloads_dir: Path,
        editorial_context: str = "",
    ) -> bool:
        """Jointly select the detail-image combination from all slot candidates.

        Per-slot review removes hard defects first. This second pass is deliberately
        set-aware: it may choose a slightly lower local candidate when that candidate
        removes semantic duplication and improves commercial role coverage.
        """

        if self.client is None:
            return False
        with self._detail_candidate_pool_lock:
            raw_pools = copy.deepcopy(self._detail_candidate_pools)
        pools: dict[int, list[dict[str, Any]]] = {}
        for index, raw_pool in raw_pools.items():
            if index not in detail_assets:
                continue
            if isinstance(raw_pool, dict):
                raw_candidates = raw_pool.get("candidates")
                records = (
                    [
                        dict(item)
                        for item in raw_candidates
                        if isinstance(item, dict) and str(item.get("url") or "")
                    ]
                    if isinstance(raw_candidates, list)
                    else []
                )
            elif isinstance(raw_pool, list):
                # Backward-compatible normalization for persisted/test state.
                records = [
                    {"url": str(url), "origin": "legacy", "local_review": {}}
                    for url in raw_pool
                    if str(url)
                ]
            else:
                records = []
            current_url = detail_assets[index].source_url
            if current_url and all(item["url"] != current_url for item in records):
                records.append(
                    {
                        "url": current_url,
                        "origin": "current_artifact",
                        "local_review": {},
                    }
                )
            # Preserve generation order only as identity metadata.  No candidate
            # is dropped or semantically ranked by the host.
            seen_urls: set[str] = set()
            unique_records: list[dict[str, Any]] = []
            for record in records:
                url = str(record.get("url") or "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_records.append(record)
            if unique_records:
                pools[index] = unique_records
        if len(pools) < 2 or sum(len(rows) for rows in pools.values()) <= len(pools):
            self.trace.emit(
                "image.detail_pool_selection_skipped",
                reason="no-cross-slot-alternatives",
                pool_sizes={str(key): len(value) for key, value in pools.items()},
            )
            return False
        if self.deadline - time.monotonic() <= 4 * 60:
            self.trace.emit(
                "image.detail_pool_selection_skipped",
                reason="insufficient-stage-budget",
                remaining_seconds=round(self.deadline - time.monotonic(), 1),
            )
            return False

        source_references = self._source_urls_for_use(
            _unique(
                facts.product_image_urls
                + facts.sku_image_urls
                + facts.description_image_urls
            ),
            use="reference",
            preferred_roles=("hero", "front", "detail", "variant"),
        )[:1]
        if not source_references or not main_asset.source_url:
            self.trace.emit(
                "image.detail_pool_selection_skipped",
                reason="no-trusted-source-reference",
            )
            return False

        candidate_urls: list[str] = []
        candidate_jobs: list[dict[str, Any]] = []
        current_selection: dict[str, int] = {}
        indices_by_slot: dict[str, set[int]] = {}
        for index in sorted(pools):
            slot = f"detail_image_{index}.jpeg"
            indices_by_slot[slot] = set()
            role = (
                creative_plan.detail_roles[index - 1]
                if index <= len(creative_plan.detail_roles)
                else f"detail_slot_{index}"
            )
            for record in pools[index]:
                url = str(record["url"])
                candidate_index = len(candidate_urls)
                candidate_urls.append(url)
                indices_by_slot[slot].add(candidate_index)
                candidate_jobs.append(
                    {
                        "candidate_index": candidate_index,
                        "slot": slot,
                        "canonical_role": role,
                        "current": url == detail_assets[index].source_url,
                        "origin": str(record.get("origin") or "generated"),
                        "local_review": record.get("local_review")
                        if isinstance(record.get("local_review"), dict)
                        else {},
                    }
                )
                if url == detail_assets[index].source_url:
                    current_selection[slot] = candidate_index
            if slot not in current_selection:
                # A missing current candidate is corrupted state, not permission
                # to pretend that another candidate is current.
                self.trace.emit(
                    "image.detail_pool_selection_skipped",
                    reason="current-candidate-missing",
                    slot=slot,
                )
                return False

        try:
            review = self.client.select_best_detail_set(
                json.dumps(facts.compact_dict(), ensure_ascii=False),
                source_references,
                main_asset.source_url,
                candidate_urls,
                candidate_jobs,
                current_selection,
                editorial_context=editorial_context,
            )
        except (ApiError, AttributeError) as exc:
            self.logger.warning("详情图候选池全局选片不可用，保留逐槽结果: %s", exc)
            self.warnings.append(f"详情图候选池全局选片未完成: {exc}")
            self.trace.emit(
                "image.detail_pool_selection_failed",
                error=str(exc),
            )
            return False

        current_score = review.get("current_set_score")
        selected_score = review.get("selected_set_score")
        if (
            review.get("selection_improves_current_set") is not True
            or not isinstance(current_score, (int, float))
            or not isinstance(selected_score, (int, float))
            or float(selected_score) <= float(current_score)
        ):
            self.trace.emit(
                "image.detail_pool_selection_kept_current",
                review=review,
            )
            return False

        rows = review.get("candidates")
        selections = review.get("selections")
        if not isinstance(rows, list) or not isinstance(selections, list):
            self.warnings.append("详情图候选池评审结构不完整，保留逐槽结果")
            return False
        by_index = {
            item.get("candidate_index"): item
            for item in rows
            if isinstance(item, dict) and isinstance(item.get("candidate_index"), int)
        }
        selected_by_slot = {
            str(item.get("slot")): item.get("candidate_index")
            for item in selections
            if isinstance(item, dict) and isinstance(item.get("candidate_index"), int)
        }
        if set(selected_by_slot) != set(indices_by_slot):
            self.warnings.append("详情图候选池未覆盖全部可选槽位，保留逐槽结果")
            return False

        staged: dict[int, tuple[Path, str]] = {}
        try:
            for index in sorted(pools):
                slot = f"detail_image_{index}.jpeg"
                candidate_index = selected_by_slot[slot]
                row = by_index.get(candidate_index)
                if candidate_index not in indices_by_slot[slot] or not isinstance(
                    row, dict
                ):
                    raise PipelineError(
                        f"全局选片返回了槽位外候选: {slot}/{candidate_index}"
                    )
                hard_ok = bool(
                    row.get("usable") is True
                    and row.get("identity_consistent") is True
                    and row.get("construction_consistent") is True
                    and row.get("color_consistent") is True
                    and row.get("pattern_consistent") is True
                    and row.get("slot_match") is True
                    and row.get("single_composition") is True
                    and row.get("unwanted_text") is not True
                    and row.get("unwanted_brand_or_logo") is not True
                    and row.get("prohibited_visual") is not True
                    and row.get("major_artifacts") is not True
                )
                if not hard_ok:
                    raise PipelineError(f"全局选片候选未通过硬门禁: {slot}")
                selected_url = candidate_urls[candidate_index]
                if selected_url == detail_assets[index].source_url:
                    continue
                path = work_dir / f".global-detail-{index}-{uuid.uuid4().hex}.jpeg"
                self._download_and_normalize(
                    selected_url,
                    path,
                    downloads_dir,
                    canvas=(1200, 1500),
                    white_background=False,
                )
                staged[index] = (path, selected_url)
        except (ApiError, MediaError, OSError, PipelineError) as exc:
            for path, _ in staged.values():
                path.unlink(missing_ok=True)
            self.logger.warning("详情图候选池全局组合安装失败，保留逐槽结果: %s", exc)
            self.warnings.append(f"详情图候选池全局组合未安装: {exc}")
            return False

        for index, (path, selected_url) in staged.items():
            asset = detail_assets[index]
            os.replace(path, Path(asset.path))
            asset.source_url = selected_url
            asset.description += "; globally selected for set diversity"
        self.trace.emit(
            "image.detail_pool_selection",
            changed_slots=sorted(staged),
            current_set_score=current_score,
            selected_set_score=selected_score,
            review=review,
        )
        return bool(staged)

    def _generate_detail_with_semantic_retry(
        self,
        index: int,
        facts: ProductFacts,
        prompt: str,
        *,
        references: list[str],
        incumbent_url: str = "",
        minimum_improvement: float = 0.0,
        record_pool: bool = False,
        candidate_count: int | None = None,
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
                count=(
                    max(1, min(int(candidate_count), 4))
                    if candidate_count is not None
                    else (1 if self.fast_mode else 2)
                ),
            )
            if record_pool:
                # Installation happens only after local selection.  Candidate
                # state is committed below together with the current selection
                # and the selector's evidence.
                pass
            if self.fast_mode:
                self.trace.emit(
                    "image.detail_review_skipped",
                    asset=f"detail_image_{index}.jpeg",
                    reason="fast-profile",
                )
                selected = candidate_urls[0]
                if record_pool:
                    self._record_detail_candidate_pool(
                        index,
                        candidate_urls,
                        selected_url=selected,
                        model=model,
                        purpose=active_prompt,
                        review={},
                    )
                return selected, model
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
                if record_pool:
                    self._record_detail_candidate_pool(
                        index,
                        candidate_urls,
                        selected_url=selected,
                        model=model,
                        purpose=active_prompt,
                        review=self._detail_candidate_reviews.get(index, {}),
                    )
                return selected, model
            except SemanticRejection as exc:
                last_rejection = exc
                if (
                    semantic_attempt + 1 >= semantic_attempts
                    or self.deadline - time.monotonic() < 360
                ):
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

    def _record_detail_candidate_pool(
        self,
        index: int,
        candidate_urls: list[str],
        *,
        selected_url: str,
        model: str,
        purpose: str,
        review: dict[str, Any],
    ) -> None:
        """Commit complete, source-addressable candidate state for set editing."""

        review_rows = review.get("candidates") if isinstance(review, dict) else []
        by_index = (
            {
                item.get("index"): dict(item)
                for item in review_rows
                if isinstance(item, dict) and isinstance(item.get("index"), int)
            }
            if isinstance(review_rows, list)
            else {}
        )
        records = [
            {
                "url": url,
                "origin": "generated",
                "generation_index": candidate_index,
                "model": model,
                "local_review": by_index.get(candidate_index, {}),
            }
            for candidate_index, url in enumerate(_unique(candidate_urls))
        ]
        if selected_url and all(item["url"] != selected_url for item in records):
            records.append(
                {
                    "url": selected_url,
                    "origin": "current_artifact",
                    "model": model,
                    "local_review": {},
                }
            )
        with self._detail_candidate_pool_lock:
            self._detail_candidate_pools[index] = {
                "slot": f"detail_image_{index}.jpeg",
                "purpose": purpose,
                "current_url": selected_url,
                "candidates": records,
            }

    def _detail_reference_selection(
        self,
        index: int,
        facts: ProductFacts,
        main_reference_url: str,
        *,
        preferred_roles: list[str] | tuple[str, ...] = (),
        preferred_indexes: list[int] | tuple[int, ...] = (),
    ) -> list[str]:
        product = facts.product_image_urls
        sku = facts.sku_image_urls
        description = facts.description_image_urls
        if preferred_indexes:
            indexed = [
                url
                for url, observation in self._source_image_observations.items()
                if observation.get("index") in preferred_indexes
                and observation.get("safe_for_generation_reference") is True
            ]
            if indexed:
                return indexed[:3]
        visual_variant_values = {
            str(attribute.value or "").strip().casefold()
            for item in facts.skus
            for attribute in item.attributes
            if str(attribute.value or "").strip()
            and re.search(
                r"(?:颜色|色号|花色|款式|图案|\bcolou?r\b|\bstyle\b|\bpattern\b)",
                str(attribute.name or ""),
                re.I,
            )
        }
        if self._source_image_observations:
            safe_variant_references = [
                url
                for url in sku
                if self._source_image_observations.get(url, {}).get(
                    "safe_for_generation_reference"
                )
                is True
            ]
        else:
            safe_variant_references = list(sku)
        if index == 4 and len(visual_variant_values) >= 2 and safe_variant_references:
            return self._source_urls_for_use(
                _unique(
                    _even_sample(safe_variant_references, 3)
                    + [main_reference_url]
                    + product
                ),
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
            preferred_roles=tuple(preferred_roles) or role_preferences.get(index, ()),
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
        asset.fallback_reason = (
            "source size chart transcribed and deterministically rendered"
        )
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
        video_asset.description = "Eight-second multi-shot catalog fallback assembled from perceptually distinct available images"

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
                    asset.description = f"Seller-source {focus_crop} close-up repaired from a duplicate slot"
                    note = (
                        f"自动修复近重复详情图: {asset.name} -> {focus_crop} close-up"
                    )
                    if note not in self.warnings:
                        self.warnings.append(note)
                    repaired = True
                    break
                except (ApiError, MediaError, OSError):
                    staged.unlink(missing_ok=True)
            if not repaired:
                # The normal QA report retains the unresolved duplicate warning.
                continue

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
        schema_id = (
            taxonomy.attribute_schema_category_id or taxonomy.category.category_id
        )
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
        }
        agent_plan_controls = {
            key: value
            for key, value in agent_plan_controls.items()
            if value is not None
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
            "源图理解完成后，两个独立模型按统一的通用证据协议裁决结构化外观属性与可信像素之间的冲突；"
            "代码按属性索引聚合发布、拒绝或仅机器字段决定，不包含具体商品特征的硬编码。裁决后的事实账本是文案、"
            "素材生成和最终评估共同使用的权威外观证据。",
            "三份文案只发布目标语言的买家文案和本地化显示值，不暴露中文原值或原始 JSON Pointer；"
            "本地化 Source 列标明商品事实、平台映射或卖家声明。平台类目 ID、属性 ID/Value ID、"
            "SKU ID/Spec ID 仍保留在精简表格中，兼顾上架解析与阅读体验。",
            "",
            f"本次共读取 {len(facts.attributes)} 条商品属性、{len(facts.skus)} 个 SKU、"
            f"{len(facts.all_image_urls())} 个不重复源图片 URL。",
            f"从源详情图核验并结构化 {len(facts.size_chart_rows)} 行服装尺码数据。",
            f"内部 Claim Ledger 共 {len(state.claim_ledger)} 条；每条记录来源类型、原始字段和证据指针，"
            "买家文案只开放 allowed_surfaces 包含 buyer_copy 的声明。",
            f"Canonical Product State 版本：{state.canonical_product_state.get('version', '未生成')}。",
            f"Evidence Sufficiency 版本：{state.evidence_sufficiency.get('version', '未生成')}；"
            f"可用于生成的明确源图索引为 {state.evidence_sufficiency.get('generation_reference_indexes', [])}。",
            f"Expected Delivery Spec 版本：{state.expected_delivery_spec.get('version', '未生成')}；"
            f"冻结保留 {len(state.expected_delivery_spec.get('required_mapping_sources', []))} 条平台映射来源覆盖。",
            f"当前依赖状态：{json.dumps(state.dependency_state, ensure_ascii=False)}。",
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
            "在线模式由 Taxonomy ReAct agent 使用通用 query/read 工具自行探索类目、schema、属性和值集合；"
            "代码不预排语义候选，只校验最终叶子节点以及模型提交的每个 ID、枚举关系和来源字段。"
            "本地词法排序仅在显式离线或模型协议失败时作为可审计降级。",
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
            "- 买家文案与机器附录独立构建：模型只生成标题、概述、卖点和尺码提示；代码从已核验的类目、属性、"
            "SKU 和媒体契约确定性渲染附录，因此文案修订不能改变任何可解析 ID 或表格行。",
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
            f"- 顶层模型选择主图候选数 {state.creative_plan.main_candidate_count}，参考图角色优先级为 "
            f"{json.dumps(state.creative_plan.main_reference_roles, ensure_ascii=False)}；候选仍须通过身份、结构、颜色和文件硬门禁。",
            "- 主图回退源图须优先满足无人物、无关道具、单一完整商品和干净中性背景；没有合格源图时明确记录质量降级。",
            f"- 五张详情图的商业职责由顶层模型按当前证据选择：{json.dumps(state.creative_plan.detail_roles, ensure_ascii=False)}。"
            "若源详情图存在可核验尺码表，确定性重绘是事实/可读性边界，不依赖品类关键词推断。",
            f"- 各详情图候选数由顶层模型选择：{json.dumps(state.creative_plan.detail_candidate_counts, ensure_ascii=False)}。"
            "候选先执行逐槽身份与结构硬门禁，再把主图和全部详情候选"
            "交给集合级编辑器联合选片；只有六图组合至少提升 3 分且所有替换图无硬伤时才原子安装。",
            "- 视频以最终主图或其源 URL 为首帧，镜头语义由顶层模型规划，代码仅维护身份稳定、禁用不受支持内容并默认移除未审核音轨。",
            f"- 本次模型直接生成并通过校验的素材数：{generated_count}。",
            "",
            "## 6. 有界 Agent 规划、评估与定向修复",
            "",
            "同一个顶层工具调用 Agent 对话贯穿规划和成品控制。它可按需查看商品事实、类目、源图证据、产物状态与工具能力，"
            "并自主选择分镜、参考角色、候选数量、生产启动次序、评审时机、返修目标和完成时机。独立多模态评估器只向顶层 Agent"
            "返回有证据的缺陷；宿主不再通过固定 repair planner 规定修复路线。",
            "所有修复均先写入临时文件，完成候选语义选优、文案事实/schema 校验或视频播放校验后才原子替换；"
            "修复失败会保留上一版，不会因为评估意见自动降级为源图或幻灯片。",
            "控制器对完整交付生成内容指纹，同一指纹的独立评审只执行一次。工具报告 completed 后仍须确认目标 hash 变化；"
            "每个目标独立保存检查点，并在依赖同步和本地一致性通过后提交。变化后的状态必须再次 review；finish 工具会拒绝"
            "过期评审、文件契约失败，以及未解决的 A1/A2/A5 重大问题。除此之外，是否继续优化由顶层模型结合剩余预算判断。",
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
                f"- 评估轮次 {item.round_index}：模型 {json.dumps(item.evaluator_models, ensure_ascii=False)}；"
                f"单模型证据惩罚分 {json.dumps(item.model_weighted_scores, ensure_ascii=False)}；"
                f"裁决后加权分 {item.weighted_score:.1f}；裁决后维度分 {json.dumps(item.dimension_scores, ensure_ascii=False)}；"
                f"ready={item.ready_for_delivery}；{brief(item.summary)}"
                for item in state.agent_evaluations
            )
        if state.agent_snapshots:
            lines.extend(["", "版本快照与最终选择：", ""])
            lines.extend(
                f"- {item.get('snapshot_id')}：完成 {item.get('after_repair_rounds')} 轮修复；"
                f"加权分 {float(item.get('weighted_score', 0.0)):.1f}；"
                f"{'最终提交' if item.get('selected') else '保留备选'}。"
                for item in state.agent_snapshots
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
