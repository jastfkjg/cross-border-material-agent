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
from ..localization import _localized_display, generate_copy_payload, render_description
from ..media import (
    MediaError,
    create_catalog_video,
    create_evidence_table_image,
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
from ..table_evidence import presentation_view, select_render_table
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

_MAIN_IMAGE_MAX_BYTES = 1024 * 1024 - 1


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
                "evidence_table": "Source table rendered from exact, model-selected cells.",
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
                "evidence_table": "판매자 원본에서 확인된 표를 모델이 선택한 셀 그대로 정리한 이미지입니다.",
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
                "evidence_table": "Tabela da fonte reproduzida com as células exatas selecionadas pelo modelo.",
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
                    or asset.model == "deterministic-evidence-table"
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
                if asset.model == "deterministic-evidence-table":
                    kind = "evidence_table"
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
            raise PipelineError(f"Repair target does not exist: {name}")
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
                max_bytes=_MAIN_IMAGE_MAX_BYTES,
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
        selected_table = select_render_table(facts.evidence_tables)
        protected_index = (
            int(selected_table.presentation.get("target_detail_index") or 0)
            if selected_table is not None
            else 0
        )
        if index == protected_index:
            return ToolExecution(
                "skipped",
                "the model-selected source table is protected from generative replacement",
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
        max_bytes: int = 5 * 1024 * 1024,
    ) -> None:
        raw_path = self._next_raw_path(downloads_dir, ".asset")
        self.downloader.download(url, raw_path, max_bytes=30 * 1024 * 1024, timeout=180)
        normalize_image(
            raw_path,
            destination,
            canvas=canvas,
            max_bytes=max_bytes,
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
        max_bytes: int = 5 * 1024 * 1024,
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
                    max_bytes=max_bytes,
                )
                if avoid_hashes:
                    quality = inspect_image_quality(candidate_destination)
                    if quality is not None and any(
                        hash_distance(quality.difference_hash, seen_hash) <= 10
                        for seen_hash in avoid_hashes
                    ):
                        errors.append(
                            f"Candidate source image is a near-duplicate of an accepted detail image: {url}"
                        )
                        candidate_destination.unlink(missing_ok=True)
                        continue
                    os.replace(candidate_destination, destination)
                return url
            except (ApiError, MediaError) as exc:
                errors.append(str(exc))
                candidate_destination.unlink(missing_ok=True)
        raise PipelineError("All source-image fallbacks failed: " + "; ".join(errors[-3:]))

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
                        max_bytes=_MAIN_IMAGE_MAX_BYTES,
                    )
                except (ApiError, MediaError) as download_error:
                    if self.deadline - time.monotonic() < 360:
                        raise
                    self.logger.warning(
                        "Main-image candidate download or physical validation failed; regenerating once: %s",
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
                        max_bytes=_MAIN_IMAGE_MAX_BYTES,
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
                self.logger.warning(
                    "Main-image generation failed; using a source-image fallback: %s",
                    exc,
                )
                self.warnings.append(f"Main-image generation fallback: {exc}")
        fallback_url = self._fallback_image(
            self._fallback_source_urls(facts, asset_name="main_image.jpeg"),
            destination,
            downloads_dir,
            canvas=(1600, 1600),
            white_background=True,
            max_bytes=_MAIN_IMAGE_MAX_BYTES,
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
                    "Main image has a confirmed semantic defect; regenerating once with review feedback: %s",
                    exc.feedback[:500],
                )
                active_prompt = (
                    prompt
                    + "\nMandatory correction after semantic rejection: "
                    + exc.feedback[:1200]
                    + "\nPreserve exact product identity and correct only these hard defects."
                )
        raise last_rejection or SemanticRejection("Main-image semantic correction failed")

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
        selected_table = select_render_table(facts.evidence_tables)
        target_index = (
            int(selected_table.presentation.get("target_detail_index") or 0)
            if selected_table is not None
            else 0
        )
        if selected_table is not None and index == target_index:
            create_evidence_table_image(selected_table, destination)
            view = presentation_view(selected_table)
            return AssetResult(
                name=destination.name,
                path=str(destination),
                source_url=selected_table.source_url,
                model="deterministic-evidence-table",
                generated=False,
                fallback_reason=(
                    "model-selected source table rendered from exact grounded cells"
                ),
                description=str(view["title"]),
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
                        "Detail-image %d candidate download or physical validation failed; regenerating once: %s",
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
                self.logger.warning(
                    "Detail-image %d generation failed; using a source-image fallback: %s",
                    index,
                    exc,
                )
                self.warnings.append(f"Detail-image {index} generation fallback: {exc}")

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
            self.logger.warning(
                "Global detail-image candidate selection is unavailable; keeping per-slot results: %s",
                exc,
            )
            self.warnings.append(
                f"Global detail-image candidate selection was not completed: {exc}"
            )
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
            self.warnings.append(
                "Detail-image candidate-pool review is incomplete; keeping per-slot results"
            )
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
            self.warnings.append(
                "Detail-image candidate-pool review did not cover every slot; keeping per-slot results"
            )
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
                        f"Global selection returned a candidate outside its slot: {slot}/{candidate_index}"
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
                    raise PipelineError(
                        f"Globally selected candidate failed a hard gate: {slot}"
                    )
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
            self.logger.warning(
                "Global detail-image set installation failed; keeping per-slot results: %s",
                exc,
            )
            self.warnings.append(
                f"Global detail-image set was not installed: {exc}"
            )
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
                    "Detail image %d has a confirmed semantic defect; regenerating once with review feedback: %s",
                    index,
                    exc.feedback[:500],
                )
                active_prompt = (
                    prompt
                    + "\nMandatory correction after semantic rejection: "
                    + exc.feedback[:1200]
                    + "\nPreserve exact product identity and storyboard purpose."
                )
        raise last_rejection or SemanticRejection(
            f"Detail-image {index} semantic correction failed"
        )

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
        excluded_roles = {"data_table", "size_chart", "packaging", "unknown"}
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
                self.logger.warning(
                    "Video model failed; creating a deterministic video fallback: %s",
                    exc,
                )
                self.warnings.append(f"Video generation fallback: {exc}")
        elif self.client is not None:
            if self.fast_mode:
                generation_failure = "fast profile skips video-model generation"
                self.trace.emit("video.generation_skipped", reason="fast-profile")
            else:
                self.warnings.append(
                    "The first-frame source image triggered IP or visual-risk signals; derivative video generation was skipped"
                )
        create_slideshow_video(main_image_path, destination, duration=8)
        return AssetResult(
            name=destination.name,
            path=str(destination),
            model="ffmpeg-slideshow-fallback",
            generated=False,
            fallback_reason=generation_failure,
            description="Playable H.264 product presentation fallback",
        )

    def _install_evidence_table_detail(
        self, facts: ProductFacts, assets: list[AssetResult], work_dir: Path
    ) -> None:
        table = select_render_table(facts.evidence_tables)
        if table is None:
            return
        target_index = int(table.presentation.get("target_detail_index") or 0)
        asset = next(
            (
                item
                for item in assets
                if item.name == f"detail_image_{target_index}.jpeg"
            ),
            None,
        )
        if asset is None:
            return
        destination = work_dir / f"detail_image_{target_index}.jpeg"
        try:
            create_evidence_table_image(table, destination)
            view = presentation_view(table)
        except (MediaError, ValueError) as exc:
            self.logger.warning(
                "Source-table rendering failed; keeping the existing detail image: %s",
                exc,
            )
            self.warnings.append(f"Source-table detail-image generation failed: {exc}")
            return
        asset.path = str(destination)
        asset.source_url = table.source_url or asset.source_url
        asset.model = "deterministic-evidence-table"
        asset.generated = False
        asset.fallback_reason = (
            "model-selected source table rendered from exact grounded cells"
        )
        asset.description = str(view["title"])

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
            self.logger.warning(
                "Multi-shot video fallback is unavailable; keeping the stable single-image video: %s",
                exc,
            )
            self.warnings.append(f"Multi-shot video fallback is unavailable: {exc}")
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
                or asset.model == "deterministic-evidence-table"
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
                        f"Automatically repaired a near-duplicate detail image: "
                        f"{asset.name} -> {focus_crop} close-up"
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
        english_payload = localization_payloads.get("en", {})
        raw_english_terms = (
            english_payload.get("localized_terms")
            if isinstance(english_payload, dict)
            else None
        )
        english_terms = raw_english_terms if isinstance(raw_english_terms, dict) else {}

        def english_display(value: Any) -> str:
            raw = str(value or "")
            rendered = _localized_display("en", raw, english_terms)
            if re.search(r"[\u4e00-\u9fff]", rendered):
                rendered = _localized_display("en", raw, {})
            return (
                rendered
                if not re.search(r"[\u4e00-\u9fff]", rendered)
                else "Seller-declared source value"
            )

        schema_id = (
            taxonomy.attribute_schema_category_id or taxonomy.category.category_id
        )
        schema_note = (
            f"The leaf category has no independent attribute metadata, so attribute mapping uses "
            f"parent/general schema {schema_id} from the same marketplace snapshot; the listing leaf "
            f"category remains {taxonomy.category.category_id}."
            if schema_id != taxonomy.category.category_id
            else f"Attribute mapping uses leaf-category schema {schema_id}."
        )

        role_labels = {
            "complete_product": "Complementary Full-Product View",
            "primary_verified_detail": "Primary Verified Detail",
            "secondary_verified_detail": "Secondary Verified Detail",
            "verified_variants": "Verified Options",
            "verified_alternate_view": "Verified Alternate View",
            "verified_use_context": "Source-Supported Use Context",
            "product_only_context": "Product-Only Context",
            "texture_closeup": "Material and Surface Close-Up",
            "collar_and_closure": "Collar and Closure Construction",
            "color_variant_grid": "Verified Color Options",
            "hem_and_trim": "Hem and Trim Detail",
            "back_silhouette": "Back Silhouette",
            "neckline_closeup": "Neckline Construction Close-Up",
            "pattern_texture": "Pattern and Surface Close-Up",
            "sleeve_cuff": "Sleeve and Cuff Construction",
            "back_view": "Back View",
            "hem_shape": "Hem Silhouette",
            "data_table": "Source Data Table",
        }

        def role_purpose(role: str) -> str:
            normalized = str(role or "").strip().casefold()
            if "table" in normalized:
                return "Turns the model-selected source table into a clear buying reference using the exact source cells"
            if any(token in normalized for token in ("variant", "color")):
                return "Presents source-supported options together for quick comparison"
            if any(token in normalized for token in ("texture", "surface", "material")):
                return "Shows visible surface characteristics that the hero image cannot convey"
            if any(
                token in normalized
                for token in ("collar", "closure", "construction", "detail", "trim", "hem")
            ):
                return "Highlights source-visible construction without inventing unseen parts"
            if any(token in normalized for token in ("back", "side", "alternate")):
                return "Adds a product angle not covered by the hero image to clarify the silhouette"
            if any(token in normalized for token in ("use", "context", "lifestyle")):
                return "Shows practical presentation only within the limits of source evidence"
            return "Adds product information without duplicating the other images"

        reconciled = (
            facts.reconciled_fact_ledger
            if isinstance(facts.reconciled_fact_ledger, dict)
            else {}
        )
        decision_rows = [
            item
            for item in reconciled.get("attribute_decisions", [])
            if isinstance(item, dict)
        ]
        market_labels = {
            "en": "United States English (en-US)",
            "ko": "South Korean market (ko-KR)",
            "pt": "Brazilian Portuguese (pt-BR)",
        }
        default_market_angles = {
            "en": "Use natural, direct commerce language; preserve source metric values and add imperial conversions only when the data supports them.",
            "ko": "Use concise, natural Korean shopping language and metric units without inferring unsupported Korean size equivalents.",
            "pt": "Use Brazilian commerce vocabulary and metric units while avoiding European Portuguese phrasing and unsupported regional size mappings.",
        }
        market_lines = [
            f"- **{market_labels[language]}:** {default_market_angles[language]}"
            for language in ("en", "ko", "pt")
        ]

        detail_lines: list[str] = []
        for index in range(1, 6):
            role = (
                state.creative_plan.detail_roles[index - 1]
                if index <= len(state.creative_plan.detail_roles)
                else f"detail_{index}"
            )
            role_label = role_labels.get(str(role), str(role).replace("_", " "))
            detail_lines.append(
                f"- **detail_image_{index}.jpeg | {role_label}:** {role_purpose(str(role))}."
            )

        if decision_rows:
            published: list[str] = []
            withheld: list[str] = []
            for row in decision_rows:
                index = row.get("attribute_index")
                if not isinstance(index, int) or not 0 <= index < len(facts.attributes):
                    continue
                source = facts.attributes[index]
                decision = str(row.get("decision") or "publish")
                if decision == "publish" and len(published) < 5:
                    published.append(
                        f"{english_display(source.name)}={english_display(source.value)}"
                    )
                elif decision != "publish" and len(withheld) < 3:
                    withheld.append(
                        f"{english_display(source.name)}={english_display(source.value)}"
                    )
            conflict_summaries = []
            for conflict in reconciled.get("conflicts", []):
                if not isinstance(conflict, dict) or len(conflict_summaries) >= 3:
                    continue
                structured = english_display(
                    " ".join(str(conflict.get("structured_value") or "").split())
                )
                visual = english_display(
                    " ".join(str(conflict.get("visual_value") or "").split())
                )
                if structured and visual:
                    conflict_summaries.append(
                        f'The seller states "{structured}", while source-image review verifies "{visual}"; public appearance claims use the image-verified result'
                    )
            fact_decision_lines = [
                "Primary facts available for shopper-facing copy: "
                + (", ".join(published) if published else "features jointly supported by source data and source images")
                + ".",
                "Claims withheld from shopper-facing copy: "
                + ("; ".join(withheld) if withheld else "no conflict requires special disclosure")
                + ". These fields may remain in evidence-backed marketplace mappings but never drive unsupported visual generation.",
            ]
            if conflict_summaries:
                fact_decision_lines.append(
                    "Key conflict handling for this run: "
                    + "; ".join(conflict_summaries)
                    + "."
                )
        else:
            fact_decision_lines = [
                "This run produced no separate visual-conflict adjudication details. Public facts still come only from source data, directly verifiable source-image evidence, and deterministic conversions; unverified fields are excluded from expanded marketing claims.",
            ]

        category_method = str(taxonomy.category.method or "").casefold()
        if "react" in category_method or "model" in category_method:
            category_selection_note = (
                "The model independently searched candidates and relationships in the marketplace snapshot before submitting the leaf category. Host code verified that the node exists, is a leaf, and that every mapping ID can be resolved."
            )
        else:
            category_selection_note = (
                "Model-based category exploration was unavailable, so a general local evidence ranking preserved delivery availability. Category nodes and mapping IDs were still validated before submission, but semantic confidence is lower than for completed model exploration."
            )

        model_locales = [
            language
            for language, source in localization_sources.items()
            if source and "fallback" not in source and "guard" not in source
        ]
        localization_delivery_note = (
            "All three language versions were drafted by target-language writers and independently fact-audited; code then generated the machine appendix deterministically from marketplace mappings."
            if len(model_locales) == 3
            else "At least one language used an availability fallback. Fallback copy still derives from adjudicated facts and retains the complete machine appendix; it never substitutes fixed product answers for unknown semantics."
        )

        video_asset = next(
            (item for item in state.assets if item.name == "product_video.mp4"), None
        )
        if video_asset is not None and video_asset.generated:
            video_delivery = (
                "Uses a verified first frame to create a short dynamic presentation with restrained camera motion across the product silhouette and key details."
            )
        else:
            video_delivery = (
                "Uses the final quality-checked images in a short catalog presentation so the video remains playable and synchronized with the delivered image set."
            )

        selected_table = select_render_table(facts.evidence_tables)
        if selected_table is not None:
            table_strategy = (
                "A table found in a source image passed reference, row/column, and presentation-purpose validation and appears in the selected detail asset and copy appendix. Code reproduces only referenced source cells and does not invent fields."
            )
        elif facts.evidence_tables:
            table_strategy = (
                "Structured tables were detected in source images, but the model did not judge them suitable as buying information for this delivery. The copy therefore makes no unsupported size-chart or measurement promises."
            )
        else:
            table_strategy = "No source-image table could be transcribed safely, so the copy uses only structured product and SKU facts."

        lines = [
            "# Product Localization Asset Strategy",
            "",
            "## 1. Product Positioning and Delivery Objective",
            "",
            f"- Product ID: {facts.offer_id}",
            f"- Source platform: {english_display(facts.platform)}",
            f"- Source product URL: {facts.source_url}",
            f"- AliExpress leaf category: {english_display(taxonomy.category.name)} ({taxonomy.category.category_id})",
            "- Delivery objective: produce listing-ready copy in three languages, one hero image, five complementary detail images, and one product video.",
            "",
            "The Agent follows five stages: (1) consolidate product JSON, SKUs, and source images; "
            "(2) let the model explore the marketplace taxonomy and explain its semantic choice; "
            "(3) adjudicate structured seller statements against visible appearance; "
            "(4) generate three localized copy versions and a six-image narrative from the same fact state; and "
            "(5) review individual assets and the complete set, replacing an accepted asset only with a safer candidate. "
            "Shopper-facing prose and machine-parseable listing data are kept separate so natural language cannot alter IDs, SKUs, or enumerated values.",
            "",
            "## 2. Factual Consistency and Source-Conflict Handling",
            "",
            "The Agent consolidates product JSON, SKUs, the marketplace taxonomy snapshot, and source-image observations into one traceable fact state. Titles and structured attributes provide seller statements, while source images verify appearance. When they conflict, the field is not copied directly into shopper prose; verified visible facts, machine-only fields, and rejected publication claims are handled separately.",
            "",
            *fact_decision_lines,
            table_strategy,
            "All three copy versions, image prompts, and media descriptions share the same adjudicated facts. Numeric conversions are deterministic only; material composition, performance, certifications, regional sizing, and any other unsupported information are never filled in.",
            "",
            "## 3. Marketplace Taxonomy and Listing Structure",
            "",
            f"- Leaf category ID: {taxonomy.category.category_id}",
            f"- Leaf category name: {english_display(taxonomy.category.name)}",
            f"- Category path: {english_display(taxonomy.category.path)}",
            f"- Mapped marketplace product/sales attributes: {len(taxonomy.attributes)}",
            "",
            "Category selection is grounded in product type, intended wearer, form, and verified attributes, and accepts only leaf nodes present in the marketplace snapshot. Submitted attribute IDs, enumerated value IDs, sales attributes, and SKU combinations are resolved back to platform relationships. Required fields without sufficient evidence remain unresolved instead of being guessed.",
            category_selection_note,
            schema_note,
            "",
            "## 4. Localization Strategy for the Three Target Markets",
            "",
            *market_lines,
            "",
            f"- {localization_delivery_note}",
            "- Shopper copy is organized from product characteristics to verifiable construction to conservative buying value; expansion stops whenever fact or pixel evidence is missing.",
            "- Shopper prose and the machine appendix are built independently. The model authors only the title, overview, highlights, and sizing note, while code renders the appendix deterministically from verified taxonomy, attributes, SKUs, and media contracts. Product IDs, URLs, marketplace mappings, and SKU data therefore remain consistent in all three languages.",
            "",
            "## 5. Six-Image Narrative and Video Role",
            "",
            "To serve all three markets while reducing cultural ambiguity, the visual system uses restrained, clear, product-centered commerce imagery. The hero image enables rapid recognition; detail images add evidence-backed views, construction, surfaces, options, or usage information; and the video connects the final assets. The six images maintain product identity, controlled backgrounds, and coordinated lighting while avoiding role duplication disguised as composition changes.",
            "",
            "- **main_image.jpeg | Conversion Entry Point:** Uses one complete product on a clean near-white background as the primary recognition anchor, with no text, prices, badges, watermarks, or unsupported props.",
            *detail_lines,
            f"- **product_video.mp4 | Dynamic Overview:** {video_delivery}",
            "",
            "Each image candidate is first checked for product identity, construction, color, unwanted text, and major defects. The complete six-image set is then reviewed for duplicate roles and information coverage. A candidate replaces an accepted asset only after passing hard gates without weakening the current delivery.",
            "",
            "## 6. Factual Consistency, Compliance, and Asset Quality Control",
            "",
            "- **Content compliance:** Shopper copy excludes prices, discounts, contact details, off-platform marketing, ratings, unsupported certifications, and body-altering claims. Generated images and video may not add text, marketplace marks, QR codes, watermarks, or brand identifiers.",
            "- **Factual consistency:** Copy claims, marketplace mappings, SKU tables, image roles, and media descriptions are projections of the same adjudicated fact state. When pixels and structured fields conflict, public content uses the verified result while the private evidence trail is preserved.",
            "- **Image quality control:** Checks format, dimensions, file size, blank or low-information output, near-duplicates, subject completeness, construction fidelity, and six-image role coverage.",
            "- **Video quality control:** Fully decodes the video stream and checks duration, container, resolution, playability, identity stability, and unacceptable defects.",
            "- **Repair strategy:** Repairs only defects backed by concrete evidence. A failed new candidate never replaces an accepted version, and every change synchronizes dependent copy, media descriptions, and video before review runs again.",
            "",
            "## 7. Listing-Ready Delivery Assurance",
            "",
            "The final delivery uses exactly the 11 required filenames. Each copy file includes a title, source platform, product ID and URL, leaf category, marketplace attributes, SKU/Spec IDs, descriptions of all five detail images, and a video description. The machine appendix remains field-parseable while shopper-facing prose stays natural in its target language.",
            "",
            "Every asset completes factual, format, specification, and semantic checks in a temporary directory before an atomic write to the output directory and one final full-delivery review. The objective is not to maximize generation volume, but to deliver a stable, fact-consistent, role-complete, playable localization package ready for the AliExpress listing workflow in a single run.",
        ]
        lines.append("")
        (work_dir / "strategy_document.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
