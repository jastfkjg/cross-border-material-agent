"""Transaction responsibilities for the delivery pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from ..agent_tools import BoundedToolRegistry, ToolExecution
from ..media import (
    MediaError,
    create_catalog_video,
    create_emergency_image,
    create_slideshow_video,
    inspect_image,
    inspect_video,
    normalize_image,
)
from ..localization import generate_copy_payload
from ..models import (
    AgentActionResult,
    AssetResult,
    CreativePlan,
    ProductFacts,
    RunState,
    TaxonomyResult,
)
from ..qa import EXPECTED_FILES, _description_language_surfaces
from .common import (
    AGENT_SNAPSHOT_FILES as _AGENT_SNAPSHOT_FILES,
)


class TransactionPipelineMixin:
    @staticmethod
    def _artifact_hash(path: Path) -> str:
        if not path.is_file():
            return "missing"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _delivery_fingerprint(
        self,
        *,
        state: RunState,
        localization_payloads: dict[str, dict[str, Any]],
        localization_sources: dict[str, str],
        work_dir: Path,
    ) -> str:
        """Identify the exact evidence bundle seen by evaluators."""

        digest = hashlib.sha256(b"delivery-evidence-v1\0")
        for filename in _AGENT_SNAPSHOT_FILES:
            digest.update(filename.encode("utf-8"))
            digest.update(self._artifact_hash(work_dir / filename).encode("ascii"))
        mutable_state = {
            "assets": [
                {
                    "name": item.name,
                    "source_url": item.source_url,
                    "model": item.model,
                    "generated": item.generated,
                }
                for item in state.assets
            ],
            "localization_payloads": localization_payloads,
            "localization_sources": localization_sources,
            "visual_set_review": state.visual_set_review,
            "reconciled_fact_ledger": state.facts.reconciled_fact_ledger,
            "claim_ledger": [
                {
                    "claim_id": item.claim_id,
                    "concept": item.concept,
                    "value": item.value,
                    "evidence_pointer": item.evidence_pointer,
                    "allowed_surfaces": item.allowed_surfaces,
                }
                for item in state.claim_ledger
            ],
            "taxonomy": {
                "category_id": state.taxonomy.category.category_id,
                "schema_id": state.taxonomy.attribute_schema_category_id,
                "attributes": [
                    (item.attr_id, item.value_id, item.platform_value)
                    for item in state.taxonomy.attributes
                ],
            },
            "expected_delivery_spec_version": state.expected_delivery_spec.get(
                "version", ""
            ),
        }
        digest.update(
            json.dumps(
                mutable_state, ensure_ascii=False, sort_keys=True, default=str
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def _capture_repair_checkpoint(
        self,
        *,
        state: RunState,
        localization_payloads: dict[str, dict[str, Any]],
        localization_sources: dict[str, str],
        work_dir: Path,
    ) -> dict[str, Any]:
        checkpoint_id = uuid.uuid4().hex
        checkpoint_dir = work_dir / ".agent-checkpoints" / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        for filename in _AGENT_SNAPSHOT_FILES:
            source = work_dir / filename
            if source.is_file():
                shutil.copy2(source, checkpoint_dir / filename)
        return {
            "directory": checkpoint_dir,
            "assets": copy.deepcopy(state.assets),
            "localization_payloads": copy.deepcopy(localization_payloads),
            "localization_sources": copy.deepcopy(localization_sources),
            "visual_set_review": copy.deepcopy(state.visual_set_review),
            "reconciled_fact_ledger": copy.deepcopy(state.facts.reconciled_fact_ledger),
            "taxonomy": copy.deepcopy(state.taxonomy),
            "claim_ledger": copy.deepcopy(state.claim_ledger),
            "canonical_product_state": copy.deepcopy(state.canonical_product_state),
            "evidence_sufficiency": copy.deepcopy(state.evidence_sufficiency),
            "expected_delivery_spec": copy.deepcopy(state.expected_delivery_spec),
            "dependency_state": copy.deepcopy(state.dependency_state),
        }

    def _restore_repair_checkpoint(
        self,
        checkpoint: dict[str, Any],
        *,
        state: RunState,
        localization_payloads: dict[str, dict[str, Any]],
        localization_sources: dict[str, str],
        work_dir: Path,
    ) -> None:
        checkpoint_dir = checkpoint["directory"]
        for filename in _AGENT_SNAPSHOT_FILES:
            source = checkpoint_dir / filename
            if not source.is_file():
                continue
            staged = work_dir / f".restore-{uuid.uuid4().hex}-{filename}"
            shutil.copy2(source, staged)
            os.replace(staged, work_dir / filename)
        state.assets = copy.deepcopy(checkpoint["assets"])
        localization_payloads.clear()
        localization_payloads.update(copy.deepcopy(checkpoint["localization_payloads"]))
        localization_sources.clear()
        localization_sources.update(copy.deepcopy(checkpoint["localization_sources"]))
        state.visual_set_review = copy.deepcopy(checkpoint["visual_set_review"])
        state.facts.reconciled_fact_ledger = copy.deepcopy(
            checkpoint["reconciled_fact_ledger"]
        )
        restored_taxonomy = checkpoint["taxonomy"]
        state.taxonomy.category = copy.deepcopy(restored_taxonomy.category)
        state.taxonomy.attributes = copy.deepcopy(restored_taxonomy.attributes)
        state.taxonomy.missing_required = copy.deepcopy(
            restored_taxonomy.missing_required
        )
        state.taxonomy.attribute_schema_category_id = (
            restored_taxonomy.attribute_schema_category_id
        )
        state.claim_ledger = copy.deepcopy(checkpoint["claim_ledger"])
        state.canonical_product_state = copy.deepcopy(
            checkpoint["canonical_product_state"]
        )
        state.evidence_sufficiency = copy.deepcopy(checkpoint["evidence_sufficiency"])
        state.expected_delivery_spec = copy.deepcopy(
            checkpoint["expected_delivery_spec"]
        )
        state.dependency_state = copy.deepcopy(checkpoint["dependency_state"])

    def _rebuild_synchronized_catalog_video(
        self,
        assets: list[AssetResult],
        work_dir: Path,
    ) -> ToolExecution:
        """Rebuild a local video from the current image set as a consistency fallback."""

        video_asset = next(
            (item for item in assets if item.name == "product_video.mp4"), None
        )
        if video_asset is None:
            return ToolExecution("failed", "product video asset missing")
        image_paths = [work_dir / "main_image.jpeg"] + [
            work_dir / f"detail_image_{index}.jpeg" for index in range(1, 6)
        ]
        staged = work_dir / f".synchronized-video-{uuid.uuid4().hex}.mp4"
        try:
            create_catalog_video(image_paths, staged, duration=8)
            inspect_video(staged)
            os.replace(staged, Path(video_asset.path))
        except (MediaError, OSError) as exc:
            return ToolExecution(
                "failed", f"catalog video synchronization failed: {exc}"
            )
        finally:
            staged.unlink(missing_ok=True)
        video_asset.source_url = ""
        video_asset.model = "ffmpeg-catalog-synchronized"
        video_asset.generated = False
        video_asset.fallback_reason = (
            "rebuilt after final image changes for cross-asset consistency"
        )
        video_asset.description = (
            "Eight-second catalog video synchronized with the current final image set"
        )
        return ToolExecution("completed", "video rebuilt from the current final images")

    def _synchronize_repair_dependencies(
        self,
        *,
        round_index: int,
        changed_targets: set[str],
        registry: BoundedToolRegistry,
        facts: ProductFacts,
        taxonomy: TaxonomyResult,
        creative_plan: CreativePlan,
        state: RunState,
        localization_payloads: dict[str, dict[str, Any]],
        work_dir: Path,
    ) -> bool:
        """Refresh media descriptions, visual review and dependent video after repairs."""

        changed_images = {
            target for target in changed_targets if target.endswith(".jpeg")
        }
        if not changed_images:
            return True

        video_changed = "product_video.mp4" in changed_targets
        main_changed = "main_image.jpeg" in changed_images
        video_asset = next(
            (item for item in state.assets if item.name == "product_video.mp4"), None
        )
        needs_catalog_refresh = bool(
            video_asset is not None and not video_asset.generated and not video_changed
        )
        if main_changed and not video_changed:
            result = registry.execute(
                "regenerate_video",
                "product_video.mp4",
                "Synchronize every video frame with the newly accepted final hero while preserving exact source-backed product identity.",
            )
            state.agent_actions.append(
                AgentActionResult(
                    round_index=round_index,
                    tool="regenerate_video",
                    target="product_video.mp4",
                    status=result.status,
                    detail="automatic dependency repair: " + result.detail,
                )
            )
            if result.status != "completed":
                result = self._rebuild_synchronized_catalog_video(
                    state.assets, work_dir
                )
                state.agent_actions.append(
                    AgentActionResult(
                        round_index=round_index,
                        tool="synchronize_catalog_video",
                        target="product_video.mp4",
                        status=result.status,
                        detail=result.detail,
                    )
                )
            if result.status != "completed":
                return False
        elif needs_catalog_refresh:
            result = self._rebuild_synchronized_catalog_video(state.assets, work_dir)
            state.agent_actions.append(
                AgentActionResult(
                    round_index=round_index,
                    tool="synchronize_catalog_video",
                    target="product_video.mp4",
                    status=result.status,
                    detail=result.detail,
                )
            )
            if result.status != "completed":
                return False

        refreshed_review = self._review_visual_set(facts, state.assets)
        state.visual_set_review = refreshed_review or {}
        self._write_localized_descriptions(
            facts,
            taxonomy,
            creative_plan,
            localization_payloads,
            state.assets,
            work_dir,
            state.visual_set_review,
        )
        self.trace.emit(
            "agent.dependencies_synchronized",
            round_index=round_index,
            changed_targets=sorted(changed_targets),
            visual_review_refreshed=bool(refreshed_review),
        )
        return True

    @staticmethod
    def _repair_batch_consistent(
        changed_targets: set[str],
        *,
        state: RunState,
        localization_payloads: dict[str, dict[str, Any]],
        work_dir: Path,
    ) -> tuple[bool, str]:
        """Perform a local post-batch consistency checkpoint before re-evaluation."""

        affected = set(changed_targets)
        conceptual_targets = {"fact_ledger", "taxonomy", "detail_image_set"}
        affected.difference_update(conceptual_targets)
        if "detail_image_set" in changed_targets:
            affected.update(f"detail_image_{index}.jpeg" for index in range(1, 6))
        if "taxonomy" in changed_targets:
            affected.update(
                f"product_description_{language}.md" for language in ("en", "ko", "pt")
            )
        if any(name.endswith(".jpeg") for name in changed_targets):
            affected.update(
                f"product_description_{language}.md" for language in ("en", "ko", "pt")
            )
        try:
            for target in sorted(affected):
                path = work_dir / target
                if not path.is_file() or path.stat().st_size <= 0:
                    return False, f"missing or empty synchronized target: {target}"
                if target.endswith(".jpeg"):
                    inspect_image(path)
                elif target.endswith(".mp4"):
                    inspect_video(path)
                elif target.endswith(".md"):
                    language = target.removeprefix("product_description_").removesuffix(
                        ".md"
                    )
                    payload = localization_payloads.get(language)
                    if not isinstance(payload, dict) or not all(
                        payload.get(key)
                        for key in ("title", "overview", "highlights", "fit_note")
                    ):
                        return (
                            False,
                            f"localized payload incomplete after batch: {language}",
                        )
                    text = path.read_text(encoding="utf-8")
                    buyer_surface, _ = _description_language_surfaces(text, language)
                    buyer_surface = re.sub(r"https?://[^\s)>]+", "", buyer_surface)
                    if re.search(r"[\u4e00-\u9fff]", buyer_surface):
                        return (
                            False,
                            f"buyer copy contains Chinese after batch: {target}",
                        )
        except (MediaError, OSError, UnicodeError) as exc:
            return False, str(exc)

        changed_images = {name for name in changed_targets if name.endswith(".jpeg")}
        rows = (
            state.visual_set_review.get("assets", [])
            if isinstance(state.visual_set_review, dict)
            else []
        )
        if changed_images and isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict) or row.get("name") not in changed_images:
                    continue
                if any(
                    row.get(key) is False
                    for key in (
                        "usable",
                        "identity_consistent",
                        "construction_consistent",
                        "slot_match",
                    )
                ) or any(
                    row.get(key) is True for key in ("unwanted_text", "major_artifacts")
                ):
                    return (
                        False,
                        f"set review rejects synchronized image: {row.get('name')}",
                    )
        return (
            True,
            "post-batch files, payloads, and explicit set-review gates are consistent",
        )

    def _ensure_minimum_delivery(
        self,
        *,
        facts: ProductFacts,
        taxonomy: TaxonomyResult,
        creative_plan: CreativePlan,
        state: RunState,
        localization_payloads: dict[str, dict[str, Any]],
        localization_sources: dict[str, str],
        work_dir: Path,
        downloads_dir: Path,
        plan_model: str,
        reason: str,
    ) -> None:
        """Fill every required slot from the best surviving grounded material.

        This is an availability boundary, not a second semantic workflow. It
        preserves existing artifacts, uses seller-source pixels when reachable,
        derives missing media locally, and falls back to a claim-free neutral
        image only when no product pixels survive.
        """

        clean_reason = " ".join(str(reason).split())[:1200]
        if clean_reason:
            warning = f"Delivery entered availability fallback: {clean_reason}"
            if warning not in self.warnings:
                self.warnings.append(warning)
            self.logger.warning(warning)

        def image_is_readable(path: Path) -> bool:
            try:
                inspect_image(path)
                return True
            except (MediaError, OSError):
                return False

        def replace_normalized(
            source: Path,
            destination: Path,
            *,
            canvas: tuple[int, int],
            white_background: bool,
            focus_crop: str = "",
            max_bytes: int = 5 * 1024 * 1024,
        ) -> None:
            staged = destination.with_name(
                f".{destination.stem}-{uuid.uuid4().hex}.recovery.jpeg"
            )
            try:
                normalize_image(
                    source,
                    staged,
                    canvas=canvas,
                    max_bytes=max_bytes,
                    white_background=white_background,
                    focus_crop=focus_crop,
                )
                os.replace(staged, destination)
            finally:
                staged.unlink(missing_ok=True)

        by_name = {item.name: item for item in state.assets}
        main_path = work_dir / "main_image.jpeg"
        if image_is_readable(main_path) and main_path.stat().st_size >= 1024 * 1024:
            replace_normalized(
                main_path,
                main_path,
                canvas=(1600, 1600),
                white_background=True,
                max_bytes=1024 * 1024 - 1,
            )
        if not image_is_readable(main_path):
            local_source = next(
                (
                    work_dir / f"detail_image_{index}.jpeg"
                    for index in range(1, 6)
                    if image_is_readable(work_dir / f"detail_image_{index}.jpeg")
                ),
                None,
            )
            if local_source is not None:
                replace_normalized(
                    local_source,
                    main_path,
                    canvas=(1600, 1600),
                    white_background=True,
                    max_bytes=1024 * 1024 - 1,
                )
            else:
                downloaded = False
                if self.deadline - time.monotonic() > 20:
                    try:
                        self._fallback_image(
                            facts.all_image_urls(),
                            main_path,
                            downloads_dir,
                            canvas=(1600, 1600),
                            white_background=True,
                            max_bytes=1024 * 1024 - 1,
                        )
                        downloaded = True
                    except Exception as exc:
                        self.logger.warning(
                            "Final source-image fallback is unavailable; creating a neutral placeholder: %s",
                            exc,
                        )
                if not downloaded:
                    create_emergency_image(main_path, canvas=(1600, 1600))
            by_name["main_image.jpeg"] = AssetResult(
                name="main_image.jpeg",
                path=str(main_path),
                model="availability-fallback",
                generated=False,
                fallback_reason=clean_reason or "missing main image recovered",
                description="Best surviving source-faithful availability fallback",
            )
        elif "main_image.jpeg" not in by_name:
            by_name["main_image.jpeg"] = AssetResult(
                name="main_image.jpeg",
                path=str(main_path),
                model="surviving-local-artifact",
                generated=False,
                fallback_reason=clean_reason,
                description="Best surviving local hero artifact",
            )
        else:
            by_name["main_image.jpeg"].path = str(main_path)

        focus_crops = ("", "upper", "lower", "left", "right")
        for index, focus_crop in enumerate(focus_crops, start=1):
            name = f"detail_image_{index}.jpeg"
            destination = work_dir / name
            if not image_is_readable(destination):
                replace_normalized(
                    main_path,
                    destination,
                    canvas=(1200, 1500),
                    white_background=False,
                    focus_crop=focus_crop,
                )
                by_name[name] = AssetResult(
                    name=name,
                    path=str(destination),
                    model="availability-fallback",
                    generated=False,
                    fallback_reason=clean_reason or "missing detail image recovered",
                    description=(
                        f"Local {focus_crop or 'complete'} view derived from the surviving image"
                    ),
                )
            elif name not in by_name:
                by_name[name] = AssetResult(
                    name=name,
                    path=str(destination),
                    model="surviving-local-artifact",
                    generated=False,
                    fallback_reason=clean_reason,
                    description="Best surviving local detail artifact",
                )
            else:
                by_name[name].path = str(destination)

        video_name = "product_video.mp4"
        video_path = work_dir / video_name
        try:
            inspect_video(video_path)
        except (MediaError, OSError):
            video_path.unlink(missing_ok=True)
            image_paths = [main_path] + [
                work_dir / f"detail_image_{index}.jpeg" for index in range(1, 6)
            ]
            try:
                create_catalog_video(image_paths, video_path, duration=8)
            except (MediaError, OSError):
                create_slideshow_video(main_path, video_path, duration=8)
            by_name[video_name] = AssetResult(
                name=video_name,
                path=str(video_path),
                model="ffmpeg-availability-fallback",
                generated=False,
                fallback_reason=clean_reason or "missing video recovered",
                description="Playable silent catalog fallback",
            )
        else:
            if video_name not in by_name:
                by_name[video_name] = AssetResult(
                    name=video_name,
                    path=str(video_path),
                    model="surviving-local-artifact",
                    generated=False,
                    fallback_reason=clean_reason,
                    description="Best surviving local video artifact",
                )
            else:
                by_name[video_name].path = str(video_path)

        state.assets = [
            by_name[name]
            for name in (
                "main_image.jpeg",
                "detail_image_1.jpeg",
                "detail_image_2.jpeg",
                "detail_image_3.jpeg",
                "detail_image_4.jpeg",
                "detail_image_5.jpeg",
                "product_video.mp4",
            )
            if name in by_name
        ]

        for language in ("en", "ko", "pt"):
            if not isinstance(localization_payloads.get(language), dict):
                # Recovery still prefers model-authored prose when the chat
                # capability and a safe commit reserve survive. The same
                # function owns validation and falls back deterministically if
                # that one bounded draft fails.
                recovery_client = (
                    self.client
                    if self.client is not None
                    and self.client.operation_available("chat")
                    and self.deadline - time.monotonic() > 90
                    else None
                )
                payload, source = generate_copy_payload(
                    language,
                    facts,
                    taxonomy,
                    creative_plan,
                    recovery_client,
                    claim_ledger=state.claim_ledger,
                    agent_guidance=(
                        "This is an availability recovery. Produce a complete, conservative draft from the supplied "
                        "evidence; omit uncertain claims and do not mention the runtime failure."
                    ),
                    audit_valid_draft=False,
                )
                localization_payloads[language] = payload
                localization_sources[language] = source
        self._write_localized_descriptions(
            facts,
            taxonomy,
            creative_plan,
            localization_payloads,
            state.assets,
            work_dir,
            state.visual_set_review,
        )
        try:
            self._write_strategy_document(
                state,
                localization_sources,
                localization_payloads,
                plan_model,
                work_dir,
            )
        except Exception as exc:
            self.logger.warning(
                "Full strategy-document generation failed; writing a minimal parseable version: %s",
                exc,
            )
            (work_dir / "strategy_document.md").write_text(
                "\n".join(
                    (
                        "# Agent Localization Delivery Strategy",
                        "",
                        f"- Product ID: {facts.offer_id}",
                        f"- Marketplace leaf category ID: {taxonomy.category.category_id}",
                        "- Facts: all public content is grounded in input product data and saved assets.",
                        "- Localization: separate English, Korean, and Brazilian Portuguese versions are delivered.",
                        "- Quality control: format, dimensions, naming, and playability are checked; the best available snapshot is submitted after an unexpected failure.",
                        "",
                    )
                ),
                encoding="utf-8",
            )

    def _commit_delivery(self, work_dir: Path) -> None:
        for filename in sorted(EXPECTED_FILES):
            source = work_dir / filename
            destination = self.output_dir / filename
            os.replace(source, destination)
