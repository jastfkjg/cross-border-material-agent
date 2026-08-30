"""Deterministic delivery gates aligned with the competition rubric."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .media import (
    MediaError,
    hash_distance,
    inspect_image,
    inspect_image_quality,
    inspect_video,
)
from .models import ProductFacts, TaxonomyResult


EXPECTED_FILES = {
    "product_description_en.md",
    "product_description_ko.md",
    "product_description_pt.md",
    "main_image.jpeg",
    "detail_image_1.jpeg",
    "detail_image_2.jpeg",
    "detail_image_3.jpeg",
    "detail_image_4.jpeg",
    "detail_image_5.jpeg",
    "product_video.mp4",
    "strategy_document.md",
}


@dataclass(slots=True)
class ValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_MEDIA_HEADINGS = {
    "en": "Media guide",
    "ko": "미디어 안내",
    "pt": "Guia de mídia",
}


def _description_language_surfaces(text: str, language: str) -> tuple[str, str]:
    """Separate locale-facing prose from the exact machine listing appendix.

    Media descriptions are shopper-facing, but their exact artifact filenames
    are protocol metadata.  Keep the complete media block in the machine
    appendix while exposing only the captions to language-quality checks.
    """

    headings = list(re.finditer(r"(?m)^## ", text))
    appendix_start = headings[2].start() if len(headings) >= 3 else len(text)
    buyer_surface = text[:appendix_start]
    machine_surface = text[appendix_start:]

    media_heading = _MEDIA_HEADINGS.get(language)
    if media_heading:
        match = re.search(rf"(?m)^## {re.escape(media_heading)}\s*$", text)
        if match:
            next_heading = re.search(r"(?m)^## ", text[match.end() :])
            media_end = (
                match.end() + next_heading.start()
                if next_heading
                else len(text)
            )
            media_surface = text[match.start() : media_end]
            # Generated descriptions use ``- **filename:** caption``.  The
            # caption must be localized, while showing filenames to a copy
            # evaluator caused machine metadata to be judged as buyer prose.
            media_captions = re.sub(
                r"(?m)^\s*-\s+\*\*[^*\n]+:\*\*\s*",
                "- ",
                media_surface,
            )
            buyer_surface += "\n" + media_captions
    return buyer_surface, machine_surface


def _validate_description(
    path: Path,
    facts: ProductFacts,
    taxonomy: TaxonomyResult,
    report: ValidationReport,
) -> None:
    if path.stat().st_size >= 1024 * 1024:
        report.errors.append(f"{path.name} 达到或超过 1MB")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        report.errors.append(f"{path.name} 无法作为 UTF-8 文本读取: {exc}")
        return
    required_values = [
        facts.offer_id,
        facts.source_url,
        taxonomy.category.category_id,
        "main_image.jpeg",
        "detail_image_1.jpeg",
        "detail_image_2.jpeg",
        "detail_image_3.jpeg",
        "detail_image_4.jpeg",
        "detail_image_5.jpeg",
        "product_video.mp4",
    ]
    for value in required_values:
        if value and value not in text:
            report.errors.append(f"{path.name} 缺少必需内容: {value}")

    language = path.stem.removeprefix("product_description_")
    buyer_surface, machine_surface = _description_language_surfaces(text, language)
    # Source URLs, identifiers and exact proper labels belong to the evidence
    # appendix. Source script is a hard defect only when it leaks into shopper
    # prose or media descriptions; appendix residue remains visible as a warning.
    buyer_surface = re.sub(r"https?://[^\s)>]+", "", buyer_surface)
    machine_surface = re.sub(r"https?://[^\s)>]+", "", machine_surface)
    if re.search(r"[\u4e00-\u9fff]", buyer_surface):
        report.errors.append(f"{path.name} 买家文案或媒体描述含未本地化中文")
    if re.search(r"[\u4e00-\u9fff]", machine_surface):
        report.warnings.append(f"{path.name} 机器附录保留了中文来源值")
    forbidden_internal_markers = (
        "/ret/result/result",
        "Source evidence",
        "Canonical source",
        "Machine-readable source",
        "원본 근거",
        "기계 판독",
        "Evidência na origem",
        "dados canônicos",
    )
    for marker in forbidden_internal_markers:
        if marker.casefold() in text.casefold():
            report.errors.append(f"{path.name} 暴露内部证据字段: {marker}")

    sections = re.split(r"(?m)^## ", text)
    if len(sections) < 3:
        report.errors.append(f"{path.name} 缺少商品描述或卖点章节")
    else:
        overview_body = sections[1].split("\n", 1)[-1].strip()
        paragraphs = [
            item.strip()
            for item in re.split(r"\n\s*\n", overview_body)
            if item.strip()
        ]
        minimum_overview_chars = {"en": 70, "ko": 35, "pt": 70}.get(language, 70)
        overview_length = len(
            re.sub(r"\s+", "", " ".join(paragraphs))
        )
        if overview_length < minimum_overview_chars:
            report.warnings.append(f"{path.name} 商品描述内容偏短")

    for item in taxonomy.attributes:
        for value in (item.attr_id, item.value_id):
            if value and value not in text:
                report.errors.append(
                    f"{path.name} 缺少平台上架属性: {item.attr_id}/{item.value_id}"
                )
                break
    for sku in facts.skus:
        if sku.sku_id not in text:
            report.errors.append(f"{path.name} 缺少 SKU: {sku.sku_id}")
            continue
        if sku.spec_id and sku.spec_id not in text:
            report.errors.append(f"{path.name} 缺少 Spec ID: {sku.sku_id}")
        for item in sku.attributes:
            if item.attribute_id and item.attribute_id not in text:
                report.errors.append(
                    f"{path.name} 缺少 SKU 分解项: {sku.sku_id}/{item.attribute_id}"
                )
                break
    for item in facts.size_chart_rows:
        for value in (
            item.bust_cm,
            item.length_cm,
        ):
            if value and value not in text:
                report.errors.append(
                    f"{path.name} 缺少卖家尺码数据: {item.size_label}/{value}"
                )
                break


def validate_delivery(
    output_dir: Path, facts: ProductFacts, taxonomy: TaxonomyResult
) -> ValidationReport:
    report = ValidationReport(valid=True)
    actual_files = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    }
    missing = sorted(EXPECTED_FILES - actual_files)
    unexpected = sorted(actual_files - EXPECTED_FILES)
    if missing:
        report.errors.append("缺少交付文件: " + ", ".join(missing))
    if unexpected:
        report.warnings.append("输出目录包含额外文件: " + ", ".join(unexpected))
    if missing:
        report.valid = False
        return report

    for language in ("en", "ko", "pt"):
        _validate_description(
            output_dir / f"product_description_{language}.md", facts, taxonomy, report
        )

    image_hashes: list[tuple[str, int]] = []
    try:
        main = inspect_image(output_dir / "main_image.jpeg")
        if main.format not in {"JPEG", "JPG", "PNG"}:
            report.errors.append(f"主图格式不合规: {main.format}")
        if main.width < 800 or main.height < 800:
            report.errors.append(f"主图尺寸不足: {main.width}x{main.height}")
        if main.size_bytes > 5 * 1024 * 1024:
            report.errors.append("主图超过 5MB")
        main_quality = inspect_image_quality(output_dir / "main_image.jpeg")
        if main_quality is not None:
            if main_quality.luminance_stddev < 2 or main_quality.entropy < 0.8:
                report.errors.append("主图疑似空白或信息量过低")
            image_hashes.append(("main_image.jpeg", main_quality.difference_hash))
    except MediaError as exc:
        report.errors.append(str(exc))

    for index in range(1, 6):
        try:
            detail = inspect_image(output_dir / f"detail_image_{index}.jpeg")
            if detail.format not in {"JPEG", "JPG", "PNG"}:
                report.errors.append(f"详情图 {index} 格式不合规: {detail.format}")
            if detail.width <= 260 or detail.height <= 260:
                report.errors.append(
                    f"详情图 {index} 尺寸不足: {detail.width}x{detail.height}"
                )
            if detail.size_bytes > 5 * 1024 * 1024:
                report.errors.append(f"详情图 {index} 超过 5MB")
            detail_quality = inspect_image_quality(
                output_dir / f"detail_image_{index}.jpeg"
            )
            if detail_quality is not None:
                if (
                    detail_quality.luminance_stddev < 2
                    or detail_quality.entropy < 0.8
                ):
                    report.errors.append(f"详情图 {index} 疑似空白或信息量过低")
                image_hashes.append(
                    (f"detail_image_{index}.jpeg", detail_quality.difference_hash)
                )
        except MediaError as exc:
            report.errors.append(str(exc))

    duplicate_threshold = 10
    for left_index, (left_name, left_hash) in enumerate(image_hashes):
        for right_name, right_hash in image_hashes[left_index + 1 :]:
            if hash_distance(left_hash, right_hash) <= duplicate_threshold:
                report.warnings.append(
                    f"商品图片视觉内容可能重复: {left_name}, {right_name}"
                )

    distinct_hashes: list[int] = []
    for _, image_hash in image_hashes:
        if all(
            hash_distance(image_hash, seen) > duplicate_threshold
            for seen in distinct_hashes
        ):
            distinct_hashes.append(image_hash)
    if image_hashes and len(distinct_hashes) / len(image_hashes) < 0.8:
        report.warnings.append(
            "商品图片的感知差异率低于 A6 的 80% 可用率参考线"
        )

    try:
        inspect_video(output_dir / "product_video.mp4")
    except MediaError as exc:
        report.errors.append(str(exc))

    strategy = output_dir / "strategy_document.md"
    if strategy.stat().st_size >= 1024 * 1024:
        report.errors.append("strategy_document.md 达到或超过 1MB")
    try:
        strategy_text = strategy.read_text(encoding="utf-8")
        for value in (
            facts.offer_id,
            taxonomy.category.category_id,
            "事实",
            "本地化",
            "质检",
        ):
            if value not in strategy_text:
                report.errors.append(f"strategy_document.md 缺少策略信息: {value}")
    except (OSError, UnicodeError) as exc:
        report.errors.append(f"策略说明无法读取: {exc}")

    report.valid = not report.errors
    return report
