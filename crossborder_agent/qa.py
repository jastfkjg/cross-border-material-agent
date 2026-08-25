"""Deterministic delivery gates aligned with the competition rubric."""

from __future__ import annotations

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
        facts.platform,
        facts.source_url,
        taxonomy.category.category_id,
        taxonomy.category.name,
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
    for item in facts.attributes:
        for value in (
            item.attribute_id,
            item.name,
            item.value,
            item.evidence_pointer,
        ):
            if value and value not in text:
                report.errors.append(
                    f"{path.name} 缺少商品属性证据: {item.attribute_id}/{item.name}"
                )
                break
    for item in taxonomy.attributes:
        for value in (
            item.attr_id,
            item.name,
            item.source_value,
            item.platform_value,
            item.source_evidence_pointer,
        ):
            if value and value not in text:
                report.errors.append(
                    f"{path.name} 缺少平台属性映射: {item.attr_id}/{item.name}"
                )
                break
    detailed_sku_evidence = sum(len(sku.attributes) for sku in facts.skus) <= 1500
    for sku in facts.skus:
        if sku.sku_id not in text:
            report.errors.append(f"{path.name} 缺少 SKU: {sku.sku_id}")
            continue
        if sku.evidence_pointer not in text:
            report.errors.append(f"{path.name} 缺少 SKU 证据: {sku.sku_id}")
        for item in sku.attributes:
            required_component_values = [item.value]
            if detailed_sku_evidence:
                required_component_values.extend(
                    [item.attribute_id, item.evidence_pointer]
                )
            if any(
                value and value not in text for value in required_component_values
            ):
                report.errors.append(
                    f"{path.name} 缺少 SKU 分解项: {sku.sku_id}/{item.attribute_id}"
                )
                break
    for item in facts.size_chart_rows:
        for value in (
            item.size_label,
            item.bust_cm,
            item.length_cm,
            item.evidence_pointer,
        ):
            if value and value not in text:
                report.errors.append(
                    f"{path.name} 缺少详情图尺码证据: {item.size_label}/{value}"
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

    for left_index, (left_name, left_hash) in enumerate(image_hashes):
        for right_name, right_hash in image_hashes[left_index + 1 :]:
            if hash_distance(left_hash, right_hash) <= 2:
                report.warnings.append(
                    f"商品图片视觉内容可能重复: {left_name}, {right_name}"
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
