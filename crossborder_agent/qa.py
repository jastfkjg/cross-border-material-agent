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
from .table_evidence import presentation_view, select_render_table


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
        report.errors.append(f"{path.name} is 1 MB or larger")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        report.errors.append(f"{path.name} cannot be read as UTF-8 text: {exc}")
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
            report.errors.append(f"{path.name} is missing required content: {value}")

    language = path.stem.removeprefix("product_description_")
    buyer_surface, machine_surface = _description_language_surfaces(text, language)
    # Source URLs, identifiers and exact proper labels belong to the evidence
    # appendix. Source script is a hard defect only when it leaks into shopper
    # prose or media descriptions; appendix residue remains visible as a warning.
    buyer_surface = re.sub(r"https?://[^\s)>]+", "", buyer_surface)
    machine_surface = re.sub(r"https?://[^\s)>]+", "", machine_surface)
    if re.search(r"[\u4e00-\u9fff]", buyer_surface):
        report.errors.append(
            f"{path.name} contains unlocalized Chinese in shopper copy or media descriptions"
        )
    if re.search(r"[\u4e00-\u9fff]", machine_surface):
        report.warnings.append(
            f"{path.name} retains Chinese source values in the machine appendix"
        )
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
            report.errors.append(f"{path.name} exposes an internal evidence field: {marker}")

    sections = re.split(r"(?m)^## ", text)
    if len(sections) < 3:
        report.errors.append(f"{path.name} is missing an overview or highlights section")
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
            report.warnings.append(f"{path.name} has a short product overview")

    for item in taxonomy.attributes:
        for value in (item.attr_id, item.value_id):
            if value and value not in text:
                report.errors.append(
                    f"{path.name} is missing a marketplace listing attribute: {item.attr_id}/{item.value_id}"
                )
                break
    for sku in facts.skus:
        if sku.sku_id not in text:
            report.errors.append(f"{path.name} is missing SKU: {sku.sku_id}")
            continue
        if sku.spec_id and sku.spec_id not in text:
            report.errors.append(f"{path.name} is missing a Spec ID for SKU: {sku.sku_id}")
        for item in sku.attributes:
            if item.attribute_id and item.attribute_id not in text:
                report.errors.append(
                    f"{path.name} is missing an SKU component: {sku.sku_id}/{item.attribute_id}"
                )
                break
    table = select_render_table(facts.evidence_tables)
    if table is not None:
        try:
            table_view = presentation_view(table, language)
        except ValueError as exc:
            report.errors.append(
                f"{path.name} has an invalid source-table presentation contract: {exc}"
            )
        else:
            for row in table_view["rows"]:
                missing = next((value for value in row if value and value not in text), "")
                if missing:
                    report.errors.append(
                        f"{path.name} is missing a source-table cell: {table.table_id}/{missing}"
                    )


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
        report.errors.append("Missing delivery files: " + ", ".join(missing))
    if unexpected:
        report.warnings.append(
            "The output directory contains unexpected files: " + ", ".join(unexpected)
        )
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
            report.errors.append(f"Main-image format is not compliant: {main.format}")
        if main.width < 800 or main.height < 800:
            report.errors.append(
                f"Main-image dimensions are too small: {main.width}x{main.height}"
            )
        if main.size_bytes >= 1024 * 1024:
            report.errors.append("Main image is 1 MB or larger")
        main_quality = inspect_image_quality(output_dir / "main_image.jpeg")
        if main_quality is not None:
            if main_quality.luminance_stddev < 2 or main_quality.entropy < 0.8:
                report.errors.append("Main image appears blank or contains too little information")
            image_hashes.append(("main_image.jpeg", main_quality.difference_hash))
    except MediaError as exc:
        report.errors.append(str(exc))

    for index in range(1, 6):
        try:
            detail = inspect_image(output_dir / f"detail_image_{index}.jpeg")
            if detail.format not in {"JPEG", "JPG", "PNG"}:
                report.errors.append(
                    f"Detail-image {index} format is not compliant: {detail.format}"
                )
            if detail.width <= 260 or detail.height <= 260:
                report.errors.append(
                    f"Detail-image {index} dimensions are too small: {detail.width}x{detail.height}"
                )
            if detail.size_bytes > 5 * 1024 * 1024:
                report.errors.append(f"Detail image {index} exceeds 5 MB")
            detail_quality = inspect_image_quality(
                output_dir / f"detail_image_{index}.jpeg"
            )
            if detail_quality is not None:
                if (
                    detail_quality.luminance_stddev < 2
                    or detail_quality.entropy < 0.8
                ):
                    report.errors.append(
                        f"Detail image {index} appears blank or contains too little information"
                    )
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
                    f"Product images may contain duplicate visual content: {left_name}, {right_name}"
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
            "Perceptually distinct product images fall below the A6 80% usability reference"
        )

    try:
        inspect_video(output_dir / "product_video.mp4")
    except MediaError as exc:
        report.errors.append(str(exc))

    strategy = output_dir / "strategy_document.md"
    if strategy.stat().st_size >= 1024 * 1024:
        report.errors.append("strategy_document.md is 1 MB or larger")
    try:
        strategy_text = strategy.read_text(encoding="utf-8")
        for value in (
            facts.offer_id,
            taxonomy.category.category_id,
            "Fact",
            "Localization",
            "Quality",
        ):
            if value not in strategy_text:
                report.errors.append(
                    f"strategy_document.md is missing strategy information: {value}"
                )
    except (OSError, UnicodeError) as exc:
        report.errors.append(f"The strategy document cannot be read: {exc}")

    report.valid = not report.errors
    return report
