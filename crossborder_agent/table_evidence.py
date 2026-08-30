"""Domain-neutral source-table grounding and presentation materialization."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from .models import EvidenceCell, EvidenceTable


_PRESENTATION_DECISIONS = {
    "render",
    "preserve_source",
    "omit",
    "request_verification",
}
_TARGET_LOCALES = ("en", "ko", "pt")
_MAX_TABLES = 8
_MAX_CELLS = 240
MAX_SOURCE_TABLE_ROWS = 40
MAX_SOURCE_TABLE_COLUMNS = 16
MAX_RENDER_TABLE_ROWS = 24
MAX_RENDER_TABLE_COLUMNS = 10


def _clean_text(value: Any, limit: int = 300) -> str:
    return " ".join(str(value or "").split())[:limit]


def _cell_signature(cells: Iterable[EvidenceCell], source_index: int) -> str:
    payload = [
        source_index,
        *[
            [cell.row, cell.column, cell.text, cell.row_span, cell.column_span]
            for cell in sorted(cells, key=lambda item: (item.row, item.column))
        ],
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def _localized_strings(value: Any, *, list_values: bool = False) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    result: dict[str, Any] = {}
    for locale in _TARGET_LOCALES:
        candidate = raw.get(locale)
        if list_values:
            if not isinstance(candidate, list):
                continue
            values = [
                clean
                for item in candidate[:8]
                if (clean := _clean_text(item, 240))
            ]
            if values:
                result[locale] = values
        else:
            clean = _clean_text(candidate, 240)
            if clean:
                result[locale] = clean
    return result


def _normalize_presentation(
    value: Any,
    *,
    rows: set[int],
    columns: set[int],
) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    validation_errors: list[str] = []
    decision = _clean_text(raw.get("decision"), 40).casefold()
    if decision not in _PRESENTATION_DECISIONS:
        validation_errors.append(
            "decision must be one of render, preserve_source, omit, request_verification"
        )
        decision = "request_verification"

    try:
        priority = int(raw.get("priority", 0))
    except (TypeError, ValueError):
        priority = 0
    priority = max(0, min(100, priority))

    target = raw.get("target_detail_index")
    target = target if isinstance(target, int) and not isinstance(target, bool) else 0
    display_locale = _clean_text(raw.get("display_locale"), 8).casefold()
    if display_locale not in _TARGET_LOCALES:
        display_locale = "en"

    selected_columns: list[dict[str, Any]] = []
    seen_columns: set[int] = set()
    raw_columns = raw.get("columns")
    if isinstance(raw_columns, list):
        if len(raw_columns) > MAX_RENDER_TABLE_COLUMNS:
            validation_errors.append(
                "columns exceeds the single-image rendering capacity of "
                f"{MAX_RENDER_TABLE_COLUMNS} entries"
            )
        for item in raw_columns[:MAX_SOURCE_TABLE_COLUMNS]:
            if not isinstance(item, dict):
                validation_errors.append("every columns entry must be an object")
                continue
            source_column = item.get("source_column")
            if not isinstance(source_column, int) or isinstance(source_column, bool):
                validation_errors.append("source_column must be an integer")
                continue
            if source_column not in columns:
                validation_errors.append(
                    f"source_column {source_column} does not exist in observed cells"
                )
                continue
            if source_column in seen_columns:
                validation_errors.append(
                    f"source_column {source_column} is selected more than once"
                )
                continue
            labels = _localized_strings(item.get("labels"))
            missing_label_locales = [
                locale for locale in _TARGET_LOCALES if locale not in labels
            ]
            if missing_label_locales:
                validation_errors.append(
                    f"source_column {source_column} is missing labels for: "
                    + ", ".join(missing_label_locales)
                )
                continue
            selected_columns.append(
                {"source_column": source_column, "labels": labels}
            )
            seen_columns.add(source_column)

    else:
        validation_errors.append("columns must be an array")

    included_rows: list[int] = []
    raw_rows = raw.get("included_rows")
    if isinstance(raw_rows, list):
        if len(raw_rows) > MAX_RENDER_TABLE_ROWS:
            validation_errors.append(
                "included_rows exceeds the single-image rendering capacity of "
                f"{MAX_RENDER_TABLE_ROWS} entries"
            )
        for row in raw_rows[:MAX_SOURCE_TABLE_ROWS]:
            if not isinstance(row, int) or isinstance(row, bool):
                validation_errors.append("every included_rows entry must be an integer")
                continue
            if row not in rows:
                validation_errors.append(
                    f"included row {row} does not exist in observed cells"
                )
                continue
            if row in included_rows:
                validation_errors.append(f"included row {row} is selected more than once")
                continue
            included_rows.append(row)
    else:
        validation_errors.append("included_rows must be an array")

    titles = _localized_strings(raw.get("titles"))
    missing_title_locales = [locale for locale in _TARGET_LOCALES if locale not in titles]
    if missing_title_locales:
        validation_errors.append(
            "titles are missing locales: " + ", ".join(missing_title_locales)
        )
    notes = _localized_strings(raw.get("notes"), list_values=True)
    reason = _clean_text(raw.get("reason"), 1000)

    # A render decision is executable only when every presentation reference is
    # grounded in the observed grid.  The host does not invent missing labels,
    # columns, rows, titles or target placement on the model's behalf.
    if target not in range(1, 6):
        validation_errors.append("target_detail_index must be an integer from 1 to 5")
    if not selected_columns:
        validation_errors.append("at least one fully localized source column is required")
    if not included_rows:
        validation_errors.append("at least one observed row must be selected")

    if decision == "render" and validation_errors:
        decision = "request_verification"
        reason = reason or "presentation is structurally incomplete or ungrounded"

    return {
        "decision": decision,
        "priority": priority,
        "target_detail_index": target,
        "display_locale": display_locale,
        "titles": titles,
        "columns": selected_columns,
        "included_rows": included_rows,
        "notes": notes,
        "reason": reason,
        "validation_errors": validation_errors,
    }


def extract_evidence_tables(vision: dict[str, Any]) -> list[EvidenceTable]:
    """Validate model-observed grids without assigning product-specific meaning."""

    raw_tables = vision.get("tables") if isinstance(vision, dict) else None
    source_images = vision.get("source_images") if isinstance(vision, dict) else None
    if not isinstance(raw_tables, list) or not isinstance(source_images, list):
        return []
    images = {
        item.get("index"): item
        for item in source_images
        if isinstance(item, dict) and isinstance(item.get("index"), int)
    }
    result: list[EvidenceTable] = []
    seen_signatures: set[str] = set()
    for raw_table in raw_tables[:_MAX_TABLES]:
        if not isinstance(raw_table, dict):
            continue
        source_index = raw_table.get("source_image_index")
        source = images.get(source_index)
        if (
            not isinstance(source_index, int)
            or not isinstance(source, dict)
            or source.get("inspection_complete") is not True
            or source.get("has_text") is not True
        ):
            continue

        cells: list[EvidenceCell] = []
        coordinates: set[tuple[int, int]] = set()
        raw_cells = raw_table.get("cells")
        if not isinstance(raw_cells, list):
            continue
        for raw_cell in raw_cells[:_MAX_CELLS]:
            if not isinstance(raw_cell, dict):
                continue
            row = raw_cell.get("row")
            column = raw_cell.get("column")
            text = _clean_text(raw_cell.get("text"))
            if (
                not isinstance(row, int)
                or isinstance(row, bool)
                or not 0 <= row < MAX_SOURCE_TABLE_ROWS
                or not isinstance(column, int)
                or isinstance(column, bool)
                or not 0 <= column < MAX_SOURCE_TABLE_COLUMNS
                or not text
                or (row, column) in coordinates
            ):
                continue
            try:
                confidence = float(raw_cell.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            row_span = raw_cell.get("row_span", 1)
            column_span = raw_cell.get("column_span", 1)
            row_span = row_span if isinstance(row_span, int) and 1 <= row_span <= 8 else 1
            column_span = (
                column_span
                if isinstance(column_span, int) and 1 <= column_span <= 8
                else 1
            )
            coordinates.add((row, column))
            cells.append(
                EvidenceCell(
                    row=row,
                    column=column,
                    text=text,
                    confidence=confidence,
                    row_span=row_span,
                    column_span=column_span,
                )
            )

        row_indexes = {cell.row for cell in cells}
        column_indexes = {cell.column for cell in cells}
        if len(cells) < 4 or len(row_indexes) < 2 or len(column_indexes) < 2:
            continue
        signature = _cell_signature(cells, source_index)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        table_id = f"table-{signature}"
        pointer = f"source-image:{source_index}#table:{table_id}"
        for cell in cells:
            cell.evidence_pointer = f"{pointer}/r{cell.row}c{cell.column}"
        result.append(
            EvidenceTable(
                table_id=table_id,
                source_image_index=source_index,
                cells=sorted(cells, key=lambda item: (item.row, item.column)),
                source_url=_clean_text(source.get("url"), 4000),
                evidence_pointer=pointer,
                presentation=_normalize_presentation(
                    raw_table.get("presentation"),
                    rows=row_indexes,
                    columns=column_indexes,
                ),
            )
        )
    return result


def select_render_table(tables: Iterable[EvidenceTable]) -> EvidenceTable | None:
    """Execute the model's render/priority decision without semantic ranking."""

    candidates = [
        table
        for table in tables
        if table.presentation.get("decision") == "render"
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda table: (
            -int(table.presentation.get("priority") or 0),
            table.table_id,
        ),
    )[0]


def presentation_view(table: EvidenceTable, locale: str | None = None) -> dict[str, Any]:
    """Materialize an already validated model presentation from exact cells."""

    presentation = table.presentation
    requested_locale = _clean_text(
        locale or presentation.get("display_locale"), 8
    ).casefold()
    if requested_locale not in _TARGET_LOCALES:
        requested_locale = "en"
    titles = presentation.get("titles")
    title = str(titles.get(requested_locale) or "") if isinstance(titles, dict) else ""
    columns = presentation.get("columns")
    included_rows = presentation.get("included_rows")
    if (
        presentation.get("decision") != "render"
        or not title
        or not isinstance(columns, list)
        or not isinstance(included_rows, list)
    ):
        raise ValueError("evidence table has no executable model presentation")

    by_coordinate = {(cell.row, cell.column): cell for cell in table.cells}
    selected_columns: list[int] = []
    headers: list[str] = []
    for item in columns:
        source_column = item.get("source_column") if isinstance(item, dict) else None
        labels = item.get("labels") if isinstance(item, dict) else None
        label = str(labels.get(requested_locale) or "") if isinstance(labels, dict) else ""
        if not isinstance(source_column, int) or not label:
            raise ValueError("evidence table presentation contains an invalid column")
        selected_columns.append(source_column)
        headers.append(label)

    rows: list[list[str]] = []
    cell_refs: list[list[str]] = []
    for row_index in included_rows:
        values: list[str] = []
        refs: list[str] = []
        for column_index in selected_columns:
            cell = by_coordinate.get((row_index, column_index))
            values.append(cell.text if cell is not None else "")
            refs.append(cell.evidence_pointer if cell is not None else "")
        if any(values):
            rows.append(values)
            cell_refs.append(refs)
    if not rows or any(not any(row[index] for row in rows) for index in range(len(headers))):
        raise ValueError("evidence table presentation contains an empty row or column")
    notes = presentation.get("notes")
    localized_notes = notes.get(requested_locale) if isinstance(notes, dict) else []
    return {
        "table_id": table.table_id,
        "source_ref": table.evidence_pointer,
        "locale": requested_locale,
        "title": title,
        "headers": headers,
        "rows": rows,
        "cell_refs": cell_refs,
        "notes": list(localized_notes) if isinstance(localized_notes, list) else [],
        "target_detail_index": int(presentation.get("target_detail_index") or 0),
    }
