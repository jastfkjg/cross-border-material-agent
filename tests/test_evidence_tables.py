from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from crossborder_agent.media import MediaError, create_evidence_table_image, inspect_image
from crossborder_agent.models import AgentEvaluation
from crossborder_agent.pipeline_parts.review import review_acceptance_decision
from crossborder_agent.table_evidence import (
    extract_evidence_tables,
    presentation_view,
    select_render_table,
)


def _vision(*, presentation: dict[str, object]) -> dict[str, object]:
    return {
        "source_images": [
            {
                "index": 0,
                "url": "https://example.test/source-table.png",
                "inspection_complete": True,
                "has_text": True,
                "role": "data_table",
            }
        ],
        "tables": [
            {
                "source_image_index": 0,
                "cells": [
                    {"row": 0, "column": 0, "text": "Profile"},
                    {"row": 0, "column": 1, "text": "Orbit"},
                    {"row": 0, "column": 2, "text": "Flux"},
                    {"row": 1, "column": 0, "text": "A-17"},
                    {"row": 1, "column": 1, "text": "41.2"},
                    {"row": 1, "column": 2, "text": "ZX"},
                ],
                "presentation": presentation,
            }
        ],
    }


def _valid_presentation() -> dict[str, object]:
    return {
        "decision": "render",
        "priority": 83,
        "target_detail_index": 3,
        "display_locale": "en",
        "titles": {"en": "Source profile", "ko": "원본 표", "pt": "Tabela da fonte"},
        "columns": [
            {
                "source_column": 0,
                "labels": {"en": "Profile", "ko": "프로필", "pt": "Perfil"},
            },
            {
                "source_column": 2,
                "labels": {"en": "Flux", "ko": "플럭스", "pt": "Fluxo"},
            },
        ],
        "included_rows": [1],
        "notes": {"en": ["Exact source cells."], "ko": [], "pt": []},
        "reason": "These source columns are useful for the listing.",
    }


class EvidenceTableTests(unittest.TestCase):
    def test_arbitrary_columns_are_model_selected_and_exactly_grounded(self) -> None:
        tables = extract_evidence_tables(_vision(presentation=_valid_presentation()))
        self.assertEqual(len(tables), 1)
        selected = select_render_table(tables)
        self.assertIsNotNone(selected)
        view = presentation_view(selected)
        self.assertEqual(view["target_detail_index"], 3)
        self.assertEqual(view["headers"], ["Profile", "Flux"])
        self.assertEqual(view["rows"], [["A-17", "ZX"]])
        self.assertNotIn("Orbit", view["headers"])

    def test_nonexistent_column_rejects_the_entire_render_request(self) -> None:
        presentation = _valid_presentation()
        presentation["columns"] = [
            *presentation["columns"],
            {
                "source_column": 9,
                "labels": {"en": "Ghost", "ko": "고스트", "pt": "Fantasma"},
            },
        ]
        table = extract_evidence_tables(_vision(presentation=presentation))[0]
        self.assertEqual(table.presentation["decision"], "request_verification")
        self.assertIn(
            "source_column 9 does not exist in observed cells",
            table.presentation["validation_errors"],
        )
        self.assertIsNone(select_render_table([table]))

    def test_missing_localized_label_returns_actionable_feedback(self) -> None:
        presentation = _valid_presentation()
        presentation["columns"] = [
            {
                "source_column": 0,
                "labels": {"en": "Profile", "ko": "프로필"},
            }
        ]
        table = extract_evidence_tables(_vision(presentation=presentation))[0]
        self.assertEqual(table.presentation["decision"], "request_verification")
        self.assertIn(
            "source_column 0 is missing labels for: pt",
            table.presentation["validation_errors"],
        )

    def test_render_capacity_is_rejected_before_media_generation(self) -> None:
        vision = _vision(presentation=_valid_presentation())
        table_payload = vision["tables"][0]
        table_payload["cells"] = [
            {"row": row, "column": column, "text": f"R{row}C{column}"}
            for row in range(26)
            for column in range(2)
        ]
        table_payload["presentation"]["columns"] = [
            {
                "source_column": column,
                "labels": {
                    "en": f"Column {column}",
                    "ko": f"열 {column}",
                    "pt": f"Coluna {column}",
                },
            }
            for column in range(2)
        ]
        table_payload["presentation"]["included_rows"] = list(range(1, 26))

        table = extract_evidence_tables(vision)[0]
        self.assertEqual(table.presentation["decision"], "request_verification")
        self.assertIn(
            "included_rows exceeds the single-image rendering capacity of 24 entries",
            table.presentation["validation_errors"],
        )

    def test_generic_renderer_accepts_the_model_presentation(self) -> None:
        presentation = _valid_presentation()
        presentation["display_locale"] = "ko"
        vision = _vision(presentation=presentation)
        vision["tables"][0]["cells"][3]["text"] = "来源值"
        vision["tables"][0]["cells"][5]["text"] = "한국"
        table = extract_evidence_tables(vision)[0]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "table.jpeg"
            try:
                create_evidence_table_image(table, output)
            except MediaError as exc:
                if "Pillow" in str(exc):
                    self.skipTest("Pillow is not installed in the host test environment")
                raise
            info = inspect_image(output)
        self.assertEqual((info.width, info.height), (1200, 1500))


class UnifiedReviewAcceptanceTests(unittest.TestCase):
    def test_not_ready_is_advisory_and_never_blocks_submission(self) -> None:
        evaluation = AgentEvaluation(
            round_index=0,
            ready_for_delivery=False,
            weighted_score=99.0,
            artifact_fingerprint="current",
            summary="More evidence is required.",
        )
        decision = review_acceptance_decision(evaluation, "current")
        self.assertTrue(decision["accepted"])
        self.assertFalse(decision["advisory_ready"])
        self.assertEqual(decision["summary"], "More evidence is required.")

    def test_finish_gate_does_not_reimplement_issue_severity_policy(self) -> None:
        evaluation = AgentEvaluation(
            round_index=0,
            ready_for_delivery=True,
            weighted_score=96.0,
            artifact_fingerprint="current",
            issues=[{"dimension": "A1", "severity": "blocker"}],
        )
        decision = review_acceptance_decision(evaluation, "current")
        self.assertTrue(decision["accepted"])

        stale = replace(evaluation, artifact_fingerprint="old")
        stale_decision = review_acceptance_decision(stale, "current")
        self.assertTrue(stale_decision["accepted"])
        self.assertEqual(stale_decision["review_status"], "stale")

        missing_decision = review_acceptance_decision(None, "current")
        self.assertTrue(missing_decision["accepted"])
        self.assertEqual(missing_decision["review_status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
