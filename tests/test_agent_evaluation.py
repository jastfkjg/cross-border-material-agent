from __future__ import annotations

import logging
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from crossborder_agent.agent_tools import BoundedToolRegistry, ToolSpec
from crossborder_agent.bounded_agent import BoundedDeliveryAgent
from crossborder_agent.claims import build_claim_ledger
from crossborder_agent.input_loader import load_json, load_product_facts
from crossborder_agent.models import AgentAction, AgentEvaluation
from crossborder_agent.pipeline import Pipeline
from crossborder_agent.planning import fallback_creative_plan
from crossborder_agent.taxonomy import resolve_taxonomy


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data_for_Users"


def evaluation_payload(score: float, actions: list[dict]) -> dict:
    return {
        "ready_for_delivery": False,
        "weighted_score": score,
        "dimension_scores": {f"A{index}": score for index in range(1, 8)},
        "summary": f"score {score}",
        "issues": [
            {
                "dimension": "A5",
                "severity": "minor",
                "target": actions[0]["target"] if actions else "delivery",
                "evidence": "model evidence",
                "expected": "correct it",
            }
        ],
        "repair_actions": actions,
    }


class MultiModelEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.categories = load_json(DATA / "clothing_categories.json")
        cls.attributes = load_json(DATA / "clothing_attributes.json")
        cls.facts = load_product_facts(
            DATA / "product_info/product_6786311895552.json"
        )
        cls.taxonomy = resolve_taxonomy(cls.facts, cls.categories, cls.attributes)
        cls.plan = fallback_creative_plan(cls.facts, cls.taxonomy)

    def test_two_models_are_averaged_and_all_repair_targets_are_merged(self) -> None:
        actions_by_model = {
            "judge-a": [
                {
                    "tool": "revise_localized_copy",
                    "target": "product_description_en.md",
                    "instruction": "fix English",
                    "reason": "A5",
                    "dimension": "A5",
                    "priority": 1,
                },
                {
                    "tool": "revise_localized_copy",
                    "target": "product_description_ko.md",
                    "instruction": "fix Korean",
                    "reason": "A4",
                    "dimension": "A4",
                    "priority": 2,
                },
            ],
            "judge-b": [
                {
                    "tool": "revise_localized_copy",
                    "target": "product_description_pt.md",
                    "instruction": "fix Portuguese",
                    "reason": "A5",
                    "dimension": "A5",
                    "priority": 1,
                },
                {
                    "tool": "regenerate_main_image",
                    "target": "main_image.jpeg",
                    "instruction": "fix hero",
                    "reason": "A6",
                    "dimension": "A6",
                    "priority": 3,
                },
            ],
        }

        class Client:
            evaluation_models = ("judge-a", "judge-b")
            config = SimpleNamespace(
                review_model="judge-a", review_fallback_model="judge-b"
            )

            def __init__(self) -> None:
                self.prompts: list[str] = []

            def chat_json(self, system, prompt, *, model, **kwargs):
                self.prompts.append(prompt)
                score = 80 if model == "judge-a" else 100
                return evaluation_payload(score, actions_by_model[model])

        registry = BoundedToolRegistry()
        registry.add_spec(
            ToolSpec(
                "revise_localized_copy",
                "repair copy",
                (
                    "product_description_en.md",
                    "product_description_ko.md",
                    "product_description_pt.md",
                ),
                1,
            )
        )
        registry.add_spec(
            ToolSpec(
                "regenerate_main_image", "repair hero", ("main_image.jpeg",), 1
            )
        )
        client = Client()
        agent = BoundedDeliveryAgent(client, logging.getLogger("multi-review-test"))
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            agent, "_artifact_evidence", return_value=([], [], [])
        ), mock.patch.object(agent, "_copy_artifact_evidence", return_value=[]):
            result = agent.evaluate_delivery(
                round_index=0,
                facts=self.facts,
                taxonomy=self.taxonomy,
                creative_plan=self.plan,
                agent_plan={"minimum_weighted_score": 90},
                assets=[],
                localization_payloads={},
                localization_sources={},
                visual_set_review={},
                work_dir=Path(temporary),
                tools=registry,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.evaluator_models, ["judge-a", "judge-b"])
        self.assertEqual(result.weighted_score, 90)
        self.assertTrue(all(value == 90 for value in result.dimension_scores.values()))
        self.assertEqual(len(result.repair_actions), 4)
        self.assertIn("Artifact specification compliance", client.prompts[0])
        self.assertIn("Product fact consistency", client.prompts[0])

    def test_generic_reconciliation_votes_by_attribute_index(self) -> None:
        class Client:
            evaluation_models = ("judge-a", "judge-b")
            config = SimpleNamespace(
                review_model="judge-a", review_fallback_model="judge-b"
            )

            @staticmethod
            def chat_json(*args, model, **kwargs):
                return {
                    "seller_title_decision": (
                        "publish" if model == "judge-a" else "machine_only"
                    ),
                    "attribute_decisions": [
                        {
                            "attribute_index": 0,
                            "decision": "reject",
                            "canonical_value": "directly observed alternative",
                            "reason": "trusted pixels contradict the structured appearance value",
                            "visual_evidence": ["multiple clear source views"],
                        }
                    ],
                    "canonical_visual_claims": [
                        {
                            "concept": "observed construction",
                            "value": "directly observed alternative",
                            "confidence": 0.95,
                            "evidence": ["multiple clear source views"],
                        }
                    ],
                    "conflicts": [],
                }

        facts = load_product_facts(
            DATA / "product_info/product_6786311895552.json"
        )
        agent = BoundedDeliveryAgent(Client(), logging.getLogger("reconcile-test"))
        ledger = agent.reconcile_facts(
            facts,
            {
                "product_type": "generic apparel",
                "visible_design_features": ["directly observed alternative"],
                "source_images": [
                    {"inspection_complete": True, "role": "front"},
                    {"inspection_complete": True, "role": "detail"},
                ],
            },
        )
        facts.reconciled_fact_ledger = ledger
        claims = build_claim_ledger(facts, self.taxonomy, {})

        self.assertEqual(ledger["attribute_decisions"][0]["decision"], "reject")
        self.assertEqual(ledger["seller_title_decision"], "machine_only")
        rejected = [
            item
            for item in claims
            if item.source_type == "seller_attribute"
            and item.source_name == facts.attributes[0].name
            and item.value == facts.attributes[0].value
        ]
        self.assertEqual(len(rejected), 1)
        self.assertNotIn("buyer_copy", rejected[0].allowed_surfaces)

    def test_copy_repair_safety_does_not_require_style_score_gain(self) -> None:
        class Reviewer:
            config = SimpleNamespace(review_model="judge")

            @staticmethod
            def chat_json(*args, **kwargs):
                return {
                    "selected_index": 0,
                    "candidates": [
                        {"index": 0, "score": 100, "facts_supported": False},
                        {
                            "index": 1,
                            "score": 60,
                            "facts_supported": True,
                            "complete": True,
                            "native_and_natural": True,
                            "has_source_script_contamination": False,
                        },
                    ],
                }

        with tempfile.TemporaryDirectory() as temporary:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temporary),
                logger=logging.getLogger("copy-safety-test"),
                offline=True,
            )
            pipeline.client = Reviewer()
            self.assertTrue(
                pipeline._copy_revision_is_safe(
                    "en", self.facts, {"title": "old"}, {"title": "corrected"}
                )
            )

    def test_no_change_repairs_do_not_stop_three_evaluation_rounds(self) -> None:
        scores = [70.0, 90.0, 80.0]

        class Evaluator:
            def __init__(self) -> None:
                self.calls = 0

            def evaluate_delivery(self, **kwargs):
                score = scores[self.calls]
                self.calls += 1
                return AgentEvaluation(
                    round_index=kwargs["round_index"],
                    ready_for_delivery=True,
                    weighted_score=score,
                    dimension_scores={f"A{index}": score for index in range(1, 8)},
                    repair_actions=[],
                    evaluator_models=["a", "b"],
                    model_weighted_scores={"a": score, "b": score},
                )

        with tempfile.TemporaryDirectory() as temporary:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temporary) / "out",
                logger=logging.getLogger("three-round-test"),
                offline=True,
            )
            pipeline.client = SimpleNamespace()
            pipeline.agent = Evaluator()
            pipeline.deadline = time.monotonic() + 1800
            state = SimpleNamespace(
                assets=[],
                visual_set_review={},
                agent_evaluations=[],
                agent_actions=[],
                agent_snapshots=[],
            )
            captured = []

            def capture(*, evaluation, **kwargs):
                snapshot = {
                    "metadata": {
                        "snapshot_id": f"snapshot_{len(captured):02d}",
                        "weighted_score": evaluation.weighted_score,
                        "dimension_scores": evaluation.dimension_scores,
                        "after_repair_rounds": evaluation.round_index,
                    }
                }
                captured.append(snapshot)
                return snapshot

            registry = BoundedToolRegistry()
            with mock.patch.object(
                pipeline, "_capture_agent_snapshot", side_effect=capture
            ), mock.patch.object(pipeline, "_restore_agent_snapshot") as restore:
                pipeline._run_bounded_agent_loop(
                    registry,
                    facts=self.facts,
                    taxonomy=self.taxonomy,
                    creative_plan=self.plan,
                    agent_plan={"minimum_weighted_score": 90},
                    state=state,
                    localization_payloads={},
                    localization_sources={},
                    work_dir=Path(temporary),
                )

        self.assertEqual(pipeline.agent.calls, 3)
        self.assertEqual(len(captured), 3)
        selected = restore.call_args.args[0]
        self.assertEqual(selected["metadata"]["weighted_score"], 90)


if __name__ == "__main__":
    unittest.main()
