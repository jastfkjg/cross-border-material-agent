from __future__ import annotations

import logging
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from crossborder_agent.agent_tools import BoundedToolRegistry, ToolExecution, ToolSpec
from crossborder_agent.bounded_agent import BoundedDeliveryAgent
from crossborder_agent.claims import build_claim_ledger
from crossborder_agent.input_loader import load_json, load_product_facts
from crossborder_agent.models import AgentAction, AgentEvaluation, AssetResult
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
        tiers = {
            item.target: item.execution_tier for item in result.repair_actions
        }
        self.assertEqual(
            tiers["product_description_en.md"], "critical_verification"
        )
        self.assertEqual(
            tiers["product_description_pt.md"], "critical_verification"
        )
        self.assertEqual(tiers["product_description_ko.md"], "discretionary")
        self.assertEqual(tiers["main_image.jpeg"], "discretionary")
        self.assertTrue(all(item.votes == 1 for item in result.repair_actions))
        self.assertIn("Artifact specification compliance", client.prompts[0])
        self.assertIn("Product fact consistency", client.prompts[0])

    def test_internal_acceptance_threshold_cannot_be_lowered_below_95(self) -> None:
        default = {"minimum_weighted_score": 95}

        result = BoundedDeliveryAgent._normalize_plan(
            {"minimum_weighted_score": 90}, default
        )

        self.assertEqual(result["minimum_weighted_score"], 95)

    def test_local_final_artifact_does_not_use_provenance_as_delivery_visual(self) -> None:
        facts = SimpleNamespace(
            product_image_urls=[],
            sku_image_urls=[],
            description_image_urls=[],
            size_chart_rows=[
                SimpleNamespace(
                    size_label="L",
                    bust_cm="98",
                    length_cm="95",
                    weight_kg="",
                    weight_lb="",
                )
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            final_path = work_dir / "detail_image_5.jpeg"
            final_path.write_bytes(b"final-local-pixels")
            asset = AssetResult(
                name=final_path.name,
                path=str(final_path),
                source_url="https://example.test/chinese-source-chart.jpeg",
                model="deterministic-size-chart",
                generated=False,
                description="locally rendered size chart",
            )
            agent = BoundedDeliveryAgent(
                None, logging.getLogger("local-final-evidence-test")
            )
            image_info = SimpleNamespace(
                format="JPEG", width=1200, height=1500, size_bytes=18
            )
            image_quality = SimpleNamespace(
                entropy=4.0, luminance_stddev=20.0, difference_hash=123
            )
            with mock.patch(
                "crossborder_agent.bounded_agent.inspect_image",
                return_value=image_info,
            ), mock.patch(
                "crossborder_agent.bounded_agent.inspect_image_quality",
                return_value=image_quality,
            ):
                manifest, image_urls, video_urls = agent._artifact_evidence(
                    facts, [asset], work_dir
                )

        self.assertEqual(image_urls, [])
        self.assertEqual(video_urls, [])
        self.assertEqual(manifest[0]["evidence_mode"], "local-final-inspection")
        self.assertEqual(manifest[0]["delivery_visual_url"], "")
        self.assertIsNone(manifest[0]["visual_input_index"])
        self.assertEqual(
            manifest[0]["provenance_source_url"],
            "https://example.test/chinese-source-chart.jpeg",
        )
        self.assertFalse(manifest[0]["deterministic_render"]["contains_cjk"])
        self.assertEqual(manifest[0]["deterministic_render"]["rows"][0]["bust"], "98 cm")

    def test_tool_catalog_and_execution_respect_target_preconditions(self) -> None:
        registry = BoundedToolRegistry()
        registry.add_spec(
            ToolSpec("repair", "repair target", ("ready", "blocked"), 1)
        )
        registry.bind("repair", lambda target, instruction: ToolExecution("completed", "ok"))
        registry.bind_precondition(
            "repair",
            lambda target: (
                (True, "ready") if target == "ready" else (False, "missing input")
            ),
        )

        self.assertEqual(registry.catalog()[0]["allowed_targets"], ["ready"])
        self.assertEqual(registry.execute("repair", "ready", "fix").status, "completed")
        blocked = registry.execute("repair", "blocked", "fix")
        self.assertEqual(blocked.status, "rejected")
        self.assertIn("missing input", blocked.detail)

    def test_qualifier_defers_consensus_action_when_tool_precondition_fails(self) -> None:
        registry = BoundedToolRegistry()
        registry.add_spec(
            ToolSpec(
                "regenerate_detail_image",
                "repair detail",
                ("detail_image_4.jpeg",),
                1,
            )
        )
        registry.bind_precondition(
            "regenerate_detail_image",
            lambda target: (False, "no trusted detail reference for slot 4"),
        )
        action = AgentAction(
            tool="regenerate_detail_image",
            target="detail_image_4.jpeg",
            instruction="repair it",
            dimension="A6",
            votes=2,
            execution_tier="consensus",
        )
        evaluation = AgentEvaluation(
            round_index=0,
            ready_for_delivery=False,
            weighted_score=80,
            dimension_scores={f"A{index}": 80 for index in range(1, 8)},
            repair_actions=[action],
            evaluator_models=["judge-a", "judge-b"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temporary) / "out",
                logger=logging.getLogger("tool-precondition-test"),
                offline=True,
            )
            pipeline.deadline = time.monotonic() + 1800
            eligible, deferred = pipeline._qualify_repair_actions(
                evaluation,
                facts=self.facts,
                taxonomy=self.taxonomy,
                state=SimpleNamespace(assets=[], visual_set_review={}),
                work_dir=Path(temporary),
                registry=registry,
            )

        self.assertEqual(eligible, [])
        self.assertEqual(deferred[0]["reason"], "tool-precondition-unavailable")
        self.assertIn("no trusted detail reference", deferred[0]["detail"])

    def test_consensus_action_records_both_models_and_skips_verification_tier(self) -> None:
        action_a = AgentAction(
            tool="revise_localized_copy",
            target="product_description_en.md",
            instruction="correct the fact",
            dimension="A5",
            priority=1,
        )
        action_b = AgentAction(
            tool="revise_localized_copy",
            target="product_description_en.md",
            instruction="remove the unsupported claim",
            dimension="A5",
            priority=2,
        )
        evaluations = {
            model: AgentEvaluation(
                round_index=0,
                ready_for_delivery=False,
                weighted_score=80,
                dimension_scores={f"A{index}": 80 for index in range(1, 8)},
                repair_actions=[action],
            )
            for model, action in (("judge-a", action_a), ("judge-b", action_b))
        }

        result = BoundedDeliveryAgent._aggregate_evaluations(
            evaluations, round_index=0, minimum_weighted_score=90
        )

        self.assertEqual(len(result.repair_actions), 1)
        action = result.repair_actions[0]
        self.assertEqual(action.execution_tier, "consensus")
        self.assertEqual(action.votes, 2)
        self.assertEqual(action.supporting_models, ["judge-a", "judge-b"])

    def test_single_model_critical_action_needs_independent_verification(self) -> None:
        class VerifyingAgent:
            def __init__(self) -> None:
                self.calls = 0

            def verify_repair_action(self, action, **kwargs):
                self.calls += 1
                self.assertion_target = action.target
                return {
                    "supported": True,
                    "status": "verified",
                    "verifier_model": "judge-b",
                    "corrected_scope": "Apply only the verified factual correction.",
                }

        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            (work_dir / "product_description_en.md").write_text(
                "# Product\n\n## Description\n\nGrounded product copy.\n\n"
                "## Features\n\n- Verified feature\n",
                encoding="utf-8",
            )
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=work_dir / "out",
                logger=logging.getLogger("critical-verification-test"),
                offline=True,
            )
            verifier = VerifyingAgent()
            pipeline.agent = verifier
            pipeline.deadline = time.monotonic() + 1800
            action = AgentAction(
                tool="revise_localized_copy",
                target="product_description_en.md",
                instruction="Correct a disputed product fact.",
                dimension="A5",
                priority=1,
                supporting_models=["judge-a"],
                votes=1,
                execution_tier="critical_verification",
            )
            evaluation = AgentEvaluation(
                round_index=0,
                ready_for_delivery=False,
                weighted_score=80,
                dimension_scores={f"A{index}": 80 for index in range(1, 8)},
                repair_actions=[action],
                evaluator_models=["judge-a", "judge-b"],
            )
            state = SimpleNamespace(assets=[], visual_set_review={})

            eligible, deferred = pipeline._qualify_repair_actions(
                evaluation,
                facts=self.facts,
                taxonomy=self.taxonomy,
                state=state,
                work_dir=work_dir,
            )

        self.assertEqual(verifier.calls, 1)
        self.assertFalse(deferred)
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0].execution_tier, "verified_critical")
        self.assertEqual(
            eligible[0].instruction,
            "Apply only the verified factual correction.",
        )

    def test_dependency_batches_group_copy_atomically_and_isolate_images(self) -> None:
        def action(tool: str, target: str) -> AgentAction:
            return AgentAction(
                tool=tool,
                target=target,
                instruction="repair",
                priority=1,
                votes=2,
                execution_tier="consensus",
            )

        batches = Pipeline._build_repair_batches(
            [
                action("regenerate_main_image", "main_image.jpeg"),
                action("regenerate_detail_image", "detail_image_1.jpeg"),
                action("regenerate_detail_image", "detail_image_2.jpeg"),
                action("revise_localized_copy", "product_description_en.md"),
                action("revise_localized_copy", "product_description_ko.md"),
                action("regenerate_video", "product_video.mp4"),
            ]
        )

        by_id = {item["batch_id"]: item for item in batches}
        self.assertEqual(len(batches), 5)
        self.assertTrue(by_id["localized_copy:direct"]["atomic"])
        self.assertEqual(len(by_id["localized_copy:direct"]["actions"]), 2)
        self.assertIn("detail:detail_image_1.jpeg", by_id)
        self.assertIn("detail:detail_image_2.jpeg", by_id)

    def test_unverified_single_model_critical_action_is_deferred(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            (work_dir / "product_description_en.md").write_text(
                "# Product\n\n## Description\n\nGrounded product copy.\n\n"
                "## Features\n\n- Verified feature\n",
                encoding="utf-8",
            )
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=work_dir / "out",
                logger=logging.getLogger("critical-rejection-test"),
                offline=True,
            )
            pipeline.agent = SimpleNamespace(
                verify_repair_action=mock.Mock(
                    return_value={
                        "supported": False,
                        "status": "rejected",
                        "verifier_model": "judge-b",
                    }
                )
            )
            pipeline.deadline = time.monotonic() + 1800
            action = AgentAction(
                tool="revise_localized_copy",
                target="product_description_en.md",
                instruction="Speculative correction.",
                dimension="A5",
                supporting_models=["judge-a"],
                votes=1,
                execution_tier="critical_verification",
            )
            evaluation = AgentEvaluation(
                round_index=0,
                ready_for_delivery=False,
                weighted_score=80,
                dimension_scores={f"A{index}": 80 for index in range(1, 8)},
                repair_actions=[action],
                evaluator_models=["judge-a", "judge-b"],
            )

            eligible, deferred = pipeline._qualify_repair_actions(
                evaluation,
                facts=self.facts,
                taxonomy=self.taxonomy,
                state=SimpleNamespace(assets=[], visual_set_review={}),
                work_dir=work_dir,
            )

        self.assertFalse(eligible)
        self.assertEqual(len(deferred), 1)
        self.assertEqual(
            deferred[0]["reason"], "single-model-critical-not-verified"
        )

    def test_only_one_successful_dependency_batch_is_applied_before_reevaluation(self) -> None:
        copy_actions = [
            AgentAction(
                tool="revise_localized_copy",
                target=f"product_description_{language}.md",
                instruction="repair locale",
                dimension="A5",
                priority=1,
                votes=2,
                execution_tier="consensus",
            )
            for language in ("en", "ko")
        ]
        detail_action = AgentAction(
            tool="regenerate_detail_image",
            target="detail_image_1.jpeg",
            instruction="repair detail",
            dimension="A6",
            priority=1,
            votes=2,
            execution_tier="consensus",
        )

        class Evaluator:
            def __init__(self) -> None:
                self.calls = 0

            def evaluate_delivery(self, **kwargs):
                actions = [*copy_actions, detail_action] if self.calls == 0 else []
                self.calls += 1
                return AgentEvaluation(
                    round_index=kwargs["round_index"],
                    ready_for_delivery=False,
                    weighted_score=80,
                    dimension_scores={f"A{index}": 80 for index in range(1, 8)},
                    repair_actions=actions,
                    evaluator_models=["judge-a", "judge-b"],
                )

        executed: list[str] = []
        registry = BoundedToolRegistry()
        registry.add_spec(
            ToolSpec(
                "revise_localized_copy",
                "copy",
                (
                    "product_description_en.md",
                    "product_description_ko.md",
                ),
                1,
            )
        )
        registry.add_spec(
            ToolSpec(
                "regenerate_detail_image",
                "detail",
                ("detail_image_1.jpeg",),
                1,
            )
        )
        registry.bind(
            "revise_localized_copy",
            lambda target, instruction: (
                executed.append(target) or ToolExecution("completed", "ok")
            ),
        )
        registry.bind(
            "regenerate_detail_image",
            lambda target, instruction: (
                executed.append(target) or ToolExecution("completed", "ok")
            ),
        )

        with tempfile.TemporaryDirectory() as temporary:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temporary) / "out",
                logger=logging.getLogger("one-batch-test"),
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
            captured: list[dict] = []

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

            with mock.patch.object(
                pipeline, "_capture_agent_snapshot", side_effect=capture
            ), mock.patch.object(
                pipeline,
                "_qualify_repair_actions",
                side_effect=lambda evaluation, **kwargs: (
                    list(evaluation.repair_actions),
                    [],
                ),
            ), mock.patch.object(
                pipeline, "_synchronize_repair_dependencies", return_value=True
            ), mock.patch.object(
                pipeline, "_repair_batch_consistent", return_value=(True, "ok")
            ), mock.patch.object(pipeline, "_restore_agent_snapshot"):
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
        self.assertEqual(
            executed,
            ["product_description_en.md", "product_description_ko.md"],
        )

    def test_partial_atomic_copy_batch_is_rolled_back(self) -> None:
        actions = [
            AgentAction(
                tool="revise_localized_copy",
                target=f"product_description_{language}.md",
                instruction="repair locale",
                dimension="A5",
                priority=1,
                votes=2,
                execution_tier="consensus",
            )
            for language in ("en", "ko")
        ]

        class Evaluator:
            def __init__(self) -> None:
                self.calls = 0

            def evaluate_delivery(self, **kwargs):
                selected = actions if self.calls == 0 else []
                self.calls += 1
                return AgentEvaluation(
                    round_index=kwargs["round_index"],
                    ready_for_delivery=False,
                    weighted_score=80,
                    dimension_scores={f"A{index}": 80 for index in range(1, 8)},
                    repair_actions=selected,
                    evaluator_models=["judge-a", "judge-b"],
                )

        registry = BoundedToolRegistry()
        registry.add_spec(
            ToolSpec(
                "revise_localized_copy",
                "copy",
                (
                    "product_description_en.md",
                    "product_description_ko.md",
                ),
                1,
            )
        )
        registry.bind(
            "revise_localized_copy",
            lambda target, instruction: ToolExecution(
                "completed" if target.endswith("en.md") else "failed",
                "result",
            ),
        )

        with tempfile.TemporaryDirectory() as temporary:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temporary) / "out",
                logger=logging.getLogger("atomic-rollback-test"),
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
            captured: list[dict] = []

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

            with mock.patch.object(
                pipeline, "_capture_agent_snapshot", side_effect=capture
            ), mock.patch.object(
                pipeline,
                "_qualify_repair_actions",
                side_effect=lambda evaluation, **kwargs: (
                    list(evaluation.repair_actions),
                    [],
                ),
            ), mock.patch.object(
                pipeline, "_restore_agent_snapshot"
            ) as restore:
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

        self.assertEqual(restore.call_count, 2)
        self.assertTrue(
            any(item.status == "rolled_back" for item in state.agent_actions)
        )

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
