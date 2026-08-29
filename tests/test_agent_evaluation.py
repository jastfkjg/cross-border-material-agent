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
from crossborder_agent.input_loader import load_json, load_product_facts
from crossborder_agent.models import AgentAction, AgentEvaluation
from crossborder_agent.pipeline import Pipeline
from crossborder_agent.planning import fallback_creative_plan
from crossborder_agent.taxonomy import resolve_taxonomy


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data_for_Users"


def finding(target: str = "product_description_en.md") -> dict:
    return {
        "dimension": "A5",
        "criterion": "unsupported-claim",
        "severity": "major",
        "target": target,
        "evidence": "The artifact makes a claim absent from the fact ledger.",
        "expected": "Remove the unsupported claim.",
    }


def evaluation(*, ready: bool = False, issues: list[dict] | None = None, round_index: int = 0) -> AgentEvaluation:
    return AgentEvaluation(
        round_index=round_index,
        ready_for_delivery=ready,
        weighted_score=100.0 if ready else 97.0,
        dimension_scores={f"A{index}": 100.0 for index in range(1, 8)},
        issues=issues or [],
        evaluator_models=["judge-a", "judge-b"],
        model_weighted_scores={"judge-a": 97.0, "judge-b": 97.0},
    )


class EvaluationRepairControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        categories = load_json(DATA / "clothing_categories.json")
        attributes = load_json(DATA / "clothing_attributes.json")
        cls.facts = load_product_facts(DATA / "product_info/product_6786311895552.json")
        cls.taxonomy = resolve_taxonomy(cls.facts, categories, attributes)
        cls.plan = fallback_creative_plan(cls.facts, cls.taxonomy)

    def registry(self) -> BoundedToolRegistry:
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
                "regenerate_detail_image",
                "repair detail",
                ("detail_image_1.jpeg", "detail_image_2.jpeg"),
                1,
            )
        )
        return registry

    def state(self) -> SimpleNamespace:
        return SimpleNamespace(
            assets=[],
            visual_set_review={},
            taxonomy=self.taxonomy,
            agent_evaluations=[],
            agent_actions=[],
            agent_snapshots=[],
        )

    @staticmethod
    def snapshot(result: AgentEvaluation) -> dict:
        return {
            "metadata": {
                "snapshot_id": f"snapshot_{result.round_index:02d}",
                "artifact_fingerprint": result.artifact_fingerprint,
                "weighted_score": result.weighted_score,
                "selected": False,
            }
        }

    def test_evaluators_only_report_findings_and_code_derives_score(self) -> None:
        class Client:
            evaluation_models = ("judge-a", "judge-b")
            config = SimpleNamespace(review_model="judge-a", review_fallback_model="judge-b")

            def __init__(self) -> None:
                self.systems: list[str] = []

            def chat_json(self, system, prompt, *, model, **kwargs):
                self.systems.append(system)
                return {
                    "summary": model,
                    "weighted_score": 10 if model == "judge-a" else 100,
                    "repair_actions": [{"tool": "unsafe"}],
                    "findings": [finding()],
                }

        client = Client()
        agent = BoundedDeliveryAgent(client, logging.getLogger("evidence-eval"))
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            agent, "_artifact_evidence", return_value=([], [], [])
        ), mock.patch.object(agent, "_copy_artifact_evidence", return_value=[]):
            result = agent.evaluate_delivery(
                round_index=0,
                facts=self.facts,
                taxonomy=self.taxonomy,
                creative_plan=self.plan,
                agent_plan={"minimum_weighted_score": 95},
                assets=[],
                localization_payloads={},
                localization_sources={},
                visual_set_review={},
                work_dir=Path(temporary),
                tools=self.registry(),
                artifact_fingerprint="artifact-a",
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.weighted_score, 97.0)
        self.assertEqual(result.dimension_scores["A5"], 70.0)
        self.assertFalse(result.ready_for_delivery)
        self.assertEqual(result.repair_actions, [])
        self.assertEqual(result.issues[0]["votes"], 2)
        self.assertEqual(result.issues[0]["adjudication"], "consensus")
        self.assertIn("not a scorer", client.systems[0])

    def test_disputed_finding_is_decided_by_separate_adjudicator(self) -> None:
        report_a = BoundedDeliveryAgent._parse_evaluator_report(
            {"summary": "a", "findings": [finding()]},
            round_index=0,
            evaluator_model="judge-a",
        )
        report_b = BoundedDeliveryAgent._parse_evaluator_report(
            {"summary": "b", "findings": []},
            round_index=0,
            evaluator_model="judge-b",
        )

        class Client:
            config = SimpleNamespace(review_model="adjudicator", review_fallback_model="fallback")

            def chat_json(self, system, prompt, **kwargs):
                return {
                    "decisions": [{
                        "defect_id": "A5:product_description_en.md:unsupported-claim",
                        "supported": True,
                        "reason": "Evidence supports the finding.",
                    }]
                }

        agent = BoundedDeliveryAgent(Client(), logging.getLogger("adjudicator"))
        result = agent._adjudicate_evaluations(
            {"judge-a": report_a, "judge-b": report_b},
            round_index=0,
            minimum_weighted_score=95,
            artifact_fingerprint="artifact-a",
        )

        self.assertTrue(result.disagreement)
        self.assertEqual(result.adjudication["status"], "completed")
        self.assertEqual(len(result.issues), 1)
        self.assertEqual(result.issues[0]["votes"], 1)

    def test_repair_planner_cannot_invent_defects(self) -> None:
        issue = {
            **finding(),
            "defect_id": "A5:product_description_en.md:unsupported-claim",
            "models": ["judge-a", "judge-b"],
            "votes": 2,
        }

        class Client:
            config = SimpleNamespace(review_model="planner", review_fallback_model="fallback")

            def chat_json(self, system, prompt, **kwargs):
                return {
                    "repair_actions": [
                        {
                            "defect_id": issue["defect_id"],
                            "tool": "revise_localized_copy",
                            "target": "product_description_en.md",
                            "instruction": "Remove only the unsupported sentence.",
                            "acceptance_criteria": "The sentence is absent.",
                            "priority": 1,
                        },
                        {
                            "defect_id": "invented-defect",
                            "tool": "revise_localized_copy",
                            "target": "product_description_ko.md",
                            "instruction": "Unrelated change.",
                        },
                    ]
                }

        agent = BoundedDeliveryAgent(Client(), logging.getLogger("planner"))
        actions = agent.plan_repairs(evaluation(issues=[issue]), tools=self.registry())

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].defect_id, issue["defect_id"])
        self.assertEqual(actions[0].execution_tier, "planned")

    def test_preflight_and_batches_use_independent_target_transactions(self) -> None:
        registry = self.registry()
        registry.bind_precondition(
            "regenerate_detail_image",
            lambda target: (False, "no trusted source reference"),
        )
        blocked = AgentAction(
            defect_id="A6:detail_image_1.jpeg:identity",
            tool="regenerate_detail_image",
            target="detail_image_1.jpeg",
            instruction="repair",
        )
        eligible, rejected = Pipeline._preflight_repair_actions([blocked], registry)
        self.assertEqual(eligible, [])
        self.assertEqual(rejected[0]["status"], "rejected_precondition")

        actions = [
            AgentAction(
                defect_id=f"A4:product_description_{language}.md:native-language",
                tool="revise_localized_copy",
                target=f"product_description_{language}.md",
                instruction="repair",
            )
            for language in ("en", "ko")
        ]
        batches = Pipeline._build_repair_batches(actions)
        self.assertEqual(len(batches), 2)
        self.assertTrue(all(item["atomic"] is False for item in batches))

    def test_ready_evaluation_stops_immediately(self) -> None:
        class Agent:
            def __init__(self) -> None:
                self.evaluations = 0

            def evaluate_delivery(self, **kwargs):
                self.evaluations += 1
                return evaluation(ready=True, round_index=kwargs["round_index"])

        with tempfile.TemporaryDirectory() as temporary:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temporary) / "out",
                logger=logging.getLogger("ready-stop"),
                offline=True,
            )
            pipeline.client = SimpleNamespace()
            pipeline.agent = Agent()
            pipeline.deadline = time.monotonic() + 1800
            state = self.state()
            with mock.patch.object(
                pipeline, "_capture_agent_snapshot",
                side_effect=lambda evaluation, **kwargs: self.snapshot(evaluation),
            ):
                pipeline._run_bounded_agent_loop(
                    self.registry(), facts=self.facts, taxonomy=self.taxonomy,
                    creative_plan=self.plan, agent_plan={"minimum_weighted_score": 95},
                    state=state, localization_payloads={}, localization_sources={},
                    work_dir=Path(temporary),
                )

        self.assertEqual(pipeline.agent.evaluations, 1)

    def test_no_change_replans_once_without_reevaluation(self) -> None:
        issue = {
            **finding(),
            "defect_id": "A5:product_description_en.md:unsupported-claim",
            "models": ["judge-a", "judge-b"],
            "votes": 2,
        }
        action = AgentAction(
            defect_id=issue["defect_id"], tool="revise_localized_copy",
            target="product_description_en.md", instruction="remove claim", dimension="A5",
        )

        class Agent:
            def __init__(self) -> None:
                self.evaluations = 0
                self.plans = 0

            def evaluate_delivery(self, **kwargs):
                self.evaluations += 1
                return evaluation(issues=[issue], round_index=kwargs["round_index"])

            def plan_repairs(self, *args, **kwargs):
                self.plans += 1
                return [action]

        registry = self.registry()
        registry.bind("revise_localized_copy", lambda target, instruction: ToolExecution("completed", "claimed success"))
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            (work_dir / action.target).write_text("unchanged", encoding="utf-8")
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=work_dir / "out",
                logger=logging.getLogger("no-change"),
                offline=True,
            )
            pipeline.client = SimpleNamespace()
            pipeline.agent = Agent()
            pipeline.deadline = time.monotonic() + 1800
            state = self.state()
            with mock.patch.object(
                pipeline, "_capture_agent_snapshot",
                side_effect=lambda evaluation, **kwargs: self.snapshot(evaluation),
            ):
                pipeline._run_bounded_agent_loop(
                    registry, facts=self.facts, taxonomy=self.taxonomy,
                    creative_plan=self.plan, agent_plan={"minimum_weighted_score": 95},
                    state=state, localization_payloads={}, localization_sources={}, work_dir=work_dir,
                )

        self.assertEqual(pipeline.agent.evaluations, 1)
        self.assertEqual(pipeline.agent.plans, 2)
        self.assertEqual([item.status for item in state.agent_actions], ["no_change"])

    def test_verified_change_is_reevaluated_once(self) -> None:
        issue = {
            **finding(),
            "defect_id": "A5:product_description_en.md:unsupported-claim",
            "models": ["judge-a", "judge-b"],
            "votes": 2,
        }
        action = AgentAction(
            defect_id=issue["defect_id"], tool="revise_localized_copy",
            target="product_description_en.md", instruction="remove claim", dimension="A5",
        )

        class Agent:
            def __init__(self) -> None:
                self.evaluations = 0

            def evaluate_delivery(self, **kwargs):
                self.evaluations += 1
                return evaluation(
                    ready=self.evaluations == 2,
                    issues=[] if self.evaluations == 2 else [issue],
                    round_index=kwargs["round_index"],
                )

            def plan_repairs(self, *args, **kwargs):
                return [action]

            def verify_repair_outcome(self, actions, **kwargs):
                return {"accepted": True, "status": "verified", "fixed_defect_ids": [actions[0].defect_id]}

        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            target = work_dir / action.target
            target.write_text("old", encoding="utf-8")
            registry = self.registry()

            def change_target(name: str, instruction: str) -> ToolExecution:
                target.write_text("new", encoding="utf-8")
                return ToolExecution("completed", "changed")

            registry.bind("revise_localized_copy", change_target)
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=work_dir / "out",
                logger=logging.getLogger("reevaluate"),
                offline=True,
            )
            pipeline.client = SimpleNamespace()
            pipeline.agent = Agent()
            pipeline.deadline = time.monotonic() + 1800
            state = self.state()
            with mock.patch.object(
                pipeline, "_capture_agent_snapshot",
                side_effect=lambda evaluation, **kwargs: self.snapshot(evaluation),
            ), mock.patch.object(
                pipeline, "_synchronize_repair_dependencies", return_value=True
            ), mock.patch.object(
                pipeline, "_repair_batch_consistent", return_value=(True, "ok")
            ):
                pipeline._run_bounded_agent_loop(
                    registry, facts=self.facts, taxonomy=self.taxonomy,
                    creative_plan=self.plan, agent_plan={"minimum_weighted_score": 95},
                    state=state, localization_payloads={}, localization_sources={}, work_dir=work_dir,
                )
            final_text = target.read_text(encoding="utf-8")

        self.assertEqual(pipeline.agent.evaluations, 2)
        self.assertEqual(final_text, "new")
        self.assertTrue(state.agent_actions[0].changed)

    def test_verifier_rejection_rolls_back_without_reevaluation(self) -> None:
        issue = {
            **finding(),
            "defect_id": "A5:product_description_en.md:unsupported-claim",
            "models": ["judge-a", "judge-b"],
            "votes": 2,
        }
        action = AgentAction(
            defect_id=issue["defect_id"],
            tool="revise_localized_copy",
            target="product_description_en.md",
            instruction="remove claim",
            dimension="A5",
        )

        class Agent:
            def __init__(self) -> None:
                self.evaluations = 0

            def evaluate_delivery(self, **kwargs):
                self.evaluations += 1
                return evaluation(issues=[issue], round_index=kwargs["round_index"])

            def plan_repairs(self, *args, **kwargs):
                return [action]

            def verify_repair_outcome(self, *args, **kwargs):
                return {
                    "accepted": False,
                    "status": "postcondition-failed",
                    "evidence": "the defect remains",
                }

        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            target = work_dir / action.target
            target.write_text("old", encoding="utf-8")
            registry = self.registry()

            def change_target(name: str, instruction: str) -> ToolExecution:
                target.write_text("rejected", encoding="utf-8")
                return ToolExecution("completed", "changed")

            registry.bind("revise_localized_copy", change_target)
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=work_dir / "out",
                logger=logging.getLogger("rollback"),
                offline=True,
            )
            pipeline.client = SimpleNamespace()
            pipeline.agent = Agent()
            pipeline.deadline = time.monotonic() + 1800
            state = self.state()
            with mock.patch.object(
                pipeline,
                "_capture_agent_snapshot",
                side_effect=lambda evaluation, **kwargs: self.snapshot(evaluation),
            ), mock.patch.object(
                pipeline, "_synchronize_repair_dependencies", return_value=True
            ), mock.patch.object(
                pipeline, "_repair_batch_consistent", return_value=(True, "ok")
            ):
                pipeline._run_bounded_agent_loop(
                    registry,
                    facts=self.facts,
                    taxonomy=self.taxonomy,
                    creative_plan=self.plan,
                    agent_plan={"minimum_weighted_score": 95},
                    state=state,
                    localization_payloads={},
                    localization_sources={},
                    work_dir=work_dir,
                )
            final_text = target.read_text(encoding="utf-8")

        self.assertEqual(pipeline.agent.evaluations, 1)
        self.assertEqual(final_text, "old")
        self.assertEqual(state.agent_actions[-1].status, "rolled_back")


if __name__ == "__main__":
    unittest.main()
