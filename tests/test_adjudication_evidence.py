from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from crossborder_agent.agent_tools import BoundedToolRegistry
from crossborder_agent.bounded_agent import BoundedDeliveryAgent
from crossborder_agent.input_loader import load_json, load_product_facts
from crossborder_agent.models import CreativePlan
from crossborder_agent.taxonomy import resolve_taxonomy


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data_for_Users"


class AdjudicationEvidenceTests(unittest.TestCase):
    def test_disagreement_adjudicator_receives_same_taxonomy_path_and_target_spec(self) -> None:
        facts = load_product_facts(
            DATA / "product_info/product_8409262509816.json"
        )
        taxonomy = resolve_taxonomy(
            facts,
            load_json(DATA / "clothing_categories.json"),
            load_json(DATA / "clothing_attributes.json"),
        )

        class Client:
            evaluation_models = ("evaluator-a", "evaluator-b")
            config = SimpleNamespace(
                review_model="adjudicator",
                review_fallback_model="adjudicator-fallback",
            )

            def __init__(self) -> None:
                self.adjudication_prompt = ""

            def chat_json(self, system, prompt, **kwargs):
                if "evidence adjudicator" in system:
                    self.adjudication_prompt = prompt
                    return {
                        "decisions": [
                            {
                                "defect_id": "A3:taxonomy:wrong-leaf",
                                "supported": True,
                                "reason": "category path conflicts with product evidence",
                            }
                        ]
                    }
                if kwargs.get("model") == "evaluator-a":
                    return {
                        "summary": "category mismatch",
                        "findings": [
                            {
                                "dimension": "A3",
                                "criterion": "wrong-leaf",
                                "severity": "major",
                                "target": "taxonomy",
                                "evidence": "the resolved leaf path conflicts with the adult product facts",
                                "expected": "an evidence-compatible leaf",
                            }
                        ],
                    }
                return {"summary": "no defect", "findings": []}

        client = Client()
        agent = BoundedDeliveryAgent(
            client, logging.getLogger("adjudication-evidence-test")
        )
        creative = CreativePlan("test", "main", ["detail"] * 5, "video")
        with tempfile.TemporaryDirectory() as temporary:
            evaluation = agent.evaluate_delivery(
                round_index=0,
                facts=facts,
                taxonomy=taxonomy,
                creative_plan=creative,
                agent_plan={},
                assets=[],
                localization_payloads={},
                localization_sources={},
                visual_set_review={},
                work_dir=Path(temporary),
                tools=BoundedToolRegistry(),
                expected_delivery_spec={"sentinel": "frozen-target"},
            )

        self.assertIsNotNone(evaluation)
        self.assertEqual(len(evaluation.issues), 1)
        self.assertIn(taxonomy.category.path, client.adjudication_prompt)
        self.assertIn(taxonomy.attribute_schema_category_id, client.adjudication_prompt)
        self.assertIn("frozen-target", client.adjudication_prompt)


if __name__ == "__main__":
    unittest.main()
