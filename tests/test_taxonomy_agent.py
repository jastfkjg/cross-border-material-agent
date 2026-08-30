from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from crossborder_agent.api import ApiError
from crossborder_agent.input_loader import load_json, load_product_facts
from crossborder_agent.models import CategoryChoice
from crossborder_agent.pipeline import Pipeline
from crossborder_agent.taxonomy import resolve_taxonomy
from crossborder_agent.taxonomy_agent import (
    TaxonomyAgentError,
    TaxonomyExplorer,
    TaxonomyReActAgent,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data_for_Users"


class TaxonomyAgentSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = load_json(DATA / "clothing_categories.json")
        cls.attributes = load_json(DATA / "clothing_attributes.json")
        cls.facts = load_product_facts(
            next(iter(sorted((DATA / "product_info").glob("*.json"))))
        )

        explorer = TaxonomyExplorer(cls.tree, cls.attributes)
        observation = explorer.execute(
            "query_taxonomy",
            {
                "requests": [
                    {
                        "collection": "categories",
                        "filters": [
                            {"field": "is_leaf", "op": "eq", "value": True},
                            {
                                "field": "available_schema_ids",
                                "op": "exists",
                                "value": True,
                            },
                        ],
                        "limit": 1,
                    }
                ]
            },
        )
        cls.category_record = observation["result"]["results"][0]["items"][0]
        cls.category_id = str(cls.category_record["category_id"])
        cls.schema_id = str(cls.category_record["available_schema_ids"][0])
        attribute_observation = explorer.execute(
            "query_taxonomy",
            {
                "requests": [
                    {
                        "collection": "attributes",
                        "filters": [
                            {
                                "field": "schema_category_id",
                                "op": "eq",
                                "value": cls.schema_id,
                            }
                        ],
                        "limit": 60,
                    }
                ]
            },
        )
        cls.attribute_records = attribute_observation["result"]["results"][0][
            "items"
        ]
        explorer.close()

    @classmethod
    def _actions(cls) -> list[tuple[str, dict[str, Any]]]:
        return [
            (
                "search",
                {
                    "pattern": f'"category_id":"{cls.category_id}"',
                    "paths": ["taxonomy/categories.jsonl"],
                },
            ),
            (
                "finish_category",
                {
                    "selected_category_id": cls.category_id,
                    "confidence": 0.8,
                    "evidence": "observed leaf category",
                },
            ),
            (
                "search",
                {
                    "pattern": f'"schema_category_id":"{cls.schema_id}"',
                    "paths": [
                        "taxonomy/schemas.jsonl",
                        "taxonomy/attributes.jsonl",
                    ],
                    "limit": 200,
                },
            ),
            (
                "finish_attributes",
                {
                    "selected_attribute_schema_category_id": cls.schema_id,
                    "mappings": [],
                    "unresolved_mappings": [
                        {
                            "scope": str(item["scope"]),
                            "platform_attr_id": str(item["attr_id"]),
                            "reason": (
                                "The smoke client observed this schema attribute but "
                                "has no synthetic source-to-platform mapping evidence."
                            ),
                        }
                        for item in cls.attribute_records
                    ],
                },
            ),
        ]

    def test_generic_tools_query_records_and_reject_ungrounded_finish(self) -> None:
        explorer = TaxonomyExplorer(self.tree, self.attributes)
        premature, rejection = explorer.finish_category(
            {
                "selected_category_id": self.category_id,
                "confidence": 0.8,
                "evidence": "not queried yet",
            }
        )
        self.assertIsNone(premature)
        self.assertFalse(rejection["ok"])

        observation = explorer.execute(
            "query_taxonomy",
            {
                "requests": [
                    {
                        "collection": "categories",
                        "filters": [
                            {
                                "field": "category_id",
                                "op": "eq",
                                "value": self.category_id,
                            }
                        ],
                    },
                    {
                        "collection": "schemas",
                        "filters": [
                            {
                                "field": "schema_category_id",
                                "op": "eq",
                                "value": self.schema_id,
                            }
                        ],
                    },
                ]
            },
        )
        self.assertTrue(observation["ok"])
        category_result, schema_result = observation["result"]["results"]
        self.assertTrue(category_result["items"][0]["is_leaf"])
        self.assertEqual(
            schema_result["items"][0]["schema_category_id"], self.schema_id
        )
        explorer.close()

    def test_json_protocol_agent_completes_with_valid_output(self) -> None:
        actions = self._actions()

        class JsonClient:
            def chat_json(self, _system: str, _prompt: str) -> dict[str, Any]:
                action, arguments = actions.pop(0)
                return {"action": action, "arguments": arguments}

        result = TaxonomyReActAgent(
            JsonClient(), self.tree, self.attributes, max_turns=4
        ).run(self.facts)

        self.assertEqual(result.category.category_id, self.category_id)
        self.assertTrue(result.category.path)
        self.assertEqual(result.attribute_schema_category_id, self.schema_id)

    def test_native_protocol_agent_completes_with_tool_observations(self) -> None:
        actions = self._actions()
        observed_roles: list[str] = []

        class NativeClient:
            def chat_tool_step(self, _system, messages, _tools):
                observed_roles.extend(str(item.get("role") or "") for item in messages)
                action, arguments = actions.pop(0)
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"smoke-{len(observed_roles)}",
                            "type": "function",
                            "function": {
                                "name": action,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }

        result = TaxonomyReActAgent(
            NativeClient(), self.tree, self.attributes, max_turns=4
        ).run(self.facts)

        self.assertEqual(result.category.category_id, self.category_id)
        self.assertEqual(result.attribute_schema_category_id, self.schema_id)
        self.assertIn("tool", observed_roles)

    def test_finish_attributes_returns_per_row_source_ref_diagnostics(self) -> None:
        explorer = TaxonomyExplorer(self.tree, self.attributes)
        explorer.install_product_evidence(self.facts)
        explorer.execute(
            "query_taxonomy",
            {
                "requests": [
                    {
                        "collection": "schemas",
                        "filters": [
                            {
                                "field": "schema_category_id",
                                "op": "eq",
                                "value": self.schema_id,
                            }
                        ],
                    },
                    {
                        "collection": "attributes",
                        "filters": [
                            {
                                "field": "schema_category_id",
                                "op": "eq",
                                "value": self.schema_id,
                            }
                        ],
                        "limit": 60,
                    },
                ]
            },
        )
        attribute = self.attribute_records[0]
        category_node = explorer.nodes[self.category_id]
        category = CategoryChoice(
            category_id=category_node.category_id,
            name=category_node.name,
            path=category_node.path,
            confidence=0.8,
            method="test",
        )
        result, rejection = explorer.finish_attributes(
            self.facts,
            category,
            {
                "selected_attribute_schema_category_id": self.schema_id,
                "mappings": [
                    {
                        "scope": attribute["scope"],
                        "platform_attr_id": attribute["attr_id"],
                        "platform_value_id": "",
                        "source_ref": "canonical-claim/not-real",
                    }
                ],
                "unresolved_mappings": [],
            },
        )

        self.assertIsNone(result)
        self.assertEqual(rejection["code"], "attribute_mapping_rejected")
        details = rejection["details"]
        self.assertEqual(details["accepted_mapping_count"], 0)
        self.assertEqual(details["rejected_mapping_count"], 1)
        reason_codes = {
            item["code"]
            for item in details["rejected_mappings"][0]["reasons"]
        }
        self.assertIn("unknown_source_ref", reason_codes)
        self.assertIn("correction", rejection)
        explorer.close()

    def test_stable_source_ref_installs_a_grounded_mapping(self) -> None:
        explorer = TaxonomyExplorer(self.tree, self.attributes)
        evidence = explorer.install_product_evidence(self.facts)
        explorer.execute(
            "query_taxonomy",
            {
                "requests": [
                    {
                        "collection": "schemas",
                        "filters": [
                            {
                                "field": "schema_category_id",
                                "op": "eq",
                                "value": self.schema_id,
                            }
                        ],
                    },
                    {
                        "collection": "attributes",
                        "filters": [
                            {
                                "field": "schema_category_id",
                                "op": "eq",
                                "value": self.schema_id,
                            }
                        ],
                        "limit": 60,
                    },
                ]
            },
        )
        sources = evidence["source_evidence"]
        attribute = next(
            item
            for item in self.attribute_records
            if any(
                (
                    item["scope"] == "sales"
                    and source["source_kind"] == "sku"
                )
                or (
                    item["scope"] == "product"
                    and source["source_kind"] in {"product", "canonical"}
                )
                for source in sources
            )
        )
        source = next(
            source
            for source in sources
            if (
                attribute["scope"] == "sales"
                and source["source_kind"] == "sku"
            )
            or (
                attribute["scope"] == "product"
                and source["source_kind"] in {"product", "canonical"}
            )
        )
        value_id = ""
        if attribute["value_count"]:
            value_page = explorer.execute(
                "read_taxonomy", {"refs": [attribute["ref"]], "limit": 60}
            )
            value_id = value_page["result"]["items"][0]["values"]["items"][0][
                "value_id"
            ]
        category_node = explorer.nodes[self.category_id]
        category = CategoryChoice(
            category_id=category_node.category_id,
            name=category_node.name,
            path=category_node.path,
            confidence=0.8,
            method="test",
        )
        unresolved = [
            {
                "scope": item["scope"],
                "platform_attr_id": item["attr_id"],
                "reason": "No synthetic semantic mapping evidence is supplied for this test row.",
            }
            for item in self.attribute_records
            if (item["scope"], item["attr_id"])
            != (attribute["scope"], attribute["attr_id"])
        ]

        result, observation = explorer.finish_attributes(
            self.facts,
            category,
            {
                "selected_attribute_schema_category_id": self.schema_id,
                "mappings": [
                    {
                        "scope": attribute["scope"],
                        "platform_attr_id": attribute["attr_id"],
                        "platform_value_id": value_id,
                        "source_ref": source["ref"],
                    }
                ],
                "unresolved_mappings": unresolved,
            },
        )

        self.assertTrue(observation["ok"])
        self.assertIsNotNone(result)
        self.assertEqual(len(result.attributes), 1)
        self.assertEqual(result.attributes[0].source_name, source["name"])
        self.assertEqual(result.attributes[0].source_value, source["value"])
        explorer.close()

    def test_identical_rejected_finish_stops_without_exhausting_budget(self) -> None:
        invalid_finish = {
            "selected_attribute_schema_category_id": self.schema_id,
            "mappings": [],
            "unresolved_mappings": "not-an-array",
        }
        actions = [
            *self._actions()[:3],
            ("finish_attributes", invalid_finish),
            ("finish_attributes", invalid_finish),
        ]

        class NativeClient:
            def __init__(self) -> None:
                self.turns = 0

            def chat_tool_step(self, _system, _messages, _tools):
                self.turns += 1
                action, arguments = actions.pop(0)
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"turn-{self.turns}",
                            "type": "function",
                            "function": {
                                "name": action,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }

        client = NativeClient()
        with self.assertRaises(TaxonomyAgentError) as captured:
            TaxonomyReActAgent(
                client, self.tree, self.attributes, max_turns=4
            ).run(self.facts)

        self.assertEqual(client.turns, 5)
        self.assertIn("identical rejected finish", str(captured.exception))

    def test_attribute_failure_preserves_committed_model_category(self) -> None:
        fallback = resolve_taxonomy(self.facts, self.tree, self.attributes)
        explorer = TaxonomyExplorer(self.tree, self.attributes)
        target = next(
            node
            for node in explorer.nodes.values()
            if node.is_leaf and node.category_id != fallback.category.category_id
        )
        selected = CategoryChoice(
            category_id=target.category_id,
            name=target.name,
            path=target.path,
            confidence=0.9,
            method="model-test",
        )
        explorer.close()

        class PartialAgent:
            def __init__(self, *args, **kwargs) -> None:
                pass

            @staticmethod
            def resolve_category(_facts):
                return selected

            @staticmethod
            def resolve_attributes(_facts, _category):
                raise ApiError(
                    "synthetic attribute failure",
                    retryable=True,
                    category="taxonomy_agent",
                )

            @staticmethod
            def close() -> None:
                return None

        with tempfile.TemporaryDirectory() as temporary:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temporary) / "output",
                logger=logging.getLogger("taxonomy-transaction-test"),
                offline=True,
            )
            pipeline.client = object()
            with mock.patch(
                "crossborder_agent.pipeline_parts.taxonomy.TaxonomyReActAgent",
                PartialAgent,
            ):
                result = pipeline._adjudicate_taxonomy(
                    self.facts, fallback, self.tree, self.attributes
                )

        self.assertEqual(result.category.category_id, selected.category_id)
        self.assertEqual(result.category.path, selected.path)


if __name__ == "__main__":
    unittest.main()
