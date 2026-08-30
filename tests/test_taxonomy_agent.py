from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from crossborder_agent.input_loader import load_json, load_product_facts
from crossborder_agent.taxonomy_agent import TaxonomyExplorer, TaxonomyReActAgent


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

    @classmethod
    def _actions(cls) -> list[tuple[str, dict[str, Any]]]:
        return [
            (
                "query_taxonomy",
                {
                    "requests": [
                        {
                            "collection": "categories",
                            "filters": [
                                {
                                    "field": "category_id",
                                    "op": "eq",
                                    "value": cls.category_id,
                                }
                            ],
                        }
                    ]
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
                "query_taxonomy",
                {
                    "requests": [
                        {
                            "collection": "schemas",
                            "filters": [
                                {
                                    "field": "schema_category_id",
                                    "op": "eq",
                                    "value": cls.schema_id,
                                }
                            ],
                        },
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
                        },
                    ]
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


if __name__ == "__main__":
    unittest.main()
