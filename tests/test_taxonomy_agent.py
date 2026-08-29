from __future__ import annotations

import json
import unittest
from pathlib import Path

from crossborder_agent.input_loader import load_json, load_product_facts
from crossborder_agent.taxonomy_agent import TaxonomyExplorer, TaxonomyReActAgent


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data_for_Users"


class TaxonomyExplorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = load_json(DATA / "clothing_categories.json")
        cls.attributes = load_json(DATA / "clothing_attributes.json")

    def test_literal_search_keeps_all_matching_leaves_available_without_ranking(self) -> None:
        explorer = TaxonomyExplorer(self.tree, self.attributes)
        observation = explorer.execute(
            "search_categories", {"query": "工装", "leaf_only": True, "limit": 60}
        )

        self.assertTrue(observation["ok"])
        ids = [item["category_id"] for item in observation["result"]["items"]]
        self.assertIn("30408", ids)
        self.assertGreater(len(ids), 1)
        self.assertNotEqual(ids[0], "30408")
        self.assertTrue(all("score" not in item for item in observation["result"]["items"]))

    def test_leaf_without_schema_exposes_structural_ancestor_schema(self) -> None:
        explorer = TaxonomyExplorer(self.tree, self.attributes)
        observation = explorer.execute("get_attribute_schema", {"category_id": "30408"})

        self.assertTrue(observation["ok"])
        self.assertIsNone(observation["result"]["schema"])
        ancestor_ids = {
            item["category_id"]
            for item in observation["result"]["ancestor_schemas"]
        }
        self.assertIn("30382", ancestor_ids)


class TaxonomyReActAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = load_json(DATA / "clothing_categories.json")
        cls.attributes = load_json(DATA / "clothing_attributes.json")

    def test_model_controls_multi_turn_search_and_can_correct_invalid_finish(self) -> None:
        facts = load_product_facts(
            DATA / "product_info/product_6837006744133.json"
        )

        class ScriptedClient:
            def __init__(self) -> None:
                self.calls = 0
                self.prompts: list[str] = []

            def chat_json(self, _system: str, prompt: str) -> dict[str, object]:
                self.prompts.append(prompt)
                actions: list[dict[str, object]] = [
                    {
                        # Deliberately premature: the tool validator must reject this
                        # and let the model continue instead of silently accepting it.
                        "action": "finish",
                        "arguments": {
                            "selected_category_id": "30408",
                            "selected_attribute_schema_category_id": "30382",
                            "mappings": [],
                        },
                    },
                    {
                        "action": "search_categories",
                        "arguments": {
                            "query": "工装",
                            "leaf_only": True,
                            "limit": 60,
                        },
                    },
                    {
                        "action": "get_attribute_schema",
                        "arguments": {"category_id": "30408"},
                    },
                    {
                        "action": "get_attribute_schema",
                        "arguments": {"category_id": "30382"},
                    },
                    {
                        "action": "get_attribute_definition",
                        "arguments": {
                            "schema_category_id": "30382",
                            "attr_id": "100157",
                            "limit": 60,
                        },
                    },
                    {
                        "action": "finish",
                        "arguments": {
                            "selected_category_id": "30408",
                            "selected_attribute_schema_category_id": "30382",
                            "confidence": 0.91,
                            "evidence": "男士夹克标题明确包含工装，平台路径属于男士外套。",
                            "mappings": [
                                {
                                    "scope": "product",
                                    "platform_attr_id": "100157",
                                    "platform_value_id": "1000011",
                                    "source_kind": "product",
                                    "source_name": "主面料成分",
                                    "source_value": "聚酯纤维（涤纶）",
                                }
                            ],
                        },
                    },
                ]
                result = actions[self.calls]
                self.calls += 1
                return result

        client = ScriptedClient()
        result = TaxonomyReActAgent(
            client, self.tree, self.attributes, max_turns=8
        ).run(facts)

        self.assertEqual(result.category.category_id, "30408")
        self.assertEqual(result.category.method, "model-react-exploration")
        self.assertEqual(result.attribute_schema_category_id, "30382")
        self.assertEqual(len(result.attributes), 1)
        self.assertEqual(result.attributes[0].attr_id, "100157")
        self.assertEqual(result.attributes[0].value_id, "1000011")
        self.assertEqual(client.calls, 6)
        self.assertIn("was not observed", client.prompts[1])
        self.assertIn("30408", client.prompts[2])
        self.assertNotIn(facts.offer_id, json.dumps(client.prompts, ensure_ascii=False))

    def test_native_tool_protocol_returns_observations_as_tool_messages(self) -> None:
        facts = load_product_facts(
            DATA / "product_info/product_6837006744133.json"
        )

        class NativeClient:
            def __init__(self) -> None:
                self.index = 0
                self.message_snapshots: list[list[dict[str, object]]] = []

            def chat_tool_step(self, _system, messages, tools):
                self.message_snapshots.append([dict(item) for item in messages])
                actions = [
                    ("search_categories", {"query": "工装", "leaf_only": True}),
                    ("get_attribute_schema", {"category_id": "30408"}),
                    ("get_attribute_schema", {"category_id": "30382"}),
                    (
                        "get_attribute_definition",
                        {
                            "schema_category_id": "30382",
                            "attr_id": "100157",
                            "limit": 60,
                        },
                    ),
                    (
                        "finish",
                        {
                            "selected_category_id": "30408",
                            "selected_attribute_schema_category_id": "30382",
                            "confidence": 0.9,
                            "evidence": "source title and category path",
                            "mappings": [
                                {
                                    "scope": "product",
                                    "platform_attr_id": "100157",
                                    "platform_value_id": "1000011",
                                    "source_kind": "product",
                                    "source_name": "主面料成分",
                                    "source_value": "聚酯纤维（涤纶）",
                                }
                            ],
                        },
                    ),
                ]
                name, arguments = actions[self.index]
                self.index += 1
                self.assert_tool_catalog(tools)
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{self.index}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }

            @staticmethod
            def assert_tool_catalog(tools):
                names = {item["function"]["name"] for item in tools}
                assert {"search_categories", "list_children", "finish"} <= names

        client = NativeClient()
        result = TaxonomyReActAgent(
            client, self.tree, self.attributes, max_turns=8
        ).run(facts)

        self.assertEqual(result.category.category_id, "30408")
        self.assertEqual(result.category.method, "model-react-exploration")
        self.assertEqual(client.index, 5)
        second_turn_roles = [item["role"] for item in client.message_snapshots[1]]
        self.assertEqual(second_turn_roles[-2:], ["assistant", "tool"])
        first_observation = client.message_snapshots[1][-1]
        self.assertEqual(first_observation["name"], "search_categories")
        self.assertIn("30408", str(first_observation["content"]))


if __name__ == "__main__":
    unittest.main()
