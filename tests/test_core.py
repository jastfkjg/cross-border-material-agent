from __future__ import annotations

import unittest
from pathlib import Path

from crossborder_agent.input_loader import (
    load_json,
    load_product_facts,
    parse_prompt_paths,
)
from crossborder_agent.localization import generate_copy_payload, render_description
from crossborder_agent.planning import fallback_creative_plan
from crossborder_agent.taxonomy import resolve_taxonomy


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data_for_Users"


class PromptParsingTests(unittest.TestCase):
    def test_official_style_prompt(self) -> None:
        paths = parse_prompt_paths(
            "读取 `/home/user/ws/input/` 目录中的文件，并将结果输出到 `/home/user/ws/output/`。"
        )
        self.assertTrue(str(paths.input_dir).endswith("/home/user/ws/input"))
        self.assertTrue(str(paths.output_dir).endswith("/home/user/ws/output"))

    def test_unquoted_paths(self) -> None:
        paths = parse_prompt_paths(
            "input directory: /data/source output directory: /workspace/result"
        )
        self.assertTrue(str(paths.input_dir).endswith("/data/source"))
        self.assertTrue(str(paths.output_dir).endswith("/workspace/result"))

    def test_output_filename_is_treated_as_parent_directory(self) -> None:
        paths = parse_prompt_paths(
            "读取 /data/dataset 中的数据，将结果输出到 /workspace/output/result.json"
        )
        self.assertTrue(str(paths.input_dir).endswith("/data/dataset"))
        self.assertTrue(str(paths.output_dir).endswith("/workspace/output"))


class FactAndTaxonomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = load_json(DATA / "clothing_categories.json")
        cls.attributes = load_json(DATA / "clothing_attributes.json")

    def test_fact_ledger_preserves_skus_and_converts_jin(self) -> None:
        facts = load_product_facts(DATA / "product_info/product_3887087154767.json")
        self.assertEqual(facts.offer_id, "3887087154767")
        self.assertEqual(len(facts.skus), 24)
        self.assertEqual(facts.size_conversions[0].kilograms, "40–47.5 kg")
        self.assertEqual(facts.size_conversions[0].pounds, "88.2–104.7 lb")
        self.assertGreaterEqual(len(facts.product_image_urls), 5)

    def test_sample_categories_resolve_to_leaf_nodes(self) -> None:
        expected = {
            "3887087154767": "29073",
            "5681480836479": "28951",
            "5758364264251": "29069",
            "5977010166484": "28976",
            "6786311895552": "39107",
            "6837006744133": "30408",
            "8409262509816": "39107",
            "8688570444629": "30843",
            "8822221153828": "39153",
            "9451226053560": "30471",
            "9493156931235": "30341",
        }
        leaf_ids = {
            str(node["catId"])
            for node in _walk_objects(self.tree)
            if node.get("isLeaf") is True and "catId" in node
        }
        for product_path in sorted((DATA / "product_info").glob("*.json")):
            facts = load_product_facts(product_path)
            result = resolve_taxonomy(facts, self.tree, self.attributes)
            self.assertEqual(result.category.category_id, expected[facts.offer_id])
            self.assertIn(result.category.category_id, leaf_ids)

    def test_rendered_copy_contains_every_required_identifier(self) -> None:
        facts = load_product_facts(DATA / "product_info/product_3887087154767.json")
        taxonomy = resolve_taxonomy(facts, self.tree, self.attributes)
        plan = fallback_creative_plan(facts, taxonomy)
        payload, source = generate_copy_payload("en", facts, taxonomy, plan, None)
        text = render_description("en", payload, facts, taxonomy)
        self.assertEqual(source, "deterministic-fallback")
        self.assertIn(facts.offer_id, text)
        self.assertIn(taxonomy.category.category_id, text)
        self.assertIn(facts.skus[-1].sku_id, text)
        self.assertIn("product_video.mp4", text)


def _walk_objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_objects(child)


if __name__ == "__main__":
    unittest.main()
