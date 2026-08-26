from __future__ import annotations

import json
import unittest
import logging
import subprocess
import sys
import time
import tempfile
from pathlib import Path

from crossborder_agent.compliance import (
    generated_copy_violations,
    normalize_source_image_observations,
)
from crossborder_agent.input_loader import (
    load_json,
    load_product_facts,
    parse_prompt_paths,
)
from crossborder_agent.localization import generate_copy_payload, render_description
from crossborder_agent.models import AssetResult
from crossborder_agent.planning import fallback_creative_plan
from crossborder_agent.pipeline import Pipeline
from crossborder_agent.taxonomy import resolve_taxonomy


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data_for_Users"


class PromptParsingTests(unittest.TestCase):
    def test_version_contract_is_consistent(self) -> None:
        manifest = json.loads((ROOT / "agent.json").read_text(encoding="utf-8"))
        completed = subprocess.run(
            [sys.executable, str(ROOT / "agent.py"), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), manifest["version"])

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


class ComplianceTests(unittest.TestCase):
    def test_generated_contact_and_price_are_rejected(self) -> None:
        violations = generated_copy_violations(
            "en", {"overview": "Contact us at seller@example.com — only US$ 9.99"}
        )
        self.assertTrue(any(item.startswith("regex:") for item in violations))

    def test_source_observations_are_bound_by_index_and_hard_risk_rejected(self) -> None:
        analysis = {
            "images": [
                {
                    "index": 1,
                    "role": "hero",
                    "has_text": False,
                    "has_logo": False,
                    "has_qr_code": True,
                    "safe_for_generation_reference": True,
                },
                {
                    "index": 0,
                    "role": "front",
                    "has_text": False,
                    "has_logo": False,
                    "safe_for_generation_reference": True,
                },
            ]
        }
        normalized = normalize_source_image_observations(
            analysis, ["https://example.test/clean.jpg", "https://example.test/qr.jpg"]
        )
        self.assertEqual(normalized[0]["url"], "https://example.test/clean.jpg")
        self.assertTrue(normalized[0]["safe_for_generation_reference"])
        self.assertFalse(normalized[1]["safe_for_generation_reference"])
        self.assertIn("has_qr_code", normalized[1]["risk_reasons"])

    def test_missing_source_observation_fails_closed_for_generation(self) -> None:
        normalized = normalize_source_image_observations(
            {"images": []}, ["https://example.test/unseen.jpg"]
        )
        self.assertFalse(normalized[0]["safe_for_generation_reference"])
        self.assertIn("inspection_incomplete", normalized[0]["risk_reasons"])

    def test_brand_contaminated_product_is_edit_reference_not_listing_fallback(self) -> None:
        facts = load_product_facts(
            DATA / "product_info/product_8688570444629.json"
        )
        product_url = facts.product_image_urls[0]
        chart_url = facts.description_image_urls[0]
        with tempfile.TemporaryDirectory(prefix="agent-selection-") as temporary:
            root = Path(temporary)
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=root / "out",
                logger=logging.getLogger("selection-test"),
                offline=True,
            )
            observations = normalize_source_image_observations(
                {
                    "images": [
                        {
                            "index": 0,
                            "role": "hero",
                            "product_coverage": "high",
                            "sharpness": "high",
                            "has_third_party_brand": True,
                            "has_logo": True,
                            "product_obscured": False,
                            "safe_for_generation_reference": False,
                        },
                        {
                            "index": 1,
                            "role": "size_chart",
                            "product_coverage": "low",
                            "sharpness": "high",
                            "has_text": True,
                        },
                    ]
                },
                [product_url, chart_url],
            )
            pipeline._source_image_observations = {
                item["url"]: item for item in observations
            }
            self.assertTrue(observations[0]["safe_for_generation_reference"])
            self.assertTrue(observations[0]["reference_requires_cleanup"])
            self.assertFalse(observations[0]["safe_for_listing_fallback"])
            self.assertEqual(
                pipeline._source_urls_for_use(
                    [product_url], use="reference", preferred_roles=("hero",)
                ),
                [product_url],
            )
            fallback = pipeline._source_urls_for_use(
                [chart_url, product_url],
                use="fallback",
                preferred_roles=("hero", "front"),
            )
            self.assertEqual(fallback[0], product_url)


class SemanticQaPolicyTests(unittest.TestCase):
    def test_incomplete_positive_image_review_preserves_generated_asset(self) -> None:
        facts = load_product_facts(
            DATA / "product_info/product_3887087154767.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temporary),
                logger=logging.getLogger("image-review-policy"),
                offline=True,
            )
            asset = AssetResult(
                name="main_image.jpeg",
                path=str(Path(temporary) / "not-needed.jpeg"),
                source_url="https://example.test/generated.jpeg",
                model="image-model",
                generated=True,
            )
            rejected = pipeline._apply_image_review(
                facts,
                [asset],
                {
                    "assets": [
                        {
                            "index": 0,
                            "usable": True,
                            "identity_consistent": True,
                            "construction_consistent": True,
                            "major_artifacts": False,
                        }
                    ]
                },
                Path(temporary),
            )
        self.assertFalse(rejected)
        self.assertTrue(asset.generated)

    def test_low_deadline_preserves_decoded_generated_video(self) -> None:
        facts = load_product_facts(
            DATA / "product_info/product_3887087154767.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temporary),
                logger=logging.getLogger("video-review-policy"),
                offline=True,
            )
            pipeline.client = object()
            pipeline.deadline = time.monotonic() + 60
            asset = AssetResult(
                name="product_video.mp4",
                path=str(Path(temporary) / "decoded.mp4"),
                source_url="https://example.test/generated.mp4",
                model="video-model",
                generated=True,
            )
            pipeline._review_generated_video(
                facts, [asset], Path(temporary)
            )
        self.assertTrue(asset.generated)
        self.assertIn("跳过生成视频语义质检", pipeline.warnings[-1])


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

    def test_sample_taxonomy_and_key_attributes_match_golden_set(self) -> None:
        golden = load_json(ROOT / "rules/sample_taxonomy_gold_v1.json")
        for offer_id, expected in golden["products"].items():
            facts = load_product_facts(
                DATA / f"product_info/product_{offer_id}.json"
            )
            result = resolve_taxonomy(facts, self.tree, self.attributes)
            self.assertEqual(result.category.category_id, expected["category_id"])
            actual = {
                (
                    item.source_name,
                    item.source_value,
                    item.attr_id,
                    item.value_id,
                )
                for item in result.attributes
            }
            for mapping in expected["key_mappings"]:
                self.assertIn(tuple(mapping), actual, (offer_id, mapping))
            self.assertEqual(
                sum(item.sales_attribute for item in result.attributes),
                expected["sales_mapping_count"],
                offer_id,
            )
            self.assertEqual(result.missing_required, expected["missing_required"])

    def test_rendered_copy_contains_every_required_identifier(self) -> None:
        facts = load_product_facts(DATA / "product_info/product_3887087154767.json")
        taxonomy = resolve_taxonomy(facts, self.tree, self.attributes)
        plan = fallback_creative_plan(facts, taxonomy)
        payload, source = generate_copy_payload("en", facts, taxonomy, plan, None)
        text = render_description("en", payload, facts, taxonomy)
        self.assertEqual(source, "deterministic-fallback")
        self.assertIn(facts.offer_id, text)
        self.assertIn(facts.platform, text)
        self.assertIn(facts.source_category_name, text)
        self.assertIn(taxonomy.category.category_id, text)
        self.assertIn(taxonomy.category.path, text)
        self.assertIn(facts.skus[-1].sku_id, text)
        self.assertIn(facts.attributes[0].evidence_pointer, text)
        self.assertIn("product_video.mp4", text)
        self.assertIn("Seller Guidance (Metric)", text)
        self.assertIn("40–47.5 kg", text)
        first_sku_row = next(
            line for line in text.splitlines() if facts.skus[0].sku_id in line
        )
        self.assertIn("88.2–104.7 lb", first_sku_row)
        overview = text.split("## Product overview", 1)[1].split(
            "## Verified highlights", 1
        )[0]
        self.assertGreaterEqual(len([part for part in overview.split("\n\n") if part.strip()]), 2)

    def test_all_sample_descriptions_preserve_machine_scoring_fields(self) -> None:
        for product_path in sorted((DATA / "product_info").glob("*.json")):
            facts = load_product_facts(product_path)
            taxonomy = resolve_taxonomy(facts, self.tree, self.attributes)
            plan = fallback_creative_plan(facts, taxonomy)
            for language in ("en", "ko", "pt"):
                payload, _ = generate_copy_payload(
                    language, facts, taxonomy, plan, None
                )
                text = render_description(language, payload, facts, taxonomy)
                self.assertLess(
                    len(text.encode("utf-8")),
                    1024 * 1024,
                    (facts.offer_id, language),
                )
                for value in (
                    facts.platform,
                    facts.offer_id,
                    facts.source_url,
                    taxonomy.category.category_id,
                    taxonomy.category.name,
                    taxonomy.category.path,
                ):
                    self.assertIn(value, text, (facts.offer_id, language, value))
                for item in taxonomy.attributes:
                    for value in (
                        item.attr_id,
                        item.name,
                        item.source_value,
                        item.platform_value,
                        item.source_evidence_pointer,
                    ):
                        if value:
                            self.assertIn(
                                value, text, (facts.offer_id, language, value)
                            )
                for sku in facts.skus:
                    self.assertIn(
                        sku.sku_id, text, (facts.offer_id, language, sku.sku_id)
                    )
                    if sku.spec_id:
                        self.assertIn(
                            sku.spec_id,
                            text,
                            (facts.offer_id, language, sku.spec_id),
                        )

    def test_storyboard_adapts_to_children_and_bottoms(self) -> None:
        children = load_product_facts(
            DATA / "product_info/product_8688570444629.json"
        )
        children_taxonomy = resolve_taxonomy(children, self.tree, self.attributes)
        children_plan = fallback_creative_plan(children, children_taxonomy)
        self.assertIn("product-only", children_plan.detail_prompts[4])
        self.assertNotIn("show one adult wearer", children_plan.detail_prompts[4])

        shorts = load_product_facts(DATA / "product_info/product_9493156931235.json")
        shorts_taxonomy = resolve_taxonomy(shorts, self.tree, self.attributes)
        shorts_plan = fallback_creative_plan(shorts, shorts_taxonomy)
        self.assertIn("waistband", shorts_plan.detail_prompts[1])
        self.assertIn("both legs", shorts_plan.video_prompt)


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
