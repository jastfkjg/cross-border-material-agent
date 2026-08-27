from __future__ import annotations

import logging
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from crossborder_agent.api import ApiError
from crossborder_agent.compliance import (
    generated_copy_violations,
    normalize_source_image_observations,
)
from crossborder_agent.input_loader import (
    load_json,
    load_product_facts,
    parse_prompt_paths,
)
from crossborder_agent.localization import (
    _fallback_payload,
    _localized_concept_is_mentioned,
    _payload_validation_error,
    generate_copy_payload,
    render_description,
)
from crossborder_agent.models import AssetResult
from crossborder_agent.planning import create_creative_plan, fallback_creative_plan
from crossborder_agent.pipeline import (
    Pipeline,
    PipelineError,
    _merge_source_vision_batches,
)
from crossborder_agent.qa import _description_language_surfaces
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


class ComplianceTests(unittest.TestCase):
    def test_generated_contact_and_price_are_rejected(self) -> None:
        violations = generated_copy_violations(
            "en", {"overview": "Contact us at seller@example.com — only US$ 9.99"}
        )
        self.assertTrue(any(item.startswith("regex:") for item in violations))

    def test_contact_rule_does_not_match_ordinary_neckline_copy(self) -> None:
        self.assertEqual(
            generated_copy_violations(
                "en", {"overview": "A V-neckline creates a clean, open shape."}
            ),
            [],
        )
        self.assertTrue(
            generated_copy_violations(
                "en", {"overview": "For help, use LINE: seller_support"}
            )
        )

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
        self.assertTrue(observations[0]["safe_for_generation_reference"])
        self.assertTrue(observations[0]["reference_requires_cleanup"])
        self.assertFalse(observations[0]["safe_for_listing_fallback"])
        with tempfile.TemporaryDirectory(prefix="agent-selection-") as temporary:
            root = Path(temporary)
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=root / "out",
                logger=logging.getLogger("selection-test"),
                offline=True,
            )
            pipeline._source_image_observations = {
                item["url"]: item for item in observations
            }
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


class VisualSelectionTests(unittest.TestCase):
    def test_hero_wearer_requires_trusted_adult_source_support(self) -> None:
        class WearerSelectorClient:
            @staticmethod
            def select_best_generated_image(*args, **kwargs):
                return {
                    "selected_index": 0,
                    "candidates": [
                        {
                            "index": 0,
                            "usable": True,
                            "identity_consistent": True,
                            "construction_consistent": True,
                            "correct_color": True,
                            "single_product": True,
                            "product_complete": True,
                            "clean_neutral_background": True,
                            "has_person": True,
                            "has_unrelated_props": False,
                            "anatomy_natural": True,
                            "unwanted_text": False,
                            "unwanted_brand_or_logo": False,
                            "major_artifacts": False,
                            "product_coverage": "high",
                            "score": 90,
                        }
                    ],
                }

        facts = load_product_facts(DATA / "product_info/product_5681480836479.json")
        source_url = "https://example.test/adult-wearer.jpg"
        candidate_url = "https://example.test/candidate.jpg"
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temp_dir),
                logger=logging.getLogger("hero-wearer-selection-test"),
                offline=True,
            )
            pipeline.client = WearerSelectorClient()
            with self.assertRaises(Exception) as caught:
                pipeline._select_main_candidate(
                    facts, [source_url], [candidate_url]
                )
            self.assertIn("unsupported_wearer", caught.exception.feedback)

            pipeline._source_image_observations[source_url] = {
                "has_person": True,
                "safe_for_generation_reference": True,
            }
            selected = pipeline._select_main_candidate(
                facts, [source_url], [candidate_url]
            )
        self.assertEqual(selected, candidate_url)

    def test_six_image_review_uses_platform_allowed_remote_urls(self) -> None:
        facts = load_product_facts(
            DATA / "product_info/product_9451226053560.json"
        )

        class Reviewer:
            config = SimpleNamespace(review_model="test-reviewer")

            def __init__(self):
                self.generated_urls = []

            def review_generated_images(
                self, facts_json, source_urls, generated_urls, expected_assets
            ):
                self.generated_urls = list(generated_urls)
                return {
                    "assets": [
                        {"index": index, "usable": True}
                        for index in range(6)
                    ],
                    "set_usable": False,
                    "usable_count": 6,
                    "distinct_commercial_roles": 4,
                    "coherent": True,
                    "near_duplicate_pairs": [[1, 2]],
                    "missing_roles": ["construction close-up"],
                    "repair_targets": [2],
                    "summary": "two detail slots repeat the same role",
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temp_dir),
                logger=logging.getLogger("set-review-test"),
                offline=True,
            )
            reviewer = Reviewer()
            pipeline.client = reviewer
            assets = [
                AssetResult(
                    name="main_image.jpeg",
                    path="main_image.jpeg",
                    source_url="https://example.test/main.jpeg",
                    generated=True,
                    description="hero",
                )
            ] + [
                AssetResult(
                    name=f"detail_image_{index}.jpeg",
                    path=f"detail_image_{index}.jpeg",
                    source_url=f"https://example.test/detail-{index}.jpeg",
                    generated=True,
                    description=f"detail role {index}",
                )
                for index in range(1, 6)
            ]
            result = pipeline._review_visual_set(facts, assets)

        self.assertEqual(result["repair_targets"], [2])
        self.assertTrue(all(url.startswith("https://") for url in reviewer.generated_urls))
        self.assertTrue(any("语义重复" in item for item in pipeline.warnings))

    def test_non_offline_pipeline_requires_complete_model_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(
                    PipelineError,
                    "DASHSCOPE_API_KEY.*DASHSCOPE_BASE_URL.*OPENAI_BASE_URL",
                ):
                    Pipeline(
                        input_dir=DATA,
                        output_dir=Path(temp_dir),
                        logger=logging.getLogger("configuration-test"),
                    )

    def test_detail_fallback_plan_balances_distinct_views_and_detail_crops(self) -> None:
        facts = load_product_facts(
            DATA / "product_info/product_5758364264251.json"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temp_dir),
                logger=logging.getLogger("fallback-plan-test"),
                offline=True,
            )
            main_reference = facts.product_image_urls[0]
            planned = [
                pipeline._detail_fallback_plan(
                    facts, index=index, main_reference_url=main_reference
                )
                for index in range(1, 6)
            ]

        first_three_urls = [urls[0] for urls, _ in planned[:3]]
        self.assertEqual(len(set(first_three_urls)), 3)
        self.assertNotIn(main_reference, first_three_urls)
        self.assertEqual(planned[3][0][0], main_reference)
        self.assertEqual(planned[3][1], "upper")
        self.assertEqual(planned[4][0][0], main_reference)
        self.assertEqual(planned[4][1], "lower")

    def test_detail_selector_soft_scores_ambiguous_structure_for_closeup(self) -> None:
        class SelectorClient:
            @staticmethod
            def select_best_detail_image(*args, **kwargs):
                return {
                    "selected_index": 0,
                    "candidates": [
                        {
                            "index": 0,
                            "usable": True,
                            "identity_consistent": True,
                            "construction_consistent": True,
                            "color_consistent": True,
                            "pattern_consistent": True,
                            "slot_match": True,
                            "critical_structure_unambiguous": False,
                            "anatomy_natural": True,
                            "unwanted_text": False,
                            "unwanted_brand_or_logo": False,
                            "prohibited_visual": False,
                            "major_artifacts": False,
                            "product_coverage": "high",
                        }
                    ],
                }

        facts = load_product_facts(
            DATA / "product_info/product_5681480836479.json"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temp_dir),
                logger=logging.getLogger("visual-selection-test"),
                offline=True,
            )
            pipeline.client = SelectorClient()
            selected = pipeline._select_detail_candidate(
                2,
                facts,
                ["https://example.test/source.jpg"],
                ["https://example.test/candidate.jpg"],
                "construction close-up",
            )
        self.assertEqual(selected, "https://example.test/candidate.jpg")

    def test_detail_selector_keeps_candidate_when_judge_is_unavailable(self) -> None:
        class FailingSelectorClient:
            @staticmethod
            def select_best_detail_image(*args, **kwargs):
                raise ApiError(
                    "429 after retries", retryable=True, category="rate_limit"
                )

        facts = load_product_facts(
            DATA / "product_info/product_5681480836479.json"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temp_dir),
                logger=logging.getLogger("judge-unavailable-test"),
                offline=True,
            )
            pipeline.client = FailingSelectorClient()
            selected = pipeline._select_detail_candidate(
                2,
                facts,
                ["https://example.test/source.jpg"],
                [
                    "https://example.test/candidate-a.jpg",
                    "https://example.test/candidate-b.jpg",
                ],
                "construction close-up",
            )
        self.assertEqual(selected, "https://example.test/candidate-a.jpg")

    def test_all_slot_mismatches_trigger_targeted_regeneration(self) -> None:
        class SoftIssueSelectorClient:
            @staticmethod
            def select_best_detail_image(*args, **kwargs):
                return {
                    "selected_index": -1,
                    "candidates": [
                        {
                            "index": 0,
                            "usable": False,
                            "identity_consistent": True,
                            "construction_consistent": True,
                            "color_consistent": True,
                            "pattern_consistent": True,
                            "slot_match": False,
                            "critical_structure_unambiguous": True,
                            "anatomy_natural": True,
                            "unwanted_text": False,
                            "unwanted_brand_or_logo": False,
                            "prohibited_visual": False,
                            "major_artifacts": False,
                            "product_coverage": "low",
                            "score": 72,
                        }
                    ],
                }

        facts = load_product_facts(
            DATA / "product_info/product_5681480836479.json"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temp_dir),
                logger=logging.getLogger("soft-quality-test"),
                offline=True,
            )
            pipeline.client = SoftIssueSelectorClient()
            with self.assertRaisesRegex(Exception, "候选均未通过语义质检"):
                pipeline._select_detail_candidate(
                    2,
                    facts,
                    ["https://example.test/source.jpg"],
                    ["https://example.test/candidate.jpg"],
                    "construction close-up",
                )

    def test_unexpected_collage_is_a_hard_detail_defect(self) -> None:
        class CollageSelectorClient:
            @staticmethod
            def select_best_detail_image(*args, **kwargs):
                return {
                    "selected_index": 0,
                    "candidates": [{
                        "index": 0,
                        "usable": True,
                        "identity_consistent": True,
                        "construction_consistent": True,
                        "color_consistent": True,
                        "pattern_consistent": True,
                        "slot_match": True,
                        "critical_structure_unambiguous": True,
                        "anatomy_natural": True,
                        "single_composition": False,
                        "unexpected_collage": True,
                        "unwanted_text": False,
                        "unwanted_brand_or_logo": False,
                        "prohibited_visual": False,
                        "major_artifacts": False,
                        "product_coverage": "high",
                        "score": 96,
                    }],
                }

        facts = load_product_facts(DATA / "product_info/product_9493156931235.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temp_dir),
                logger=logging.getLogger("collage-selection-test"),
                offline=True,
            )
            pipeline.client = CollageSelectorClient()
            with self.assertRaisesRegex(Exception, "候选均未通过语义质检"):
                pipeline._select_detail_candidate(
                    2,
                    facts,
                    ["https://example.test/source.jpg"],
                    ["https://example.test/candidate.jpg"],
                    "waistband close-up",
                )

    def test_repair_keeps_incumbent_without_six_point_improvement(self) -> None:
        class ComparisonSelectorClient:
            @staticmethod
            def select_best_detail_image(*args, **kwargs):
                return {
                    "selected_index": 1,
                    "candidates": [
                        {
                            "index": 0,
                            "usable": True,
                            "identity_consistent": True,
                            "construction_consistent": True,
                            "color_consistent": True,
                            "pattern_consistent": True,
                            "slot_match": True,
                            "critical_structure_unambiguous": True,
                            "anatomy_natural": True,
                            "unwanted_text": False,
                            "unwanted_brand_or_logo": False,
                            "prohibited_visual": False,
                            "major_artifacts": False,
                            "product_coverage": "high",
                            "score": 88,
                        },
                        {
                            "index": 1,
                            "usable": True,
                            "identity_consistent": True,
                            "construction_consistent": True,
                            "color_consistent": True,
                            "pattern_consistent": True,
                            "slot_match": True,
                            "critical_structure_unambiguous": True,
                            "anatomy_natural": True,
                            "unwanted_text": False,
                            "unwanted_brand_or_logo": False,
                            "prohibited_visual": False,
                            "major_artifacts": False,
                            "product_coverage": "high",
                            "score": 91,
                        },
                    ],
                }

        facts = load_product_facts(
            DATA / "product_info/product_5681480836479.json"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temp_dir),
                logger=logging.getLogger("monotonic-repair-test"),
                offline=True,
            )
            pipeline.client = ComparisonSelectorClient()
            selected = pipeline._select_detail_candidate(
                2,
                facts,
                ["https://example.test/source.jpg"],
                [
                    "https://example.test/incumbent.jpg",
                    "https://example.test/revision.jpg",
                ],
                "construction close-up",
                incumbent_index=0,
                minimum_improvement=6,
            )
        self.assertEqual(selected, "https://example.test/incumbent.jpg")

    def test_explicit_semantic_rejection_regenerates_once_with_feedback(self) -> None:
        class RetryClient:
            def __init__(self):
                self.generate_prompts = []
                self.review_calls = 0

            def generate_image_candidates(self, prompt, *args, **kwargs):
                self.generate_prompts.append(prompt)
                prefix = "first" if len(self.generate_prompts) == 1 else "retry"
                return (
                    [
                        f"https://example.test/{prefix}-a.jpg",
                        f"https://example.test/{prefix}-b.jpg",
                    ],
                    "image-model",
                )

            def select_best_detail_image(self, *args, **kwargs):
                self.review_calls += 1
                rejected = self.review_calls == 1
                return {
                    "selected_index": 0,
                    "candidates": [
                        {
                            "index": index,
                            "usable": not rejected,
                            "identity_consistent": not rejected,
                            "construction_consistent": True,
                            "color_consistent": True,
                            "pattern_consistent": True,
                            "slot_match": True,
                            "critical_structure_unambiguous": True,
                            "anatomy_natural": True,
                            "unwanted_text": False,
                            "unwanted_brand_or_logo": False,
                            "prohibited_visual": False,
                            "major_artifacts": False,
                            "product_coverage": "high",
                            "score": 90,
                            "reason": "wrong product identity" if rejected else "fixed",
                        }
                        for index in range(2)
                    ],
                }

        facts = load_product_facts(
            DATA / "product_info/product_5681480836479.json"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temp_dir),
                logger=logging.getLogger("semantic-retry-test"),
                offline=True,
            )
            client = RetryClient()
            pipeline.client = client
            selected, _ = pipeline._generate_detail_with_semantic_retry(
                2,
                facts,
                "initial detail prompt",
                references=["https://example.test/source.jpg"],
            )
        self.assertEqual(selected, "https://example.test/retry-a.jpg")
        self.assertEqual(len(client.generate_prompts), 2)
        self.assertIn("identity_consistent", client.generate_prompts[1])

    def test_fast_profile_uses_one_detail_candidate_without_per_image_judge(self) -> None:
        class FastClient:
            def __init__(self) -> None:
                self.requested_counts: list[int] = []

            def generate_image_candidates(self, *args, count=2, **kwargs):
                self.requested_counts.append(count)
                return ["https://example.test/fast-detail.jpg"], "fast-image"

            @staticmethod
            def select_best_detail_image(*args, **kwargs):
                raise AssertionError("fast profile must skip the per-detail judge")

        facts = load_product_facts(DATA / "product_info/product_5681480836479.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(
                input_dir=DATA,
                output_dir=Path(temp_dir),
                logger=logging.getLogger("fast-detail-test"),
                offline=True,
                run_profile="fast",
            )
            client = FastClient()
            pipeline.client = client
            selected, model = pipeline._generate_detail_with_semantic_retry(
                2,
                facts,
                "fast detail prompt",
                references=["https://example.test/source.jpg"],
            )
        self.assertEqual(selected, "https://example.test/fast-detail.jpg")
        self.assertEqual(model, "fast-image")
        self.assertEqual(client.requested_counts, [1])


class FactAndTaxonomyTests(unittest.TestCase):
    def test_hidden_category_ranking_uses_product_family_audience_and_specialization(self) -> None:
        facts = load_product_facts(
            DATA / "product_info/product_5681480836479.json"
        )
        hidden_facts = replace(
            facts, source_category_id="", source_category_name=""
        )
        taxonomy = resolve_taxonomy(
            hidden_facts,
            load_json(DATA / "clothing_categories.json"),
            load_json(DATA / "clothing_attributes.json"),
        )
        top = taxonomy.category.candidates[0]
        self.assertIn("针织", top["name"] + top["path"])
        self.assertIn("女", top["name"] + top["path"])
        self.assertNotIn("女童", top["name"] + top["path"])
        self.assertNotIn("运动", top["name"] + top["path"])

    def test_hidden_shorts_ranking_enforces_length_and_avoids_unverified_specialization(self) -> None:
        facts = load_product_facts(
            DATA / "product_info/product_9493156931235.json"
        )
        hidden_facts = replace(facts, source_category_id="", source_category_name="")
        taxonomy = resolve_taxonomy(
            hidden_facts,
            load_json(DATA / "clothing_categories.json"),
            load_json(DATA / "clothing_attributes.json"),
        )
        self.assertEqual(taxonomy.category.candidates[0]["category_id"], "30341")
        self.assertNotIn("长裤", taxonomy.category.candidates[0]["path"])

    def test_canonical_storyboard_cannot_be_overridden_by_model_slots(self) -> None:
        facts = load_product_facts(DATA / "product_info/product_9493156931235.json")
        taxonomy = resolve_taxonomy(
            facts,
            load_json(DATA / "clothing_categories.json"),
            load_json(DATA / "clothing_attributes.json"),
        )

        class ConflictingPlanner:
            config = SimpleNamespace(chat_model="planner")

            @staticmethod
            def chat_json(*args, **kwargs):
                base = "Use source-faithful commercial styling with neutral light and no text. " * 2
                return {
                    "visual_theme": "neutral daylight, ivory and charcoal",
                    "main_prompt": base,
                    "detail_prompts": [
                        base + "FOCUS EXCLUSIVELY ON WAISTBAND CLOSEUP"
                        for _ in range(5)
                    ],
                    "video_prompt": base,
                    "market_angles": {"en": "clarity", "ko": "명확성", "pt": "clareza"},
                }

        plan, _ = create_creative_plan(facts, taxonomy, {}, ConflictingPlanner())
        self.assertNotIn("FOCUS EXCLUSIVELY", " ".join(plan.detail_prompts))
        self.assertEqual(
            plan.detail_roles,
            [
                "complete_product",
                "primary_verified_detail",
                "secondary_verified_detail",
                "verified_alternate_view",
                "product_only_context",
            ],
        )
        prompts = " ".join(plan.detail_prompts + [plan.video_prompt]).casefold()
        self.assertNotIn("waistband", prompts)
        self.assertNotIn("both legs", prompts)
        self.assertNotIn("shoulder to hem", prompts)

    def test_copy_concept_gate_accepts_natural_synonyms(self) -> None:
        text = "A straight-leg silhouette finishes at a knee-length hem."
        self.assertTrue(_localized_concept_is_mentioned("en", "Straight Fit", text))
        self.assertTrue(
            _localized_concept_is_mentioned("en", "Knee-Length Shorts", text)
        )

    def test_natural_copy_is_not_discarded_for_canonical_wording_difference(self) -> None:
        facts = load_product_facts(DATA / "product_info/product_9493156931235.json")
        taxonomy = resolve_taxonomy(facts, self.tree, self.attributes)
        plan = fallback_creative_plan(facts, taxonomy)
        fallback, _ = generate_copy_payload("en", facts, taxonomy, plan, None)
        draft = dict(fallback)
        draft.update(
            {
                "title": "Men's Solid-Color Drawstring Knee-Length Shorts",
                "overview": (
                    "A straight-leg silhouette and knee-length hem define these solid-color shorts, "
                    "finished with a drawstring waist and polyester fabric.\n\n"
                    "Seven seller-listed colors and sizes M through 5XL are shown in the option tables below."
                ),
                "highlights": [
                    "Straight-leg silhouette",
                    "Knee-length hem",
                    "Solid-color finish",
                    "Polyester fabric",
                ],
            }
        )

        class CopyClient:
            config = SimpleNamespace(chat_model="copy-model")
            trace = None

            def __init__(self):
                self.calls = 0

            def chat_json(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return draft
                raise ApiError("auditor unavailable", retryable=True, category="queue")

        payload, source = generate_copy_payload(
            "en", facts, taxonomy, plan, CopyClient()
        )
        self.assertEqual(payload["title"], draft["title"])
        self.assertEqual(source, "copy-model-validated-draft")

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

    def test_copy_schema_accepts_natural_paragraphing_and_extra_metadata(self) -> None:
        facts = load_product_facts(DATA / "product_info/product_5758364264251.json")
        taxonomy = resolve_taxonomy(facts, self.tree, self.attributes)
        payload = _fallback_payload("en", facts, taxonomy)
        payload["overview"] = " ".join(payload["overview"].split())
        payload["diagnostics"] = {"draft": "model-note"}
        payload["localized_terms"]["__exact_source_label__"] = "原厂型号 A"

        error = _payload_validation_error(
            "en",
            payload,
            facts,
            taxonomy,
            set(payload["media_descriptions"]),
            set(payload["localized_terms"]),
        )
        self.assertEqual(error, "")

        payload["title"] += " 中文残留"
        error = _payload_validation_error(
            "en",
            payload,
            facts,
            taxonomy,
            set(payload["media_descriptions"]),
            set(payload["localized_terms"]),
        )
        self.assertEqual(error, "source-script-contamination-guard")

    def test_fast_copy_skips_second_model_call_when_draft_is_valid(self) -> None:
        facts = load_product_facts(DATA / "product_info/product_5758364264251.json")
        taxonomy = resolve_taxonomy(facts, self.tree, self.attributes)
        plan = fallback_creative_plan(facts, taxonomy)
        draft = _fallback_payload("en", facts, taxonomy)

        class CopyClient:
            config = SimpleNamespace(chat_model="fast-copy-model")
            trace = None

            def __init__(self) -> None:
                self.calls = 0

            def chat_json(self, *args, **kwargs):
                self.calls += 1
                return draft

        client = CopyClient()
        payload, source = generate_copy_payload(
            "en",
            facts,
            taxonomy,
            plan,
            client,
            audit_valid_draft=False,
        )
        self.assertEqual(client.calls, 1)
        self.assertEqual(payload["title"], draft["title"])
        self.assertEqual(source, "fast-copy-model-validated-draft-fast")

    def test_chinese_gate_separates_buyer_copy_from_machine_appendix(self) -> None:
        text = """# Natural product title

## Product description

Natural localized overview.

## Key features

- Verified feature

## Listing information

- **Exact source label:** 原厂型号 A

## Media guide

- **main_image.jpeg:** Localized media description.
"""
        buyer, machine = _description_language_surfaces(text, "en")
        self.assertNotRegex(buyer, r"[\u4e00-\u9fff]")
        self.assertIn("原厂型号", machine)

        contaminated = text.replace(
            "Localized media description.", "Localized media description 中文"
        )
        buyer, _ = _description_language_surfaces(contaminated, "en")
        self.assertRegex(buyer, r"[\u4e00-\u9fff]")

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
        self.assertIn(taxonomy.category.category_id, text)
        self.assertIn(facts.skus[-1].sku_id, text)
        self.assertIn("product_video.mp4", text)
        self.assertIn("Seller Label", text)
        self.assertIn("40–47.5 kg", text)
        self.assertNotRegex(text, r"[\u4e00-\u9fff]")
        self.assertNotIn("/ret/result/result", text)
        self.assertNotIn("Source evidence", text)
        first_sku_row = next(
            line for line in text.splitlines() if facts.skus[0].sku_id in line
        )
        self.assertIn("88.2–104.7 lb", first_sku_row)

    def test_568_localization_keeps_locale_units_and_complete_one_size_label(self) -> None:
        facts = load_product_facts(DATA / "product_info/product_5681480836479.json")
        taxonomy = resolve_taxonomy(facts, self.tree, self.attributes)
        plan = fallback_creative_plan(facts, taxonomy)

        rendered: dict[str, str] = {}
        for language in ("en", "ko", "pt"):
            payload, _ = generate_copy_payload(language, facts, taxonomy, plan, None)
            rendered[language] = render_description(
                language, payload, facts, taxonomy
            )

        self.assertIn("Halter neck", rendered["en"].splitlines()[0])
        self.assertIn("Cold-shoulder", rendered["en"].splitlines()[0])
        self.assertIn("| Product | 100157 | Material | 1001111 | Viscose |", rendered["en"])
        self.assertIn("Listed material", rendered["en"])
        self.assertNotIn("| Main material | Viscose |", rendered["en"])
        self.assertIn(
            "One Size — 40–60 kg (88.2–132.3 lb)", rendered["en"]
        )
        self.assertNotIn("Seller-declared source value", rendered["en"])
        self.assertNotIn("()", rendered["en"])
        self.assertIn("프리사이즈 — 40–60 kg", rendered["ko"])
        self.assertNotIn("88.2–132.3 lb", rendered["ko"])
        self.assertNotIn("야드파운드법", rendered["ko"])
        self.assertIn("Tamanho único — 40–60 kg", rendered["pt"])
        self.assertNotIn("88.2–132.3 lb", rendered["pt"])
        self.assertNotIn("Imperial", rendered["pt"])

    def test_storyboard_adapts_to_children_and_bottoms(self) -> None:
        children = load_product_facts(
            DATA / "product_info/product_8688570444629.json"
        )
        children_taxonomy = resolve_taxonomy(children, self.tree, self.attributes)
        children_plan = fallback_creative_plan(children, children_taxonomy)
        self.assertIn("do not add a child, adult", children_plan.main_prompt)
        self.assertIn("product-only", children_plan.detail_prompts[4])
        self.assertNotIn("show one adult wearer", children_plan.detail_prompts[4])

        shorts = load_product_facts(DATA / "product_info/product_9493156931235.json")
        shorts_taxonomy = resolve_taxonomy(shorts, self.tree, self.attributes)
        shorts_plan = fallback_creative_plan(shorts, shorts_taxonomy)
        self.assertIn("do not introduce a wearer", shorts_plan.main_prompt)
        self.assertIn("source-visible", shorts_plan.detail_prompts[1])
        self.assertNotIn("waistband", " ".join(shorts_plan.detail_prompts))
        self.assertNotIn("both legs", shorts_plan.video_prompt)

    def test_storyboard_uses_construction_instead_of_invented_single_color_variants(self) -> None:
        single_color = load_product_facts(
            DATA / "product_info/product_5758364264251.json"
        )
        single_taxonomy = resolve_taxonomy(single_color, self.tree, self.attributes)
        single_plan = fallback_creative_plan(single_color, single_taxonomy)
        self.assertIn("source-supported alternate", single_plan.detail_prompts[3])
        self.assertNotIn("color variants", single_plan.detail_prompts[3])
        self.assertIn("Campaign Style Lock", single_plan.visual_theme)

        multi_color = load_product_facts(
            DATA / "product_info/product_3887087154767.json"
        )
        multi_taxonomy = resolve_taxonomy(multi_color, self.tree, self.attributes)
        multi_plan = fallback_creative_plan(
            multi_color,
            multi_taxonomy,
            {
                "visible_colors": ["black", "blue"],
                "source_images": [
                    {"role": "variant", "dominant_color": "black"},
                    {"role": "variant", "dominant_color": "blue"},
                ],
            },
        )
        self.assertIn("distinct variants", multi_plan.detail_prompts[3])

    def test_source_vision_batches_keep_late_size_chart_indexes(self) -> None:
        urls = [f"https://example.test/{index}.jpg" for index in range(14)]
        batches = [
            (
                urls[:12],
                {
                    "images": [
                        {"index": index, "role": "detail"}
                        for index in range(12)
                    ]
                },
            ),
            (
                urls[12:],
                {
                    "images": [
                        {"index": 0, "role": "size_chart", "has_text": True},
                        {"index": 1, "role": "detail"},
                    ],
                    "size_chart_rows": [
                        {
                            "size_label": "M",
                            "bust_cm": "90",
                            "length_cm": "60",
                            "weight_guidance": "",
                            "source_image_index": 0,
                        }
                    ],
                },
            ),
        ]
        merged = _merge_source_vision_batches(batches, urls)
        self.assertEqual(len(merged["source_images"]), 14)
        self.assertEqual(merged["size_chart_rows"][0]["source_image_index"], 12)

    def test_fact_driven_copy_is_benefit_led_without_product_override(self) -> None:
        facts = load_product_facts(
            DATA / "product_info/product_5758364264251.json"
        )
        taxonomy = resolve_taxonomy(facts, self.tree, self.attributes)
        plan = fallback_creative_plan(facts, taxonomy)
        payload, _ = generate_copy_payload("en", facts, taxonomy, plan, None)
        rendered = render_description("en", payload, facts, taxonomy)

        self.assertEqual(
            rendered.splitlines()[0],
            "# Women's T-Shirt with a vintage floral print, a V-neck, long sleeves, and a relaxed fit",
        )
        self.assertIn("V-neckline creates a clean, open shape", rendered)
        self.assertIn("Relaxed fit leaves room through the body", rendered)
        self.assertIn(
            "Seller size labels: XL, XXL, XXXL, 4XL, and 5XL", rendered
        )
        self.assertIn("Mapped from seller product title", rendered)
        self.assertIn("Mapped from seller SKU attribute: Color", rendered)
        self.assertEqual(
            rendered.count("Body measurements are not provided"), 1
        )

    def test_category_titles_generalize_without_product_or_category_id_branches(self) -> None:
        cases = {
            "8688570444629": ("Boys' T-shirt with short sleeves", "Cotton"),
            "9493156931235": ("Men's flat-front shorts", "Straight cut"),
        }
        for product_id, (title_fragment, feature_fragment) in cases.items():
            with self.subTest(product_id=product_id):
                facts = load_product_facts(
                    DATA / f"product_info/product_{product_id}.json"
                )
                taxonomy = resolve_taxonomy(facts, self.tree, self.attributes)
                plan = fallback_creative_plan(facts, taxonomy)
                payload, _ = generate_copy_payload(
                    "en", facts, taxonomy, plan, None
                )
                rendered = render_description("en", payload, facts, taxonomy)
                self.assertIn(title_fragment, rendered.splitlines()[0])
                self.assertIn(feature_fragment, rendered)
                self.assertNotIn("Seller-declared source value", rendered)
                self.assertNotIn("MenChildren", rendered)


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
