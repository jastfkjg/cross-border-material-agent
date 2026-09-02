from __future__ import annotations

import unittest

from crossborder_agent.claims import build_claim_ledger
from crossborder_agent.decision_state import (
    DependencyState,
    assess_evidence_sufficiency,
    build_canonical_product_state,
    build_expected_delivery_spec,
)
from crossborder_agent.models import (
    CategoryChoice,
    MappedAttribute,
    ProductAttribute,
    ProductFacts,
    Sku,
    SkuAttribute,
    TaxonomyResult,
)
from crossborder_agent.planning import validate_creative_plan_payload


def _facts() -> ProductFacts:
    return ProductFacts(
        platform="fixture",
        source_url="https://example.invalid/item",
        offer_id="synthetic",
        source_title="Synthetic source title",
        source_category_id="source-leaf",
        source_category_name="Synthetic source category",
        attributes=[
            ProductAttribute(
                "material-source",
                "Listed construction",
                "conflicting raw value",
                evidence_pointer="$.attributes[0]",
            )
        ],
        skus=[
            Sku(
                "sku-1",
                "spec-1",
                [
                    SkuAttribute(
                        "variant-source",
                        "Variant",
                        "Option A",
                        evidence_pointer="$.skus[0].attributes[0]",
                    )
                ],
            )
        ],
        product_image_urls=["https://example.invalid/image-a.jpg"],
        sku_image_urls=[],
        description_image_urls=[],
        size_conversions=[],
        input_file="fixture.json",
        fingerprint="fixture",
        source_title_evidence_pointer="$.title",
    )


def _taxonomy() -> TaxonomyResult:
    return TaxonomyResult(
        category=CategoryChoice("leaf", "Leaf", "Root > Leaf", 0.9, "model"),
        attributes=[
            MappedAttribute(
                attr_id="variant-platform",
                name="Platform variant",
                source_name="Variant",
                source_value="Option A",
                source_evidence_pointer="$.skus[0].attributes[0]",
                value_id="option-a",
                platform_value="Option A",
                sales_attribute=True,
            )
        ],
        attribute_schema_category_id="leaf",
    )


class DecisionStateTests(unittest.TestCase):
    def test_claim_surfaces_preserve_nonvisual_fact_without_using_it_for_media(self) -> None:
        facts = _facts()
        facts.reconciled_fact_ledger = {
            "attribute_decisions": [
                {
                    "attribute_index": 0,
                    "decision": "machine_only",
                    "surface_decisions": {
                        "buyer_copy": "publish",
                        "media_generation": "machine_only",
                        "marketplace_mapping": "publish",
                        "machine_appendix": "publish",
                    },
                    "reason": "source-grounded but not visually observable",
                }
            ]
        }

        claim = next(
            item
            for item in build_claim_ledger(facts, _taxonomy())
            if item.source_type == "seller_attribute"
        )

        self.assertIn("buyer_copy", claim.allowed_surfaces)
        self.assertIn("machine_appendix", claim.allowed_surfaces)
        self.assertNotIn("image_prompt", claim.allowed_surfaces)

    def test_canonical_state_keeps_conflict_evidence_but_not_publishable_raw_claim(self) -> None:
        facts = _facts()
        facts.reconciled_fact_ledger = {
            "seller_title_decision": "publish",
            "attribute_decisions": [
                {
                    "attribute_index": 0,
                    "decision": "reject",
                    "canonical_value": "N/A",
                    "reason": "independent evidence conflicts",
                }
            ],
            "canonical_visual_claims": [
                {
                    "concept": "visible construction",
                    "value": "observed alternative",
                    "evidence": ["source-image:0"],
                }
            ],
            "conflicts": [
                {
                    "claim": "Listed construction",
                    "resolution": "use visual observation for public surfaces",
                }
            ],
        }

        canonical = build_canonical_product_state(facts, facts.reconciled_fact_ledger)
        rejected = next(item for item in canonical.claims if item.claim_id == "source-attribute-0")
        observed = next(item for item in canonical.claims if item.claim_id == "visual-claim-0")

        self.assertNotIn("buyer_copy", rejected.allowed_surfaces)
        self.assertIn("machine_appendix", rejected.allowed_surfaces)
        self.assertIn("buyer_copy", observed.allowed_surfaces)
        self.assertEqual(canonical.unresolved_questions, [])

    def test_frozen_spec_detects_mapping_coverage_lost_by_repair(self) -> None:
        facts = _facts()
        canonical = build_canonical_product_state(facts, {})
        evidence = assess_evidence_sufficiency(
            {
                "source_images": [
                    {
                        "index": 0,
                        "role": "front",
                        "inspection_complete": True,
                        "safe_for_generation_reference": True,
                        "safe_for_listing_fallback": True,
                    }
                ]
            },
            canonical,
        )
        taxonomy = _taxonomy()
        spec = build_expected_delivery_spec(
            canonical=canonical,
            taxonomy=taxonomy,
            claim_ledger=build_claim_ledger(facts, taxonomy),
            evidence=evidence,
            required_files={"one", "two"},
        )
        repaired = TaxonomyResult(
            category=taxonomy.category,
            attributes=[],
            attribute_schema_category_id="leaf",
        )

        self.assertEqual(
            spec.taxonomy_coverage_gaps(repaired),
            [{"scope": "sales", "source_name": "Variant", "source_value": "Option A"}],
        )

    def test_dependency_graph_marks_downstream_state_stale(self) -> None:
        dependencies = DependencyState()
        dependencies.record("canonical", {"version": 1})
        dependencies.record("taxonomy", {"version": 1}, canonical="one")
        dependencies.record("copy", {"version": 1}, taxonomy="one")

        dependencies.invalidate(
            "canonical", ["taxonomy", "copy", "review"], "evidence decision changed"
        )

        self.assertEqual(dependencies.stale_nodes(), ["copy", "review", "taxonomy"])

    def test_creative_plan_rejects_unavailable_evidence_index(self) -> None:
        safe_prompt = (
            "Create one source-faithful product composition with accurate visible construction, "
            "neutral lighting, complete framing, and no invented product features."
        )
        payload = {
            "visual_theme": "Grounded neutral campaign",
            "main": {
                "prompt": safe_prompt,
                "candidate_count": 1,
                "reference_roles": ["front"],
                "reference_indexes": [7],
            },
            "details": [
                {
                    "role": f"job-{index}",
                    "prompt": safe_prompt + f" Distinct evidence-backed job {index}.",
                    "candidate_count": 1,
                    "reference_roles": ["detail"],
                    "reference_indexes": [0],
                }
                for index in range(5)
            ],
            "video": {"prompt": safe_prompt + " Use restrained stable camera motion."},
            "market_angles": {"en": "clear", "ko": "명확함", "pt": "clareza"},
        }

        plan, error = validate_creative_plan_payload(
            payload, available_reference_indexes={0}
        )

        self.assertIsNone(plan)
        self.assertIn("unavailable source image indexes", error)


if __name__ == "__main__":
    unittest.main()
