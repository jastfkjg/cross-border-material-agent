from __future__ import annotations

import unittest
from pathlib import Path

from crossborder_agent.skill_runtime import SkillLibrary


ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = ROOT / "crossborder_agent" / "skills"


class SkillCompilationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.skills = SkillLibrary(SKILLS_ROOT)

    def test_copy_stage_contains_only_copy_relevant_rules(self) -> None:
        compiled = self.skills.compile(
            "copy", "product-grounding", "marketplace-materials"
        )

        self.assertIn("required product source URL", compiled)
        self.assertIn("buyer prose", compiled)
        self.assertIn("seller measurement labels", compiled)
        self.assertNotIn("The hero is", compiled)
        self.assertNotIn("eight-second video", compiled)
        self.assertNotIn("---", compiled)
        self.assertNotIn("# Marketplace", compiled)

    def test_creative_stage_removes_conflicting_palette_and_shot_caps(self) -> None:
        compiled = self.skills.compile(
            "creative-plan", "product-grounding", "marketplace-materials"
        )

        self.assertIn("every verified seller color", compiled)
        self.assertIn("assigned local structure", compiled)
        self.assertIn("continuous move or a coherent multi-shot sequence", compiled)
        self.assertNotIn("no more than three colors", compiled.casefold())
        self.assertNotIn("three-act", compiled.casefold())
        self.assertIn("single dominant subject", compiled)
        self.assertIn("person, hand or use context only when trusted source evidence supports it", compiled)

    def test_grounding_and_delivery_rules_avoid_ambiguous_or_duplicate_policy(self) -> None:
        source_rules = self.skills.compile("source-vision", "product-grounding")
        self.assertIn("sexually explicit content", source_rules)
        self.assertIn("ordinary non-sensitive product-use scene is not prohibited", source_rules)
        self.assertNotIn("Never use adult,", source_rules)

        manager_rules = self.skills.compile(
            "manager", "delivery-quality", "product-grounding"
        )
        self.assertIn("expected rubric-weighted gain", manager_rules)
        self.assertNotIn("Deliver three localized descriptions", manager_rules)
        self.assertNotIn("Keep generation failure", manager_rules)

    def test_final_review_keeps_task_boundary_and_variant_exception(self) -> None:
        compiled = self.skills.compile(
            "final-review",
            "delivery-quality",
            "product-grounding",
            "aliexpress-taxonomy",
            "marketplace-materials",
        )

        self.assertIn("excludes independent legal", compiled)
        self.assertIn("source URL in the machine appendix", compiled)
        self.assertIn("At least five of six images", compiled)
        self.assertIn("clean verified-variant composition is allowed", compiled)

    def test_skill_set_is_small_and_compiled_context_is_bounded(self) -> None:
        skill_files = sorted(SKILLS_ROOT.glob("*/SKILL.md"))
        self.assertEqual(
            [path.parent.name for path in skill_files],
            [
                "aliexpress-taxonomy",
                "delivery-quality",
                "marketplace-materials",
                "product-grounding",
            ],
        )

        raw_size = sum(len(path.read_text(encoding="utf-8")) for path in skill_files)
        compiled = self.skills.compile(
            "final-review",
            "delivery-quality",
            "product-grounding",
            "aliexpress-taxonomy",
            "marketplace-materials",
        )
        self.assertLess(len(compiled), raw_size)
        self.assertLess(len(compiled), 6000)

    def test_unknown_stage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.skills.compile("everything", "product-grounding")


if __name__ == "__main__":
    unittest.main()
