---
name: rubric-evaluation
description: Independently score a complete commerce-material delivery against task dimensions A1-A7 and choose evidence-based bounded repairs.
---

# Rubric Evaluation

Judge artifacts, not producer explanations. Score the whole delivery using the task weights and distinguish physical completeness from semantic quality.

- A1: all copy, visible text, and visual elements comply with the supplied marketplace content policy. Follow the task boundary that excludes legal and intellectual-property adjudication; unknown brand authorization alone is not an A1 failure, although unrelated marks can still make an image unusable under A4/A6 or violate explicit listing-image rules.
- A2: every required file exists, is parseable, and meets exact text/image/video specifications.
- A3: leaf category, platform attributes, value IDs, and SKU option values match the supplied platform data.
- A4: en-US, ko-KR, pt-BR, sizing, units, cultural treatment, and channel visuals are genuinely localized.
- A5: every verifiable claim and visible annotation is traceable to the source fact ledger or direct image evidence.
- A6: the complete six-image set is usable, coherent, distinct, identity-consistent, and free of major defects. Calculate the threshold: at least five of six images must be usable; similar crops of one scene and physically valid but semantically wrong fallbacks are not independently usable.
- A7: the video plays, remains temporally stable and product-faithful, and provides useful product presentation.

For every blocker or major issue, name the artifact, concrete evidence, expected outcome, and one permitted repair tool. Rank repairs by weighted impact and confidence. Do not request broad regeneration when a field-level or single-slot correction is sufficient.

Never turn reviewer uncertainty or API failure into an automatic fallback. A repair preserves the current version until a new candidate is successfully generated and accepted. Mark ready only when the configured threshold is met, there are no blockers, and A1/A2/A5 have no major issue.
