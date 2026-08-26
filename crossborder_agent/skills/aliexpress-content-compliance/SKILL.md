---
name: aliexpress-content-compliance
description: Review AliExpress listing text, taxonomy, SKU data, generated media, source references, and prompts against task-scoped content policy and platform listing rules.
---

# AliExpress Content Compliance

Apply this policy as a layered pre-publication gate. Evaluate the complete customer-facing listing: title, description, attributes, category, SKU options, images, visible text, video, and generation prompts. Judge the artifact that will be delivered, not the producer's explanation or intention.

## Rule precedence and scope

1. Follow the task's explicit contract and supplied platform snapshot.
2. Apply the AliExpress listing rules below.
3. Apply destination-market or category-specific requirements only when they are supplied as evidence. Never invent a legal requirement, certificate, license, warning, or authorization.

AliExpress rules and destination-market requirements can change. Treat the policy sources listed at the end as the verified baseline, not as a substitute for a current Seller Center check before live publication. When required evidence is unavailable, return `external_launch_check` rather than claiming compliance.

The competition's A1 check excludes legal and intellectual-property adjudication. Do not score a suspected trademark, character, patent, or copyright as an A1 violation solely because authorization is unknown. Record unknown brand or content rights as an `external_launch_check`. A known counterfeit, an explicitly unauthorized use, or a newly introduced unrelated brand remains a blocker. Separately enforce marketplace visual readiness: unrelated logos, text, props, and platform marks must not survive into the final listing asset.

## Platform listing gates

### Product eligibility and required evidence

- Block a listing when supplied evidence identifies the product as prohibited, illegal, sanctioned, unsafe, counterfeit, or unavailable for the destination market.
- For restricted or special categories, require the supplied license, registration, safety document, warning, or authorization. If the task provides no way to establish it, request an `external_launch_check`; do not fabricate approval.
- Do not imply that a generated compliance review replaces the seller's responsibility to follow applicable product-safety, consumer-protection, import/export, advertising, and destination-market laws.

### Truthfulness and product identity

- All listing content must be true, accurate, complete for the required fields, lawful, and consistent with the verified fact ledger and source pixels.
- Block false, misleading, deceptive, contradictory, or materially incomplete claims. This includes invented specifications, functions, materials, measurements, stock, certifications, compatibility, regional sizing, performance, or included accessories.
- The title, description, category, attributes, SKU options, images, and video must describe the same sellable product. Do not replace an established listing identity with a materially different product.
- Do not copy another seller's title, description, images, price, attributes, or other listing content. Source material explicitly supplied by the task may be transformed only within the task's provenance and usage constraints.

### Title and description

- The title must clearly identify the concrete product and verified differentiators. It must match the selected category, attributes, SKU data, and delivered media.
- Reject keyword stuffing, repeated search terms, irrelevant keywords, irrelevant brands, competitor names, or wording that obscures what is actually sold.
- Exclude profanity or vulgar language in every locale and in any visible image or video text.
- Exclude defamatory, threatening, harassing, obscene, sexually explicit, discriminatory, hateful, or minor-harming content unless the task explicitly covers a permitted age-restricted category with the required controls.
- Do not publish personal data, seller contact details, phone numbers, email addresses, social handles, messaging IDs, QR codes, or off-platform links. Do not direct buyers away from AliExpress. A supplied, platform-permitted product instruction link requires explicit evidence and must not be inferred.

### Claims and promotions

- Exclude prices, discounts, coupons, urgency, scarcity, ratings, review counts, sales counts, platform promotions, or comparative price claims from newly generated copy and media. Platform-managed price and promotion fields are outside this material agent's scope.
- Exclude unsupported certification, authorization, endorsement, guarantee, medical, therapeutic, body-transformation, safety, environmental, durability, comfort, or performance claims.
- Do not use superlatives, exclusivity, or comparative superiority unless the claim is both supplied and objectively substantiated.

### Category, attributes, and SKU integrity

- Publish only in the best matching supplied leaf category. Never select a strategically convenient but inaccurate category.
- Use only attributes and enumerated values allowed by the resolved platform schema. All customer-visible units must match the verified source and localization rules.
- Every SKU option must be a genuine variant of the same product and must match the supplied sellable combinations. Do not add unavailable colors or sizes, unrelated low-price items, placeholder variants, misleading quantities, or options intended to manipulate the displayed price.
- Do not treat a parent/common attribute-schema fallback as permission to replace the selected listing leaf category.

### Images and video

- The primary image must show the actual product rather than text in place of the product. Every delivered image and video must correspond to the product, its verified variants, and reality.
- Block media that misrepresents color, silhouette, construction, quantity, included items, scale, function, or results, or that introduces an unrelated product.
- Generated customer-facing media must not contain promotional text, prices, discounts, contact details, QR codes, external links, ratings, review or sales counts, unsupported seals, platform marks, watermarks, or newly invented brand marks.
- Intrinsic product print or a sewn label must remain faithful to the source. Do not erase or alter a product-defining mark merely to make an uncertain authorization issue disappear.
- Do not introduce text into generated commerce images. A verified seller size chart may be deterministically re-rendered when the task contract permits it and all values remain traceable.

## Reference and candidate handling

Separate reference safety from direct-listing safety. Contact details, QR codes, watermarks, marketplace marks, price/review graphics, certification seals and sensitive or prohibited content are hard reference blockers. A clear product photo containing a person, ordinary prop, background text or third-party styling mark may be used for identity-preserving image editing, but it requires cleanup and may never be delivered unchanged. Intrinsic product print or a sewn label must remain faithful to the source.

Preserve canonical source data and field-level evidence in the private fact ledger. Publish only localized listing fields plus the exact category, attribute/value and SKU identifiers needed for platform parsing; do not expose JSON pointers, evidence labels or raw Chinese values in shopper deliverables. Do not convert a contaminated reference into an unsupported factual claim.

## Decision and feedback contract

Classify every concrete finding as one of:

- `blocker`: explicit prohibited/illegal product evidence, known counterfeit or unauthorized content, off-platform contact or redirection, materially false or deceptive content, prohibited harmful content, or a required artifact that cannot be made publishable.
- `major`: wrong category, unsupported material claim, misleading image, mismatched attribute or SKU, keyword abuse, or another issue that must be repaired before delivery.
- `minor`: a localized or presentation issue that does not misstate the product and can be repaired without changing identity.
- `external_launch_check`: brand rights, category permits, product-safety documents, destination-market warnings, or other requirements that cannot be established from task evidence.

Compliance feedback must identify the exact artifact or field, the visible or textual evidence, the violated gate, and the smallest permitted repair. Do not trigger a fallback merely because review is unavailable or uncertain; request a targeted revision when a concrete violation exists. Never describe a known degraded fallback as compliant or validated.

Set content compliance to ready only when there are no `blocker` or `major` findings. Report `external_launch_check` items separately and never silently convert them into a pass. Under the task's A1 boundary, an external intellectual-property check alone does not lower A1 unless the supplied evidence establishes a violation.

## Policy baseline

- Last verified: 2026-08-25.
- AliExpress.com Terms of Use: https://terms.alicdn.com/legal-agreement/terms/suit_bu1_aliexpress/suit_bu1_aliexpress202204182115_66077.html
- AliExpress seller product-publication rules: https://business.aliexpress-cis.com/help/article/products-publication-rules
- These sources establish the general baseline. Live publication must also check the current seller region, destination market, leaf category, and any applicable program or channel rules.
