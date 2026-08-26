---
name: marketplace-localization
description: Create and review fact-grounded en-US, ko-KR, and pt-BR marketplace copy, sizing language, terminology, and media descriptions.
---

# Marketplace Localization

Localize meaning and shopping intent, not source-language syntax. Keep category IDs, platform attribute/value IDs, SKU IDs and Spec IDs exact in compact listing tables. Evidence pointers and raw source-language values belong to the private fact ledger and must not appear in localized deliverables.

- en-US: concise US marketplace phrasing and spelling. Display seller centimeter measurements with deterministic inch equivalents where useful, and kilograms with pounds when weight guidance is present.
- ko-KR: natural Korean retail sentence endings and option terminology. Retain metric units and avoid inventing Korean numeric size equivalents.
- pt-BR: Brazilian vocabulary and syntax such as `produto`, `tamanho` and `consulte`; avoid European Portuguese usage. Retain metric units and do not invent P/M/G equivalence.

Translate seller `均码` guidance as `One Size`, `프리사이즈`, or `Tamanho único`, followed by the exact deterministic weight conversion. Use pounds and inches only in en-US; do not add imperial-only columns to ko-KR or pt-BR tables. Never publish Chinese source values in en-US, ko-KR or pt-BR deliverables; omit a nonessential row or use an em dash when no safe localized display value exists.

Titles should identify the concrete product and distinctive source-backed design details without keyword stuffing. Put the concrete product noun and strongest differentiators into a natural market-specific phrase rather than a comma-separated attribute dump. The overview must contain two substantive shopper-facing paragraphs: product/design first, options and conservative sizing second. Translate Feature -> visible Advantage -> conservative Buyer Benefit only when each step is supported by the fact ledger or direct pixels. Highlights should read as independently extractable benefit-led feature phrases rather than an audit checklist or raw data dump. Use grammatical phrases with articles and natural inflection; never paste Title Case attribute values directly into a sentence.

If a field is named `main material` but the seller declares its content below 30% and provides no complete fiber composition, do not imply that fiber is predominant. Omit it from shopper prose or label it conservatively as a seller-listed material with incomplete composition information; platform attribute mapping may still retain the exact enumerated material value.

Avoid body shaming, gender stereotypes, cultural caricatures, unsupported benefits, urgency, superlatives, promotional claims, ratings, contact details, external links, and mixed-language fragments. Media descriptions must describe the artifact actually delivered, including a rendered size chart or deterministic catalog video when applicable.

When revising, change only the weak localized fields while preserving compact listing IDs and fact-backed details. A numeric guard failure must repair only the affected fields; it must not discard otherwise natural translations. Do not render empty localization parentheses. Media descriptions must distinguish the actual delivered slots and must not call degraded fallbacks “validated.”
