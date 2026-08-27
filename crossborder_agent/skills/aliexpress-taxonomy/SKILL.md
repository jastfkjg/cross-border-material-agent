---
name: aliexpress-taxonomy
description: Select and review AliExpress apparel leaf categories, enumerated product attributes, and sales-option values from the supplied platform snapshot.
---

# AliExpress Taxonomy

## Compilable rules

- [taxonomy][hard] Choose only a supplied leaf category and preserve its exact ID; product family, intended wearer, garment length and explicit source attributes outrank visual styling.
- [taxonomy][hard] Reject an explicit alias when it conflicts with verified product type, wearer or garment length; aliases may guide ranking but cannot bypass compatibility checks.
- [taxonomy][hard] Map only supplied schema keys and enumerated values, preserve exact attribute/value IDs and genuine SKU combinations, and never invent a missing required value.
- [taxonomy][hard] A parent or common schema may supply attribute definitions without changing the selected listing leaf ID.
- [taxonomy][soft] Prefer ordinary apparel leaves unless supplied evidence supports a specialized sport, uniform, maternity, plus-size or novelty purpose.
- [final-review][hard] Judge A3 by exact leaf, attribute, value and SKU identifiers rather than plausible free-text labels.
