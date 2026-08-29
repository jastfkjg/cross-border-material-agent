---
name: aliexpress-taxonomy
description: Select and review AliExpress leaf categories, enumerated product attributes, and sales-option values from the supplied platform snapshot.
---

# AliExpress Taxonomy

## Compilable rules

- [taxonomy][hard] Choose only a leaf exposed by the supplied taxonomy exploration tools and preserve its exact ID; product family, audience, form factor, intended use and explicit source attributes outrank visual styling or wording coincidence.
- [taxonomy][hard] Control retrieval iteratively: search with model-chosen phrases, navigate parents and children, inspect ambiguous alternatives, paginate when necessary, and revise the query after an empty or weak observation. Do not require a pre-ranked lexical Top-N and do not assume the first search is complete.
- [taxonomy][hard] Map only supplied schema keys and enumerated values, preserve exact attribute/value IDs and genuine SKU combinations, and never invent a missing required value.
- [taxonomy][hard] Inspect the chosen attribute schema and every attribute/value used in the final mapping. A parent or common schema may supply attribute definitions without changing the selected listing leaf ID.
- [taxonomy][soft] Prefer an ordinary category over a specialized use-case leaf unless the supplied title, attributes or SKU facts directly support that specialization.
- [final-review][hard] Judge A3 by exact leaf, attribute, value and SKU identifiers rather than plausible free-text labels.
