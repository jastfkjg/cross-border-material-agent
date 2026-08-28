---
name: aliexpress-taxonomy
description: Select and review AliExpress leaf categories, enumerated product attributes, and sales-option values from the supplied platform snapshot.
---

# AliExpress Taxonomy

## Compilable rules

- [taxonomy][hard] Choose only from the complete supplied leaf set and preserve its exact ID; product family, audience, form factor, intended use and explicit source attributes outrank visual styling or lexical hints.
- [taxonomy][hard] Treat local lexical ranking only as a weak hint. Inspect every supplied leaf and reject any candidate whose specialization, audience or product family is unsupported by verified facts.
- [taxonomy][hard] Map only supplied schema keys and enumerated values, preserve exact attribute/value IDs and genuine SKU combinations, and never invent a missing required value.
- [taxonomy][hard] A parent or common schema may supply attribute definitions without changing the selected listing leaf ID.
- [taxonomy][soft] Prefer an ordinary category over a specialized use-case leaf unless the supplied title, attributes or SKU facts directly support that specialization.
- [final-review][hard] Judge A3 by exact leaf, attribute, value and SKU identifiers rather than plausible free-text labels.
