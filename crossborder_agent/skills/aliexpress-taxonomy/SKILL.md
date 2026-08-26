---
name: aliexpress-taxonomy
description: Select and review AliExpress apparel leaf categories, enumerated product attributes, and sales-option values from the supplied platform snapshot.
---

# AliExpress Taxonomy

Choose only from supplied leaf-category candidates and preserve the exact category ID. Use verified product type, intended wearer, garment family, silhouette, and source category as evidence; visual styling alone must not override stronger structured facts.

Map attributes only to keys and enumerated values allowed by the selected category schema. Preserve platform attribute IDs, value IDs, sales-attribute status, source values, and evidence pointers. Never invent a required value that is absent from the source.

If the leaf category has no independent attribute metadata and the platform snapshot supplies a parent or common schema, keep the selected leaf ID for listing and report the schema category separately. Do not mistake a schema fallback for a different listing category.

When reviewing A3, compare exact IDs and enumerations rather than judging whether a free-text label merely sounds plausible.

