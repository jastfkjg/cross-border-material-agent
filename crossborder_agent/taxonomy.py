"""Constrained model-first marketplace category and attribute resolution."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Iterable

from .models import (
    CategoryChoice,
    MappedAttribute,
    ProductFacts,
    TaxonomyResult,
)


_SPACE_PUNCT_RE = re.compile(r"[\s、，。,/&（）()\-_]+")


def normalize_label(value: str) -> str:
    normalized = _SPACE_PUNCT_RE.sub("", value or "").lower()
    # Keep only spelling-level normalization here. Product-family rewrites (for
    # example treating every short-sleeve item as a T-shirt) silently bake a
    # benchmark taxonomy into the resolver and can prevent the correct unseen
    # leaf from ever reaching the model classifier.
    replacements = (
        ("女式", "女士"),
        ("男式", "男士"),
        ("t恤衫", "t恤"),
        ("聚酯纤维", "涤纶"),
        ("图案花纹", "图案"),
        ("材质", "面料成分"),
    )
    for source, target in replacements:
        normalized = normalized.replace(source, target)
    return normalized


def _flatten_category_tree(data: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("isLeaf") is True:
            result.append(
                {
                    "category_id": str(node.get("catId", "")),
                    "name": str(node.get("name", "")),
                    "path": str(node.get("categoryPath", "")),
                }
            )
        for child in node.get("children") or []:
            visit(child)

    for category in data.get("categories") or []:
        visit(category)
    return result


def _metadata_categories(data: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in data.get("categories") or []:
        if not isinstance(raw, dict):
            continue
        category_id = raw.get("categoryId")
        if category_id in (None, ""):
            continue
        result.append(
            {
                "category_id": str(category_id),
                "name": str(
                    raw.get("categoryName")
                    or raw.get("nameChinese")
                    or raw.get("name")
                    or ""
                ),
                "name_chinese": str(raw.get("nameChinese") or ""),
                "path": str(raw.get("categoryPath") or ""),
                "raw": raw,
            }
        )
    return result


def category_leaf_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every selectable leaf from the supplied taxonomy snapshot."""

    return _flatten_category_tree(data)


def attribute_schema_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact, model-safe metadata rows without benchmark-specific IDs."""

    result: list[dict[str, Any]] = []
    for item in _metadata_categories(data):
        config = (item.get("raw") or {}).get("categoryMetadata") or {}

        def names(field: str) -> list[str]:
            return [
                str(definition.get("attributeNameAlias") or definition.get("name") or "")
                for definition in config.get(field) or []
                if isinstance(definition, dict)
            ]

        result.append(
            {
                "category_id": item["category_id"],
                "name": item["name"],
                "name_chinese": item["name_chinese"],
                "path": item["path"],
                "product_attributes": names("categoryProductAttrList"),
                "sales_attributes": names("categorySaleAttrList"),
            }
        )
    return result


def attribute_schema_definition(
    data: dict[str, Any], schema_category_id: str
) -> dict[str, Any]:
    """Return one complete schema in a compact form suitable for model mapping."""

    selected = _find_category(schema_category_id, _metadata_categories(data))
    if not selected:
        return {}
    config = ((selected.get("raw") or {}).get("categoryMetadata") or {})

    def simplify(definitions: Any, *, sales: bool) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for definition in definitions or []:
            if not isinstance(definition, dict):
                continue
            result.append(
                {
                    "scope": "sales" if sales else "product",
                    "attr_id": str(definition.get("attrId") or ""),
                    "name": str(
                        definition.get("attributeNameAlias")
                        or definition.get("name")
                        or ""
                    ),
                    "required": bool(definition.get("isMandatory")),
                    "multiple": bool(definition.get("isMultipleSelected")),
                    "values": [
                        {
                            "value_id": str(value.get("id") or ""),
                            "name": str(
                                value.get("valueNameAlias")
                                or value.get("name")
                                or ""
                            ),
                        }
                        for value in definition.get("values") or []
                        if isinstance(value, dict)
                    ],
                }
            )
        return result

    return {
        "category_id": schema_category_id,
        "attributes": [
            *simplify(config.get("categoryProductAttrList"), sales=False),
            *simplify(config.get("categorySaleAttrList"), sales=True),
        ],
    }


def _char_overlap(left: str, right: str) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _category_score(facts: ProductFacts, candidate: dict[str, Any]) -> float:
    """Generic lexical recall score; the model makes the final online decision.

    The score deliberately contains no product-family, audience, category-ID or
    benchmark-value branches. Its job is to provide an offline fallback and a
    diagnostic ordering, not to encode the correct leaf answer.
    """

    source = normalize_label(facts.source_category_name)
    title = normalize_label(facts.source_title)
    name = normalize_label(
        candidate.get("name", "") + candidate.get("name_chinese", "")
    )
    path = normalize_label(candidate.get("path", ""))
    evidence = normalize_label(
        " ".join(
            [
                facts.source_category_name,
                facts.source_title,
                *[f"{item.name}:{item.value}" for item in facts.attributes],
            ]
        )
    )
    score = SequenceMatcher(None, source, name).ratio() * 45
    score += _char_overlap(source, name) * 25
    score += _char_overlap(source, path) * 15
    score += _char_overlap(title, name) * 10
    score += _char_overlap(evidence, name + path) * 5
    if source and source == name:
        score += 30
    elif source and (source in name or name in source):
        score += 15
    return score


def _best_candidates(
    facts: ProductFacts, categories: Iterable[dict[str, Any]], limit: int = 12
) -> list[dict[str, Any]]:
    scored = []
    for category in categories:
        item = dict(category)
        item["score"] = round(_category_score(facts, category), 3)
        scored.append(item)
    scored.sort(key=lambda item: item["score"], reverse=True)
    return [
        {key: value for key, value in item.items() if key != "raw"}
        for item in scored[:limit]
    ]


def _find_category(
    category_id: str, categories: Iterable[dict[str, Any]]
) -> dict[str, Any] | None:
    return next(
        (item for item in categories if item.get("category_id") == category_id), None
    )


def _category_parent_map(data: dict[str, Any]) -> dict[str, str]:
    parents: dict[str, str] = {}

    def visit(node: Any, parent_id: str = "") -> None:
        if not isinstance(node, dict):
            return
        node_id = str(node.get("catId") or "")
        if node_id and parent_id:
            parents[node_id] = parent_id
        for child in node.get("children") or []:
            visit(child, node_id)

    for category in data.get("categories") or []:
        visit(category)
    return parents


def _name_match_score(source_name: str, platform_alias: str) -> float:
    source = normalize_label(source_name)
    platform = normalize_label(platform_alias)
    if not source or not platform:
        return 0.0
    if "含量" in source and "含量" not in platform:
        return 0.0
    if source.endswith("尺码") and platform.endswith("尺码类型"):
        return 0.0
    if source == platform:
        return 1.0
    if source in platform or platform in source:
        return 0.8
    return SequenceMatcher(None, source, platform).ratio() * 0.55


def _value_match(source_value: str, values: list[dict[str, Any]]) -> tuple[str, str]:
    if not values:
        return "", source_value
    source = normalize_label(source_value)
    best: tuple[float, str, str] = (0.0, "", "")
    for value in values:
        alias = str(value.get("valueNameAlias") or "")
        name = str(value.get("name") or "")
        for candidate in (alias, name):
            normalized = normalize_label(candidate)
            if not normalized:
                continue
            if source == normalized:
                score = 1.0
            elif min(len(source), len(normalized)) >= 2 and (
                source in normalized or normalized in source
            ):
                score = 0.88
            else:
                score = SequenceMatcher(None, source, normalized).ratio() * 0.65
            if score > best[0]:
                best = (score, str(value.get("id") or ""), alias or name)
    if best[0] >= 0.57:
        return best[1], best[2]
    return "", source_value


def _multiple_value_matches(
    source_value: str, values: list[dict[str, Any]]
) -> list[tuple[str, str]]:
    """Return every enum value explicitly represented by a compound source value."""

    source = normalize_label(source_value)
    matches: list[tuple[str, str]] = []
    for value in values:
        alias = str(value.get("valueNameAlias") or "")
        name = str(value.get("name") or "")
        candidates = [normalize_label(item) for item in (alias, name) if item]
        represented = any(
            len(candidate) >= 2 and candidate in source for candidate in candidates
        )
        pair = (str(value.get("id") or ""), alias or name)
        if represented and pair[0] and pair not in matches:
            matches.append(pair)
    if matches:
        return matches
    value_id, platform_value = _value_match(source_value, values)
    return [(value_id, platform_value)] if value_id else []


def _map_attribute_group(
    facts: ProductFacts,
    definitions: list[dict[str, Any]],
    *,
    sales: bool,
) -> tuple[list[MappedAttribute], list[str]]:
    mapped: list[MappedAttribute] = []
    missing: list[str] = []
    for definition in definitions:
        if not isinstance(definition, dict):
            continue
        alias = str(
            definition.get("attributeNameAlias") or definition.get("name") or ""
        )
        required = bool(definition.get("isMandatory"))
        source_attributes = list(facts.attributes)
        matches = sorted(
            ((_name_match_score(item.name, alias), item) for item in source_attributes),
            key=lambda pair: pair[0],
            reverse=True,
        )
        accepted = [item for score, item in matches if score >= 0.72]

        mapped_before = len(mapped)
        if sales and ("颜色" in alias or "尺码" in alias):
            sku_values: list[tuple[str, str, str]] = []
            seen_sku_values: set[tuple[str, str]] = set()
            for sku in facts.skus:
                for sku_attr in sku.attributes:
                    if _name_match_score(sku_attr.name, alias) >= 0.72:
                        key = (sku_attr.name, sku_attr.value)
                        if key not in seen_sku_values:
                            seen_sku_values.add(key)
                            sku_values.append(
                                (
                                    sku_attr.name,
                                    sku_attr.value,
                                    sku_attr.evidence_pointer,
                                )
                            )
            if sku_values:
                accepted = []
                for source_name, source_value, evidence_pointer in sku_values:
                    value_id, platform_value = _value_match(
                        source_value, definition.get("values") or []
                    )
                    mapped.append(
                        MappedAttribute(
                            attr_id=str(definition.get("attrId") or ""),
                            name=alias,
                            source_name=source_name,
                            source_value=source_value,
                            source_evidence_pointer=evidence_pointer,
                            value_id=value_id,
                            platform_value=platform_value,
                            required=required,
                            sales_attribute=True,
                        )
                    )
                continue

        if not accepted:
            if required:
                missing.append(alias)
            continue
        multiple = bool(definition.get("isMultipleSelected"))
        for item in accepted:
            values = definition.get("values") or []
            resolved_values = (
                _multiple_value_matches(item.value, values)
                if multiple and values
                else [_value_match(item.value, values)]
            )
            for value_id, platform_value in resolved_values:
                if values and not value_id:
                    continue
                mapped.append(
                    MappedAttribute(
                        attr_id=str(definition.get("attrId") or ""),
                        name=alias,
                        source_name=item.name,
                        source_value=item.value,
                        source_evidence_pointer=item.evidence_pointer,
                        value_id=value_id,
                        platform_value=platform_value,
                        required=required,
                        sales_attribute=sales,
                    )
                )
            if not multiple and len(mapped) > mapped_before:
                break
        if required and len(mapped) == mapped_before and alias not in missing:
            missing.append(alias)
    return mapped, missing


def apply_model_attribute_mappings(
    facts: ProductFacts,
    taxonomy: TaxonomyResult,
    attribute_data: dict[str, Any],
    decisions: Any,
) -> TaxonomyResult:
    """Install only model mappings that are fully grounded in supplied IDs and facts.

    The model performs semantic matching, while this function owns authority:
    invented attributes, enum values, source names, and source values are rejected.
    An empty or malformed response leaves the deterministic fallback untouched.
    """

    schema = attribute_schema_definition(
        attribute_data, taxonomy.attribute_schema_category_id
    )
    definitions = {
        (item["scope"], item["attr_id"]): item
        for item in schema.get("attributes") or []
        if item.get("attr_id")
    }
    source_rows: dict[tuple[str, str, str], str] = {}
    for index, item in enumerate(facts.attributes):
        source_rows.setdefault(
            ("product", item.name, item.value),
            item.evidence_pointer or f"attributes[{index}]",
        )
    for sku_index, sku in enumerate(facts.skus):
        for attribute_index, item in enumerate(sku.attributes):
            source_rows.setdefault(
                ("sku", item.name, item.value),
                item.evidence_pointer
                or sku.evidence_pointer
                or f"skus[{sku_index}].attributes[{attribute_index}]",
            )
    canonical_claims = facts.reconciled_fact_ledger.get("canonical_visual_claims", [])
    for index, item in enumerate(canonical_claims if isinstance(canonical_claims, list) else []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("concept") or "visible_design_feature")
        value = str(item.get("value") or "")
        if value:
            source_rows.setdefault(
                ("canonical", name, value),
                f"reconciled_fact_ledger.canonical_visual_claims[{index}]",
            )

    if not isinstance(decisions, list):
        return taxonomy
    validated: list[MappedAttribute] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for row in decisions:
        if not isinstance(row, dict):
            continue
        scope = str(row.get("scope") or "")
        attr_id = str(row.get("platform_attr_id") or "")
        definition = definitions.get((scope, attr_id))
        source_kind = str(row.get("source_kind") or "")
        source_name = str(row.get("source_name") or "")
        source_value = str(row.get("source_value") or "")
        evidence_pointer = source_rows.get(
            (source_kind, source_name, source_value), ""
        )
        if not definition or not evidence_pointer:
            continue
        values = {
            str(value.get("value_id") or ""): str(value.get("name") or "")
            for value in definition.get("values") or []
            if value.get("value_id")
        }
        value_id = str(row.get("platform_value_id") or "")
        if values and value_id not in values:
            continue
        if not values:
            value_id = ""
        key = (scope, attr_id, value_id, source_name, source_value)
        if key in seen:
            continue
        seen.add(key)
        validated.append(
            MappedAttribute(
                attr_id=attr_id,
                name=str(definition.get("name") or ""),
                source_name=source_name,
                source_value=source_value,
                source_evidence_pointer=evidence_pointer,
                value_id=value_id,
                platform_value=values.get(value_id, source_value),
                required=bool(definition.get("required")),
                sales_attribute=scope == "sales",
            )
        )
    if not validated:
        return taxonomy
    present = {
        ("sales" if item.sales_attribute else "product", item.attr_id)
        for item in validated
    }
    missing_required = [
        str(item.get("name") or "")
        for item in schema.get("attributes") or []
        if item.get("required")
        and (str(item.get("scope") or ""), str(item.get("attr_id") or ""))
        not in present
    ]
    return TaxonomyResult(
        category=taxonomy.category,
        attributes=validated,
        missing_required=missing_required,
        attribute_schema_category_id=taxonomy.attribute_schema_category_id,
    )


def resolve_taxonomy(
    facts: ProductFacts,
    category_tree: dict[str, Any],
    attribute_data: dict[str, Any],
    *,
    preferred_category_id: str = "",
    preferred_attribute_schema_id: str = "",
) -> TaxonomyResult:
    leaves = _flatten_category_tree(category_tree)
    metadata = _metadata_categories(attribute_data)
    # Category accuracy is scored against leaf answers. Metadata rows often
    # describe parent schemas, so they may inform attribute mapping below but
    # must never enter the selectable category candidate set.
    candidates = _best_candidates(facts, leaves)

    selected = _find_category(preferred_category_id, leaves)
    if selected:
        choice = CategoryChoice(
            category_id=selected["category_id"],
            name=selected["name"],
            path=selected["path"],
            confidence=0.92,
            method="model-constrained-all-leaves",
            candidates=candidates,
        )
    elif candidates:
        top = candidates[0]
        second_score = (
            float(candidates[1]["score"]) if len(candidates) > 1 else float(top["score"])
        )
        margin = max(0.0, float(top["score"]) - second_score)
        absolute = min(1.0, max(0.0, float(top["score"]) / 125.0))
        separation = min(1.0, margin / 30.0)
        confidence = min(0.94, max(0.35, 0.35 + absolute * 0.4 + separation * 0.25))
        choice = CategoryChoice(
            category_id=str(top["category_id"]),
            name=str(top["name"]),
            path=str(top["path"]),
            confidence=confidence,
            method="local-lexical-ranking",
            candidates=candidates,
        )
    else:
        choice = CategoryChoice(
            category_id=facts.source_category_id,
            name=facts.source_category_name,
            path=facts.source_category_name,
            confidence=0.2,
            method="source-fallback",
            candidates=[],
        )

    selected_metadata = _find_category(preferred_attribute_schema_id, metadata)
    if not selected_metadata:
        selected_metadata = _find_category(choice.category_id, metadata)
    if not selected_metadata:
        parents = _category_parent_map(category_tree)
        ancestor_id = parents.get(choice.category_id, "")
        while ancestor_id and not selected_metadata:
            selected_metadata = _find_category(ancestor_id, metadata)
            ancestor_id = parents.get(ancestor_id, "")
    if not selected_metadata and metadata:
        # Some taxonomy snapshots omit metadata for individual leaves. Select a
        # schema generically by label/path similarity instead of maintaining a
        # benchmark category-ID substitution table. In full mode the model may
        # supply preferred_attribute_schema_id and takes precedence over this
        # conservative offline fallback.
        target = f"{choice.name} {choice.path}"
        ranked_metadata = sorted(
            metadata,
            key=lambda item: max(
                SequenceMatcher(
                    None,
                    normalize_label(target),
                    normalize_label(
                        f"{item.get('name', '')} {item.get('name_chinese', '')} {item.get('path', '')}"
                    ),
                ).ratio(),
                _char_overlap(
                    normalize_label(target),
                    normalize_label(
                        f"{item.get('name', '')} {item.get('name_chinese', '')} {item.get('path', '')}"
                    ),
                ),
            ),
            reverse=True,
        )
        selected_metadata = ranked_metadata[0]
    if not selected_metadata:
        return TaxonomyResult(category=choice)

    raw = selected_metadata.get("raw") or {}
    config = raw.get("categoryMetadata") or {}
    product_mapped, product_missing = _map_attribute_group(
        facts, config.get("categoryProductAttrList") or [], sales=False
    )
    sale_mapped, sale_missing = _map_attribute_group(
        facts, config.get("categorySaleAttrList") or [], sales=True
    )
    unique_mapped: list[MappedAttribute] = []
    seen_mappings: set[tuple[str, str, str, bool]] = set()
    for item in product_mapped + sale_mapped:
        key = (
            item.attr_id,
            item.value_id,
            normalize_label(item.platform_value),
            item.sales_attribute,
        )
        if key not in seen_mappings:
            seen_mappings.add(key)
            unique_mapped.append(item)
    return TaxonomyResult(
        category=choice,
        attributes=unique_mapped,
        missing_required=product_missing + sale_missing,
        attribute_schema_category_id=str(selected_metadata.get("category_id") or ""),
    )
