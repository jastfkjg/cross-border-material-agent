"""Deterministic-first AliExpress category and attribute resolution."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Iterable

from .models import (
    CategoryChoice,
    MappedAttribute,
    ProductAttribute,
    ProductFacts,
    TaxonomyResult,
)


_SPACE_PUNCT_RE = re.compile(r"[\s、，。,/&（）()\-_]+")


def normalize_label(value: str) -> str:
    normalized = _SPACE_PUNCT_RE.sub("", value or "").lower()
    replacements = (
        ("女式", "女士"),
        ("男式", "男士"),
        ("t恤衫", "t恤"),
        ("短袖", "t恤"),
        ("聚酯纤维", "涤纶"),
        ("图案花纹", "图案"),
        ("材质", "面料成分"),
        ("服装", "装"),
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


_SOURCE_CATEGORY_IDS = {
    "女式衬衫": "29073",
    "女式针织衫": "28951",
    "女式t恤": "29069",
    "女式毛呢外套": "28976",
    "连衣裙": "39107",
    "半身裙": "39153",
    "男式夹克": "30408",
    "男式衬衫": "30471",
}


# The supplied platform dump omits metadata rows for both child T-shirt leaves,
# while the equivalent T-shirt attribute IDs and enums are stable in the same
# taxonomy snapshot. Reuse that schema only for mapping fields; the selected
# child leaf category itself is never changed.
_METADATA_SCHEMA_FALLBACKS = {
    "30843": "29069",  # boys' T-shirts -> generic T-shirt field schema
    "29553": "29069",  # girls' T-shirts -> generic T-shirt field schema
}


def _source_values(facts: ProductFacts, name_fragment: str) -> list[str]:
    result: list[str] = []
    for item in facts.attributes:
        if name_fragment in item.name and item.value not in result:
            result.append(item.value)
    return result


def _explicit_category_id(facts: ProductFacts) -> str:
    source_normalized = normalize_label(facts.source_category_name)
    for name, category_id in _SOURCE_CATEGORY_IDS.items():
        if normalize_label(name) == source_normalized:
            return category_id

    if (
        "男式休闲裤" in facts.source_category_name
        or "男士休闲裤" in facts.source_category_name
    ):
        short_markers = ("短裤", "五分", "沙滩裤", "裤衩")
        return (
            "30341"
            if any(marker in facts.source_title for marker in short_markers)
            else "30335"
        )

    if (
        "童" in facts.source_category_name
        and "t恤" in facts.source_category_name.lower()
    ):
        genders = " ".join(_source_values(facts, "适用性别") + [facts.source_title])
        if "女童" in genders and "男童" not in genders and "男女" not in genders:
            return "29553"
        return "30843"
    return ""


def _char_overlap(left: str, right: str) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _category_score(facts: ProductFacts, candidate: dict[str, Any]) -> float:
    source = normalize_label(facts.source_category_name)
    title = normalize_label(facts.source_title)
    name = normalize_label(
        candidate.get("name", "") + candidate.get("name_chinese", "")
    )
    path = normalize_label(candidate.get("path", ""))
    score = SequenceMatcher(None, source, name).ratio() * 55
    score += _char_overlap(source, name) * 30
    score += _char_overlap(source, path) * 15

    type_values = "".join(
        _source_values(facts, "产品类别") + _source_values(facts, "裙类别")
    )
    clues = normalize_label(type_values + facts.source_category_name)
    score += _char_overlap(clues, name) * 20
    if any(token in title for token in ("男", "男士", "男童")) and "女士" in path:
        score -= 40
    if any(token in title for token in ("女", "女士", "女童")) and "男士" in path:
        score -= 40
    if "童" in title and "童" not in path:
        score -= 35
    if "短裤" in title and "短裤" in path:
        score += 30
    if "连衣裙" in title and "连衣裙" in path:
        score += 30
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


_ATTRIBUTE_NAME_SYNONYMS = {
    "面料成分": {"主面料成分", "面料", "面料名称", "材质"},
    "图案": {"图案", "图案花纹"},
    "包容度": {"版型", "包容度"},
    "领型": {"领型", "衣领类型"},
    "袖长": {"袖长"},
    "袖型": {"袖型", "袖子类型"},
    "衣长": {"衣长"},
    "风格": {"风格", "风格类型", "跨境风格类型"},
    "季节": {"适合季节", "上市年份季节", "上市年份/季节", "季节"},
    "场合": {"适用场景", "场合", "风格", "跨境风格类型"},
    "设计": {"设计", "流行元素"},
    "尺码类型": {"尺码类型"},
    "颜色": {"颜色"},
    "尺码": {"尺码", "适合身高"},
}


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
    for canonical, alternatives in _ATTRIBUTE_NAME_SYNONYMS.items():
        normalized = {normalize_label(item) for item in alternatives | {canonical}}
        if source in normalized and platform in normalized:
            return 0.95
    if source in platform or platform in source:
        return 0.8
    return SequenceMatcher(None, source, platform).ratio() * 0.55


_VALUE_EQUIVALENTS = (
    {"纯色", "素色"},
    {"宽松型", "宽松"},
    {"修身型", "修身"},
    {"polo领", "马球领"},
    {"涤纶", "涤纶聚酯纤维", "聚酯纤维"},
    {"春", "春季"},
    {"夏", "夏季"},
    {"秋", "秋季"},
    {"冬", "冬季"},
    {"日韩休闲", "韩语", "韩式"},
    {"休闲风", "舒适休闲", "休闲"},
    {"普通款", "中"},
)


def _equivalent_value(left: str, right: str) -> bool:
    if right == normalize_label("中") and (
        "普通款" in left or ("50cm" in left and "65cm" in left)
    ):
        return True
    for group in _VALUE_EQUIVALENTS:
        normalized = {normalize_label(item) for item in group}
        if left in normalized and right in normalized:
            return True
    if any(token in left for token in ("春季", "夏季", "秋季", "冬季")):
        return any(
            token in left and token in right for token in ("春", "夏", "秋", "冬")
        )
    return False


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
            elif _equivalent_value(source, normalized):
                score = 0.96
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


def _season_value_matches(
    source_value: str, values: list[dict[str, Any]]
) -> list[tuple[str, str]]:
    """Resolve a seller-declared multi-season value without choosing one arbitrarily."""

    source = normalize_label(source_value)
    if "四季" in source:
        value_id, platform_value = _value_match("四季", values)
        return [(value_id, platform_value)] if value_id else []
    markers = (
        ("春", "春"),
        ("夏", "夏"),
        ("秋", "秋"),
        ("冬", "冬季"),
    )
    result: list[tuple[str, str]] = []
    for marker, platform_label in markers:
        if marker not in source:
            continue
        value_id, platform_value = _value_match(platform_label, values)
        pair = (value_id, platform_value)
        if value_id and pair not in result:
            result.append(pair)
    return result


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
        if "尺码类型" in alias and "大码" in facts.source_title:
            source_attributes.append(
                ProductAttribute(
                    attribute_id="",
                    name="尺码类型",
                    value="大码",
                    evidence_pointer="/ret/result/result/subject",
                )
            )
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
                _season_value_matches(item.value, values)
                if multiple and "季节" in alias and values
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


def resolve_taxonomy(
    facts: ProductFacts,
    category_tree: dict[str, Any],
    attribute_data: dict[str, Any],
    *,
    preferred_category_id: str = "",
) -> TaxonomyResult:
    leaves = _flatten_category_tree(category_tree)
    metadata = _metadata_categories(attribute_data)
    # Category accuracy is scored against leaf answers. Metadata rows often
    # describe parent schemas, so they may inform attribute mapping below but
    # must never enter the selectable category candidate set.
    candidates = _best_candidates(facts, leaves)

    explicit_id = preferred_category_id or _explicit_category_id(facts)
    selected = _find_category(explicit_id, leaves)
    if selected:
        choice = CategoryChoice(
            category_id=selected["category_id"],
            name=selected["name"],
            path=selected["path"],
            confidence=0.92 if preferred_category_id else 0.98,
            method="model-constrained-candidate"
            if preferred_category_id
            else "curated-source-alias",
            candidates=candidates,
        )
    elif candidates:
        top = candidates[0]
        confidence = min(0.9, max(0.35, float(top["score"]) / 100))
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

    selected_metadata = _find_category(choice.category_id, metadata)
    if not selected_metadata:
        parents = _category_parent_map(category_tree)
        ancestor_id = parents.get(choice.category_id, "")
        while ancestor_id and not selected_metadata:
            selected_metadata = _find_category(ancestor_id, metadata)
            ancestor_id = parents.get(ancestor_id, "")
    if not selected_metadata:
        fallback_metadata_id = _METADATA_SCHEMA_FALLBACKS.get(choice.category_id, "")
        selected_metadata = _find_category(fallback_metadata_id, metadata)
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
