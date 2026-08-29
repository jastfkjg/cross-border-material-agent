"""Model-directed, read-only exploration of marketplace taxonomy snapshots.

The model owns semantic decisions.  This module only exposes bounded equivalents
of ``read``/``rg`` for the category tree and attribute metadata, then validates
the IDs returned by the model against the supplied snapshots and source facts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .api import ApiError
from .models import CategoryChoice, ProductFacts, TaxonomyResult
from .taxonomy import (
    apply_model_attribute_mappings,
    attribute_schema_candidates,
    attribute_schema_definition,
)


_SEARCH_NORMALIZE_RE = re.compile(r"[\s、，。,./\\|（）()\-_]+")


class TaxonomyAgentError(ApiError):
    """The exploration loop ended without a grounded final selection."""

    def __init__(self, message: str):
        super().__init__(message, retryable=True, category="taxonomy_agent")


def _search_text(value: Any) -> str:
    return _SEARCH_NORMALIZE_RE.sub("", str(value or "")).casefold()


def _bounded_int(value: Any, default: int, *, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


@dataclass(slots=True)
class _CategoryNode:
    category_id: str
    name: str
    path: str
    parent_id: str
    is_leaf: bool
    child_ids: list[str]

    def compact(self) -> dict[str, Any]:
        return {
            "category_id": self.category_id,
            "name": self.name,
            "path": self.path,
            "parent_id": self.parent_id,
            "is_leaf": self.is_leaf,
            "child_count": len(self.child_ids),
        }


class TaxonomyExplorer:
    """Generic, bounded, read-only queries over normalized taxonomy records."""

    MAX_OUTPUT_BYTES = 50 * 1024
    COLLECTIONS = ("categories", "schemas", "attributes", "values")

    CATEGORY_TOOL_NAMES = (
        "query_taxonomy",
        "read_taxonomy",
        "finish_category",
    )
    ATTRIBUTE_TOOL_NAMES = (
        "query_taxonomy",
        "read_taxonomy",
        "finish_attributes",
    )

    TOOL_CATALOG = (
        {
            "name": "query_taxonomy",
            "description": (
                "Batch-query normalized read-only collections: categories, schemas, attributes, and values. "
                "Each request chooses filters, AND/OR matching, projection, sorting, and pagination."
            ),
        },
        {
            "name": "read_taxonomy",
            "description": (
                "Read exact stable refs returned by query_taxonomy. Schema refs include attribute summaries; "
                "attribute refs include a bounded page of enum values."
            ),
        },
        {
            "name": "finish_category",
            "description": (
                "Finish the category phase with selected_category_id, confidence, and evidence. "
                "The selected category must be an observed leaf."
            ),
        },
        {
            "name": "finish_attributes",
            "description": (
                "Finish the attribute phase with selected_attribute_schema_category_id and mappings. "
                "The schema/attribute/value IDs used in mappings must have been observed first."
            ),
        },
    )

    @classmethod
    def openai_tools(cls, names: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """Return native function-tool definitions for OpenAI-compatible APIs."""

        def tool(
            name: str,
            description: str,
            properties: dict[str, Any],
            required: list[str] | None = None,
        ) -> dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required or [],
                        "additionalProperties": False,
                    },
                },
            }

        by_name = {item["name"]: item["description"] for item in cls.TOOL_CATALOG}
        mapping_properties = {
            "scope": {"type": "string", "enum": ["product", "sales"]},
            "platform_attr_id": {"type": "string"},
            "platform_value_id": {"type": "string"},
            "source_kind": {"type": "string", "enum": ["product", "sku"]},
            "source_name": {"type": "string"},
            "source_value": {"type": "string"},
        }
        filter_properties = {
            "field": {
                "type": "string",
                "description": "A field returned by the selected collection.",
            },
            "op": {
                "type": "string",
                "enum": [
                    "eq",
                    "neq",
                    "contains",
                    "not_contains",
                    "contains_any",
                    "contains_all",
                    "in",
                    "exists",
                    "gt",
                    "gte",
                    "lt",
                    "lte",
                ],
            },
            "value": {
                "description": "A string, number, boolean, or array used by the operator."
            },
        }
        query_request_properties = {
            "collection": {"type": "string", "enum": list(cls.COLLECTIONS)},
            "filters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": filter_properties,
                    "required": ["field", "op"],
                    "additionalProperties": False,
                },
                "maxItems": 12,
            },
            "match": {"type": "string", "enum": ["all", "any"]},
            "fields": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 24,
            },
            "sort_by": {"type": "string"},
            "descending": {"type": "boolean"},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 60},
        }
        definitions = [
            tool(
                "query_taxonomy",
                by_name["query_taxonomy"],
                {
                    "requests": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": query_request_properties,
                            "required": ["collection"],
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                        "maxItems": 8,
                    }
                },
                ["requests"],
            ),
            tool(
                "read_taxonomy",
                by_name["read_taxonomy"],
                {
                    "refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 12,
                    },
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 60},
                },
                ["refs"],
            ),
            tool(
                "finish_category",
                by_name["finish_category"],
                {
                    "selected_category_id": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string"},
                },
                ["selected_category_id", "confidence", "evidence"],
            ),
            tool(
                "finish_attributes",
                by_name["finish_attributes"],
                {
                    "selected_attribute_schema_category_id": {"type": "string"},
                    "mappings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": mapping_properties,
                            "required": list(mapping_properties),
                            "additionalProperties": False,
                        },
                    },
                },
                [
                    "selected_attribute_schema_category_id",
                    "mappings",
                ],
            ),
        ]
        if names is None:
            return definitions
        allowed = set(names)
        return [item for item in definitions if item["function"]["name"] in allowed]

    def __init__(self, category_tree: dict[str, Any], attribute_data: dict[str, Any]):
        self.category_tree = category_tree
        self.attribute_data = attribute_data
        self.nodes: dict[str, _CategoryNode] = {}
        self._tree_order: list[str] = []
        self._build_tree()
        self.schemas = {
            str(item["category_id"]): item
            for item in attribute_schema_candidates(attribute_data)
            if item.get("category_id")
        }
        self._schema_order = list(self.schemas)
        self._schema_definitions = {
            schema_id: attribute_schema_definition(attribute_data, schema_id)
            for schema_id in self._schema_order
        }
        self._records: dict[str, list[dict[str, Any]]] = {
            collection: [] for collection in self.COLLECTIONS
        }
        self._refs: dict[str, dict[str, Any]] = {}
        self._build_records()

        # Grounding ledger: final IDs must first have appeared in an observation.
        self.exposed_category_ids: set[str] = set()
        self.inspected_schema_ids: set[str] = set()
        self.inspected_attribute_ids: set[tuple[str, str]] = set()
        self.exposed_value_ids: dict[tuple[str, str], set[str]] = {}

    def _build_tree(self) -> None:
        def visit(raw: Any, parent_id: str = "") -> None:
            if not isinstance(raw, dict):
                return
            category_id = str(raw.get("catId") or "")
            if not category_id:
                return
            children = [
                str(child.get("catId") or "")
                for child in raw.get("children") or []
                if isinstance(child, dict) and child.get("catId") not in (None, "")
            ]
            self.nodes[category_id] = _CategoryNode(
                category_id=category_id,
                name=str(raw.get("name") or ""),
                path=str(raw.get("categoryPath") or ""),
                parent_id=parent_id,
                is_leaf=raw.get("isLeaf") is True,
                child_ids=children,
            )
            self._tree_order.append(category_id)
            for child in raw.get("children") or []:
                visit(child, category_id)

        for raw in self.category_tree.get("categories") or []:
            visit(raw)

    def _ancestor_ids(self, category_id: str) -> list[str]:
        result: list[str] = []
        current = self.nodes.get(category_id)
        while current is not None and current.parent_id:
            result.append(current.parent_id)
            current = self.nodes.get(current.parent_id)
        return result

    def _append_record(self, collection: str, record: dict[str, Any]) -> None:
        self._records[collection].append(record)
        self._refs[str(record["ref"])] = record

    def _build_records(self) -> None:
        for category_id in self._tree_order:
            node = self.nodes[category_id]
            ancestor_ids = self._ancestor_ids(category_id)
            self._append_record(
                "categories",
                {
                    "ref": f"categories/{category_id}",
                    **node.compact(),
                    "child_ids": list(node.child_ids),
                    "ancestor_ids": ancestor_ids,
                    "available_schema_ids": [
                        item
                        for item in [category_id, *ancestor_ids]
                        if item in self.schemas
                    ],
                },
            )

        for schema_id in self._schema_order:
            schema = self.schemas[schema_id]
            definition = self._schema_definitions[schema_id]
            attributes = list(definition.get("attributes") or [])
            self._append_record(
                "schemas",
                {
                    "ref": f"schemas/{schema_id}",
                    "schema_category_id": schema_id,
                    "name": str(
                        schema.get("name") or schema.get("name_chinese") or ""
                    ),
                    "path": str(schema.get("path") or ""),
                    "attribute_count": len(attributes),
                    "product_attribute_count": sum(
                        item.get("scope") == "product" for item in attributes
                    ),
                    "sales_attribute_count": sum(
                        item.get("scope") == "sales" for item in attributes
                    ),
                },
            )
            for attribute in attributes:
                scope = str(attribute.get("scope") or "")
                attr_id = str(attribute.get("attr_id") or "")
                attr_ref = f"attributes/{schema_id}/{scope}/{attr_id}"
                values = list(attribute.get("values") or [])
                self._append_record(
                    "attributes",
                    {
                        "ref": attr_ref,
                        "schema_category_id": schema_id,
                        "scope": scope,
                        "attr_id": attr_id,
                        "name": str(attribute.get("name") or ""),
                        "required": bool(attribute.get("required")),
                        "multiple": bool(attribute.get("multiple")),
                        "value_count": len(values),
                    },
                )
                for value in values:
                    value_id = str(value.get("value_id") or "")
                    self._append_record(
                        "values",
                        {
                            "ref": f"values/{schema_id}/{scope}/{attr_id}/{value_id}",
                            "schema_category_id": schema_id,
                            "scope": scope,
                            "attr_id": attr_id,
                            "attribute_name": str(attribute.get("name") or ""),
                            "value_id": value_id,
                            "name": str(value.get("name") or ""),
                        },
                    )

    @staticmethod
    def _page(items: list[Any], arguments: dict[str, Any]) -> dict[str, Any]:
        offset = _bounded_int(
            arguments.get("offset"), 0, low=0, high=max(0, len(items))
        )
        limit = _bounded_int(arguments.get("limit"), 30, low=1, high=60)
        page = items[offset : offset + limit]
        next_offset = offset + len(page)
        return {
            "total": len(items),
            "offset": offset,
            "next_offset": next_offset if next_offset < len(items) else None,
            "items": page,
        }

    def _collection_fields(self, collection: str) -> list[str]:
        records = self._records.get(collection) or []
        return sorted({key for record in records for key in record})

    @staticmethod
    def _comparable(value: Any) -> Any:
        if isinstance(value, str):
            return _search_text(value)
        return value

    @classmethod
    def _filter_matches(
        cls, record: dict[str, Any], condition: dict[str, Any]
    ) -> bool:
        field = str(condition.get("field") or "")
        operator = str(condition.get("op") or "eq")
        expected = condition.get("value")
        actual = record.get(field)

        if operator == "exists":
            return (field in record and actual not in (None, "", [])) is bool(
                expected if isinstance(expected, bool) else True
            )

        actual_values = actual if isinstance(actual, list) else [actual]
        expected_values = expected if isinstance(expected, list) else [expected]
        comparable_actual = [cls._comparable(item) for item in actual_values]
        comparable_expected = [cls._comparable(item) for item in expected_values]

        if operator == "eq":
            return (
                comparable_expected[0] in comparable_actual
                if isinstance(actual, list)
                else cls._comparable(actual) == cls._comparable(expected)
            )
        if operator == "neq":
            return not cls._filter_matches(
                record, {"field": field, "op": "eq", "value": expected}
            )
        if operator in {"contains", "not_contains"}:
            needle = str(cls._comparable(expected) or "")
            matched = any(
                needle in str(item or "") for item in comparable_actual if needle
            )
            return not matched if operator == "not_contains" else matched
        if operator in {"contains_any", "contains_all"}:
            needles = [str(item or "") for item in comparable_expected if str(item or "")]
            matches = [
                any(needle in str(item or "") for item in comparable_actual)
                for needle in needles
            ]
            return bool(matches) and (
                all(matches) if operator == "contains_all" else any(matches)
            )
        if operator == "in":
            return any(item in comparable_expected for item in comparable_actual)
        if operator in {"gt", "gte", "lt", "lte"}:
            try:
                left = float(actual)
                right = float(expected)
            except (TypeError, ValueError):
                return False
            return {
                "gt": left > right,
                "gte": left >= right,
                "lt": left < right,
                "lte": left <= right,
            }[operator]
        raise ValueError(f"unsupported filter operator: {operator}")

    @staticmethod
    def _identity_fields(collection: str) -> tuple[str, ...]:
        return {
            "categories": ("ref", "category_id"),
            "schemas": ("ref", "schema_category_id"),
            "attributes": ("ref", "schema_category_id", "scope", "attr_id"),
            "values": (
                "ref",
                "schema_category_id",
                "scope",
                "attr_id",
                "value_id",
            ),
        }[collection]

    def _project_record(
        self, collection: str, record: dict[str, Any], raw_fields: Any
    ) -> dict[str, Any]:
        requested = raw_fields if isinstance(raw_fields, list) else []
        fields = list(dict.fromkeys([*self._identity_fields(collection), *requested]))
        if not requested:
            fields = list(record)
        return {field: record[field] for field in fields if field in record}

    def _query_taxonomy(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_requests = arguments.get("requests")
        requests = raw_requests if isinstance(raw_requests, list) else []
        if not requests:
            return self._error("query_taxonomy", "requests must be a non-empty array")

        results: list[dict[str, Any]] = []
        exposure: list[tuple[str, list[dict[str, Any]]]] = []
        for index, raw_request in enumerate(requests[:8]):
            request = raw_request if isinstance(raw_request, dict) else {}
            collection = str(request.get("collection") or "")
            if collection not in self._records:
                results.append(
                    {
                        "request_index": index,
                        "ok": False,
                        "error": f"unknown collection: {collection}",
                        "available_collections": list(self.COLLECTIONS),
                    }
                )
                continue
            fields = self._collection_fields(collection)
            raw_filters = request.get("filters")
            filters = raw_filters if isinstance(raw_filters, list) else []
            if any(not isinstance(item, dict) for item in filters):
                results.append(
                    {
                        "request_index": index,
                        "ok": False,
                        "error": "every filter must be an object",
                    }
                )
                continue
            unknown_fields = sorted(
                {
                    str(item.get("field") or "")
                    for item in filters
                    if isinstance(item, dict)
                    and str(item.get("field") or "") not in fields
                }
            )
            requested_fields = request.get("fields")
            if isinstance(requested_fields, list):
                unknown_fields.extend(
                    str(item) for item in requested_fields if str(item) not in fields
                )
            sort_by = str(request.get("sort_by") or "")
            if sort_by and sort_by not in fields:
                unknown_fields.append(sort_by)
            if unknown_fields:
                results.append(
                    {
                        "request_index": index,
                        "ok": False,
                        "error": f"unknown fields: {sorted(set(unknown_fields))}",
                        "available_fields": fields,
                    }
                )
                continue

            match_any = request.get("match") == "any"
            try:
                matched = [
                    record
                    for record in self._records[collection]
                    if not filters
                    or (any if match_any else all)(
                        self._filter_matches(record, item)
                        for item in filters
                        if isinstance(item, dict)
                    )
                ]
            except ValueError as exc:
                results.append(
                    {"request_index": index, "ok": False, "error": str(exc)}
                )
                continue
            if sort_by:
                matched.sort(
                    key=lambda item: str(item.get(sort_by) or "").casefold(),
                    reverse=bool(request.get("descending")),
                )
            page = self._page(matched, request)
            visible = [
                self._project_record(collection, item, requested_fields)
                for item in page["items"]
            ]
            results.append(
                {
                    "request_index": index,
                    "ok": True,
                    "collection": collection,
                    **{key: value for key, value in page.items() if key != "items"},
                    "items": visible,
                }
            )
            exposure.append((collection, visible))

        result = {
            "collections": {
                collection: self._collection_fields(collection)
                for collection in self.COLLECTIONS
            },
            "results": results,
        }
        if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > self.MAX_OUTPUT_BYTES:
            return self._error(
                "query_taxonomy",
                "query output exceeds 50KB; lower limits, narrow filters, or request fewer fields",
            )
        for collection, items in exposure:
            self._record_exposure(collection, items)
        return self._ok("query_taxonomy", result)

    def _read_taxonomy(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_refs = arguments.get("refs")
        refs = raw_refs if isinstance(raw_refs, list) else []
        if not refs:
            return self._error("read_taxonomy", "refs must be a non-empty array")
        offset = _bounded_int(arguments.get("offset"), 0, low=0, high=10000)
        limit = _bounded_int(arguments.get("limit"), 30, low=1, high=60)
        items: list[dict[str, Any]] = []
        unknown: list[str] = []
        exposure: list[tuple[str, list[dict[str, Any]]]] = []
        for raw_ref in refs[:12]:
            ref = str(raw_ref or "")
            record = self._refs.get(ref)
            if record is None:
                unknown.append(ref)
                continue
            collection = ref.split("/", 1)[0]
            expanded = dict(record)
            if collection == "schemas":
                schema_id = str(record["schema_category_id"])
                related = [
                    item
                    for item in self._records["attributes"]
                    if item["schema_category_id"] == schema_id
                ]
                page = self._page(related, {"offset": offset, "limit": limit})
                expanded["attributes"] = page
                exposure.append(("attributes", page["items"]))
            elif collection == "attributes":
                related = [
                    item
                    for item in self._records["values"]
                    if item["schema_category_id"] == record["schema_category_id"]
                    and item["scope"] == record["scope"]
                    and item["attr_id"] == record["attr_id"]
                ]
                page = self._page(related, {"offset": offset, "limit": limit})
                expanded["values"] = page
                exposure.append(("values", page["items"]))
            items.append(expanded)
            exposure.append((collection, [expanded]))
        result = {"items": items, "unknown_refs": unknown}
        if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > self.MAX_OUTPUT_BYTES:
            return self._error(
                "read_taxonomy",
                "read output exceeds 50KB; lower limit or read fewer refs",
            )
        for collection, visible in exposure:
            self._record_exposure(collection, visible)
        return self._ok("read_taxonomy", result)

    def _record_exposure(
        self, collection: str, items: list[dict[str, Any]]
    ) -> None:
        if collection == "categories":
            self.exposed_category_ids.update(
                str(item.get("category_id") or "")
                for item in items
                if item.get("category_id")
            )
            return
        if collection == "schemas":
            self.inspected_schema_ids.update(
                str(item.get("schema_category_id") or "")
                for item in items
                if item.get("schema_category_id")
            )
            return
        for item in items:
            schema_id = str(item.get("schema_category_id") or "")
            attr_id = str(item.get("attr_id") or "")
            if not schema_id or not attr_id:
                continue
            self.inspected_schema_ids.add(schema_id)
            self.inspected_attribute_ids.add((schema_id, attr_id))
            if collection == "values" and item.get("value_id"):
                self.exposed_value_ids.setdefault((schema_id, attr_id), set()).add(
                    str(item["value_id"])
                )

    def execute(self, tool: str, arguments: Any) -> dict[str, Any]:
        args = arguments if isinstance(arguments, dict) else {}
        try:
            if tool == "query_taxonomy":
                return self._query_taxonomy(args)
            if tool == "read_taxonomy":
                return self._read_taxonomy(args)
            return self._error(tool, f"unknown tool: {tool}")
        except Exception as exc:
            return self._error(tool, f"tool execution failed: {exc}")

    @staticmethod
    def _ok(tool: str, result: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "tool": tool, "result": result}

    @staticmethod
    def _error(tool: str, error: str) -> dict[str, Any]:
        return {"ok": False, "tool": tool, "error": error}

    def finish_category(
        self, arguments: Any
    ) -> tuple[CategoryChoice | None, dict[str, Any]]:
        args = arguments if isinstance(arguments, dict) else {}
        category_id = str(args.get("selected_category_id") or "")
        errors: list[str] = []
        node = self.nodes.get(category_id)
        if category_id not in self.exposed_category_ids:
            errors.append(
                "selected_category_id was not observed through an exploration tool"
            )
        if not node or not node.is_leaf:
            errors.append("selected_category_id must be an existing leaf")

        if errors:
            return None, self._error("finish_category", "; ".join(errors))
        assert node is not None
        try:
            confidence = float(args.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        choice = CategoryChoice(
            category_id=node.category_id,
            name=node.name,
            path=node.path,
            confidence=max(0.0, min(1.0, confidence)),
            method="model-react-exploration",
            candidates=[],
        )
        return choice, self._ok(
            "finish_category",
            {
                "selected_category_id": category_id,
                "evidence": str(args.get("evidence") or "")[:1000],
            },
        )

    def finish_attributes(
        self,
        facts: ProductFacts,
        category: CategoryChoice,
        arguments: Any,
    ) -> tuple[TaxonomyResult | None, dict[str, Any]]:
        args = arguments if isinstance(arguments, dict) else {}
        schema_id = str(args.get("selected_attribute_schema_category_id") or "")
        errors: list[str] = []
        if schema_id not in self.inspected_schema_ids:
            errors.append(
                "selected_attribute_schema_category_id must be inspected first"
            )

        mappings = args.get("mappings")
        if not isinstance(mappings, list):
            errors.append("mappings must be an array")
            mappings = []
        for index, row in enumerate(mappings):
            if not isinstance(row, dict):
                errors.append(f"mapping {index} is not an object")
                continue
            attr_id = str(row.get("platform_attr_id") or "")
            key = (schema_id, attr_id)
            if key not in self.inspected_attribute_ids:
                errors.append(
                    f"mapping {index} uses attribute {attr_id} before inspecting it"
                )
                continue
            value_id = str(row.get("platform_value_id") or "")
            schema = attribute_schema_definition(self.attribute_data, schema_id)
            definition = next(
                (
                    item
                    for item in schema.get("attributes") or []
                    if item.get("attr_id") == attr_id
                ),
                None,
            )
            allowed_values = list((definition or {}).get("values") or [])
            if allowed_values and value_id not in self.exposed_value_ids.get(
                key, set()
            ):
                errors.append(
                    f"mapping {index} uses enum value {value_id} before observing it"
                )
            if not allowed_values and value_id:
                errors.append(
                    f"mapping {index} must leave value ID empty for a free-text attribute"
                )

        if errors:
            return None, self._error("finish_attributes", "; ".join(errors))
        base = TaxonomyResult(
            category=category,
            attributes=[],
            attribute_schema_category_id=schema_id,
        )
        mapped = apply_model_attribute_mappings(
            facts, base, self.attribute_data, mappings
        )
        if mappings and len(mapped.attributes) != len(mappings):
            return None, self._error(
                "finish_attributes",
                "one or more mappings failed source/schema validation; inspect exact source pairs, scopes, and IDs",
            )
        if not mappings:
            schema = attribute_schema_definition(self.attribute_data, schema_id)
            mapped.missing_required = [
                str(item.get("name") or "")
                for item in schema.get("attributes") or []
                if item.get("required")
            ]
        return mapped, self._ok(
            "finish_attributes",
            {
                "selected_category_id": category.category_id,
                "selected_attribute_schema_category_id": schema_id,
                "accepted_mapping_count": len(mapped.attributes),
            },
        )

    def finish(
        self, facts: ProductFacts, arguments: Any
    ) -> tuple[TaxonomyResult | None, dict[str, Any]]:
        """Backward-compatible combined validator used by external callers/tests."""

        category, observation = self.finish_category(arguments)
        if category is None:
            return None, self._error(
                "finish", str(observation.get("error") or "invalid category")
            )
        result, observation = self.finish_attributes(facts, category, arguments)
        if result is None:
            return None, self._error(
                "finish", str(observation.get("error") or "invalid mappings")
            )
        return result, self._ok("finish", observation["result"])


class TaxonomyReActAgent:
    """Run category and attribute exploration with equal, independent budgets."""

    FINISH_RESERVE_TURNS = 2

    def __init__(
        self,
        client: Any,
        category_tree: dict[str, Any],
        attribute_data: dict[str, Any],
        *,
        skill_instructions: str = "",
        trace: Any = None,
        max_turns: int = 50,
    ):
        if max_turns < 4:
            raise ValueError("each taxonomy task requires at least 4 turns")
        self.client = client
        self.explorer = TaxonomyExplorer(category_tree, attribute_data)
        self.skill_instructions = skill_instructions
        self.trace = trace
        self.max_turns = max_turns

    @staticmethod
    def _product_evidence(
        facts: ProductFacts, *, decision_context: str = ""
    ) -> dict[str, Any]:
        sku_options: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for sku in facts.skus:
            for item in sku.attributes:
                key = (item.name, item.value)
                if key not in seen:
                    seen.add(key)
                    sku_options.append({"name": item.name, "value": item.value})
        evidence = {
            "source_title": facts.source_title,
            "source_category_name": facts.source_category_name,
            "attributes": [
                {"name": item.name, "value": item.value} for item in facts.attributes
            ],
            "sku_options": sku_options,
            # This remains evidence, not a host-side routing rule.  The model
            # decides whether a reconciled conflict changes category/schema or
            # only affects a particular publication surface.
            "reconciled_fact_ledger": facts.reconciled_fact_ledger,
        }
        if decision_context:
            evidence["orchestrator_reconsideration_context"] = decision_context[:3000]
        return evidence

    def run(
        self, facts: ProductFacts, *, decision_context: str = ""
    ) -> TaxonomyResult:
        evidence = self._product_evidence(
            facts, decision_context=decision_context
        )
        native_step = getattr(self.client, "chat_tool_step", None)
        if callable(native_step):
            category = self._run_native_category(evidence)
            return self._run_native_attributes(facts, category, evidence)
        category = self._run_json_category(evidence)
        return self._run_json_attributes(facts, category, evidence)

    def _system_prompt(self, phase: str) -> str:
        shared = (
            "You are the reasoning controller of a marketplace taxonomy exploration agent. "
            "You receive product evidence and generic bounded read-only access to normalized taxonomy records, "
            "never the complete snapshots in context. Use query_taxonomy like a safe data-query shell: issue up "
            "to eight independent requests in one call, choosing a collection, filters, match=all/any, fields, "
            "sorting, and pagination. Operators are eq, neq, contains, not_contains, contains_any, contains_all, "
            "in, exists, gt/gte/lt/lte. Use read_taxonomy to expand exact refs returned by a query. Categories "
            "expose parent/child/ancestor IDs and structurally available schema IDs; schemas, attributes, and "
            "values are separate flat collections linked by schema_category_id and attr_id. Field map: categories "
            "have category_id/name/path/parent_id/is_leaf/child_ids/ancestor_ids/available_schema_ids; schemas have "
            "schema_category_id/name/path and attribute counts; attributes have schema_category_id/scope/attr_id/"
            "name/required/multiple/value_count; values have schema_category_id/scope/attr_id/attribute_name/"
            "value_id/name. Semantic decisions are yours; tools only query and validate. Batch independent lookups "
            "instead of spending one turn per term or attribute. Separate alternatives into batched requests when "
            "OR would be too broad, and follow next_offset when later pages remain. Do not guess IDs or use product "
            "identifiers/benchmark memories. Adapt after empty or ambiguous results. Every turn includes a budget "
            "notice. The last two phase turns are reserved "
            "for a validated finish and one correction, so complete prerequisite inspection before that window. "
            "Return one JSON action object only when using the JSON protocol: "
            '{"action": tool name, "arguments": object, "reason": concise evidence note}. '
        )
        if phase == "category":
            phase_rules = (
                "This task selects only the best platform leaf category. Prefer source category, title, and "
                "structured facts. Query multiple useful phrases together and inspect exact refs only when needed. "
                "Call finish_category as soon as a suitable observed leaf is identified; provide "
                "selected_category_id, confidence, and evidence."
            )
        else:
            phase_rules = (
                "This separate task selects the attribute schema and grounded mappings for an already selected "
                "leaf. Query its category record for available_schema_ids, then batch schema, attribute, and value "
                "filters using source names and values. Before finish_attributes, every submitted schema, attribute, "
                "and enum value ID must have appeared in query/read output. Each mapping contains scope, "
                "platform_attr_id, platform_value_id, source_kind, and source_name/source_value copied exactly from "
                "PRODUCT EVIDENCE. Omit uncertain mappings; an empty mapping list is valid."
            )
        return (
            shared
            + phase_rules
            + ("\n\n" + self.skill_instructions if self.skill_instructions else "")
        )

    def _budget_notice(
        self,
        phase: str,
        phase_turn: int,
        phase_budget: int,
    ) -> str:
        phase_remaining = phase_budget - phase_turn + 1
        if phase_remaining <= self.FINISH_RESERVE_TURNS:
            instruction = (
                "FINISH WINDOW: only the phase finish tool is available. Submit the best grounded result now; "
                "if validation rejects it, use the next reserved turn to correct the exact error."
            )
        elif phase_remaining == self.FINISH_RESERVE_TURNS + 1:
            instruction = "LAST EXPLORATION TURN: inspect any missing prerequisite now, then finish on the next turn."
        else:
            instruction = "Finish early once the result is grounded; do not spend turns on optional mappings."
        return (
            f"BUDGET: independent task={phase}; task turn {phase_turn}/{phase_budget}; "
            f"task turns remaining including this one={phase_remaining}. {instruction}"
        )

    def _phase_tools(
        self, names: tuple[str, ...], finish_name: str, remaining: int
    ) -> list[dict[str, Any]]:
        available = self._available_names(names, finish_name, remaining)
        return self.explorer.openai_tools(available)

    @staticmethod
    def _available_names(
        names: tuple[str, ...], finish_name: str, remaining: int
    ) -> tuple[str, ...]:
        return (
            (finish_name,)
            if remaining <= TaxonomyReActAgent.FINISH_RESERVE_TURNS
            else names
        )

    @staticmethod
    def _catalog(
        names: tuple[str, ...], finish_name: str, remaining: int
    ) -> list[dict[str, str]]:
        available = set(
            TaxonomyReActAgent._available_names(names, finish_name, remaining)
        )
        return [
            item for item in TaxonomyExplorer.TOOL_CATALOG if item["name"] in available
        ]

    def _run_native_category(self, evidence: dict[str, Any]) -> CategoryChoice:
        prompt = (
            "CATEGORY TASK: resolve only the best leaf category.\n\n"
            f"PRODUCT EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}"
        )
        result, _ = self._run_native_phase(
            phase="category",
            system=self._system_prompt("category"),
            initial_prompt=prompt,
            tool_names=TaxonomyExplorer.CATEGORY_TOOL_NAMES,
            finish_name="finish_category",
            phase_budget=self.max_turns,
            finish=lambda arguments: self.explorer.finish_category(arguments),
        )
        assert isinstance(result, CategoryChoice)
        return result

    def _run_native_attributes(
        self,
        facts: ProductFacts,
        category: CategoryChoice,
        evidence: dict[str, Any],
    ) -> TaxonomyResult:
        prompt = (
            "ATTRIBUTE TASK: resolve only the schema and grounded attribute mappings for the selected leaf.\n\n"
            f"SELECTED CATEGORY:\n{json.dumps({'category_id': category.category_id, 'name': category.name, 'path': category.path}, ensure_ascii=False)}\n\n"
            f"PRODUCT EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}"
        )
        result, _ = self._run_native_phase(
            phase="attributes",
            system=self._system_prompt("attributes"),
            initial_prompt=prompt,
            tool_names=TaxonomyExplorer.ATTRIBUTE_TOOL_NAMES,
            finish_name="finish_attributes",
            phase_budget=self.max_turns,
            finish=lambda arguments: self.explorer.finish_attributes(
                facts, category, arguments
            ),
        )
        assert isinstance(result, TaxonomyResult)
        return result

    def _run_native_phase(
        self,
        *,
        phase: str,
        system: str,
        initial_prompt: str,
        tool_names: tuple[str, ...],
        finish_name: str,
        phase_budget: int,
        finish: Any,
    ) -> tuple[CategoryChoice | TaxonomyResult, int]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": initial_prompt}]
        repeated_actions: dict[str, int] = {}
        for phase_turn in range(1, phase_budget + 1):
            remaining = phase_budget - phase_turn + 1
            available_names = set(
                self._available_names(tool_names, finish_name, remaining)
            )
            messages.append(
                {
                    "role": "user",
                    "content": self._budget_notice(phase, phase_turn, phase_budget),
                }
            )
            assistant = self.client.chat_tool_step(
                system,
                messages,
                self._phase_tools(tool_names, finish_name, remaining),
            )
            raw_calls = assistant.get("tool_calls")
            if not isinstance(raw_calls, list) or not raw_calls:
                raise TaxonomyAgentError(
                    f"native taxonomy {phase} turn returned no tool call"
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant.get("content"),
                    "tool_calls": raw_calls,
                }
            )
            completed: CategoryChoice | TaxonomyResult | None = None
            for call_index, raw_call in enumerate(raw_calls):
                call = raw_call if isinstance(raw_call, dict) else {}
                function = call.get("function")
                function = function if isinstance(function, dict) else {}
                action = str(function.get("name") or "")
                raw_arguments = function.get("arguments")
                try:
                    arguments = (
                        json.loads(raw_arguments)
                        if isinstance(raw_arguments, str)
                        else raw_arguments
                    )
                except json.JSONDecodeError as exc:
                    arguments = {}
                    observation = TaxonomyExplorer._error(
                        action or "unknown", f"tool arguments are not valid JSON: {exc}"
                    )
                else:
                    signature = json.dumps(
                        {"action": action, "arguments": arguments},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    repeated_actions[signature] = repeated_actions.get(signature, 0) + 1
                    if action not in available_names:
                        observation = TaxonomyExplorer._error(
                            action or "unknown",
                            f"tool is unavailable in the current {phase} budget window; "
                            f"available tools: {sorted(available_names)}",
                        )
                    elif repeated_actions[signature] > 2:
                        observation = TaxonomyExplorer._error(
                            action or "unknown",
                            "identical action repeated; revise the query or browse another node",
                        )
                    elif action == finish_name:
                        completed, observation = finish(arguments)
                    else:
                        observation = self.explorer.execute(action, arguments)
                self._trace(
                    phase_turn, phase, phase_turn, action, arguments, observation
                )
                call_id = str(
                    call.get("id") or f"taxonomy-{phase}-{phase_turn}-{call_index}"
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": action,
                        "content": json.dumps(observation, ensure_ascii=False),
                    }
                )
            if completed is not None:
                return completed, phase_turn
        raise TaxonomyAgentError(
            f"native taxonomy {phase} exploration did not finish within its {phase_budget}-turn "
            "independent task budget"
        )

    def _run_json_category(self, evidence: dict[str, Any]) -> CategoryChoice:
        result, _ = self._run_json_phase(
            phase="category",
            system=self._system_prompt("category"),
            evidence=evidence,
            selected_category=None,
            tool_names=TaxonomyExplorer.CATEGORY_TOOL_NAMES,
            finish_name="finish_category",
            phase_budget=self.max_turns,
            finish=lambda arguments: self.explorer.finish_category(arguments),
        )
        assert isinstance(result, CategoryChoice)
        return result

    def _run_json_attributes(
        self,
        facts: ProductFacts,
        category: CategoryChoice,
        evidence: dict[str, Any],
    ) -> TaxonomyResult:
        result, _ = self._run_json_phase(
            phase="attributes",
            system=self._system_prompt("attributes"),
            evidence=evidence,
            selected_category=category,
            tool_names=TaxonomyExplorer.ATTRIBUTE_TOOL_NAMES,
            finish_name="finish_attributes",
            phase_budget=self.max_turns,
            finish=lambda arguments: self.explorer.finish_attributes(
                facts, category, arguments
            ),
        )
        assert isinstance(result, TaxonomyResult)
        return result

    def _run_json_phase(
        self,
        *,
        phase: str,
        system: str,
        evidence: dict[str, Any],
        selected_category: CategoryChoice | None,
        tool_names: tuple[str, ...],
        finish_name: str,
        phase_budget: int,
        finish: Any,
    ) -> tuple[CategoryChoice | TaxonomyResult, int]:
        history: list[dict[str, Any]] = []
        repeated_actions: dict[str, int] = {}
        for phase_turn in range(1, phase_budget + 1):
            remaining = phase_budget - phase_turn + 1
            available_names = set(
                self._available_names(tool_names, finish_name, remaining)
            )
            category_context = (
                json.dumps(
                    {
                        "category_id": selected_category.category_id,
                        "name": selected_category.name,
                        "path": selected_category.path,
                    },
                    ensure_ascii=False,
                )
                if selected_category is not None
                else "not selected yet"
            )
            prompt = (
                f"Continue the separate {phase} taxonomy task. Choose the next action from TOOL CATALOG.\n\n"
                f"{self._budget_notice(phase, phase_turn, phase_budget)}\n\n"
                f"SELECTED CATEGORY:\n{category_context}\n\n"
                f"PRODUCT EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}\n\n"
                f"TOOL CATALOG:\n{json.dumps(self._catalog(tool_names, finish_name, remaining), ensure_ascii=False)}\n\n"
                f"ACTION/OBSERVATION HISTORY:\n{json.dumps(history, ensure_ascii=False)}"
            )
            response = self.client.chat_json(system, prompt)
            action = str(response.get("action") or "")
            arguments = response.get("arguments")
            signature = json.dumps(
                {"action": action, "arguments": arguments},
                ensure_ascii=False,
                sort_keys=True,
            )
            repeated_actions[signature] = repeated_actions.get(signature, 0) + 1
            if action not in available_names:
                observation = TaxonomyExplorer._error(
                    action or "unknown",
                    f"tool is unavailable in the current {phase} budget window; "
                    f"available tools: {sorted(available_names)}",
                )
                self._trace(
                    phase_turn, phase, phase_turn, action, arguments, observation
                )
            elif repeated_actions[signature] > 2:
                observation = TaxonomyExplorer._error(
                    action or "unknown",
                    "identical action repeated; revise the query or browse another node",
                )
                self._trace(
                    phase_turn, phase, phase_turn, action, arguments, observation
                )
            elif action == finish_name:
                result, observation = finish(arguments)
                self._trace(
                    phase_turn, phase, phase_turn, action, arguments, observation
                )
                if result is not None:
                    return result, phase_turn
            else:
                observation = self.explorer.execute(action, arguments)
                self._trace(
                    phase_turn, phase, phase_turn, action, arguments, observation
                )
            history.append(
                {
                    "action": action,
                    "arguments": arguments if isinstance(arguments, dict) else {},
                    "reason": str(response.get("reason") or "")[:500],
                    "observation": observation,
                }
            )
        raise TaxonomyAgentError(
            f"taxonomy {phase} exploration did not produce a grounded finish within its "
            f"{phase_budget}-turn independent task budget"
        )

    def _trace(
        self,
        turn: int,
        phase: str,
        phase_turn: int,
        action: str,
        arguments: Any,
        observation: dict[str, Any],
    ) -> None:
        if self.trace is not None:
            self.trace.emit(
                "taxonomy.react_step",
                turn=turn,
                phase=phase,
                phase_turn=phase_turn,
                independent_task_budget=self.max_turns,
                action=action,
                arguments=arguments if isinstance(arguments, dict) else {},
                observation=observation,
            )
