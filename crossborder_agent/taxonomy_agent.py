"""Model-directed, read-only exploration of marketplace taxonomy snapshots.

The model owns semantic decisions.  This module only exposes bounded equivalents
of ``read``/``rg`` for the category tree and attribute metadata, then validates
the IDs returned by the model against the supplied snapshots and source facts.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from .agent_workspace import BoundedAgentWorkspace
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
        "list",
        "read",
        "search",
        "bash",
        "write_staging",
        "finish_category",
    )
    ATTRIBUTE_TOOL_NAMES = (
        "query_taxonomy",
        "read_taxonomy",
        "list",
        "read",
        "search",
        "bash",
        "write_staging",
        "finish_attributes",
    )

    TOOL_CATALOG = (
        {
            "name": "list",
            "description": "List normalized evidence and taxonomy files in the bounded workspace.",
        },
        {
            "name": "read",
            "description": "Read a bounded line range from a normalized UTF-8 workspace file.",
        },
        {
            "name": "search",
            "description": "Regex-search normalized evidence/taxonomy files and return matching records.",
        },
        {
            "name": "bash",
            "description": "Run one restricted rg/jq/find/file/ffprobe command inside the workspace.",
        },
        {
            "name": "write_staging",
            "description": "Write notes or proposed intermediate data inside staging only.",
        },
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
            "source_ref": {
                "type": "string",
                "description": (
                    "Exact stable ref from evidence/product.json, such as "
                    "product-attribute/3, sku-option/7, or canonical-claim/2."
                ),
            },
        }
        unresolved_properties = {
            "scope": {"type": "string", "enum": ["product", "sales"]},
            "platform_attr_id": {"type": "string"},
            "reason": {
                "type": "string",
                "minLength": 8,
                "description": (
                    "Evidence-based reason this required or source-relevant attribute "
                    "cannot be mapped without guessing."
                ),
            },
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
            *BoundedAgentWorkspace.openai_tools(),
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
                    "unresolved_mappings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": unresolved_properties,
                            "required": list(unresolved_properties),
                            "additionalProperties": False,
                        },
                    },
                },
                [
                    "selected_attribute_schema_category_id",
                    "mappings",
                    "unresolved_mappings",
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

        self._temporary_workspace = tempfile.TemporaryDirectory(
            prefix="taxonomy-agent-workspace-"
        )
        self.workspace = BoundedAgentWorkspace(
            Path(self._temporary_workspace.name),
            on_observation=self._record_workspace_observation,
        )
        self._source_refs: dict[str, dict[str, str]] = {}
        self._install_workspace()

        # Grounding ledger: final IDs must first have appeared in an observation.
        self.exposed_category_ids: set[str] = set()
        self.inspected_schema_ids: set[str] = set()
        self.inspected_attribute_ids: set[tuple[str, str, str]] = set()
        self.exposed_value_ids: dict[tuple[str, str, str], set[str]] = {}

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

    def install_product_evidence(self, facts: ProductFacts) -> dict[str, Any]:
        """Install exact source rows and return the compact model-visible index."""

        self._source_refs.clear()
        source_rows: list[dict[str, Any]] = []
        for index, item in enumerate(facts.attributes):
            ref = f"product-attribute/{index}"
            row = {
                "ref": ref,
                "source_kind": "product",
                "name": item.name,
                "value": item.value,
                "evidence_pointer": item.evidence_pointer or f"attributes[{index}]",
            }
            self._source_refs[ref] = {
                "source_kind": "product",
                "source_name": item.name,
                "source_value": item.value,
            }
            source_rows.append(row)

        unique_sku_rows: dict[tuple[str, str], dict[str, Any]] = {}
        for sku in facts.skus:
            for item in sku.attributes:
                key = (item.name, item.value)
                row = unique_sku_rows.setdefault(
                    key,
                    {
                        "source_kind": "sku",
                        "name": item.name,
                        "value": item.value,
                        "evidence_pointer": item.evidence_pointer,
                        "sku_ids": [],
                    },
                )
                if sku.sku_id and sku.sku_id not in row["sku_ids"]:
                    row["sku_ids"].append(sku.sku_id)
                if not row["evidence_pointer"]:
                    row["evidence_pointer"] = sku.evidence_pointer
        for index, row in enumerate(unique_sku_rows.values()):
            ref = f"sku-option/{index}"
            row = {
                "ref": ref,
                **row,
                "evidence_pointer": row["evidence_pointer"] or f"skus[*].attributes[{index}]",
            }
            self._source_refs[ref] = {
                "source_kind": "sku",
                "source_name": str(row["name"]),
                "source_value": str(row["value"]),
            }
            source_rows.append(row)

        canonical_claims = facts.reconciled_fact_ledger.get(
            "canonical_visual_claims", []
        )
        for index, item in enumerate(
            canonical_claims if isinstance(canonical_claims, list) else []
        ):
            if not isinstance(item, dict):
                continue
            value = str(item.get("value") or "").strip()
            if not value:
                continue
            name = str(item.get("concept") or "visible_design_feature")
            ref = f"canonical-claim/{index}"
            row = {
                "ref": ref,
                "source_kind": "canonical",
                "name": name,
                "value": value,
                "evidence_pointer": str(
                    item.get("evidence_pointer")
                    or f"reconciled_fact_ledger.canonical_visual_claims[{index}]"
                ),
            }
            self._source_refs[ref] = {
                "source_kind": "canonical",
                "source_name": name,
                "source_value": value,
            }
            source_rows.append(row)

        evidence = {
            "source_title": facts.source_title,
            "source_category_name": facts.source_category_name,
            "source_evidence": source_rows,
            "reconciled_fact_ledger": facts.reconciled_fact_ledger,
        }
        self.workspace.host_write_json("evidence/product.json", evidence)
        return evidence

    def _install_workspace(self) -> None:
        for collection in self.COLLECTIONS:
            self.workspace.host_write_json(
                f"taxonomy/{collection}.jsonl",
                self._records[collection],
                jsonl=True,
            )
        self.workspace.host_write_json(
            "workspace/index.json",
            {
                "evidence": ["evidence/product.json"],
                "taxonomy": [
                    f"taxonomy/{collection}.jsonl"
                    for collection in self.COLLECTIONS
                ],
                "notes": (
                    "JSONL records carry stable refs. Use search or restricted rg for discovery, "
                    "then read bounded line ranges. Writes are allowed only below staging/."
                ),
            },
        )

    def _record_workspace_observation(self, text: str) -> None:
        refs = set(
            re.findall(
                r"(?:categories|schemas|attributes|values)/[^\s\"'\\]+",
                text,
            )
        )
        by_collection: dict[str, list[dict[str, Any]]] = {}
        for ref in refs:
            record = self._refs.get(ref.rstrip(",}]"))
            if record is None:
                continue
            collection = str(record["ref"]).split("/", 1)[0]
            by_collection.setdefault(collection, []).append(record)
        for collection, records in by_collection.items():
            self._record_exposure(collection, records)

    def source_record(self, ref: str) -> dict[str, str] | None:
        row = self._source_refs.get(str(ref or ""))
        return dict(row) if row is not None else None

    def close(self) -> None:
        self._temporary_workspace.cleanup()

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
            scope = str(item.get("scope") or "")
            attr_id = str(item.get("attr_id") or "")
            if not schema_id or not scope or not attr_id:
                continue
            self.inspected_schema_ids.add(schema_id)
            self.inspected_attribute_ids.add((schema_id, scope, attr_id))
            if collection == "values" and item.get("value_id"):
                self.exposed_value_ids.setdefault(
                    (schema_id, scope, attr_id), set()
                ).add(str(item["value_id"]))

    def execute(self, tool: str, arguments: Any) -> dict[str, Any]:
        args = arguments if isinstance(arguments, dict) else {}
        try:
            if tool in {"list", "read", "search", "bash", "write_staging"}:
                return self.workspace.execute(tool, args)
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
    def _error(
        tool: str,
        error: str,
        *,
        code: str = "validation_error",
        details: Any = None,
        correction: str = "",
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "tool": tool,
            "error": error,
            "code": code,
        }
        if details is not None:
            result["details"] = details
        if correction:
            result["correction"] = correction
        return result

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
        schema = attribute_schema_definition(self.attribute_data, schema_id)
        definitions = {
            (str(item.get("scope") or ""), str(item.get("attr_id") or "")): item
            for item in schema.get("attributes") or []
            if item.get("attr_id")
        }
        request_errors: list[dict[str, Any]] = []
        if schema_id not in self.inspected_schema_ids:
            request_errors.append(
                {
                    "code": "schema_not_observed",
                    "field": "selected_attribute_schema_category_id",
                    "actual": schema_id,
                    "message": "schema must appear in read/search/bash output before commit",
                }
            )

        mappings = args.get("mappings")
        if not isinstance(mappings, list):
            request_errors.append(
                {
                    "code": "invalid_mapping_array",
                    "field": "mappings",
                    "actual_type": type(mappings).__name__,
                    "message": "mappings must be an array",
                }
            )
            mappings = []
        accepted_rows: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []
        expanded_mappings: list[dict[str, str]] = []
        seen_mapping_keys: set[tuple[str, str, str, str]] = set()
        for index, row in enumerate(mappings):
            if not isinstance(row, dict):
                rejected_rows.append(
                    {
                        "mapping_index": index,
                        "submitted": row,
                        "reasons": [
                            {
                                "code": "mapping_not_object",
                                "field": f"mappings[{index}]",
                                "message": "mapping must be an object",
                            }
                        ],
                    }
                )
                continue
            scope = str(row.get("scope") or "")
            attr_id = str(row.get("platform_attr_id") or "")
            key = (schema_id, scope, attr_id)
            definition = definitions.get((scope, attr_id))
            value_id = str(row.get("platform_value_id") or "")
            source_ref = str(row.get("source_ref") or "")
            source = self.source_record(source_ref)
            reasons: list[dict[str, Any]] = []
            if definition is None:
                reasons.append(
                    {
                        "code": "attribute_not_in_schema",
                        "field": "platform_attr_id",
                        "actual": attr_id,
                        "scope": scope,
                        "message": "attribute ID/scope is not defined by the selected schema",
                    }
                )
            elif key not in self.inspected_attribute_ids:
                reasons.append(
                    {
                        "code": "attribute_not_observed",
                        "field": "platform_attr_id",
                        "actual": attr_id,
                        "scope": scope,
                        "expected_ref": f"attributes/{schema_id}/{scope}/{attr_id}",
                        "message": "read or search the exact attribute record before commit",
                    }
                )
            allowed_values = list((definition or {}).get("values") or [])
            allowed_value_ids = {
                str(item.get("value_id") or "") for item in allowed_values
            }
            if allowed_values and value_id not in allowed_value_ids:
                reasons.append(
                    {
                        "code": "value_not_in_attribute",
                        "field": "platform_value_id",
                        "actual": value_id,
                        "message": "enum value ID is not defined by this schema attribute",
                    }
                )
            elif allowed_values and value_id not in self.exposed_value_ids.get(key, set()):
                reasons.append(
                    {
                        "code": "value_not_observed",
                        "field": "platform_value_id",
                        "actual": value_id,
                        "expected_ref": f"values/{schema_id}/{scope}/{attr_id}/{value_id}",
                        "message": "read or search the exact enum value record before commit",
                    }
                )
            if not allowed_values and value_id:
                reasons.append(
                    {
                        "code": "free_text_value_id_must_be_empty",
                        "field": "platform_value_id",
                        "actual": value_id,
                        "message": "leave platform_value_id empty for a free-text attribute",
                    }
                )
            if source is None:
                same_scope_refs = [
                    ref
                    for ref, candidate in self._source_refs.items()
                    if (scope == "sales" and candidate["source_kind"] == "sku")
                    or (
                        scope == "product"
                        and candidate["source_kind"] in {"product", "canonical"}
                    )
                ]
                reasons.append(
                    {
                        "code": "unknown_source_ref",
                        "field": "source_ref",
                        "actual": source_ref,
                        "available_source_refs_for_scope": same_scope_refs[:80],
                        "message": "source_ref must be copied exactly from evidence/product.json",
                    }
                )
            elif (
                scope == "sales" and source["source_kind"] != "sku"
            ) or (
                scope == "product"
                and source["source_kind"] not in {"product", "canonical"}
            ):
                reasons.append(
                    {
                        "code": "source_scope_mismatch",
                        "field": "source_ref",
                        "actual": source_ref,
                        "source_kind": source["source_kind"],
                        "expected": "sku" if scope == "sales" else "product or canonical",
                        "message": "source evidence kind does not match platform attribute scope",
                    }
                )
            duplicate_key = (scope, attr_id, value_id, source_ref)
            if duplicate_key in seen_mapping_keys:
                reasons.append(
                    {
                        "code": "duplicate_mapping",
                        "field": f"mappings[{index}]",
                        "message": "an identical mapping already appears in this proposal",
                    }
                )
            seen_mapping_keys.add(duplicate_key)
            if reasons:
                rejected_rows.append(
                    {
                        "mapping_index": index,
                        "submitted": row,
                        "reasons": reasons,
                    }
                )
                continue
            assert source is not None
            expanded = {
                "scope": scope,
                "platform_attr_id": attr_id,
                "platform_value_id": value_id,
                **source,
            }
            expanded_mappings.append(expanded)
            accepted_rows.append(
                {
                    "mapping_index": index,
                    "submitted": row,
                    "resolved_source": source,
                }
            )

        if request_errors or rejected_rows:
            return None, self._error(
                "finish_attributes",
                "attribute proposal was not committed; correct only the rejected rows and resubmit the complete proposal",
                code="attribute_mapping_rejected",
                details={
                    "request_errors": request_errors,
                    "accepted_mappings": accepted_rows,
                    "rejected_mappings": rejected_rows,
                    "accepted_mapping_count": len(accepted_rows),
                    "rejected_mapping_count": len(rejected_rows),
                },
                correction=(
                    "Keep every accepted mapping unchanged. Replace each rejected row using its exact reason, "
                    "expected ref, and the stable source refs in evidence/product.json; then resubmit once."
                ),
            )
        base = TaxonomyResult(
            category=category,
            attributes=[],
            attribute_schema_category_id=schema_id,
        )
        mapped = apply_model_attribute_mappings(
            facts, base, self.attribute_data, expanded_mappings
        )
        if expanded_mappings and len(mapped.attributes) != len(expanded_mappings):
            return None, self._error(
                "finish_attributes",
                "host mapping installation disagreed with the prevalidated proposal",
                code="mapping_installation_invariant_failed",
                details={
                    "accepted_mappings": accepted_rows,
                    "expected_count": len(expanded_mappings),
                    "installed_count": len(mapped.attributes),
                },
                correction="Do not retry the identical proposal; inspect the reported source refs and schema records.",
            )
        if not expanded_mappings:
            mapped.missing_required = [
                str(item.get("name") or "")
                for item in schema.get("attributes") or []
                if item.get("required")
            ]

        unresolved_rows = args.get("unresolved_mappings")
        if unresolved_rows is None:
            unresolved_rows = []
        elif not isinstance(unresolved_rows, list):
            return None, self._error(
                "finish_attributes",
                "unresolved_mappings must be an array",
                code="invalid_unresolved_mapping_array",
                details={
                    "field": "unresolved_mappings",
                    "actual_type": type(unresolved_rows).__name__,
                    "accepted_mappings": accepted_rows,
                },
                correction=(
                    "Keep accepted mappings unchanged and submit unresolved_mappings as an array; "
                    "use an empty array when every required/relevant attribute is mapped."
                ),
            )
        unresolved: dict[tuple[str, str], str] = {}
        accepted_unresolved: list[dict[str, Any]] = []
        rejected_unresolved: list[dict[str, Any]] = []
        for index, row in enumerate(unresolved_rows):
            if not isinstance(row, dict):
                rejected_unresolved.append(
                    {
                        "unresolved_index": index,
                        "submitted": row,
                        "reasons": [
                            {
                                "code": "unresolved_not_object",
                                "message": "unresolved mapping must be an object",
                            }
                        ],
                    }
                )
                continue
            key = (
                str(row.get("scope") or ""),
                str(row.get("platform_attr_id") or ""),
            )
            reason = str(row.get("reason") or "").strip()
            reasons: list[dict[str, Any]] = []
            if key not in definitions:
                reasons.append(
                    {
                        "code": "unresolved_attribute_not_in_schema",
                        "scope": key[0],
                        "platform_attr_id": key[1],
                        "message": "attribute ID/scope is not defined by the selected schema",
                    }
                )
            elif (schema_id, key[0], key[1]) not in self.inspected_attribute_ids:
                reasons.append(
                    {
                        "code": "unresolved_attribute_not_observed",
                        "expected_ref": f"attributes/{schema_id}/{key[0]}/{key[1]}",
                        "message": "read or search the exact attribute before marking it unresolved",
                    }
                )
            if len(reason) < 8:
                reasons.append(
                    {
                        "code": "unresolved_reason_too_short",
                        "field": "reason",
                        "minimum_characters": 8,
                        "message": "provide the concrete missing or conflicting source evidence",
                    }
                )
            if reasons:
                rejected_unresolved.append(
                    {
                        "unresolved_index": index,
                        "submitted": row,
                        "reasons": reasons,
                    }
                )
            else:
                unresolved[key] = reason[:1000]
                accepted_unresolved.append(
                    {
                        "unresolved_index": index,
                        "scope": key[0],
                        "platform_attr_id": key[1],
                        "reason": reason[:1000],
                    }
                )
        if rejected_unresolved:
            return None, self._error(
                "finish_attributes",
                "one or more unresolved dispositions were rejected",
                code="unresolved_mapping_rejected",
                details={
                    "accepted_mappings": accepted_rows,
                    "accepted_unresolved_mappings": accepted_unresolved,
                    "rejected_unresolved_mappings": rejected_unresolved,
                },
                correction=(
                    "Keep accepted rows unchanged. Correct only each rejected unresolved row using its exact "
                    "schema ref and a concrete evidence-gap reason, then resubmit the complete proposal."
                ),
            )

        present = {
            ("sales" if item.sales_attribute else "product", item.attr_id)
            for item in mapped.attributes
        }

        def evidence_candidates(definition: dict[str, Any]) -> list[dict[str, str]]:
            scope = str(definition.get("scope") or "")
            attribute_name = _search_text(definition.get("name"))
            values = {
                _search_text(item.get("name"))
                for item in definition.get("values") or []
                if _search_text(item.get("name"))
            }
            candidates: list[dict[str, str]] = []
            for source_ref, source in self._source_refs.items():
                source_kind = source["source_kind"]
                if scope == "sales" and source_kind != "sku":
                    continue
                if scope == "product" and source_kind not in {"product", "canonical"}:
                    continue
                source_name = source["source_name"]
                source_value = source["source_value"]
                if (
                    _search_text(source_name) == attribute_name
                    or _search_text(source_value) in values
                ):
                    candidates.append(
                        {
                            "source_ref": source_ref,
                            "source_kind": source_kind,
                            "source_name": source_name,
                            "source_value": source_value,
                        }
                    )
            return candidates[:20]

        completion_gaps: list[dict[str, Any]] = []
        for key, definition in definitions.items():
            if key in present or key in unresolved:
                continue
            candidates = evidence_candidates(definition)
            required = bool(definition.get("required"))
            # Sales dimensions are relevant when their schema name or one of
            # their enum values has a spelling-level match in the supplied SKU
            # evidence.  This is a grounding/completeness signal, not a host
            # semantic mapping decision.
            source_relevant_sales = key[0] == "sales" and bool(candidates)
            if required or source_relevant_sales:
                completion_gaps.append(
                    {
                        "scope": key[0],
                        "platform_attr_id": key[1],
                        "platform_attribute_name": str(
                            definition.get("name") or ""
                        ),
                        "required": required,
                        "matching_source_evidence": candidates,
                    }
                )
        if completion_gaps:
            return None, self._error(
                "finish_attributes",
                "attribute selection is incomplete",
                code="attribute_disposition_incomplete",
                details={
                    "completion_gaps": completion_gaps,
                    "accepted_mappings": accepted_rows,
                },
                correction=(
                    "For every completion gap, either map one of matching_source_evidence refs or add the "
                    "inspected attribute to unresolved_mappings with an evidence-based reason."
                ),
            )
        return mapped, self._ok(
            "finish_attributes",
            {
                "selected_category_id": category.category_id,
                "selected_attribute_schema_category_id": schema_id,
                "accepted_mapping_count": len(mapped.attributes),
                "accepted_mappings": accepted_rows,
                "unresolved_mapping_count": len(unresolved),
                "unresolved_mappings": [
                    {
                        "scope": scope,
                        "platform_attr_id": attr_id,
                        "reason": reason,
                    }
                    for (scope, attr_id), reason in unresolved.items()
                ],
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

    def _product_evidence(
        self, facts: ProductFacts, *, decision_context: str = ""
    ) -> dict[str, Any]:
        evidence = self.explorer.install_product_evidence(facts)
        if decision_context:
            evidence["orchestrator_reconsideration_context"] = decision_context[:3000]
            self.explorer.workspace.host_write_json("evidence/product.json", evidence)
        return evidence

    def run(
        self, facts: ProductFacts, *, decision_context: str = ""
    ) -> TaxonomyResult:
        try:
            evidence = self._product_evidence(
                facts, decision_context=decision_context
            )
            category = self.resolve_category(facts, evidence=evidence)
            return self.resolve_attributes(facts, category, evidence=evidence)
        finally:
            self.close()

    def resolve_category(
        self,
        facts: ProductFacts,
        *,
        decision_context: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> CategoryChoice:
        """Run and commit only the independently validated category transaction."""

        phase_evidence = evidence or self._product_evidence(
            facts, decision_context=decision_context
        )
        native_step = getattr(self.client, "chat_tool_step", None)
        if callable(native_step):
            return self._run_native_category(phase_evidence)
        return self._run_json_category(phase_evidence)

    def resolve_attributes(
        self,
        facts: ProductFacts,
        category: CategoryChoice,
        *,
        decision_context: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> TaxonomyResult:
        """Run and commit attributes without changing an accepted category."""

        phase_evidence = evidence or self._product_evidence(
            facts, decision_context=decision_context
        )
        native_step = getattr(self.client, "chat_tool_step", None)
        if callable(native_step):
            return self._run_native_attributes(facts, category, phase_evidence)
        return self._run_json_attributes(facts, category, phase_evidence)

    def close(self) -> None:
        self.explorer.close()

    def _system_prompt(self, phase: str) -> str:
        shared = (
            "You are the reasoning controller of a marketplace taxonomy exploration agent. "
            "You receive a bounded workspace instead of complete snapshots in context. Start with workspace/index.json. "
            "Product evidence is in evidence/product.json; normalized categories, schemas, attributes, and values are "
            "JSONL files below taxonomy/. Prefer query_taxonomy for bounded batch discovery and read_taxonomy for "
            "exact referenced records; use list/read/search or a restricted rg/jq/find/file/ffprobe command through "
            "bash only when the structured queries cannot express the needed inspection. Use write_staging only for "
            "notes. Every taxonomy record carries a stable ref. Category "
            "records expose parent/child/ancestor IDs and available_schema_ids; schema, attribute and value records are "
            "linked by schema_category_id, scope and attr_id. Semantic decisions are yours; host tools only retrieve "
            "and validate. Put independent collection lookups into one query_taxonomy request when practical, then "
            "read only useful refs. Do not guess IDs or use product "
            "identifiers/benchmark memories. Adapt after empty or ambiguous results. Every turn includes a budget "
            "notice. The last two phase turns are reserved "
            "for a validated finish and one correction, so complete prerequisite inspection before that window. "
            "Return one JSON action object only when using the JSON protocol: "
            '{"action": tool name, "arguments": object, "reason": concise evidence note}. '
        )
        if phase == "category":
            phase_rules = (
                "This task selects only the best platform leaf category. Prefer source category, title, and "
                "structured facts. Search multiple useful phrases and inspect exact records only when needed. "
                "Call finish_category as soon as a suitable observed leaf is identified; provide "
                "selected_category_id, confidence, and evidence."
            )
        else:
            phase_rules = (
                "This separate task selects the attribute schema and grounded mappings for an already selected "
                "leaf. Inspect its category record for available_schema_ids, then search schema, attribute, and value "
                "records using source names and values. Before finish_attributes, every submitted schema, attribute, "
                "and enum value ID must have appeared in read/search/bash output. Each mapping contains scope, "
                "platform_attr_id, platform_value_id, and one source_ref copied exactly from evidence/product.json. "
                "Stable source refs eliminate manual copying of source kind/name/value. Never substitute a nearby ref. "
                "Prioritize every required product attribute and every sales "
                "dimension directly represented by SKU evidence. finish_attributes requires unresolved_mappings: for "
                "each required or source-relevant attribute you cannot safely map, name the inspected platform attr ID "
                "and explain the evidence gap. This is a disposition ledger, not permission to skip inspection. "
                "Do not treat a source-grounded non-visual fact as invalid merely because pixels cannot verify it."
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
            stalled_error = ""
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
                    elif action == finish_name and repeated_actions[signature] > 1:
                        observation = TaxonomyExplorer._error(
                            action or "unknown",
                            "identical rejected finish repeated without changing the proposal",
                            code="no_progress",
                            correction=(
                                "Use the previous rejection's rejected_mappings, expected refs, and correction "
                                "fields. An unchanged proposal cannot pass deterministic validation."
                            ),
                        )
                        stalled_error = (
                            f"native taxonomy {phase} stopped after an identical rejected finish; "
                            "the successful prior phase remains committed"
                        )
                    elif repeated_actions[signature] > 2:
                        observation = TaxonomyExplorer._error(
                            action or "unknown",
                            "identical exploration action repeated; change the query or read another range",
                            code="no_progress",
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
            if stalled_error:
                raise TaxonomyAgentError(stalled_error)
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
            elif action == finish_name and repeated_actions[signature] > 1:
                observation = TaxonomyExplorer._error(
                    action or "unknown",
                    "identical rejected finish repeated without changing the proposal",
                    code="no_progress",
                    correction=(
                        "Use the previous rejection's rejected_mappings, expected refs, and correction fields. "
                        "An unchanged proposal cannot pass deterministic validation."
                    ),
                )
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
                    f"taxonomy {phase} stopped after an identical rejected finish; "
                    "the successful prior phase remains committed"
                )
            elif repeated_actions[signature] > 2:
                observation = TaxonomyExplorer._error(
                    action or "unknown",
                    "identical exploration action repeated; change the query or read another range",
                    code="no_progress",
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
