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
    """Bounded read/search tools over one supplied taxonomy snapshot."""

    TOOL_CATALOG = (
        {
            "name": "list_root_categories",
            "description": "List the top-level category nodes. Arguments: offset?, limit?.",
        },
        {
            "name": "list_children",
            "description": "List direct children of a category. Arguments: category_id, offset?, limit?.",
        },
        {
            "name": "search_categories",
            "description": (
                "Literal search over names and full paths. The model chooses the query; results are not "
                "semantically ranked. Arguments: query, under_category_id?, leaf_only?, offset?, limit?."
            ),
        },
        {
            "name": "inspect_categories",
            "description": "Inspect exact category nodes. Arguments: category_ids (up to 20 IDs).",
        },
        {
            "name": "search_attribute_schemas",
            "description": (
                "Literal search over schema names, paths, and attribute names. "
                "Arguments: query, offset?, limit?."
            ),
        },
        {
            "name": "get_attribute_schema",
            "description": (
                "Inspect one schema's attribute IDs and names. When given a leaf without a direct schema, "
                "also returns structurally available ancestor schemas. Arguments: category_id."
            ),
        },
        {
            "name": "get_attribute_definition",
            "description": (
                "Inspect one attribute and a page of its allowed enum values. "
                "Arguments: schema_category_id, attr_id, offset?, limit?."
            ),
        },
        {
            "name": "search_attribute_values",
            "description": (
                "Literal-search enum values for one inspected attribute. Arguments: "
                "schema_category_id, attr_id, queries (string array), offset?, limit?."
            ),
        },
        {
            "name": "finish",
            "description": (
                "Submit selected_category_id, selected_attribute_schema_category_id, mappings, confidence, "
                "and evidence. The selected leaf/schema/attribute/value IDs must have been observed first."
            ),
        },
    )

    @classmethod
    def openai_tools(cls) -> list[dict[str, Any]]:
        """Return native function-tool definitions for OpenAI-compatible APIs."""

        paging = {
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 60},
        }

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
        return [
            tool("list_root_categories", by_name["list_root_categories"], dict(paging)),
            tool(
                "list_children",
                by_name["list_children"],
                {"category_id": {"type": "string"}, **paging},
                ["category_id"],
            ),
            tool(
                "search_categories",
                by_name["search_categories"],
                {
                    "query": {"type": "string"},
                    "under_category_id": {"type": "string"},
                    "leaf_only": {"type": "boolean"},
                    **paging,
                },
                ["query"],
            ),
            tool(
                "inspect_categories",
                by_name["inspect_categories"],
                {
                    "category_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 20,
                    }
                },
                ["category_ids"],
            ),
            tool(
                "search_attribute_schemas",
                by_name["search_attribute_schemas"],
                {"query": {"type": "string"}, **paging},
                ["query"],
            ),
            tool(
                "get_attribute_schema",
                by_name["get_attribute_schema"],
                {"category_id": {"type": "string"}},
                ["category_id"],
            ),
            tool(
                "get_attribute_definition",
                by_name["get_attribute_definition"],
                {
                    "schema_category_id": {"type": "string"},
                    "attr_id": {"type": "string"},
                    **paging,
                },
                ["schema_category_id", "attr_id"],
            ),
            tool(
                "search_attribute_values",
                by_name["search_attribute_values"],
                {
                    "schema_category_id": {"type": "string"},
                    "attr_id": {"type": "string"},
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 20,
                    },
                    **paging,
                },
                ["schema_category_id", "attr_id", "queries"],
            ),
            tool(
                "finish",
                by_name["finish"],
                {
                    "selected_category_id": {"type": "string"},
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
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string"},
                },
                [
                    "selected_category_id",
                    "selected_attribute_schema_category_id",
                    "mappings",
                    "confidence",
                    "evidence",
                ],
            ),
        ]

    def __init__(self, category_tree: dict[str, Any], attribute_data: dict[str, Any]):
        self.category_tree = category_tree
        self.attribute_data = attribute_data
        self.nodes: dict[str, _CategoryNode] = {}
        self.root_ids: list[str] = []
        self._tree_order: list[str] = []
        self._build_tree()
        self.schemas = {
            str(item["category_id"]): item
            for item in attribute_schema_candidates(attribute_data)
            if item.get("category_id")
        }
        self._schema_order = list(self.schemas)

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
            if isinstance(raw, dict) and raw.get("catId") not in (None, ""):
                self.root_ids.append(str(raw["catId"]))
            visit(raw)

    @staticmethod
    def _page(items: list[Any], arguments: dict[str, Any]) -> dict[str, Any]:
        offset = _bounded_int(arguments.get("offset"), 0, low=0, high=max(0, len(items)))
        limit = _bounded_int(arguments.get("limit"), 30, low=1, high=60)
        page = items[offset : offset + limit]
        next_offset = offset + len(page)
        return {
            "total": len(items),
            "offset": offset,
            "next_offset": next_offset if next_offset < len(items) else None,
            "items": page,
        }

    def _category_page(self, ids: list[str], arguments: dict[str, Any]) -> dict[str, Any]:
        nodes = [self.nodes[item].compact() for item in ids if item in self.nodes]
        page = self._page(nodes, arguments)
        self.exposed_category_ids.update(item["category_id"] for item in page["items"])
        return page

    def _descendant_ids(self, root_id: str) -> set[str]:
        if not root_id:
            return set(self._tree_order)
        if root_id not in self.nodes:
            return set()
        result: set[str] = set()
        pending = [root_id]
        while pending:
            current = pending.pop()
            if current in result:
                continue
            result.add(current)
            pending.extend(self.nodes[current].child_ids)
        return result

    def _ancestor_schema_summaries(self, category_id: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        current = self.nodes.get(category_id)
        while current is not None:
            schema = self.schemas.get(current.category_id)
            if schema:
                result.append(self._compact_schema(schema))
            current = self.nodes.get(current.parent_id)
        return result

    @staticmethod
    def _compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "category_id": str(schema.get("category_id") or ""),
            "name": str(schema.get("name") or schema.get("name_chinese") or ""),
            "path": str(schema.get("path") or ""),
            "product_attributes": list(schema.get("product_attributes") or []),
            "sales_attributes": list(schema.get("sales_attributes") or []),
        }

    def execute(self, tool: str, arguments: Any) -> dict[str, Any]:
        args = arguments if isinstance(arguments, dict) else {}
        try:
            if tool == "list_root_categories":
                return self._ok(tool, self._category_page(self.root_ids, args))

            if tool == "list_children":
                category_id = str(args.get("category_id") or "")
                node = self.nodes.get(category_id)
                if not node:
                    return self._error(tool, f"unknown category_id: {category_id}")
                self.exposed_category_ids.add(category_id)
                return self._ok(tool, self._category_page(node.child_ids, args))

            if tool == "search_categories":
                query = _search_text(args.get("query"))
                if not query:
                    return self._error(tool, "query must be a non-empty literal phrase")
                scope_id = str(args.get("under_category_id") or "")
                scope = self._descendant_ids(scope_id)
                if scope_id and not scope:
                    return self._error(tool, f"unknown under_category_id: {scope_id}")
                leaf_only = bool(args.get("leaf_only"))
                matched = [
                    category_id
                    for category_id in self._tree_order
                    if category_id in scope
                    and (not leaf_only or self.nodes[category_id].is_leaf)
                    and query
                    in _search_text(
                        f"{self.nodes[category_id].name} {self.nodes[category_id].path}"
                    )
                ]
                result = self._category_page(matched, args)
                result["query"] = str(args.get("query") or "")
                result["under_category_id"] = scope_id
                return self._ok(tool, result)

            if tool == "inspect_categories":
                raw_ids = args.get("category_ids")
                category_ids = raw_ids if isinstance(raw_ids, list) else []
                category_ids = [str(item) for item in category_ids[:20]]
                unknown = [item for item in category_ids if item not in self.nodes]
                if unknown:
                    return self._error(tool, f"unknown category IDs: {unknown}")
                return self._ok(tool, self._category_page(category_ids, {"limit": 20}))

            if tool == "search_attribute_schemas":
                query = _search_text(args.get("query"))
                if not query:
                    return self._error(tool, "query must be a non-empty literal phrase")
                matched = []
                for schema_id in self._schema_order:
                    schema = self.schemas[schema_id]
                    haystack = _search_text(
                        " ".join(
                            [
                                str(schema.get("name") or ""),
                                str(schema.get("name_chinese") or ""),
                                str(schema.get("path") or ""),
                                *list(schema.get("product_attributes") or []),
                                *list(schema.get("sales_attributes") or []),
                            ]
                        )
                    )
                    if query in haystack:
                        matched.append(self._compact_schema(schema))
                page = self._page(matched, args)
                # Search reveals candidates, but a schema must be explicitly inspected before finish.
                page["query"] = str(args.get("query") or "")
                return self._ok(tool, page)

            if tool == "get_attribute_schema":
                category_id = str(args.get("category_id") or "")
                schema = self.schemas.get(category_id)
                if not schema and category_id not in self.nodes:
                    return self._error(tool, f"unknown category/schema ID: {category_id}")
                if not schema:
                    return self._ok(
                        tool,
                        {
                            "category_id": category_id,
                            "schema": None,
                            "ancestor_schemas": self._ancestor_schema_summaries(category_id),
                        },
                    )
                definition = attribute_schema_definition(self.attribute_data, category_id)
                attributes = [
                    {
                        "scope": item.get("scope"),
                        "attr_id": item.get("attr_id"),
                        "name": item.get("name"),
                        "required": item.get("required"),
                        "multiple": item.get("multiple"),
                        "value_count": len(item.get("values") or []),
                    }
                    for item in definition.get("attributes") or []
                ]
                self.inspected_schema_ids.add(category_id)
                return self._ok(
                    tool,
                    {
                        "category_id": category_id,
                        "name": schema.get("name"),
                        "path": schema.get("path"),
                        "attributes": attributes,
                    },
                )

            if tool in {"get_attribute_definition", "search_attribute_values"}:
                return self._execute_attribute_tool(tool, args)

            return self._error(tool, f"unknown tool: {tool}")
        except Exception as exc:
            return self._error(tool, f"tool execution failed: {exc}")

    def _execute_attribute_tool(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        schema_id = str(args.get("schema_category_id") or "")
        attr_id = str(args.get("attr_id") or "")
        if schema_id not in self.inspected_schema_ids:
            return self._error(tool, "inspect the schema with get_attribute_schema first")
        definition = attribute_schema_definition(self.attribute_data, schema_id)
        attribute = next(
            (
                item
                for item in definition.get("attributes") or []
                if str(item.get("attr_id") or "") == attr_id
            ),
            None,
        )
        if not attribute:
            return self._error(tool, f"attr_id {attr_id} is not in schema {schema_id}")
        self.inspected_attribute_ids.add((schema_id, attr_id))
        values = list(attribute.get("values") or [])
        if tool == "search_attribute_values":
            raw_queries = args.get("queries")
            queries = raw_queries if isinstance(raw_queries, list) else []
            normalized = [_search_text(item) for item in queries[:20] if _search_text(item)]
            if not normalized:
                return self._error(tool, "queries must contain at least one non-empty literal value")
            values = [
                value
                for value in values
                if any(query in _search_text(value.get("name")) for query in normalized)
            ]
        page = self._page(values, args)
        key = (schema_id, attr_id)
        self.exposed_value_ids.setdefault(key, set()).update(
            str(item.get("value_id") or "")
            for item in page["items"]
            if item.get("value_id")
        )
        return self._ok(
            tool,
            {
                "schema_category_id": schema_id,
                "attribute": {
                    key: value
                    for key, value in attribute.items()
                    if key != "values"
                },
                "values": page,
                **({"queries": args.get("queries")} if tool == "search_attribute_values" else {}),
            },
        )

    @staticmethod
    def _ok(tool: str, result: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "tool": tool, "result": result}

    @staticmethod
    def _error(tool: str, error: str) -> dict[str, Any]:
        return {"ok": False, "tool": tool, "error": error}

    def finish(self, facts: ProductFacts, arguments: Any) -> tuple[TaxonomyResult | None, dict[str, Any]]:
        args = arguments if isinstance(arguments, dict) else {}
        category_id = str(args.get("selected_category_id") or "")
        schema_id = str(args.get("selected_attribute_schema_category_id") or "")
        errors: list[str] = []
        node = self.nodes.get(category_id)
        if category_id not in self.exposed_category_ids:
            errors.append("selected_category_id was not observed through an exploration tool")
        if not node or not node.is_leaf:
            errors.append("selected_category_id must be an existing leaf")
        if schema_id not in self.inspected_schema_ids:
            errors.append("selected_attribute_schema_category_id must be inspected first")

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
                errors.append(f"mapping {index} uses attribute {attr_id} before inspecting it")
                continue
            value_id = str(row.get("platform_value_id") or "")
            schema = attribute_schema_definition(self.attribute_data, schema_id)
            definition = next(
                (item for item in schema.get("attributes") or [] if item.get("attr_id") == attr_id),
                None,
            )
            allowed_values = list((definition or {}).get("values") or [])
            if allowed_values and value_id not in self.exposed_value_ids.get(key, set()):
                errors.append(f"mapping {index} uses enum value {value_id} before observing it")
            if not allowed_values and value_id:
                errors.append(f"mapping {index} must leave value ID empty for a free-text attribute")

        if errors:
            return None, self._error("finish", "; ".join(errors))
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
        base = TaxonomyResult(
            category=choice,
            attributes=[],
            attribute_schema_category_id=schema_id,
        )
        mapped = apply_model_attribute_mappings(facts, base, self.attribute_data, mappings)
        if mappings and len(mapped.attributes) != len(mappings):
            return None, self._error(
                "finish",
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
            "finish",
            {
                "selected_category_id": category_id,
                "selected_attribute_schema_category_id": schema_id,
                "accepted_mapping_count": len(mapped.attributes),
            },
        )


class TaxonomyReActAgent:
    """Run a bounded action/observation loop controlled by the chat model."""

    def __init__(
        self,
        client: Any,
        category_tree: dict[str, Any],
        attribute_data: dict[str, Any],
        *,
        skill_instructions: str = "",
        trace: Any = None,
        max_turns: int = 16,
    ):
        self.client = client
        self.explorer = TaxonomyExplorer(category_tree, attribute_data)
        self.skill_instructions = skill_instructions
        self.trace = trace
        self.max_turns = max_turns

    @staticmethod
    def _product_evidence(facts: ProductFacts) -> dict[str, Any]:
        sku_options: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for sku in facts.skus:
            for item in sku.attributes:
                key = (item.name, item.value)
                if key not in seen:
                    seen.add(key)
                    sku_options.append({"name": item.name, "value": item.value})
        return {
            "source_title": facts.source_title,
            "source_category_name": facts.source_category_name,
            "attributes": [
                {"name": item.name, "value": item.value} for item in facts.attributes
            ],
            "sku_options": sku_options,
        }

    def run(self, facts: ProductFacts) -> TaxonomyResult:
        system = self._system_prompt()
        evidence = self._product_evidence(facts)
        native_step = getattr(self.client, "chat_tool_step", None)
        if callable(native_step):
            return self._run_native(facts, system, evidence)
        return self._run_json_actions(facts, system, evidence)

    def _system_prompt(self) -> str:
        return (
            "You are the reasoning controller of a marketplace taxonomy exploration agent. "
            "You do not receive the complete taxonomy. Explore it with the supplied read-only tools, "
            "and revise your approach from each observation. You may call several independent inspection "
            "tools in parallel after their shared prerequisite schema is observed, but never call finish in "
            "parallel with tools whose unseen results it depends on. Semantic category and "
            "attribute decisions are yours; tools perform only literal lookup, tree navigation, and exact "
            "validation. Do not guess IDs. Prefer verified source category, title, structured attributes, "
            "and SKU facts; do not use product identifiers or benchmark-specific memories. If a search is "
            "empty or ambiguous, change the query, browse the tree, inspect neighboring branches, or paginate. "
            "Before finish, inspect the chosen schema and every attribute/value used in mappings. Return one "
            "JSON object only: {\"action\": tool name, \"arguments\": object, \"reason\": concise evidence note}. "
            "Use action=finish only when grounded. Its arguments are: selected_category_id, "
            "selected_attribute_schema_category_id, confidence (0..1), evidence, and mappings. Each mapping "
            "must contain scope (product or sales), platform_attr_id, platform_value_id (empty only for a "
            "non-enum attribute), source_kind (product or sku), and source_name/source_value copied exactly "
            "from PRODUCT EVIDENCE. Omit uncertain mappings instead of guessing."
            + ("\n\n" + self.skill_instructions if self.skill_instructions else "")
        )

    def _run_native(
        self,
        facts: ProductFacts,
        system: str,
        evidence: dict[str, Any],
    ) -> TaxonomyResult:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    "Resolve the leaf category and platform attribute mappings. Start by calling "
                    "the exploration tool that best reduces uncertainty.\n\n"
                    f"PRODUCT EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}"
                ),
            }
        ]
        repeated_actions: dict[str, int] = {}
        for turn in range(1, self.max_turns + 1):
            assistant = self.client.chat_tool_step(
                system,
                messages,
                self.explorer.openai_tools(),
            )
            raw_calls = assistant.get("tool_calls")
            if not isinstance(raw_calls, list) or not raw_calls:
                raise TaxonomyAgentError("native taxonomy turn returned no tool call")
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant.get("content"),
                    "tool_calls": raw_calls,
                }
            )
            completed: TaxonomyResult | None = None
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
                    if repeated_actions[signature] > 2:
                        observation = TaxonomyExplorer._error(
                            action or "unknown",
                            "identical action repeated; revise the query or browse another node",
                        )
                    elif action == "finish":
                        completed, observation = self.explorer.finish(facts, arguments)
                    else:
                        observation = self.explorer.execute(action, arguments)
                self._trace(turn, action, arguments, observation)
                call_id = str(call.get("id") or f"taxonomy-{turn}-{call_index}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": action,
                        "content": json.dumps(observation, ensure_ascii=False),
                    }
                )
            if completed is not None:
                return completed
        raise TaxonomyAgentError(
            f"native taxonomy exploration did not finish within {self.max_turns} turns"
        )

    def _run_json_actions(
        self,
        facts: ProductFacts,
        system: str,
        evidence: dict[str, Any],
    ) -> TaxonomyResult:
        initial = self.explorer.execute("list_root_categories", {"limit": 60})
        history: list[dict[str, Any]] = [
            {"action": "list_root_categories", "arguments": {"limit": 60}, "observation": initial}
        ]
        repeated_actions: dict[str, int] = {}
        for turn in range(1, self.max_turns + 1):
            prompt = (
                "Continue the taxonomy exploration. Choose the next tool action from TOOL CATALOG.\n\n"
                f"PRODUCT EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}\n\n"
                f"TOOL CATALOG:\n{json.dumps(self.explorer.TOOL_CATALOG, ensure_ascii=False)}\n\n"
                f"ACTION/OBSERVATION HISTORY:\n{json.dumps(history, ensure_ascii=False)}"
            )
            response = self.client.chat_json(system, prompt)
            action = str(response.get("action") or "")
            arguments = response.get("arguments")
            signature = json.dumps(
                {"action": action, "arguments": arguments}, ensure_ascii=False, sort_keys=True
            )
            repeated_actions[signature] = repeated_actions.get(signature, 0) + 1
            if repeated_actions[signature] > 2:
                observation = TaxonomyExplorer._error(
                    action or "unknown", "identical action repeated; revise the query or browse another node"
                )
            elif action == "finish":
                result, observation = self.explorer.finish(facts, arguments)
                self._trace(turn, action, arguments, observation)
                if result is not None:
                    return result
            else:
                observation = self.explorer.execute(action, arguments)
                self._trace(turn, action, arguments, observation)
            history.append(
                {
                    "action": action,
                    "arguments": arguments if isinstance(arguments, dict) else {},
                    "reason": str(response.get("reason") or "")[:500],
                    "observation": observation,
                }
            )
        raise TaxonomyAgentError(
            f"taxonomy exploration did not produce a grounded finish within {self.max_turns} turns"
        )

    def _trace(self, turn: int, action: str, arguments: Any, observation: dict[str, Any]) -> None:
        if self.trace is not None:
            self.trace.emit(
                "taxonomy.react_step",
                turn=turn,
                action=action,
                arguments=arguments if isinstance(arguments, dict) else {},
                observation=observation,
            )
