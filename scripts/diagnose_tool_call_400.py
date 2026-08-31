#!/usr/bin/env python3
"""Reproduce and isolate the planning tool-call HTTP 400 with a small request budget.

The script runs the real ``BoundedDeliveryAgent.plan_delivery`` prompt, schemas,
and observations against an instrumented OpenAI-compatible client.  It records
only structural metadata.  If the baseline reproduces a 400, it sends a few
single-purpose variants of the failed request to distinguish transcript,
parallel-tool, null-content, and observation-size compatibility problems.

The API key is read from DASHSCOPE_API_KEY or from a hidden prompt.  It is never
accepted as a command-line argument and is never written to the report.
"""

from __future__ import annotations

import argparse
import copy
import getpass
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossborder_agent.agent_tools import BoundedToolRegistry  # noqa: E402
from crossborder_agent.api import ApiError  # noqa: E402
from crossborder_agent.bounded_agent import BoundedDeliveryAgent  # noqa: E402
from crossborder_agent.input_loader import load_product_facts  # noqa: E402
from crossborder_agent.models import (  # noqa: E402
    CategoryChoice,
    MappedAttribute,
    TaxonomyResult,
)


DEFAULT_BASE_URL = (
    "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)
RUNTIME_BUDGET_PREFIX = "RUNTIME BUDGET:"


class RequestBudgetExhausted(ApiError):
    pass


class ProbeHttpError(RuntimeError):
    def __init__(self, status: int | None, error: dict[str, Any]):
        super().__init__(f"HTTP {status or 'transport'}")
        self.status = status
        self.error = error


@dataclass
class DiagnosticState:
    max_requests: int
    requests: list[dict[str, Any]] = field(default_factory=list)
    failed_body: dict[str, Any] | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def reserve(self) -> int:
        if len(self.requests) >= self.max_requests:
            raise RequestBudgetExhausted(
                f"diagnostic request cap reached ({self.max_requests})",
                category="diagnostic_budget",
            )
        return len(self.requests) + 1


def _json_bytes(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def transcript_issues(messages: list[dict[str, Any]]) -> list[str]:
    """Return OpenAI tool-message pairing problems without exposing content."""

    issues: list[str] = []
    pending: set[str] = set()
    seen_ids: set[str] = set()
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            if pending:
                issues.append(
                    f"message[{index}] assistant appears before tool replies: {sorted(pending)}"
                )
                pending.clear()
            calls = message.get("tool_calls") or []
            for call in calls if isinstance(calls, list) else []:
                call_id = str(call.get("id") or "") if isinstance(call, dict) else ""
                if not call_id:
                    issues.append(f"message[{index}] contains a tool call without id")
                elif call_id in seen_ids:
                    issues.append(f"message[{index}] reuses tool_call_id {call_id}")
                else:
                    seen_ids.add(call_id)
                    pending.add(call_id)
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id not in pending:
                issues.append(
                    f"message[{index}] replies to unknown/non-pending tool_call_id {call_id}"
                )
            else:
                pending.remove(call_id)
        elif pending:
            issues.append(
                f"message[{index}] role={role} appears before tool replies: {sorted(pending)}"
            )
            pending.clear()
    if pending:
        issues.append(f"transcript ends before tool replies: {sorted(pending)}")
    return issues


def request_metadata(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages") or []
    tools = body.get("tools") or []
    role_counts: dict[str, int] = {}
    role_bytes: dict[str, int] = {}
    largest_tool_observation = 0
    assistant_tool_call_counts: list[int] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        size = _json_bytes(message)
        role_counts[role] = role_counts.get(role, 0) + 1
        role_bytes[role] = role_bytes.get(role, 0) + size
        if role == "tool":
            largest_tool_observation = max(largest_tool_observation, size)
        if role == "assistant":
            calls = message.get("tool_calls")
            assistant_tool_call_counts.append(len(calls) if isinstance(calls, list) else 0)
    return {
        "request_bytes": _json_bytes(body),
        "message_count": len(messages),
        "role_sequence": [
            str(item.get("role") or "unknown")
            for item in messages
            if isinstance(item, dict)
        ],
        "role_counts": role_counts,
        "role_bytes": role_bytes,
        "tool_schema_bytes": _json_bytes(tools),
        "tool_count": len(tools),
        "largest_tool_observation_bytes": largest_tool_observation,
        "assistant_tool_call_counts": assistant_tool_call_counts,
        "parallel_tool_calls": body.get("parallel_tool_calls", "absent"),
        "assistant_null_content_count": sum(
            1
            for item in messages
            if isinstance(item, dict)
            and item.get("role") == "assistant"
            and item.get("content") is None
        ),
        "tool_name_field_count": sum(
            1
            for item in messages
            if isinstance(item, dict)
            and item.get("role") == "tool"
            and "name" in item
        ),
        "protocol_issues": transcript_issues(messages),
    }


def _safe_provider_error(raw: bytes) -> dict[str, Any]:
    """Keep provider codes/messages, but never arbitrary response payloads."""

    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {"type": "non_json_error", "message": raw[:300].decode(errors="replace")}
    error = payload.get("error") if isinstance(payload, dict) else None
    error = error if isinstance(error, dict) else {}
    return {
        "code": str(error.get("code") or "")[:200],
        "type": str(error.get("type") or "")[:200],
        "param": str(error.get("param") or "")[:200],
        "message": str(error.get("message") or "")[:500],
        "request_id": str(payload.get("id") or "")[:200]
        if isinstance(payload, dict)
        else "",
    }


def _tool_names(message: dict[str, Any]) -> list[str]:
    result: list[str] = []
    calls = message.get("tool_calls")
    for call in calls if isinstance(calls, list) else []:
        function = call.get("function") if isinstance(call, dict) else None
        if isinstance(function, dict):
            result.append(str(function.get("name") or ""))
    return result


def post_chat_completion(
    *,
    api_key: str,
    endpoint: str,
    body: dict[str, Any],
    timeout: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    request = Request(
        endpoint,
        data=encoded,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "crossborder-tool-call-diagnostic/1.0",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        raise ProbeHttpError(exc.code, _safe_provider_error(exc.read())) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ProbeHttpError(
            None,
            {"type": "transport_error", "message": str(exc)[:500]},
        ) from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
        message = payload["choices"][0]["message"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ProbeHttpError(
            None,
            {"type": "invalid_success_response", "message": str(exc)[:500]},
        ) from exc
    if not isinstance(message, dict):
        raise ProbeHttpError(
            None,
            {"type": "invalid_success_response", "message": "message is not an object"},
        )
    usage = payload.get("usage") if isinstance(payload, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    return message, {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


class InstrumentedClient:
    """Small QwenClient-compatible adapter with no fallback or hidden retries."""

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        model: str,
        timeout: float,
        state: DiagnosticState,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout
        self.state = state
        self.config = SimpleNamespace(chat_model=model)
        self.http = SimpleNamespace(deadline=time.monotonic() + 600)

    def chat_tool_step(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        model: str | None = None,
        fallback_model: str | None = None,
    ) -> dict[str, Any]:
        del fallback_model
        body = {
            "model": model or self.config.chat_model,
            "messages": [{"role": "system", "content": system}, *messages],
            "tools": tools,
            "tool_choice": "required",
            "parallel_tool_calls": True,
            "temperature": 0.0,
            "enable_thinking": False,
        }
        request_number = self.state.reserve()
        metadata = request_metadata(body)
        started = time.monotonic()
        try:
            message, usage = post_chat_completion(
                api_key=self.api_key,
                endpoint=self.endpoint,
                body=body,
                timeout=self.timeout,
            )
        except ProbeHttpError as exc:
            record = {
                "number": request_number,
                "kind": "baseline",
                "http_status": exc.status,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "request": metadata,
                "provider_error": exc.error,
            }
            self.state.requests.append(record)
            if exc.status == 400:
                self.state.failed_body = copy.deepcopy(body)
            raise ApiError(
                f"HTTP {exc.status or 'transport'}: {exc.error}",
                status_code=exc.status,
                retryable=False,
                category="invalid_request" if exc.status == 400 else "transport",
            ) from exc
        self.state.total_prompt_tokens += usage["prompt_tokens"]
        self.state.total_completion_tokens += usage["completion_tokens"]
        self.state.requests.append(
            {
                "number": request_number,
                "kind": "baseline",
                "http_status": 200,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "request": metadata,
                "response_tool_names": _tool_names(message),
                "usage": usage,
            }
        )
        return message


def _load_trace_fixture(
    product_path: Path, debug_log: Path
) -> tuple[Any, TaxonomyResult, dict[str, Any]]:
    facts = load_product_facts(product_path)
    events: dict[str, dict[str, Any]] = {}
    with debug_log.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            if raw_line.startswith("TRACE_JSON "):
                raw_line = raw_line[len("TRACE_JSON ") :]
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            event = item.get("event") if isinstance(item, dict) else None
            if event in {"facts.reconciled", "taxonomy.resolved", "vision.source_review"}:
                events[str(event)] = item

    missing = sorted(
        {"facts.reconciled", "taxonomy.resolved", "vision.source_review"} - set(events)
    )
    if missing:
        raise ValueError(f"debug log lacks required events: {', '.join(missing)}")

    ledger = events["facts.reconciled"].get("ledger")
    facts.reconciled_fact_ledger = ledger if isinstance(ledger, dict) else {}

    taxonomy_event = events["taxonomy.resolved"]
    category_raw = taxonomy_event.get("category")
    if not isinstance(category_raw, dict):
        raise ValueError("taxonomy.resolved.category is not an object")
    category = CategoryChoice(
        category_id=str(category_raw.get("category_id") or ""),
        name=str(category_raw.get("name") or ""),
        path=str(category_raw.get("path") or ""),
        confidence=float(category_raw.get("confidence") or 0.0),
        method=str(category_raw.get("method") or "diagnostic-trace"),
        candidates=category_raw.get("candidates")
        if isinstance(category_raw.get("candidates"), list)
        else [],
    )
    attributes: list[MappedAttribute] = []
    for raw in taxonomy_event.get("mapped_attributes") or []:
        if not isinstance(raw, dict):
            continue
        attributes.append(
            MappedAttribute(
                attr_id=str(raw.get("attr_id") or ""),
                name=str(raw.get("name") or ""),
                source_name=str(raw.get("source_name") or ""),
                source_value=str(raw.get("source_value") or ""),
                source_evidence_pointer=str(raw.get("source_evidence_pointer") or ""),
                value_id=str(raw.get("value_id") or ""),
                platform_value=str(raw.get("platform_value") or ""),
                required=bool(raw.get("required")),
                sales_attribute=bool(raw.get("sales_attribute")),
            )
        )
    taxonomy = TaxonomyResult(
        category=category,
        attributes=attributes,
        missing_required=[str(item) for item in taxonomy_event.get("missing_required") or []],
    )
    vision = events["vision.source_review"].get("result")
    if not isinstance(vision, dict):
        raise ValueError("vision.source_review.result is not an object")
    return facts, taxonomy, vision


def make_probe_body(name: str, failed_body: dict[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(failed_body)
    messages = body.get("messages")
    messages = messages if isinstance(messages, list) else []
    if name == "fresh_history":
        system = next(
            (item for item in messages if isinstance(item, dict) and item.get("role") == "system"),
            {"role": "system", "content": "Use one available tool."},
        )
        body["messages"] = [
            system,
            {
                "role": "user",
                "content": (
                    "Diagnostic request: call exactly one inspection tool with valid arguments. "
                    "Do not submit the final delivery plan."
                ),
            },
        ]
    elif name == "without_parallel_tool_calls":
        body.pop("parallel_tool_calls", None)
    elif name == "without_budget_messages":
        body["messages"] = [
            item
            for item in messages
            if not (
                isinstance(item, dict)
                and item.get("role") == "user"
                and str(item.get("content") or "").startswith(RUNTIME_BUDGET_PREFIX)
            )
        ]
    elif name == "normalized_assistant_content":
        for item in messages:
            if (
                isinstance(item, dict)
                and item.get("role") == "assistant"
                and item.get("content") is None
            ):
                item["content"] = ""
    elif name == "without_tool_name_fields":
        for item in messages:
            if isinstance(item, dict) and item.get("role") == "tool":
                item.pop("name", None)
    elif name == "compacted_tool_observations":
        for item in messages:
            if not isinstance(item, dict) or item.get("role") != "tool":
                continue
            original = item.get("content")
            item["content"] = json.dumps(
                {
                    "ok": True,
                    "diagnostic_compacted": True,
                    "original_bytes": len(str(original).encode("utf-8")),
                },
                separators=(",", ":"),
            )
    else:
        raise ValueError(f"unknown probe: {name}")
    return body


def run_probe(
    name: str,
    failed_body: dict[str, Any],
    *,
    api_key: str,
    endpoint: str,
    timeout: float,
    max_tokens: int,
    state: DiagnosticState,
) -> dict[str, Any]:
    body = make_probe_body(name, failed_body)
    # Probes only need an HTTP acceptance/rejection signal.  Capping their
    # completion avoids paying for another full delivery-plan argument object.
    body["max_tokens"] = max_tokens
    request_number = state.reserve()
    started = time.monotonic()
    record: dict[str, Any] = {
        "number": request_number,
        "kind": "probe",
        "probe": name,
        "request": request_metadata(body),
    }
    try:
        message, usage = post_chat_completion(
            api_key=api_key,
            endpoint=endpoint,
            body=body,
            timeout=timeout,
        )
    except ProbeHttpError as exc:
        record.update(
            {
                "http_status": exc.status,
                "provider_error": exc.error,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )
    else:
        state.total_prompt_tokens += usage["prompt_tokens"]
        state.total_completion_tokens += usage["completion_tokens"]
        record.update(
            {
                "http_status": 200,
                "response_tool_names": _tool_names(message),
                "usage": usage,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )
    state.requests.append(record)
    return record


def _diagnose(requests: list[dict[str, Any]]) -> list[str]:
    probes = {
        str(item.get("probe")): item
        for item in requests
        if item.get("kind") == "probe"
    }
    conclusions: list[str] = []
    fresh = probes.get("fresh_history")
    if fresh and fresh.get("http_status") == 200:
        conclusions.append(
            "同一模型、工具 schema 和静态参数在清空历史后可用：故障依赖累计 transcript，而不是模型整体不可用。"
        )
    elif fresh:
        conclusions.append(
            "清空历史后仍失败：优先检查静态工具参数、工具 schema 或服务端模型工具能力。"
        )
    mapping = {
        "without_parallel_tool_calls": "移除 parallel_tool_calls 后成功，说明该兼容接口与并行工具参数/历史组合不兼容。",
        "without_budget_messages": "移除逐轮 RUNTIME BUDGET user 消息后成功，说明交错 user 消息破坏了服务端接受的工具回放序列。",
        "normalized_assistant_content": "将 assistant.content 从 null 规范化为空字符串后成功，说明接口不接受历史 tool-call assistant 的 null content。",
        "without_tool_name_fields": "移除 role=tool 消息中的 name 后成功，说明接口只接受 tool_call_id/content 的严格格式。",
        "compacted_tool_observations": "压缩 tool observation 后成功，说明请求长度或某个 observation 的大小/内容触发了内部限制。",
    }
    for name, text in mapping.items():
        probe = probes.get(name)
        if probe and probe.get("http_status") == 200:
            conclusions.append(text)
            break
    if probes and not any(item.get("http_status") == 200 for item in probes.values()):
        conclusions.append(
            "所有已执行变体均失败：现有单变量未定位，较可能是 provider 的工具服务故障或 schema/arguments 内部限制。"
        )
    return conclusions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--product-json",
        type=Path,
        default=REPO_ROOT / "Data_for_Users/product_info/product_5977010166484.json",
    )
    parser.add_argument(
        "--debug-log",
        type=Path,
        default=REPO_ROOT / "output/5977010166484_20260829_215232/agent_debug.jsonl",
    )
    parser.add_argument("--model", default="qwen3.8-max")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL or full /chat/completions URL",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--probe-max-tokens",
        type=int,
        default=128,
        help="completion cap for each post-failure probe (default: 128)",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=7,
        help="hard cap across baseline and probes (default: 7)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optional JSON report path; otherwise print to stdout",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate fixtures only; make no API calls",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_requests < 1 or args.max_requests > 12:
        raise SystemExit("--max-requests must be between 1 and 12")
    if args.probe_max_tokens < 16 or args.probe_max_tokens > 512:
        raise SystemExit("--probe-max-tokens must be between 16 and 512")
    facts, taxonomy, vision = _load_trace_fixture(args.product_json, args.debug_log)
    fixture = {
        "product_facts_bytes": _json_bytes(facts.compact_dict()),
        "taxonomy_attribute_count": len(taxonomy.attributes),
        "visual_evidence_bytes": _json_bytes(
            BoundedDeliveryAgent._compact_visual_evidence(vision)
        ),
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "fixture": fixture}, ensure_ascii=False, indent=2))
        return 0

    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key and sys.stdin.isatty():
        api_key = getpass.getpass("DASHSCOPE_API_KEY (hidden): ").strip()
    if not api_key:
        raise SystemExit(
            "DASHSCOPE_API_KEY is not set. Export it in your terminal or run interactively."
        )
    base_url = args.base_url.rstrip("/")
    endpoint = (
        base_url
        if base_url.endswith("/chat/completions")
        else f"{base_url}/chat/completions"
    )
    state = DiagnosticState(max_requests=args.max_requests)
    client = InstrumentedClient(
        api_key=api_key,
        endpoint=endpoint,
        model=args.model,
        timeout=args.timeout,
        state=state,
    )
    logger = logging.getLogger("tool-call-diagnostic")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    agent = BoundedDeliveryAgent(client, logger)
    baseline_exception = ""
    try:
        agent.plan_delivery(
            facts,
            taxonomy,
            vision,
            BoundedToolRegistry(),
            use_model=True,
        )
    except RequestBudgetExhausted as exc:
        baseline_exception = str(exc)

    failed_body = state.failed_body
    if failed_body is not None and len(state.requests) < state.max_requests:
        fresh = run_probe(
            "fresh_history",
            failed_body,
            api_key=api_key,
            endpoint=endpoint,
            timeout=args.timeout,
            max_tokens=args.probe_max_tokens,
            state=state,
        )
        if fresh.get("http_status") == 200:
            probe_order = [
                "compacted_tool_observations",
                "without_budget_messages",
                "normalized_assistant_content",
                "without_tool_name_fields",
            ]
        else:
            probe_order = [
                "without_parallel_tool_calls",
                "normalized_assistant_content",
                "without_tool_name_fields",
                "compacted_tool_observations",
            ]
        for name in probe_order:
            if len(state.requests) >= state.max_requests:
                break
            result = run_probe(
                name,
                failed_body,
                api_key=api_key,
                endpoint=endpoint,
                timeout=args.timeout,
                max_tokens=args.probe_max_tokens,
                state=state,
            )
            if result.get("http_status") == 200:
                break

    diagnosis = _diagnose(state.requests)
    if failed_body is None:
        diagnosis.append(
            "未在请求上限内复现 HTTP 400；本次运行不能证明历史故障已消失，建议保留报告并与失败运行比较。"
        )
    endpoint_parts = urlsplit(endpoint)
    report = {
        "model": args.model,
        "endpoint_origin": f"{endpoint_parts.scheme}://{endpoint_parts.netloc}",
        "fixture": fixture,
        "request_cap": state.max_requests,
        "requests_made": len(state.requests),
        "usage": {
            "prompt_tokens": state.total_prompt_tokens,
            "completion_tokens": state.total_completion_tokens,
        },
        "reproduced_http_400": failed_body is not None,
        "baseline_exception": baseline_exception,
        "requests": state.requests,
        "diagnosis": diagnosis,
        "secrets_recorded": False,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
        print(f"diagnostic report written to {args.report}")
    else:
        print(rendered)
    return 2 if failed_body is not None and not report["diagnosis"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
