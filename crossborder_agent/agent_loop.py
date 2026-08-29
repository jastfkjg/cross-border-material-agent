"""Minimal native tool-calling loop for the delivery orchestrator.

The loop deliberately mirrors Pi's small control surface: the model chooses a
tool, code validates and executes it, and the observation is appended as a
``role=tool`` message before the next model turn.  Domain workflow belongs in
tool implementations and model-visible evidence, not in this loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any, Callable


@dataclass(slots=True)
class AgentToolOutcome:
    observation: dict[str, Any]
    terminate: bool = False


ToolHandler = Callable[[dict[str, Any]], AgentToolOutcome | dict[str, Any]]


@dataclass(frozen=True, slots=True)
class AgentLoopTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler = field(compare=False, repr=False)
    terminal: bool = False

    def openai_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class AgentLoopResult:
    messages: list[dict[str, Any]]
    turns: int
    stop_reason: str
    final_observation: dict[str, Any] = field(default_factory=dict)


def _validate_arguments(schema: dict[str, Any], value: Any, path: str = "arguments") -> None:
    """Validate the small JSON-Schema subset used by this project's tools."""

    expected = schema.get("type")
    valid_type = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
    }.get(expected)
    if valid_type is not None and not valid_type(value):
        raise ValueError(f"{path} must be {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']}")
    if isinstance(value, dict):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        for name in schema.get("required") or []:
            if name not in value:
                raise ValueError(f"{path}.{name} is required")
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                raise ValueError(f"{path} contains unknown fields: {unknown}")
        for name, item in value.items():
            child = properties.get(name)
            if isinstance(child, dict):
                _validate_arguments(child, item, f"{path}.{name}")
    elif isinstance(value, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            raise ValueError(f"{path} requires at least {minimum} items")
        if isinstance(maximum, int) and len(value) > maximum:
            raise ValueError(f"{path} allows at most {maximum} items")
        if schema.get("uniqueItems") is True:
            serialized = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in value]
            if len(serialized) != len(set(serialized)):
                raise ValueError(f"{path} items must be unique")
        child = schema.get("items")
        if isinstance(child, dict):
            for index, item in enumerate(value):
                _validate_arguments(child, item, f"{path}[{index}]")
    elif isinstance(value, str):
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if isinstance(minimum, int) and len(value) < minimum:
            raise ValueError(f"{path} requires at least {minimum} characters")
        if isinstance(maximum, int) and len(value) > maximum:
            raise ValueError(f"{path} allows at most {maximum} characters")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            raise ValueError(f"{path} must be at least {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            raise ValueError(f"{path} must be at most {maximum}")


def _normalize_single_tool_call(raw_call: Any, fallback_id: str) -> dict[str, Any]:
    """Return the strict, single-call shape safe for provider history replay."""

    call = raw_call if isinstance(raw_call, dict) else {}
    function = call.get("function")
    function = function if isinstance(function, dict) else {}
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(
            arguments if isinstance(arguments, dict) else {},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return {
        "id": str(call.get("id") or fallback_id),
        "type": "function",
        "function": {
            "name": str(function.get("name") or ""),
            "arguments": arguments,
        },
    }


class NativeToolAgentLoop:
    """Run an OpenAI-compatible function-calling loop with bounded host authority."""

    def __init__(
        self,
        client: Any,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]] | None = None,
        trace: Any = None,
    ):
        self.client = client
        self.system_prompt = system_prompt
        self.messages = list(messages or [])
        self.trace = trace

    def run(
        self,
        prompt: str,
        tools: list[AgentLoopTool],
        *,
        max_turns: int,
        deadline: float,
        reserve_seconds: float = 0.0,
    ) -> AgentLoopResult:
        if max_turns < 1:
            raise ValueError("agent loop requires at least one turn")
        if not tools:
            raise ValueError("agent loop requires at least one tool")
        by_name = {tool.name: tool for tool in tools}
        if len(by_name) != len(tools):
            raise ValueError("agent loop tool names must be unique")
        self.messages.append({"role": "user", "content": prompt})
        repeated: dict[str, int] = {}
        final_observation: dict[str, Any] = {}

        for turn in range(1, max_turns + 1):
            remaining_seconds = max(0.0, deadline - time.monotonic())
            if remaining_seconds <= reserve_seconds:
                return AgentLoopResult(
                    messages=list(self.messages),
                    turns=turn - 1,
                    stop_reason="deadline-reserve",
                    final_observation=final_observation,
                )
            budget_message = {
                "role": "user",
                "content": (
                    f"RUNTIME BUDGET: turn {turn}/{max_turns}; "
                    f"approximately {remaining_seconds:.0f} seconds remain. "
                    "Choose the highest-value grounded action. Finish as soon as the delivery state is sufficient."
                ),
            }
            self.messages.append(budget_message)
            assistant = self.client.chat_tool_step(
                self.system_prompt,
                self.messages,
                [tool.openai_definition() for tool in tools],
            )
            calls = assistant.get("tool_calls")
            if not isinstance(calls, list) or not calls:
                observation = {
                    "ok": False,
                    "error": "assistant returned no tool call",
                }
                self.messages.append(
                    {"role": "user", "content": json.dumps(observation)}
                )
                continue
            raw_call_count = len(calls)
            # Defense in depth: some compatible providers ignore
            # parallel_tool_calls=false.  Replaying multiple distinct call IDs
            # from one assistant turn is rejected by the configured endpoint,
            # so serialize the batch and let the model request remaining tools
            # on later turns.  Rebuild the call to remove provider-only fields
            # and guarantee that assistant/tool messages share a non-empty ID.
            calls = [_normalize_single_tool_call(calls[0], f"agent-{turn}-0")]
            if raw_call_count > 1 and self.trace is not None:
                self.trace.emit(
                    "orchestrator.parallel_tool_calls_serialized",
                    turn=turn,
                    received_call_count=raw_call_count,
                    retained_tool=calls[0]["function"]["name"],
                )
            self.messages.append(
                {
                    "role": "assistant",
                    "content": assistant.get("content"),
                    "tool_calls": calls,
                }
            )
            outcomes: list[AgentToolOutcome] = []
            for call_index, raw_call in enumerate(calls):
                call = raw_call if isinstance(raw_call, dict) else {}
                function = call.get("function")
                function = function if isinstance(function, dict) else {}
                name = str(function.get("name") or "")
                raw_arguments = function.get("arguments")
                try:
                    arguments = (
                        json.loads(raw_arguments)
                        if isinstance(raw_arguments, str)
                        else raw_arguments
                    )
                except json.JSONDecodeError as exc:
                    arguments = {}
                    outcome = AgentToolOutcome(
                        {"ok": False, "error": f"invalid JSON arguments: {exc}"}
                    )
                else:
                    arguments = arguments if isinstance(arguments, dict) else {}
                    signature = json.dumps(
                        {"name": name, "arguments": arguments},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    repeated[signature] = repeated.get(signature, 0) + 1
                    tool = by_name.get(name)
                    if tool is None:
                        outcome = AgentToolOutcome(
                            {"ok": False, "error": f"unknown tool: {name}"}
                        )
                    elif repeated[signature] > 2:
                        outcome = AgentToolOutcome(
                            {
                                "ok": False,
                                "error": "identical tool call repeated; change the action or finish",
                            }
                        )
                    elif tool.terminal and len(calls) > 1:
                        outcome = AgentToolOutcome(
                            {
                                "ok": False,
                                "error": "terminal tools must be called alone so no mutation can occur after acceptance",
                            }
                        )
                    else:
                        try:
                            _validate_arguments(tool.parameters, arguments)
                            raw_outcome = tool.handler(arguments)
                            outcome = (
                                raw_outcome
                                if isinstance(raw_outcome, AgentToolOutcome)
                                else AgentToolOutcome(raw_outcome)
                            )
                        except Exception as exc:  # tool failures are observations
                            outcome = AgentToolOutcome(
                                {"ok": False, "error": str(exc)[:2000]}
                            )
                observation = dict(outcome.observation)
                observation.setdefault("ok", True)
                observation["runtime"] = {
                    "turn": turn,
                    "max_turns": max_turns,
                    "remaining_seconds": round(
                        max(0.0, deadline - time.monotonic()), 1
                    ),
                }
                final_observation = observation
                call_id = str(call.get("id") or f"agent-{turn}-{call_index}")
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(observation, ensure_ascii=False),
                    }
                )
                if self.trace is not None:
                    self.trace.emit(
                        "orchestrator.tool_result",
                        turn=turn,
                        tool=name,
                        arguments=arguments,
                        observation=observation,
                        terminate=outcome.terminate,
                    )
                outcomes.append(outcome)
            # Match Pi's safe batch semantics: a turn terminates only when every
            # finalized call is terminal. A mixed batch always returns to the
            # model with observations.
            if outcomes and all(outcome.terminate for outcome in outcomes):
                return AgentLoopResult(
                    messages=list(self.messages),
                    turns=turn,
                    stop_reason="finish-tool",
                    final_observation=final_observation,
                )

        return AgentLoopResult(
            messages=list(self.messages),
            turns=max_turns,
            stop_reason="max-turns",
            final_observation=final_observation,
        )
