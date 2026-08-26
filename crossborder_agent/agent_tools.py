"""Typed tool registry used by the bounded delivery agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    targets: tuple[str, ...]
    estimated_seconds: int
    retry_safe: bool = True
    side_effects: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "allowed_targets": list(self.targets),
            "estimated_seconds": self.estimated_seconds,
            "retry_safe": self.retry_safe,
            "side_effects": self.side_effects,
        }


@dataclass(slots=True)
class ToolExecution:
    status: str
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)


ToolCallback = Callable[[str, str], ToolExecution]


class BoundedToolRegistry:
    """Expose only explicitly registered, reversible repair operations."""

    def __init__(self):
        self._specs: dict[str, ToolSpec] = {}
        self._callbacks: dict[str, ToolCallback] = {}

    def register(self, spec: ToolSpec, callback: ToolCallback) -> None:
        self.add_spec(spec)
        self.bind(spec.name, callback)

    def add_spec(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._specs[spec.name] = spec

    def bind(self, name: str, callback: ToolCallback) -> None:
        if name not in self._specs:
            raise ValueError(f"unknown tool: {name}")
        self._callbacks[name] = callback

    def catalog(self) -> list[dict[str, Any]]:
        return [self._specs[name].to_dict() for name in sorted(self._specs)]

    def accepts(self, tool: str, target: str) -> bool:
        spec = self._specs.get(tool)
        return spec is not None and target in spec.targets

    def estimated_seconds(self, tool: str) -> int:
        spec = self._specs.get(tool)
        return spec.estimated_seconds if spec is not None else 0

    def execute(self, tool: str, target: str, instruction: str) -> ToolExecution:
        if not self.accepts(tool, target):
            return ToolExecution("rejected", f"tool/target not allowed: {tool}/{target}")
        if tool not in self._callbacks:
            return ToolExecution("rejected", f"tool is not bound: {tool}")
        try:
            return self._callbacks[tool](target, instruction)
        except Exception as exc:  # caller records the bounded failure and keeps prior asset
            return ToolExecution("failed", str(exc))
