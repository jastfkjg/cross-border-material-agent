"""Structured, redacted debug tracing that is inert unless explicitly enabled."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from typing import Any
from urllib.parse import urlsplit, urlunsplit


_SECRET_KEYS = {
    "api_key",
    "authorization",
    "token",
    "access_token",
    "secret",
    "signature",
}


def _safe_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "[invalid-url]"
    if parsed.scheme not in {"http", "https"}:
        return value
    clean = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{clean}#url-sha256={digest}"


def _sanitize(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return "[depth-limited]"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for raw_key, child in list(value.items())[:120]:
            key = str(raw_key)
            if key.casefold() in _SECRET_KEYS:
                result[key] = "[redacted]"
            else:
                result[key] = _sanitize(child, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, depth=depth + 1) for item in list(value)[:120]]
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            return _safe_url(value)
        if value.startswith("data:"):
            return f"[data-url bytes={len(value)}]"
        return value[:8000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:2000]


class DebugTrace:
    """Emit parseable TRACE_JSON lines into agent.log when debug is enabled."""

    def __init__(self, logger: logging.Logger, *, enabled: bool):
        self.logger = logger
        self.enabled = enabled
        self.run_id = uuid.uuid4().hex
        self.started = time.monotonic()
        self._sequence = 0
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        # Event producers historically used ``elapsed_seconds`` for operation
        # duration, which overwrote the run timeline field during ``**fields``
        # expansion. Keep the two clocks explicit and collision-free.
        operation_elapsed = fields.pop("elapsed_seconds", None)
        payload = {
            "run_id": self.run_id,
            "seq": sequence,
            "run_elapsed_seconds": round(time.monotonic() - self.started, 3),
            "event": event,
            **fields,
        }
        if operation_elapsed is not None:
            payload["operation_duration_seconds"] = operation_elapsed
        self.logger.debug(
            "TRACE_JSON %s",
            json.dumps(_sanitize(payload), ensure_ascii=False, separators=(",", ":")),
        )
