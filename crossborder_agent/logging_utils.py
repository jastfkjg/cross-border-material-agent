"""Logging helpers that comply with AGENT_LOG_DIR."""

from __future__ import annotations

import logging
import os
from pathlib import Path


class _TraceFilter(logging.Filter):
    def __init__(self, *, trace_only: bool):
        super().__init__()
        self.trace_only = trace_only

    def filter(self, record: logging.LogRecord) -> bool:
        is_trace = record.getMessage().startswith("TRACE_JSON ")
        return is_trace if self.trace_only else not is_trace


def configure_logging(*, debug: bool = False) -> logging.Logger:
    log_dir = Path(os.environ.get("AGENT_LOG_DIR", ".agent-logs")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "agent.log"

    logger = logging.getLogger("crossborder_agent")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except OSError:
            pass

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_TraceFilter(trace_only=False))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(_TraceFilter(trace_only=False))
    logger.addHandler(stream_handler)
    if debug:
        debug_handler = logging.FileHandler(
            log_dir / "agent_debug.jsonl", encoding="utf-8"
        )
        debug_handler.setFormatter(logging.Formatter("%(message)s"))
        debug_handler.addFilter(_TraceFilter(trace_only=True))
        logger.addHandler(debug_handler)
    return logger
