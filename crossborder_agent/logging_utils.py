"""Logging helpers that comply with AGENT_LOG_DIR."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def configure_logging() -> logging.Logger:
    log_dir = Path(os.environ.get("AGENT_LOG_DIR", ".agent-logs")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "agent.log"

    logger = logging.getLogger("crossborder_agent")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
