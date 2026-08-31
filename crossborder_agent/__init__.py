"""Cross-border e-commerce localization material agent."""

from __future__ import annotations

import json
from pathlib import Path


def _manifest_version() -> str:
    """Load the one release version declared by the submission manifest."""

    manifest_path = Path(__file__).resolve().parents[1] / "agent.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = manifest["version"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"invalid or missing agent manifest: {manifest_path}") from exc
    if not isinstance(version, str):
        raise RuntimeError(f"agent manifest version must be a string: {manifest_path}")
    return version


VERSION = _manifest_version()
