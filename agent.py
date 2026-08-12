#!/usr/bin/env python3
"""Competition entry point for the cross-border material agent."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT))

from crossborder_agent.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
