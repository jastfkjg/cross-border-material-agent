#!/usr/bin/env python3
"""Competition entry point for the cross-border material agent."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))


def _prepare_vendor_binaries() -> None:
    """Restore executable bits even when the ZIP extractor ignores Unix metadata."""

    binary_dir = VENDOR / "imageio_ffmpeg" / "binaries"
    if not binary_dir.is_dir():
        return
    for candidate in binary_dir.glob("ffmpeg-*"):
        try:
            candidate.chmod(candidate.stat().st_mode | 0o111)
        except OSError:
            continue


_prepare_vendor_binaries()
sys.path.insert(0, str(ROOT))

from crossborder_agent.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
