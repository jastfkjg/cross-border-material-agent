from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SubmissionContractTests(unittest.TestCase):
    def test_cli_version_matches_manifest(self) -> None:
        manifest = json.loads((ROOT / "agent.json").read_text(encoding="utf-8"))
        completed = subprocess.run(
            [sys.executable, str(ROOT / "agent.py"), "--version"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), manifest["version"])


if __name__ == "__main__":
    unittest.main()
