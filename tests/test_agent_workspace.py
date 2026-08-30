from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crossborder_agent.agent_workspace import BoundedAgentWorkspace


class BoundedAgentWorkspaceTests(unittest.TestCase):
    def test_read_search_and_staging_are_composable(self) -> None:
        observed: list[str] = []
        with tempfile.TemporaryDirectory() as temporary:
            workspace = BoundedAgentWorkspace(
                Path(temporary), on_observation=observed.append
            )
            workspace.host_write_json(
                "evidence/records.jsonl",
                [{"ref": "record/1", "name": "grounded"}],
                jsonl=True,
            )

            listing = workspace.execute("list", {"path": "evidence"})
            search = workspace.execute(
                "search",
                {
                    "pattern": "grounded",
                    "paths": ["evidence/records.jsonl"],
                },
            )
            write = workspace.execute(
                "write_staging", {"path": "notes.txt", "content": "candidate"}
            )

            self.assertTrue(listing["ok"])
            self.assertTrue(search["ok"])
            self.assertEqual(search["matches"][0]["text"], observed[-1])
            self.assertTrue(write["ok"])
            self.assertTrue((Path(temporary) / "staging/notes.txt").is_file())

    def test_paths_and_commands_cannot_escape_or_mutate_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = BoundedAgentWorkspace(Path(temporary))
            outside = workspace.execute("read", {"path": "../secret"})
            shell = workspace.execute(
                "bash", {"command": "rg evidence | jq .", "cwd": "."}
            )
            find_exec = workspace.execute(
                "bash", {"command": "find . -exec file {} ;", "cwd": "."}
            )
            remote_probe = workspace.execute(
                "bash", {"command": "ffprobe https://example.test/video.mp4"}
            )

            self.assertFalse(outside["ok"])
            self.assertFalse(shell["ok"])
            self.assertFalse(find_exec["ok"])
            self.assertFalse(remote_probe["ok"])


if __name__ == "__main__":
    unittest.main()
