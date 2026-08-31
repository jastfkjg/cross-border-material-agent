"""Restricted, composable filesystem tools for model-directed agents.

The workspace deliberately offers Pi-like primitives without giving a model an
ambient shell.  Paths stay below one explicit root, writes stay below staging,
and ``bash`` executes one allow-listed argv vector with a sanitized environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Any, Callable, Iterable


class AgentWorkspaceError(ValueError):
    """A requested workspace operation crossed a stable safety boundary."""


ObservationCallback = Callable[[str], None]


class BoundedAgentWorkspace:
    """A bounded read/search/command surface rooted at one local directory."""

    DEFAULT_PROGRAMS = ("rg", "jq", "find", "file", "ffprobe")
    MAX_READ_LINES = 400
    MAX_SEARCH_MATCHES = 200
    MAX_OUTPUT_BYTES = 64 * 1024
    MAX_WRITE_BYTES = 128 * 1024
    MAX_TIMEOUT_SECONDS = 30

    def __init__(
        self,
        root: Path,
        *,
        staging_dir: str = "staging",
        allowed_programs: Iterable[str] = DEFAULT_PROGRAMS,
        on_observation: ObservationCallback | None = None,
    ):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.staging_root = self._resolve(staging_dir, must_exist=False)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.allowed_programs = frozenset(str(item) for item in allowed_programs)
        self.on_observation = on_observation

    def _resolve(self, raw_path: str, *, must_exist: bool = True) -> Path:
        text = str(raw_path or ".").strip()
        candidate = Path(text)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise AgentWorkspaceError("path must be relative and stay inside the workspace")
        resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise AgentWorkspaceError("path escapes the workspace")
        if must_exist and not resolved.exists():
            raise AgentWorkspaceError(f"workspace path does not exist: {text}")
        return resolved

    @staticmethod
    def _relative(root: Path, path: Path) -> str:
        value = path.relative_to(root).as_posix()
        return value or "."

    def _observe(self, text: str) -> None:
        if self.on_observation is not None and text:
            self.on_observation(text)

    @classmethod
    def _bounded_text(cls, value: str, requested: int | None = None) -> tuple[str, bool]:
        maximum = cls.MAX_OUTPUT_BYTES
        if requested is not None:
            maximum = max(256, min(maximum, int(requested)))
        encoded = value.encode("utf-8", errors="replace")
        if len(encoded) <= maximum:
            return value, False
        return encoded[:maximum].decode("utf-8", errors="replace"), True

    def host_write_json(self, path: str, payload: Any, *, jsonl: bool = False) -> None:
        """Install host-owned evidence; this is not a model-authorized write."""

        destination = self._resolve(path, must_exist=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if jsonl:
            rows = payload if isinstance(payload, list) else []
            text = "\n".join(
                json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                for item in rows
            )
            if text:
                text += "\n"
        else:
            text = json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
        destination.write_text(text, encoding="utf-8")

    def list(self, arguments: dict[str, Any]) -> dict[str, Any]:
        target = self._resolve(str(arguments.get("path") or "."))
        if not target.is_dir():
            raise AgentWorkspaceError("list path must be a directory")
        recursive = arguments.get("recursive") is True
        limit = max(1, min(500, int(arguments.get("limit") or 200)))
        iterator = target.rglob("*") if recursive else target.iterdir()
        entries: list[dict[str, Any]] = []
        for path in sorted(iterator, key=lambda item: item.as_posix()):
            if len(entries) >= limit:
                break
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append(
                {
                    "path": self._relative(self.root, path),
                    "type": "directory" if path.is_dir() else "file",
                    "size_bytes": stat.st_size if path.is_file() else None,
                }
            )
        return {"ok": True, "entries": entries, "truncated": len(entries) >= limit}

    def read(self, arguments: dict[str, Any]) -> dict[str, Any]:
        target = self._resolve(str(arguments.get("path") or ""))
        if not target.is_file():
            raise AgentWorkspaceError("read path must be a file")
        offset = max(0, int(arguments.get("offset") or 0))
        limit = max(1, min(self.MAX_READ_LINES, int(arguments.get("limit") or 120)))
        if target.stat().st_size > 8 * 1024 * 1024:
            raise AgentWorkspaceError("read refuses files larger than 8MB; use file/ffprobe")
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except UnicodeError as exc:
            raise AgentWorkspaceError("read accepts UTF-8 text only; use file/ffprobe for media") from exc
        selected = lines[offset : offset + limit]
        content, byte_truncated = self._bounded_text("\n".join(selected))
        self._observe(content)
        next_offset = offset + len(selected)
        return {
            "ok": True,
            "path": self._relative(self.root, target),
            "offset": offset,
            "next_offset": next_offset if next_offset < len(lines) else None,
            "total_lines": len(lines),
            "content": content,
            "truncated": byte_truncated or next_offset < len(lines),
        }

    def search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        pattern = str(arguments.get("pattern") or "")
        if not pattern:
            raise AgentWorkspaceError("search pattern must be non-empty")
        try:
            expression = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise AgentWorkspaceError(f"invalid search regex: {exc}") from exc
        raw_paths = arguments.get("paths")
        paths = raw_paths if isinstance(raw_paths, list) and raw_paths else ["."]
        limit = max(1, min(self.MAX_SEARCH_MATCHES, int(arguments.get("limit") or 80)))
        candidates: list[Path] = []
        for raw_path in paths[:20]:
            target = self._resolve(str(raw_path or "."))
            if target.is_file():
                candidates.append(target)
            elif target.is_dir():
                candidates.extend(path for path in target.rglob("*") if path.is_file())
        matches: list[dict[str, Any]] = []
        observation_parts: list[str] = []
        for path in sorted(set(candidates), key=lambda item: item.as_posix()):
            if len(matches) >= limit:
                break
            try:
                if path.stat().st_size > 8 * 1024 * 1024:
                    continue
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if expression.search(line) is None:
                    continue
                bounded_line, _ = self._bounded_text(line, 4000)
                matches.append(
                    {
                        "path": self._relative(self.root, path),
                        "line": line_number,
                        "text": bounded_line,
                    }
                )
                observation_parts.append(bounded_line)
                if len(matches) >= limit:
                    break
        self._observe("\n".join(observation_parts))
        return {"ok": True, "matches": matches, "truncated": len(matches) >= limit}

    @staticmethod
    def _unsafe_command_token(token: str) -> bool:
        if token.startswith("/") or ".." in Path(token).parts:
            return True
        return any(marker in token for marker in ("\n", "\r", "\x00", "`", "$("))

    @staticmethod
    def _program_arguments_are_safe(program: str, arguments: list[str]) -> bool:
        lowered = [item.casefold() for item in arguments]
        if program == "rg" and any(
            original == "-L"
            or item in {"--follow", "--pre"}
            or item.startswith("--pre=")
            for original, item in zip(arguments, lowered)
        ):
            return False
        if program == "find" and any(
            original == "-L"
            or item.startswith(
                ("-exec", "-ok", "-delete", "-fls", "-fprintf", "-fprint")
            )
            for original, item in zip(arguments, lowered)
        ):
            return False
        if program == "file" and any(
            original == "-L" or item == "--dereference"
            for original, item in zip(arguments, lowered)
        ):
            return False
        if program == "ffprobe" and any(
            "://" in item
            or item.startswith(("concat:", "crypto:", "data:", "file:", "pipe:", "subfile:"))
            for item in lowered
        ):
            return False
        return True

    def bash(self, arguments: dict[str, Any]) -> dict[str, Any]:
        command = str(arguments.get("command") or "").strip()
        if not command:
            raise AgentWorkspaceError("bash command must be non-empty")
        try:
            argv = shlex.split(command, posix=True)
        except ValueError as exc:
            raise AgentWorkspaceError(f"invalid command quoting: {exc}") from exc
        if not argv:
            raise AgentWorkspaceError("bash command must be non-empty")
        program = Path(argv[0]).name
        if argv[0] != program or program not in self.allowed_programs:
            raise AgentWorkspaceError(
                "command is not allowed; available programs: "
                + ", ".join(sorted(self.allowed_programs))
            )
        shell_operators = {"|", "||", "&&", ";", ">", ">>", "<", "2>", "&"}
        if (
            any(token in shell_operators or self._unsafe_command_token(token) for token in argv[1:])
            or not self._program_arguments_are_safe(program, argv[1:])
        ):
            raise AgentWorkspaceError(
                "shell operators, absolute paths, parent traversal, and substitutions are not allowed"
            )
        executable = shutil.which(program)
        if executable is None:
            raise AgentWorkspaceError(f"allowed program is unavailable: {program}")
        cwd = self._resolve(str(arguments.get("cwd") or "."))
        if not cwd.is_dir():
            raise AgentWorkspaceError("bash cwd must be a directory")
        if any(path.is_symlink() for path in self.root.rglob("*")):
            raise AgentWorkspaceError(
                "bash is unavailable while the workspace contains symbolic links"
            )
        timeout = max(
            1,
            min(self.MAX_TIMEOUT_SECONDS, int(arguments.get("timeout") or 10)),
        )
        max_output = max(
            256,
            min(self.MAX_OUTPUT_BYTES, int(arguments.get("max_output") or 16000)),
        )
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        try:
            safe_argv = [executable, *argv[1:]]
            if program == "ffprobe":
                safe_argv[1:1] = ["-protocol_whitelist", "file,crypto,data"]
            completed = subprocess.run(
                safe_argv,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
                close_fds=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise AgentWorkspaceError(f"command exceeded {timeout}s timeout") from exc
        stdout, stdout_truncated = self._bounded_text(
            completed.stdout.decode("utf-8", errors="replace"), max_output
        )
        remaining = max(256, max_output - len(stdout.encode("utf-8")))
        stderr, stderr_truncated = self._bounded_text(
            completed.stderr.decode("utf-8", errors="replace"), remaining
        )
        self._observe(stdout)
        return {
            "ok": completed.returncode == 0,
            "argv": [program, *argv[1:]],
            "cwd": self._relative(self.root, cwd),
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": stdout_truncated or stderr_truncated,
        }

    def write_staging(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(arguments.get("path") or "")
        content = str(arguments.get("content") or "")
        if not raw_path:
            raise AgentWorkspaceError("staging path must be non-empty")
        if len(content.encode("utf-8")) > self.MAX_WRITE_BYTES:
            raise AgentWorkspaceError("staging write exceeds 128KB")
        candidate = Path(raw_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise AgentWorkspaceError("staging path must be relative")
        destination = (self.staging_root / candidate).resolve()
        if destination != self.staging_root and self.staging_root not in destination.parents:
            raise AgentWorkspaceError("staging path escapes the staging directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        return {
            "ok": True,
            "path": self._relative(self.root, destination),
            "size_bytes": len(content.encode("utf-8")),
        }

    def execute(self, name: str, arguments: Any) -> dict[str, Any]:
        args = arguments if isinstance(arguments, dict) else {}
        handler = {
            "list": self.list,
            "read": self.read,
            "search": self.search,
            "bash": self.bash,
            "write_staging": self.write_staging,
        }.get(name)
        if handler is None:
            return {"ok": False, "tool": name, "error": f"unknown workspace tool: {name}"}
        try:
            result = handler(args)
            result.setdefault("tool", name)
            return result
        except (AgentWorkspaceError, OSError, ValueError) as exc:
            return {"ok": False, "tool": name, "error": str(exc)[:2000]}

    @classmethod
    def tool_catalog(cls) -> tuple[dict[str, str], ...]:
        return (
            {"name": "list", "description": "List files and directories inside the bounded workspace."},
            {"name": "read", "description": "Read a bounded line range from one UTF-8 workspace file."},
            {"name": "search", "description": "Regex-search bounded workspace files and return matching lines."},
            {"name": "bash", "description": "Run one restricted rg/jq/find/file/ffprobe command without a shell."},
            {"name": "write_staging", "description": "Write model notes or proposed data only inside staging."},
        )

    @classmethod
    def openai_tools(cls) -> list[dict[str, Any]]:
        def definition(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }

        catalog = {item["name"]: item["description"] for item in cls.tool_catalog()}
        return [
            definition(
                "list",
                catalog["list"],
                {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                },
                ["path"],
            ),
            definition(
                "read",
                catalog["read"],
                {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": cls.MAX_READ_LINES},
                },
                ["path"],
            ),
            definition(
                "search",
                catalog["search"],
                {
                    "pattern": {"type": "string", "minLength": 1},
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 20,
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": cls.MAX_SEARCH_MATCHES},
                },
                ["pattern"],
            ),
            definition(
                "bash",
                catalog["bash"],
                {
                    "command": {"type": "string", "minLength": 1},
                    "cwd": {"type": "string"},
                    "timeout": {"type": "integer", "minimum": 1, "maximum": cls.MAX_TIMEOUT_SECONDS},
                    "max_output": {"type": "integer", "minimum": 256, "maximum": cls.MAX_OUTPUT_BYTES},
                },
                ["command"],
            ),
            definition(
                "write_staging",
                catalog["write_staging"],
                {
                    "path": {"type": "string", "minLength": 1},
                    "content": {"type": "string", "maxLength": cls.MAX_WRITE_BYTES},
                },
                ["path", "content"],
            ),
        ]
