"""Load the small, local skill library shipped with the competition agent."""

from __future__ import annotations

from pathlib import Path


class SkillLibrary:
    """Progressively load task-specific instructions without external retrieval."""

    def __init__(self, root: Path | None = None):
        self.root = root or Path(__file__).resolve().parent / "skills"
        self._cache: dict[str, str] = {}

    def load(self, name: str) -> str:
        if name in self._cache:
            return self._cache[name]
        path = self.root / name / "SKILL.md"
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(f"skill unavailable: {name}") from exc
        if not text:
            raise RuntimeError(f"skill is empty: {name}")
        self._cache[name] = text
        return text

    def combine(self, *names: str) -> str:
        return "\n\n---\n\n".join(self.load(name) for name in names)

