"""Compile concise, stage-scoped rules from the local competition skills."""

from __future__ import annotations

from pathlib import Path
import re


_RULE_RE = re.compile(
    r"^- \[(?P<stages>[a-z0-9, -]+)\]\[(?P<level>hard|soft)\]\s*(?P<text>.+)$"
)

_VALID_STAGES = {
    "manager",
    "taxonomy",
    "source-vision",
    "creative-plan",
    "copy",
    "final-review",
}

# These are direct task.md contract clauses or necessary disambiguations of
# that contract. They are injected before skill rules and therefore cannot be
# weakened by a domain skill.
_TASK_RULES: dict[str, tuple[tuple[str, str], ...]] = {
    "all": (
        (
            "hard",
            "The supplied task contract and platform snapshot take precedence over every domain preference.",
        ),
    ),
    "taxonomy": (
        ("hard", "Output one supplied leaf category and exact supplied attribute, value and SKU identifiers."),
    ),
    "copy": (
        (
            "hard",
            "Each locale must include title, product attributes, SKU breakdown, source platform, product ID and URL, and descriptions for every required image and video.",
        ),
        (
            "hard",
            "The required product source URL is allowed only in the machine-oriented listing appendix; do not treat it as off-platform marketing or remove it.",
        ),
    ),
    "creative-plan": (
        ("hard", "Plan exactly one hero, five detail images and one playable product video using the required filenames."),
    ),
    "final-review": (
        (
            "hard",
            "Judge A1 only within the task boundary: content compliance excludes independent legal and intellectual-property adjudication.",
        ),
        (
            "hard",
            "The required source URL in the machine appendix is contract evidence, not an off-platform-link violation.",
        ),
    ),
}


class SkillLibrary:
    """Load skills once and compile only rules relevant to the active stage."""

    def __init__(self, root: Path | None = None):
        self.root = root or Path(__file__).resolve().parent / "skills"
        self._cache: dict[str, tuple[tuple[frozenset[str], str, str], ...]] = {}

    def load(self, name: str) -> tuple[tuple[frozenset[str], str, str], ...]:
        if name in self._cache:
            return self._cache[name]
        path = self.root / name / "SKILL.md"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"skill unavailable: {name}") from exc
        rules: list[tuple[frozenset[str], str, str]] = []
        for raw_line in text.splitlines():
            match = _RULE_RE.fullmatch(raw_line.strip())
            if not match:
                continue
            stages = frozenset(
                item.strip() for item in match.group("stages").split(",")
            )
            unknown = stages - (_VALID_STAGES | {"all"})
            if unknown:
                raise RuntimeError(
                    f"skill {name} contains unknown stages: {sorted(unknown)}"
                )
            rules.append((stages, match.group("level"), match.group("text").strip()))
        if not rules:
            raise RuntimeError(f"skill contains no compilable rules: {name}")
        compiled = tuple(rules)
        self._cache[name] = compiled
        return compiled

    def compile(self, stage: str, *names: str) -> str:
        """Return a compact hard-rule/preference capsule for one model call."""

        if stage not in _VALID_STAGES:
            raise ValueError(f"unknown skill compilation stage: {stage}")
        selected: list[tuple[str, str]] = [*_TASK_RULES.get("all", ())]
        selected.extend(_TASK_RULES.get(stage, ()))
        for name in names:
            for stages, level, text in self.load(name):
                if "all" in stages or stage in stages:
                    selected.append((level, text))

        deduplicated: list[tuple[str, str]] = []
        seen: set[str] = set()
        for level, text in selected:
            key = re.sub(r"\s+", " ", text).strip().casefold()
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append((level, text))

        hard = [text for level, text in deduplicated if level == "hard"]
        soft = [text for level, text in deduplicated if level == "soft"]
        lines = [f"Applicable compiled policy for stage: {stage}", "Hard requirements:"]
        lines.extend(f"- {item}" for item in hard)
        if soft:
            lines.append("Quality preferences (never override hard requirements):")
            lines.extend(f"- {item}" for item in soft)
        return "\n".join(lines)
