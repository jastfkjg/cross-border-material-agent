"""Shared contracts and small pure helpers for pipeline implementation blocks."""

from __future__ import annotations

import re
from typing import Any

from ..api import ApiError
from ..qa import EXPECTED_FILES


class PipelineError(RuntimeError):
    """Raised when the agent cannot produce a complete, validated delivery."""


class SemanticRejection(ApiError):
    """All candidates contain a concrete product-identity or compliance defect."""

    def __init__(self, message: str, *, feedback: str = ""):
        super().__init__(message, retryable=True, category="semantic_rejection")
        self.feedback = feedback or message


IMAGE_NEGATIVE_PROMPT = (
    "written text, letters, numbers, watermark, logo, brand mark, price tag, promotional badge, "
    "unreadable typography, distorted anatomy, extra limbs, malformed hands, product deformation, "
    "changed buttons, changed fasteners, changed pattern, changed color, blur, low resolution"
)

SINGLE_COMPOSITION_NEGATIVE_PROMPT = ", split screen, inset panel, repeated panel, mixed close-up and full-product composition"

MAIN_NEGATIVE_PROMPT = (
    IMAGE_NEGATIVE_PROMPT
    + ", collage, montage, split screen, inset, duplicate product, multiple products, unsupported variants, "
    "cropped product, person, mannequin body"
)

VIDEO_NEGATIVE_PROMPT = (
    "product morphing, changed product construction, changed color, changed pattern, added or removed components, "
    "changed fastenings or trims, duplicate product, extra product, warped material, flicker, scene cut, camera shake, "
    "hands covering product, text, subtitles, watermark, logo animation, speech, music"
)

AGENT_SNAPSHOT_FILES = tuple(sorted(EXPECTED_FILES - {"strategy_document.md"}))


def reviewed_media_description(
    review: dict[str, Any] | None,
    name: str,
    language: str,
    stale_names: set[str],
) -> str:
    """Use a caption only when it describes the current, accepted final asset."""

    if not review or name in stale_names:
        return ""
    rows = review.get("assets")
    if not isinstance(rows, list):
        return ""
    row = next(
        (item for item in rows if isinstance(item, dict) and item.get("name") == name),
        None,
    )
    if not isinstance(row, dict) or row.get("usable") is not True:
        return ""
    if any(
        row.get(field) is False
        for field in (
            "identity_consistent",
            "construction_consistent",
            "color_consistent",
        )
    ):
        return ""
    if row.get("description_confidence") not in {"high", "medium"}:
        return ""
    descriptions = row.get("media_descriptions")
    value = descriptions.get(language) if isinstance(descriptions, dict) else ""
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if not 12 <= len(value) <= 300 or re.search(r"[\u4e00-\u9fff]", value):
        return ""
    if language == "ko" and not re.search(r"[\uac00-\ud7a3]", value):
        return ""
    if language in {"en", "pt"} and re.search(r"[\uac00-\ud7a3]", value):
        return ""
    return value


def unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def even_sample(values: list[str], count: int) -> list[str]:
    """Choose stable, evenly distributed values without duplicating entries."""

    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return list(values)
    if count == 1:
        return [values[0]]
    indexes = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return [values[index] for index in indexes]
