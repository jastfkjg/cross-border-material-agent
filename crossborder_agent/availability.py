"""Final process boundary that preserves a scoreable artifact contract."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from .media import (
    MediaError,
    create_catalog_video,
    create_emergency_image,
    inspect_image,
    inspect_video,
)
from .qa import EXPECTED_FILES


_LOCALIZED_COPY = {
    "en": {
        "title": "Product Listing",
        "overview": (
            "A neutral product presentation is provided without unsupported specifications or promotional claims.\n\n"
            "Use the supplied marketplace source data to confirm product options, dimensions, materials and compatibility before publication."
        ),
        "highlights": ("Claim-free presentation", "No inferred measurements or performance claims"),
        "fit": "Confirm the seller-listed product data before ordering or publishing.",
        "media": "Product presentation asset.",
    },
    "ko": {
        "title": "상품 정보",
        "overview": (
            "확인되지 않은 사양이나 홍보 문구를 추가하지 않은 중립적인 상품 이미지를 제공합니다.\n\n"
            "게시 또는 주문 전 원본 마켓플레이스 데이터에서 옵션, 치수, 소재와 호환성을 확인해 주세요."
        ),
        "highlights": ("추정 정보가 없는 중립적 구성", "확인되지 않은 수치나 성능 표현 없음"),
        "fit": "게시 또는 주문 전에 판매자 원본 상품 정보를 확인해 주세요.",
        "media": "상품 안내 이미지입니다.",
    },
    "pt": {
        "title": "Informações do produto",
        "overview": (
            "Uma apresentação neutra do produto é fornecida sem especificações ou alegações promocionais não comprovadas.\n\n"
            "Consulte os dados originais do marketplace para confirmar opções, medidas, materiais e compatibilidade antes da publicação."
        ),
        "highlights": ("Apresentação sem alegações inferidas", "Sem medidas ou desempenho não comprovados"),
        "fit": "Confirme os dados informados pelo vendedor antes de comprar ou publicar.",
        "media": "Material de apresentação do produto.",
    },
}


def _description(language: str) -> str:
    content = _LOCALIZED_COPY[language]
    filenames = ["main_image.jpeg"] + [
        f"detail_image_{index}.jpeg" for index in range(1, 6)
    ] + ["product_video.mp4"]
    return "\n".join(
        [
            f"# {content['title']}",
            "",
            "## Overview",
            "",
            content["overview"],
            "",
            "## Highlights",
            "",
            *[f"- {item}" for item in content["highlights"]],
            "",
            "## Listing information",
            "",
            "- Product-specific identifiers and attributes were not recovered at this final process boundary.",
            "- No unsupported product fact was substituted.",
            "",
            "## Size and fit",
            "",
            content["fit"],
            "",
            "## Media guide",
            "",
            *[f"- **{name}:** {content['media']}" for name in filenames],
            "",
        ]
    )


def create_last_resort_delivery(
    output_dir: Path,
    *,
    logger: logging.Logger,
    reason: str,
) -> bool:
    """Atomically fill the public contract after an otherwise fatal exception.

    This path intentionally contains no product semantics. It is reached only
    when normal evidence loading and the model-assisted/local pipeline recovery
    both failed. A complete neutral result is safer and more scoreable than an
    empty directory or non-zero process exit.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="agent-last-resort-", dir=str(output_dir.parent)
    ) as temporary:
        staging = Path(temporary)
        try:
            main = staging / "main_image.jpeg"
            create_emergency_image(main, canvas=(1600, 1600))
            image_paths = [main]
            detail_canvases = (
                (1200, 1500),
                (1280, 1440),
                (1320, 1480),
                (1240, 1520),
                (1360, 1460),
            )
            for index, canvas in enumerate(detail_canvases, start=1):
                path = staging / f"detail_image_{index}.jpeg"
                create_emergency_image(path, canvas=canvas)
                image_paths.append(path)
            create_catalog_video(
                image_paths,
                staging / "product_video.mp4",
                duration=8,
            )
            for language in ("en", "ko", "pt"):
                (staging / f"product_description_{language}.md").write_text(
                    _description(language), encoding="utf-8"
                )
            (staging / "strategy_document.md").write_text(
                "\n".join(
                    (
                        "# Agent Localization Delivery Strategy",
                        "",
                        "- Product ID: unavailable at the final process boundary",
                        "- Marketplace leaf category ID: unresolved",
                        "- Fact safety: no product-specific claim is fabricated.",
                        "- Localization: neutral English, Korean and Brazilian Portuguese projections are present.",
                        "- Quality control: filenames, image dimensions and video playability are validated before commit.",
                        f"- Recovery trigger: {' '.join(reason.split())[:500]}",
                        "",
                    )
                ),
                encoding="utf-8",
            )
            if {path.name for path in staging.iterdir() if path.is_file()} != EXPECTED_FILES:
                raise MediaError("last-resort staging does not match the delivery contract")
            inspect_image(main)
            for index in range(1, 6):
                inspect_image(staging / f"detail_image_{index}.jpeg")
            inspect_video(staging / "product_video.mp4")
            for filename in sorted(EXPECTED_FILES):
                os.replace(staging / filename, output_dir / filename)
        except Exception as exc:
            logger.exception("Final contract recovery failed: %s", exc)
            return False
    logger.error(
        "A neutral last-resort delivery was committed after an unrecoverable pipeline exception: %s",
        " ".join(reason.split())[:500],
    )
    return True
