"""Local media normalization, validation, and deterministic fallbacks."""

from __future__ import annotations

import math
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import EvidenceTable
from .table_evidence import (
    MAX_RENDER_TABLE_COLUMNS,
    MAX_RENDER_TABLE_ROWS,
    presentation_view,
)


_BUNDLED_CJK_FONT = (
    Path(__file__).resolve().parent
    / "assets"
    / "fonts"
    / "NotoSansCJK-Regular.ttc"
)


try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat
except (
    ImportError
):  # pragma: no cover - exercised in a deliberately dependency-free environment
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFilter = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]
    ImageStat = None  # type: ignore[assignment]


class MediaError(RuntimeError):
    """Raised when a media file cannot be made delivery-safe."""


@dataclass(frozen=True, slots=True)
class ImageInfo:
    width: int
    height: int
    format: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ImageQualityInfo:
    luminance_stddev: float
    entropy: float
    edge_mean: float
    difference_hash: int


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", header[16:24])
    return None


def _jpeg_dimensions(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            return None
        while True:
            marker_start = handle.read(1)
            if not marker_start:
                return None
            if marker_start != b"\xff":
                continue
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if not marker or marker in {b"\xd8", b"\xd9"}:
                continue
            length_raw = handle.read(2)
            if len(length_raw) != 2:
                return None
            length = struct.unpack(">H", length_raw)[0]
            if marker[0] in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                data = handle.read(5)
                if len(data) != 5:
                    return None
                height, width = struct.unpack(">HH", data[1:5])
                return width, height
            handle.seek(max(0, length - 2), os.SEEK_CUR)


def inspect_image(path: Path) -> ImageInfo:
    if not path.is_file():
        raise MediaError(f"Image does not exist: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise MediaError(f"Image is empty: {path}")
    if Image is not None:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                image_format = str(image.format or "").upper()
            return ImageInfo(
                width=width, height=height, format=image_format, size_bytes=size
            )
        except Exception as exc:
            raise MediaError(f"Image cannot be decoded: {path}: {exc}") from exc
    dimensions = _png_dimensions(path)
    if dimensions:
        return ImageInfo(dimensions[0], dimensions[1], "PNG", size)
    dimensions = _jpeg_dimensions(path)
    if dimensions:
        return ImageInfo(dimensions[0], dimensions[1], "JPEG", size)
    raise MediaError(f"Unsupported image format and Pillow is unavailable: {path}")


def inspect_image_quality(path: Path) -> ImageQualityInfo | None:
    """Return cheap deterministic quality signals without changing the image."""

    inspect_image(path)
    if Image is None or ImageFilter is None or ImageStat is None:
        return None
    try:
        with Image.open(path) as opened:
            gray = ImageOps.exif_transpose(opened).convert("L")
            sample = gray.resize((256, 256), Image.Resampling.LANCZOS)
            inner = sample.crop((4, 4, 252, 252))
            edges = inner.filter(ImageFilter.FIND_EDGES).crop((4, 4, 244, 244))
            hash_sample = gray.resize((9, 8), Image.Resampling.LANCZOS)
            pixels = list(hash_sample.getdata())
    except Exception as exc:
        raise MediaError(f"Image-quality analysis failed: {path}: {exc}") from exc

    difference_hash = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            difference_hash = (difference_hash << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return ImageQualityInfo(
        luminance_stddev=float(ImageStat.Stat(inner).stddev[0]),
        entropy=float(sample.entropy()),
        edge_mean=float(ImageStat.Stat(edges).mean[0]),
        difference_hash=difference_hash,
    )


def hash_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def normalize_image(
    source: Path,
    destination: Path,
    *,
    canvas: tuple[int, int],
    max_bytes: int = 5 * 1024 * 1024,
    white_background: bool = False,
    focus_crop: str = "",
) -> ImageInfo:
    """Normalize an image to an RGB JPEG.

    Full-view assets preserve the entire source. Deterministic detail fallbacks may
    request one of a small set of bounded spatial crops so a limited source set can
    still provide useful source-visible detail coverage without inventing pixels.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    if Image is None:
        info = inspect_image(source)
        if info.format not in {"JPEG", "JPG", "PNG"}:
            raise MediaError("Pillow is unavailable, so the source image format cannot be converted")
        if info.width < min(canvas[0], 800) or info.height < min(canvas[1], 800):
            raise MediaError("Pillow is unavailable, so a small source image cannot be enlarged")
        shutil.copyfile(source, destination)
        return inspect_image(destination)

    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            crop_boxes = {
                "upper": (0.12, 0.02, 0.88, 0.68),
                "lower": (0.12, 0.30, 0.88, 0.98),
                "left": (0.00, 0.10, 0.70, 0.90),
                "right": (0.30, 0.10, 1.00, 0.90),
                "center": (0.14, 0.12, 0.86, 0.88),
            }
            normalized_focus = focus_crop.strip().lower()
            if normalized_focus:
                if normalized_focus not in crop_boxes:
                    raise MediaError(f"Unsupported detail crop region: {focus_crop}")
                left, top, right, bottom = crop_boxes[normalized_focus]
                focused = image.crop(
                    (
                        round(image.width * left),
                        round(image.height * top),
                        round(image.width * right),
                        round(image.height * bottom),
                    )
                )
                canvas_image = ImageOps.fit(
                    focused, canvas, method=Image.Resampling.LANCZOS
                )
            else:
                contained = ImageOps.contain(
                    image, canvas, method=Image.Resampling.LANCZOS
                )
                if white_background:
                    canvas_image = Image.new("RGB", canvas, (255, 255, 255))
                else:
                    background = ImageOps.fit(
                        image, canvas, method=Image.Resampling.LANCZOS
                    ).filter(ImageFilter.GaussianBlur(radius=36))
                    veil = Image.new("RGB", canvas, (255, 255, 255))
                    canvas_image = Image.blend(background, veil, 0.42)
                paste_left = (canvas[0] - contained.width) // 2
                paste_top = (canvas[1] - contained.height) // 2
                canvas_image.paste(contained, (paste_left, paste_top))

            quality = 92
            while True:
                canvas_image.save(
                    destination,
                    format="JPEG",
                    quality=quality,
                    optimize=True,
                    progressive=True,
                    subsampling=0 if quality >= 88 else 2,
                )
                if destination.stat().st_size <= max_bytes or quality <= 72:
                    break
                quality -= 5
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise MediaError(f"Image normalization failed: {exc}") from exc
    info = inspect_image(destination)
    if info.size_bytes > max_bytes:
        raise MediaError(f"Normalized image still exceeds {max_bytes} bytes")
    return info


def create_emergency_image(
    destination: Path, *, canvas: tuple[int, int] = (1200, 1500)
) -> ImageInfo:
    """Create a neutral, specification-safe last-resort image without claims.

    Normal delivery always prefers generated or seller-source pixels. This is
    only an availability fallback for the case where no remote or local product
    image survives. It deliberately contains no text, logo, or product claim.
    """

    if Image is None or ImageDraw is None:
        raise MediaError("Pillow is unavailable, so the final image fallback cannot be created")
    width, height = canvas
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        image = Image.new("RGB", canvas, (246, 247, 249))
        draw = ImageDraw.Draw(image)
        for y in range(height):
            shade = 246 - round(12 * y / max(1, height - 1))
            draw.line((0, y, width, y), fill=(shade, shade + 1, min(255, shade + 3)))
        margin_x = max(40, width // 7)
        margin_y = max(50, height // 8)
        draw.rounded_rectangle(
            (margin_x, margin_y, width - margin_x, height - margin_y),
            radius=max(24, min(width, height) // 24),
            fill=(224, 228, 233),
            outline=(190, 197, 205),
            width=max(4, width // 240),
        )
        inner = max(24, min(width, height) // 32)
        draw.rounded_rectangle(
            (
                margin_x + inner,
                margin_y + inner,
                width - margin_x - inner,
                height - margin_y - inner,
            ),
            radius=max(18, min(width, height) // 32),
            fill=(238, 240, 243),
        )
        image.save(
            destination,
            format="JPEG",
            quality=90,
            optimize=True,
            progressive=True,
        )
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise MediaError(f"Final image fallback creation failed: {exc}") from exc
    return inspect_image(destination)


def create_evidence_table_image(table: EvidenceTable, destination: Path) -> None:
    """Render the model's grounded, domain-neutral table presentation."""

    if Image is None or ImageDraw is None or ImageFont is None:
        raise MediaError("Pillow is unavailable, so the evidence-table detail image cannot be generated")
    try:
        view = presentation_view(table)
    except ValueError as exc:
        raise MediaError(str(exc)) from exc
    headers = view["headers"]
    rows = view["rows"]
    if (
        not 1 <= len(headers) <= MAX_RENDER_TABLE_COLUMNS
        or not 1 <= len(rows) <= MAX_RENDER_TABLE_ROWS
    ):
        raise MediaError(
            "The model-selected table exceeds single-page rendering capacity: "
            f"columns={len(headers)}/{MAX_RENDER_TABLE_COLUMNS}, "
            f"rows={len(rows)}/{MAX_RENDER_TABLE_ROWS}"
        )

    width, height = 1200, 1500
    canvas = Image.new("RGB", (width, height), (248, 246, 242))
    draw = ImageDraw.Draw(canvas)

    font_collection_index = {"ko": 1, "en": 2, "pt": 2}.get(
        str(view.get("locale") or "en"), 2
    )

    def needs_cjk_font(text: str) -> bool:
        return any(
            0x2E80 <= ord(character) <= 0x9FFF
            or 0xAC00 <= ord(character) <= 0xD7AF
            or 0x3040 <= ord(character) <= 0x30FF
            for character in text
        )

    def font(size: int, text: str, *, bold: bool = False) -> Any:
        bundled = str(_BUNDLED_CJK_FONT)
        latin_names = (
            (
                "DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                bundled,
            )
            if bold
            else (
                "DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
                bundled,
            )
        )
        cjk_names = (
            bundled,
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
        names = cjk_names if needs_cjk_font(text) else latin_names
        for name in names:
            try:
                return ImageFont.truetype(
                    name,
                    size=size,
                    index=(font_collection_index if name.endswith(".ttc") else 0),
                )
            except OSError:
                continue
        return ImageFont.load_default()

    ink = (41, 39, 45)
    accent = (112, 88, 132)
    muted = (105, 100, 110)

    def fit_font(text: str, maximum: int, box_width: int, *, bold: bool = False) -> Any:
        for size in range(maximum, 13, -1):
            candidate = font(size, text, bold=bold)
            bounds = draw.textbbox((0, 0), text, font=candidate)
            if bounds[2] - bounds[0] <= box_width:
                return candidate
        raise MediaError(f"Table text cannot fit completely in its cell: {text[:80]}")

    draw.rounded_rectangle((70, 70, 1130, 1430), radius=30, fill=(255, 255, 255))
    title = str(view["title"])
    title_font = fit_font(title, 54, 960, bold=True)
    draw.text((120, 125), title, font=title_font, fill=ink)
    draw.line((120, 235, 1080, 235), fill=accent, width=5)

    column_texts = [
        [str(headers[index]), *[str(row[index]) for row in rows]]
        for index in range(len(headers))
    ]
    weights = [
        max(4, min(24, max(len(value) for value in values)))
        for values in column_texts
    ]
    total_weight = sum(weights)
    table_left, table_right = 110, 1090
    available_width = table_right - table_left
    boundaries = [table_left]
    consumed = 0
    for weight in weights:
        consumed += weight
        boundaries.append(table_left + round(available_width * consumed / total_weight))

    header_y = 285
    row_height = min(105, 860 // max(1, len(rows)))
    if row_height < 34:
        raise MediaError("The model-selected table has too many rows for lossless rendering")
    for index, label in enumerate(headers):
        cell_left = boundaries[index] + 10
        cell_width = boundaries[index + 1] - boundaries[index] - 20
        header_font = fit_font(str(label), 25, cell_width, bold=True)
        draw.text((cell_left, header_y), str(label), font=header_font, fill=accent)

    y = 360
    for row_index, values in enumerate(rows):
        if row_index % 2 == 0:
            draw.rounded_rectangle(
                (105, y - 16, 1095, y + row_height - 16),
                radius=14,
                fill=(248, 246, 250),
            )
        for column_index, value in enumerate(values):
            text_value = str(value)
            if not text_value:
                continue
            cell_left = boundaries[column_index] + 10
            cell_width = boundaries[column_index + 1] - boundaries[column_index] - 20
            row_font = fit_font(text_value, 27, cell_width)
            draw.text((cell_left, y), text_value, font=row_font, fill=ink)
        y += row_height

    notes = [str(item) for item in view["notes"] if str(item)]
    if notes:
        note_y = max(1220, y + 35)
        required_height = 42 * len(notes)
        if note_y + required_height > 1385:
            raise MediaError("The model-provided table note cannot fit completely on the canvas")
        draw.line((120, note_y, 1080, note_y), fill=(221, 216, 225), width=2)
        for note_index, note in enumerate(notes):
            note_font = fit_font(note, 22, 960)
            draw.text(
                (120, note_y + 30 + note_index * 42),
                note,
                font=note_font,
                fill=muted,
            )
    canvas.save(
        destination,
        format="JPEG",
        quality=92,
        optimize=True,
        progressive=True,
        subsampling=0,
    )
    inspect_image(destination)


def _edge_background(image: Any) -> tuple[int, int, int]:
    width, height = image.size
    points = [
        image.getpixel((0, 0)),
        image.getpixel((max(0, width - 1), 0)),
        image.getpixel((0, max(0, height - 1))),
        image.getpixel((max(0, width - 1), max(0, height - 1))),
    ]
    return tuple(
        int(sum(point[channel] for point in points) / len(points))
        for channel in range(3)
    )


def _ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
        if executable and Path(executable).is_file():
            return executable
    except (ImportError, OSError, RuntimeError):
        pass
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    raise MediaError("No usable ffmpeg binary was found for video fallback generation")


def create_slideshow_video(
    image_path: Path, destination: Path, *, duration: int = 8
) -> None:
    """Create a standards-compliant H.264 fallback video from a verified still."""

    ffmpeg = _ffmpeg_executable()
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames = duration * 25
    filter_graph = (
        "scale=1280:720:force_original_aspect_ratio=decrease,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=white,"
        f"zoompan=z='min(zoom+0.00035,1.05)':d={frames}:s=1280x720:fps=25,"
        "format=yuv420p"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-vf",
        filter_graph,
        "-t",
        str(duration),
        "-an",
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-level",
        "4.0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=120, check=False
    )
    if completed.returncode != 0:
        destination.unlink(missing_ok=True)
        raise MediaError(f"ffmpeg video fallback failed: {completed.stderr[-1200:]}")
    inspect_video(destination)


def create_catalog_video(
    image_paths: list[Path], destination: Path, *, duration: int = 8
) -> None:
    """Create a compact multi-shot catalog video from already validated images."""

    usable: list[Path] = []
    seen_hashes: list[int] = []
    for path in image_paths:
        if path in usable:
            continue
        inspect_image(path)
        quality = inspect_image_quality(path)
        if quality is not None and any(
            hash_distance(quality.difference_hash, seen) <= 10 for seen in seen_hashes
        ):
            continue
        usable.append(path)
        if quality is not None:
            seen_hashes.append(quality.difference_hash)
    if len(usable) < 2:
        if not usable:
            raise MediaError("No image is available for the video fallback")
        create_slideshow_video(usable[0], destination, duration=duration)
        return

    ffmpeg = _ffmpeg_executable()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fps = 25
    transition_seconds = 0.18
    shot_duration = duration / len(usable)
    frames_per_shot = max(25, math.ceil(shot_duration * fps))
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    for path in usable:
        command.extend(
            ["-loop", "1", "-framerate", str(fps), "-t", f"{shot_duration:.3f}", "-i", str(path)]
        )

    filters: list[str] = []
    streams: list[str] = []
    for index in range(len(usable)):
        label = f"v{index}"
        pan_direction = "zoom+0.00045" if index % 2 == 0 else "zoom+0.0003"
        fade_out = (
            f"fade=t=out:st={max(0.0, shot_duration - transition_seconds):.3f}:"
            f"d={transition_seconds:.3f}:color=white,"
            if index < len(usable) - 1
            else ""
        )
        filters.append(
            f"[{index}:v]split=2[bg{index}][fg{index}];"
            f"[bg{index}]scale=1280:720:force_original_aspect_ratio=increase,"
            f"crop=1280:720,eq=contrast=0.68:brightness=0.12:saturation=0.45,"
            f"gblur=sigma=38[bgfill{index}];"
            f"[fg{index}]scale=1280:720:force_original_aspect_ratio=decrease[fgfit{index}];"
            f"[bgfill{index}][fgfit{index}]overlay=(W-w)/2:(H-h)/2,"
            f"zoompan=z='min({pan_direction},1.045)':d=1:"
            f"s=1280x720:fps={fps},fps={fps},settb=AVTB,"
            f"trim=end_frame={frames_per_shot},setpts=PTS-STARTPTS,"
            f"fade=t=in:st=0:d={transition_seconds:.3f}:color=white,"
            f"{fade_out}setsar=1,format=yuv420p[{label}]"
        )
        streams.append(f"[{label}]")
    filters.append(
        "".join(streams)
        + f"concat=n={len(usable)}:v=1:a=0,fps={fps},"
        "tpad=stop_mode=clone:stop_duration=1,format=yuv420p[outv]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outv]",
            "-t",
            str(duration),
            "-an",
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-level",
            "4.0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=150, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        destination.unlink(missing_ok=True)
        raise MediaError(f"Multi-shot video fallback failed: {exc}") from exc
    if completed.returncode != 0:
        destination.unlink(missing_ok=True)
        raise MediaError(f"Multi-shot video fallback failed: {completed.stderr[-1200:]}")
    inspect_video(destination)


def strip_video_audio(source: Path, destination: Path) -> None:
    """Copy the video stream into a fresh MP4 while removing unreviewed audio."""

    ffmpeg = _ffmpeg_executable()
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-an",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        destination.unlink(missing_ok=True)
        raise MediaError(f"Video audio removal failed: {exc}") from exc
    if completed.returncode != 0:
        destination.unlink(missing_ok=True)
        raise MediaError(f"Video audio removal failed: {completed.stderr[-1200:]}")
    inspect_video(destination)


def inspect_video(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise MediaError(f"Video does not exist or is empty: {path}")
    if path.stat().st_size >= 200 * 1024 * 1024:
        raise MediaError("Video is 200 MB or larger")
    with path.open("rb") as handle:
        header = handle.read(64)
    if b"ftyp" not in header:
        raise MediaError("Video is not a recognizable MP4/MOV container")
    result: dict[str, Any] = {
        "size_bytes": path.stat().st_size,
        "container": "mp4/mov",
        "decoded": False,
    }
    try:
        ffmpeg = _ffmpeg_executable()
    except MediaError:
        return result
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaError(f"Full video decode check failed: {exc}") from exc
    if completed.returncode != 0:
        raise MediaError(f"Video cannot be decoded completely: {completed.stderr[-1200:]}")
    result["decoded"] = True
    # Supply objective temporal evidence for local-only fallbacks that cannot be
    # uploaded to a reviewer model. This never claims semantic quality; it only
    # distinguishes an unchanged frame from a presentation with visible change.
    if Image is not None:
        try:
            with tempfile.TemporaryDirectory(prefix="agent-video-frames-") as temporary:
                frame_pattern = str(Path(temporary) / "frame-%02d.png")
                sample = subprocess.run(
                    [
                        ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-nostdin",
                        "-i",
                        str(path),
                        "-vf",
                        "fps=1",
                        "-frames:v",
                        "10",
                        frame_pattern,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                if sample.returncode == 0:
                    hashes = [
                        quality.difference_hash
                        for frame in sorted(Path(temporary).glob("frame-*.png"))
                        for quality in [inspect_image_quality(frame)]
                        if quality is not None
                    ]
                    changes = [
                        hash_distance(left, right)
                        for left, right in zip(hashes, hashes[1:])
                    ]
                    result["temporal_samples"] = len(hashes)
                    result["temporal_changed_pairs"] = sum(
                        distance > 4 for distance in changes
                    )
                    result["temporal_pair_count"] = len(changes)
                    result["temporal_change_ratio"] = (
                        round(sum(distance > 4 for distance in changes) / len(changes), 3)
                        if changes
                        else 0.0
                    )
        except (OSError, subprocess.TimeoutExpired, MediaError):
            # Full decode and physical validity already succeeded. Optional
            # temporal telemetry must never turn a playable delivery into a
            # process failure.
            pass
    return result
