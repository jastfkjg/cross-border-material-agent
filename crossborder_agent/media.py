"""Local media normalization, validation, and deterministic fallbacks."""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    from PIL import Image, ImageFilter, ImageOps, ImageStat
except (
    ImportError
):  # pragma: no cover - exercised in a deliberately dependency-free environment
    Image = None  # type: ignore[assignment]
    ImageFilter = None  # type: ignore[assignment]
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
        raise MediaError(f"图片不存在: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise MediaError(f"图片为空: {path}")
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
            raise MediaError(f"图片无法解码: {path}: {exc}") from exc
    dimensions = _png_dimensions(path)
    if dimensions:
        return ImageInfo(dimensions[0], dimensions[1], "PNG", size)
    dimensions = _jpeg_dimensions(path)
    if dimensions:
        return ImageInfo(dimensions[0], dimensions[1], "JPEG", size)
    raise MediaError(f"不支持的图片格式且 Pillow 不可用: {path}")


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
        raise MediaError(f"图片质量分析失败: {path}: {exc}") from exc

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
) -> ImageInfo:
    """Normalize an image to an RGB JPEG while preserving the entire product."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if Image is None:
        info = inspect_image(source)
        if info.format not in {"JPEG", "JPG", "PNG"}:
            raise MediaError("Pillow 不可用，无法转换源图片格式")
        if info.width < min(canvas[0], 800) or info.height < min(canvas[1], 800):
            raise MediaError("Pillow 不可用，无法放大小尺寸源图片")
        shutil.copyfile(source, destination)
        return inspect_image(destination)

    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            background_color = (
                (255, 255, 255) if white_background else _edge_background(image)
            )
            contained = ImageOps.contain(image, canvas, method=Image.Resampling.LANCZOS)
            canvas_image = Image.new("RGB", canvas, background_color)
            left = (canvas[0] - contained.width) // 2
            top = (canvas[1] - contained.height) // 2
            canvas_image.paste(contained, (left, top))

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
        raise MediaError(f"图片归一化失败: {exc}") from exc
    info = inspect_image(destination)
    if info.size_bytes > max_bytes:
        raise MediaError(f"归一化图片仍超过 {max_bytes} 字节")
    return info


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
    raise MediaError("找不到可用的 ffmpeg，无法生成视频回退")


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
        raise MediaError(f"ffmpeg 视频回退失败: {completed.stderr[-1200:]}")
    inspect_video(destination)


def inspect_video(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise MediaError(f"视频不存在或为空: {path}")
    if path.stat().st_size >= 200 * 1024 * 1024:
        raise MediaError("视频达到或超过 200MB")
    with path.open("rb") as handle:
        header = handle.read(64)
    if b"ftyp" not in header:
        raise MediaError("视频不是可识别的 MP4/MOV 容器")
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
        raise MediaError(f"视频完整解码检查失败: {exc}") from exc
    if completed.returncode != 0:
        raise MediaError(f"视频无法完整解码: {completed.stderr[-1200:]}")
    result["decoded"] = True
    return result
