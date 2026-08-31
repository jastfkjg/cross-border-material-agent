#!/usr/bin/env python3
"""Build a self-contained Debian 12 x86_64 submission ZIP under dist/."""

from __future__ import annotations

import argparse
import compileall
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
STAGING = DIST / "agent"
ZIP_PATH = DIST / "agent.zip"
MAX_ZIP_BYTES = 100 * 1024 * 1024
NOTO_CJK_SHA256 = "b76b0433203017ca80401b2ee0dd69350349871c4b19d504c34dbdd80541690a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-dependencies", action="store_true", help="build source-only archive"
    )
    return parser.parse_args()


def copy_sources() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    for filename in ("agent.py", "agent.json", "requirements.txt"):
        shutil.copy2(ROOT / filename, STAGING / filename)
    shutil.copytree(ROOT / "crossborder_agent", STAGING / "crossborder_agent")
    shutil.copytree(ROOT / "rules", STAGING / "rules")
    for cache in STAGING.rglob("__pycache__"):
        shutil.rmtree(cache)
    for pyc in STAGING.rglob("*.pyc"):
        pyc.unlink()


def vendor_linux_dependencies() -> None:
    vendor = STAGING / "vendor"
    vendor.mkdir()
    with tempfile.TemporaryDirectory(prefix="agent-wheels-") as temporary:
        wheel_dir = Path(temporary)
        command = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--disable-pip-version-check",
            "--only-binary=:all:",
            "--platform=manylinux2014_x86_64",
            "--python-version=312",
            "--implementation=cp",
            "--abi=cp312",
            f"--dest={wheel_dir}",
            f"--requirement={ROOT / 'requirements.txt'}",
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            # Local mirrors occasionally stop serving a pinned manylinux wheel.
            # Reuse only an earlier submission whose dependency manifest is
            # byte-for-byte identical; source files are still rebuilt below.
            if not ZIP_PATH.is_file():
                raise
            with zipfile.ZipFile(ZIP_PATH) as previous:
                if (
                    previous.read("requirements.txt")
                    != (ROOT / "requirements.txt").read_bytes()
                ):
                    raise
                vendor_members = [
                    name
                    for name in previous.namelist()
                    if name.startswith("vendor/") and not name.endswith("/")
                ]
                if not vendor_members:
                    raise
                for name in vendor_members:
                    relative = Path(name)
                    destination = STAGING / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(previous.read(name))
            print("Reused matching Linux vendor dependencies from previous agent.zip")
            return
        wheels = sorted(wheel_dir.glob("*.whl"))
        if not wheels:
            raise RuntimeError("pip did not download any dependency wheels")
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as archive:
                archive.extractall(vendor)


def validate_staging(skip_dependencies: bool) -> None:
    manifest = json.loads((STAGING / "agent.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {"runtime", "version"}:
        raise RuntimeError(f"Unexpected agent.json: {manifest}")
    if manifest["runtime"] != "python":
        raise RuntimeError(f"Unexpected agent runtime: {manifest['runtime']}")
    version = manifest["version"]
    if not isinstance(version, str) or re.fullmatch(r"\d+\.\d+\.\d+", version) is None:
        raise RuntimeError(f"Invalid agent version: {version!r}")
    try:
        version_check = subprocess.run(
            [sys.executable, str(STAGING / "agent.py"), "--version"],
            cwd=STAGING,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Agent version check could not run: {exc}") from exc
    reported_version = version_check.stdout.strip()
    if version_check.returncode != 0 or reported_version != version:
        raise RuntimeError(
            "Agent --version does not match agent.json: "
            f"expected={version!r}, reported={reported_version!r}, "
            f"stderr={version_check.stderr[-1000:]!r}"
        )
    font_path = (
        STAGING
        / "crossborder_agent"
        / "assets"
        / "fonts"
        / "NotoSansCJK-Regular.ttc"
    )
    font_license = font_path.with_name("OFL-1.1.txt")
    if not font_path.is_file() or not font_license.is_file():
        raise RuntimeError("Bundled CJK font or its license is missing")
    font_digest = hashlib.sha256(font_path.read_bytes()).hexdigest()
    if font_digest != NOTO_CJK_SHA256:
        raise RuntimeError(f"Bundled CJK font checksum mismatch: {font_digest}")
    if not skip_dependencies:
        pillow = STAGING / "vendor" / "PIL"
        imageio = STAGING / "vendor" / "imageio_ffmpeg"
        if not pillow.is_dir() or not imageio.is_dir():
            raise RuntimeError("Required vendored packages are missing")
        binaries = list(imageio.rglob("ffmpeg-*"))
        if not binaries:
            raise RuntimeError("imageio-ffmpeg wheel does not contain an ffmpeg binary")
        for binary in binaries:
            binary.chmod(binary.stat().st_mode | 0o111)
    if not compileall.compile_dir(STAGING / "crossborder_agent", quiet=1):
        raise RuntimeError("Python source compilation failed")
    for cache in STAGING.rglob("__pycache__"):
        shutil.rmtree(cache)


def write_zip() -> None:
    ZIP_PATH.unlink(missing_ok=True)
    with zipfile.ZipFile(
        ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(STAGING.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(STAGING)
            info = zipfile.ZipInfo.from_file(path, arcname=str(relative))
            if relative == Path("agent.py") or "imageio_ffmpeg/binaries/ffmpeg-" in str(
                relative
            ):
                info.external_attr = (0o100755 & 0xFFFF) << 16
            with path.open("rb") as handle:
                archive.writestr(
                    info,
                    handle.read(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
    size = ZIP_PATH.stat().st_size
    if size > MAX_ZIP_BYTES:
        raise RuntimeError(f"Submission ZIP exceeds 100MB: {size} bytes")
    digest = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()
    print(f"Built {ZIP_PATH} ({size / 1024 / 1024:.2f} MiB)")
    print(f"SHA256 {digest}")


def main() -> int:
    args = parse_args()
    DIST.mkdir(exist_ok=True)
    copy_sources()
    if not args.skip_dependencies:
        vendor_linux_dependencies()
    validate_staging(args.skip_dependencies)
    write_zip()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
