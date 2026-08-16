"""Small, deterministic video-tool helpers shared by the application and API."""

from __future__ import annotations

import os
from pathlib import Path
from shutil import which
import subprocess


def resolve_ffmpeg_binary(configured_binary: str | None = None) -> str:
    """Prefer an explicit/system FFmpeg, then use imageio-ffmpeg's bundled Windows binary."""
    if configured_binary:
        return configured_binary
    if system_binary := which("ffmpeg"):
        return system_binary
    try:
        import imageio_ffmpeg

        bundled_binary = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError as error:
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and add it to PATH, set FFMPEG_BINARY, "
            "or install the project's imageio-ffmpeg dependency."
        ) from error
    if bundled_binary.is_file():
        return str(bundled_binary)
    raise RuntimeError(
        "imageio-ffmpeg is installed but did not provide an executable. Set FFMPEG_BINARY to a valid ffmpeg.exe path."
    )


def run_ffmpeg(arguments: list[str], configured_binary: str | None = None) -> None:
    """Run an FFmpeg argument list and turn process failures into compact UI-safe errors."""
    command = [resolve_ffmpeg_binary(configured_binary or os.getenv("FFMPEG_BINARY")), *arguments]
    try:
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"FFmpeg executable was not found: {command[0]}") from error
    if process.returncode:
        raise RuntimeError(process.stderr[-1_500:] or "FFmpeg failed without a diagnostic message.")
