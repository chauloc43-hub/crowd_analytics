"""Synchronous, bounded short-video analysis for the demo API.

This deliberately is not a durable job system.  It accepts one small uploaded
clip, creates an isolated tracker state, returns the final analytics snapshot,
and removes the temporary source immediately.  Live WebRTC remains the primary
demo path for annotated video.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, BinaryIO, Callable, Protocol

import cv2

from src.api.sessions import PipelineFactory, UnsupportedSessionModeError
from src.inference.live_stream import LivePipeline


class VideoAnalysisError(RuntimeError):
    """Base error for safe client-facing upload diagnostics."""


class UnsupportedVideoError(VideoAnalysisError):
    pass


class VideoTooLargeError(VideoAnalysisError):
    pass


class VideoTooLongError(VideoAnalysisError):
    pass


class VideoAnalysisBusyError(VideoAnalysisError):
    """A live stateful tracker already owns the one-GPU demo capacity."""


class VideoDecodeError(VideoAnalysisError):
    pass


class VideoAnalyzer(Protocol):
    def analyze(
        self,
        stream: BinaryIO,
        *,
        filename: str | None,
        content_type: str | None,
        mode: str,
    ) -> dict[str, Any]: ...


class ShortVideoAnalyzer:
    """Analyze one bounded upload with fresh per-video tracker state."""

    _SUPPORTED_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".jpg", ".jpeg", ".png"})
    _CHUNK_BYTES = 1_024 * 1_024


    def __init__(
        self,
        pipeline_factory: PipelineFactory,
        *,
        allowed_modes: tuple[str, ...] = ("default", "classroom_demo"),
        max_bytes: int = 64 * 1024 * 1024,
        max_seconds: float = 60.0,
        max_frames: int = 1_800,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive.")
        if max_seconds <= 0.0:
            raise ValueError("max_seconds must be positive.")
        if max_frames < 1:
            raise ValueError("max_frames must be positive.")
        self._pipeline_factory = pipeline_factory
        self._allowed_modes = frozenset(allowed_modes)
        self._max_bytes = int(max_bytes)
        self._max_seconds = float(max_seconds)
        self._max_frames = int(max_frames)

    def analyze(
        self,
        stream: BinaryIO,
        *,
        filename: str | None,
        content_type: str | None,
        mode: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        suffix = self._validate_upload_metadata(filename, content_type)
        with TemporaryDirectory(prefix="crowd_api_video_") as temporary_directory:
            source_path = Path(temporary_directory) / f"upload{suffix}"
            received_bytes = self._copy_upload(stream, source_path)
            return self._process_file(source_path, normalized_mode, received_bytes)

    def _process_file(self, source_path: Path, mode: str, received_bytes: int) -> dict[str, Any]:
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            capture.release()
            raise VideoDecodeError("OpenCV could not open the uploaded video.")
        pipeline: LivePipeline | None = None
        try:
            reported_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            fps = reported_fps if reported_fps > 0.0 else 25.0
            reported_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
            reported_duration = reported_frames / fps if reported_frames else None
            if reported_duration is not None and reported_duration > self._max_seconds:
                raise VideoTooLongError(
                    f"The uploaded video is {reported_duration:.1f}s; demo limit is {self._max_seconds:.1f}s."
                )
            if reported_frames and reported_frames > self._max_frames:
                raise VideoTooLongError(
                    f"The uploaded video has {reported_frames} frames; demo limit is {self._max_frames}."
                )

            pipeline = self._pipeline_factory(mode)
            started_at = perf_counter()
            frames_processed = 0
            last_stats: dict[str, Any] | None = None
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frames_processed >= self._max_frames:
                    raise VideoTooLongError(f"The uploaded video exceeds the {self._max_frames}-frame demo limit.")
                timestamp_seconds = frames_processed / fps
                _annotated, last_stats = pipeline.process_frame(frame, timestamp_seconds=timestamp_seconds)
                frames_processed += 1
                if timestamp_seconds > self._max_seconds:
                    raise VideoTooLongError(
                        f"The uploaded video exceeds the {self._max_seconds:.1f}s demo limit."
                    )
            elapsed_seconds = max(0.0, perf_counter() - started_at)
            if frames_processed == 0 or last_stats is None:
                raise VideoDecodeError("The uploaded video did not contain a decodable frame.")
            return {
                "status": "completed",
                "mode": mode,
                "input": {
                    "bytes": received_bytes,
                    "frames_reported": reported_frames or None,
                    "frames_processed": frames_processed,
                    "fps": round(fps, 3),
                    "duration_seconds": round(frames_processed / fps, 3),
                },
                "performance": {
                    "wall_time_seconds": round(elapsed_seconds, 3),
                    "average_processing_fps": round(frames_processed / elapsed_seconds, 3)
                    if elapsed_seconds > 0.0
                    else 0.0,
                },
                # Return the same full analytics envelope used by the live
                # dashboard.  No input or crop data is retained after return.
                "analytics": last_stats,
                "artifacts": {
                    "annotated_video_url": None,
                    "note": "The bounded demo endpoint returns analytics only; WebRTC renders live annotated video.",
                },
            }
        finally:
            capture.release()
            if pipeline is not None:
                close = getattr(pipeline, "close", None)
                if callable(close):
                    with suppress(Exception):
                        close()
                else:
                    with suppress(Exception):
                        pipeline.reset()

    def _copy_upload(self, stream: BinaryIO, target_path: Path) -> int:
        with suppress(Exception):
            stream.seek(0)
        total_bytes = 0
        with target_path.open("wb") as destination:
            while chunk := stream.read(self._CHUNK_BYTES):
                total_bytes += len(chunk)
                if total_bytes > self._max_bytes:
                    raise VideoTooLargeError(
                        f"Upload exceeds the {self._max_bytes // (1024 * 1024)} MiB demo limit."
                    )
                destination.write(chunk)
        if total_bytes == 0:
            raise UnsupportedVideoError("Upload is empty.")
        return total_bytes

    def _normalize_mode(self, mode: str) -> str:
        normalized = str(mode).strip().lower()
        if normalized not in self._allowed_modes:
            allowed = ", ".join(sorted(self._allowed_modes))
            raise UnsupportedSessionModeError(f"Unsupported mode {mode!r}. Allowed modes: {allowed}.")
        return normalized

    def _validate_upload_metadata(self, filename: str | None, content_type: str | None) -> str:
        suffix = Path(filename or "upload.mp4").suffix.lower() or ".mp4"
        if suffix not in self._SUPPORTED_SUFFIXES:
            supported = ", ".join(sorted(self._SUPPORTED_SUFFIXES))
            raise UnsupportedVideoError(f"Unsupported video extension {suffix!r}. Supported extensions: {supported}.")
        if content_type and content_type not in {"application/octet-stream", "binary/octet-stream"}:
            if not (content_type.lower().startswith("video/") or content_type.lower().startswith("image/")):
                raise UnsupportedVideoError("Content-Type must be a video or image MIME type.")
        return suffix

