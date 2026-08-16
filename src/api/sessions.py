"""Session ownership for the stateful live FastAPI demo.

Every live session owns exactly one ``CrowdGenderPipeline`` and one
``LiveFrameProcessor``.  Keeping that boundary explicit prevents a FastTracker
or stream-scoped ``person_id`` from leaking across cameras or browser peers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock
from time import monotonic
from typing import Any, Callable, Protocol
from uuid import uuid4

import numpy as np

from src.inference.live_stream import LiveFrameProcessor, LiveFrameResult, LivePipeline


class ApiSessionError(RuntimeError):
    """Base error translated to a compact HTTP response by the API layer."""


class SessionNotFoundError(ApiSessionError):
    pass


class SessionCapacityError(ApiSessionError):
    pass


class UnsupportedSessionModeError(ApiSessionError):
    pass


class SessionInitializationError(ApiSessionError):
    """The requested model/profile could not be prepared for a new session."""


class SessionClosedError(ApiSessionError):
    pass


PipelineFactory = Callable[[str], LivePipeline]
ReadinessProbe = Callable[[], dict[str, Any]]


@dataclass(frozen=True)
class SessionInfo:
    """JSON-safe public metadata; no model, image, or tracker object escapes."""

    session_id: str
    mode: str
    camera_id: str | None
    created_at: str
    last_used_at: str
    expires_in_seconds: float | None
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.session_id,
            "mode": self.mode,
            "camera_id": self.camera_id,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "expires_in_seconds": self.expires_in_seconds,
            "status": self.status,
        }


@dataclass(frozen=True)
class SessionState:
    """A point-in-time dashboard view of a session and its newest result."""

    info: SessionInfo
    result: LiveFrameResult | None
    telemetry: dict[str, Any]


@dataclass
class _SessionEntry:
    session_id: str
    mode: str
    camera_id: str | None
    processor: LiveFrameProcessor
    created_at: datetime
    last_used_at: datetime
    created_monotonic: float
    last_used_monotonic: float


class SessionManager(Protocol):
    """Minimal manager contract shared by HTTP and WebRTC adapters."""

    def create_session(self, mode: str = "default", camera_id: str | None = None) -> SessionInfo: ...

    def get(self, session_id: str) -> SessionInfo: ...

    def get_processor(self, session_id: str) -> LiveFrameProcessor: ...

    def submit_frame(
        self,
        session_id: str,
        frame: np.ndarray,
        *,
        submitted_at: float | None = None,
    ) -> int: ...

    def latest_result(self, session_id: str) -> LiveFrameResult | None: ...

    def get_state(self, session_id: str) -> SessionState: ...

    def reset(self, session_id: str) -> SessionInfo: ...

    def close(self, session_id: str) -> None: ...

    def health(self) -> dict[str, Any]: ...

    def readiness(self) -> dict[str, Any]: ...

    def close_all(self) -> None: ...


class DemoSessionManager:
    """Capacity-bounded owner of live pipeline workers.

    The factory is injected so tests never instantiate GPU models.  The default
    app uses a factory that calls ``warmup`` only when a session is explicitly
    created, not at ASGI import time.
    """

    def __init__(
        self,
        pipeline_factory: PipelineFactory,
        *,
        allowed_modes: tuple[str, ...] = ("default", "classroom_demo"),
        max_sessions: int = 1,
        ttl_seconds: float = 600.0,
        cadence_seconds: float = 0.15,
        profiling_window: int = 120,
        readiness_probe: ReadinessProbe | None = None,
        clock: Callable[[], float] = monotonic,
        wall_clock: Callable[[], datetime] | None = None,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive.")
        if ttl_seconds < 0.0:
            raise ValueError("ttl_seconds must be non-negative.")
        if cadence_seconds <= 0.0:
            raise ValueError("cadence_seconds must be positive.")
        if profiling_window < 1:
            raise ValueError("profiling_window must be positive.")
        if not allowed_modes:
            raise ValueError("allowed_modes cannot be empty.")
        self._pipeline_factory = pipeline_factory
        self._allowed_modes = frozenset(allowed_modes)
        self._max_sessions = int(max_sessions)
        self._ttl_seconds = float(ttl_seconds)
        self._cadence_seconds = float(cadence_seconds)
        self._profiling_window = int(profiling_window)
        self._readiness_probe = readiness_probe
        self._clock = clock
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._lock = RLock()
        self._entries: dict[str, _SessionEntry] = {}
        self._last_creation_error: str | None = None

    @property
    def allowed_modes(self) -> tuple[str, ...]:
        return tuple(sorted(self._allowed_modes))

    def create_session(self, mode: str = "default", camera_id: str | None = None) -> SessionInfo:
        normalized_mode = self._normalize_mode(mode)
        normalized_camera_id = self._normalize_camera_id(camera_id)
        with self._lock:
            self._evict_expired_locked(self._clock())
            if len(self._entries) >= self._max_sessions:
                if self._max_sessions == 1:
                    evict_ids = list(self._entries.keys())
                    for old_id in evict_ids:
                        old_entry = self._entries.pop(old_id)
                        old_entry.processor.close()
                else:
                    raise SessionCapacityError(
                        f"Live demo capacity is {self._max_sessions} session(s); close the existing session first."
                    )
            try:
                pipeline = self._pipeline_factory(normalized_mode)
                processor = LiveFrameProcessor(
                    pipeline,
                    cadence_seconds=self._cadence_seconds,
                    profiling_window=self._profiling_window,
                )
            except Exception as exc:
                self._last_creation_error = f"{type(exc).__name__}: {exc}"
                raise SessionInitializationError(
                    "The selected pipeline could not be initialized. Check /api/v1/ready and server logs."
                ) from exc
            now_monotonic = self._clock()
            now_wall = self._wall_clock()
            entry = _SessionEntry(
                session_id=f"demo_{uuid4().hex[:12]}",
                mode=normalized_mode,
                camera_id=normalized_camera_id,
                processor=processor,
                created_at=now_wall,
                last_used_at=now_wall,
                created_monotonic=now_monotonic,
                last_used_monotonic=now_monotonic,
            )
            self._entries[entry.session_id] = entry
            self._last_creation_error = None
            return self._snapshot_locked(entry, now_monotonic)

    def get(self, session_id: str) -> SessionInfo:
        with self._lock:
            entry, now = self._entry_locked(session_id)
            self._touch_locked(entry, now)
            return self._snapshot_locked(entry, now)

    def get_processor(self, session_id: str) -> LiveFrameProcessor:
        with self._lock:
            entry, now = self._entry_locked(session_id)
            self._touch_locked(entry, now)
            return entry.processor

    def submit_frame(
        self,
        session_id: str,
        frame: np.ndarray,
        *,
        submitted_at: float | None = None,
    ) -> int:
        processor = self.get_processor(session_id)
        sequence = processor.submit(frame, submitted_at=submitted_at)
        if sequence is None:
            raise SessionClosedError("The live session was closed while a frame was being submitted.")
        return sequence

    def latest_result(self, session_id: str) -> LiveFrameResult | None:
        processor = self.get_processor(session_id)
        return processor.latest_result()

    def get_state(self, session_id: str) -> SessionState:
        with self._lock:
            entry, now = self._entry_locked(session_id)
            self._touch_locked(entry, now)
            info = self._snapshot_locked(entry, now)
            processor = entry.processor
        return SessionState(info=info, result=processor.latest_result(), telemetry=processor.telemetry())

    def reset(self, session_id: str) -> SessionInfo:
        processor = self.get_processor(session_id)
        processor.reset()
        with self._lock:
            entry, now = self._entry_locked(session_id)
            self._touch_locked(entry, now)
            return self._snapshot_locked(entry, now)

    def close(self, session_id: str) -> None:
        with self._lock:
            self._evict_expired_locked(self._clock())
            entry = self._entries.pop(session_id, None)
        if entry is None:
            raise SessionNotFoundError(f"Unknown or expired session: {session_id}")
        entry.processor.close()

    def close_all(self) -> None:
        with self._lock:
            entries = tuple(self._entries.values())
            self._entries.clear()
        for entry in entries:
            entry.processor.close()

    def health(self) -> dict[str, Any]:
        with self._lock:
            self._evict_expired_locked(self._clock())
            return {
                "active_sessions": len(self._entries),
                "max_live_sessions": self._max_sessions,
                "allowed_modes": self.allowed_modes,
                "session_ttl_seconds": self._ttl_seconds if self._ttl_seconds > 0.0 else None,
                "last_session_creation_error": self._last_creation_error,
            }

    def readiness(self) -> dict[str, Any]:
        details = dict(self._readiness_probe() if self._readiness_probe is not None else {"ready": True})
        details["ready"] = bool(details.get("ready", True)) and self._last_creation_error is None
        details["active_sessions"] = self.health()["active_sessions"]
        details["max_live_sessions"] = self._max_sessions
        if self._last_creation_error is not None:
            details["last_session_creation_error"] = self._last_creation_error
        return details

    def _entry_locked(self, session_id: str) -> tuple[_SessionEntry, float]:
        now = self._clock()
        self._evict_expired_locked(now)
        entry = self._entries.get(session_id)
        if entry is None:
            raise SessionNotFoundError(f"Unknown or expired session: {session_id}")
        return entry, now

    def _evict_expired_locked(self, now: float) -> None:
        if self._ttl_seconds <= 0.0:
            return
        expired_ids = [
            session_id
            for session_id, entry in self._entries.items()
            if now - entry.last_used_monotonic >= self._ttl_seconds
        ]
        for session_id in expired_ids:
            entry = self._entries.pop(session_id)
            # This waits for an in-flight inference before model teardown.  It
            # is intentionally serialized with session creation because the
            # demo has a single GPU/session capacity.
            entry.processor.close()

    def _touch_locked(self, entry: _SessionEntry, now_monotonic: float) -> None:
        entry.last_used_monotonic = now_monotonic
        entry.last_used_at = self._wall_clock()

    def _snapshot_locked(self, entry: _SessionEntry, now: float) -> SessionInfo:
        expires = None
        if self._ttl_seconds > 0.0:
            expires = max(0.0, self._ttl_seconds - (now - entry.last_used_monotonic))
        return SessionInfo(
            session_id=entry.session_id,
            mode=entry.mode,
            camera_id=entry.camera_id,
            created_at=entry.created_at.isoformat(),
            last_used_at=entry.last_used_at.isoformat(),
            expires_in_seconds=round(expires, 2) if expires is not None else None,
        )

    def _normalize_mode(self, mode: str) -> str:
        normalized = str(mode).strip().lower()
        if normalized not in self._allowed_modes:
            allowed = ", ".join(sorted(self._allowed_modes))
            raise UnsupportedSessionModeError(f"Unsupported mode {mode!r}. Allowed modes: {allowed}.")
        return normalized

    @staticmethod
    def _normalize_camera_id(camera_id: str | None) -> str | None:
        if camera_id is None:
            return None
        normalized = str(camera_id).strip()
        if not normalized:
            return None
        if len(normalized) > 128:
            raise ValueError("camera_id must contain at most 128 characters.")
        return normalized


def build_crowd_pipeline_factory(
    mode_configs: dict[str, str],
    *,
    gender_model_path: str | None = None,
) -> PipelineFactory:
    """Create a lazy production factory without constructing a model yet."""

    normalized_configs = {str(mode).strip().lower(): str(path) for mode, path in mode_configs.items()}

    def factory(mode: str) -> LivePipeline:
        normalized_mode = str(mode).strip().lower()
        try:
            config_path = normalized_configs[normalized_mode]
        except KeyError as error:
            raise UnsupportedSessionModeError(f"No pipeline config is registered for mode {mode!r}.") from error
        # Keep this import and construction inside the factory.  ASGI startup
        # stays fast and tests can import the API without CUDA/Ultralytics work.
        from src.inference.pipeline import CrowdGenderPipeline

        pipeline = CrowdGenderPipeline(
            config_path=config_path,
            model_path=gender_model_path,
        )
        pipeline.warmup()
        return pipeline

    return factory
