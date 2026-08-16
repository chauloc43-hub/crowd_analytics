"""FastAPI factory for the bounded Crowd Analytics demonstration API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from src.api.config import ApiSettings
from src.api.sessions import (
    ApiSessionError,
    DemoSessionManager,
    SessionCapacityError,
    SessionInitializationError,
    SessionManager,
    SessionNotFoundError,
    UnsupportedSessionModeError,
    build_crowd_pipeline_factory,
)
from src.api.video import (
    ShortVideoAnalyzer,
    VideoAnalysisBusyError,
    UnsupportedVideoError,
    VideoAnalysisError,
    VideoAnalyzer,
    VideoTooLargeError,
    VideoTooLongError,
)
from src.api.webrtc import WebRTCPeerRegistry, create_webrtc_router


API_PREFIX = "/api/v1"


class CreateSessionRequest(BaseModel):
    """Only allow named demo profiles; never accept client filesystem paths."""

    mode: str = Field(default="default", description="`default` or `classroom_demo`.")
    camera_id: str | None = Field(default=None, max_length=128)


def create_api_app(
    *,
    settings: ApiSettings | None = None,
    session_manager: SessionManager | None = None,
    video_analyzer: VideoAnalyzer | None = None,
) -> FastAPI:
    """Create the API without loading a model until a session/job is requested.

    ``session_manager`` and ``video_analyzer`` are explicit injection points for
    tests, hardware-specific runtimes, and the WebRTC adapter.  This keeps the
    REST layer free from singleton model state at import time.
    """

    effective_settings = settings or ApiSettings.from_environment()
    pipeline_factory = build_crowd_pipeline_factory(
        {mode: str(path) for mode, path in effective_settings.modes.items()},
        gender_model_path=effective_settings.gender_model_path,
    )
    manager: SessionManager = session_manager or DemoSessionManager(
        pipeline_factory,
        allowed_modes=tuple(effective_settings.modes),
        max_sessions=effective_settings.max_live_sessions,
        ttl_seconds=effective_settings.session_ttl_seconds,
        cadence_seconds=effective_settings.live_cadence_seconds,
        profiling_window=effective_settings.profiling_window,
        readiness_probe=effective_settings.readiness,
    )
    analyzer: VideoAnalyzer = video_analyzer or ShortVideoAnalyzer(
        pipeline_factory,
        allowed_modes=tuple(effective_settings.modes),
        max_bytes=effective_settings.max_video_bytes,
        max_seconds=effective_settings.max_video_seconds,
        max_frames=effective_settings.max_video_frames,
    )
    webrtc_peers = WebRTCPeerRegistry(manager)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            # A tracker worker owns native/GPU resources.  Sessions are still
            # released by DELETE or TTL during normal use, but app shutdown
            # must clean up any remaining worker deterministically.
            # Close media transports before their associated model workers.
            # The registry also closes WebRTC-owned sessions; close_all then
            # releases any REST-only sessions that have no peer record.
            await webrtc_peers.close_all()
            manager.close_all()

    app = FastAPI(
        title="Crowd Analytics Demo API",
        version="0.1.0",
        description=(
            "Small stateful API for demonstrating FastTracker, stream-scoped "
            "person IDs, visual presentation analytics, heatmap, and live telemetry."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.api_settings = effective_settings
    app.state.session_manager = manager
    app.state.video_analyzer = analyzer
    app.state.webrtc_peers = webrtc_peers
    app.include_router(
        create_webrtc_router(
            manager,
            registry=webrtc_peers,
            session_ttl_seconds=(
                int(effective_settings.session_ttl_seconds)
                if effective_settings.session_ttl_seconds > 0.0
                else None
            ),
        )
    )

    @app.exception_handler(SessionNotFoundError)
    async def session_not_found_handler(_request, exc: SessionNotFoundError) -> JSONResponse:
        return _error_response(404, "session_not_found", str(exc))

    @app.exception_handler(SessionCapacityError)
    async def session_capacity_handler(_request, exc: SessionCapacityError) -> JSONResponse:
        return _error_response(429, "live_session_capacity_reached", str(exc))

    @app.exception_handler(SessionInitializationError)
    async def session_initialization_handler(_request, exc: SessionInitializationError) -> JSONResponse:
        return _error_response(503, "pipeline_unavailable", str(exc))

    @app.exception_handler(UnsupportedSessionModeError)
    async def mode_handler(_request, exc: UnsupportedSessionModeError) -> JSONResponse:
        return _error_response(422, "unsupported_mode", str(exc))

    @app.exception_handler(ApiSessionError)
    async def session_error_handler(_request, exc: ApiSessionError) -> JSONResponse:
        return _error_response(409, "session_state_error", str(exc))

    @app.exception_handler(VideoTooLargeError)
    async def video_size_handler(_request, exc: VideoTooLargeError) -> JSONResponse:
        return _error_response(413, "video_too_large", str(exc))

    @app.exception_handler(VideoAnalysisBusyError)
    async def video_busy_handler(_request, exc: VideoAnalysisBusyError) -> JSONResponse:
        return _error_response(429, "video_analysis_busy", str(exc))

    @app.exception_handler(VideoTooLongError)
    async def video_length_handler(_request, exc: VideoTooLongError) -> JSONResponse:
        return _error_response(422, "video_too_long", str(exc))

    @app.exception_handler(UnsupportedVideoError)
    async def unsupported_video_handler(_request, exc: UnsupportedVideoError) -> JSONResponse:
        return _error_response(415, "unsupported_video", str(exc))

    @app.exception_handler(VideoAnalysisError)
    async def video_error_handler(_request, exc: VideoAnalysisError) -> JSONResponse:
        return _error_response(422, "video_analysis_failed", str(exc))

    @app.get("/", include_in_schema=False)
    def api_root() -> RedirectResponse:
        """Redirect root to the mobile-first Cyber-HUD web interface."""

        return RedirectResponse(url="/static/index.html", status_code=307)

    # Mount static files for the Cyber-HUD Web Frontend
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


    @app.get(f"{API_PREFIX}/health", tags=["service"])
    def health() -> dict[str, Any]:
        """Liveness check: does not initialize a pipeline or a GPU model."""

        return _json_safe(
            {
                "status": "ok",
                "service": "crowd-analytics-demo-api",
                "sessions": manager.health(),
            }
        )

    @app.get(f"{API_PREFIX}/ready", tags=["service"])
    def ready() -> JSONResponse:
        """Report whether the configured default profile can be created on demand."""

        details = _json_safe(manager.readiness())
        is_ready = bool(details.get("ready", False))
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content={"status": "ready" if is_ready else "not_ready", "service": "crowd-analytics-demo-api", **details},
        )

    @app.post(f"{API_PREFIX}/sessions", status_code=201, tags=["sessions"])
    def create_session(request: CreateSessionRequest) -> dict[str, Any]:
        """Create the one persistent tracker/person-ID state for a media peer."""

        session = manager.create_session(request.mode, request.camera_id)
        return _json_safe({"status": "created", "session": session.to_dict()})

    @app.get(f"{API_PREFIX}/sessions/{{session_id}}", tags=["sessions"])
    def get_session(session_id: str) -> dict[str, Any]:
        return _json_safe({"status": "active", "session": manager.get(session_id).to_dict()})

    @app.get(f"{API_PREFIX}/sessions/{{session_id}}/stats", tags=["sessions"])
    def get_session_stats(session_id: str) -> dict[str, Any]:
        """Return the whole latest analytics envelope for dashboard polling."""

        state = manager.get_state(session_id)
        result = state.result
        payload: dict[str, Any] = {
            "status": "ready" if result is not None else "waiting_for_frame",
            "session": state.info.to_dict(),
            "frame": (
                {
                    "sequence": result.sequence,
                    "submitted_monotonic_seconds": result.submitted_at,
                    "completed_monotonic_seconds": result.completed_at,
                }
                if result is not None
                else None
            ),
            # This envelope contains all model measurements (tracking,
            # identity, attributes, spatial/heatmap, classroom, and runtime).
            "analytics": result.stats if result is not None else None,
            "live_stream": state.telemetry,
        }
        return _json_safe(payload)

    @app.post(f"{API_PREFIX}/sessions/{{session_id}}/reset", tags=["sessions"])
    def reset_session(session_id: str) -> dict[str, Any]:
        """Reset FastTracker, stream person IDs, counters, and heatmap together."""

        session = manager.reset(session_id)
        return _json_safe({"status": "reset", "session": session.to_dict()})

    @app.post(f"{API_PREFIX}/sessions/{{session_id}}/frame", tags=["sessions"])
    async def submit_session_frame(
        session_id: str,
        file: UploadFile = File(..., description="JPEG/PNG camera frame for live session tracking."),
        boxes: bool = Query(True),
        ids: bool = Query(True),
        attributes: bool = Query(True),
        motion: bool = Query(False),
        zones: bool = Query(True),
        seats: bool = Query(True),
    ) -> dict[str, Any]:
        """Submit a camera frame to an active stateful live session."""

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return _error_response(422, "invalid_image", "Could not decode image frame.")
        
        processor = manager.get_processor(session_id)
        # Dynamic overlay flags set by frontend control panel
        processor.pipeline.config.setdefault("visualization", {})
        processor.pipeline.config["visualization"].update({
            "show_boxes": boxes,
            "show_ids": ids,
            "show_attributes": attributes,
            "show_motion": motion,
            "show_motion_labels": motion,
            "show_trajectories": motion,
            "show_zones": zones,
            "show_seats": seats,
        })

        # Directly process frame synchronously using session's pipeline worker
        annotated_frame, stats = processor.pipeline.process_frame(frame)
        
        # Encode annotated output frame with bounding boxes & HUD into base64 JPEG for live preview
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        import base64
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        return _json_safe({
            "status": "completed",
            "annotated_frame": f"data:image/jpeg;base64,{frame_base64}",
            "analytics": stats,
            "session": manager.get(session_id).to_dict()
        })





    @app.delete(f"{API_PREFIX}/sessions/{{session_id}}", status_code=204, tags=["sessions"])
    async def delete_session(session_id: str) -> Response:
        await webrtc_peers.close_peer(session_id)
        manager.close(session_id)
        return Response(status_code=204)

    @app.post(f"{API_PREFIX}/video/analyze", tags=["video"])
    def analyze_short_video(
        file: UploadFile = File(..., description="Short video clip; default demo limit is 60 seconds / 64 MiB."),
        mode: str = Form("default"),
    ) -> dict[str, Any]:
        """Synchronous short-clip fallback; it never changes a live session's tracker."""

        # Short video analysis fallback
        result = analyzer.analyze(
            file.file,
            filename=file.filename,
            content_type=file.content_type,
            mode=mode,
        )
        return _json_safe(result)

    return app


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": {"code": code, "message": message}})


def _json_safe(value: Any) -> Any:
    """Normalize numpy/OpenCV analytics values before JSON serialization."""

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    return str(value)
