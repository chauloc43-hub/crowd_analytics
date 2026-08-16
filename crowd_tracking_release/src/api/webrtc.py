"""Optional WebRTC media-plane router for the Crowd Analytics demo.

The REST API remains usable without WebRTC dependencies.  ``aiortc`` and its
PyAV dependency are intentionally imported only while handling an SDP offer:
the lightweight HTTP API must still boot in environments that only need the
upload/frame-demo path.  In such an environment the offer endpoint returns a
clear 503 rather than pretending that a peer connection was established.

The browser must use *non-trickle ICE* for this small demo: it waits until its
offer has finished ICE gathering, then posts the complete offer here.  This
keeps the public API to one signalling endpoint.  aiortc peers use a default
public STUN server (overridable with ``WEBRTC_STUN_SERVERS``) for candidate
discovery.  TURN and a Modal-specific peer/TURN adapter remain intentionally
out of scope for this one-camera demo; restrictive NATs may therefore still
need a later production transport layer.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable, Literal, Protocol

import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field


LOGGER = logging.getLogger(__name__)

DEFAULT_STUN_SERVER = "stun:stun.l.google.com:19302"
STUN_SERVERS_ENV = "WEBRTC_STUN_SERVERS"


class WebRTCDependencyUnavailable(RuntimeError):
    """Raised when the optional WebRTC media runtime is not installed."""


@dataclass(frozen=True)
class AiortcBackend:
    """The small subset of aiortc/PyAV used by this module.

    Keeping this injectable makes the router testable without installing or
    opening real peer connections in the normal unit-test environment.
    """

    peer_connection_type: type[Any]
    session_description_type: type[Any]
    video_stream_track_type: type[Any]
    video_frame_type: Any
    configuration_type: type[Any] | None = None
    ice_server_type: type[Any] | None = None


def load_aiortc_backend() -> AiortcBackend:
    """Load aiortc lazily, with an actionable error for the API caller."""

    try:
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
        try:
            # Publicly re-exported by current aiortc releases.
            from aiortc import VideoStreamTrack
        except ImportError:  # pragma: no cover - compatibility with older aiortc layouts.
            from aiortc.mediastreams import VideoStreamTrack
        from av import VideoFrame
    except (ImportError, OSError) as exc:
        raise WebRTCDependencyUnavailable(
            "WebRTC is not enabled in this deployment. Install the optional "
            "media dependencies (for example: pip install aiortc av) and redeploy."
        ) from exc
    return AiortcBackend(
        peer_connection_type=RTCPeerConnection,
        session_description_type=RTCSessionDescription,
        video_stream_track_type=VideoStreamTrack,
        video_frame_type=VideoFrame,
        configuration_type=RTCConfiguration,
        ice_server_type=RTCIceServer,
    )


def _normalize_stun_servers(servers: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(server).strip() for server in servers if str(server).strip()))
    for server in normalized:
        if not server.lower().startswith(("stun:", "stuns:")):
            raise ValueError("STUN server URLs must start with stun: or stuns:.")
    return normalized


def resolve_stun_servers(raw_value: str | None = None) -> tuple[str, ...]:
    """Resolve comma-separated STUN URLs without exposing TURN credentials.

    The default helps browser-to-Modal candidate discovery. Set
    ``WEBRTC_STUN_SERVERS`` to a comma-separated list to use an organisation's
    STUN service, or to an empty string to intentionally disable STUN. TURN is
    not accepted here because this demo has no credential-management or relay
    policy yet.
    """

    configured = os.getenv(STUN_SERVERS_ENV) if raw_value is None else raw_value
    if configured is None:
        return (DEFAULT_STUN_SERVER,)
    return _normalize_stun_servers(configured.split(","))


def _create_peer_connection(
    backend: AiortcBackend,
    *,
    stun_servers: tuple[str, ...],
) -> Any:
    """Construct an aiortc peer with STUN, preserving dependency-free mocks."""

    # Test/alternate backends can omit aiortc's ICE configuration classes.
    # Existing lightweight mock peers then retain their zero-argument
    # constructor while real aiortc peers always receive an RTCConfiguration.
    if backend.configuration_type is None or backend.ice_server_type is None:
        return backend.peer_connection_type()
    ice_servers = [backend.ice_server_type(urls=server) for server in stun_servers]
    configuration = backend.configuration_type(iceServers=ice_servers)
    return backend.peer_connection_type(configuration=configuration)


class WebRTCSessionManager(Protocol):
    """Duck-typed contract supplied by ``src.api.sessions.DemoSessionManager``.

    The media router deliberately never imports the concrete REST/session
    manager.  It only needs a per-session latest-frame processor and the
    lifecycle methods below, which keeps the stateful FastTracker ownership in
    one place.
    """

    def create_session(
        self,
        *,
        mode: str = "default",
        camera_id: str | None = None,
    ) -> Any: ...

    def submit_frame(
        self,
        session_id: str,
        frame: np.ndarray,
        submitted_at: float | None = None,
    ) -> Any: ...

    def latest_result(self, session_id: str) -> Any: ...

    def close(self, session_id: str) -> Any: ...


class WebRTCOfferRequest(BaseModel):
    """A complete, non-trickle browser SDP offer."""

    sdp: str = Field(min_length=1, max_length=200_000)
    type: Literal["offer"] = "offer"
    mode: Literal["default", "classroom_demo"] = "default"
    camera_id: str | None = Field(default=None, max_length=128)


class WebRTCOfferResponse(BaseModel):
    session_id: str
    sdp: str
    type: Literal["answer"]
    mode: Literal["default", "classroom_demo"]
    ice_mode: Literal["non_trickle"] = "non_trickle"
    expires_in_seconds: int | None = None


@dataclass(frozen=True)
class _PeerRecord:
    session_id: str
    peer_connection: Any


class WebRTCPeerRegistry:
    """Own and close aiortc peers alongside their tracker sessions.

    ``DemoSessionManager`` owns the actual model/processor.  This registry only
    ensures that a browser disconnect also closes that manager session, and
    allows the REST ``DELETE /sessions/{id}`` handler to close a peer first.
    """

    def __init__(self, session_manager: WebRTCSessionManager) -> None:
        self._session_manager = session_manager
        self._records: dict[str, _PeerRecord] = {}
        self._lock = asyncio.Lock()

    async def register(self, session_id: str, peer_connection: Any) -> None:
        """Register a peer, replacing an old peer for the same session safely."""

        previous: _PeerRecord | None
        async with self._lock:
            previous = self._records.get(session_id)
            self._records[session_id] = _PeerRecord(session_id, peer_connection)
        if previous is not None and previous.peer_connection is not peer_connection:
            await self._close_peer_connection(previous.peer_connection)

    async def close_peer(
        self,
        session_id: str,
        *,
        expected_peer: Any | None = None,
    ) -> bool:
        """Close and forget a peer without touching the session manager."""

        record = await self._take(session_id, expected_peer=expected_peer)
        if record is None:
            return False
        await self._close_peer_connection(record.peer_connection)
        return True

    async def close_session(
        self,
        session_id: str,
        *,
        expected_peer: Any | None = None,
    ) -> bool:
        """Close a peer (if present) and release its stateful tracker session."""

        peer_closed = await self.close_peer(session_id, expected_peer=expected_peer)
        self._close_manager_session(session_id)
        return peer_closed

    async def close_all(self) -> None:
        """Close all current peers and their sessions during application shutdown."""

        async with self._lock:
            records = list(self._records.values())
            self._records.clear()
        for record in records:
            await self._close_peer_connection(record.peer_connection)
            self._close_manager_session(record.session_id)

    async def peer_count(self) -> int:
        async with self._lock:
            return len(self._records)

    async def _take(self, session_id: str, *, expected_peer: Any | None) -> _PeerRecord | None:
        async with self._lock:
            record = self._records.get(session_id)
            if record is None:
                return None
            # An old connection-state callback must never tear down a newly
            # registered peer for the same session id.
            if expected_peer is not None and record.peer_connection is not expected_peer:
                return None
            return self._records.pop(session_id)

    async def _close_peer_connection(self, peer_connection: Any) -> None:
        try:
            result = peer_connection.close()
            if inspect.isawaitable(result):
                await result
        except Exception:  # Browser teardown is best effort and must not leak tracker state.
            LOGGER.debug("Unable to close WebRTC peer cleanly.", exc_info=True)

    def _close_manager_session(self, session_id: str) -> None:
        try:
            self._session_manager.close(session_id)
        except Exception:
            # REST deletion and a connection-state callback may race. Closing
            # an already-released demo session is intentionally idempotent. We
            # intentionally do not import the concrete SessionNotFoundError so
            # this optional adapter remains independent of the REST layer.
            LOGGER.debug("WebRTC cleanup found no live session %s.", session_id, exc_info=True)


def _session_id_from(entry: Any) -> str:
    """Accept the manager's SessionEntry object or a simple string in tests."""

    if isinstance(entry, str):
        return entry
    session_id = getattr(entry, "session_id", None)
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError("Session manager create_session() did not return a session_id.")
    return session_id


def _latest_annotated_frame(result: Any) -> np.ndarray | None:
    """Extract a completed ``LiveFrameResult`` without coupling to its class."""

    frame = getattr(result, "annotated_frame", None)
    if isinstance(frame, np.ndarray) and frame.ndim == 3:
        return frame
    return None


def _make_annotated_video_track(
    backend: AiortcBackend,
    session_manager: WebRTCSessionManager,
    session_id: str,
) -> type[Any]:
    """Build a transform track only after aiortc is available.

    The source frame is offered to ``LiveFrameProcessor`` and the most recent
    completed annotation is returned.  This deliberately does not await model
    inference in ``recv``: the processor's capacity-one mailbox retains only
    the newest input frame, so a slow detector cannot create a growing media
    backlog or mutate one FastTracker concurrently.
    """

    class AnnotatedVideoTrack(backend.video_stream_track_type):
        kind = "video"

        def __init__(self, source_track: Any) -> None:
            super().__init__()
            self._source_track = source_track

        async def recv(self) -> Any:
            source_frame = await self._source_track.recv()
            try:
                source_bgr = source_frame.to_ndarray(format="bgr24")
                session_manager.submit_frame(
                    session_id,
                    source_bgr,
                    submitted_at=monotonic(),
                )
                completed = session_manager.latest_result(session_id)
                annotated_bgr = _latest_annotated_frame(completed)
                # Keep a valid live image flowing until the first asynchronous
                # inference result is ready, and guard against a mismatched
                # result after a caller changes source resolution.
                if annotated_bgr is None or annotated_bgr.shape[:2] != source_bgr.shape[:2]:
                    annotated_bgr = source_bgr
            except Exception:
                # A bad media frame must not permanently terminate the browser
                # preview. Pipeline errors are separately recorded by the live
                # processor telemetry exposed via /stats.
                LOGGER.debug("WebRTC video frame could not be submitted.", exc_info=True)
                return source_frame

            output_frame = backend.video_frame_type.from_ndarray(
                np.ascontiguousarray(annotated_bgr),
                format="bgr24",
            )
            output_frame.pts = source_frame.pts
            output_frame.time_base = source_frame.time_base
            return output_frame

    return AnnotatedVideoTrack


def _unavailable_detail(exc: WebRTCDependencyUnavailable) -> dict[str, str]:
    return {
        "code": "webrtc_unavailable",
        "message": str(exc),
        "remediation": "Install the optional aiortc and av packages in the API image, then redeploy.",
    }


def create_webrtc_router(
    session_manager: WebRTCSessionManager,
    *,
    prefix: str = "/api/v1",
    registry: WebRTCPeerRegistry | None = None,
    backend_loader: Callable[[], AiortcBackend] = load_aiortc_backend,
    session_ttl_seconds: int | None = 600,
    stun_servers: tuple[str, ...] | None = None,
) -> APIRouter:
    """Create the minimal non-trickle WebRTC offer endpoint.

    The caller should retain the supplied/returned registry (normally in
    ``app.state.webrtc_peers``).  A REST session-delete handler can then call
    ``await registry.close_peer(session_id)`` before it calls
    ``session_manager.close(session_id)``.  Connection failures call
    ``registry.close_session`` automatically.  Real aiortc peers use the
    default STUN server unless ``stun_servers`` is explicitly supplied or the
    ``WEBRTC_STUN_SERVERS`` environment variable overrides it.
    """

    if session_ttl_seconds is not None and session_ttl_seconds < 0:
        raise ValueError("session_ttl_seconds must be non-negative or None.")
    if session_ttl_seconds == 0:
        session_ttl_seconds = None
    effective_stun_servers = (
        resolve_stun_servers() if stun_servers is None else _normalize_stun_servers(stun_servers)
    )
    peer_registry = registry or WebRTCPeerRegistry(session_manager)
    router = APIRouter(prefix=prefix, tags=["webrtc"])

    @router.post(
        "/webrtc/offer",
        response_model=WebRTCOfferResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_webrtc_offer(offer: WebRTCOfferRequest) -> WebRTCOfferResponse:
        # Do this before creating a session, avoiding a model/session leak when
        # the optional WebRTC media runtime was omitted from the deployment.
        try:
            backend = backend_loader()
        except WebRTCDependencyUnavailable as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=_unavailable_detail(exc),
                headers={"Retry-After": "60"},
            ) from exc

        if not offer.sdp.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"code": "invalid_sdp", "message": "SDP offer must not be blank."},
            )

        entry = session_manager.create_session(mode=offer.mode, camera_id=offer.camera_id)
        session_id = _session_id_from(entry)
        response_mode = str(getattr(entry, "mode", offer.mode))
        response_ttl = getattr(entry, "expires_in_seconds", session_ttl_seconds)
        if response_ttl is not None:
            response_ttl = max(0, int(float(response_ttl)))
        try:
            peer_connection = _create_peer_connection(backend, stun_servers=effective_stun_servers)
        except Exception as exc:
            # The tracker was created first, so explicitly release it if an
            # optional media-runtime/configuration error prevents peer setup.
            await peer_registry.close_session(session_id)
            LOGGER.exception("Unable to initialize WebRTC peer for demo session %s", session_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "code": "webrtc_peer_initialization_failed",
                    "message": "The WebRTC peer could not be initialized by this deployment.",
                },
            ) from exc
        await peer_registry.register(session_id, peer_connection)

        video_attached = False

        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            connection_state = str(getattr(peer_connection, "connectionState", "")).lower()
            if connection_state in {"failed", "closed"}:
                await peer_registry.close_session(session_id, expected_peer=peer_connection)

        @peer_connection.on("track")
        def on_track(track: Any) -> None:
            nonlocal video_attached
            if getattr(track, "kind", None) != "video" or video_attached:
                return
            video_attached = True
            annotated_track = _make_annotated_video_track(
                backend,
                session_manager,
                session_id,
            )(track)
            peer_connection.addTrack(annotated_track)

        try:
            await peer_connection.setRemoteDescription(
                backend.session_description_type(sdp=offer.sdp, type=offer.type)
            )
            answer = await peer_connection.createAnswer()
            await peer_connection.setLocalDescription(answer)
            local_description = getattr(peer_connection, "localDescription", None)
            if local_description is None:
                raise RuntimeError("WebRTC peer did not create a local SDP answer.")
        except Exception as exc:
            await peer_registry.close_session(session_id, expected_peer=peer_connection)
            LOGGER.info("Rejected WebRTC offer for demo session %s: %s", session_id, type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "invalid_webrtc_offer",
                    "message": "The SDP offer could not be accepted by the WebRTC peer.",
                },
            ) from exc

        answer_type = str(getattr(local_description, "type", "answer"))
        answer_sdp = str(getattr(local_description, "sdp", ""))
        if answer_type != "answer" or not answer_sdp:
            await peer_registry.close_session(session_id, expected_peer=peer_connection)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "code": "webrtc_answer_unavailable",
                    "message": "The peer did not produce a usable SDP answer.",
                },
            )
        return WebRTCOfferResponse(
            session_id=session_id,
            sdp=answer_sdp,
            type="answer",
            mode=response_mode,  # manager is the source of truth for normalized mode.
            expires_in_seconds=response_ttl,
        )

    # APIRouter intentionally has no State object. This attribute is a small,
    # explicit hand-off for an application factory that wants to store the
    # registry on ``FastAPI.state`` and use it in DELETE/lifespan handlers.
    setattr(router, "webrtc_peer_registry", peer_registry)
    return router


__all__ = [
    "AiortcBackend",
    "DEFAULT_STUN_SERVER",
    "STUN_SERVERS_ENV",
    "WebRTCDependencyUnavailable",
    "WebRTCOfferRequest",
    "WebRTCOfferResponse",
    "WebRTCPeerRegistry",
    "WebRTCSessionManager",
    "create_webrtc_router",
    "load_aiortc_backend",
    "resolve_stun_servers",
]
