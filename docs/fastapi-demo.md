# FastAPI demo API

The Modal deployment exposes a small, stateful API at `/api/v1`. It is
**API-only**: FastAPI's interactive schema is at `/docs` and the root URL
redirects there. The existing Gradio interface remains a separate local
application (`python app.py`).

The live pipeline is intentionally session-owned: a FastTracker instance,
session-scoped `person_id` resolver, counters, and heatmap belong to exactly
one active browser peer.  The demo limit is one live session, with a default
idle expiry of 600 seconds.

Keeping Modal API-only ensures its single session manager is the only owner of
the one GPU pipeline. A future shared-manager refactor can add a hosted UI
without risking two independent FastTracker states.

## Run locally

Install the normal runtime and the optional media transport dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r deploy\requirements-webrtc.txt
.\.venv\Scripts\python.exe -m uvicorn "src.api.app:create_api_app" --factory --host 127.0.0.1 --port 8000
```

`deploy/modal_app.py` also installs both dependency sets. Its current Modal
ASGI target exposes the REST API only; run `app.py` locally if a Gradio UI is
needed. Direct WebRTC media on a public Modal deployment has an additional
caveat documented below; no deployment command is needed merely to develop or
test the API locally.

The person detector is an application artifact at
`artifacts/person_detector/yolo11n.pt`, alongside the face and classification
models. Keep that checkpoint in place before starting the API or deploying to
Modal; the image copies this exact file and does not depend on Ultralytics'
automatic download cache.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Lightweight liveness and live-session capacity; never warms a model. |
| `GET` | `/api/v1/ready` | Checks that the configured profiles are present. Models warm on session creation. |
| `POST` | `/api/v1/sessions` | Creates a stateful tracker session for lifecycle/testing use. |
| `GET` | `/api/v1/sessions/{session_id}` | Gets session metadata and remaining TTL. |
| `GET` | `/api/v1/sessions/{session_id}/stats` | Gets the latest complete analytics/dashboard snapshot. |
| `POST` | `/api/v1/sessions/{session_id}/reset` | Clears FastTracker, `person_id`, counters, heatmap, and cadence telemetry. |
| `DELETE` | `/api/v1/sessions/{session_id}` | Closes the live worker and releases its tracking state. |
| `POST` | `/api/v1/webrtc/offer` | Self-hosted browser WebRTC signaling; creates the live session and returns the SDP answer. |
| `POST` | `/api/v1/video/analyze` | Optional bounded short-video analysis fallback. |

All application errors use a compact envelope such as:

```json
{
  "detail": {
    "code": "live_session_capacity_reached",
    "message": "Live demo capacity is 1 session(s); close the existing session first."
  }
}
```

## REST control and dashboard

Create a session only when testing the lifecycle via REST.  A WebRTC offer
creates its own session, so do not create a second one first.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"mode":"default","camera_id":"demo-camera"}'
```

The only allowed modes are `default` and `classroom_demo`.  `classroom_demo`
uses the deployed classroom profile; it reports an unconfigured/calibration
state until a real camera geometry config has been supplied.

Poll the dashboard endpoint roughly once per second:

```bash
curl http://127.0.0.1:8000/api/v1/sessions/demo_abc123/stats
```

Its response is deliberately one complete snapshot:

```json
{
  "status": "ready",
  "session": {"id": "demo_abc123", "mode": "default"},
  "frame": {"sequence": 42},
  "analytics": {
    "identity": {},
    "crowd": {},
    "spatial": {},
    "runtime": {}
  },
  "live_stream": {}
}
```

Before the first media frame, `status` is `waiting_for_frame` and `analytics`
is `null`.  Once frames arrive, `analytics` contains the current pipeline
envelope: local/confirmed tracks, session `person_id` mappings and recovery
counts, visual-presentation estimates, motion/crossing counters, spatial
heatmap and zone values, classroom state, detector recovery telemetry, and
latency/FPS/drop measurements.  `live_stream` contains the newest-frame queue
and worker telemetry used by the realtime path.

Resetting is intentionally destructive to only that in-memory session:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/sessions/demo_abc123/reset
curl -X DELETE http://127.0.0.1:8000/api/v1/sessions/demo_abc123
```

## WebRTC live camera (self-hosted ASGI)

`POST /api/v1/webrtc/offer` is the media entry point.  It accepts a complete
**non-trickle ICE** offer and returns an SDP answer plus a newly allocated
`session_id`:

```json
{
  "sdp": "v=0\\r\\n...",
  "type": "offer",
  "mode": "default",
  "camera_id": "browser-camera"
}
```

The returned annotated video track is the live preview.  The browser should
poll `/stats` using the returned `session_id` for analytics rather than trying
to infer telemetry from the video overlay.  The demo does not implement a
DataChannel, candidate endpoint, TURN credentials, or multi-camera routing.
It uses `stun:stun.l.google.com:19302` by default; set
`WEBRTC_STUN_SERVERS` to a comma-separated `stun:`/`stuns:` allow-list when
self-hosting. For a simple public test, configure the browser peer with that
same STUN service:

```js
new RTCPeerConnection({
  iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
})
```

It expects the browser to wait for ICE gathering to complete before posting
the offer. An empty `WEBRTC_STUN_SERVERS` value disables server-side STUN;
TURN URLs and credentials are deliberately rejected by this demo. If the
optional media packages were not installed, the endpoint returns `503` with
code `webrtc_unavailable`.

### Modal WebRTC boundary

The current API-only `deploy/modal_app.py` target is suitable for REST control
and analytics. Do **not** claim that its plain ASGI `aiortc` offer route is
production-ready WebRTC on Modal: Modal maps web requests to function-call
lifetimes, and its documented WebRTC architecture uses a persistent WebSocket
signaling function, a spawned `@app.cls` cloud peer, and TURN configuration
when NAT/firewall traversal needs it. The FastAPI route is tested for
local/self-hosted ASGI development; adding that Modal-specific peer/signaling
adapter is a separate deployment task. See [Modal's WebRTC YOLO
example](https://modal.com/docs/examples/webrtc_yolo).

## Short video fallback

Upload one short clip with `multipart/form-data`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/video/analyze \
  -F "file=@sample.mp4;type=video/mp4" \
  -F "mode=default"
```

The demo accepts `.mp4`, `.mov`, `.avi`, `.mkv`, and `.webm`, with default
limits of 64 MiB, 60 seconds, and 1,800 frames.  It uses a fresh tracking
state, returns final analytics and processing performance, deletes the source
afterward, and does **not** retain or publish an annotated video artifact.

## Demo boundaries

- Session IDs are ephemeral; `person_id` is session-scoped and resets with
  `POST /reset`, session expiry, or `DELETE`.
- A second live session is rejected with `429` until the first is closed or
  expires. This preserves tracker/session affinity on one demo GPU container.
- No authentication, durable storage, video-job queue, TURN service,
  Modal-specific WebRTC peer worker, or multi-camera scheduler is included at
  this stage.
