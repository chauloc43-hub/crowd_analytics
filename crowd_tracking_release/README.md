# Crowd Analytics MVP

Production-ready demo for webcam and short-video crowd analytics.

```text
YOLO11n -> FastTracker -> session-scoped person_id -> visual-presentation evidence -> crowd analytics
```

The selected live pipeline is FastTracker in `configs/pipeline-live.yaml`. It
keeps tracking state per stream and adds a conservative, session-scoped
`person_id` above tracker-local IDs. A person ID is not biometric identity and
is cleared when a session resets or expires.

## Included in this repository

- Local Gradio application (`app.py`)
- FastAPI demo API, including short-video analysis and WebRTC offer signaling
- FastTracker, persistent session person IDs, detector recovery, crowd
  statistics, heatmap, zones, and classroom-layout support
- Modal API deployment definition

Datasets, evaluation outputs, notebooks, tests, A/B profiles, development
scripts, virtual environments, and binary model checkpoints are deliberately
kept out of Git. They stay local and are not deleted by this cleanup.

## Provision model assets

The production application needs these operator-provided local files:

```text
artifacts/person_detector/yolo11n.pt
artifacts/face_detector/face_detection_yunet_2023mar.onnx
artifacts/gender_classifier/best_model.pth
artifacts/body_gender_classifier/best_body_gender_model.pth
```

Their expected checksums and safe verification/copy workflow are in
[docs/model-assets.md](docs/model-assets.md). The runtime fails clearly if an
asset is missing; it does not download a model during a request.

## Run locally

Create an environment, provision the four assets, then install the local UI
runtime:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools\prepare_production_assets.py
.\.venv\Scripts\python.exe app.py
```

`app.py` opens the local Gradio demo. It uses `configs/pipeline-live.yaml` by
default. Copy `.env.example` to `.env` only when environment-specific paths or
limits need to change.

## FastAPI demo

Install the API dependencies and run the application factory with Uvicorn:

```powershell
.\.venv\Scripts\python.exe -m pip install -r deploy\requirements-api-runtime.txt
.\.venv\Scripts\python.exe -m uvicorn src.api.app:create_api_app --factory --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API. The compact API
guide, request examples, and response metrics are in
[docs/fastapi-demo.md](docs/fastapi-demo.md).

The WebRTC offer endpoint is optional. For a self-hosted WebRTC demo, install
`deploy/requirements-webrtc.txt`. Public Modal WebRTC needs a dedicated
signaling/peer/TURN adapter and is not represented as a production promise by
this MVP.

## Modal API deployment

Modal packages the API and the four provisioned local assets into its image;
it does not upload the local dataset or development material.

```powershell
.\.venv\Scripts\python.exe tools\prepare_production_assets.py
.\.venv\Scripts\python.exe -m pip install -r deploy\requirements-modal.txt
.\.venv\Scripts\modal.exe setup
.\.venv\Scripts\modal.exe deploy deploy\modal_app.py
```

The Modal deployment is API-only: `/` redirects to `/docs`, and routes live
under `/api/v1`. It is configured for one stateful live session, appropriate
for the demo's GPU capacity.

## Scope and privacy

Face/body classifier results are visual-presentation estimates, not verified
identity attributes. This demo has no authentication, storage service, or
cross-session identity. Add access control, consent/retention policy, and
camera-specific calibration before any real deployment.
