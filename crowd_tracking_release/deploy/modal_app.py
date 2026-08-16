"""Modal entrypoint for the single-stream Crowd Analytics MVP.

Install the local deploy CLI with:
    pip install -r deploy/requirements-modal.txt

Then authenticate once and deploy:
    modal setup
    modal deploy deploy/modal_app.py

The Image below is an explicit allow-list.  It intentionally excludes datasets,
notebooks, test videos, benchmark artifacts, documentation and experimental
profiles even if they remain in the local workspace.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/crowd_tracking"
# Keep the detector alongside the other application-owned artifacts.  This
# avoids relying on Ultralytics' working-directory cache (and an implicit
# first-request download) in the deployed container.
PERSON_DETECTOR_MODEL = PROJECT_ROOT / "artifacts" / "person_detector" / "yolo11n.pt"
PRODUCTION_ASSET_MANIFEST = PROJECT_ROOT / "models" / "production-assets.json"
PRODUCTION_ASSET_BOOTSTRAP_COMMAND = "python tools/prepare_production_assets.py"
_REQUIRED_MODEL_ASSET_IDS = (
    "person_detector",
    "face_detector",
    "gender_classifier",
    "body_gender_classifier",
)
# `api` is a first-class runtime package: the Modal ASGI app is API-only. This
# preserves one owner for every stateful FastTracker/person_id pipeline. The
# legacy Gradio UI remains a separate local demo in `app.py`.
RUNTIME_PACKAGES = ("analytics", "api", "inference", "models", "tracking")


def _required_local_assets(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    """Load and validate the model files that must exist before an image build.

    Checkpoints are intentionally ignored by Git.  Failing here is much more
    useful than letting ``add_local_file`` fail halfway through a Modal deploy
    (or letting Ultralytics attempt an implicit download at runtime).
    """

    manifest_path = project_root / "models" / "production-assets.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = payload["assets"]
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise FileNotFoundError(
            "Production asset manifest is unavailable or invalid: "
            f"{manifest_path}. Restore it, then run `{PRODUCTION_ASSET_BOOTSTRAP_COMMAND}` "
            "before deploying Modal."
        ) from error

    if not isinstance(entries, list):
        raise ValueError(
            f"Production asset manifest must contain an 'assets' list: {manifest_path}. "
            f"Run `{PRODUCTION_ASSET_BOOTSTRAP_COMMAND}` after restoring the manifest."
        )

    root = project_root.resolve()
    by_id: dict[str, Path] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        asset_id = entry.get("id")
        configured_path = entry.get("path")
        if not isinstance(asset_id, str) or not isinstance(configured_path, str):
            continue
        candidate = Path(configured_path)
        if candidate.is_absolute():
            continue
        resolved = (project_root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        by_id[asset_id] = resolved

    absent_from_manifest = [asset_id for asset_id in _REQUIRED_MODEL_ASSET_IDS if asset_id not in by_id]
    missing_files = [
        f"{asset_id}: {by_id[asset_id]}"
        for asset_id in _REQUIRED_MODEL_ASSET_IDS
        if asset_id in by_id and not by_id[asset_id].is_file()
    ]
    if absent_from_manifest or missing_files:
        details = [
            *(f"{asset_id}: missing manifest entry" for asset_id in absent_from_manifest),
            *missing_files,
        ]
        raise FileNotFoundError(
            "Required production model assets are missing before the Modal image can be built:\n"
            + "\n".join(f"- {detail}" for detail in details)
            + f"\nRun `{PRODUCTION_ASSET_BOOTSTRAP_COMMAND}` and retry `modal deploy deploy/modal_app.py`."
        )
    return {asset_id: by_id[asset_id] for asset_id in _REQUIRED_MODEL_ASSET_IDS}


def _runtime_image() -> modal.Image:
    """Build a reproducible image containing only the selected live profile."""

    # Do this before instantiating ``modal.Image`` so a deployment fails with
    # an actionable local remediation rather than an opaque image-build error.
    model_assets = _required_local_assets()
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
        .pip_install_from_requirements(str(PROJECT_ROOT / "deploy" / "requirements-api-runtime.txt"))
        # WebRTC is intentionally separate from the normal runtime list so
        # local Gradio-only development does not need media transport wheels.
        # It keeps the self-hosted FastAPI `/api/v1/webrtc/offer` contract
        # available in the image. A production Modal WebRTC peer still needs
        # Modal's dedicated signaling/peer/TURN adapter; see the API guide.
        .pip_install_from_requirements(str(PROJECT_ROOT / "deploy" / "requirements-webrtc.txt"))
        .env(
            {
                "PIPELINE_CONFIG": f"{REMOTE_ROOT}/configs/pipeline-live.yaml",
                "GENDER_MODEL_PATH": f"{REMOTE_ROOT}/artifacts/gender_classifier/best_model.pth",
                # API-only Modal deployment: one manager owns the one active
                # pipeline. `app.py` is intentionally not mounted here.
                "API_MAX_LIVE_SESSIONS": "1",
                "API_SESSION_TTL_SECONDS": "600",
                "LIVE_STREAM_EVERY_SECONDS": "0.15",
                "YOLO_CONFIG_DIR": "/tmp/ultralytics",
            }
        )
        .workdir(REMOTE_ROOT)
        .add_local_file(
            PROJECT_ROOT / "src" / "__init__.py",
            remote_path=f"{REMOTE_ROOT}/src/__init__.py",
            copy=True,
        )
        .add_local_file(
            PROJECT_ROOT / "configs" / "pipeline-live.yaml",
            remote_path=f"{REMOTE_ROOT}/configs/pipeline-live.yaml",
            copy=True,
        )
        .add_local_file(
            PROJECT_ROOT / "configs" / "pipeline-classroom-template.yaml",
            remote_path=f"{REMOTE_ROOT}/configs/pipeline-classroom-template.yaml",
            copy=True,
        )
        .add_local_file(
            PROJECT_ROOT / "configs" / "fasttrack-live.yaml",
            remote_path=f"{REMOTE_ROOT}/configs/fasttrack-live.yaml",
            copy=True,
        )
        .add_local_file(
            model_assets["person_detector"],
            remote_path=f"{REMOTE_ROOT}/artifacts/person_detector/yolo11n.pt",
            copy=True,
        )
        .add_local_file(
            model_assets["face_detector"],
            remote_path=f"{REMOTE_ROOT}/artifacts/face_detector/face_detection_yunet_2023mar.onnx",
            copy=True,
        )
        .add_local_file(
            model_assets["gender_classifier"],
            remote_path=f"{REMOTE_ROOT}/artifacts/gender_classifier/best_model.pth",
            copy=True,
        )
        .add_local_file(
            model_assets["body_gender_classifier"],
            remote_path=f"{REMOTE_ROOT}/artifacts/body_gender_classifier/best_body_gender_model.pth",
            copy=True,
        )
    )
    for package in RUNTIME_PACKAGES:
        image = image.add_local_dir(
            PROJECT_ROOT / "src" / package,
            remote_path=f"{REMOTE_ROOT}/src/{package}",
            copy=True,
            ignore=["__pycache__", "**/__pycache__", "*.pyc", "ultralytics_reid_compat.py"],
        )
    return image


app = modal.App("crowd-analytics-mvp")
image = _runtime_image()


@app.function(
    image=image,
    gpu="T4",
    timeout=900,
    max_containers=1,
    scaledown_window=600,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def web():
    """Serve the API-only Modal demonstration.

    The FastAPI session manager is the sole owner of the T4's live pipeline,
    preserving FastTracker and session-scoped person-ID continuity.  The
    existing Gradio application is intentionally left as a separate local
    entry point until both surfaces can share one manager safely.
    """

    from src.api.app import create_api_app

    return create_api_app()
