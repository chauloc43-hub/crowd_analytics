"""Modal entrypoint for Crowd Analytics MVP v2 (Clean Standalone CPU Cloud AI)."""

from __future__ import annotations

import json
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/crowd_tracking"
_REQUIRED_MODEL_ASSET_IDS = (
    "person_detector",
    "face_detector",
    "gender_classifier",
    "body_gender_classifier",
)
RUNTIME_PACKAGES = ("analytics", "api", "inference", "models", "tracking")


def _required_local_assets(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    manifest_path = project_root / "models" / "production-assets.json"
    if not manifest_path.exists():
        remote_root = Path(REMOTE_ROOT)
        return {
            "person_detector": remote_root / "artifacts" / "person_detector" / "yolo11n.pt",
            "face_detector": remote_root / "artifacts" / "face_detector" / "face_detection_yunet_2023mar.onnx",
            "gender_classifier": remote_root / "artifacts" / "gender_classifier" / "face_gender_classifier_mobilenet_v3_large.pth",
            "body_gender_classifier": remote_root / "artifacts" / "body_gender_classifier" / "body_gender_classifier_mobilenet_v3_small.pth",
        }
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload["assets"]

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
        by_id[asset_id] = resolved

    return {asset_id: by_id[asset_id] for asset_id in _REQUIRED_MODEL_ASSET_IDS}


def _runtime_image() -> modal.Image:
    model_assets = _required_local_assets()
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
        .pip_install(
            "fastapi==0.141.1",
            "uvicorn",
            "python-multipart==0.0.32",
            "opencv-python-headless==4.12.0.88",
            "torch==2.8.0",
            "torchvision==0.23.0",
            "ultralytics==8.4.63",
            "numpy==2.2.6",
            "pillow==11.3.0",
            "PyYAML==6.0.3",
        )
        .env(
            {
                "PIPELINE_CONFIG": f"{REMOTE_ROOT}/configs/pipeline-live.yaml",
                "GENDER_MODEL_PATH": f"{REMOTE_ROOT}/artifacts/gender_classifier/face_gender_classifier_mobilenet_v3_large.pth",
                "API_MAX_LIVE_SESSIONS": "10",
                "API_SESSION_TTL_SECONDS": "1800",
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
            remote_path=f"{REMOTE_ROOT}/artifacts/gender_classifier/face_gender_classifier_mobilenet_v3_large.pth",
            copy=True,
        )
        .add_local_file(
            model_assets["body_gender_classifier"],
            remote_path=f"{REMOTE_ROOT}/artifacts/body_gender_classifier/body_gender_classifier_mobilenet_v3_small.pth",
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


app = modal.App("crowd-analytics-v2")
image = _runtime_image()


@app.function(
    image=image,
    timeout=900,
    min_containers=1,
    max_containers=3,
    scaledown_window=600,
)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def web():
    from src.api.app import create_api_app

    return create_api_app()
