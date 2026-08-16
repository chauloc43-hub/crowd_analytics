"""Configuration for the intentionally small demo API.

All values are environment-overridable so the same source can be used locally
and in Modal.  Importing this module only resolves paths; it never loads a
YOLO, gender, or tracker model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


_PRODUCTION_ASSET_MANIFEST = Path("models") / "production-assets.json"
_PRODUCTION_ASSET_BOOTSTRAP_COMMAND = "python tools/prepare_production_assets.py"
_REQUIRED_MODEL_ASSET_IDS = (
    "person_detector",
    "face_detector",
    "gender_classifier",
    "body_gender_classifier",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _path_from_environment(name: str, default: Path, project_root: Path) -> Path:
    value = os.getenv(name)
    candidate = Path(value) if value else default
    return candidate if candidate.is_absolute() else project_root / candidate


def _positive_int_from_environment(name: str, default: int, *, minimum: int = 1) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _nonnegative_float_from_environment(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _production_asset_status(
    project_root: Path,
    *,
    gender_model_path: str | None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return local production model availability without loading any model.

    The manifest is deliberately the source of truth rather than the Python
    dependency graph: the same four artifacts are copied into the Modal image
    and are intentionally excluded from the source repository.  This makes a
    missing bootstrap visible through ``/ready`` before a client opens a live
    session.
    """

    manifest_path = project_root / _PRODUCTION_ASSET_MANIFEST
    if not manifest_path.is_file():
        error = f"Production asset manifest not found: {manifest_path}"
        return _unavailable_asset_statuses(error), error

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assets = payload["assets"]
    except (OSError, TypeError, ValueError, KeyError) as error:
        message = f"Could not read production asset manifest {manifest_path}: {error}"
        return _unavailable_asset_statuses(message), message

    if not isinstance(assets, list) or not assets:
        message = f"Production asset manifest {manifest_path} must contain a non-empty 'assets' list."
        return _unavailable_asset_statuses(message), message

    entries_by_id: dict[str, dict[str, Any]] = {}
    for entry in assets:
        if not isinstance(entry, dict):
            continue
        asset_id = entry.get("id")
        if not isinstance(asset_id, str) or not asset_id.strip():
            continue
        entries_by_id[asset_id] = entry

    statuses: list[dict[str, Any]] = []
    for asset_id in _REQUIRED_MODEL_ASSET_IDS:
        entry = entries_by_id.get(asset_id)
        if entry is None:
            statuses.append(
                {
                    "id": asset_id,
                    "path": None,
                    "present": False,
                    "error": "missing manifest entry",
                }
            )
            continue
        configured_path = entry.get("path")
        if not isinstance(configured_path, str) or not configured_path.strip():
            statuses.append(
                {
                    "id": asset_id,
                    "path": None,
                    "present": False,
                    "error": "invalid manifest path",
                }
            )
            continue

        # The face classifier is the one production asset with a documented
        # environment override.  Report the effective checkpoint rather than
        # incorrectly requiring the default one as well.
        effective_path = gender_model_path if asset_id == "gender_classifier" and gender_model_path else configured_path
        path = Path(effective_path)
        resolved_path = path if path.is_absolute() else project_root / path
        statuses.append(
            {
                "id": asset_id,
                "path": str(resolved_path),
                "present": resolved_path.is_file(),
            }
        )

    return statuses, None


def _unavailable_asset_statuses(error: str) -> list[dict[str, Any]]:
    return [
        {
            "id": asset_id,
            "path": None,
            "present": False,
            "error": error,
        }
        for asset_id in _REQUIRED_MODEL_ASSET_IDS
    ]


@dataclass(frozen=True)
class ApiSettings:
    """Bounded configuration appropriate for the one-camera demo.

    ``max_live_sessions`` intentionally defaults to one: one stateful tracker
    must remain pinned to one source/peer for ``person_id`` continuity.
    """

    project_root: Path
    default_pipeline_config: Path
    classroom_pipeline_config: Path
    gender_model_path: str | None = None
    max_live_sessions: int = 1
    session_ttl_seconds: float = 600.0
    live_cadence_seconds: float = 0.15
    profiling_window: int = 120
    max_video_bytes: int = 64 * 1024 * 1024
    max_video_seconds: float = 60.0
    max_video_frames: int = 1_800

    @classmethod
    def from_environment(cls) -> "ApiSettings":
        root = _project_root()
        return cls(
            project_root=root,
            default_pipeline_config=_path_from_environment(
                "PIPELINE_CONFIG", root / "configs" / "pipeline-live.yaml", root
            ),
            classroom_pipeline_config=_path_from_environment(
                "CLASSROOM_PIPELINE_CONFIG", root / "configs" / "pipeline-classroom-template.yaml", root
            ),
            gender_model_path=os.getenv("GENDER_MODEL_PATH") or None,
            max_live_sessions=_positive_int_from_environment("API_MAX_LIVE_SESSIONS", 1),
            session_ttl_seconds=_nonnegative_float_from_environment("API_SESSION_TTL_SECONDS", 600.0),
            live_cadence_seconds=max(
                0.01,
                _nonnegative_float_from_environment("LIVE_STREAM_EVERY_SECONDS", 0.15),
            ),
            profiling_window=_positive_int_from_environment("API_PROFILING_WINDOW", 120),
            max_video_bytes=_positive_int_from_environment("API_MAX_VIDEO_BYTES", 64 * 1024 * 1024),
            max_video_seconds=max(
                1.0,
                _nonnegative_float_from_environment("API_MAX_VIDEO_SECONDS", 60.0),
            ),
            max_video_frames=_positive_int_from_environment("API_MAX_VIDEO_FRAMES", 1_800),
        )

    @property
    def modes(self) -> dict[str, Path]:
        return {
            "default": self.default_pipeline_config,
            "classroom_demo": self.classroom_pipeline_config,
        }

    def readiness(self) -> dict[str, Any]:
        """Validate profiles and all required local model assets without warming them."""

        assets, asset_manifest_error = _production_asset_status(
            self.project_root,
            gender_model_path=self.gender_model_path,
        )
        assets_ready = asset_manifest_error is None and all(asset["present"] for asset in assets)
        missing_assets = [asset for asset in assets if not asset["present"]]
        modes = {}
        for name, path in self.modes.items():
            configured = path.is_file()
            modes[name] = {
                "config_path": str(path),
                "configured": configured,
                "model_assets": assets,
                "assets_ready": assets_ready,
                "ready": configured and assets_ready,
            }
        return {
            "ready": modes["default"]["ready"],
            "model_initialization": "on_session_create",
            "modes": modes,
            "production_asset_manifest": str(self.project_root / _PRODUCTION_ASSET_MANIFEST),
            "missing_model_assets": missing_assets,
            "asset_bootstrap_command": _PRODUCTION_ASSET_BOOTSTRAP_COMMAND,
            **({"asset_manifest_error": asset_manifest_error} if asset_manifest_error else {}),
        }
