from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch

from src.analytics.geometry import scale_point, scale_polygon
from src.inference.attribute_router import AttributeRoute
from src.inference.body_gender_batch import BodyGenderCandidate
from src.inference.gender_batch import GenderCandidate
from src.inference.runtime import BBox, ModelRuntime
from src.inference.stream_state import CrowdStreamState, validate_tracker_state_retention
from src.tracking.tracker_config import (
    TrackerProfile,
    load_tracker_profile,
)


def _merge_yaml(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge one YAML configuration over a complete base mapping."""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_yaml(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(path: Path, _parents: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a profile, optionally overlaying it on a sibling ``extends`` YAML.

    Inheritance keeps related deployment configurations concise without
    duplicating a complete base profile.
    """

    resolved_path = path.resolve()
    if resolved_path in _parents:
        chain = " -> ".join(str(item) for item in (*_parents, resolved_path))
        raise ValueError(f"Cyclic YAML profile inheritance: {chain}")
    import yaml

    with resolved_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Pipeline config must contain a YAML mapping: {resolved_path}")
    values = dict(loaded)
    parent_value = values.pop("extends", None)
    if parent_value is None:
        return values
    if not isinstance(parent_value, str) or not parent_value.strip():
        raise ValueError(f"{resolved_path}: 'extends' must be a non-empty relative or absolute YAML path.")
    parent_path = Path(parent_value)
    if not parent_path.is_absolute():
        parent_path = resolved_path.parent / parent_path
    return _merge_yaml(_load_yaml(parent_path, (*_parents, resolved_path)), values)


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


class CrowdGenderPipeline:
    """One stateful video stream backed by a model runtime and isolated analytics state.

    The public API remains ``process_frame`` and ``reset``. ``ModelRuntime`` intentionally
    remains exclusive to this pipeline for now because Ultralytics stores tracker state inside
    ``YOLO.track``. A future detector/tracker adapter can safely share detector weights.
    """

    def __init__(
        self,
        # The public API follows the selected production profile.
        config_path: str = "configs/pipeline-live.yaml",
        model_path: str | None = None,
        device: str | None = None,
        runtime: ModelRuntime | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {self.config_path}")
        self.project_root = self.config_path.parent.parent
        self.config = _load_yaml(self.config_path)
        self.tracker_profile = load_tracker_profile(
            _resolve_path(self.config["tracker"]["config_path"], self.project_root)
        )
        self.stream_state = CrowdStreamState(self.config)
        validate_tracker_state_retention(
            self.stream_state.max_missing_frames,
            int(self.tracker_profile.values["track_buffer"]),
            int(self.tracker_profile.values["occ_reappear_window"]),
        )
        self.tsm = self.stream_state.tsm
        self.people_counter = self.stream_state.people_counter
        self.line_counter = self.stream_state.line_counter
        self.zone_manager = self.stream_state.zone_manager

        if runtime is None:
            runtime_device = torch.device(
                device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            )
            gender_config = self.config["gender_classifier"]
            router_mode = str(self.config.get("attribute_router", {}).get("mode", "face_first")).strip().lower()
            gender_model_path = (
                _resolve_path(model_path or gender_config["model_path"], self.project_root)
                if router_mode != "body_only"
                else None
            )
            if gender_model_path is not None and not gender_model_path.exists():
                raise FileNotFoundError(
                    f"Gender checkpoint not found: {gender_model_path}. "
                    "Provision the configured local asset with "
                    "`python tools/prepare_production_assets.py` before starting the pipeline."
                )
            self.runtime = ModelRuntime(
                pipeline_config=self.config,
                project_root=self.project_root,
                gender_model_path=gender_model_path,
                device=runtime_device,
                tracker_profile=self.tracker_profile,
            )
        else:
            if runtime.tracker_profile.path.resolve() != self.tracker_profile.path.resolve():
                raise ValueError("An injected ModelRuntime must use the same tracker profile as this pipeline.")
            self.runtime = runtime
        self.device = self.runtime.device
        self._last_gender_batch_size = 0
        self._last_body_gender_batch_size = 0
        self._last_active_track_ids: tuple[int, ...] = ()
        self._last_active_person_ids: tuple[int, ...] = ()
        runtime_config = self.config.get("runtime", {})
        self._latency_history_ms: deque[float] = deque(
            maxlen=max(1, int(runtime_config.get("profiling_window", 120)))
        )
        self._last_timing_ms: dict[str, float] = {}

    def reset(self) -> None:
        """Clear per-stream analytics and reset the runtime's persistent tracker."""
        self.runtime.reset_tracker()
        self.stream_state.reset()
        self._last_gender_batch_size = 0
        self._last_body_gender_batch_size = 0
        self._last_active_track_ids = ()
        self._last_active_person_ids = ()
        self._latency_history_ms.clear()
        self._last_timing_ms = {}

    def apply_session_layout(self, session_layout: dict[str, Any]) -> dict[str, object]:
        """Apply a lightweight classroom seating change without resetting tracking.

        This is the intended backend hook for a future room-setup UI.  It
        resets only seat-assignment hysteresis; the loaded models, FastTracker
        state and existing crowd counters remain intact.
        """

        return self.stream_state.apply_session_layout(session_layout)

    def close(self) -> None:
        """Release this stream's tracker and analytics state."""

        self.reset()

    def warmup(self) -> None:
        """Prepare model execution without retaining a synthetic track or analytics event."""
        runtime_config = self.config.get("runtime", {})
        if not bool(runtime_config.get("warmup_models", True)):
            return
        reference_width, reference_height = self.stream_state.reference_size
        frame = self._prepare_frame(np.zeros((reference_height, reference_width, 3), dtype=np.uint8))
        self.runtime.warmup(frame.shape[:2])
        self.reset()

    @property
    def last_active_track_ids(self) -> tuple[int, ...]:
        """Tracker-local IDs emitted on the last processed frame."""
        return self._last_active_track_ids

    @property
    def last_active_person_ids(self) -> tuple[int, ...]:
        """Session-local person IDs resolved above the current tracker IDs."""

        return self._last_active_person_ids

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        runtime_config = self.config.get("runtime", {})
        max_frame_width = int(runtime_config.get("max_frame_width", 0) or 0)
        if max_frame_width <= 0 or frame.shape[1] <= max_frame_width:
            return frame
        scale = max_frame_width / frame.shape[1]
        resized_height = max(1, int(round(frame.shape[0] * scale)))
        return cv2.resize(frame, (max_frame_width, resized_height), interpolation=cv2.INTER_AREA)

    def _collect_gender_candidates(
        self, frame: np.ndarray, active_tracks: dict[int, BBox]
    ) -> tuple[list[GenderCandidate], list[BodyGenderCandidate]]:
        """Route each track before YuNet, preserving face-first evidence priority."""
        face_candidates: list[GenderCandidate] = []
        frame_shape = frame.shape[:2]
        routes = self.stream_state.resolve_attribute_routes(active_tracks, frame_shape)

        # Far/body-only tracks are selected before legacy fallbacks. This keeps a
        # bounded body batch even when the camera sees many people.
        body_candidates: list[BodyGenderCandidate] = []
        body_track_ids = self.stream_state.scheduled_body_track_ids(
            [track_id for track_id, route in routes.items() if route == AttributeRoute.BODY],
            direct=True,
        )
        for track_id in body_track_ids:
            self.stream_state.record_body_attempt(track_id)
            candidate = self.runtime.extract_body_gender_candidate(frame, track_id, active_tracks[track_id])
            if candidate is not None:
                self.stream_state.record_body_candidate(track_id)
                body_candidates.append(candidate)

        # Preserve ordinary face-first fallback scheduling for tracks that have
        # already accumulated face misses or unresolved face observations.
        remaining_body_capacity = self.stream_state.max_body_tracks_per_frame - len(body_track_ids)
        fallback_track_ids = self.stream_state.scheduled_body_track_ids(
            [
                track_id
                for track_id, route in routes.items()
                # A close-up box clipped at the image edge is routed FACE, but
                # it still needs the configured body fallback after repeated
                # low-confidence face predictions. ``scheduled_body_track_ids``
                # applies the miss/unknown-prediction guard, so including FACE
                # here does not cause body inference on the first face attempt.
                if route in {AttributeRoute.FACE, AttributeRoute.FACE_THEN_BODY}
                and track_id not in set(body_track_ids)
            ],
            limit=max(0, remaining_body_capacity),
        )
        for track_id in fallback_track_ids:
            self.stream_state.record_body_attempt(track_id)
            candidate = self.runtime.extract_body_gender_candidate(frame, track_id, active_tracks[track_id])
            if candidate is not None:
                self.stream_state.record_body_candidate(track_id)
                body_candidates.append(candidate)

        body_track_id_set = set(body_track_ids) | set(fallback_track_ids)
        face_track_ids = self.stream_state.scheduled_face_track_ids(
            [track_id for track_id in active_tracks if track_id not in body_track_id_set],
            routes,
        )
        for track_id in face_track_ids:
            self.stream_state.record_face_attempt(track_id)
            detailed_extractor = getattr(self.runtime, "extract_gender_candidate_result", None)
            if callable(detailed_extractor):
                extraction = detailed_extractor(frame, track_id, active_tracks[track_id])
                candidate = extraction.candidate
                self.stream_state.record_face_extraction(
                    track_id,
                    extraction.status,
                    extraction.quality,
                )
            else:
                # Keep injected runtimes and older extensions compatible with
                # the original crop-only contract.
                candidate = self.runtime.extract_gender_candidate(frame, track_id, active_tracks[track_id])
                if candidate is not None:
                    self.stream_state.record_face_candidate(track_id)
                else:
                    self.stream_state.record_face_miss(track_id)
            if candidate is not None:
                face_candidates.append(candidate)
            else:
                # Adaptive medium-distance tracks may use their body crop on the
                # same frame as a YuNet miss. Face-first compatibility mode keeps
                # its established delayed fallback behavior.
                if (
                    self.stream_state.router_config.mode == "adaptive"
                    and routes[track_id] in {AttributeRoute.FACE, AttributeRoute.FACE_THEN_BODY}
                    and len(body_track_id_set) < self.stream_state.max_body_tracks_per_frame
                    and self.tsm.should_attempt_body_direct(
                        track_id, self.stream_state.frame_index, self.stream_state.body_retry_interval
                    )
                ):
                    self.stream_state.record_body_attempt(track_id)
                    body_candidate = self.runtime.extract_body_gender_candidate(frame, track_id, active_tracks[track_id])
                    body_track_id_set.add(track_id)
                    if body_candidate is not None:
                        self.stream_state.record_body_candidate(track_id)
                        body_candidates.append(body_candidate)
        return face_candidates, body_candidates

    def _apply_gender_batch(self, candidates: list[GenderCandidate]) -> None:
        self._last_gender_batch_size = len(candidates)
        for evidence in self.runtime.classify_gender_batch(candidates):
            source_before = self.tsm.get_gender_source(evidence.track_id)
            evidence_updated = self.tsm.update_gender(
                evidence.track_id,
                evidence.logits,
                face_detection_confidence=evidence.face_detection_confidence,
                gender_confidence=evidence.gender_confidence,
                face_width=evidence.face_width,
                frame_index=self.stream_state.frame_index,
            )
            source_after = self.tsm.get_gender_source(evidence.track_id)
            self.stream_state.record_face_inference(
                evidence.track_id,
                evidence.gender_confidence,
                evidence_updated,
                resolution_transition=source_before != "face" and source_after == "face",
                body_override=source_before == "body" and source_after == "face",
            )

    def _apply_body_gender_batch(self, candidates: list[BodyGenderCandidate]) -> None:
        self._last_body_gender_batch_size = len(candidates)
        for evidence in self.runtime.classify_body_gender_batch(candidates):
            source_before = self.tsm.get_gender_source(evidence.track_id)
            evidence_updated = self.tsm.update_body_gender(
                evidence.track_id,
                evidence.logits,
                gender_confidence=evidence.gender_confidence,
                person_height=evidence.person_height,
                frame_index=self.stream_state.frame_index,
            )
            self.stream_state.record_body_inference(
                evidence.track_id,
                evidence.gender_confidence,
                evidence_updated,
                resolution_transition=source_before == "unknown" and self.tsm.get_gender_source(evidence.track_id) == "body",
            )

    def _record_timing(self, timings: dict[str, float]) -> None:
        total = timings["total"]
        self._latency_history_ms.append(total)
        samples = np.asarray(self._latency_history_ms, dtype=np.float32)
        timings["p50"] = float(np.percentile(samples, 50))
        timings["p95"] = float(np.percentile(samples, 95))
        self._last_timing_ms = {name: round(value, 2) for name, value in timings.items()}

    def process_frame(self, frame: np.ndarray, timestamp_seconds: float | None = None) -> tuple[np.ndarray, dict]:
        """Analyse one frame, optionally using source/camera time for trajectory metrics."""
        if frame is None or frame.ndim != 3:
            raise ValueError("frame must be a BGR image with shape (height, width, channels)")
        started = perf_counter()
        frame = self._prepare_frame(frame)
        after_prepare = perf_counter()
        self.stream_state.frame_index += 1
        active_tracks = self.runtime.active_tracks(frame)
        self._last_active_track_ids = tuple(sorted(active_tracks))
        person_ids = self.stream_state.resolve_person_ids(
            active_tracks,
            (frame.shape[1], frame.shape[0]),
        )
        self._last_active_person_ids = tuple(person_ids[track_id] for track_id in sorted(person_ids))
        after_tracking = perf_counter()
        for track_id, bbox in active_tracks.items():
            self.tsm.update_track(track_id, bbox, self.stream_state.frame_index)
        frame_size = (frame.shape[1], frame.shape[0])
        event_timestamp = self.stream_state.resolve_timestamp(timestamp_seconds)
        self.stream_state.trajectory_engine.update(
            active_tracks,
            frame_size,
            self.stream_state.frame_index,
            event_timestamp,
        )

        face_candidates, body_candidates = self._collect_gender_candidates(frame, active_tracks)
        after_face_detection = perf_counter()
        self._apply_gender_batch(face_candidates)
        after_face_classification = perf_counter()
        self._apply_body_gender_batch(body_candidates)
        after_body_classification = perf_counter()

        # The canonical SpaceConfig can intentionally omit a counting line.
        # Keep the legacy LineCrossingCounter dormant in that case rather than
        # inventing a geometric IN/OUT measurement.
        configured_line = self.stream_state.primary_counting_line
        line_start: tuple[float, float] | None = None
        line_end: tuple[float, float] | None = None
        if configured_line is not None:
            line_start = scale_point(configured_line.p1, frame_size, self.stream_state.reference_size)
            line_end = scale_point(configured_line.p2, frame_size, self.stream_state.reference_size)
        active_ids = list(active_tracks)
        self.people_counter.update(active_ids)
        if line_start is not None and line_end is not None:
            self.line_counter.update(active_ids, line_start, line_end)
        confirmed_tracks = {
            track_id: bbox
            for track_id, bbox in active_tracks.items()
            if self.tsm.is_stable(track_id, self.config["analytics"]["stable_track_frames"])
        }
        spatial_config = self.config["analytics"].get("spatial", {})
        if bool(spatial_config.get("enabled", True)):
            spatial = self.stream_state.spatial_engine.update(confirmed_tracks, frame_size, event_timestamp)
        else:
            spatial = self.stream_state.spatial_engine.get_statistics()
        classroom_statistics = self.stream_state.update_classroom(
            confirmed_tracks,
            frame_size,
            event_timestamp,
        )
        finalized = self.stream_state.finalize_stale_tracks(event_timestamp)
        # ``update`` already returns the current snapshots. Finalization only
        # changes those snapshots when a stale track is actually consumed;
        # avoid rebuilding heatmap/zone/classroom payloads on every frame.
        if finalized:
            self.people_counter.consume_finalized(finalized)
            # Finalisation transfers zone dwell into compact aggregates. Read
            # the engines again so this frame includes completed dwell and a
            # released seat assignment for a finalized track.
            spatial = self.stream_state.spatial_engine.get_statistics()
            classroom_statistics = self.stream_state.classroom_statistics()
        crowd_statistics = self.people_counter.get_statistics(active_ids)
        crossing_statistics = self.line_counter.get_counts()
        space_statistics, history_statistics = self.stream_state.update_space_and_history(
            event_timestamp,
            crowd_statistics,
            crossing_statistics,
            spatial,
        )
        overlay_tracks = []
        for track_id, bbox in sorted(active_tracks.items()):
            gender, confidence = self.tsm.get_gender(track_id)
            source = self.tsm.get_gender_source(track_id)
            person_id = person_ids.get(track_id)
            person_label = self.stream_state.person_identity.display_label(person_id)
            overlay_tracks.append(
                {
                    "track_id": int(track_id),
                    "person_id": int(person_id) if person_id is not None else None,
                    "label": f"{person_label} | T{track_id} | {source}: {gender}",
                    "bbox": [int(value) for value in bbox],
                    "gender": str(gender),
                    "source": str(source),
                    "confidence": round(float(confidence), 3),
                }
            )
        stats = {
            "identity": self.stream_state.person_identity_statistics(),
            "attributes": self.stream_state.attribute_branch.summarize(active_ids),
            "crowd": crowd_statistics,
            "crossing": crossing_statistics,
            "trajectory": self.stream_state.trajectory_engine.get_summary(active_ids),
            "spatial": spatial,
            "space": space_statistics,
            "history": history_statistics,
            "classroom": classroom_statistics,
            # The static CyberHUD client draws this lightweight metadata over
            # its local <video>. Coordinates belong to the processed frame,
            # not to the browser viewport, so the client can account for
            # object-fit/crop and portrait camera dimensions explicitly.
            "overlay": {
                "coordinate_space": "processed_frame",
                "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
                "tracks": overlay_tracks,
            },
            # Compatibility field retained for callers that used the original ZoneManager response.
            "zones": spatial["zones"],
            "runtime": {
                "tracker_type": self.tracker_profile.tracker_type,
                "detector": (
                    self.runtime.detector_statistics()
                    if callable(getattr(self.runtime, "detector_statistics", None))
                    else {}
                ),
                "gender_batch_size": self._last_gender_batch_size,
                "body_gender_batch_size": self._last_body_gender_batch_size,
                "body_classifier_timing": (
                    self.runtime.body_classifier_statistics()
                    if callable(getattr(self.runtime, "body_classifier_statistics", None))
                    else {}
                ),
                "attribute_router": self.stream_state.attribute_router_statistics(),
                "active_track_states": len(self.tsm.track_states),
                "finalized_this_frame": len(finalized),
                "timing_ms": self._last_timing_ms,
                "processing_fps": (
                    round(1_000 / self._last_timing_ms["total"], 2)
                    if self._last_timing_ms.get("total", 0.0) > 0.0
                    else 0.0
                ),
            },
        }
        after_analytics = perf_counter()
        output = self._draw_results(frame, active_tracks, person_ids, stats, line_start, line_end)
        after_drawing = perf_counter()
        timing_ms = {
            "prepare": (after_prepare - started) * 1_000,
            "tracking": (after_tracking - after_prepare) * 1_000,
            "face_detection": (after_face_detection - after_tracking) * 1_000,
            "face_classifier": (after_face_classification - after_face_detection) * 1_000,
            "body_classifier": (after_body_classification - after_face_classification) * 1_000,
            "analytics": (after_analytics - after_body_classification) * 1_000,
            "drawing": (after_drawing - after_analytics) * 1_000,
            "total": (after_drawing - started) * 1_000,
        }
        self._record_timing(timing_ms)
        stats["runtime"]["timing_ms"] = self._last_timing_ms
        stats["runtime"]["processing_fps"] = round(1_000 / timing_ms["total"], 2) if timing_ms["total"] > 0 else 0.0
        return output, stats

    def _draw_results(
        self,
        frame: np.ndarray,
        active_tracks: dict[int, BBox],
        person_ids: dict[int, int],
        stats: dict,
        line_start: tuple[float, float] | None,
        line_end: tuple[float, float] | None,
    ) -> np.ndarray:
        output = frame.copy()
        frame_size = (frame.shape[1], frame.shape[0])
        visualization = self.config.get("visualization", {})
        spatial = stats.get("spatial", {})
        heatmap = spatial.get("heatmap", {})
        if bool(visualization.get("show_heatmap", False)) and heatmap.get("total_weight", 0.0) > 0.0:
            overlay = self.stream_state.spatial_engine.heatmap_image(frame_size)
            alpha = float(visualization.get("heatmap_alpha", 0.25))
            output = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0.0)
        if (
            bool(self.config["analytics"].get("spatial", {}).get("enabled", True))
            and bool(self.config["analytics"].get("enable_zones", True))
            and bool(visualization.get("show_zones", True))
        ):
            for name, polygon in self.zone_manager.scaled_polygons(frame_size, self.stream_state.reference_size).items():
                points = np.asarray(polygon, dtype=np.int32)
                cv2.polylines(output, [points], True, (255, 255, 0), 2)
                count = spatial["zones"][name]["current_count"]
                cv2.putText(
                    output,
                    f"{name}: {count}",
                    tuple(points[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )
        classroom = stats.get("classroom", {})
        classroom_config = self.stream_state.classroom_config
        if classroom_config is not None and bool(visualization.get("show_classroom_regions", False)):
            reference_size = classroom_config.room_profile.reference_resolution or self.stream_state.reference_size
            room_boundary = classroom_config.room_profile.room_boundary
            if room_boundary is not None:
                boundary = np.asarray(scale_polygon(room_boundary, frame_size, reference_size), dtype=np.int32)
                cv2.polylines(output, [boundary], True, (220, 120, 255), 1)
            door_roi = classroom_config.room_profile.door_roi
            if door_roi is not None:
                door = np.asarray(scale_polygon(door_roi, frame_size, reference_size), dtype=np.int32)
                cv2.polylines(output, [door], True, (0, 165, 255), 2)
                cv2.putText(output, "door", tuple(door[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1)
            for aisle in classroom.get("aisles", {}).get("items", []):
                polygon = aisle.get("polygon")
                if not polygon:
                    continue
                points = np.asarray(polygon, dtype=np.int32)
                cv2.polylines(output, [points], True, (255, 160, 0), 1)

        if classroom_config is not None and bool(visualization.get("show_seats", False)):
            seat_statistics = classroom.get("seats", {})
            status_by_seat = {
                str(item.get("seat_id")): str(item.get("status", "vacant"))
                for item in seat_statistics.get("seats", [])
                if isinstance(item, dict) and item.get("seat_id") is not None
            }
            reference_size = classroom_config.room_profile.reference_resolution or self.stream_state.reference_size
            colors = {
                "occupied": (60, 210, 60),
                "pending": (0, 180, 255),
                "uncertain": (0, 180, 255),
                "disabled": (110, 110, 110),
                "vacant": (255, 210, 0),
            }
            for seat in classroom_config.seat_definitions:
                if seat.polygon is None:
                    continue
                points = np.asarray(scale_polygon(seat.polygon, frame_size, reference_size), dtype=np.int32)
                status = status_by_seat.get(seat.seat_id, "disabled" if not seat.enabled else "vacant")
                color = colors.get(status, colors["vacant"])
                cv2.polylines(output, [points], True, color, 1)
                if bool(visualization.get("show_seat_labels", False)):
                    cv2.putText(output, seat.seat_id, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
        if line_start is not None and line_end is not None and bool(visualization.get("show_counting_line", False)):
            cv2.line(output, tuple(map(int, line_start)), tuple(map(int, line_end)), (0, 0, 255), 2)

        for track_id, (x1, y1, x2, y2) in active_tracks.items():
            gender, confidence = self.tsm.get_gender(track_id)
            source = self.tsm.get_gender_source(track_id)
            color = {"female": (203, 192, 255), "male": (255, 128, 0), "unknown": (128, 128, 128)}[gender]
            
            # Show/hide bounding boxes according to show_boxes flag
            if bool(visualization.get("show_boxes", True)):
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Show/hide IDs and Attributes according to show_ids & show_attributes flags
            show_ids = bool(visualization.get("show_ids", True))
            show_attrs = bool(visualization.get("show_attributes", True))
            person_label = self.stream_state.person_identity.display_label(person_ids.get(track_id))
            
            label_parts = []
            if show_ids:
                label_parts.append(f"{person_label} | T{track_id}")
            if show_attrs:
                label_parts.append(f"{source}: {gender}" + (f" {confidence:.2f}" if gender != "unknown" else ""))
            
            if label_parts:
                label = " | ".join(label_parts)
                # Draw black background tag for text readability
                (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                tag_y = max(18, y1 - 7)
                cv2.rectangle(output, (x1, tag_y - txt_h - 4), (x1 + txt_w + 6, tag_y + 4), (15, 23, 42), -1)
                cv2.putText(output, label, (x1 + 3, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if bool(visualization.get("show_motion_labels", False)) or bool(visualization.get("show_motion", False)):
                motion = stats.get("trajectory", {}).get("active_tracks", {}).get(str(track_id), {})
                if motion:
                    motion_label = (
                        f"{motion['direction']} {motion['speed_reference_px_per_second']:.0f}rpx/s "
                        f"| dwell {motion['dwell_seconds']:.1f}s"
                    )
                    cv2.putText(output, motion_label, (x1, min(output.shape[0] - 6, y2 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            trajectory = self.tsm.get_trajectory(track_id)[-20:]
            if (bool(visualization.get("show_trajectories", False)) or bool(visualization.get("show_motion", False))) and len(trajectory) > 1:
                cv2.polylines(output, [np.asarray(trajectory, dtype=np.int32)], False, color, 1)

        if not bool(visualization.get("show_hud", True)):
            return output
        crowd, crossing, trajectory = stats["crowd"], stats["crossing"], stats["trajectory"]
        identity = stats.get("identity", {})
        runtime = stats["runtime"]
        timing = runtime.get("timing_ms", {})
        density = spatial.get("density", {})
        lines = [
            "CROWD ANALYTICS",
            f"Tracker: {runtime['tracker_type']}",
            f"Active tracks: {crowd['current_count']} | Confirmed: {crowd['confirmed_active_count']}",
            f"Session persons: {identity.get('unique_person_count', 0)} | local tracks: {crowd['unique_track_count']}",
            f"Female {crowd['gender_counts']['female']} | Male {crowd['gender_counts']['male']} | Unknown {crowd['gender_counts']['unknown']}",
            f"Coverage: {crowd['gender_coverage'] * 100:.0f}% | Face batch: {runtime['gender_batch_size']} | Body: {runtime['body_gender_batch_size']}",
            f"Motion: {trajectory['moving_count']} moving | {trajectory['stationary_count']} stationary | mean {trajectory['mean_speed_reference_px_per_second']:.0f} rpx/s",
            f"Screen density: {density.get('density_per_100k_reference_pixels', 0.0):.2f} confirmed / 100k px",
            f"Line crossings  IN {crossing['in']} | OUT {crossing['out']}",
            f"Processing: {runtime.get('processing_fps', 0.0):.1f} FPS | p95: {timing.get('p95', 0.0):.0f} ms",
        ]
        hud_width = min(output.shape[1] - 10, 560)
        hud_height = min(output.shape[0] - 10, 42 + len(lines) * 23)
        cv2.rectangle(output, (10, 10), (hud_width, hud_height), (0, 0, 0), -1)
        for index, text in enumerate(lines):
            cv2.putText(output, text, (20, 32 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return output
