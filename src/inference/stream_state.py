"""Per-stream analytics state. This module intentionally owns no neural model objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from src.analytics.attributes import AttributeBranch
from src.analytics.classroom import ClassroomAnalytics
from src.analytics.classroom_config import ClassroomConfig, SessionLayout, parse_classroom_config
from src.analytics.history import HistoryEngine
from src.analytics.line_crossing import LineCrossingCounter
from src.analytics.people_counter import PeopleCounter
from src.analytics.space import SpaceAnalytics
from src.analytics.space_config import CountingLineConfig, parse_space_config
from src.analytics.spatial import SpatialEngine
from src.analytics.trajectory import TrajectoryEngine
from src.analytics.zones import ZoneManager
from src.inference.attribute_router import AttributeRoute, AttributeRouter, AttributeRouterConfig
from src.inference.gender_batch import FaceQuality
from src.tracking.person_identity import PersonIdentityResolver
from src.tracking.track_state import FinalizedTrack, TrackStateManager


BBox = tuple[int, int, int, int]


def _positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"'{key}' must be a positive integer.")
    return value


def _nonnegative_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"'{key}' must be a non-negative integer.")
    return value


def _probability(config: dict[str, Any], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{key}' must be a numeric probability.")
    return float(value)


def _positive_number(config: dict[str, Any], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise ValueError(f"'{key}' must be a positive number.")
    return float(value)


def validate_tracker_state_retention(
    max_missing_frames: int, track_buffer: int, extended_recovery_frames: int | None = None
) -> None:
    """Do not prune evidence before the selected tracker can recover an ID.

    FastTracker can retain a recently occluded *lost* track through its
    ``occ_reappear_window`` after the normal tracking buffer would have
    expired. The analytics state must outlive whichever recovery window is
    longer or a recovered track loses its accumulated gender/trajectory state.
    """

    required_recovery_frames = max(track_buffer, extended_recovery_frames or 0)
    if max_missing_frames < required_recovery_frames:
        raise ValueError(
            "analytics.max_missing_frames must be at least the selected tracker recovery window "
            f"({required_recovery_frames} frames) so a recoverable ID keeps its evidence."
        )


class CrowdStreamState:
    """Tracker-adjacent state belonging to exactly one video/webcam stream."""

    def __init__(self, pipeline_config: dict[str, Any]) -> None:
        analytics = pipeline_config["analytics"]
        gender = pipeline_config["gender_classifier"]
        body = pipeline_config.get("body_gender_classifier", {})
        router_values = dict(pipeline_config.get("attribute_router", {}) or {})
        self._validate_settings(analytics, gender, body)
        self.reference_size = tuple(pipeline_config["reference_resolution"])
        # This is deliberately independent from ``TrackStateManager``.  The
        # latter is keyed by a tracker-local ID so it can retain model evidence
        # through the selected tracker's own recovery window; the resolver
        # exposes a separate, conservative person-level label above it.
        self.person_identity = PersonIdentityResolver(
            self.reference_size,
            pipeline_config.get("person_identity"),
        )
        self.space_config = parse_space_config(pipeline_config)
        self.classroom_config, classroom_options = self._parse_classroom_settings(pipeline_config)
        self._classroom_engine_options = classroom_options
        self.classroom_analytics = ClassroomAnalytics(self.classroom_config, **classroom_options)
        self._classroom_layout_revision = 0
        self._validate_classroom_space_consistency()
        legacy_space = self.space_config.to_legacy_analytics_mapping()
        self.primary_counting_line = self._resolve_primary_counting_line()
        trajectory_config = analytics.get("trajectory", {})
        spatial_config = analytics.get("spatial", {})
        history_config = analytics.get("history", {})
        self.max_missing_frames = int(analytics["max_missing_frames"])
        self.max_face_tracks_per_frame = int(gender["max_face_tracks_per_frame"])
        self.face_retry_interval = int(gender["face_retry_interval"])
        self.body_enabled = bool(body.get("enabled", False))
        self.max_body_tracks_per_frame = int(body.get("max_body_tracks_per_frame", 0)) if self.body_enabled else 0
        self.body_retry_interval = int(body.get("body_retry_interval", 1)) if self.body_enabled else 1
        self.body_face_misses_before_attempt = (
            int(body.get("face_misses_before_body", 1)) if self.body_enabled else 1
        )
        self.body_face_unknown_predictions_before_attempt = (
            int(body.get("face_unknown_predictions_before_body", self.body_face_misses_before_attempt))
            if self.body_enabled
            else 1
        )
        # A face aggregate only becomes a body-fallback signal once it has had
        # enough observations to be meaningful.  Keep this separate from the
        # number of consecutive unresolved aggregate decisions.
        self.body_face_fallback_min_observations = (
            int(body.get("face_fallback_min_observations", gender["maximum_observations"]))
            if self.body_enabled
            else int(gender["minimum_observations"])
        )
        body_minimum_width = int(body.get("minimum_person_width", 1)) if self.body_enabled else 1
        body_minimum_height = int(body.get("minimum_person_height", 1)) if self.body_enabled else 1
        self.router_config = AttributeRouterConfig.from_mapping(
            router_values,
            body_minimum_width=body_minimum_width,
            body_minimum_height=body_minimum_height,
        )
        if self.router_config.mode == "body_only" and not self.body_enabled:
            raise ValueError("attribute_router.mode=body_only requires body_gender_classifier.enabled=true.")
        self.attribute_router = AttributeRouter(
            self.router_config,
            body_minimum_width=body_minimum_width,
            body_minimum_height=body_minimum_height,
        )
        self.route_refresh_interval = _positive_int(
            {"route_refresh_interval": router_values.get("route_refresh_interval", 10)}, "route_refresh_interval"
        )
        self.route_stability_frames = _positive_int(
            {"route_stability_frames": router_values.get("route_stability_frames", 1)}, "route_stability_frames"
        )
        self.minimum_track_age_for_attributes = _positive_int(
            {"minimum_track_age_frames": router_values.get("minimum_track_age_frames", 1)},
            "minimum_track_age_frames",
        )
        self.face_failure_limit = _positive_int(
            {"face_failure_limit": router_values.get("face_failure_limit", 2)}, "face_failure_limit"
        )
        self.face_cooldown_frames = _positive_int(
            {"face_cooldown_frames": router_values.get("face_cooldown_frames", 30)}, "face_cooldown_frames"
        )

        self.tsm = TrackStateManager(
            confidence_threshold=float(gender["confidence_threshold"]),
            minimum_observations=int(gender["minimum_observations"]),
            maximum_observations=int(gender["maximum_observations"]),
            stable_confidence=float(gender["stable_confidence"]),
            success_interval=int(gender["gender_success_interval"]),
            refresh_interval=int(gender["refresh_interval"]),
            trajectory_length=int(analytics["trajectory_length"]),
            finalized_history_size=int(analytics["finalized_history_size"]),
            body_confidence_threshold=float(body.get("confidence_threshold", gender["confidence_threshold"])),
            body_minimum_observations=int(body.get("minimum_observations", gender["minimum_observations"])),
            body_maximum_observations=int(body.get("maximum_observations", gender["maximum_observations"])),
            body_stable_confidence=float(body.get("stable_confidence", gender["stable_confidence"])),
            body_success_interval=int(body.get("gender_success_interval", gender["gender_success_interval"])),
            body_refresh_interval=int(body.get("refresh_interval", gender["refresh_interval"])),
            body_unknown_retry_interval=int(body.get("unknown_retry_interval", body.get("refresh_interval", gender["refresh_interval"]))),
            body_face_refresh_interval=int(body.get("face_upgrade_interval", body.get("refresh_interval", gender["refresh_interval"]))),
        )
        self.people_counter = PeopleCounter(self.tsm, int(analytics["stable_track_frames"]))
        self.line_counter = LineCrossingCounter(self.tsm, int(analytics["stable_track_frames"]))
        self.attribute_branch = AttributeBranch(self.tsm)
        self.assumed_fps = float(trajectory_config.get("assumed_fps", 25.0))
        self.trajectory_engine = TrajectoryEngine(
            self.reference_size,
            assumed_fps=self.assumed_fps,
            history_length=int(trajectory_config.get("history_length", analytics["trajectory_length"])),
            velocity_window_frames=int(trajectory_config.get("velocity_window_frames", 8)),
            stationary_speed_threshold=float(trajectory_config.get("stationary_speed_threshold", 8.0)),
            stationary_duration_seconds=float(trajectory_config.get("stationary_duration_seconds", 2.0)),
        )
        self.spatial_engine = SpatialEngine(
            legacy_space["zones"],
            self.reference_size,
            zone_history_length=int(spatial_config.get("zone_history_length", 300)),
            heatmap_grid_size=tuple(spatial_config.get("heatmap_grid_size", (16, 12))),
            heatmap_decay=float(spatial_config.get("heatmap_decay", 0.995)),
            density_area_scale=float(spatial_config.get("density_area_scale", 100_000.0)),
            visible_floor_area_m2=self.space_config.visible_floor_area_m2,
        )
        self.space_analytics = SpaceAnalytics(self.space_config)
        self.history_engine = HistoryEngine(
            snapshot_interval_seconds=float(history_config.get("snapshot_interval_seconds", 5.0)),
            max_snapshots=int(history_config.get("max_snapshots", 720)),
            flow_windows_seconds=tuple(history_config.get("flow_windows_seconds", (60.0, 300.0))),
            max_flow_events=int(history_config.get("max_flow_events", 2_000)),
        )
        # Kept as a compatibility alias for callers that draw/edit zone polygons directly.
        self.zone_manager: ZoneManager = self.spatial_engine.zone_manager
        self.frame_index = 0
        self._last_resolved_timestamp: float | None = None
        self._last_line_counts = {"in": 0, "out": 0}
        self._face_scheduler_cursor = 0
        self._body_scheduler_cursor = 0
        self._reset_router_telemetry()

    @staticmethod
    def _parse_classroom_settings(
        pipeline_config: Mapping[str, Any],
    ) -> tuple[ClassroomConfig | None, dict[str, float]]:
        """Read the optional classroom branch without changing legacy profiles.

        The active camera's static room profile and one session's seat layout
        live under a root ``classroom`` block.  A missing block (or an explicit
        ``enabled: false``) leaves the entire branch dormant, which keeps the
        current webcam profiles and tests behaviourally unchanged.
        """

        raw_classroom = pipeline_config.get("classroom")
        if raw_classroom is None:
            return None, {}
        if not isinstance(raw_classroom, Mapping):
            raise ValueError("classroom must be a YAML mapping when configured.")
        enabled = raw_classroom.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("classroom.enabled must be a boolean when configured.")
        if not enabled:
            return None, {}

        raw_seat_options = raw_classroom.get("seat_occupancy", {})
        if raw_seat_options is None:
            raw_seat_options = {}
        if not isinstance(raw_seat_options, Mapping):
            raise ValueError("classroom.seat_occupancy must be a YAML mapping when configured.")
        options: dict[str, float] = {}
        for key, default in (
            ("occupancy_confirm_seconds", 1.5),
            ("vacancy_grace_seconds", 2.0),
            ("aspect_ratio_tolerance", 0.02),
        ):
            value = raw_seat_options.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0:
                raise ValueError(f"classroom.seat_occupancy.{key} must be a non-negative number.")
            if key == "aspect_ratio_tolerance" and float(value) >= 1.0:
                raise ValueError("classroom.seat_occupancy.aspect_ratio_tolerance must be in [0, 1).")
            options[key] = float(value)
        return parse_classroom_config(pipeline_config), options

    def _validate_classroom_space_consistency(self) -> None:
        """Avoid publishing two incompatible physical calibrations for one camera."""

        if self.classroom_config is None:
            return
        room = self.classroom_config.room_profile
        space_area = self.space_config.visible_floor_area_m2
        if (
            space_area is not None
            and room.visible_floor_area_m2 is not None
            and abs(space_area - room.visible_floor_area_m2) > 1e-9
        ):
            raise ValueError(
                "space.visible_floor_area_m2 and classroom.room_profile.visible_floor_area_m2 "
                "must match when both are configured for the same camera."
            )
        if (
            self.space_config.capacity is not None
            and room.maximum_capacity is not None
            and self.space_config.capacity != room.maximum_capacity
        ):
            raise ValueError(
                "space.capacity and classroom.room_profile.maximum_capacity must match "
                "when both are configured for the same camera."
            )

    def _resolve_primary_counting_line(self) -> CountingLineConfig | None:
        """Use a classroom door line only when canonical space has none.

        ``LineCrossingCounter`` still has a one-line contract.  This small
        bridge lets a room profile declare its entrance without treating the
        door ROI as an overlapping density zone, while rejecting two different
        line definitions for one camera.
        """

        space_line = self.space_config.primary_counting_line
        classroom_line = (
            None if self.classroom_config is None else self.classroom_config.room_profile.counting_line
        )
        if classroom_line is None:
            return space_line
        classroom_reference = self.classroom_config.room_profile.reference_resolution
        if classroom_reference is None or classroom_reference == self.reference_size:
            p1, p2 = classroom_line.p1, classroom_line.p2
        else:
            source_width, source_height = classroom_reference
            target_width, target_height = self.reference_size
            p1 = (
                classroom_line.p1[0] * target_width / source_width,
                classroom_line.p1[1] * target_height / source_height,
            )
            p2 = (
                classroom_line.p2[0] * target_width / source_width,
                classroom_line.p2[1] * target_height / source_height,
            )
        converted = CountingLineConfig(
            name="classroom_entrance",
            p1=p1,
            p2=p2,
        )
        if space_line is None:
            return converted
        if space_line.p1 != converted.p1 or space_line.p2 != converted.p2:
            raise ValueError(
                "space.counting_line and classroom.room_profile.entrance.counting_line must match "
                "when both are configured."
            )
        return space_line

    @staticmethod
    def _validate_settings(analytics: dict[str, Any], gender: dict[str, Any], body: dict[str, Any]) -> None:
        for key in ("stable_track_frames", "trajectory_length", "max_missing_frames"):
            _positive_int(analytics, key)
        _nonnegative_int(analytics, "finalized_history_size")
        if analytics["max_missing_frames"] < 1:
            raise ValueError("'max_missing_frames' must be positive.")
        for key in (
            "minimum_observations",
            "maximum_observations",
            "face_retry_interval",
            "gender_success_interval",
            "refresh_interval",
            "max_face_tracks_per_frame",
        ):
            _positive_int(gender, key)
        if gender["maximum_observations"] < gender["minimum_observations"]:
            raise ValueError("'maximum_observations' must be at least 'minimum_observations'.")
        confidence_threshold = float(gender["confidence_threshold"])
        stable_confidence = float(gender["stable_confidence"])
        if not 0.5 < confidence_threshold < 1.0:
            raise ValueError("'confidence_threshold' must be in (0.5, 1.0).")
        if not confidence_threshold <= stable_confidence <= 1.0:
            raise ValueError("'stable_confidence' must be between confidence_threshold and 1.0.")
        trajectory = analytics.get("trajectory", {})
        if trajectory:
            for key in ("assumed_fps", "stationary_speed_threshold", "stationary_duration_seconds"):
                _positive_number(trajectory, key)
            for key in ("history_length", "velocity_window_frames"):
                _positive_int(trajectory, key)
            if trajectory["history_length"] < 2 or trajectory["velocity_window_frames"] < 2:
                raise ValueError("trajectory history_length and velocity_window_frames must be at least 2.")
            if trajectory["velocity_window_frames"] > trajectory["history_length"]:
                raise ValueError("trajectory velocity_window_frames cannot exceed history_length.")
        spatial = analytics.get("spatial", {})
        if spatial:
            _positive_int(spatial, "zone_history_length")
            _positive_number(spatial, "density_area_scale")
            decay = _probability(spatial, "heatmap_decay")
            if not 0.0 < decay <= 1.0:
                raise ValueError("spatial heatmap_decay must be in (0, 1].")
            grid_size = spatial.get("heatmap_grid_size")
            if not isinstance(grid_size, (list, tuple)) or len(grid_size) != 2:
                raise ValueError("spatial heatmap_grid_size must be [columns, rows].")
            for index, size in enumerate(grid_size):
                if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                    raise ValueError(f"spatial heatmap_grid_size[{index}] must be a positive integer.")
        if not body.get("enabled", False):
            return
        for key in (
            "minimum_person_width",
            "minimum_person_height",
            "max_body_tracks_per_frame",
            "body_retry_interval",
            "face_misses_before_body",
            "minimum_observations",
            "maximum_observations",
            "gender_success_interval",
            "refresh_interval",
        ):
            _positive_int(body, key)
        if "face_unknown_predictions_before_body" in body:
            _positive_int(body, "face_unknown_predictions_before_body")
        if "face_fallback_min_observations" in body:
            _positive_int(body, "face_fallback_min_observations")
            if body["face_fallback_min_observations"] < gender["minimum_observations"]:
                raise ValueError(
                    "body face_fallback_min_observations must be at least "
                    "gender_classifier.minimum_observations."
                )
        if "unknown_retry_interval" in body:
            _positive_int(body, "unknown_retry_interval")
        if body["maximum_observations"] < body["minimum_observations"]:
            raise ValueError("body maximum_observations must be at least minimum_observations.")
        body_threshold = _probability(body, "confidence_threshold")
        body_stable = _probability(body, "stable_confidence")
        if not 0.5 < body_threshold < 1.0:
            raise ValueError("body confidence_threshold must be in (0.5, 1.0).")
        if not body_threshold <= body_stable <= 1.0:
            raise ValueError("body stable_confidence must be between confidence_threshold and 1.0.")

    def reset(self) -> None:
        self.tsm.reset()
        self.person_identity.reset()
        self.people_counter.reset()
        self.line_counter.reset()
        self.trajectory_engine.reset()
        self.spatial_engine.reset()
        self.history_engine.reset()
        # A pipeline reset starts a new observation session, but it must not
        # discard the installed room profile or the selected session layout.
        self.classroom_analytics.reset()
        self.frame_index = 0
        self._last_resolved_timestamp = None
        self._last_line_counts = {"in": 0, "out": 0}
        self._face_scheduler_cursor = 0
        self._body_scheduler_cursor = 0
        self._reset_router_telemetry()

    def resolve_person_ids(self, active_tracks: dict[int, BBox], frame_size: tuple[int, int]) -> dict[int, int]:
        """Map active FastTracker IDs to conservative stream-session person IDs."""

        return self.person_identity.update(active_tracks, self.frame_index, frame_size)

    def person_identity_statistics(self) -> dict[str, object]:
        """Return non-biometric person-ID telemetry for the latest frame."""

        return self.person_identity.get_statistics()

    def _reset_router_telemetry(self) -> None:
        self._router_route_counts = {route.value: 0 for route in AttributeRoute}
        self._router_reason_counts: dict[str, int] = {}
        # Fixed bins make calibration review bounded and JSON-friendly.  They
        # describe detector boxes, never retained image crops or identities.
        self._router_person_height_bins = {"<48": 0, "48-63": 0, "64-95": 0, "96-127": 0, ">=128": 0}
        self._router_person_width_bins = {"<16": 0, "16-23": 0, "24-31": 0, "32-47": 0, ">=48": 0}
        self._router_estimated_face_bins = {"<16": 0, "16-23": 0, "24-31": 0, "32-47": 0, ">=48": 0}
        self._router_face_attempts = 0
        self._router_face_candidates = 0
        # Face extraction is deliberately reported as aggregate diagnostics.
        # Do not retain crops, landmark coordinates, or per-person quality
        # records in a live stream.
        self._router_face_extraction_status_counts = {
            status: 0
            for status in (
                "candidate",
                "no_face",
                "person_too_small",
                "face_too_small",
                "quality_rejected",
                "empty_crop",
            )
        }
        self._router_face_quality_alignment_modes: dict[str, int] = {}
        self._router_face_quality_rejection_reasons: dict[str, int] = {}
        self._router_face_quality_samples = 0
        self._router_face_quality_accepted = 0
        self._router_face_quality_sums = {
            "detection_confidence": 0.0,
            "face_width": 0.0,
            "blur_variance": 0.0,
            "brightness": 0.0,
        }
        self._router_body_attempts = 0
        self._router_body_candidates = 0
        self._router_body_inferences = 0
        self._router_body_single_frame_threshold_hits = 0
        self._router_body_evidence_updates = 0
        self._router_body_resolved_updates = 0
        self._router_face_inferences = 0
        # Fixed confidence bins expose the common "consistently 0.65--0.74"
        # deployment case without retaining raw logits or image samples.
        self._router_face_classifier_confidence_bins = {
            "<0.60": 0,
            "0.60-0.74": 0,
            "0.75-0.89": 0,
            ">=0.90": 0,
        }
        self._router_face_single_frame_threshold_hits = 0
        self._router_face_evidence_updates = 0
        self._router_face_resolved_updates = 0
        self._router_face_body_override_updates = 0
        self._router_body_fallback_reason_counts: dict[str, int] = {}
        # ``route_counts`` above intentionally remains a compact transition
        # history. These calibration samples instead describe every expensive
        # route refresh, including a desired route held back by hysteresis.
        self._router_calibration_desired_routes = {route.value: 0 for route in AttributeRoute}
        self._router_calibration_active_routes = {route.value: 0 for route in AttributeRoute}
        self._router_calibration_reasons: dict[str, int] = {}
        self._router_calibration_track_too_young = 0
        self._router_calibration_geometry = {
            "relative_person_height": {"<0.12": 0, "0.12-0.149": 0, ">=0.15": 0},
            "person_height_px": {"<48": 0, "48-95": 0, "96-159": 0, "160-239": 0, ">=240": 0},
            "person_width_px": {"<16": 0, "16-31": 0, "32-47": 0, "48-95": 0, ">=96": 0},
            "estimated_face_size_px": {"<32": 0, "32-47": 0, ">=48": 0},
            "vertical_truncation": {"none": 0, "top": 0, "bottom": 0, "both": 0},
        }

    @staticmethod
    def _increment_geometry_bin(value: float, bins: dict[str, int]) -> None:
        """Record one scalar in a fixed, bounded router-telemetry histogram."""

        if value < 16:
            key = "<16"
        elif value < 24:
            key = "16-23"
        elif value < 32:
            key = "24-31"
        elif value < 48:
            key = "32-47"
        else:
            key = ">=48"
        # Person-height bins intentionally distinguish useful body-crop sizes.
        if "<48" in bins:
            if value < 48:
                key = "<48"
            elif value < 64:
                key = "48-63"
            elif value < 96:
                key = "64-95"
            elif value < 128:
                key = "96-127"
            else:
                key = ">=128"
        bins[key] += 1

    @staticmethod
    def _increment_calibration_bin(value: float, bins: dict[str, int], thresholds: tuple[float, ...]) -> None:
        """Increment ordered, fixed calibration bins without retaining boxes."""

        labels = tuple(bins)
        for threshold, label in zip(thresholds, labels):
            if value < threshold:
                bins[label] += 1
                return
        bins[labels[-1]] += 1

    def _record_calibration_geometry(
        self,
        bbox: BBox,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> None:
        x1, y1, x2, y2 = bbox
        frame_height = int(frame_shape[0])
        width, height = max(0, x2 - x1), max(0, y2 - y1)
        geometry = self._router_calibration_geometry
        self._increment_calibration_bin(
            height / max(1, frame_height), geometry["relative_person_height"], (0.12, 0.15)
        )
        self._increment_calibration_bin(height, geometry["person_height_px"], (48, 96, 160, 240))
        self._increment_calibration_bin(width, geometry["person_width_px"], (16, 32, 48, 96))
        self._increment_calibration_bin(
            height * self.router_config.estimated_face_ratio,
            geometry["estimated_face_size_px"],
            (32, 48),
        )
        margin = self.router_config.edge_margin_px
        touches_top = y1 <= margin
        touches_bottom = y2 >= frame_height - margin
        truncation = "both" if touches_top and touches_bottom else "top" if touches_top else "bottom" if touches_bottom else "none"
        geometry["vertical_truncation"][truncation] += 1

    def resolve_attribute_routes(
        self, active_tracks: dict[int, BBox], frame_shape: tuple[int, int] | tuple[int, int, int]
    ) -> dict[int, AttributeRoute]:
        """Resolve one stable, cheap route for each active tracker ID."""

        routes: dict[int, AttributeRoute] = {}
        for track_id, bbox in sorted(active_tracks.items()):
            state = self.tsm.ensure_track(track_id)
            first_decision = state.last_attribute_route_frame < 0
            # The initial young-track decision is deliberately UNKNOWN.  Once
            # the age gate is reached, force one immediate geometry decision;
            # otherwise a route_refresh_interval of 10 would make a 3-frame
            # gate wait until frame 11.
            reached_attribute_age_gate = state.active_frames == self.minimum_track_age_for_attributes
            should_refresh = (
                first_decision
                or reached_attribute_age_gate
                or self.frame_index - state.last_attribute_route_frame >= self.route_refresh_interval
            )
            if should_refresh:
                x1, y1, x2, y2 = bbox
                person_width, person_height = max(0, x2 - x1), max(0, y2 - y1)
                self._increment_geometry_bin(person_width, self._router_person_width_bins)
                self._increment_geometry_bin(person_height, self._router_person_height_bins)
                self._increment_geometry_bin(
                    person_height * self.router_config.estimated_face_ratio,
                    self._router_estimated_face_bins,
                )
                self._record_calibration_geometry(bbox, frame_shape)
            if state.active_frames < self.minimum_track_age_for_attributes:
                desired = AttributeRoute.UNKNOWN
                reason = "track_too_young"
            elif should_refresh:
                decision = self.attribute_router.decide(
                    bbox,
                    frame_shape,
                    face_cooldown_active=self.tsm.face_cooldown_active(track_id, self.frame_index),
                )
                desired = decision.route
                reason = decision.reason
            else:
                desired = AttributeRoute(state.attribute_route)
                reason = "route_hysteresis_hold"

            # ``last_attribute_route_frame`` is the timestamp of the last
            # geometry decision, not the last frame on which the current route
            # was reused.  Updating it during hysteresis holds would postpone
            # the next refresh forever (the common track-too-young -> face
            # transition would remain ``unknown`` for the whole stream).
            if should_refresh:
                active_route, changed = self.tsm.update_attribute_route(
                    track_id,
                    desired.value,
                    self.frame_index,
                    stability_frames=self.route_stability_frames,
                )
            else:
                active_route = state.attribute_route
                changed = False
            route = AttributeRoute(active_route)
            routes[track_id] = route
            if should_refresh:
                self._router_reason_counts[reason] = self._router_reason_counts.get(reason, 0) + 1
                self._router_calibration_desired_routes[desired.value] += 1
                self._router_calibration_active_routes[route.value] += 1
                self._router_calibration_reasons[reason] = self._router_calibration_reasons.get(reason, 0) + 1
                if reason == "track_too_young":
                    self._router_calibration_track_too_young += 1
            if first_decision or changed:
                self._router_route_counts[route.value] += 1
        return routes

    def attribute_router_statistics(self) -> dict[str, object]:
        """Compact whole-stream telemetry, including finalized tracks."""

        face_evidence_states: dict[str, int] = {}
        for state in self.tsm.track_states.values():
            face_evidence_states[state.face_evidence_state] = (
                face_evidence_states.get(state.face_evidence_state, 0) + 1
            )
        quality_count = self._router_face_quality_samples
        quality_means = {
            name: round(total / quality_count, 3) if quality_count else None
            for name, total in self._router_face_quality_sums.items()
        }
        return {
            "mode": self.router_config.mode,
            "route_counts": dict(self._router_route_counts),
            "reason_counts": dict(self._router_reason_counts),
            "box_geometry": {
                "person_width_px": dict(self._router_person_width_bins),
                "person_height_px": dict(self._router_person_height_bins),
                "estimated_face_size_px": dict(self._router_estimated_face_bins),
            },
            "calibration": {
                "desired_route_counts": dict(self._router_calibration_desired_routes),
                "active_route_counts": dict(self._router_calibration_active_routes),
                "reason_counts": dict(self._router_calibration_reasons),
                "track_too_young_count": self._router_calibration_track_too_young,
                "geometry_around_gates": {
                    name: dict(bins) for name, bins in self._router_calibration_geometry.items()
                },
            },
            "face_attempts": self._router_face_attempts,
            "face_candidate_successes": self._router_face_candidates,
            "face_extraction_status_counts": dict(self._router_face_extraction_status_counts),
            "face_quality": {
                "samples": quality_count,
                "accepted": self._router_face_quality_accepted,
                "rejected": quality_count - self._router_face_quality_accepted,
                "means": quality_means,
                "alignment_modes": dict(self._router_face_quality_alignment_modes),
                "rejection_reasons": dict(self._router_face_quality_rejection_reasons),
            },
            "face_inferences": self._router_face_inferences,
            "face_classifier_confidence_bins": dict(self._router_face_classifier_confidence_bins),
            "face_single_frame_threshold_hits": self._router_face_single_frame_threshold_hits,
            "face_evidence_updates": self._router_face_evidence_updates,
            "face_resolved_updates": self._router_face_resolved_updates,
            "face_body_override_updates": self._router_face_body_override_updates,
            "active_face_evidence_states": face_evidence_states,
            "body_attempts": self._router_body_attempts,
            "body_candidate_successes": self._router_body_candidates,
            "body_inferences": self._router_body_inferences,
            "body_single_frame_threshold_hits": self._router_body_single_frame_threshold_hits,
            "body_evidence_updates": self._router_body_evidence_updates,
            "body_resolved_updates": self._router_body_resolved_updates,
            "body_fallback_reason_counts": dict(self._router_body_fallback_reason_counts),
        }

    def record_face_attempt(self, track_id: int) -> None:
        self.tsm.mark_face_attempt(track_id, self.frame_index)
        self._router_face_attempts += 1

    def record_face_candidate(self, track_id: int) -> None:
        self.tsm.ensure_track(track_id)
        self._router_face_candidates += 1

    def record_face_extraction(
        self,
        track_id: int,
        status: str,
        quality: FaceQuality | None = None,
    ) -> None:
        """Record a crop outcome without retaining biometric image data.

        Every scheduled YuNet call produces exactly one status.  A quality
        rejection remains a no-evidence attempt for temporal routing, while
        its distinct reason stays visible in aggregate telemetry instead of
        being misreported as a YuNet detector miss.
        """

        normalized_status = str(status).strip().lower() or "unknown"
        self._router_face_extraction_status_counts[normalized_status] = (
            self._router_face_extraction_status_counts.get(normalized_status, 0) + 1
        )
        if quality is not None:
            self._router_face_quality_samples += 1
            if quality.accepted:
                self._router_face_quality_accepted += 1
            self._router_face_quality_alignment_modes[quality.alignment_mode] = (
                self._router_face_quality_alignment_modes.get(quality.alignment_mode, 0) + 1
            )
            for name, value in (
                ("detection_confidence", quality.detection_confidence),
                ("face_width", quality.face_width),
                ("blur_variance", quality.blur_variance),
                ("brightness", quality.brightness),
            ):
                self._router_face_quality_sums[name] += float(value)
            for reason in quality.rejection_reasons:
                self._router_face_quality_rejection_reasons[reason] = (
                    self._router_face_quality_rejection_reasons.get(reason, 0) + 1
                )
        if normalized_status == "candidate":
            self.record_face_candidate(track_id)
        else:
            self.record_face_miss(track_id)

    def record_face_inference(
        self,
        track_id: int,
        confidence: float,
        evidence_updated: bool,
        *,
        resolution_transition: bool,
        body_override: bool,
    ) -> None:
        """Record face-model work and true source transitions exactly once."""

        self.tsm.ensure_track(track_id)
        self._router_face_inferences += 1
        if confidence < 0.60:
            confidence_bin = "<0.60"
        elif confidence < 0.75:
            confidence_bin = "0.60-0.74"
        elif confidence < 0.90:
            confidence_bin = "0.75-0.89"
        else:
            confidence_bin = ">=0.90"
        self._router_face_classifier_confidence_bins[confidence_bin] += 1
        if confidence >= self.tsm.confidence_threshold:
            self._router_face_single_frame_threshold_hits += 1
        if evidence_updated:
            self._router_face_evidence_updates += 1
        if resolution_transition:
            self._router_face_resolved_updates += 1
        if body_override:
            self._router_face_body_override_updates += 1

    def record_face_miss(self, track_id: int) -> None:
        self.tsm.record_face_miss(track_id)
        state = self.tsm.ensure_track(track_id)
        if (
            self.router_config.mode == "adaptive"
            and self.body_enabled
            and state.consecutive_face_misses >= self.face_failure_limit
        ):
            self.tsm.set_face_cooldown(track_id, self.frame_index + self.face_cooldown_frames)

    def record_body_attempt(self, track_id: int) -> None:
        self.tsm.mark_body_attempt(track_id, self.frame_index)
        self._router_body_attempts += 1

    def record_body_candidate(self, track_id: int) -> None:
        self.tsm.ensure_track(track_id)
        self._router_body_candidates += 1

    def record_body_inference(
        self,
        track_id: int,
        confidence: float,
        evidence_updated: bool,
        *,
        resolution_transition: bool,
    ) -> None:
        """Keep model-stage telemetry separate from crop-stage telemetry."""

        self._router_body_inferences += 1
        if confidence >= self.tsm.body_confidence_threshold:
            self._router_body_single_frame_threshold_hits += 1
        if evidence_updated:
            self._router_body_evidence_updates += 1
        if resolution_transition:
            self._router_body_resolved_updates += 1

    @staticmethod
    def _round_robin(
        eligible: list[int], limit: int, cursor: int
    ) -> tuple[list[int], int]:
        if len(eligible) <= limit:
            return eligible, cursor
        start = cursor % len(eligible)
        rotated = eligible[start:] + eligible[:start]
        selected = rotated[:limit]
        return selected, (start + len(selected)) % len(eligible)

    def scheduled_face_track_ids(
        self, active_track_ids: list[int], routes: dict[int, AttributeRoute] | None = None
    ) -> list[int]:
        """Select eligible tracks fairly so low IDs do not monopolize the face budget."""
        eligible = [
            track_id
            for track_id in sorted(active_track_ids)
            if (
                routes is None or routes.get(track_id) in {AttributeRoute.FACE, AttributeRoute.FACE_THEN_BODY}
            )
            and self.tsm.should_attempt_face(track_id, self.frame_index, self.face_retry_interval)
        ]
        selected, self._face_scheduler_cursor = self._round_robin(
            eligible, self.max_face_tracks_per_frame, self._face_scheduler_cursor
        )
        return selected

    def scheduled_body_track_ids(
        self, active_track_ids: list[int], *, direct: bool = False, limit: int | None = None
    ) -> list[int]:
        """Round-robin body fallbacks; caller gives these tracks priority over face work this frame."""
        if not self.body_enabled:
            return []
        eligible: list[int] = []
        fallback_reasons: dict[int, str] = {}
        for track_id in sorted(active_track_ids):
            if direct:
                allowed = self.tsm.should_attempt_body_direct(track_id, self.frame_index, self.body_retry_interval)
                reason = "direct_route"
            else:
                allowed = self.tsm.should_attempt_body(
                    track_id,
                    self.frame_index,
                    self.body_retry_interval,
                    self.body_face_misses_before_attempt,
                    self.body_face_unknown_predictions_before_attempt,
                    self.body_face_fallback_min_observations,
                )
                reason = self.tsm.ensure_track(track_id).last_body_fallback_reason
            if allowed:
                eligible.append(track_id)
                fallback_reasons[track_id] = reason
        selection_limit = self.max_body_tracks_per_frame if limit is None else int(limit)
        if selection_limit < 0:
            raise ValueError("body scheduling limit cannot be negative")
        selected, self._body_scheduler_cursor = self._round_robin(
            eligible, selection_limit, self._body_scheduler_cursor
        )
        for track_id in selected:
            reason = fallback_reasons[track_id]
            self._router_body_fallback_reason_counts[reason] = (
                self._router_body_fallback_reason_counts.get(reason, 0) + 1
            )
        return selected

    def update_classroom(
        self,
        confirmed_tracks: Mapping[int, BBox],
        frame_size: tuple[int, int],
        timestamp_seconds: float,
    ) -> dict[str, object]:
        """Update optional seat/room analytics from already-confirmed tracks only."""

        statistics = self.classroom_analytics.update(confirmed_tracks, frame_size, timestamp_seconds)
        statistics["layout_revision"] = self._classroom_layout_revision
        return statistics

    def classroom_statistics(self) -> dict[str, object]:
        """Return the compact current classroom snapshot with layout revision."""

        statistics = self.classroom_analytics.get_statistics()
        statistics["layout_revision"] = self._classroom_layout_revision
        return statistics

    def apply_session_layout(self, session_layout: Mapping[str, Any]) -> dict[str, object]:
        """Replace the current session's rows/disabled seats without reloading models.

        A web setup screen can pass only the dynamic fields.  The room name
        and current template are filled in when unambiguous; callers can also
        supply an explicit complete ``SessionLayout`` mapping.
        """

        if self.classroom_config is None:
            raise RuntimeError("A configured classroom room_profile is required before applying a session layout.")
        if not isinstance(session_layout, Mapping):
            raise ValueError("session_layout must be a mapping.")
        values = dict(session_layout)
        values.setdefault("room_profile", self.classroom_config.room_profile.name)
        if self.classroom_config.active_template is not None:
            values.setdefault("template", self.classroom_config.active_template.name)
        template_name = values.get("template", values.get("layout_template"))
        if not isinstance(template_name, str) or not template_name.strip():
            raise ValueError("session_layout.template is required when no current template is active.")
        template = next(
            (candidate for candidate in self.classroom_config.layout_templates if candidate.name == template_name.strip()),
            None,
        )
        if template is None:
            raise ValueError(f"Unknown classroom layout template: {template_name!r}.")
        updated_layout = SessionLayout.from_mapping(values, template=template)
        self.classroom_config = replace(self.classroom_config, session_layout=updated_layout)
        # Rebuilding only this lightweight analytics branch deliberately clears
        # old seat reservations while FastTracker, models and person counters
        # stay alive for the webcam session.
        self.classroom_analytics = ClassroomAnalytics(self.classroom_config, **self._classroom_engine_options)
        self._classroom_layout_revision += 1
        return self.classroom_statistics()

    def finalize_stale_tracks(self, timestamp_seconds: float | None = None) -> list[FinalizedTrack]:
        finalized = self.tsm.prune_stale(self.frame_index, self.max_missing_frames)
        if finalized:
            self.people_counter.consume_finalized(finalized)
            self.line_counter.consume_finalized(finalized)
            self.trajectory_engine.consume_finalized(finalized)
            self.spatial_engine.consume_finalized(finalized)
            self.classroom_analytics.consume_finalized(finalized, timestamp_seconds=timestamp_seconds)
        return finalized

    def resolve_timestamp(self, timestamp_seconds: float | None) -> float:
        """Use source/camera time when supplied, otherwise preserve a deterministic FPS clock."""
        if timestamp_seconds is None:
            resolved = self.frame_index / self.assumed_fps
        elif timestamp_seconds < 0.0:
            raise ValueError("timestamp_seconds cannot be negative.")
        else:
            resolved = float(timestamp_seconds)
        if self._last_resolved_timestamp is not None:
            resolved = max(resolved, self._last_resolved_timestamp)
        self._last_resolved_timestamp = resolved
        return resolved

    def update_space_and_history(
        self,
        timestamp_seconds: float,
        crowd_statistics: dict[str, object],
        crossing_statistics: dict[str, object],
        spatial_statistics: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Derive calibrated space metrics and bounded session history once per frame."""

        confirmed_count = int(crowd_statistics.get("confirmed_active_count", 0))
        space_statistics = self.space_analytics.build(confirmed_count, spatial_statistics)
        current_in = int(crossing_statistics.get("in", 0))
        current_out = int(crossing_statistics.get("out", 0))
        in_events = max(0, current_in - self._last_line_counts["in"])
        out_events = max(0, current_out - self._last_line_counts["out"])
        self._last_line_counts = {"in": current_in, "out": current_out}
        occupancy = space_statistics["occupancy"]
        physical_density = space_statistics["physical_density"]
        self.history_engine.update(
            timestamp_seconds,
            confirmed_count,
            in_events=in_events,
            out_events=out_events,
            occupancy_rate=occupancy["rate"],
            people_per_m2=physical_density["people_per_m2"],
        )
        # The Gradio webcam callback needs only the compact current/peak/flow
        # summary. Keep detailed bounded snapshots available from the engine for
        # a slower chart/export endpoint instead of serialising them every frame.
        return space_statistics, self.history_engine.get_statistics(
            timestamp_seconds,
            include_snapshots=False,
        )
