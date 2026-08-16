"""Screen-space trajectory metrics derived from the tracker, without another model."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from math import atan2, degrees, hypot
from typing import Iterable

from src.tracking.track_state import FinalizedTrack


Point = tuple[float, float]
BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class MotionSample:
    timestamp_seconds: float
    point: Point


@dataclass
class TrackMotionState:
    first_timestamp_seconds: float
    last_motion_timestamp_seconds: float
    samples: deque[MotionSample] = field(default_factory=deque)


def _direction_from_vector(dx: float, dy: float) -> str:
    """Map image-space movement to a compact eight-way label (positive y is down)."""
    angle = (degrees(atan2(dy, dx)) + 360.0) % 360.0
    labels = ("right", "down_right", "down", "down_left", "left", "up_left", "up", "up_right")
    return labels[int((angle + 22.5) // 45.0) % len(labels)]


class TrajectoryEngine:
    """Maintains bounded screen-space motion summaries for one tracker stream.

    All distances are in *reference-frame pixels*, not metres. A camera calibration
    branch would be required before reporting physical speed or real-world density.
    """

    def __init__(
        self,
        reference_size: tuple[int, int],
        *,
        assumed_fps: float = 25.0,
        history_length: int = 60,
        velocity_window_frames: int = 8,
        stationary_speed_threshold: float = 8.0,
        stationary_duration_seconds: float = 2.0,
    ) -> None:
        reference_width, reference_height = reference_size
        if reference_width <= 0 or reference_height <= 0:
            raise ValueError("reference_size must contain positive dimensions.")
        if assumed_fps <= 0.0:
            raise ValueError("assumed_fps must be positive.")
        if history_length < 2 or velocity_window_frames < 2:
            raise ValueError("history_length and velocity_window_frames must be at least 2.")
        if velocity_window_frames > history_length:
            raise ValueError("velocity_window_frames cannot exceed history_length.")
        if stationary_speed_threshold < 0.0 or stationary_duration_seconds < 0.0:
            raise ValueError("stationary thresholds cannot be negative.")
        self.reference_size = reference_size
        self.assumed_fps = float(assumed_fps)
        self.history_length = int(history_length)
        self.velocity_window_frames = int(velocity_window_frames)
        self.stationary_speed_threshold = float(stationary_speed_threshold)
        self.stationary_duration_seconds = float(stationary_duration_seconds)
        self.reset()

    def reset(self) -> None:
        self._tracks: dict[int, TrackMotionState] = {}
        self._latest_metrics: dict[int, dict[str, object]] = {}

    @staticmethod
    def _foot_point(bbox: BBox) -> Point:
        x1, _y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, float(y2))

    def _to_reference(self, point: Point, frame_size: tuple[int, int]) -> Point:
        frame_width, frame_height = frame_size
        reference_width, reference_height = self.reference_size
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame_size must contain positive dimensions.")
        return (
            point[0] * reference_width / frame_width,
            point[1] * reference_height / frame_height,
        )

    def _timestamp(self, frame_index: int, timestamp_seconds: float | None) -> float:
        if timestamp_seconds is not None:
            if timestamp_seconds < 0.0:
                raise ValueError("timestamp_seconds cannot be negative.")
            return float(timestamp_seconds)
        return float(frame_index) / self.assumed_fps

    def update(
        self,
        tracked_persons: dict[int, BBox],
        frame_size: tuple[int, int],
        frame_index: int,
        timestamp_seconds: float | None = None,
    ) -> dict[int, dict[str, object]]:
        """Update active tracks and return their latest screen-space metrics."""
        timestamp = self._timestamp(frame_index, timestamp_seconds)
        metrics: dict[int, dict[str, object]] = {}
        for track_id, bbox in tracked_persons.items():
            point = self._to_reference(self._foot_point(bbox), frame_size)
            state = self._tracks.get(track_id)
            if state is None:
                state = TrackMotionState(timestamp, timestamp, deque(maxlen=self.history_length))
                self._tracks[track_id] = state
            if state.samples and timestamp <= state.samples[-1].timestamp_seconds:
                # A caller may provide repeated video timestamps. Preserve monotonic time
                # without exposing a division-by-zero speed spike.
                timestamp = state.samples[-1].timestamp_seconds + 1.0 / self.assumed_fps
            state.samples.append(MotionSample(timestamp, point))
            window_start = state.samples[max(0, len(state.samples) - self.velocity_window_frames)]
            elapsed = max(timestamp - window_start.timestamp_seconds, 1.0 / self.assumed_fps)
            dx = point[0] - window_start.point[0]
            dy = point[1] - window_start.point[1]
            speed = hypot(dx, dy) / elapsed
            moving = speed > self.stationary_speed_threshold
            if moving:
                state.last_motion_timestamp_seconds = timestamp
            stationary_duration = max(0.0, timestamp - state.last_motion_timestamp_seconds)
            stationary = not moving and stationary_duration >= self.stationary_duration_seconds
            direction = _direction_from_vector(dx, dy) if moving else "stationary"
            metrics[track_id] = {
                "speed_reference_px_per_second": round(speed, 2),
                "direction": direction,
                "stationary": stationary,
                "stationary_duration_seconds": round(stationary_duration, 2),
                "dwell_seconds": round(max(0.0, timestamp - state.first_timestamp_seconds), 2),
            }
        self._latest_metrics = metrics
        return metrics

    def get_track_metrics(self, track_id: int) -> dict[str, object] | None:
        metric = self._latest_metrics.get(track_id)
        return metric.copy() if metric is not None else None

    def get_summary(self, active_track_ids: Iterable[int]) -> dict[str, object]:
        active_metrics = {track_id: self._latest_metrics[track_id] for track_id in active_track_ids if track_id in self._latest_metrics}
        speeds = [float(metric["speed_reference_px_per_second"]) for metric in active_metrics.values()]
        stationary_count = sum(bool(metric["stationary"]) for metric in active_metrics.values())
        direction_counts = Counter(
            str(metric["direction"]) for metric in active_metrics.values() if metric["direction"] != "stationary"
        )
        return {
            "unit": "reference_px_per_second",
            "active_tracks": {str(track_id): metric.copy() for track_id, metric in sorted(active_metrics.items())},
            "moving_count": len(active_metrics) - stationary_count,
            "stationary_count": stationary_count,
            "mean_speed_reference_px_per_second": round(sum(speeds) / len(speeds), 2) if speeds else 0.0,
            "max_speed_reference_px_per_second": round(max(speeds), 2) if speeds else 0.0,
            "direction_counts": dict(sorted(direction_counts.items())),
        }

    def consume_finalized(self, finalized_tracks: Iterable[FinalizedTrack]) -> None:
        for summary in finalized_tracks:
            self._tracks.pop(summary.track_id, None)
            self._latest_metrics.pop(summary.track_id, None)
