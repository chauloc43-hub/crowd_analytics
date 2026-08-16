"""Zone, screen-density and bounded heatmap analytics for a single camera stream."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import cv2
import numpy as np

from src.analytics.geometry import scale_polygon
from src.analytics.zones import ZoneManager
from src.tracking.track_state import FinalizedTrack


BBox = tuple[int, int, int, int]
Point = tuple[float, float]


def _polygon_area(polygon: list[Point]) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(
        sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
    ) / 2.0


class SpatialEngine:
    """Computes camera-plane density and a decayed foot-point heatmap.

    Values are screen-space proxies. They become physical density only after a
    camera-calibration/homography step, which is deliberately outside this MVP.
    """

    def __init__(
        self,
        zones_config: list[dict],
        reference_size: tuple[int, int],
        *,
        zone_history_length: int = 300,
        heatmap_grid_size: tuple[int, int] = (16, 12),
        heatmap_decay: float = 0.995,
        density_area_scale: float = 100_000.0,
        visible_floor_area_m2: float | None = None,
    ) -> None:
        reference_width, reference_height = reference_size
        columns, rows = heatmap_grid_size
        if reference_width <= 0 or reference_height <= 0:
            raise ValueError("reference_size must contain positive dimensions.")
        if columns < 1 or rows < 1:
            raise ValueError("heatmap_grid_size must contain positive integers.")
        if not 0.0 < heatmap_decay <= 1.0:
            raise ValueError("heatmap_decay must be in (0, 1].")
        if density_area_scale <= 0.0:
            raise ValueError("density_area_scale must be positive.")
        if visible_floor_area_m2 is not None and visible_floor_area_m2 <= 0.0:
            raise ValueError("visible_floor_area_m2 must be positive when configured.")
        self.reference_size = reference_size
        self.heatmap_grid_size = (int(columns), int(rows))
        self.heatmap_decay = float(heatmap_decay)
        self.density_area_scale = float(density_area_scale)
        self.visible_floor_area_m2 = None if visible_floor_area_m2 is None else float(visible_floor_area_m2)
        self._zones_by_name = {str(zone["name"]): dict(zone) for zone in zones_config}
        self.zone_manager = ZoneManager(zones_config, history_length=zone_history_length)
        self.reset()

    def reset(self) -> None:
        columns, rows = self.heatmap_grid_size
        self.zone_manager.reset()
        self._heatmap = np.zeros((rows, columns), dtype=np.float32)
        self._accumulated_heatmap = np.zeros((rows, columns), dtype=np.float32)
        self._last_timestamp_by_track: dict[int, float] = {}
        self._previous_zones_by_track: dict[int, set[str]] = {}
        self._zone_dwell_seconds: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._zone_tracks_seen: dict[int, set[str]] = defaultdict(set)
        self._finalized_zone_dwell_seconds: dict[str, float] = defaultdict(float)
        self._finalized_zone_track_counts: dict[str, int] = defaultdict(int)
        self._latest_zone_states: dict[str, dict[str, object]] = {}
        self._latest_density: dict[str, object] = {}
        self._latest_distribution: dict[str, object] = {}

    @staticmethod
    def _foot_point(bbox: BBox) -> Point:
        x1, _y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, float(y2))

    def _reference_point(self, point: Point, frame_size: tuple[int, int]) -> Point:
        frame_width, frame_height = frame_size
        reference_width, reference_height = self.reference_size
        return (point[0] * reference_width / frame_width, point[1] * reference_height / frame_height)

    def _zone_names_for_point(self, point: Point, polygons: dict[str, list[Point]]) -> set[str]:
        return {
            name
            for name, polygon in polygons.items()
            if ZoneManager._is_point_in_polygon(point, polygon)
        }

    def _add_heatmap_point(self, point: Point) -> None:
        reference_width, reference_height = self.reference_size
        columns, rows = self.heatmap_grid_size
        column = min(columns - 1, max(0, int(point[0] * columns / reference_width)))
        row = min(rows - 1, max(0, int(point[1] * rows / reference_height)))
        self._heatmap[row, column] += 1.0
        self._accumulated_heatmap[row, column] += 1.0

    def _refresh_zone_dwell_statistics(self) -> None:
        """Refresh compact active + completed dwell aggregates in current zone stats."""

        for name, state in self._latest_zone_states.items():
            active_dwell = [
                dwell.get(name, 0.0)
                for dwell in self._zone_dwell_seconds.values()
                if name in dwell
            ]
            active_seen = sum(name in zones for zones in self._zone_tracks_seen.values())
            finalized_dwell = self._finalized_zone_dwell_seconds[name]
            finalized_tracks = self._finalized_zone_track_counts[name]
            total_dwell = finalized_dwell + sum(active_dwell)
            total_dwell_tracks = finalized_tracks + active_seen
            state["mean_active_dwell_seconds"] = (
                round(sum(active_dwell) / len(active_dwell), 2) if active_dwell else 0.0
            )
            state["completed_dwell_seconds"] = round(finalized_dwell, 2)
            state["completed_track_count"] = finalized_tracks
            state["total_dwell_seconds"] = round(total_dwell, 2)
            state["mean_session_dwell_seconds"] = (
                round(total_dwell / total_dwell_tracks, 2) if total_dwell_tracks else 0.0
            )

    def update(
        self,
        tracked_persons: dict[int, BBox],
        frame_size: tuple[int, int],
        timestamp_seconds: float,
    ) -> dict[str, object]:
        """Update confirmed tracks only; unconfirmed detections do not pollute spatial metrics."""
        frame_width, frame_height = frame_size
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame_size must contain positive dimensions.")
        if timestamp_seconds < 0.0:
            raise ValueError("timestamp_seconds cannot be negative.")
        self._heatmap *= self.heatmap_decay
        zone_counts = self.zone_manager.update(tracked_persons, frame_size, self.reference_size)
        primary_zone_by_track = self.zone_manager.primary_zone_by_track()
        primary_zone_counts = {name: 0 for name in zone_counts}
        outside_zone_count = 0
        for track_id in tracked_persons:
            primary_zone = primary_zone_by_track.get(track_id)
            if primary_zone is None:
                outside_zone_count += 1
            else:
                primary_zone_counts[primary_zone] += 1
        polygons = self.zone_manager.scaled_polygons(frame_size, self.reference_size)
        for track_id, bbox in tracked_persons.items():
            previous_timestamp = self._last_timestamp_by_track.get(track_id)
            if previous_timestamp is not None:
                elapsed = max(0.0, timestamp_seconds - previous_timestamp)
                for zone_name in self._previous_zones_by_track.get(track_id, set()):
                    self._zone_dwell_seconds[track_id][zone_name] += elapsed
            point = self._foot_point(bbox)
            current_zones = self._zone_names_for_point(point, polygons)
            self._previous_zones_by_track[track_id] = current_zones
            self._zone_tracks_seen[track_id].update(current_zones)
            self._last_timestamp_by_track[track_id] = timestamp_seconds
            self._add_heatmap_point(self._reference_point(point, frame_size))

        states = self.zone_manager.get_zone_states(zone_counts)
        reference_polygons = self.zone_manager.scaled_polygons(self.reference_size, self.reference_size)
        for name, state in states.items():
            area = _polygon_area(reference_polygons[name])
            zone_config = self._zones_by_name[name]
            area_m2 = zone_config.get("area_m2")
            capacity = zone_config.get("capacity")
            state["area_reference_pixels"] = round(area, 2)
            state["density_per_100k_reference_pixels"] = round(
                state["current_count"] * self.density_area_scale / area, 3
            ) if area else 0.0
            state["area_m2"] = float(area_m2) if area_m2 is not None else None
            state["density_people_per_m2"] = (
                round(state["current_count"] / float(area_m2), 3) if area_m2 is not None else None
            )
            state["capacity"] = int(capacity) if capacity is not None else None
            state["occupancy_rate"] = (
                round(state["current_count"] / int(capacity), 4) if capacity is not None else None
            )
        reference_area = float(self.reference_size[0] * self.reference_size[1])
        self._latest_zone_states = states
        self._latest_density = {
            "confirmed_count": len(tracked_persons),
            "density_per_100k_reference_pixels": round(len(tracked_persons) * self.density_area_scale / reference_area, 3),
            "unit": "confirmed_tracks_per_100k_reference_pixels",
            "visible_floor_area_m2": self.visible_floor_area_m2,
            "people_per_m2": (
                round(len(tracked_persons) / self.visible_floor_area_m2, 3)
                if self.visible_floor_area_m2 is not None
                else None
            ),
        }
        self._latest_distribution = {
            "assignment_policy": "highest_priority_then_config_order",
            "primary_zone_counts": primary_zone_counts,
            "outside_zone_count": outside_zone_count,
            "assigned_confirmed_count": sum(primary_zone_counts.values()),
            "confirmed_count": len(tracked_persons),
        }
        self._refresh_zone_dwell_statistics()
        return self.get_statistics()

    def get_statistics(self) -> dict[str, object]:
        peak_index = np.unravel_index(int(np.argmax(self._heatmap)), self._heatmap.shape)
        accumulated_peak_index = np.unravel_index(int(np.argmax(self._accumulated_heatmap)), self._accumulated_heatmap.shape)
        return {
            "zones": {name: state.copy() for name, state in self._latest_zone_states.items()},
            "density": self._latest_density.copy(),
            "distribution": {
                **self._latest_distribution,
                "primary_zone_counts": dict(self._latest_distribution.get("primary_zone_counts", {})),
            },
            "heatmap": {
                "grid_size": list(self.heatmap_grid_size),
                "reference_resolution": list(self.reference_size),
                "decay": self.heatmap_decay,
                "total_weight": round(float(self._heatmap.sum()), 3),
                "peak_cell": [int(peak_index[1]), int(peak_index[0])],
                "peak_value": round(float(self._heatmap[peak_index]), 3),
                "values": np.round(self._heatmap, 3).tolist(),
                "accumulated": {
                    "total_weight": round(float(self._accumulated_heatmap.sum()), 3),
                    "peak_cell": [int(accumulated_peak_index[1]), int(accumulated_peak_index[0])],
                    "peak_value": round(float(self._accumulated_heatmap[accumulated_peak_index]), 3),
                    "values": np.round(self._accumulated_heatmap, 3).tolist(),
                },
            },
        }

    def heatmap_image(self, frame_size: tuple[int, int], *, mode: str = "decayed") -> np.ndarray:
        """Return a BGR overlay-sized heatmap; caller chooses alpha blending.

        ``decayed`` communicates recent activity; ``accumulated`` shows
        session-long foot-point usage.  Both are bounded fixed-size grids.
        """
        frame_width, frame_height = frame_size
        if mode == "decayed":
            values = self._heatmap
        elif mode == "accumulated":
            values = self._accumulated_heatmap
        else:
            raise ValueError("heatmap mode must be 'decayed' or 'accumulated'.")
        maximum = float(values.max())
        if maximum <= 0.0:
            return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        normalized = np.uint8(np.clip(values / maximum * 255.0, 0.0, 255.0))
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        return cv2.resize(colored, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

    def consume_finalized(self, finalized_tracks: Iterable[FinalizedTrack]) -> None:
        for summary in finalized_tracks:
            track_id = summary.track_id
            self._last_timestamp_by_track.pop(track_id, None)
            self._previous_zones_by_track.pop(track_id, None)
            dwell_by_zone = self._zone_dwell_seconds.pop(track_id, {})
            seen_zones = self._zone_tracks_seen.pop(track_id, set())
            for zone_name in seen_zones:
                self._finalized_zone_dwell_seconds[zone_name] += float(dwell_by_zone.get(zone_name, 0.0))
                self._finalized_zone_track_counts[zone_name] += 1
        self._refresh_zone_dwell_statistics()
