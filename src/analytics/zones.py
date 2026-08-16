from __future__ import annotations

from src.analytics.geometry import scale_polygon


class ZoneManager:
    """Counts confirmed tracks inside reference-resolution polygons.

    ``current_count`` remains inclusive for backward compatibility: a person in
    two overlapping zones contributes to both counts.  The separate primary
    assignment API is for distributions, where each confirmed person must be
    counted exactly once.
    """

    def __init__(self, zones_config: list[dict], history_length: int = 300) -> None:
        self.zones = zones_config
        self.history_length = history_length
        self.reset()

    def reset(self) -> None:
        self.zone_histories = {zone["name"]: [] for zone in self.zones}
        self._primary_zone_by_track: dict[int, str] = {}

    def _primary_zone_order(self) -> list[dict]:
        """Return deterministic distribution priority: higher priority, then YAML order."""

        return [
            zone
            for _index, zone in sorted(
                enumerate(self.zones),
                key=lambda item: (-int(item[1].get("priority", 0)), item[0]),
            )
        ]

    @staticmethod
    def _is_point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
        x, y = point
        inside = False
        previous_x, previous_y = polygon[-1]
        for current_x, current_y in polygon:
            crosses_y = (current_y > y) != (previous_y > y)
            if crosses_y:
                intersection_x = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
                if x < intersection_x:
                    inside = not inside
            previous_x, previous_y = current_x, current_y
        return inside

    def scaled_polygons(
        self, frame_size: tuple[int, int], reference_size: tuple[int, int]
    ) -> dict[str, list[tuple[float, float]]]:
        return {
            zone["name"]: scale_polygon(zone["polygon"], frame_size, reference_size)
            for zone in self.zones
        }

    def update(
        self,
        tracked_persons: dict[int, tuple[int, int, int, int]],
        frame_size: tuple[int, int],
        reference_size: tuple[int, int],
    ) -> dict[str, int]:
        polygons = self.scaled_polygons(frame_size, reference_size)
        zone_counts = {zone["name"]: 0 for zone in self.zones}
        primary_zone_by_track: dict[int, str] = {}
        primary_order = self._primary_zone_order()
        for track_id, (x1, _y1, x2, y2) in tracked_persons.items():
            foot_point = ((x1 + x2) / 2.0, float(y2))
            for zone in self.zones:
                if self._is_point_in_polygon(foot_point, polygons[zone["name"]]):
                    zone_counts[zone["name"]] += 1
            for zone in primary_order:
                if self._is_point_in_polygon(foot_point, polygons[zone["name"]]):
                    primary_zone_by_track[track_id] = str(zone["name"])
                    break

        for name, count in zone_counts.items():
            history = self.zone_histories[name]
            history.append(count)
            if len(history) > self.history_length:
                history.pop(0)
        self._primary_zone_by_track = primary_zone_by_track
        return zone_counts

    def primary_zone_by_track(self) -> dict[int, str]:
        """Return one non-overlapping assignment per active track from the last update."""

        return dict(self._primary_zone_by_track)

    def get_zone_states(self, current_counts: dict[str, int]) -> dict[str, dict]:
        states = {}
        thresholds = {zone["name"]: zone.get("crowded_threshold", 10) for zone in self.zones}
        for name, count in current_counts.items():
            history = self.zone_histories[name]
            average = sum(history) / len(history) if history else 0.0
            states[name] = {
                "current_count": count,
                "average_count": round(average, 1),
                "status": "CROWDED" if count >= thresholds[name] else "NORMAL",
            }
        return states
