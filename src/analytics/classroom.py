"""Classroom-specific facade built on top of tracking and seat occupancy.

The general crowd pipeline intentionally keeps counting, spatial analytics and
tracking independent from any particular room.  This module is the optional
bridge for a classroom installation.  It consumes *confirmed* tracker boxes,
never creates or changes tracker IDs, and reports three distinct concepts:

* people visible in the configured room boundary;
* official room capacity / visible-floor density; and
* seat, block, aisle and entrance state for the active session layout.

The facade is deliberately conservative when geometry is absent.  A semantic
``2-4-2`` layout without a reference grid still reports its configured seats,
but it never invents seat occupancy from image pixels.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
from numbers import Real
from typing import Any, TypeAlias

from src.analytics.classroom_config import ClassroomConfig, Point, SeatDefinition, parse_classroom_config
from src.analytics.seat_occupancy import SeatOccupancyEngine


TrackId: TypeAlias = int | str
BBox: TypeAlias = tuple[float, float, float, float]
Polygon: TypeAlias = tuple[Point, ...]


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _frame_size(value: object) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError("frame_size must contain [width, height].")
    width = _finite_number(value[0], "frame_size[0]")
    height = _finite_number(value[1], "frame_size[1]")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("frame_size must contain positive dimensions.")
    return (int(width), int(height))


def _track_id(value: object, name: str = "track_id") -> TrackId:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{name} must be an int or a non-empty string.")
    if isinstance(value, str):
        result = value.strip()
        if not result:
            raise ValueError(f"{name} must be an int or a non-empty string.")
        return result
    return int(value)


def _bbox(value: object, name: str) -> BBox:
    if isinstance(value, Mapping):
        if "bbox" in value:
            return _bbox(value["bbox"], f"{name}.bbox")
        try:
            value = (value["x1"], value["y1"], value["x2"], value["y2"])
        except KeyError as error:
            raise ValueError(f"{name} must contain a bbox.") from error
    elif not isinstance(value, (str, bytes)) and hasattr(value, "bbox"):
        return _bbox(getattr(value, "bbox"), f"{name}.bbox")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError(f"{name} must be [x1, y1, x2, y2].")
    x1, y1, x2, y2 = (_finite_number(item, f"{name}[{index}]") for index, item in enumerate(value))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} must have x2 > x1 and y2 > y1.")
    return (x1, y1, x2, y2)


def _tracked_boxes(value: Mapping[TrackId, Sequence[float]] | Iterable[object]) -> dict[TrackId, BBox]:
    """Normalise the same lightweight record shapes accepted by seat occupancy."""

    if isinstance(value, Mapping):
        if "track_id" in value and ("bbox" in value or {"x1", "y1", "x2", "y2"}.issubset(value)):
            raw_items: Iterable[object] = (value,)
        else:
            raw_items = tuple(value.items())
    else:
        raw_items = value
    tracks: dict[TrackId, BBox] = {}
    for index, item in enumerate(raw_items):
        if isinstance(item, Mapping):
            raw_identifier = item.get("track_id", item.get("id"))
            identifier = _track_id(raw_identifier, f"tracked_persons[{index}].track_id")
            box = _bbox(item, f"tracked_persons[{index}]")
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            identifier = _track_id(item[0], f"tracked_persons[{index}][0]")
            box = _bbox(item[1], f"tracked_persons[{index}][1]")
        else:
            identifier = _track_id(
                getattr(item, "track_id", getattr(item, "id", None)),
                f"tracked_persons[{index}].track_id",
            )
            box = _bbox(item, f"tracked_persons[{index}]")
        if identifier in tracks:
            raise ValueError(f"tracked_persons contains duplicate track_id {identifier!r}.")
        tracks[identifier] = box
    return tracks


def _finalized_track_id(value: object) -> TrackId:
    if isinstance(value, (int, str)) and not isinstance(value, bool):
        return _track_id(value)
    if isinstance(value, Mapping):
        return _track_id(value.get("track_id", value.get("id")))
    return _track_id(getattr(value, "track_id", getattr(value, "id", None)))


def _foot_point(bbox: BBox) -> Point:
    return ((bbox[0] + bbox[2]) / 2.0, bbox[3])


def _point_on_segment(point: Point, first: Point, second: Point) -> bool:
    cross = (point[1] - first[1]) * (second[0] - first[0]) - (point[0] - first[0]) * (second[1] - first[1])
    if abs(cross) > 1e-9:
        return False
    return (
        min(first[0], second[0]) - 1e-9 <= point[0] <= max(first[0], second[0]) + 1e-9
        and min(first[1], second[1]) - 1e-9 <= point[1] <= max(first[1], second[1]) + 1e-9
    )


def _contains_point(point: Point, polygon: Sequence[Point]) -> bool:
    """Return true for a point inside or on an edge of a simple polygon."""

    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        crosses = (current[1] > point[1]) != (previous[1] > point[1])
        if crosses:
            x_intercept = (previous[0] - current[0]) * (point[1] - current[1]) / (previous[1] - current[1]) + current[0]
            if point[0] < x_intercept:
                inside = not inside
        previous = current
    return inside


def _scale_polygon(
    polygon: Sequence[Point],
    *,
    frame_size: tuple[int, int] | None,
    reference_size: tuple[int, int] | None,
) -> Polygon:
    if frame_size is None or reference_size is None:
        return tuple((float(x_coord), float(y_coord)) for x_coord, y_coord in polygon)
    frame_width, frame_height = frame_size
    reference_width, reference_height = reference_size
    return tuple(
        (x_coord * frame_width / reference_width, y_coord * frame_height / reference_height)
        for x_coord, y_coord in polygon
    )


def _rounded_ratio(numerator: int, denominator: int | None) -> tuple[float | None, float | None]:
    if denominator is None or denominator <= 0:
        return (None, None)
    ratio = numerator / denominator
    return (round(ratio, 4), round(ratio * 100.0, 1))


class ClassroomAnalytics:
    """Optional classroom view over confirmed tracker observations.

    Args:
        config: A parsed :class:`ClassroomConfig`, its YAML-compatible mapping,
            or ``None``.  ``None`` produces a safe ``not_configured`` snapshot
            and is useful for preserving existing non-classroom deployments.
        seat_engine: An optional injected engine, primarily for tests or a
            caller that owns a custom seat-assignment implementation.  When it
            is omitted, an engine is built only if every configured seat has a
            usable polygon.
        occupancy_confirm_seconds: Confirmation time for the internally
            created seat engine.
        vacancy_grace_seconds: Lost-track grace time for the internally
            created seat engine.
    """

    def __init__(
        self,
        config: ClassroomConfig | Mapping[str, Any] | None = None,
        *,
        seat_engine: SeatOccupancyEngine | None = None,
        occupancy_confirm_seconds: float = 1.5,
        vacancy_grace_seconds: float = 2.0,
        aspect_ratio_tolerance: float = 0.02,
    ) -> None:
        self.occupancy_confirm_seconds = _finite_number(
            occupancy_confirm_seconds, "occupancy_confirm_seconds"
        )
        self.vacancy_grace_seconds = _finite_number(vacancy_grace_seconds, "vacancy_grace_seconds")
        self.aspect_ratio_tolerance = _finite_number(aspect_ratio_tolerance, "aspect_ratio_tolerance")
        if self.occupancy_confirm_seconds < 0.0 or self.vacancy_grace_seconds < 0.0:
            raise ValueError("occupancy confirmation and vacancy grace must be non-negative.")
        if not 0.0 <= self.aspect_ratio_tolerance < 1.0:
            raise ValueError("aspect_ratio_tolerance must be in [0, 1).")
        self.config: ClassroomConfig | None = None
        self.seat_engine: SeatOccupancyEngine | None = None
        self._last_timestamp: float | None = None
        self._last_frame_size: tuple[int, int] | None = None
        self._last_tracks: dict[TrackId, BBox] = {}
        self._last_statistics: dict[str, object] = {}
        self._geometry_status = "not_observed"
        self.configure(config, seat_engine=seat_engine)

    @property
    def is_configured(self) -> bool:
        return self.config is not None

    @property
    def has_seat_geometry(self) -> bool:
        return self.seat_engine is not None

    def configure(
        self,
        config: ClassroomConfig | Mapping[str, Any] | None,
        *,
        seat_engine: SeatOccupancyEngine | None = None,
    ) -> None:
        """Replace the optional room/session configuration and reset its state.

        This is intentionally explicit: calling :meth:`reset` clears only live
        session state and preserves the same room/layout definitions.
        """

        if config is None:
            parsed: ClassroomConfig | None = None
        elif isinstance(config, ClassroomConfig):
            parsed = config
        elif isinstance(config, Mapping):
            parsed = parse_classroom_config(config)
        else:
            raise ValueError("config must be ClassroomConfig, a mapping, or None.")
        self.config = parsed
        self.seat_engine = seat_engine if seat_engine is not None else self._build_seat_engine(parsed)
        self.reset()

    def _build_seat_engine(self, config: ClassroomConfig | None) -> SeatOccupancyEngine | None:
        if config is None:
            return None
        seats = config.seat_definitions
        # A semantic layout can be configured before camera points are drawn.
        # Do not turn absent geometry into fabricated occupied/vacant seats.
        if not seats or any(seat.polygon is None for seat in seats):
            return None
        template = config.active_template
        grid = None if template is None else template.reference_grid
        anchor_ratio = 0.65 if grid is None else grid.seat_anchor_ratio
        return SeatOccupancyEngine(
            seats,
            reference_size=config.room_profile.reference_resolution,
            anchor_ratio=anchor_ratio,
            occupancy_confirm_seconds=self.occupancy_confirm_seconds,
            vacancy_grace_seconds=self.vacancy_grace_seconds,
        )

    def reset(self) -> None:
        """Clear live assignments while retaining the fixed room/session layout."""

        if self.seat_engine is not None:
            self.seat_engine.reset()
        self._last_timestamp = None
        self._last_frame_size = None
        self._last_tracks = {}
        self._geometry_status = "not_observed"
        self._last_statistics = self._build_statistics()

    def _geometry_status_for_frame(self, frame_size: tuple[int, int]) -> str:
        """Reject reference-seat geometry when the camera aspect ratio changed.

        Independent x/y scaling is fine for a resize of the same camera feed,
        but it silently corrupts a calibrated seating plane after a crop or a
        different webcam aspect ratio.  In that case retain normal crowd
        tracking and report the classroom branch as unavailable instead.
        """

        if self.config is None or self.config.room_profile.reference_resolution is None:
            return "compatible"
        reference_width, reference_height = self.config.room_profile.reference_resolution
        frame_width, frame_height = frame_size
        reference_ratio = reference_width / reference_height
        frame_ratio = frame_width / frame_height
        relative_error = abs(frame_ratio / reference_ratio - 1.0)
        return "compatible" if relative_error <= self.aspect_ratio_tolerance else "aspect_ratio_mismatch"

    def update(
        self,
        confirmed_tracks: Mapping[TrackId, Sequence[float]] | Iterable[object],
        frame_size: tuple[int, int],
        timestamp_seconds: float,
    ) -> dict[str, object]:
        """Consume currently confirmed tracker boxes and return a fresh snapshot."""

        timestamp = _finite_number(timestamp_seconds, "timestamp_seconds")
        if timestamp < 0.0:
            raise ValueError("timestamp_seconds cannot be negative.")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("timestamp_seconds must be monotonic within a classroom session.")
        normalized_frame_size = _frame_size(frame_size)
        tracks = _tracked_boxes(confirmed_tracks)
        self._last_timestamp = timestamp
        self._last_frame_size = normalized_frame_size
        self._last_tracks = tracks
        self._geometry_status = self._geometry_status_for_frame(normalized_frame_size)
        if self.seat_engine is not None and self._geometry_status == "compatible":
            self.seat_engine.update(tracks, normalized_frame_size, timestamp)
        self._last_statistics = self._build_statistics()
        return self.get_statistics()

    def consume_finalized(
        self,
        finalized_tracks: Iterable[object],
        *,
        timestamp_seconds: float | None = None,
    ) -> dict[str, object]:
        """Release finalized tracker IDs without treating a short loss as vacant."""

        values = tuple(finalized_tracks)
        finalized_ids = {_finalized_track_id(value) for value in values}
        if timestamp_seconds is not None:
            timestamp = _finite_number(timestamp_seconds, "timestamp_seconds")
            if timestamp < 0.0:
                raise ValueError("timestamp_seconds cannot be negative.")
            if self._last_timestamp is not None and timestamp < self._last_timestamp:
                raise ValueError("timestamp_seconds must not precede the latest classroom update.")
            self._last_timestamp = timestamp
        if self.seat_engine is not None:
            self.seat_engine.consume_finalized(values, timestamp_seconds=timestamp_seconds)
        self._last_tracks = {
            track_id: bbox for track_id, bbox in self._last_tracks.items() if track_id not in finalized_ids
        }
        self._last_statistics = self._build_statistics()
        return self.get_statistics()

    def get_statistics(self) -> dict[str, object]:
        """Return a defensive, JSON-friendly snapshot of the last update."""

        return _copy_json_mapping(self._last_statistics)

    def _seat_records(self) -> tuple[SeatDefinition, ...]:
        if self.config is None:
            return ()
        return self.config.seat_definitions

    def _seat_statistics(self) -> dict[str, object]:
        records = self._seat_records()
        capacity = None if self.config is None else self.config.capacity_summary
        if self.seat_engine is not None and self._geometry_status == "aspect_ratio_mismatch":
            total = None if capacity is None else capacity.total_seats
            enabled = None if capacity is None else capacity.enabled_seats
            disabled = None if capacity is None else capacity.disabled_seats
            return {
                "status": "aspect_ratio_mismatch",
                "geometry_configured": True,
                "assignment_enabled": False,
                "configured_seats": total,
                "enabled_seats": enabled,
                "disabled_seats": disabled,
                "occupied_seats": None,
                "pending_seats": None,
                "uncertain_seats": None,
                "vacant_seats": None,
                "utilization": None,
                "assignments": {},
                "seats": [],
                "lifetime": self.seat_engine.get_statistics()["lifetime"],
            }
        if self.seat_engine is not None:
            statistics = self.seat_engine.get_statistics()
            statistics["geometry_configured"] = True
            statistics["assignment_enabled"] = True
            return statistics

        if self.config is None or not records:
            status = "not_configured"
        else:
            status = "geometry_required"
        total = None if capacity is None else capacity.total_seats
        enabled = None if capacity is None else capacity.enabled_seats
        disabled = None if capacity is None else capacity.disabled_seats
        return {
            "status": status,
            "geometry_configured": False,
            "assignment_enabled": False,
            "configured_seats": total,
            "enabled_seats": enabled,
            "disabled_seats": disabled,
            "occupied_seats": None,
            "pending_seats": None,
            "uncertain_seats": None,
            "vacant_seats": None,
            "utilization": None,
            "assignments": {},
            "seats": [],
            "lifetime": {
                "confirmed_occupancy_events": 0,
                "vacated_events": 0,
                "completed_occupancy_seconds": 0.0,
                "finalized_track_count": 0,
            },
        }

    def _block_statistics(self, seat_statistics: Mapping[str, object]) -> dict[str, object]:
        records = self._seat_records()
        if not records:
            return {"status": "not_configured", "items": {}}

        state_by_seat = {
            str(item.get("seat_id")): str(item.get("status"))
            for item in seat_statistics.get("seats", [])
            if isinstance(item, Mapping) and item.get("seat_id") is not None
        }
        bucket_names: list[str] = []
        if self.config is not None and self.config.active_template is not None:
            bucket_names.extend(self.config.active_template.block_names)
        for record in records:
            if record.block not in bucket_names:
                bucket_names.append(record.block)
        items: dict[str, dict[str, object]] = {}
        has_live_status = self.seat_engine is not None and self._geometry_status == "compatible"
        for block_name in bucket_names:
            block_records = [record for record in records if record.block == block_name]
            enabled_records = [record for record in block_records if record.enabled]
            item: dict[str, object] = {
                "configured_seats": len(block_records),
                "enabled_seats": len(enabled_records),
                "disabled_seats": len(block_records) - len(enabled_records),
            }
            if not has_live_status:
                item.update(
                    {
                        "occupied_seats": None,
                        "pending_seats": None,
                        "uncertain_seats": None,
                        "vacant_seats": None,
                        "utilization": None,
                    }
                )
            else:
                counts = {"occupied": 0, "pending": 0, "uncertain": 0, "vacant": 0}
                for record in enabled_records:
                    status = state_by_seat.get(record.seat_id, "vacant")
                    if status in counts:
                        counts[status] += 1
                utilization, _percentage = _rounded_ratio(counts["occupied"], len(enabled_records))
                item.update(
                    {
                        "occupied_seats": counts["occupied"],
                        "pending_seats": counts["pending"],
                        "uncertain_seats": counts["uncertain"],
                        "vacant_seats": counts["vacant"],
                        "utilization": utilization,
                    }
                )
            items[block_name] = item
        return {
            "status": (
                "configured"
                if has_live_status
                else "aspect_ratio_mismatch"
                if self._geometry_status == "aspect_ratio_mismatch"
                else "geometry_required"
            ),
            "items": items,
        }

    def _row_statistics(self, seat_statistics: Mapping[str, object]) -> dict[str, object]:
        """Expose row roll-ups alongside blocks for a classroom dashboard."""

        records = self._seat_records()
        if not records:
            return {"status": "not_configured", "items": {}}
        state_by_seat = {
            str(item.get("seat_id")): str(item.get("status"))
            for item in seat_statistics.get("seats", [])
            if isinstance(item, Mapping) and item.get("seat_id") is not None
        }
        items: dict[str, dict[str, object]] = {}
        for row in sorted({record.row for record in records}):
            row_records = [record for record in records if record.row == row]
            enabled_records = [record for record in row_records if record.enabled]
            item: dict[str, object] = {
                "configured_seats": len(row_records),
                "enabled_seats": len(enabled_records),
                "disabled_seats": len(row_records) - len(enabled_records),
            }
            if self.seat_engine is None or self._geometry_status != "compatible":
                item.update(
                    {
                        "occupied_seats": None,
                        "pending_seats": None,
                        "uncertain_seats": None,
                        "vacant_seats": None,
                        "utilization": None,
                    }
                )
            else:
                counts = {"occupied": 0, "pending": 0, "uncertain": 0, "vacant": 0}
                for record in enabled_records:
                    status = state_by_seat.get(record.seat_id, "vacant")
                    if status in counts:
                        counts[status] += 1
                utilization, _percentage = _rounded_ratio(counts["occupied"], len(enabled_records))
                item.update(
                    {
                        "occupied_seats": counts["occupied"],
                        "pending_seats": counts["pending"],
                        "uncertain_seats": counts["uncertain"],
                        "vacant_seats": counts["vacant"],
                        "utilization": utilization,
                    }
                )
            items[str(row)] = item
        return {
            "status": (
                "configured"
                if self.seat_engine is not None and self._geometry_status == "compatible"
                else "aspect_ratio_mismatch"
                if self._geometry_status == "aspect_ratio_mismatch"
                else "geometry_required"
            ),
            "items": items,
        }

    def _aisle_definitions(self) -> tuple[dict[str, object], ...]:
        if self.config is None or self.config.active_template is None or self.config.session_layout is None:
            return ()
        template = self.config.active_template
        grid = template.reference_grid
        if grid is None or len(template.blocks) < 2 or grid.block_gap_units <= 0.0:
            return ()
        total_units = float(template.column_count) + grid.block_gap_units * (len(template.blocks) - 1)
        cursor = 0.0
        aisles: list[dict[str, object]] = []
        for index, left_block in enumerate(template.blocks[:-1]):
            cursor += left_block.columns
            u_start = cursor / total_units
            u_end = (cursor + grid.block_gap_units) / total_units
            polygon = (
                grid.project(u_start, 0.0),
                grid.project(u_end, 0.0),
                grid.project(u_end, 1.0),
                grid.project(u_start, 1.0),
            )
            right_block = template.blocks[index + 1]
            aisles.append(
                {
                    "name": f"{left_block.name}_to_{right_block.name}",
                    "left_block": left_block.name,
                    "right_block": right_block.name,
                    "reference_polygon": polygon,
                }
            )
            cursor += grid.block_gap_units
        return tuple(aisles)

    def _aisle_statistics(self, tracks: Mapping[TrackId, BBox]) -> dict[str, object]:
        definitions = self._aisle_definitions()
        if self.config is None:
            return {"status": "not_configured", "items": [], "total_current_people": None}
        if not definitions:
            return {"status": "not_configured", "items": [], "total_current_people": 0}
        if self._geometry_status == "aspect_ratio_mismatch":
            return {
                "status": "aspect_ratio_mismatch",
                "measurement_status": "reference_geometry_rejected",
                "minimum_required_width_m": self.config.room_profile.layout_rules.minimum_aisle_width_m,
                "items": [],
                "total_current_people": None,
            }

        reference_size = self.config.room_profile.reference_resolution
        points = {track_id: _foot_point(bbox) for track_id, bbox in tracks.items()}
        items: list[dict[str, object]] = []
        for definition in definitions:
            reference_polygon = definition["reference_polygon"]
            assert isinstance(reference_polygon, tuple)
            polygon = _scale_polygon(
                reference_polygon,
                frame_size=self._last_frame_size,
                reference_size=reference_size,
            )
            matches = sorted(
                (track_id for track_id, point in points.items() if _contains_point(point, polygon)),
                key=str,
            )
            items.append(
                {
                    "name": definition["name"],
                    "left_block": definition["left_block"],
                    "right_block": definition["right_block"],
                    "current_people": len(matches),
                    "track_ids": list(matches),
                    "reference_polygon": [list(point) for point in reference_polygon],
                    "polygon": [list(point) for point in polygon],
                }
            )
        return {
            "status": "configured",
            "measurement_status": "image_geometry_only",
            "minimum_required_width_m": self.config.room_profile.layout_rules.minimum_aisle_width_m,
            "items": items,
            "total_current_people": sum(int(item["current_people"]) for item in items),
        }

    def _entrance_statistics(self, tracks: Mapping[TrackId, BBox]) -> dict[str, object]:
        if self.config is None:
            return {
                "status": "not_configured",
                "door_roi_configured": False,
                "counting_line_configured": False,
                "current_people_in_door_roi": None,
                "track_ids": [],
            }
        profile = self.config.room_profile
        reference_size = profile.reference_resolution
        if self._geometry_status == "aspect_ratio_mismatch":
            return {
                "status": "aspect_ratio_mismatch",
                "door_roi_configured": profile.door_roi is not None,
                "counting_line_configured": profile.counting_line is not None,
                "current_people_in_door_roi": None,
                "track_ids": [],
                "door_roi": None,
                "counting_line": None,
                "event_counting": "provided_by_line_crossing_counter",
            }
        door_polygon = (
            None
            if profile.door_roi is None
            else _scale_polygon(profile.door_roi, frame_size=self._last_frame_size, reference_size=reference_size)
        )
        matches = [] if door_polygon is None else sorted(
            (track_id for track_id, bbox in tracks.items() if _contains_point(_foot_point(bbox), door_polygon)),
            key=str,
        )
        line = profile.counting_line
        return {
            "status": "configured" if door_polygon is not None or line is not None else "not_configured",
            "door_roi_configured": door_polygon is not None,
            "counting_line_configured": line is not None,
            "current_people_in_door_roi": None if door_polygon is None else len(matches),
            "track_ids": list(matches),
            "door_roi": None if door_polygon is None else [list(point) for point in door_polygon],
            "counting_line": (
                None
                if line is None
                else {
                    "p1": list(_scale_polygon((line.p1,), frame_size=self._last_frame_size, reference_size=reference_size)[0]),
                    "p2": list(_scale_polygon((line.p2,), frame_size=self._last_frame_size, reference_size=reference_size)[0]),
                }
            ),
            "event_counting": "provided_by_line_crossing_counter",
        }

    def _room_statistics(self, tracks: Mapping[TrackId, BBox]) -> dict[str, object]:
        if self.config is None:
            return {
                "name": None,
                "confirmed_active_tracks": 0,
                "current_people": None,
                "outside_room_people": None,
                "maximum_capacity": None,
                "occupancy_rate": None,
                "occupancy_percentage": None,
                "over_capacity": False,
                "visible_floor_area_m2": None,
                "people_per_m2": None,
                "room_boundary_configured": False,
                "physical_density_calibrated": False,
            }
        profile = self.config.room_profile
        geometry_usable = self._geometry_status != "aspect_ratio_mismatch"
        boundary = (
            None
            if profile.room_boundary is None or not geometry_usable
            else _scale_polygon(
                profile.room_boundary,
                frame_size=self._last_frame_size,
                reference_size=profile.reference_resolution,
            )
        )
        if profile.room_boundary is not None and not geometry_usable:
            current_track_ids: set[TrackId] | None = None
        elif boundary is None:
            current_track_ids = set(tracks)
        else:
            current_track_ids = {
                track_id for track_id, bbox in tracks.items() if _contains_point(_foot_point(bbox), boundary)
            }
        current_people = None if current_track_ids is None else len(current_track_ids)
        maximum_capacity = profile.maximum_capacity
        occupancy_rate, occupancy_percentage = (
            (None, None)
            if current_people is None
            else _rounded_ratio(current_people, maximum_capacity)
        )
        area_m2 = profile.visible_floor_area_m2
        people_per_m2 = (
            round(current_people / area_m2, 3)
            if area_m2 is not None and current_people is not None and geometry_usable
            else None
        )
        return {
            "name": profile.name,
            "confirmed_active_tracks": len(tracks),
            "current_people": current_people,
            "current_track_ids": [] if current_track_ids is None else sorted(current_track_ids, key=str),
            "outside_room_people": None if current_people is None else len(tracks) - current_people,
            "maximum_capacity": maximum_capacity,
            "occupancy_rate": occupancy_rate,
            "occupancy_percentage": occupancy_percentage,
            "over_capacity": bool(occupancy_rate is not None and occupancy_rate > 1.0),
            "visible_floor_area_m2": area_m2,
            "people_per_m2": people_per_m2,
            "room_boundary_configured": profile.room_boundary is not None,
            "room_boundary": None if boundary is None else [list(point) for point in boundary],
            "physical_density_calibrated": area_m2 is not None and geometry_usable,
            "camera_calibration_configured": profile.calibration is not None,
        }

    def _layout_statistics(self) -> dict[str, object]:
        if self.config is None:
            return {
                "status": "not_configured",
                "template": None,
                "rows": None,
                "rules": None,
                "reference_geometry_configured": False,
            }
        profile = self.config.room_profile
        template = self.config.active_template
        session = self.config.session_layout
        return {
            "status": "configured" if session is not None else "not_configured",
            "template": None if template is None else template.name,
            "rows": None if session is None else session.rows,
            "blocks": [] if template is None else list(template.block_names),
            "reference_geometry_configured": bool(template is not None and template.reference_grid is not None),
            "reference_resolution": None if profile.reference_resolution is None else list(profile.reference_resolution),
            "rules": profile.layout_rules.to_mapping(),
            "capacity": self.config.capacity_summary.to_mapping(),
        }

    def _build_statistics(self) -> dict[str, object]:
        seats = self._seat_statistics()
        if self.config is None:
            status = "not_configured"
        elif self._geometry_status == "aspect_ratio_mismatch":
            status = "aspect_ratio_mismatch"
        elif self.seat_engine is not None:
            status = "ready"
        elif self.config.session_layout is None:
            status = "room_configured"
        else:
            status = "geometry_required"
        return {
            "enabled": self.config is not None,
            "status": status,
            "geometry": {
                "status": self._geometry_status,
                "reference_resolution": (
                    None
                    if self.config is None or self.config.room_profile.reference_resolution is None
                    else list(self.config.room_profile.reference_resolution)
                ),
                "frame_size": None if self._last_frame_size is None else list(self._last_frame_size),
                "aspect_ratio_tolerance": self.aspect_ratio_tolerance,
            },
            "room": self._room_statistics(self._last_tracks),
            "layout": self._layout_statistics(),
            "seats": seats,
            "blocks": self._block_statistics(seats),
            "rows": self._row_statistics(seats),
            "aisles": self._aisle_statistics(self._last_tracks),
            "entrance": self._entrance_statistics(self._last_tracks),
        }


def _copy_json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Return a small recursive copy without depending on JSON serialisation."""

    def copy_item(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): copy_item(nested) for key, nested in item.items()}
        if isinstance(item, list):
            return [copy_item(nested) for nested in item]
        if isinstance(item, tuple):
            return [copy_item(nested) for nested in item]
        if isinstance(item, set):
            return [copy_item(nested) for nested in sorted(item, key=str)]
        return item

    return {str(key): copy_item(item) for key, item in value.items()}


__all__ = ["BBox", "ClassroomAnalytics", "TrackId"]
