"""Optional, stateful seat-occupancy analytics for a calibrated classroom.

The tracker remains the source of truth for people.  This module only maps
*confirmed* tracked person boxes to a configured set of seats.  It deliberately
does not depend on a particular classroom-config implementation: callers may
pass :class:`SeatDefinition`, mappings, or small objects that expose
``seat_id``/``polygon`` attributes.

Seat occupancy is intentionally conservative:

* an anchor is taken below the centre of a person box, rather than using the
  floor foot point which is often hidden by a desk;
* a candidate must remain associated for ``occupancy_confirm_seconds`` before
  it becomes occupied;
* an occupied seat remains reserved for ``vacancy_grace_seconds`` after a
  short tracker loss; and
* conflicting new assignments are reported as ``uncertain`` rather than
  claiming that a seat contains multiple people.

Coordinates in seat polygons are reference-frame pixels when ``reference_size``
is configured.  Otherwise they are interpreted in the supplied frame directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Any, TypeAlias


Point: TypeAlias = tuple[float, float]
BBox: TypeAlias = tuple[float, float, float, float]
TrackId: TypeAlias = int | str


def _finite_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{path} must be a finite number.")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{path} must be a finite number.")
    return result


def _non_empty_id(value: object, path: str) -> str:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{path} must be a non-empty identifier.")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{path} must be a non-empty identifier.")
    return result


def _point(value: object, path: str) -> Point:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{path} must be [x, y].")
    return (_finite_number(value[0], f"{path}[0]"), _finite_number(value[1], f"{path}[1]"))


def _polygon_signed_area(polygon: Sequence[Point]) -> float:
    return sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    ) / 2.0


def _polygon(value: object, path: str) -> tuple[Point, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a sequence of [x, y] points.")
    result = tuple(_point(point, f"{path}[{index}]") for index, point in enumerate(value))
    if len(result) >= 2 and result[0] == result[-1]:
        result = result[:-1]
    if len(result) < 3 or len(set(result)) < 3:
        raise ValueError(f"{path} must contain at least three distinct points.")
    if abs(_polygon_signed_area(result)) <= 1e-9:
        raise ValueError(f"{path} must enclose a non-zero area.")
    return result


def _point_on_segment(point: Point, first: Point, second: Point) -> bool:
    cross = (point[1] - first[1]) * (second[0] - first[0]) - (point[0] - first[0]) * (second[1] - first[1])
    if abs(cross) > 1e-9:
        return False
    return (
        min(first[0], second[0]) - 1e-9 <= point[0] <= max(first[0], second[0]) + 1e-9
        and min(first[1], second[1]) - 1e-9 <= point[1] <= max(first[1], second[1]) + 1e-9
    )


def _contains_point(point: Point, polygon: Sequence[Point]) -> bool:
    """Return true for points inside *or on the edge* of a polygon."""
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        crosses = (current[1] > point[1]) != (previous[1] > point[1])
        if crosses:
            intercept = (previous[0] - current[0]) * (point[1] - current[1]) / (previous[1] - current[1]) + current[0]
            if point[0] < intercept:
                inside = not inside
        previous = current
    return inside


def _centroid(polygon: Sequence[Point]) -> Point:
    """A stable centroid used only as a deterministic tie breaker."""
    signed_area = _polygon_signed_area(polygon)
    if abs(signed_area) <= 1e-9:
        return (
            sum(point[0] for point in polygon) / len(polygon),
            sum(point[1] for point in polygon) / len(polygon),
        )
    factor = 1.0 / (6.0 * signed_area)
    x = 0.0
    y = 0.0
    for index, current in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        cross = current[0] * following[1] - following[0] * current[1]
        x += (current[0] + following[0]) * cross
        y += (current[1] + following[1]) * cross
    return (x * factor, y * factor)


@dataclass(frozen=True)
class SeatDefinition:
    """A single seat polygon and optional presentation metadata.

    ``metadata`` can carry layout fields such as ``row``, ``block`` and
    ``column``.  The occupancy engine does not interpret these values.
    """

    seat_id: str
    polygon: tuple[Point, ...]
    enabled: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "seat_id", _non_empty_id(self.seat_id, "seat_id"))
        object.__setattr__(self, "polygon", _polygon(self.polygon, f"seat {self.seat_id}.polygon"))
        if not isinstance(self.enabled, bool):
            raise ValueError("seat.enabled must be a boolean.")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("seat.metadata must be a mapping.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_value(cls, value: "SeatDefinition | Mapping[str, object] | object") -> "SeatDefinition":
        """Coerce a small config object or mapping into a seat definition."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            raw_id = value.get("seat_id", value.get("id", value.get("name")))
            polygon = value.get("polygon")
            enabled = value.get("enabled", True)
            raw_metadata = value.get("metadata", {})
            if not isinstance(raw_metadata, Mapping):
                raise ValueError("seat.metadata must be a mapping.")
            metadata = dict(raw_metadata)
            for key in ("row", "block", "column", "label"):
                if key in value:
                    metadata.setdefault(key, value[key])
        else:
            raw_id = getattr(value, "seat_id", getattr(value, "id", getattr(value, "name", None)))
            polygon = getattr(value, "polygon", None)
            enabled = getattr(value, "enabled", True)
            raw_metadata = getattr(value, "metadata", {})
            if not isinstance(raw_metadata, Mapping):
                raise ValueError("seat.metadata must be a mapping.")
            metadata = dict(raw_metadata)
            for key in ("row", "block", "column", "label"):
                if hasattr(value, key):
                    metadata.setdefault(key, getattr(value, key))
        return cls(
            seat_id=_non_empty_id(raw_id, "seat.seat_id"),
            polygon=_polygon(polygon, "seat.polygon"),
            enabled=enabled,
            metadata=metadata,
        )


@dataclass
class _SeatRuntime:
    active_track_id: TrackId | None = None
    candidate_since: float | None = None
    occupied_since: float | None = None
    last_seen_at: float | None = None
    last_anchor: Point | None = None

    @property
    def is_occupied(self) -> bool:
        return self.occupied_since is not None

    @property
    def is_pending(self) -> bool:
        return self.active_track_id is not None and self.candidate_since is not None and not self.is_occupied


class SeatOccupancyEngine:
    """Map tracked person boxes to seats with conservative temporal smoothing.

    Args:
        seats: Generic seat definitions.  A mapping needs ``seat_id`` (or
            ``id``/``name``) and ``polygon``.  An object can expose equivalent
            attributes.
        reference_size: Reference resolution of seat polygons.  ``None`` means
            polygons are already expressed in the incoming frame resolution.
        anchor_ratio: Vertical location of the body anchor in a person bbox.
            ``0.65`` means 65% from the top edge toward the bottom edge.
        occupancy_confirm_seconds: Continuous candidate time required before a
            seat becomes occupied.  Set to zero for immediate confirmation.
        vacancy_grace_seconds: How long an occupied seat remains reserved after
            its tracked occupant is not observed.
    """

    def __init__(
        self,
        seats: Iterable[SeatDefinition | Mapping[str, object] | object],
        *,
        reference_size: tuple[int, int] | None = None,
        anchor_ratio: float = 0.65,
        occupancy_confirm_seconds: float = 1.5,
        vacancy_grace_seconds: float = 2.0,
    ) -> None:
        self.seats = tuple(SeatDefinition.from_value(seat) for seat in seats)
        seat_ids = [seat.seat_id for seat in self.seats]
        if len(set(seat_ids)) != len(seat_ids):
            raise ValueError("seat definitions contain duplicate seat_id values.")
        if reference_size is not None:
            if len(reference_size) != 2 or reference_size[0] <= 0 or reference_size[1] <= 0:
                raise ValueError("reference_size must contain two positive dimensions.")
            self.reference_size = (int(reference_size[0]), int(reference_size[1]))
        else:
            self.reference_size = None
        self.anchor_ratio = _finite_number(anchor_ratio, "anchor_ratio")
        if not 0.0 <= self.anchor_ratio <= 1.0:
            raise ValueError("anchor_ratio must be within [0, 1].")
        self.occupancy_confirm_seconds = _finite_number(occupancy_confirm_seconds, "occupancy_confirm_seconds")
        self.vacancy_grace_seconds = _finite_number(vacancy_grace_seconds, "vacancy_grace_seconds")
        if self.occupancy_confirm_seconds < 0.0:
            raise ValueError("occupancy_confirm_seconds cannot be negative.")
        if self.vacancy_grace_seconds < 0.0:
            raise ValueError("vacancy_grace_seconds cannot be negative.")
        self._seats_by_id = {seat.seat_id: seat for seat in self.seats}
        self.reset()

    def reset(self) -> None:
        """Discard all session assignments and finalized occupancy statistics."""
        self._runtime = {seat.seat_id: _SeatRuntime() for seat in self.seats}
        self._last_timestamp: float | None = None
        self._last_scaled_polygons = {seat.seat_id: seat.polygon for seat in self.seats}
        self._visible_track_ids: set[TrackId] = set()
        self._ambiguous_seat_ids: set[str] = set()
        self._confirmed_occupancy_events = 0
        self._vacated_events = 0
        self._completed_occupancy_seconds = 0.0
        self._finalized_track_count = 0

    @staticmethod
    def anchor_for_bbox(bbox: Sequence[float], *, anchor_ratio: float = 0.65) -> Point:
        """Return the classroom seat anchor for a standard ``[x1,y1,x2,y2]`` box."""
        ratio = _finite_number(anchor_ratio, "anchor_ratio")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("anchor_ratio must be within [0, 1].")
        x1, y1, x2, y2 = _bbox(bbox, "bbox")
        return ((x1 + x2) / 2.0, y1 + ratio * (y2 - y1))

    def _scaled_polygons(self, frame_size: tuple[int, int] | None) -> dict[str, tuple[Point, ...]]:
        if self.reference_size is None:
            return {seat.seat_id: seat.polygon for seat in self.seats}
        if frame_size is None:
            raise ValueError("frame_size is required when reference_size is configured.")
        width, height = _frame_size(frame_size)
        reference_width, reference_height = self.reference_size
        return {
            seat.seat_id: tuple(
                (point[0] * width / reference_width, point[1] * height / reference_height)
                for point in seat.polygon
            )
            for seat in self.seats
        }

    def _release(self, seat_id: str, timestamp_seconds: float) -> None:
        runtime = self._runtime[seat_id]
        if runtime.is_occupied:
            self._vacated_events += 1
            occupied_since = runtime.occupied_since
            if occupied_since is not None:
                self._completed_occupancy_seconds += max(0.0, timestamp_seconds - occupied_since)
        self._runtime[seat_id] = _SeatRuntime()

    def _confirm(self, seat_id: str, timestamp_seconds: float) -> None:
        runtime = self._runtime[seat_id]
        if runtime.is_occupied or runtime.candidate_since is None:
            return
        if timestamp_seconds - runtime.candidate_since + 1e-9 >= self.occupancy_confirm_seconds:
            runtime.occupied_since = timestamp_seconds
            self._confirmed_occupancy_events += 1

    def _claim(self, seat_id: str, track_id: TrackId, anchor: Point, timestamp_seconds: float) -> None:
        runtime = self._runtime[seat_id]
        runtime.active_track_id = track_id
        runtime.candidate_since = timestamp_seconds
        runtime.last_seen_at = timestamp_seconds
        runtime.last_anchor = anchor
        self._confirm(seat_id, timestamp_seconds)

    def _update_existing_assignments(
        self,
        anchors: Mapping[TrackId, Point],
        polygons: Mapping[str, Sequence[Point]],
        timestamp_seconds: float,
    ) -> set[TrackId]:
        """Refresh sticky assignments and return tracks that remain reserved."""
        owned_tracks: set[TrackId] = set()
        for seat in self.seats:
            runtime = self._runtime[seat.seat_id]
            track_id = runtime.active_track_id
            if track_id is None:
                if runtime.is_occupied and runtime.last_seen_at is not None and (
                    timestamp_seconds - runtime.last_seen_at > self.vacancy_grace_seconds
                ):
                    self._release(seat.seat_id, timestamp_seconds)
                continue

            anchor = anchors.get(track_id)
            if anchor is not None and _contains_point(anchor, polygons[seat.seat_id]):
                runtime.last_seen_at = timestamp_seconds
                runtime.last_anchor = anchor
                owned_tracks.add(track_id)
                self._confirm(seat.seat_id, timestamp_seconds)
                continue

            if runtime.is_pending:
                # A provisional claim has no established occupancy to smooth;
                # clear it immediately so the track can try another seat.
                self._runtime[seat.seat_id] = _SeatRuntime()
                continue

            # The track is visible but moved outside this seat.  Keep the old
            # occupied seat inside the grace period, yet allow the track to be
            # evaluated against a new seat on this frame.
            if anchor is not None:
                runtime.active_track_id = None
                runtime.candidate_since = None
            if runtime.last_seen_at is not None and timestamp_seconds - runtime.last_seen_at > self.vacancy_grace_seconds:
                self._release(seat.seat_id, timestamp_seconds)
        return owned_tracks

    @staticmethod
    def _track_identifier(value: object, path: str = "track_id") -> TrackId:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError(f"{path} must be an int or non-empty string.")
        if isinstance(value, str):
            result = value.strip()
            if not result:
                raise ValueError(f"{path} must be an int or non-empty string.")
            return result
        return int(value)

    def update(
        self,
        tracked_persons: Mapping[TrackId, Sequence[float]] | Iterable[object],
        frame_size: tuple[int, int] | None = None,
        timestamp_seconds: float = 0.0,
    ) -> dict[str, object]:
        """Update seat assignments from the current confirmed tracker boxes.

        ``tracked_persons`` normally has the same shape as ``SpatialEngine``:
        ``{track_id: (x1, y1, x2, y2)}``.  For API adapters an iterable of
        records with ``track_id`` and ``bbox`` is accepted as well.
        """
        timestamp = _timestamp(timestamp_seconds)
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("timestamp_seconds must be monotonic within a seat-occupancy session.")
        if frame_size is not None:
            _frame_size(frame_size)
        polygons = self._scaled_polygons(frame_size)
        tracks = _tracked_boxes(tracked_persons)
        anchors = {
            track_id: self.anchor_for_bbox(bbox, anchor_ratio=self.anchor_ratio)
            for track_id, bbox in tracks.items()
        }
        self._last_timestamp = timestamp
        self._last_scaled_polygons = polygons
        self._visible_track_ids = set(anchors)
        self._ambiguous_seat_ids.clear()

        owned_tracks = self._update_existing_assignments(anchors, polygons, timestamp)
        available_seats = {
            seat.seat_id
            for seat in self.seats
            if seat.enabled
            and not self._runtime[seat.seat_id].is_occupied
            and not self._runtime[seat.seat_id].is_pending
        }

        # A person can geometrically overlap a neighbouring polygon near a
        # boundary.  Select its closest seat-centre deterministically before
        # checking collisions between people.
        claims: dict[str, list[tuple[TrackId, Point]]] = {}
        for track_id, anchor in anchors.items():
            if track_id in owned_tracks:
                continue
            candidates = [
                seat_id
                for seat_id in available_seats
                if _contains_point(anchor, polygons[seat_id])
            ]
            if not candidates:
                continue
            selected = min(
                candidates,
                key=lambda seat_id: (
                    (anchor[0] - _centroid(polygons[seat_id])[0]) ** 2
                    + (anchor[1] - _centroid(polygons[seat_id])[1]) ** 2,
                    seat_id,
                ),
            )
            claims.setdefault(selected, []).append((track_id, anchor))

        for seat_id, contenders in claims.items():
            if len(contenders) != 1:
                # Do not pick a person arbitrarily when a new observation
                # cannot distinguish two people inside one seat polygon.
                self._ambiguous_seat_ids.add(seat_id)
                continue
            track_id, anchor = contenders[0]
            self._claim(seat_id, track_id, anchor, timestamp)

        return self.get_statistics()

    def finalize_tracks(
        self,
        track_ids: Iterable[TrackId | object],
        *,
        timestamp_seconds: float | None = None,
    ) -> dict[str, object]:
        """Finalize lost tracks and record completed seat occupancy durations.

        This is intended for the stream state's finalized tracker summaries.
        It releases a confirmed seat immediately because its tracker has
        already exhausted its own lost-track buffer.
        """
        if timestamp_seconds is None:
            timestamp = self._last_timestamp if self._last_timestamp is not None else 0.0
        else:
            timestamp = _timestamp(timestamp_seconds)
            if self._last_timestamp is not None and timestamp < self._last_timestamp:
                raise ValueError("timestamp_seconds must not precede the latest update.")
            self._last_timestamp = timestamp
        normalized_ids = {_finalized_track_id(value) for value in track_ids}
        self._finalized_track_count += len(normalized_ids)
        for seat in self.seats:
            runtime = self._runtime[seat.seat_id]
            if runtime.active_track_id not in normalized_ids:
                continue
            if runtime.is_occupied:
                self._release(seat.seat_id, timestamp)
            else:
                self._runtime[seat.seat_id] = _SeatRuntime()
        return self.get_statistics()

    def consume_finalized(
        self,
        finalized_tracks: Iterable[object],
        *,
        timestamp_seconds: float | None = None,
    ) -> dict[str, object]:
        """Compatibility alias accepting objects with a ``track_id`` attribute."""
        return self.finalize_tracks(finalized_tracks, timestamp_seconds=timestamp_seconds)

    def get_statistics(self) -> dict[str, object]:
        """Return a JSON-friendly snapshot without mutating the engine."""
        seat_views: list[dict[str, object]] = []
        assignments: dict[str, str] = {}
        occupied = pending = uncertain = 0
        enabled = 0
        for seat in self.seats:
            runtime = self._runtime[seat.seat_id]
            if not seat.enabled:
                status = "disabled"
            elif runtime.is_occupied:
                status = "occupied"
                occupied += 1
            elif runtime.is_pending:
                status = "pending"
                pending += 1
            elif seat.seat_id in self._ambiguous_seat_ids:
                status = "uncertain"
                uncertain += 1
            else:
                status = "vacant"
            if seat.enabled:
                enabled += 1
            track_id = runtime.active_track_id
            if track_id is not None and status in {"occupied", "pending"}:
                assignments[str(track_id)] = seat.seat_id
            seat_views.append(
                {
                    "seat_id": seat.seat_id,
                    "enabled": seat.enabled,
                    "status": status,
                    "track_id": track_id,
                    "visible": track_id in self._visible_track_ids if track_id is not None else False,
                    "anchor": list(runtime.last_anchor) if runtime.last_anchor is not None else None,
                    "metadata": dict(seat.metadata),
                }
            )
        vacant = enabled - occupied - pending - uncertain
        return {
            "status": "configured" if self.seats else "not_configured",
            "configured_seats": len(self.seats),
            "enabled_seats": enabled,
            "disabled_seats": len(self.seats) - enabled,
            "occupied_seats": occupied,
            "pending_seats": pending,
            "uncertain_seats": uncertain,
            "vacant_seats": vacant,
            "utilization": round(occupied / enabled, 4) if enabled else None,
            "anchor_ratio": self.anchor_ratio,
            "occupancy_confirm_seconds": self.occupancy_confirm_seconds,
            "vacancy_grace_seconds": self.vacancy_grace_seconds,
            "assignments": assignments,
            "seats": seat_views,
            "lifetime": {
                "confirmed_occupancy_events": self._confirmed_occupancy_events,
                "vacated_events": self._vacated_events,
                "completed_occupancy_seconds": round(self._completed_occupancy_seconds, 3),
                "finalized_track_count": self._finalized_track_count,
            },
        }


def _frame_size(value: object) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError("frame_size must contain [width, height].")
    width = _finite_number(value[0], "frame_size[0]")
    height = _finite_number(value[1], "frame_size[1]")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("frame_size must contain positive dimensions.")
    return (int(width), int(height))


def _bbox(value: object, path: str) -> BBox:
    if isinstance(value, Mapping):
        if "bbox" in value:
            return _bbox(value["bbox"], f"{path}.bbox")
        try:
            value = (value["x1"], value["y1"], value["x2"], value["y2"])
        except KeyError as error:
            raise ValueError(f"{path} must be [x1, y1, x2, y2] or a bbox mapping.") from error
    elif not isinstance(value, (str, bytes)) and hasattr(value, "bbox"):
        return _bbox(getattr(value, "bbox"), f"{path}.bbox")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError(f"{path} must be [x1, y1, x2, y2].")
    x1, y1, x2, y2 = (_finite_number(item, f"{path}[{index}]") for index, item in enumerate(value))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{path} must have x2 > x1 and y2 > y1.")
    return (x1, y1, x2, y2)


def _tracked_boxes(value: Mapping[TrackId, Sequence[float]] | Iterable[object]) -> dict[TrackId, BBox]:
    if isinstance(value, Mapping):
        if "track_id" in value and ("bbox" in value or {"x1", "y1", "x2", "y2"}.issubset(value)):
            raw_items: Iterable[object] = (value,)
        else:
            raw_items = tuple(value.items())
    else:
        raw_items = value
    result: dict[TrackId, BBox] = {}
    for index, item in enumerate(raw_items):
        if isinstance(item, Mapping):
            track_id = SeatOccupancyEngine._track_identifier(item.get("track_id", item.get("id")), f"tracks[{index}].track_id")
            bbox = _bbox(item, f"tracks[{index}]")
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            track_id = SeatOccupancyEngine._track_identifier(item[0], f"tracks[{index}][0]")
            bbox = _bbox(item[1], f"tracks[{index}][1]")
        else:
            track_id = SeatOccupancyEngine._track_identifier(
                getattr(item, "track_id", getattr(item, "id", None)), f"tracks[{index}].track_id"
            )
            bbox = _bbox(item, f"tracks[{index}]")
        if track_id in result:
            raise ValueError(f"tracked_persons contains duplicate track_id {track_id!r}.")
        result[track_id] = bbox
    return result


def _timestamp(value: object) -> float:
    result = _finite_number(value, "timestamp_seconds")
    if result < 0.0:
        raise ValueError("timestamp_seconds cannot be negative.")
    return result


def _finalized_track_id(value: object) -> TrackId:
    if isinstance(value, (int, str)) and not isinstance(value, bool):
        return SeatOccupancyEngine._track_identifier(value)
    if isinstance(value, Mapping):
        return SeatOccupancyEngine._track_identifier(value.get("track_id", value.get("id")))
    return SeatOccupancyEngine._track_identifier(getattr(value, "track_id", getattr(value, "id", None)))


__all__ = ["BBox", "Point", "SeatDefinition", "SeatOccupancyEngine", "TrackId"]
