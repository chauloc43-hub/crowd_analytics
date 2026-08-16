"""Validated, serialisable contract for a single camera's physical space.

The current stream analytics configuration predates physical-space metrics and
uses ``analytics.zones`` plus one ``analytics.counting_line``.  This module is
deliberately independent from the runtime so a future dashboard or calibration
UI can consume a single, validated representation without changing the live
pipeline first.

``parse_space_config`` accepts both the new ``space`` block and the legacy
analytics fields.  A legacy input without a name is assigned ``default_space``.
Coordinates are reference-frame pixels, not world coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any, TypeAlias

import yaml


Point: TypeAlias = tuple[float, float]


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a YAML mapping.")
    return value


def _sequence(value: object, path: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a YAML list.")
    return value


def _name(value: object, path: str) -> str:
    if not isinstance(value, str) or not (result := value.strip()):
        raise ValueError(f"{path} must be a non-empty string.")
    return result


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{path} must be a positive integer.")
    return int(value)


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{path} must be an integer.")
    return int(value)


def _positive_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{path} must be a positive number.")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{path} must be a finite positive number.")
    return result


def _coordinate(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{path} must be a finite non-negative number.")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{path} must be a finite non-negative number.")
    return result


def _point(value: object, path: str) -> Point:
    values = _sequence(value, path)
    if len(values) != 2:
        raise ValueError(f"{path} must contain exactly [x, y].")
    return (_coordinate(values[0], f"{path}[0]"), _coordinate(values[1], f"{path}[1]"))


def _polygon_signed_area(polygon: Sequence[Point]) -> float:
    return sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    ) / 2.0


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    """Whether b lies on the closed segment a--c, assuming collinearity."""
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Return true for proper or touching segment intersections."""
    first_a = _orientation(a, b, c)
    first_b = _orientation(a, b, d)
    second_a = _orientation(c, d, a)
    second_b = _orientation(c, d, b)
    epsilon = 1e-9

    if (first_a > epsilon) != (first_b > epsilon) and (second_a > epsilon) != (second_b > epsilon):
        return True
    return (
        (abs(first_a) <= epsilon and _on_segment(a, c, b))
        or (abs(first_b) <= epsilon and _on_segment(a, d, b))
        or (abs(second_a) <= epsilon and _on_segment(c, a, d))
        or (abs(second_b) <= epsilon and _on_segment(c, b, d))
    )


def _polygon(value: object, path: str) -> tuple[Point, ...]:
    raw_points = _sequence(value, path)
    points = tuple(_point(point, f"{path}[{index}]") for index, point in enumerate(raw_points))
    # A repeated first vertex is conventional in some GIS/YAML exports.  Store
    # the canonical open polygon because ZoneManager closes it internally.
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError(f"{path} must contain at least three distinct vertices.")
    if len(set(points)) < 3:
        raise ValueError(f"{path} must contain at least three distinct vertices.")
    for index, point in enumerate(points):
        if point == points[(index + 1) % len(points)]:
            raise ValueError(f"{path} cannot contain consecutive duplicate vertices.")
    if abs(_polygon_signed_area(points)) <= 1e-9:
        raise ValueError(f"{path} must enclose a non-zero area.")

    edge_count = len(points)
    for first_index in range(edge_count):
        a = points[first_index]
        b = points[(first_index + 1) % edge_count]
        for second_index in range(first_index + 1, edge_count):
            # Adjacent edges share one endpoint by design; only test genuinely
            # non-adjacent pairs, including the first/last edge pair.
            if second_index in {first_index, (first_index + 1) % edge_count}:
                continue
            if first_index == 0 and second_index == edge_count - 1:
                continue
            c = points[second_index]
            d = points[(second_index + 1) % edge_count]
            if _segments_intersect(a, b, c, d):
                raise ValueError(f"{path} must not self-intersect.")
    return points


@dataclass(frozen=True)
class CountingLineConfig:
    """Named virtual counting line in reference-frame pixels."""

    name: str
    p1: Point
    p2: Point

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        source: str,
        default_name: str | None = None,
    ) -> "CountingLineConfig":
        values = _mapping(raw, source)
        raw_name = values.get("name", default_name)
        name = _name(raw_name, f"{source}.name")
        if "p1" not in values or "p2" not in values:
            raise ValueError(f"{source} requires both p1 and p2.")
        p1 = _point(values["p1"], f"{source}.p1")
        p2 = _point(values["p2"], f"{source}.p2")
        if p1 == p2:
            raise ValueError(f"{source}.p1 and {source}.p2 must be different points.")
        return cls(name=name, p1=p1, p2=p2)

    def to_mapping(self) -> dict[str, object]:
        return {"name": self.name, "p1": list(self.p1), "p2": list(self.p2)}

    def to_legacy_mapping(self) -> dict[str, list[float]]:
        return {"p1": list(self.p1), "p2": list(self.p2)}


@dataclass(frozen=True)
class ZoneConfig:
    """Validated zone polygon plus optional calibrated physical properties."""

    name: str
    polygon: tuple[Point, ...]
    area_m2: float | None = None
    capacity: int | None = None
    priority: int = 0
    # Kept as a typed legacy extension because ZoneManager already consumes it.
    crowded_threshold: int | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, source: str) -> "ZoneConfig":
        values = _mapping(raw, source)
        if "name" not in values:
            raise ValueError(f"{source}.name is required.")
        if "polygon" not in values:
            raise ValueError(f"{source}.polygon is required.")
        area_m2 = (
            None
            if values.get("area_m2") is None
            else _positive_float(values["area_m2"], f"{source}.area_m2")
        )
        capacity = (
            None
            if values.get("capacity") is None
            else _positive_int(values["capacity"], f"{source}.capacity")
        )
        priority = 0 if values.get("priority") is None else _integer(values["priority"], f"{source}.priority")
        crowded_threshold = (
            None
            if values.get("crowded_threshold") is None
            else _positive_int(values["crowded_threshold"], f"{source}.crowded_threshold")
        )
        return cls(
            name=_name(values["name"], f"{source}.name"),
            polygon=_polygon(values["polygon"], f"{source}.polygon"),
            area_m2=area_m2,
            capacity=capacity,
            priority=priority,
            crowded_threshold=crowded_threshold,
        )

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self.name,
            "polygon": [list(point) for point in self.polygon],
        }
        if self.area_m2 is not None:
            result["area_m2"] = self.area_m2
        if self.capacity is not None:
            result["capacity"] = self.capacity
        if self.priority:
            result["priority"] = self.priority
        if self.crowded_threshold is not None:
            result["crowded_threshold"] = self.crowded_threshold
        return result

    def to_legacy_mapping(self) -> dict[str, object]:
        """Return a ZoneManager-compatible mapping, retaining useful metadata."""
        return self.to_mapping()


@dataclass(frozen=True)
class SpaceConfig:
    """Single-camera space calibration and analytics contract.

    ``capacity`` (also accepted as the clearer YAML alias
    ``maximum_capacity``) and ``visible_floor_area_m2`` are intentionally optional.
    Consumers must report occupancy or people/m² only when their corresponding
    calibration is present; neither can be inferred from image pixels.
    """

    name: str
    capacity: int | None = None
    visible_floor_area_m2: float | None = None
    zones: tuple[ZoneConfig, ...] = ()
    counting_lines: tuple[CountingLineConfig, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        source: str = "space",
        default_name: str | None = None,
    ) -> "SpaceConfig":
        values = _mapping(raw, source)
        raw_name = values.get("name", default_name)
        name = _name(raw_name, f"{source}.name")
        # ``capacity`` was the original public schema.  Classroom profiles use
        # ``maximum_capacity`` to distinguish the formal room limit from a
        # session's dynamically configured usable seats.  Keep one internal
        # field for compatibility while rejecting an ambiguous dual value.
        raw_capacity = values.get("capacity")
        raw_maximum_capacity = values.get("maximum_capacity")
        if raw_capacity is not None and raw_maximum_capacity is not None:
            capacity_value = _positive_int(raw_capacity, f"{source}.capacity")
            maximum_capacity_value = _positive_int(
                raw_maximum_capacity, f"{source}.maximum_capacity"
            )
            if capacity_value != maximum_capacity_value:
                raise ValueError(
                    f"{source}.capacity and {source}.maximum_capacity must match when both are set."
                )
            capacity = capacity_value
        elif raw_capacity is not None:
            capacity = _positive_int(raw_capacity, f"{source}.capacity")
        elif raw_maximum_capacity is not None:
            capacity = _positive_int(raw_maximum_capacity, f"{source}.maximum_capacity")
        else:
            capacity = None
        visible_floor_area_m2 = (
            None
            if values.get("visible_floor_area_m2") is None
            else _positive_float(values["visible_floor_area_m2"], f"{source}.visible_floor_area_m2")
        )

        raw_zones = values.get("zones", ())
        if raw_zones is None:
            raw_zones = ()
        zones = tuple(
            ZoneConfig.from_mapping(_mapping(zone, f"{source}.zones[{index}]"), source=f"{source}.zones[{index}]")
            for index, zone in enumerate(_sequence(raw_zones, f"{source}.zones"))
        )
        zone_names = [zone.name for zone in zones]
        if len(set(zone_names)) != len(zone_names):
            raise ValueError(f"{source}.zones contains duplicate names.")

        has_plural_lines = "counting_lines" in values
        has_legacy_line = "counting_line" in values and values["counting_line"] is not None
        if has_plural_lines and has_legacy_line:
            raise ValueError(f"{source} cannot define both counting_lines and counting_line.")
        if has_plural_lines:
            raw_lines = values["counting_lines"]
            if raw_lines is None:
                raw_lines = ()
            lines = tuple(
                CountingLineConfig.from_mapping(
                    _mapping(line, f"{source}.counting_lines[{index}]"),
                    source=f"{source}.counting_lines[{index}]",
                    default_name=f"line_{index + 1}",
                )
                for index, line in enumerate(_sequence(raw_lines, f"{source}.counting_lines"))
            )
        elif has_legacy_line:
            lines = (
                CountingLineConfig.from_mapping(
                    _mapping(values["counting_line"], f"{source}.counting_line"),
                    source=f"{source}.counting_line",
                    default_name="primary",
                ),
            )
        else:
            lines = ()
        line_names = [line.name for line in lines]
        if len(set(line_names)) != len(line_names):
            raise ValueError(f"{source}.counting_lines contains duplicate names.")

        return cls(
            name=name,
            capacity=capacity,
            visible_floor_area_m2=visible_floor_area_m2,
            zones=zones,
            counting_lines=lines,
        )

    @property
    def primary_counting_line(self) -> CountingLineConfig | None:
        """The first configured line, for the existing single-line analytics engine."""
        return self.counting_lines[0] if self.counting_lines else None

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self.name,
            "zones": [zone.to_mapping() for zone in self.zones],
            "counting_lines": [line.to_mapping() for line in self.counting_lines],
        }
        if self.capacity is not None:
            result["capacity"] = self.capacity
        if self.visible_floor_area_m2 is not None:
            result["visible_floor_area_m2"] = self.visible_floor_area_m2
        return result

    def to_legacy_analytics_mapping(self) -> dict[str, object]:
        """Adapt to today's ``analytics.zones`` / single ``counting_line`` API.

        The current LineCrossingCounter accepts one line only.  Refuse a lossy
        conversion when a future profile contains multiple named lines.
        """
        if len(self.counting_lines) > 1:
            raise ValueError("The current legacy analytics API supports one counting line only.")
        result: dict[str, object] = {"zones": [zone.to_legacy_mapping() for zone in self.zones]}
        if self.counting_lines:
            result["counting_line"] = self.counting_lines[0].to_legacy_mapping()
        return result


def _reject_conflicting_legacy_geometry(
    canonical: SpaceConfig,
    analytics: Mapping[str, Any],
    *,
    source: str,
) -> None:
    """Avoid silently choosing one set of camera polygons over another.

    A migration profile may carry legacy analytics knobs next to the canonical
    ``space`` block. Only geometry fields are compared here; trajectory and
    heatmap settings remain owned by ``analytics``.
    """

    geometry_keys = ("zones", "counting_line", "counting_lines")
    # An extending YAML profile commonly clears inherited legacy geometry with
    # ``zones: []`` / ``counting_line: null`` before supplying ``space``. Those
    # empty values are not a competing camera definition.
    supplied_keys = [
        key
        for key in geometry_keys
        if key in analytics and analytics[key] not in (None, [])
    ]
    if not supplied_keys:
        return
    legacy_values: dict[str, Any] = {"name": "legacy_comparison"}
    for key in supplied_keys:
        if key in analytics:
            legacy_values[key] = analytics[key]
    legacy = SpaceConfig.from_mapping(legacy_values, source=f"{source}.legacy_geometry")
    canonical_zone_geometry = tuple((zone.name, zone.polygon) for zone in canonical.zones)
    legacy_zone_geometry = tuple((zone.name, zone.polygon) for zone in legacy.zones)
    canonical_line_geometry = tuple((line.p1, line.p2) for line in canonical.counting_lines)
    legacy_line_geometry = tuple((line.p1, line.p2) for line in legacy.counting_lines)
    if canonical_zone_geometry != legacy_zone_geometry or canonical_line_geometry != legacy_line_geometry:
        raise ValueError(
            f"{source} defines conflicting canonical space and legacy analytics geometry; "
            "keep one source of zones/counting lines."
        )


def parse_space_config(raw: Mapping[str, Any], *, source: str = "config") -> SpaceConfig:
    """Parse either a canonical space block or current pipeline analytics fields.

    Accepted layouts are ``{space: {...}}``, ``{analytics: {space: {...}}}``,
    a direct canonical space mapping, and the legacy ``analytics.zones`` /
    ``analytics.counting_line`` shape.  Legacy shapes receive the stable name
    ``default_space`` because they had no name field.
    """
    values = _mapping(raw, source)
    if "space" in values:
        canonical = SpaceConfig.from_mapping(
            _mapping(values["space"], f"{source}.space"), source=f"{source}.space"
        )
        if "analytics" in values:
            _reject_conflicting_legacy_geometry(
                canonical,
                _mapping(values["analytics"], f"{source}.analytics"),
                source=source,
            )
        return canonical
    if "analytics" in values:
        analytics = _mapping(values["analytics"], f"{source}.analytics")
        if "space" in analytics:
            canonical = SpaceConfig.from_mapping(
                _mapping(analytics["space"], f"{source}.analytics.space"),
                source=f"{source}.analytics.space",
            )
            _reject_conflicting_legacy_geometry(canonical, analytics, source=f"{source}.analytics")
            return canonical
        return SpaceConfig.from_mapping(
            analytics,
            source=f"{source}.analytics",
            default_name="default_space",
        )

    legacy_default = "default_space" if {"zones", "counting_line"} & set(values) else None
    return SpaceConfig.from_mapping(values, source=source, default_name=legacy_default)


def load_space_config(path: str | Path) -> SpaceConfig:
    """Load a YAML mapping and parse its canonical or legacy space contract."""
    resolved_path = Path(path)
    with resolved_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return parse_space_config(_mapping(raw, str(resolved_path)), source=str(resolved_path))
