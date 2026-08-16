"""Validated, serialisable classroom configuration contracts.

This module deliberately has no dependency on the live tracker or on
``space_config``.  It describes three different layers that must not be
conflated:

* :class:`RoomProfile` is fixed for a camera/room installation.
* :class:`LayoutTemplate` describes a reusable seating pattern such as 2-4-2.
* :class:`SessionLayout` selects a template and the seats enabled for one
  teaching session.

The template can optionally contain a reference-frame quadrilateral.  In that
case it generates deterministic seat polygons and anchors in camera pixels;
otherwise it still generates semantic seat definitions.  It intentionally does
not claim a physical calibration or infer a room capacity from the seat count.
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
Polygon: TypeAlias = tuple[Point, ...]
SeatKey: TypeAlias = tuple[int, str, int]


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


def _finite_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{path} must be a finite number.")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{path} must be a finite number.")
    return result


def _positive_float(value: object, path: str) -> float:
    result = _finite_float(value, path)
    if result <= 0.0:
        raise ValueError(f"{path} must be a finite positive number.")
    return result


def _non_negative_float(value: object, path: str) -> float:
    result = _finite_float(value, path)
    if result < 0.0:
        raise ValueError(f"{path} must be a finite non-negative number.")
    return result


def _unit_interval(value: object, path: str) -> float:
    result = _finite_float(value, path)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{path} must be between 0 and 1.")
    return result


def _reference_coordinate(value: object, path: str) -> float:
    return _non_negative_float(value, path)


def _reference_point(value: object, path: str) -> Point:
    values = _sequence(value, path)
    if len(values) != 2:
        raise ValueError(f"{path} must contain exactly [x, y].")
    return (
        _reference_coordinate(values[0], f"{path}[0]"),
        _reference_coordinate(values[1], f"{path}[1]"),
    )


def _world_point(value: object, path: str) -> Point:
    values = _sequence(value, path)
    if len(values) != 2:
        raise ValueError(f"{path} must contain exactly [x, y].")
    return (_finite_float(values[0], f"{path}[0]"), _finite_float(values[1], f"{path}[1]"))


def _polygon_signed_area(polygon: Sequence[Point]) -> float:
    return sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    ) / 2.0


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])


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


def _polygon(value: object, path: str) -> Polygon:
    raw_points = _sequence(value, path)
    points = tuple(_reference_point(point, f"{path}[{index}]") for index, point in enumerate(raw_points))
    # Permit conventional closed GIS/YAML polygons while storing an open form.
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3 or len(set(points)) < 3:
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
            if second_index in {first_index, (first_index + 1) % edge_count}:
                continue
            if first_index == 0 and second_index == edge_count - 1:
                continue
            c = points[second_index]
            d = points[(second_index + 1) % edge_count]
            if _segments_intersect(a, b, c, d):
                raise ValueError(f"{path} must not self-intersect.")
    return points


def _quadrilateral(value: object, path: str) -> Polygon:
    """Validate a convex, ordered reference-frame quadrilateral.

    The expected point order is top-left, top-right, bottom-right,
    bottom-left in image/reference coordinates.  Keeping this convention
    explicit makes the generated grid reproducible and avoids a folded
    bilinear transform.
    """
    polygon = _polygon(value, path)
    if len(polygon) != 4:
        raise ValueError(f"{path} must contain exactly four corners.")
    if _polygon_signed_area(polygon) <= 0.0:
        raise ValueError(
            f"{path} must be ordered top-left, top-right, bottom-right, bottom-left."
        )
    orientations = tuple(
        _orientation(polygon[index], polygon[(index + 1) % 4], polygon[(index + 2) % 4])
        for index in range(4)
    )
    if any(value <= 1e-9 for value in orientations):
        raise ValueError(f"{path} must be a convex quadrilateral.")
    return polygon


def _optional_capacity(values: Mapping[str, Any], source: str) -> int | None:
    """Read canonical maximum_capacity and its legacy capacity alias safely."""
    raw_maximum = values.get("maximum_capacity")
    raw_alias = values.get("capacity")
    maximum = None if raw_maximum is None else _positive_int(raw_maximum, f"{source}.maximum_capacity")
    alias = None if raw_alias is None else _positive_int(raw_alias, f"{source}.capacity")
    if maximum is not None and alias is not None and maximum != alias:
        raise ValueError(f"{source}.maximum_capacity and {source}.capacity must match when both are set.")
    return maximum if maximum is not None else alias


def _resolution(value: object, path: str) -> tuple[int, int]:
    values = _sequence(value, path)
    if len(values) != 2:
        raise ValueError(f"{path} must contain exactly [width, height].")
    return (_positive_int(values[0], f"{path}[0]"), _positive_int(values[1], f"{path}[1]"))


def _validate_reference_bounds(points: Sequence[Point], resolution: tuple[int, int] | None, path: str) -> None:
    if resolution is None:
        return
    width, height = resolution
    for index, (x_coord, y_coord) in enumerate(points):
        if x_coord > width or y_coord > height:
            raise ValueError(f"{path}[{index}] lies outside reference_resolution {list(resolution)}.")


@dataclass(frozen=True)
class ReferenceLine:
    """A named virtual line in reference-frame pixels."""

    p1: Point
    p2: Point

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, source: str) -> "ReferenceLine":
        values = _mapping(raw, source)
        if "p1" not in values or "p2" not in values:
            raise ValueError(f"{source} requires both p1 and p2.")
        p1 = _reference_point(values["p1"], f"{source}.p1")
        p2 = _reference_point(values["p2"], f"{source}.p2")
        if p1 == p2:
            raise ValueError(f"{source}.p1 and {source}.p2 must be different points.")
        return cls(p1=p1, p2=p2)

    def to_mapping(self) -> dict[str, list[float]]:
        return {"p1": list(self.p1), "p2": list(self.p2)}


@dataclass(frozen=True)
class LayoutRules:
    """Physical layout rules recorded as metadata, in metres.

    These values are not used to derive distances from detection boxes.  A
    future calibrated spatial engine may validate measured geometry against
    them.
    """

    minimum_row_gap_m: float = 0.70
    minimum_aisle_width_m: float = 0.60
    desk_height_m: float = 0.75
    chair_height_m: float = 0.45

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None, *, source: str) -> "LayoutRules":
        if raw is None:
            return cls()
        values = _mapping(raw, source)
        return cls(
            minimum_row_gap_m=_positive_float(
                values.get("minimum_row_gap_m", 0.70), f"{source}.minimum_row_gap_m"
            ),
            minimum_aisle_width_m=_positive_float(
                values.get("minimum_aisle_width_m", 0.60), f"{source}.minimum_aisle_width_m"
            ),
            desk_height_m=_positive_float(values.get("desk_height_m", 0.75), f"{source}.desk_height_m"),
            chair_height_m=_positive_float(values.get("chair_height_m", 0.45), f"{source}.chair_height_m"),
        )

    def to_mapping(self) -> dict[str, float]:
        return {
            "minimum_row_gap_m": self.minimum_row_gap_m,
            "minimum_aisle_width_m": self.minimum_aisle_width_m,
            "desk_height_m": self.desk_height_m,
            "chair_height_m": self.chair_height_m,
        }


@dataclass(frozen=True)
class CameraCalibration:
    """Optional paired calibration correspondences, without computing homography."""

    floor_points_px: tuple[Point, ...]
    floor_points_m: tuple[Point, ...]
    maximum_error_cm: float = 10.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, source: str) -> "CameraCalibration":
        values = _mapping(raw, source)
        if "floor_points_px" not in values or "floor_points_m" not in values:
            raise ValueError(f"{source} requires floor_points_px and floor_points_m together.")
        raw_pixels = _sequence(values["floor_points_px"], f"{source}.floor_points_px")
        raw_metres = _sequence(values["floor_points_m"], f"{source}.floor_points_m")
        pixels = tuple(_reference_point(point, f"{source}.floor_points_px[{index}]") for index, point in enumerate(raw_pixels))
        metres = tuple(_world_point(point, f"{source}.floor_points_m[{index}]") for index, point in enumerate(raw_metres))
        if len(pixels) < 4 or len(metres) < 4:
            raise ValueError(f"{source} requires at least four paired floor points.")
        if len(pixels) != len(metres):
            raise ValueError(f"{source}.floor_points_px and floor_points_m must have the same length.")
        if len(set(pixels)) < 4 or len(set(metres)) < 4:
            raise ValueError(f"{source} requires at least four distinct paired floor points.")
        return cls(
            floor_points_px=pixels,
            floor_points_m=metres,
            maximum_error_cm=_positive_float(values.get("maximum_error_cm", 10.0), f"{source}.maximum_error_cm"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "floor_points_px": [list(point) for point in self.floor_points_px],
            "floor_points_m": [list(point) for point in self.floor_points_m],
            "maximum_error_cm": self.maximum_error_cm,
        }


@dataclass(frozen=True)
class RoomProfile:
    """Fixed camera/room facts that do not change from one session to another."""

    name: str
    visible_floor_area_m2: float | None = None
    maximum_capacity: int | None = None
    reference_resolution: tuple[int, int] | None = None
    room_boundary: Polygon | None = None
    door_roi: Polygon | None = None
    counting_line: ReferenceLine | None = None
    layout_rules: LayoutRules = LayoutRules()
    calibration: CameraCalibration | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, source: str = "room_profile") -> "RoomProfile":
        values = _mapping(raw, source)
        if "name" not in values:
            raise ValueError(f"{source}.name is required.")
        raw_camera = values.get("camera")
        camera = {} if raw_camera is None else _mapping(raw_camera, f"{source}.camera")
        direct_resolution = values.get("reference_resolution")
        camera_resolution = camera.get("reference_resolution")
        if direct_resolution is not None and camera_resolution is not None:
            direct = _resolution(direct_resolution, f"{source}.reference_resolution")
            nested = _resolution(camera_resolution, f"{source}.camera.reference_resolution")
            if direct != nested:
                raise ValueError(
                    f"{source}.reference_resolution and {source}.camera.reference_resolution must match."
                )
            reference_resolution = direct
        elif direct_resolution is not None:
            reference_resolution = _resolution(direct_resolution, f"{source}.reference_resolution")
        elif camera_resolution is not None:
            reference_resolution = _resolution(camera_resolution, f"{source}.camera.reference_resolution")
        else:
            reference_resolution = None

        room_boundary = (
            None if values.get("room_boundary") is None else _polygon(values["room_boundary"], f"{source}.room_boundary")
        )
        raw_entrance = values.get("entrance")
        entrance = {} if raw_entrance is None else _mapping(raw_entrance, f"{source}.entrance")
        direct_door = values.get("door_roi")
        nested_door = entrance.get("door_roi")
        if direct_door is not None and nested_door is not None:
            direct = _polygon(direct_door, f"{source}.door_roi")
            nested = _polygon(nested_door, f"{source}.entrance.door_roi")
            if direct != nested:
                raise ValueError(f"{source}.door_roi and {source}.entrance.door_roi must match.")
            door_roi = direct
        elif direct_door is not None:
            door_roi = _polygon(direct_door, f"{source}.door_roi")
        elif nested_door is not None:
            door_roi = _polygon(nested_door, f"{source}.entrance.door_roi")
        else:
            door_roi = None

        direct_line = values.get("counting_line")
        nested_line = entrance.get("counting_line")
        if direct_line is not None and nested_line is not None:
            direct = ReferenceLine.from_mapping(_mapping(direct_line, f"{source}.counting_line"), source=f"{source}.counting_line")
            nested = ReferenceLine.from_mapping(
                _mapping(nested_line, f"{source}.entrance.counting_line"),
                source=f"{source}.entrance.counting_line",
            )
            if direct != nested:
                raise ValueError(f"{source}.counting_line and {source}.entrance.counting_line must match.")
            counting_line = direct
        elif direct_line is not None:
            counting_line = ReferenceLine.from_mapping(
                _mapping(direct_line, f"{source}.counting_line"), source=f"{source}.counting_line"
            )
        elif nested_line is not None:
            counting_line = ReferenceLine.from_mapping(
                _mapping(nested_line, f"{source}.entrance.counting_line"),
                source=f"{source}.entrance.counting_line",
            )
        else:
            counting_line = None

        calibration = (
            None
            if values.get("calibration") is None
            else CameraCalibration.from_mapping(
                _mapping(values["calibration"], f"{source}.calibration"), source=f"{source}.calibration"
            )
        )
        if room_boundary is not None:
            _validate_reference_bounds(room_boundary, reference_resolution, f"{source}.room_boundary")
        if door_roi is not None:
            _validate_reference_bounds(door_roi, reference_resolution, f"{source}.door_roi")
        if counting_line is not None:
            _validate_reference_bounds((counting_line.p1, counting_line.p2), reference_resolution, f"{source}.counting_line")
        if calibration is not None:
            _validate_reference_bounds(
                calibration.floor_points_px, reference_resolution, f"{source}.calibration.floor_points_px"
            )

        return cls(
            name=_name(values["name"], f"{source}.name"),
            visible_floor_area_m2=(
                None
                if values.get("visible_floor_area_m2") is None
                else _positive_float(values["visible_floor_area_m2"], f"{source}.visible_floor_area_m2")
            ),
            maximum_capacity=_optional_capacity(values, source),
            reference_resolution=reference_resolution,
            room_boundary=room_boundary,
            door_roi=door_roi,
            counting_line=counting_line,
            layout_rules=LayoutRules.from_mapping(
                None
                if values.get("layout_rules") is None
                else _mapping(values["layout_rules"], f"{source}.layout_rules"),
                source=f"{source}.layout_rules",
            ),
            calibration=calibration,
        )

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self.name,
            "layout_rules": self.layout_rules.to_mapping(),
        }
        if self.visible_floor_area_m2 is not None:
            result["visible_floor_area_m2"] = self.visible_floor_area_m2
        if self.maximum_capacity is not None:
            result["maximum_capacity"] = self.maximum_capacity
        if self.reference_resolution is not None:
            result["camera"] = {"reference_resolution": list(self.reference_resolution)}
        if self.room_boundary is not None:
            result["room_boundary"] = [list(point) for point in self.room_boundary]
        entrance: dict[str, object] = {}
        if self.door_roi is not None:
            entrance["door_roi"] = [list(point) for point in self.door_roi]
        if self.counting_line is not None:
            entrance["counting_line"] = self.counting_line.to_mapping()
        if entrance:
            result["entrance"] = entrance
        if self.calibration is not None:
            result["calibration"] = self.calibration.to_mapping()
        return result


@dataclass(frozen=True)
class ReferenceGrid:
    """A quadrilateral grid from which reference-frame seat geometry is derived.

    ``region`` uses the point order top-left, top-right, bottom-right,
    bottom-left.  Gap sizes are measured in fractions of one seat cell, so a
    template remains portable across different reference resolutions.
    """

    region: Polygon
    block_gap_units: float = 0.50
    row_gap_units: float = 0.15
    seat_anchor_ratio: float = 0.65

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, source: str) -> "ReferenceGrid":
        values = _mapping(raw, source)
        raw_region = values.get("region", values.get("polygon"))
        if raw_region is None:
            raise ValueError(f"{source}.region is required.")
        return cls(
            region=_quadrilateral(raw_region, f"{source}.region"),
            block_gap_units=_non_negative_float(
                values.get("block_gap_units", 0.50), f"{source}.block_gap_units"
            ),
            row_gap_units=_non_negative_float(values.get("row_gap_units", 0.15), f"{source}.row_gap_units"),
            seat_anchor_ratio=_unit_interval(
                values.get("seat_anchor_ratio", 0.65), f"{source}.seat_anchor_ratio"
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "region": [list(point) for point in self.region],
            "block_gap_units": self.block_gap_units,
            "row_gap_units": self.row_gap_units,
            "seat_anchor_ratio": self.seat_anchor_ratio,
        }

    def project(self, u_coord: float, v_coord: float) -> Point:
        """Project normalised grid coordinates using bilinear interpolation."""
        if not 0.0 <= u_coord <= 1.0 or not 0.0 <= v_coord <= 1.0:
            raise ValueError("Reference grid coordinates must be between 0 and 1.")
        top_left, top_right, bottom_right, bottom_left = self.region
        left_x = top_left[0] * (1.0 - v_coord) + bottom_left[0] * v_coord
        left_y = top_left[1] * (1.0 - v_coord) + bottom_left[1] * v_coord
        right_x = top_right[0] * (1.0 - v_coord) + bottom_right[0] * v_coord
        right_y = top_right[1] * (1.0 - v_coord) + bottom_right[1] * v_coord
        return (
            left_x * (1.0 - u_coord) + right_x * u_coord,
            left_y * (1.0 - u_coord) + right_y * u_coord,
        )

    def cell_geometry(self, u_start: float, u_end: float, v_start: float, v_end: float) -> tuple[Polygon, Point]:
        if not 0.0 <= u_start < u_end <= 1.0 or not 0.0 <= v_start < v_end <= 1.0:
            raise ValueError("A reference-grid cell must have positive normalised width and height.")
        polygon = (
            self.project(u_start, v_start),
            self.project(u_end, v_start),
            self.project(u_end, v_end),
            self.project(u_start, v_end),
        )
        anchor = self.project((u_start + u_end) / 2.0, v_start + (v_end - v_start) * self.seat_anchor_ratio)
        return polygon, anchor


@dataclass(frozen=True)
class LayoutBlock:
    """One named block in a generic seating pattern (for example left: 2)."""

    name: str
    columns: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, source: str) -> "LayoutBlock":
        values = _mapping(raw, source)
        if "name" not in values or "columns" not in values:
            raise ValueError(f"{source} requires name and columns.")
        return cls(name=_name(values["name"], f"{source}.name"), columns=_positive_int(values["columns"], f"{source}.columns"))

    def to_mapping(self) -> dict[str, object]:
        return {"name": self.name, "columns": self.columns}


def _layout_blocks(value: object, path: str) -> tuple[LayoutBlock, ...]:
    """Accept a readable mapping (`left: 2`) or a sequence of block mappings."""
    if isinstance(value, Mapping):
        blocks = tuple(
            LayoutBlock(name=_name(name, f"{path} key"), columns=_positive_int(columns, f"{path}.{name}"))
            for name, columns in value.items()
        )
    else:
        blocks = tuple(
            LayoutBlock.from_mapping(_mapping(item, f"{path}[{index}]"), source=f"{path}[{index}]")
            for index, item in enumerate(_sequence(value, path))
        )
    if not blocks:
        raise ValueError(f"{path} must contain at least one block.")
    names = [block.name for block in blocks]
    if len(set(names)) != len(names):
        raise ValueError(f"{path} contains duplicate block names.")
    return blocks


@dataclass(frozen=True)
class LayoutTemplate:
    """Reusable, generic seat block structure; it never fixes a row count."""

    name: str
    blocks: tuple[LayoutBlock, ...]
    reference_grid: ReferenceGrid | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, source: str = "layout_template") -> "LayoutTemplate":
        values = _mapping(raw, source)
        if "name" not in values or "blocks" not in values:
            raise ValueError(f"{source} requires name and blocks.")
        raw_grid = values.get("reference_grid")
        return cls(
            name=_name(values["name"], f"{source}.name"),
            blocks=_layout_blocks(values["blocks"], f"{source}.blocks"),
            reference_grid=(
                None
                if raw_grid is None
                else ReferenceGrid.from_mapping(
                    _mapping(raw_grid, f"{source}.reference_grid"), source=f"{source}.reference_grid"
                )
            ),
        )

    @property
    def column_count(self) -> int:
        return sum(block.columns for block in self.blocks)

    @property
    def block_names(self) -> tuple[str, ...]:
        return tuple(block.name for block in self.blocks)

    def column_count_for(self, block_name: str) -> int:
        for block in self.blocks:
            if block.name == block_name:
                return block.columns
        raise ValueError(f"Unknown layout block: {block_name!r}.")

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {"name": self.name, "blocks": [block.to_mapping() for block in self.blocks]}
        if self.reference_grid is not None:
            result["reference_grid"] = self.reference_grid.to_mapping()
        return result


def _seat_key(raw: Mapping[str, Any], *, source: str) -> SeatKey:
    values = _mapping(raw, source)
    for field in ("row", "block", "column"):
        if field not in values:
            raise ValueError(f"{source}.{field} is required.")
    return (
        _positive_int(values["row"], f"{source}.row"),
        _name(values["block"], f"{source}.block"),
        _positive_int(values["column"], f"{source}.column"),
    )


def _seat_identifier(key: SeatKey) -> str:
    row, block, column = key
    return f"r{row}-{block}-c{column}"


@dataclass(frozen=True)
class SeatDefinition:
    """A concrete session seat, optionally with a reference-frame polygon."""

    seat_id: str
    row: int
    block: str
    column: int
    enabled: bool = True
    polygon: Polygon | None = None
    anchor: Point | None = None

    @property
    def key(self) -> SeatKey:
        return (self.row, self.block, self.column)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, source: str = "seat") -> "SeatDefinition":
        values = _mapping(raw, source)
        row, block, column = _seat_key(values, source=source)
        raw_enabled = values.get("enabled", True)
        if not isinstance(raw_enabled, bool):
            raise ValueError(f"{source}.enabled must be a boolean.")
        raw_polygon = values.get("polygon")
        raw_anchor = values.get("anchor")
        return cls(
            seat_id=_name(values.get("seat_id", _seat_identifier((row, block, column))), f"{source}.seat_id"),
            row=row,
            block=block,
            column=column,
            enabled=raw_enabled,
            polygon=None if raw_polygon is None else _polygon(raw_polygon, f"{source}.polygon"),
            anchor=None if raw_anchor is None else _reference_point(raw_anchor, f"{source}.anchor"),
        )

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "seat_id": self.seat_id,
            "row": self.row,
            "block": self.block,
            "column": self.column,
            "enabled": self.enabled,
        }
        if self.polygon is not None:
            result["polygon"] = [list(point) for point in self.polygon]
        if self.anchor is not None:
            result["anchor"] = list(self.anchor)
        return result


@dataclass(frozen=True)
class SessionLayout:
    """Mutable-per-session layout choice; disabled seats do not alter the template."""

    room_profile: str
    template: str
    rows: int
    disabled_seats: frozenset[SeatKey] = frozenset()

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        source: str = "session_layout",
        template: LayoutTemplate | None = None,
    ) -> "SessionLayout":
        values = _mapping(raw, source)
        for field in ("room_profile", "rows"):
            if field not in values:
                raise ValueError(f"{source}.{field} is required.")
        raw_template = values.get("template", values.get("layout_template"))
        if values.get("template") is not None and values.get("layout_template") is not None:
            first = _name(values["template"], f"{source}.template")
            second = _name(values["layout_template"], f"{source}.layout_template")
            if first != second:
                raise ValueError(f"{source}.template and {source}.layout_template must match.")
        template_name = _name(raw_template, f"{source}.template")
        raw_disabled = values.get("disabled_seats", ())
        if raw_disabled is None:
            raw_disabled = ()
        disabled = tuple(
            _seat_key(_mapping(item, f"{source}.disabled_seats[{index}]"), source=f"{source}.disabled_seats[{index}]")
            for index, item in enumerate(_sequence(raw_disabled, f"{source}.disabled_seats"))
        )
        if len(set(disabled)) != len(disabled):
            raise ValueError(f"{source}.disabled_seats contains duplicate seats.")
        result = cls(
            room_profile=_name(values["room_profile"], f"{source}.room_profile"),
            template=template_name,
            rows=_positive_int(values["rows"], f"{source}.rows"),
            disabled_seats=frozenset(disabled),
        )
        if template is not None:
            if result.template != template.name:
                raise ValueError(
                    f"{source}.template {result.template!r} does not match supplied template {template.name!r}."
                )
            result.validate_against(template, source=source)
        return result

    def validate_against(self, template: LayoutTemplate, *, source: str = "session_layout") -> None:
        for row, block, column in self.disabled_seats:
            if row > self.rows:
                raise ValueError(f"{source}.disabled_seats references row {row}, but rows is {self.rows}.")
            try:
                maximum_column = template.column_count_for(block)
            except ValueError as error:
                raise ValueError(f"{source}.disabled_seats references unknown block {block!r}.") from error
            if column > maximum_column:
                raise ValueError(
                    f"{source}.disabled_seats references column {column} in block {block!r}, "
                    f"which has only {maximum_column} columns."
                )

    @property
    def disabled_seat_count(self) -> int:
        return len(self.disabled_seats)

    def total_seat_count(self, template: LayoutTemplate) -> int:
        self._require_template(template)
        return self.rows * template.column_count

    def enabled_seat_count(self, template: LayoutTemplate) -> int:
        return self.total_seat_count(template) - self.disabled_seat_count

    def seat_definitions(self, template: LayoutTemplate) -> tuple[SeatDefinition, ...]:
        """Generate one stable seat record per template row/block/column."""
        self._require_template(template)
        grid = template.reference_grid
        total_horizontal_units = float(template.column_count)
        if grid is not None:
            total_horizontal_units += grid.block_gap_units * (len(template.blocks) - 1)
        total_vertical_units = float(self.rows)
        if grid is not None:
            total_vertical_units += grid.row_gap_units * (self.rows - 1)

        definitions: list[SeatDefinition] = []
        horizontal_cursor = 0.0
        for block_index, layout_block in enumerate(template.blocks):
            for column in range(1, layout_block.columns + 1):
                u_start = (horizontal_cursor + column - 1) / total_horizontal_units
                u_end = (horizontal_cursor + column) / total_horizontal_units
                for row in range(1, self.rows + 1):
                    key = (row, layout_block.name, column)
                    polygon: Polygon | None = None
                    anchor: Point | None = None
                    if grid is not None:
                        row_start_units = (row - 1) * (1.0 + grid.row_gap_units)
                        v_start = row_start_units / total_vertical_units
                        v_end = (row_start_units + 1.0) / total_vertical_units
                        polygon, anchor = grid.cell_geometry(u_start, u_end, v_start, v_end)
                    definitions.append(
                        SeatDefinition(
                            seat_id=_seat_identifier(key),
                            row=row,
                            block=layout_block.name,
                            column=column,
                            enabled=key not in self.disabled_seats,
                            polygon=polygon,
                            anchor=anchor,
                        )
                    )
            horizontal_cursor += layout_block.columns
            if grid is not None and block_index < len(template.blocks) - 1:
                horizontal_cursor += grid.block_gap_units
        return tuple(definitions)

    def to_mapping(self) -> dict[str, object]:
        return {
            "room_profile": self.room_profile,
            "template": self.template,
            "rows": self.rows,
            "disabled_seats": [
                {"row": row, "block": block, "column": column}
                for row, block, column in sorted(self.disabled_seats)
            ],
        }

    def _require_template(self, template: LayoutTemplate) -> None:
        if template.name != self.template:
            raise ValueError(f"Session template {self.template!r} does not match layout template {template.name!r}.")
        self.validate_against(template)


@dataclass(frozen=True)
class CapacitySummary:
    """Explicit capacity semantics; physical seats are not room capacity."""

    maximum_capacity: int | None
    total_seats: int | None
    enabled_seats: int | None
    disabled_seats: int | None

    @property
    def maximum_capacity_minus_enabled_seats(self) -> int | None:
        if self.maximum_capacity is None or self.enabled_seats is None:
            return None
        return self.maximum_capacity - self.enabled_seats

    def to_mapping(self) -> dict[str, int | None]:
        return {
            "maximum_capacity": self.maximum_capacity,
            "total_seats": self.total_seats,
            "enabled_seats": self.enabled_seats,
            "disabled_seats": self.disabled_seats,
            "maximum_capacity_minus_enabled_seats": self.maximum_capacity_minus_enabled_seats,
        }


@dataclass(frozen=True)
class ClassroomConfig:
    """Bundle a fixed room profile with optional templates and one active session."""

    room_profile: RoomProfile
    layout_templates: tuple[LayoutTemplate, ...] = ()
    session_layout: SessionLayout | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, source: str = "classroom") -> "ClassroomConfig":
        values = _mapping(raw, source)
        if "classroom" in values:
            values = _mapping(values["classroom"], f"{source}.classroom")
            source = f"{source}.classroom"
        raw_room = values.get("room_profile", values.get("room"))
        if raw_room is None:
            raise ValueError(f"{source}.room_profile is required.")
        room = RoomProfile.from_mapping(_mapping(raw_room, f"{source}.room_profile"), source=f"{source}.room_profile")

        has_plural_templates = "layout_templates" in values
        has_single_template = "layout_template" in values
        if has_plural_templates and has_single_template:
            raise ValueError(f"{source} cannot define both layout_templates and layout_template.")
        if has_plural_templates:
            raw_templates = values["layout_templates"]
            if raw_templates is None:
                raw_templates = ()
            templates = tuple(
                LayoutTemplate.from_mapping(
                    _mapping(template, f"{source}.layout_templates[{index}]"),
                    source=f"{source}.layout_templates[{index}]",
                )
                for index, template in enumerate(_sequence(raw_templates, f"{source}.layout_templates"))
            )
        elif has_single_template:
            templates = (
                LayoutTemplate.from_mapping(
                    _mapping(values["layout_template"], f"{source}.layout_template"),
                    source=f"{source}.layout_template",
                ),
            )
        else:
            templates = ()
        names = [template.name for template in templates]
        if len(set(names)) != len(names):
            raise ValueError(f"{source}.layout_templates contains duplicate template names.")
        for index, template in enumerate(templates):
            if template.reference_grid is not None:
                _validate_reference_bounds(
                    template.reference_grid.region,
                    room.reference_resolution,
                    f"{source}.layout_templates[{index}].reference_grid.region",
                )

        raw_session = values.get("session_layout")
        if raw_session is None:
            session = None
        else:
            session = SessionLayout.from_mapping(
                _mapping(raw_session, f"{source}.session_layout"), source=f"{source}.session_layout"
            )
            if session.room_profile != room.name:
                raise ValueError(
                    f"{source}.session_layout.room_profile {session.room_profile!r} does not match room {room.name!r}."
                )
            template = next((candidate for candidate in templates if candidate.name == session.template), None)
            if template is None:
                raise ValueError(
                    f"{source}.session_layout.template {session.template!r} is not present in layout_templates."
                )
            session.validate_against(template, source=f"{source}.session_layout")
        return cls(room_profile=room, layout_templates=templates, session_layout=session)

    @property
    def active_template(self) -> LayoutTemplate | None:
        if self.session_layout is None:
            return None
        return next(
            (template for template in self.layout_templates if template.name == self.session_layout.template),
            None,
        )

    @property
    def seat_definitions(self) -> tuple[SeatDefinition, ...]:
        if self.session_layout is None:
            return ()
        template = self.active_template
        if template is None:  # Defensive guard for direct dataclass construction.
            return ()
        return self.session_layout.seat_definitions(template)

    @property
    def capacity_summary(self) -> CapacitySummary:
        if self.session_layout is None or self.active_template is None:
            return CapacitySummary(
                maximum_capacity=self.room_profile.maximum_capacity,
                total_seats=None,
                enabled_seats=None,
                disabled_seats=None,
            )
        template = self.active_template
        return CapacitySummary(
            maximum_capacity=self.room_profile.maximum_capacity,
            total_seats=self.session_layout.total_seat_count(template),
            enabled_seats=self.session_layout.enabled_seat_count(template),
            disabled_seats=self.session_layout.disabled_seat_count,
        )

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "room_profile": self.room_profile.to_mapping(),
            "layout_templates": [template.to_mapping() for template in self.layout_templates],
        }
        if self.session_layout is not None:
            result["session_layout"] = self.session_layout.to_mapping()
        return result


def parse_classroom_config(raw: Mapping[str, Any], *, source: str = "config") -> ClassroomConfig:
    """Parse a canonical classroom mapping or a root-level ``classroom`` block."""
    return ClassroomConfig.from_mapping(_mapping(raw, source), source=source)


def load_classroom_config(path: str | Path) -> ClassroomConfig:
    """Load a YAML classroom profile without coupling it to the live pipeline."""
    resolved_path = Path(path)
    with resolved_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return parse_classroom_config(_mapping(raw, str(resolved_path)), source=str(resolved_path))
