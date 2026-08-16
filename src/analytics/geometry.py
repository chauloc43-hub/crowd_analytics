from __future__ import annotations

from typing import Iterable, Sequence


Point = tuple[float, float]


def scale_point(point: Sequence[float], frame_size: tuple[int, int], reference_size: tuple[int, int]) -> Point:
    """Scale a point stored in reference pixels to the current frame size."""
    frame_width, frame_height = frame_size
    reference_width, reference_height = reference_size
    if reference_width <= 0 or reference_height <= 0:
        raise ValueError("reference_size must contain positive values")
    return (
        float(point[0]) * frame_width / reference_width,
        float(point[1]) * frame_height / reference_height,
    )


def scale_polygon(
    polygon: Iterable[Sequence[float]], frame_size: tuple[int, int], reference_size: tuple[int, int]
) -> list[Point]:
    return [scale_point(point, frame_size, reference_size) for point in polygon]
