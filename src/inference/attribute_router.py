"""Cheap, camera-aware routing before face or body neural inference.

The router deliberately uses only a person box and frame geometry.  It never
predicts an attribute and it never combines face/body evidence; those jobs stay
with the existing temporal state manager.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


BBox = tuple[int, int, int, int]


class AttributeRoute(str, Enum):
    """The one branch a track may spend attribute compute on this frame."""

    FACE = "face"
    FACE_THEN_BODY = "face_then_body"
    BODY = "body"
    UNKNOWN = "unknown"


ROUTER_MODES = {"face_first", "adaptive", "body_only"}


@dataclass(frozen=True)
class RouteDecision:
    """A transparent routing result suitable for telemetry and unit tests."""

    route: AttributeRoute
    person_width: int
    person_height: int
    relative_height: float
    estimated_face_size_px: float
    vertically_truncated: bool
    body_usable: bool
    reason: str
    # Kept separately from the combined flag so calibration reports can tell a
    # top-clipped distant person from a close seated person cropped at bottom.
    touches_top: bool = False
    touches_bottom: bool = False


@dataclass(frozen=True)
class AttributeRouterConfig:
    """Validated, conservative geometry thresholds for one camera profile."""

    mode: str = "face_first"
    estimated_face_ratio: float = 0.20
    minimum_estimated_face_size_px: int = 48
    # A seated/close webcam subject often has a person box clipped at the
    # bottom edge. Its height then underestimates face size, while visible
    # upper-body width remains a useful signal to try YuNet once.
    minimum_bottom_truncated_person_width_px: int = 96
    minimum_relative_person_height: float = 0.15
    edge_margin_px: int = 3
    skip_severely_truncated: bool = True

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object] | None,
        *,
        body_minimum_width: int,
        body_minimum_height: int,
    ) -> "AttributeRouterConfig":
        config = dict(values or {})
        mode = str(config.get("mode", "face_first")).strip().lower()
        if mode not in ROUTER_MODES:
            raise ValueError(f"attribute_router.mode must be one of {sorted(ROUTER_MODES)}; received {mode!r}.")

        ratio = float(config.get("estimated_face_ratio", 0.20))
        face_size = int(config.get("minimum_estimated_face_size_px", 48))
        bottom_truncated_width = int(config.get("minimum_bottom_truncated_person_width_px", 96))
        relative_height = float(config.get("minimum_relative_person_height", 0.15))
        edge_margin = int(config.get("edge_margin_px", 3))
        if not 0.0 < ratio <= 1.0:
            raise ValueError("attribute_router.estimated_face_ratio must be in (0, 1].")
        if face_size < 1:
            raise ValueError("attribute_router.minimum_estimated_face_size_px must be positive.")
        if bottom_truncated_width < 1:
            raise ValueError("attribute_router.minimum_bottom_truncated_person_width_px must be positive.")
        if not 0.0 <= relative_height <= 1.0:
            raise ValueError("attribute_router.minimum_relative_person_height must be in [0, 1].")
        if edge_margin < 0:
            raise ValueError("attribute_router.edge_margin_px cannot be negative.")
        if body_minimum_width < 1 or body_minimum_height < 1:
            raise ValueError("body minimum dimensions must be positive before configuring the router.")
        return cls(
            mode=mode,
            estimated_face_ratio=ratio,
            minimum_estimated_face_size_px=face_size,
            minimum_bottom_truncated_person_width_px=bottom_truncated_width,
            minimum_relative_person_height=relative_height,
            edge_margin_px=edge_margin,
            skip_severely_truncated=bool(config.get("skip_severely_truncated", True)),
        )


class AttributeRouter:
    """Route a track before calling YuNet or either classifier.

    The estimated face ratio is only a camera-specific seed.  It must be
    calibrated from router telemetry, not interpreted as a human-anatomy rule.
    """

    def __init__(self, config: AttributeRouterConfig, *, body_minimum_width: int, body_minimum_height: int) -> None:
        self.config = config
        self.body_minimum_width = body_minimum_width
        self.body_minimum_height = body_minimum_height

    @property
    def face_enabled(self) -> bool:
        return self.config.mode != "body_only"

    def decide(
        self,
        bbox: Sequence[int],
        frame_shape: tuple[int, int] | tuple[int, int, int],
        *,
        face_cooldown_active: bool = False,
    ) -> RouteDecision:
        """Choose a route using only observable image geometry.

        The function is intentionally side-effect free.  Track hysteresis and
        cooldown ownership live in TrackStateManager.
        """

        if len(bbox) != 4:
            raise ValueError("bbox must contain x1, y1, x2, y2.")
        frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
        if frame_height < 1 or frame_width < 1:
            raise ValueError("frame dimensions must be positive.")

        x1, y1, x2, y2 = (int(value) for value in bbox)
        width, height = x2 - x1, y2 - y1
        if width < 1 or height < 1:
            return RouteDecision(
                AttributeRoute.UNKNOWN, max(0, width), max(0, height), 0.0, 0.0, False, False, "invalid_bbox"
            )

        margin = self.config.edge_margin_px
        touches_top = y1 <= margin
        touches_bottom = y2 >= frame_height - margin
        vertically_truncated = touches_top or touches_bottom
        relative_height = height / frame_height
        estimated_face_size = height * self.config.estimated_face_ratio
        height_face_usable = estimated_face_size >= self.config.minimum_estimated_face_size_px
        # Recover close seated subjects whose lower body is outside the image.
        # A top-clipped box is deliberately excluded: YuNet needs the full face.
        bottom_truncated_width_proxy = (
            touches_bottom
            and not touches_top
            and width >= self.config.minimum_bottom_truncated_person_width_px
        )
        face_usable = height_face_usable or bottom_truncated_width_proxy
        body_usable = (
            width >= self.body_minimum_width
            and height >= self.body_minimum_height
            and relative_height >= self.config.minimum_relative_person_height
            and not (self.config.skip_severely_truncated and vertically_truncated)
        )

        if self.config.mode == "body_only":
            route = AttributeRoute.BODY if body_usable else AttributeRoute.UNKNOWN
            reason = "body_only" if body_usable else "body_quality_gate"
        elif self.config.mode == "face_first":
            route = AttributeRoute.FACE_THEN_BODY if self.face_enabled else AttributeRoute.UNKNOWN
            reason = "compatibility_face_first"
        elif face_cooldown_active:
            route = AttributeRoute.BODY if body_usable else AttributeRoute.UNKNOWN
            reason = "face_cooldown_body" if body_usable else "face_cooldown_no_body_evidence"
        elif vertically_truncated:
            route = AttributeRoute.FACE if face_usable else AttributeRoute.UNKNOWN
            if bottom_truncated_width_proxy and not height_face_usable:
                reason = "bottom_truncated_width_proxy"
            else:
                reason = "truncated_face" if face_usable else "truncated_no_face_evidence"
        elif face_usable and body_usable:
            route = AttributeRoute.FACE_THEN_BODY
            reason = "both_feasible"
        elif face_usable:
            route = AttributeRoute.FACE
            reason = "face_only_feasible"
        elif body_usable:
            route = AttributeRoute.BODY
            reason = "face_too_small_body_feasible"
        else:
            route = AttributeRoute.UNKNOWN
            reason = "insufficient_visual_evidence"

        return RouteDecision(
            route=route,
            person_width=width,
            person_height=height,
            relative_height=relative_height,
            estimated_face_size_px=estimated_face_size,
            vertically_truncated=vertically_truncated,
            body_usable=body_usable,
            reason=reason,
            touches_top=touches_top,
            touches_bottom=touches_bottom,
        )
