"""Derived occupancy, calibrated density and exclusive crowd distribution.

This branch consumes only confirmed tracker counts plus bounded spatial output.
It deliberately does not own track IDs, detector calls or persistent state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.analytics.space_config import SpaceConfig


class SpaceAnalytics:
    """Build camera-space metrics without confusing pixel proxies with metres."""

    def __init__(self, config: SpaceConfig) -> None:
        self.config = config

    @staticmethod
    def _nonnegative_count(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
        return value

    def build(self, confirmed_active_count: int, spatial_statistics: Mapping[str, Any]) -> dict[str, object]:
        """Return only metrics justified by the supplied camera calibration.

        A missing capacity or floor area is represented by ``None``.  Values are
        never guessed from image pixels and occupancy is deliberately not capped:
        a real over-capacity situation should remain visible to the demonstrator.
        """

        confirmed = self._nonnegative_count(confirmed_active_count, "confirmed_active_count")
        spatial = dict(spatial_statistics)
        distribution_source = spatial.get("distribution", {})
        if not isinstance(distribution_source, Mapping):
            distribution_source = {}
        primary_raw = distribution_source.get("primary_zone_counts", {})
        if not isinstance(primary_raw, Mapping):
            primary_raw = {}
        primary_counts = {
            zone.name: self._nonnegative_count(primary_raw.get(zone.name, 0), f"primary count for {zone.name}")
            for zone in self.config.zones
        }
        assigned = sum(primary_counts.values())
        reported_outside = distribution_source.get("outside_zone_count", confirmed - assigned)
        outside = self._nonnegative_count(reported_outside, "outside_zone_count")
        # Spatial output can be unavailable (or created under a previous
        # config) for a frame. Preserve the invariant instead of emitting a
        # distribution whose percentages do not sum to 100%.
        if assigned + outside != confirmed:
            outside = max(0, confirmed - assigned)
        if assigned > confirmed:
            # This should only happen with malformed external statistics. Do
            # not silently turn an inclusive overlap count into a percentage.
            raise ValueError("primary zone counts cannot exceed confirmed_active_count.")

        capacity = self.config.capacity
        occupancy_rate = confirmed / capacity if capacity is not None else None
        area_m2 = self.config.visible_floor_area_m2
        people_per_m2 = confirmed / area_m2 if area_m2 is not None else None
        denominator = max(confirmed, 1)
        zone_items = [
            {
                "name": zone.name,
                "count": primary_counts[zone.name],
                "share": round(primary_counts[zone.name] / denominator, 4) if confirmed else 0.0,
                "percentage": round(primary_counts[zone.name] * 100.0 / denominator, 1) if confirmed else 0.0,
            }
            for zone in self.config.zones
        ]
        outside_payload = {
            "count": outside,
            "share": round(outside / denominator, 4) if confirmed else 0.0,
            "percentage": round(outside * 100.0 / denominator, 1) if confirmed else 0.0,
        }

        return {
            "name": self.config.name,
            "occupancy": {
                "confirmed_active_count": confirmed,
                "capacity": capacity,
                "rate": round(occupancy_rate, 4) if occupancy_rate is not None else None,
                "percentage": round(occupancy_rate * 100.0, 1) if occupancy_rate is not None else None,
                "over_capacity": bool(occupancy_rate is not None and occupancy_rate > 1.0),
                "calibrated": capacity is not None,
            },
            "physical_density": {
                "visible_floor_area_m2": area_m2,
                "people_per_m2": round(people_per_m2, 3) if people_per_m2 is not None else None,
                "calibrated": area_m2 is not None,
                "unit": "confirmed_people_per_m2" if area_m2 is not None else None,
            },
            "crowd_distribution": {
                "assignment_policy": str(
                    distribution_source.get("assignment_policy", "highest_priority_then_config_order")
                ),
                "denominator_confirmed_count": confirmed,
                "zones": zone_items,
                "outside_zones": outside_payload,
                "total_count": assigned + outside,
            },
        }
