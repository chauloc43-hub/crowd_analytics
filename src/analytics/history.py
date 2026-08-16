"""Bounded session history and event-driven flow analytics.

This module intentionally has no dependency on the tracker, UI, or persistence
layer.  A stream owner supplies the number of *confirmed* active tracks each
frame and emits an ``IN``/``OUT`` event only when its line-crossing component
commits one.  That separation keeps history metrics from changing tracker
behaviour and makes a reset deterministic for a new video/webcam session.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from typing import Literal, Sequence


FlowDirection = Literal["IN", "OUT"]


@dataclass(frozen=True)
class OccupancySnapshot:
    """One bounded, display-friendly sample of confirmed occupancy."""

    timestamp_seconds: float
    confirmed_occupancy: int
    total_in: int
    total_out: int
    occupancy_rate: float | None = None
    people_per_m2: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "confirmed_occupancy": self.confirmed_occupancy,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "net_flow": self.total_in - self.total_out,
            "occupancy_rate": round(self.occupancy_rate, 4) if self.occupancy_rate is not None else None,
            "people_per_m2": round(self.people_per_m2, 4) if self.people_per_m2 is not None else None,
        }


@dataclass(frozen=True)
class FlowEvent:
    """A confirmed virtual-line crossing supplied by the caller."""

    timestamp_seconds: float
    direction: FlowDirection


class HistoryEngine:
    """Keep bounded occupancy snapshots and rolling line-crossing rates.

    ``update`` should be called once per processed frame with the confirmed
    active count.  It uses every update for the session average while retaining
    only display snapshots at ``snapshot_interval_seconds``.  The average is
    time-weighted: an occupancy value is assumed to hold until the next update.

    Flow is event-driven.  Call :meth:`record_flow_event` for every committed
    crossing, or pass frame-local ``in_events``/``out_events`` to ``update``.
    Events older than the largest configured rolling window are discarded, and
    the remaining queue is hard-capped by ``max_flow_events``.
    """

    def __init__(
        self,
        *,
        snapshot_interval_seconds: float = 5.0,
        max_snapshots: int = 720,
        flow_windows_seconds: Sequence[float] = (60.0, 300.0),
        max_flow_events: int = 2_000,
    ) -> None:
        if snapshot_interval_seconds <= 0.0:
            raise ValueError("snapshot_interval_seconds must be positive.")
        if max_snapshots < 1:
            raise ValueError("max_snapshots must be at least 1.")
        if max_flow_events < 1:
            raise ValueError("max_flow_events must be at least 1.")
        if not flow_windows_seconds:
            raise ValueError("flow_windows_seconds must contain at least one window.")

        normalized_windows = tuple(sorted({float(window) for window in flow_windows_seconds}))
        if any(window <= 0.0 for window in normalized_windows):
            raise ValueError("Every flow window must be positive.")

        self.snapshot_interval_seconds = float(snapshot_interval_seconds)
        self.max_snapshots = int(max_snapshots)
        self.flow_windows_seconds = normalized_windows
        self.max_flow_events = int(max_flow_events)
        self.reset()

    def reset(self) -> None:
        """Forget all stream-local state; this module never writes to disk."""
        self._snapshots: deque[OccupancySnapshot] = deque(maxlen=self.max_snapshots)
        self._flow_events: deque[FlowEvent] = deque()
        self._latest_timestamp_seconds: float | None = None
        self._first_timestamp_seconds: float | None = None
        self._last_snapshot_timestamp_seconds: float | None = None
        self._latest_confirmed_occupancy = 0
        self._previous_occupancy_timestamp_seconds: float | None = None
        self._previous_confirmed_occupancy = 0
        self._occupancy_area_seconds = 0.0
        self._occupancy_observed_seconds = 0.0
        self._peak_confirmed_occupancy = 0
        self._total_in = 0
        self._total_out = 0
        self._events_dropped_due_capacity = 0
        self._events_pruned_due_age = 0

    @staticmethod
    def _validate_timestamp(timestamp_seconds: float) -> float:
        timestamp = float(timestamp_seconds)
        if timestamp < 0.0:
            raise ValueError("timestamp_seconds cannot be negative.")
        return timestamp

    @staticmethod
    def _validate_count(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
        return value

    @staticmethod
    def _optional_nonnegative_number(value: float | None, name: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a non-negative finite number or None.")
        result = float(value)
        if not isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be a non-negative finite number or None.")
        return result

    def _observe_timestamp(self, timestamp_seconds: float) -> float:
        timestamp = self._validate_timestamp(timestamp_seconds)
        latest = self._latest_timestamp_seconds
        if latest is not None and timestamp < latest:
            raise ValueError("HistoryEngine requires non-decreasing timestamps.")
        if self._first_timestamp_seconds is None:
            self._first_timestamp_seconds = timestamp
        self._latest_timestamp_seconds = timestamp
        return timestamp

    def _prune_expired_flow_events(self, now_seconds: float) -> None:
        cutoff = now_seconds - self.flow_windows_seconds[-1]
        while self._flow_events and self._flow_events[0].timestamp_seconds < cutoff:
            self._flow_events.popleft()
            self._events_pruned_due_age += 1

    def _append_flow_event(self, timestamp_seconds: float, direction: FlowDirection) -> FlowEvent:
        self._prune_expired_flow_events(timestamp_seconds)
        if len(self._flow_events) >= self.max_flow_events:
            self._flow_events.popleft()
            self._events_dropped_due_capacity += 1
        event = FlowEvent(timestamp_seconds=timestamp_seconds, direction=direction)
        self._flow_events.append(event)
        if direction == "IN":
            self._total_in += 1
        else:
            self._total_out += 1
        return event

    def record_flow_event(self, direction: str, timestamp_seconds: float) -> FlowEvent:
        """Record one already-confirmed virtual-line crossing.

        The caller owns deduplication and confirmation.  This engine therefore
        does not infer flow from changes in occupancy, which would count tracker
        fragmentation and missed detections as people entering or leaving.
        """
        normalized_direction = str(direction).upper()
        if normalized_direction not in {"IN", "OUT"}:
            raise ValueError("direction must be 'IN' or 'OUT'.")
        timestamp = self._observe_timestamp(timestamp_seconds)
        return self._append_flow_event(timestamp, normalized_direction)  # type: ignore[arg-type]

    def record_flow_events(
        self,
        timestamp_seconds: float,
        *,
        in_events: int = 0,
        out_events: int = 0,
    ) -> tuple[FlowEvent, ...]:
        """Record a frame-local batch of already-confirmed events."""
        in_count = self._validate_count(in_events, "in_events")
        out_count = self._validate_count(out_events, "out_events")
        timestamp = self._observe_timestamp(timestamp_seconds)
        return tuple(
            self._append_flow_event(timestamp, "IN")
            for _ in range(in_count)
        ) + tuple(
            self._append_flow_event(timestamp, "OUT")
            for _ in range(out_count)
        )

    def _maybe_snapshot(
        self,
        timestamp_seconds: float,
        *,
        occupancy_rate: float | None,
        people_per_m2: float | None,
    ) -> OccupancySnapshot | None:
        last = self._last_snapshot_timestamp_seconds
        if last is not None and timestamp_seconds - last < self.snapshot_interval_seconds:
            return None
        snapshot = OccupancySnapshot(
            timestamp_seconds=timestamp_seconds,
            confirmed_occupancy=self._latest_confirmed_occupancy,
            total_in=self._total_in,
            total_out=self._total_out,
            occupancy_rate=occupancy_rate,
            people_per_m2=people_per_m2,
        )
        self._snapshots.append(snapshot)
        self._last_snapshot_timestamp_seconds = timestamp_seconds
        return snapshot

    def update(
        self,
        timestamp_seconds: float,
        confirmed_occupancy: int,
        *,
        in_events: int = 0,
        out_events: int = 0,
        occupancy_rate: float | None = None,
        people_per_m2: float | None = None,
    ) -> OccupancySnapshot | None:
        """Update confirmed occupancy and create a snapshot when its interval elapses.

        Passing same-frame events here ensures that a newly created snapshot
        contains their cumulative totals.  For direct event emission after an
        update, use :meth:`record_flow_event` instead.
        """
        confirmed_count = self._validate_count(confirmed_occupancy, "confirmed_occupancy")
        in_count = self._validate_count(in_events, "in_events")
        out_count = self._validate_count(out_events, "out_events")
        validated_occupancy_rate = self._optional_nonnegative_number(occupancy_rate, "occupancy_rate")
        validated_people_per_m2 = self._optional_nonnegative_number(people_per_m2, "people_per_m2")
        timestamp = self._observe_timestamp(timestamp_seconds)

        previous_timestamp = self._previous_occupancy_timestamp_seconds
        if previous_timestamp is not None:
            elapsed = timestamp - previous_timestamp
            self._occupancy_area_seconds += self._previous_confirmed_occupancy * elapsed
            self._occupancy_observed_seconds += elapsed
        self._previous_occupancy_timestamp_seconds = timestamp
        self._previous_confirmed_occupancy = confirmed_count
        self._latest_confirmed_occupancy = confirmed_count
        self._peak_confirmed_occupancy = max(self._peak_confirmed_occupancy, confirmed_count)
        self._prune_expired_flow_events(timestamp)
        for _ in range(in_count):
            self._append_flow_event(timestamp, "IN")
        for _ in range(out_count):
            self._append_flow_event(timestamp, "OUT")
        return self._maybe_snapshot(
            timestamp,
            occupancy_rate=validated_occupancy_rate,
            people_per_m2=validated_people_per_m2,
        )

    def snapshots(self) -> tuple[OccupancySnapshot, ...]:
        """Return an immutable view of the bounded display history."""
        return tuple(self._snapshots)

    def _flow_window_statistics(self, now_seconds: float, window_seconds: float) -> dict[str, object]:
        cutoff = now_seconds - window_seconds
        events = [event for event in self._flow_events if event.timestamp_seconds >= cutoff]
        in_count = sum(event.direction == "IN" for event in events)
        out_count = sum(event.direction == "OUT" for event in events)
        first_timestamp = self._first_timestamp_seconds
        observed_window_seconds = (
            min(window_seconds, max(0.0, now_seconds - first_timestamp))
            if first_timestamp is not None
            else 0.0
        )
        scale = 60.0 / observed_window_seconds if observed_window_seconds > 0.0 else 0.0
        return {
            "window_seconds": window_seconds,
            "observed_window_seconds": round(observed_window_seconds, 3),
            "in_count": int(in_count),
            "out_count": int(out_count),
            "net_count": int(in_count - out_count),
            "in_per_minute": round(in_count * scale, 3),
            "out_per_minute": round(out_count * scale, 3),
            "net_per_minute": round((in_count - out_count) * scale, 3),
        }

    def get_statistics(
        self,
        timestamp_seconds: float | None = None,
        *,
        include_snapshots: bool = True,
    ) -> dict[str, object]:
        """Return session aggregates, bounded snapshots, and rolling flow rates.

        ``timestamp_seconds`` is optional for a reporting caller that needs rates
        at a later current time.  It never mutates the engine; real updates and
        events are the only operations that advance session state. Real-time
        callers can set ``include_snapshots=False`` to avoid serialising the
        bounded timeline on every webcam callback.
        """
        if timestamp_seconds is None:
            now = self._latest_timestamp_seconds if self._latest_timestamp_seconds is not None else 0.0
        else:
            now = self._validate_timestamp(timestamp_seconds)
            if self._latest_timestamp_seconds is not None and now < self._latest_timestamp_seconds:
                raise ValueError("timestamp_seconds cannot precede the latest observed timestamp.")

        if self._occupancy_observed_seconds > 0.0:
            average = self._occupancy_area_seconds / self._occupancy_observed_seconds
        else:
            average = float(self._latest_confirmed_occupancy)
        flow_windows = [self._flow_window_statistics(now, window) for window in self.flow_windows_seconds]
        return {
            "current_confirmed_occupancy": self._latest_confirmed_occupancy,
            "peak_confirmed_occupancy": self._peak_confirmed_occupancy,
            "average_confirmed_occupancy": round(average, 3),
            "occupancy_observed_seconds": round(self._occupancy_observed_seconds, 3),
            "history": {
                "snapshot_interval_seconds": self.snapshot_interval_seconds,
                "retained_snapshot_count": len(self._snapshots),
                "max_snapshots": self.max_snapshots,
                "snapshots": [snapshot.as_dict() for snapshot in self._snapshots] if include_snapshots else [],
                "snapshots_omitted": not include_snapshots,
            },
            "flow": {
                "total_in": self._total_in,
                "total_out": self._total_out,
                "net_total": self._total_in - self._total_out,
                "retained_event_count": len(self._flow_events),
                "max_flow_events": self.max_flow_events,
                "events_dropped_due_capacity": self._events_dropped_due_capacity,
                "events_pruned_due_age": self._events_pruned_due_age,
                "windows": flow_windows,
            },
        }
