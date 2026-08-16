"""Bounded, latest-frame execution for one live video stream.

The browser is allowed to send webcam updates faster than model inference can
finish.  Feeding every one of those updates into a tracker creates two bad
failure modes: stale frames accumulate (high display latency), or concurrent
calls mutate the same persistent tracker.  ``LiveFrameProcessor`` owns one
pipeline and one worker thread, accepts at most one pending frame, and replaces
that pending frame with the newest arrival.

The class deliberately does *not* reset a tracker when frames are replaced.
Only frames actually handed to ``pipeline.process_frame`` advance its tracker
and analytics state.  Their timestamps are server-receive monotonic timestamps
and are clamped to remain non-decreasing, so downstream trajectory analytics
remain time-ordered even if concurrent HTTP arrivals are scheduled oddly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Condition, RLock, Thread
from time import monotonic
from typing import Any, Callable, Protocol

import numpy as np


class LivePipeline(Protocol):
    """Small pipeline contract needed by the live scheduler."""

    def process_frame(
        self, frame: np.ndarray, timestamp_seconds: float | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class LiveFrameResult:
    """Most recent successfully processed result for one webcam session."""

    annotated_frame: np.ndarray
    stats: dict[str, Any]
    sequence: int
    submitted_at: float
    completed_at: float


@dataclass(frozen=True)
class _PendingFrame:
    frame: np.ndarray
    sequence: int
    submitted_at: float
    generation: int


def _percentile(values: deque[float], percentile: float) -> float:
    """Return a small dependency-free linear percentile for telemetry."""

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


class LiveFrameProcessor:
    """Run one persistent tracker with a capacity-one newest-frame mailbox.

    ``submit`` is intentionally quick and thread-safe: it never runs model
    inference.  A private worker owns the pipeline, so no two frames can touch
    a FastTracker (or any other stateful Ultralytics tracker) concurrently.
    The most recent output can be returned by a lightweight Gradio callback
    while the worker processes the next newest frame.
    """

    POLICY = "latest_frame_only"
    TIMESTAMP_SOURCE = "server_receive_monotonic"

    def __init__(
        self,
        pipeline: LivePipeline,
        *,
        cadence_seconds: float = 0.15,
        profiling_window: int = 120,
        clock: Callable[[], float] = monotonic,
        worker_name: str = "crowd-live-frame-worker",
    ) -> None:
        if cadence_seconds <= 0.0:
            raise ValueError("cadence_seconds must be positive.")
        if profiling_window < 1:
            raise ValueError("profiling_window must be positive.")
        self.pipeline = pipeline
        self.cadence_seconds = float(cadence_seconds)
        self._clock = clock
        # Serializes reset/close themselves. The worker never takes this lock:
        # reset waits for ``_processing`` to clear first, and close joins the
        # worker first, so model methods cannot overlap either lifecycle call.
        self._lifecycle_lock = RLock()
        self._condition = Condition(RLock())
        self._pending: _PendingFrame | None = None
        self._processing = False
        self._resetting = False
        self._closed = False
        self._generation = 0
        self._next_sequence = 0
        self._last_pipeline_timestamp: float | None = None
        self._last_result: LiveFrameResult | None = None
        self._last_error: str | None = None

        self._received = 0
        self._processed = 0
        self._failed = 0
        self._dropped_replaced = 0
        self._dropped_reset = 0
        self._dropped_shutdown = 0
        self._timestamp_adjustments = 0
        self._submitted_at_history: deque[float] = deque(maxlen=profiling_window)
        self._arrival_interval_ms: deque[float] = deque(maxlen=profiling_window)
        self._queue_wait_ms: deque[float] = deque(maxlen=profiling_window)
        self._processing_ms: deque[float] = deque(maxlen=profiling_window)
        self._end_to_end_ms: deque[float] = deque(maxlen=profiling_window)

        self._worker = Thread(target=self._run, name=worker_name, daemon=True)
        self._worker.start()

    def submit(self, frame: np.ndarray, *, submitted_at: float | None = None) -> int | None:
        """Offer a webcam frame and return its sequence, or ``None`` if closed.

        The caller should pass a frame that remains valid until the worker has
        consumed it.  ``cv2.cvtColor`` in the app creates such a private BGR
        array.  Avoiding an additional full-frame copy keeps the ingress path
        cheap; there is still only one bounded pending reference.
        """

        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("Live frames must be numpy HxWxC arrays.")
        received_at = self._clock() if submitted_at is None else float(submitted_at)
        if received_at < 0.0:
            raise ValueError("submitted_at cannot be negative.")
        with self._condition:
            if self._closed:
                return None
            if self._resetting:
                # Reset is a deliberate session boundary.  Do not allow an
                # old browser callback to re-populate the mailbox mid-reset.
                self._dropped_reset += 1
                return None
            self._received += 1
            if self._submitted_at_history:
                interval = max(0.0, received_at - self._submitted_at_history[-1]) * 1_000.0
                self._arrival_interval_ms.append(interval)
            self._submitted_at_history.append(received_at)
            self._next_sequence += 1
            if self._pending is not None:
                self._dropped_replaced += 1
            self._pending = _PendingFrame(
                frame=frame,
                sequence=self._next_sequence,
                submitted_at=received_at,
                generation=self._generation,
            )
            self._condition.notify_all()
            return self._next_sequence

    def latest_result(self) -> LiveFrameResult | None:
        """Return the completed frame, if the worker has produced one yet."""

        with self._condition:
            return self._last_result

    def telemetry(self) -> dict[str, Any]:
        """Return aggregate scheduler telemetry without retaining images."""

        with self._condition:
            return self._telemetry_locked(self._clock())

    def reset(self) -> None:
        """Clear pending work, then reset the owned pipeline exactly once.

        A running inference is allowed to finish before ``pipeline.reset`` so
        a tracker is never reset concurrently with ``process_frame``.  Its
        result is discarded at the session boundary.
        """

        with self._lifecycle_lock:
            with self._condition:
                if self._closed:
                    return
                self._resetting = True
                self._generation += 1
                if self._pending is not None:
                    self._dropped_reset += 1
                    self._pending = None
                while self._processing:
                    self._condition.wait()
            try:
                self.pipeline.reset()
            finally:
                with self._condition:
                    self._last_pipeline_timestamp = None
                    self._last_result = None
                    self._last_error = None
                    self._reset_telemetry_locked()
                    self._resetting = False
                    self._condition.notify_all()

    def close(self) -> None:
        """Stop the worker before releasing the pipeline's tracker/model state."""

        with self._lifecycle_lock:
            with self._condition:
                if self._closed:
                    return
                self._closed = True
                self._generation += 1
                if self._pending is not None:
                    self._dropped_shutdown += 1
                    self._pending = None
                self._condition.notify_all()
            self._worker.join()
            close = getattr(self.pipeline, "close", None)
            if callable(close):
                close()
            else:
                self.pipeline.reset()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._closed and (self._pending is None or self._resetting):
                    self._condition.wait()
                if self._closed:
                    return
                pending = self._pending
                self._pending = None
                self._processing = True
                started_at = self._clock()
                queue_wait_ms = max(0.0, started_at - pending.submitted_at) * 1_000.0
                timestamp_seconds = pending.submitted_at
                if self._last_pipeline_timestamp is not None and timestamp_seconds < self._last_pipeline_timestamp:
                    timestamp_seconds = self._last_pipeline_timestamp
                    self._timestamp_adjustments += 1
                self._last_pipeline_timestamp = timestamp_seconds

            try:
                annotated, stats = self.pipeline.process_frame(
                    pending.frame,
                    timestamp_seconds=timestamp_seconds,
                )
                error: str | None = None
            except Exception as exc:  # Keep the latest-frame worker alive after one bad callback.
                annotated, stats = None, None
                error = f"{type(exc).__name__}: {exc}"
            completed_at = self._clock()
            processing_ms = max(0.0, completed_at - started_at) * 1_000.0
            end_to_end_ms = max(0.0, completed_at - pending.submitted_at) * 1_000.0

            with self._condition:
                self._processing = False
                current_generation = pending.generation == self._generation and not self._resetting and not self._closed
                if current_generation:
                    self._queue_wait_ms.append(queue_wait_ms)
                    self._processing_ms.append(processing_ms)
                    self._end_to_end_ms.append(end_to_end_ms)
                    if error is None and annotated is not None and stats is not None:
                        self._processed += 1
                        self._last_error = None
                        telemetry = self._telemetry_locked(completed_at)
                        self._last_result = LiveFrameResult(
                            annotated_frame=annotated,
                            stats=self._with_telemetry(stats, telemetry),
                            sequence=pending.sequence,
                            submitted_at=pending.submitted_at,
                            completed_at=completed_at,
                        )
                    else:
                        self._failed += 1
                        self._last_error = error or "Unknown live-frame processing error."
                else:
                    if self._closed:
                        self._dropped_shutdown += 1
                    else:
                        self._dropped_reset += 1
                self._condition.notify_all()

    def _reset_telemetry_locked(self) -> None:
        self._received = 0
        self._processed = 0
        self._failed = 0
        self._dropped_replaced = 0
        self._dropped_reset = 0
        self._dropped_shutdown = 0
        self._timestamp_adjustments = 0
        self._submitted_at_history.clear()
        self._arrival_interval_ms.clear()
        self._queue_wait_ms.clear()
        self._processing_ms.clear()
        self._end_to_end_ms.clear()

    def _telemetry_locked(self, now: float) -> dict[str, Any]:
        latest_age_ms = (
            max(0.0, now - self._last_result.completed_at) * 1_000.0
            if self._last_result is not None
            else None
        )
        return {
            "policy": self.POLICY,
            "timestamp_source": self.TIMESTAMP_SOURCE,
            "queue_capacity": 1,
            "configured_cadence_ms": round(self.cadence_seconds * 1_000.0, 2),
            "frames_received": self._received,
            "frames_processed": self._processed,
            "frames_failed": self._failed,
            "frames_dropped_replaced": self._dropped_replaced,
            "frames_dropped_reset": self._dropped_reset,
            "frames_dropped_shutdown": self._dropped_shutdown,
            "frames_dropped_total": self._dropped_replaced + self._dropped_reset + self._dropped_shutdown,
            "pending_frames": int(self._pending is not None),
            "is_processing": self._processing,
            "timestamp_adjustments": self._timestamp_adjustments,
            "last_completed_sequence": self._last_result.sequence if self._last_result is not None else None,
            "last_result_age_ms": round(latest_age_ms, 2) if latest_age_ms is not None else None,
            "arrival_cadence_ms": self._summary(self._arrival_interval_ms),
            "queue_wait_ms": self._summary(self._queue_wait_ms),
            "worker_processing_ms": self._summary(self._processing_ms),
            "end_to_end_ms": self._summary(self._end_to_end_ms),
            "last_error": self._last_error,
        }

    @staticmethod
    def _summary(values: deque[float]) -> dict[str, float | int]:
        if not values:
            return {"samples": 0, "last": 0.0, "p50": 0.0, "p95": 0.0}
        return {
            "samples": len(values),
            "last": round(float(values[-1]), 2),
            "p50": round(_percentile(values, 50.0), 2),
            "p95": round(_percentile(values, 95.0), 2),
        }

    @staticmethod
    def _with_telemetry(stats: dict[str, Any], telemetry: dict[str, Any]) -> dict[str, Any]:
        """Copy only the report envelope, keeping pipeline payloads intact."""

        output = dict(stats)
        runtime = dict(stats.get("runtime", {}))
        runtime["live_stream"] = telemetry
        output["runtime"] = runtime
        return output
