from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field

import numpy as np


GENDER_LABELS = ("female", "male")
_NEVER = -10**9


@dataclass
class TrackState:
    """Mutable evidence retained while a tracker ID can still be recovered."""

    first_seen_frame: int = -1
    active_frames: int = 0
    last_seen_frame: int = -1
    last_face_attempt_frame: int = _NEVER
    last_gender_success_frame: int = _NEVER
    consecutive_face_misses: int = 0
    # YuNet can find a face while the face classifier remains below its calibrated
    # threshold. Keep this separate from detection misses so the body fallback can
    # help without pretending that an uncertain face result is a face-detection error.
    consecutive_face_unknown_predictions: int = 0
    # The stream telemetry must be able to distinguish a face that has not yet
    # collected enough temporal evidence from a genuine aggregate decision that
    # remains uncertain.  This is deliberately independent from the body branch.
    face_evidence_state: str = "no_face_evidence"
    last_body_fallback_reason: str = "none"
    gender_observations: int = 0
    gender_refresh_observations: int = 0
    weighted_logits: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    weight_sum: float = 0.0
    final_gender: str = "unknown"
    gender_confidence: float = 0.0
    gender_frozen: bool = False
    last_body_attempt_frame: int = _NEVER
    last_body_success_frame: int = _NEVER
    body_gender_observations: int = 0
    body_gender_refresh_observations: int = 0
    body_weighted_logits: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    body_weight_sum: float = 0.0
    body_final_gender: str = "unknown"
    body_gender_confidence: float = 0.0
    body_gender_frozen: bool = False
    # Router state is independent of gender evidence. It prevents a person box
    # near a size threshold from switching face/body branches every frame.
    attribute_route: str = "unknown"
    attribute_route_candidate: str = "unknown"
    attribute_route_candidate_frames: int = 0
    last_attribute_route_frame: int = _NEVER
    face_cooldown_until_frame: int = _NEVER
    trajectory: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=120))


@dataclass(frozen=True)
class FinalizedTrack:
    """Small immutable summary retained after trajectory/logits have been released."""

    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    active_frames: int
    gender: str
    gender_confidence: float
    gender_observations: int
    gender_source: str
    trajectory: tuple[tuple[float, float], ...]


class TrackStateManager:
    """Accumulates per-track gender evidence and bounds stream-state memory."""

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        minimum_observations: int = 5,
        maximum_observations: int = 8,
        stable_confidence: float = 0.90,
        success_interval: int = 3,
        refresh_interval: int = 60,
        trajectory_length: int = 120,
        finalized_history_size: int = 1_000,
        ema_momentum: float = 0.70,
        body_confidence_threshold: float | None = None,
        body_minimum_observations: int | None = None,
        body_maximum_observations: int | None = None,
        body_stable_confidence: float | None = None,
        body_success_interval: int | None = None,
        body_refresh_interval: int | None = None,
        body_unknown_retry_interval: int | None = None,
        body_face_refresh_interval: int | None = None,
    ) -> None:
        if not 0.5 < confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must be greater than 0.5 and below 1.0")
        if minimum_observations < 1:
            raise ValueError("minimum_observations must be positive")
        if maximum_observations < minimum_observations:
            raise ValueError("maximum_observations must be at least minimum_observations")
        if not confidence_threshold <= stable_confidence <= 1.0:
            raise ValueError("stable_confidence must be between confidence_threshold and 1.0")
        if success_interval < 1:
            raise ValueError("success_interval must be positive")
        if refresh_interval < 1:
            raise ValueError("refresh_interval must be positive")
        if trajectory_length < 2:
            raise ValueError("trajectory_length must be at least 2")
        if finalized_history_size < 0:
            raise ValueError("finalized_history_size cannot be negative")
        if not 0.0 <= ema_momentum < 1.0:
            raise ValueError("ema_momentum must be in [0, 1)")
        self.confidence_threshold = confidence_threshold
        self.minimum_observations = minimum_observations
        self.maximum_observations = maximum_observations
        self.stable_confidence = stable_confidence
        self.success_interval = success_interval
        self.refresh_interval = refresh_interval
        self.trajectory_length = trajectory_length
        self.finalized_history_size = finalized_history_size
        self.ema_momentum = ema_momentum
        self.body_confidence_threshold = (
            confidence_threshold if body_confidence_threshold is None else body_confidence_threshold
        )
        self.body_minimum_observations = (
            minimum_observations if body_minimum_observations is None else body_minimum_observations
        )
        self.body_maximum_observations = (
            maximum_observations if body_maximum_observations is None else body_maximum_observations
        )
        self.body_stable_confidence = (
            stable_confidence if body_stable_confidence is None else body_stable_confidence
        )
        self.body_success_interval = success_interval if body_success_interval is None else body_success_interval
        self.body_refresh_interval = refresh_interval if body_refresh_interval is None else body_refresh_interval
        self.body_unknown_retry_interval = (
            self.body_refresh_interval if body_unknown_retry_interval is None else body_unknown_retry_interval
        )
        self.body_face_refresh_interval = (
            self.body_refresh_interval if body_face_refresh_interval is None else body_face_refresh_interval
        )
        if not 0.5 < self.body_confidence_threshold < 1.0:
            raise ValueError("body_confidence_threshold must be greater than 0.5 and below 1.0")
        if self.body_minimum_observations < 1:
            raise ValueError("body_minimum_observations must be positive")
        if self.body_maximum_observations < self.body_minimum_observations:
            raise ValueError("body_maximum_observations must be at least body_minimum_observations")
        if not self.body_confidence_threshold <= self.body_stable_confidence <= 1.0:
            raise ValueError("body_stable_confidence must be between body_confidence_threshold and 1.0")
        if (
            self.body_success_interval < 1
            or self.body_refresh_interval < 1
            or self.body_unknown_retry_interval < 1
            or self.body_face_refresh_interval < 1
        ):
            raise ValueError("body success and retry intervals must be positive")
        self.track_states: dict[int, TrackState] = {}
        self.finalized_tracks: OrderedDict[int, FinalizedTrack] = OrderedDict()

    def reset(self) -> None:
        self.track_states.clear()
        self.finalized_tracks.clear()

    def _new_state(self) -> TrackState:
        state = TrackState()
        state.trajectory = deque(maxlen=self.trajectory_length)
        return state

    def ensure_track(self, track_id: int) -> TrackState:
        if track_id not in self.track_states:
            # A revived/reused ID belongs to the active stream again; its archived summary is historical only.
            self.finalized_tracks.pop(track_id, None)
            self.track_states[track_id] = self._new_state()
        return self.track_states[track_id]

    def update_track(self, track_id: int, bbox: tuple[int, int, int, int], frame_index: int) -> None:
        """Record one detected track and its foot point."""
        state = self.ensure_track(track_id)
        x1, _y1, x2, y2 = bbox
        if state.first_seen_frame < 0:
            state.first_seen_frame = frame_index
        state.active_frames += 1
        state.last_seen_frame = frame_index
        state.trajectory.append(((x1 + x2) / 2.0, float(y2)))

    def should_attempt_face(self, track_id: int, frame_index: int, retry_interval: int) -> bool:
        """Return whether this track may spend a YuNet attempt on the current frame."""
        if retry_interval < 1:
            raise ValueError("retry_interval must be positive")
        state = self.ensure_track(track_id)
        # A stable body label is useful fallback evidence. Do not keep spending a YuNet
        # call every normal face-retry interval merely to upgrade it; probe face much
        # less often, while preserving the opportunity for face evidence to take priority.
        if (
            state.final_gender == "unknown"
            and state.body_gender_frozen
            and state.body_final_gender != "unknown"
        ):
            return frame_index - state.last_face_attempt_frame >= self.body_face_refresh_interval
        if frame_index - state.last_face_attempt_frame < retry_interval:
            return False
        if not state.gender_frozen:
            if state.last_gender_success_frame > _NEVER:
                return frame_index - state.last_gender_success_frame >= self.success_interval
            return True
        # A frozen result is refreshed occasionally, but face misses do not reset that timer.
        return frame_index - state.last_gender_success_frame >= self.refresh_interval

    def should_attempt_gender(self, track_id: int, frame_index: int, interval: int) -> bool:
        """Compatibility alias for callers that still use the old name."""
        return self.should_attempt_face(track_id, frame_index, interval)

    def mark_face_attempt(self, track_id: int, frame_index: int) -> None:
        self.ensure_track(track_id).last_face_attempt_frame = frame_index

    def record_face_miss(self, track_id: int) -> None:
        """Record a completed YuNet attempt with no usable face crop."""
        state = self.ensure_track(track_id)
        state.consecutive_face_misses += 1
        state.face_evidence_state = "face_detection_miss"

    def face_evidence_state(self, track_id: int) -> str:
        """Return the current, non-image telemetry state of face evidence."""

        return self.ensure_track(track_id).face_evidence_state

    def body_fallback_reason(
        self,
        track_id: int,
        required_face_misses: int,
        required_face_unknown_predictions: int | None = None,
        face_fallback_min_observations: int | None = None,
    ) -> str | None:
        """Explain whether body may run as a fallback without mixing branch logits.

        Face detection misses are a separate failure mode and may open body early.
        In contrast, classifier uncertainty is only actionable after a full face
        evidence budget has been collected.
        """

        if required_face_misses < 1:
            raise ValueError("required_face_misses must be positive")
        if required_face_unknown_predictions is not None and required_face_unknown_predictions < 1:
            raise ValueError("required_face_unknown_predictions must be positive when supplied")
        if face_fallback_min_observations is not None and face_fallback_min_observations < self.minimum_observations:
            raise ValueError("face_fallback_min_observations must be at least minimum_observations")

        state = self.ensure_track(track_id)
        if state.final_gender != "unknown":
            return None
        if state.consecutive_face_misses >= required_face_misses:
            return "face_detection_miss"
        if (
            required_face_unknown_predictions is not None
            and state.gender_observations >= (face_fallback_min_observations or self.minimum_observations)
            and state.consecutive_face_unknown_predictions >= required_face_unknown_predictions
        ):
            return "face_uncertain_after_minimum_observations"
        return None

    def set_face_cooldown(self, track_id: int, until_frame: int) -> None:
        """Suppress futile YuNet retries while retaining a future face upgrade path."""
        state = self.ensure_track(track_id)
        state.face_cooldown_until_frame = max(state.face_cooldown_until_frame, int(until_frame))

    def face_cooldown_active(self, track_id: int, frame_index: int) -> bool:
        return frame_index < self.ensure_track(track_id).face_cooldown_until_frame

    def update_attribute_route(
        self,
        track_id: int,
        desired_route: str,
        frame_index: int,
        *,
        stability_frames: int = 1,
    ) -> tuple[str, bool]:
        """Apply route hysteresis and return the active route plus change flag.

        This owns only dispatch policy; face and body logits remain completely
        separate in their existing accumulators.
        """

        if stability_frames < 1:
            raise ValueError("stability_frames must be positive")
        state = self.ensure_track(track_id)
        desired_route = str(desired_route)
        previous = state.attribute_route
        if desired_route == previous:
            state.attribute_route_candidate = desired_route
            state.attribute_route_candidate_frames = 0
        elif desired_route == state.attribute_route_candidate:
            state.attribute_route_candidate_frames += 1
        else:
            state.attribute_route_candidate = desired_route
            state.attribute_route_candidate_frames = 1

        # The first useful route should not wait for hysteresis. Later changes
        # require repeated geometry evidence.
        if previous == "unknown" or state.attribute_route_candidate_frames >= stability_frames:
            state.attribute_route = desired_route
            state.attribute_route_candidate = desired_route
            state.attribute_route_candidate_frames = 0
        state.last_attribute_route_frame = int(frame_index)
        return state.attribute_route, state.attribute_route != previous

    def mark_gender_attempt(self, track_id: int, frame_index: int) -> None:
        """Compatibility alias: an old gender attempt was actually a face-detection attempt."""
        self.mark_face_attempt(track_id, frame_index)

    def should_attempt_body(
        self,
        track_id: int,
        frame_index: int,
        retry_interval: int,
        required_face_misses: int,
        required_face_unknown_predictions: int | None = None,
        face_fallback_min_observations: int | None = None,
    ) -> bool:
        """Use body after repeated YuNet misses *or* unresolved face-classifier evidence."""
        if retry_interval < 1:
            raise ValueError("body retry_interval must be positive")
        state = self.ensure_track(track_id)
        reason = self.body_fallback_reason(
            track_id,
            required_face_misses,
            required_face_unknown_predictions,
            face_fallback_min_observations,
        )
        state.last_body_fallback_reason = reason or "none"
        if reason is None:
            return False
        return self.should_attempt_body_direct(track_id, frame_index, retry_interval)

    def should_attempt_body_direct(self, track_id: int, frame_index: int, retry_interval: int) -> bool:
        """Schedule body evidence without requiring a preceding face failure.

        Used by the adaptive router for far tracks and by the explicit
        body-only camera profile. It preserves existing confidence, temporal
        aggregation, freeze and refresh semantics.
        """

        if retry_interval < 1:
            raise ValueError("body retry_interval must be positive")
        state = self.ensure_track(track_id)
        if state.final_gender != "unknown":
            return False
        if frame_index - state.last_body_attempt_frame < retry_interval:
            return False
        if state.body_gender_frozen:
            return frame_index - state.last_body_success_frame >= self.body_refresh_interval
        # Do not permanently lock a low-quality/occluded person as unknown.
        # After the initial evidence budget, reduce its retry cadence instead
        # of spending a body forward pass every success interval.
        if state.body_gender_observations >= self.body_maximum_observations:
            return frame_index - state.last_body_success_frame >= self.body_unknown_retry_interval
        if state.last_body_success_frame > _NEVER:
            return frame_index - state.last_body_success_frame >= self.body_success_interval
        return True

    def mark_body_attempt(self, track_id: int, frame_index: int) -> None:
        self.ensure_track(track_id).last_body_attempt_frame = frame_index

    def update_gender(
        self,
        track_id: int,
        logits: np.ndarray,
        face_detection_confidence: float,
        gender_confidence: float,
        face_width: int,
        frame_index: int | None = None,
    ) -> bool:
        """Add one valid classifier result and return whether it was accepted as evidence."""
        state = self.ensure_track(track_id)
        logits = np.asarray(logits, dtype=np.float32)
        if logits.shape != (2,):
            raise ValueError("gender logits must have shape (2,)")

        face_quality = min(1.0, max(0.0, face_width / 96.0))
        weight = max(0.0, face_detection_confidence) * max(0.0, gender_confidence) * face_quality
        if weight <= 0.0:
            return False

        momentum = self.ema_momentum
        state.weighted_logits = momentum * state.weighted_logits + (1.0 - momentum) * logits * weight
        state.weight_sum = momentum * state.weight_sum + (1.0 - momentum) * weight
        if state.gender_frozen:
            state.gender_refresh_observations += 1
        else:
            state.gender_observations += 1
        if frame_index is not None:
            state.last_gender_success_frame = frame_index
        state.consecutive_face_misses = 0
        self._aggregate_gender(state)
        if state.gender_observations < self.minimum_observations:
            state.face_evidence_state = "insufficient_observations"
            # A young face track is expected to be unresolved; it must not spend
            # the body fallback budget before an aggregate face decision exists.
            state.consecutive_face_unknown_predictions = 0
        elif state.final_gender == "unknown":
            state.face_evidence_state = "uncertain_after_minimum_observations"
            state.consecutive_face_unknown_predictions += 1
        else:
            state.face_evidence_state = "resolved"
            state.consecutive_face_unknown_predictions = 0
        return True

    def update_body_gender(
        self,
        track_id: int,
        logits: np.ndarray,
        gender_confidence: float,
        person_height: int,
        frame_index: int | None = None,
    ) -> bool:
        """Add body-only evidence without ever mixing it into face logits."""
        state = self.ensure_track(track_id)
        logits = np.asarray(logits, dtype=np.float32)
        if logits.shape != (2,):
            raise ValueError("body gender logits must have shape (2,)")
        # Full-body labels become unreliable for tiny detections. This is a separate quality
        # model from face confidence; no YuNet confidence is reused here.
        body_quality = min(1.0, max(0.0, person_height / 160.0))
        weight = max(0.0, gender_confidence) * body_quality
        if weight <= 0.0:
            return False

        momentum = self.ema_momentum
        state.body_weighted_logits = momentum * state.body_weighted_logits + (1.0 - momentum) * logits * weight
        state.body_weight_sum = momentum * state.body_weight_sum + (1.0 - momentum) * weight
        if state.body_gender_frozen:
            state.body_gender_refresh_observations += 1
        else:
            state.body_gender_observations += 1
        if frame_index is not None:
            state.last_body_success_frame = frame_index
        self._aggregate_body_gender(state)
        return True

    def _aggregate_gender(self, state: TrackState) -> None:
        if state.gender_observations < self.minimum_observations or state.weight_sum <= 1e-8:
            state.final_gender = "unknown"
            state.gender_confidence = 0.0
            return

        average_logits = state.weighted_logits / state.weight_sum
        shifted = average_logits - np.max(average_logits)
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
        state.gender_confidence = confidence
        state.final_gender = GENDER_LABELS[predicted_index] if confidence >= self.confidence_threshold else "unknown"
        state.gender_frozen = (
            state.gender_observations >= self.maximum_observations
            or (state.final_gender != "unknown" and confidence >= self.stable_confidence)
        )

    def _aggregate_body_gender(self, state: TrackState) -> None:
        if state.body_gender_observations < self.body_minimum_observations or state.body_weight_sum <= 1e-8:
            state.body_final_gender = "unknown"
            state.body_gender_confidence = 0.0
            return

        average_logits = state.body_weighted_logits / state.body_weight_sum
        shifted = average_logits - np.max(average_logits)
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
        state.body_gender_confidence = confidence
        state.body_final_gender = (
            GENDER_LABELS[predicted_index] if confidence >= self.body_confidence_threshold else "unknown"
        )
        # A confident label may freeze, but an unresolved track must remain
        # eligible for bounded retries: later frames can be less occluded.
        state.body_gender_frozen = state.body_final_gender != "unknown" and (
            state.body_gender_observations >= self.body_maximum_observations
            or confidence >= self.body_stable_confidence
        )

    def is_gender_frozen(self, track_id: int) -> bool:
        state = self.track_states.get(track_id)
        return bool(state and state.gender_frozen)

    def is_body_gender_frozen(self, track_id: int) -> bool:
        state = self.track_states.get(track_id)
        return bool(state and state.body_gender_frozen)

    def gender_observation_count(self, track_id: int) -> int:
        state = self.track_states.get(track_id)
        return state.gender_observations if state else 0

    def body_gender_observation_count(self, track_id: int) -> int:
        state = self.track_states.get(track_id)
        return state.body_gender_observations if state else 0

    @staticmethod
    def _resolved_gender(state: TrackState) -> tuple[str, float, str]:
        """Face evidence wins; body is a fallback only and logits are never merged."""
        if state.final_gender != "unknown":
            return state.final_gender, state.gender_confidence, "face"
        if state.body_final_gender != "unknown":
            return state.body_final_gender, state.body_gender_confidence, "body"
        return "unknown", 0.0, "unknown"

    def get_gender(self, track_id: int) -> tuple[str, float]:
        state = self.track_states.get(track_id)
        if state is not None:
            gender, confidence, _source = self._resolved_gender(state)
            return gender, confidence
        finalized = self.finalized_tracks.get(track_id)
        if finalized is not None:
            return finalized.gender, finalized.gender_confidence
        return "unknown", 0.0

    def get_gender_source(self, track_id: int) -> str:
        state = self.track_states.get(track_id)
        if state is not None:
            return self._resolved_gender(state)[2]
        finalized = self.finalized_tracks.get(track_id)
        return finalized.gender_source if finalized is not None else "unknown"

    def get_trajectory(self, track_id: int) -> list[tuple[float, float]]:
        state = self.track_states.get(track_id)
        return list(state.trajectory) if state else []

    def active_frames(self, track_id: int) -> int:
        state = self.track_states.get(track_id)
        return state.active_frames if state else 0

    def is_stable(self, track_id: int, minimum_frames: int) -> bool:
        return self.active_frames(track_id) >= minimum_frames

    def prune_stale(self, frame_index: int, max_missing_frames: int) -> list[FinalizedTrack]:
        """Finalize tracks once they are beyond the tracker-recovery window.

        A track is still retained when it has been missing for exactly `max_missing_frames`;
        it is finalized one frame later. This keeps state alive for the entire tracker buffer.
        """
        if max_missing_frames < 0:
            raise ValueError("max_missing_frames cannot be negative")
        stale_ids = [
            track_id
            for track_id, state in self.track_states.items()
            if frame_index - state.last_seen_frame > max_missing_frames
        ]
        finalized: list[FinalizedTrack] = []
        for track_id in stale_ids:
            state = self.track_states.pop(track_id)
            gender, confidence, source = self._resolved_gender(state)
            summary = FinalizedTrack(
                track_id=track_id,
                first_seen_frame=state.first_seen_frame,
                last_seen_frame=state.last_seen_frame,
                active_frames=state.active_frames,
                gender=gender,
                gender_confidence=confidence,
                gender_observations=state.gender_observations + state.body_gender_observations,
                gender_source=source,
                trajectory=tuple(state.trajectory),
            )
            finalized.append(summary)
            if self.finalized_history_size:
                self.finalized_tracks[track_id] = summary
                self.finalized_tracks.move_to_end(track_id)
                while len(self.finalized_tracks) > self.finalized_history_size:
                    self.finalized_tracks.popitem(last=False)
        return finalized
