"""Conservative stream-scoped person identities above volatile tracker IDs.

The selected FastTracker owns short-term association.  This module does *not*
claim to identify a person across cameras or sessions, and deliberately does
not use the gender classifier as biometric evidence.  It only carries a
session-local ``person_id`` across a short tracker-ID break when position,
motion and box geometry make the recovery unambiguous.  Ambiguous and long-gap
cases receive a new ID rather than silently merging two people.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite, log
from typing import Mapping


BBox = tuple[int, int, int, int]
ReferenceBBox = tuple[float, float, float, float]
Point = tuple[float, float]


def _finite_number(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ValueError(f"person_identity.{name} must be a finite number.")
    number = float(value)
    if minimum is not None and number < minimum:
        comparator = "positive" if minimum > 0.0 else "non-negative"
        raise ValueError(f"person_identity.{name} must be {comparator}.")
    return number


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"person_identity.{name} must be a positive integer.")
    return int(value)


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"person_identity.{name} must be a non-negative integer.")
    return int(value)


@dataclass(frozen=True)
class PersonIdentityConfig:
    """Validated, camera-plane thresholds for session-local ID recovery."""

    enabled: bool = True
    max_inactive_frames: int = 90
    max_reassociation_gap_frames: int = 30
    base_match_distance_reference_px: float = 28.0
    max_match_distance_reference_px: float = 96.0
    max_prediction_speed_reference_px_per_frame: float = 10.0
    prediction_horizon_frames: int = 8
    max_continuity_distance_reference_px: float = 100.0
    max_area_ratio: float = 2.25
    max_aspect_ratio: float = 1.80
    match_score_threshold: float = 0.72
    ambiguity_margin: float = 0.12
    finalized_history_size: int = 1_000

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> "PersonIdentityConfig":
        if not isinstance(values, (Mapping, type(None))):
            raise ValueError("person_identity must be a YAML mapping when configured.")
        raw = {} if values is None else dict(values)
        defaults = cls()
        enabled = raw.get("enabled", defaults.enabled)
        if not isinstance(enabled, bool):
            raise ValueError("person_identity.enabled must be a boolean.")
        config = cls(
            enabled=enabled,
            max_inactive_frames=_positive_int(
                raw.get("max_inactive_frames", defaults.max_inactive_frames), "max_inactive_frames"
            ),
            max_reassociation_gap_frames=_positive_int(
                raw.get("max_reassociation_gap_frames", defaults.max_reassociation_gap_frames),
                "max_reassociation_gap_frames",
            ),
            base_match_distance_reference_px=_finite_number(
                raw.get("base_match_distance_reference_px", defaults.base_match_distance_reference_px),
                "base_match_distance_reference_px",
                minimum=0.0,
            ),
            max_match_distance_reference_px=_finite_number(
                raw.get("max_match_distance_reference_px", defaults.max_match_distance_reference_px),
                "max_match_distance_reference_px",
                minimum=0.0,
            ),
            max_prediction_speed_reference_px_per_frame=_finite_number(
                raw.get(
                    "max_prediction_speed_reference_px_per_frame",
                    defaults.max_prediction_speed_reference_px_per_frame,
                ),
                "max_prediction_speed_reference_px_per_frame",
                minimum=0.0,
            ),
            prediction_horizon_frames=_positive_int(
                raw.get("prediction_horizon_frames", defaults.prediction_horizon_frames),
                "prediction_horizon_frames",
            ),
            max_continuity_distance_reference_px=_finite_number(
                raw.get(
                    "max_continuity_distance_reference_px",
                    defaults.max_continuity_distance_reference_px,
                ),
                "max_continuity_distance_reference_px",
                minimum=0.0,
            ),
            max_area_ratio=_finite_number(raw.get("max_area_ratio", defaults.max_area_ratio), "max_area_ratio", minimum=1.0),
            max_aspect_ratio=_finite_number(
                raw.get("max_aspect_ratio", defaults.max_aspect_ratio), "max_aspect_ratio", minimum=1.0
            ),
            match_score_threshold=_finite_number(
                raw.get("match_score_threshold", defaults.match_score_threshold),
                "match_score_threshold",
                minimum=0.0,
            ),
            ambiguity_margin=_finite_number(
                raw.get("ambiguity_margin", defaults.ambiguity_margin), "ambiguity_margin", minimum=0.0
            ),
            finalized_history_size=_nonnegative_int(
                raw.get("finalized_history_size", defaults.finalized_history_size), "finalized_history_size"
            ),
        )
        if config.max_reassociation_gap_frames > config.max_inactive_frames:
            raise ValueError(
                "person_identity.max_reassociation_gap_frames cannot exceed max_inactive_frames."
            )
        if config.base_match_distance_reference_px > config.max_match_distance_reference_px:
            raise ValueError(
                "person_identity.base_match_distance_reference_px cannot exceed max_match_distance_reference_px."
            )
        if config.match_score_threshold > 1.0:
            raise ValueError("person_identity.match_score_threshold must be at most 1.0.")
        return config


@dataclass
class _PersonRecord:
    person_id: int
    first_seen_frame: int
    last_seen_frame: int
    last_bbox: ReferenceBBox
    last_center: Point
    velocity: Point = (0.0, 0.0)
    tracker_recoveries: int = 0


class PersonIdentityResolver:
    """Resolve stable, pseudonymous IDs for one stream session only.

    The resolver is intentionally conservative:

    * a continuous local tracker ID keeps its person ID unless the box jumps
      implausibly far;
    * a changed local ID can recover a recent person only through a one-to-one,
      geometry- and motion-gated *unambiguous* assignment;
    * a long changed-ID gap or any ambiguity is treated as a new person. A
      retained FastTracker numeric ID may resume within the bounded retention
      window, but only through the same geometry/motion gate.

    That policy favors an occasional new ``person_id`` over merging different
    people.  There is no cross-session persistence and no biometric matching.
    """

    def __init__(
        self,
        reference_size: tuple[int, int],
        config: PersonIdentityConfig | Mapping[str, object] | None = None,
    ) -> None:
        reference_width, reference_height = reference_size
        if reference_width <= 0 or reference_height <= 0:
            raise ValueError("reference_size must contain positive width and height.")
        self.reference_size = (int(reference_width), int(reference_height))
        self.config = (
            config
            if isinstance(config, PersonIdentityConfig)
            else PersonIdentityConfig.from_mapping(config)
        )
        self.reset()

    def reset(self) -> None:
        """Begin a new stream session; IDs are never retained across reset."""

        self._records: dict[int, _PersonRecord] = {}
        self._active_track_to_person: dict[int, int] = {}
        # A tracker can intentionally hide an existing local ID while a track
        # is lost. Keep that weak link only within the bounded stream window;
        # its geometry must still pass the normal recovery gate before reuse.
        self._dormant_track_to_person: dict[int, int] = {}
        self._next_person_id = 1
        self._last_frame_index = -1
        self._created_person_count = 0
        self._recovered_track_count = 0
        self._ambiguous_match_rejections = 0
        self._continuity_breaks = 0
        self._expired_person_count = 0
        self._finalized_person_ids: list[int] = []

    @staticmethod
    def _validate_bbox(track_id: int, bbox: object) -> BBox:
        if not isinstance(track_id, int) or isinstance(track_id, bool):
            raise ValueError("track IDs passed to person_identity must be integers.")
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            raise ValueError("person_identity boxes must be (x1, y1, x2, y2) tuples.")
        try:
            x1, y1, x2, y2 = (int(value) for value in bbox)
        except (TypeError, ValueError) as error:
            raise ValueError("person_identity boxes must contain integer coordinates.") from error
        if x2 <= x1 or y2 <= y1:
            raise ValueError("person_identity boxes must have positive width and height.")
        return x1, y1, x2, y2

    def _to_reference(self, bbox: BBox, frame_size: tuple[int, int]) -> ReferenceBBox:
        frame_width, frame_height = frame_size
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame_size must contain positive width and height.")
        reference_width, reference_height = self.reference_size
        x1, y1, x2, y2 = bbox
        return (
            x1 * reference_width / frame_width,
            y1 * reference_height / frame_height,
            x2 * reference_width / frame_width,
            y2 * reference_height / frame_height,
        )

    @staticmethod
    def _center(bbox: ReferenceBBox) -> Point:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    @staticmethod
    def _area(bbox: ReferenceBBox) -> float:
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    @staticmethod
    def _aspect_ratio(bbox: ReferenceBBox) -> float:
        return (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])

    @staticmethod
    def _iou(first: ReferenceBBox, second: ReferenceBBox) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        union = PersonIdentityResolver._area(first) + PersonIdentityResolver._area(second) - intersection
        return intersection / union if union > 0.0 else 0.0

    def _shape_is_compatible(self, record: _PersonRecord, bbox: ReferenceBBox) -> bool:
        previous_area = self._area(record.last_bbox)
        current_area = self._area(bbox)
        if previous_area <= 0.0 or current_area <= 0.0:
            return False
        area_ratio = max(previous_area / current_area, current_area / previous_area)
        if area_ratio > self.config.max_area_ratio:
            return False
        previous_aspect = self._aspect_ratio(record.last_bbox)
        current_aspect = self._aspect_ratio(bbox)
        aspect_ratio = max(previous_aspect / current_aspect, current_aspect / previous_aspect)
        return aspect_ratio <= self.config.max_aspect_ratio

    def _predicted_center(self, record: _PersonRecord, gap_frames: int) -> Point:
        horizon = min(max(0, gap_frames), self.config.prediction_horizon_frames)
        return (
            record.last_center[0] + record.velocity[0] * horizon,
            record.last_center[1] + record.velocity[1] * horizon,
        )

    def _match_distance_limit(self, gap_frames: int) -> float:
        return min(
            self.config.max_match_distance_reference_px,
            self.config.base_match_distance_reference_px
            + self.config.max_prediction_speed_reference_px_per_frame
            * min(max(0, gap_frames), self.config.prediction_horizon_frames),
        )

    def _continuity_is_plausible(self, record: _PersonRecord, bbox: ReferenceBBox) -> bool:
        if not self._shape_is_compatible(record, bbox):
            return False
        center = self._center(bbox)
        predicted = self._predicted_center(record, 1)
        return hypot(center[0] - predicted[0], center[1] - predicted[1]) <= self.config.max_continuity_distance_reference_px

    def _candidate_score(
        self,
        record: _PersonRecord,
        bbox: ReferenceBBox,
        frame_index: int,
        *,
        maximum_gap_frames: int | None = None,
    ) -> float | None:
        gap_frames = frame_index - record.last_seen_frame
        maximum_gap = self.config.max_reassociation_gap_frames if maximum_gap_frames is None else maximum_gap_frames
        if gap_frames < 1 or gap_frames > maximum_gap:
            return None
        if not self._shape_is_compatible(record, bbox):
            return None
        limit = self._match_distance_limit(gap_frames)
        if limit <= 0.0:
            return None
        center = self._center(bbox)
        predicted = self._predicted_center(record, gap_frames)
        distance_ratio = hypot(center[0] - predicted[0], center[1] - predicted[1]) / limit
        if distance_ratio > 1.0:
            return None
        area_ratio = max(self._area(record.last_bbox) / self._area(bbox), self._area(bbox) / self._area(record.last_bbox))
        # Shape compatibility above guarantees a positive finite denominator.
        size_penalty = log(area_ratio) / log(self.config.max_area_ratio) if self.config.max_area_ratio > 1.0 else 0.0
        iou_penalty = 1.0 - self._iou(record.last_bbox, bbox)
        score = 0.78 * distance_ratio + 0.15 * size_penalty + 0.07 * iou_penalty
        return score if score <= self.config.match_score_threshold else None

    def _bind(self, person_id: int, track_id: int, bbox: ReferenceBBox, frame_index: int, *, recovered: bool) -> None:
        record = self._records[person_id]
        # An identity can have exactly one active local track. Once it is bound
        # again, old dormant numeric IDs must not revive beside the new one.
        for dormant_track_id, dormant_person_id in list(self._dormant_track_to_person.items()):
            if dormant_person_id == person_id:
                self._dormant_track_to_person.pop(dormant_track_id, None)
        previous_center = record.last_center
        elapsed = max(1, frame_index - record.last_seen_frame)
        current_center = self._center(bbox)
        observed_velocity = (
            (current_center[0] - previous_center[0]) / elapsed,
            (current_center[1] - previous_center[1]) / elapsed,
        )
        # Preserve a little history during normal tracking, but let a short
        # recovery update the predicted direction for a subsequent break.
        if record.last_seen_frame >= record.first_seen_frame:
            record.velocity = (
                0.65 * record.velocity[0] + 0.35 * observed_velocity[0],
                0.65 * record.velocity[1] + 0.35 * observed_velocity[1],
            )
        else:
            record.velocity = observed_velocity
        record.last_seen_frame = frame_index
        record.last_bbox = bbox
        record.last_center = current_center
        self._active_track_to_person[track_id] = person_id
        if recovered:
            record.tracker_recoveries += 1
            self._recovered_track_count += 1

    def _create_person(self, track_id: int, bbox: ReferenceBBox, frame_index: int) -> int:
        person_id = self._next_person_id
        self._next_person_id += 1
        center = self._center(bbox)
        self._records[person_id] = _PersonRecord(
            person_id=person_id,
            first_seen_frame=frame_index,
            last_seen_frame=frame_index,
            last_bbox=bbox,
            last_center=center,
        )
        self._active_track_to_person[track_id] = person_id
        self._created_person_count += 1
        return person_id

    def _expire_stale_records(self, frame_index: int) -> None:
        active_people = set(self._active_track_to_person.values())
        expired = [
            person_id
            for person_id, record in self._records.items()
            if person_id not in active_people and frame_index - record.last_seen_frame > self.config.max_inactive_frames
        ]
        for person_id in expired:
            self._records.pop(person_id, None)
            self._expired_person_count += 1
            if self.config.finalized_history_size:
                self._finalized_person_ids.append(person_id)
                if len(self._finalized_person_ids) > self.config.finalized_history_size:
                    del self._finalized_person_ids[0]

    @staticmethod
    def _has_clear_winner(candidates: list[tuple[float, int]], margin: float) -> bool:
        return len(candidates) == 1 or candidates[1][0] - candidates[0][0] >= margin

    def update(
        self,
        active_tracks: Mapping[int, BBox],
        frame_index: int,
        frame_size: tuple[int, int],
    ) -> dict[int, int]:
        """Return a stable ``{track_id: person_id}`` mapping for this frame."""

        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("frame_index passed to person_identity must be a non-negative integer.")
        if frame_index <= self._last_frame_index:
            raise ValueError("person_identity frame_index must increase; call reset() for a new stream.")
        self._last_frame_index = frame_index
        normalized_tracks = {
            track_id: self._to_reference(self._validate_bbox(track_id, bbox), frame_size)
            for track_id, bbox in active_tracks.items()
        }
        if not self.config.enabled:
            self._active_track_to_person = {}
            return {}

        previous_active = self._active_track_to_person
        self._active_track_to_person = {}
        resolved: dict[int, int] = {}
        used_person_ids: set[int] = set()
        unresolved: dict[int, ReferenceBBox] = {}

        # A missing local tracker ID becomes a dormant hint. It is not trusted
        # by itself: returning through the same numeric ID still has to pass
        # geometry/motion gating below, which protects against tracker-ID reuse.
        for track_id, person_id in previous_active.items():
            record = self._records.get(person_id)
            if (
                record is None
                or track_id not in normalized_tracks
                or record.last_seen_frame != frame_index - 1
            ):
                self._dormant_track_to_person[track_id] = person_id
        for track_id, person_id in list(self._dormant_track_to_person.items()):
            record = self._records.get(person_id)
            if record is None or frame_index - record.last_seen_frame > self.config.max_inactive_frames:
                self._dormant_track_to_person.pop(track_id, None)

        # A tracker ID present on adjacent frames normally remains authoritative.
        # A large discontinuity is handled as a fresh association instead, which
        # prevents an ID transfer from silently becoming a person-ID transfer.
        for track_id, bbox in normalized_tracks.items():
            person_id = previous_active.get(track_id)
            record = self._records.get(person_id) if person_id is not None else None
            if (
                record is not None
                and person_id not in used_person_ids
                and record.last_seen_frame == frame_index - 1
                and self._continuity_is_plausible(record, bbox)
            ):
                self._bind(person_id, track_id, bbox, frame_index, recovered=False)
                resolved[track_id] = person_id
                used_person_ids.add(person_id)
                self._dormant_track_to_person.pop(track_id, None)
            else:
                if person_id is not None and record is not None:
                    self._continuity_breaks += 1
                    self._dormant_track_to_person[track_id] = person_id
                dormant_person_id = self._dormant_track_to_person.get(track_id)
                dormant_record = self._records.get(dormant_person_id) if dormant_person_id is not None else None
                dormant_score = (
                    self._candidate_score(
                        dormant_record,
                        bbox,
                        frame_index,
                        maximum_gap_frames=self.config.max_inactive_frames,
                    )
                    if dormant_record is not None and dormant_person_id not in used_person_ids
                    else None
                )
                if dormant_score is not None and dormant_person_id is not None:
                    self._bind(dormant_person_id, track_id, bbox, frame_index, recovered=True)
                    resolved[track_id] = dormant_person_id
                    used_person_ids.add(dormant_person_id)
                    self._dormant_track_to_person.pop(track_id, None)
                else:
                    unresolved[track_id] = bbox

        # Score every recent dormant identity.  A match must be the clear best
        # option for both sides; otherwise leave it unmatched and allocate a
        # new pseudonymous ID.  This makes crowd crossings deliberately safe.
        candidates_by_track: dict[int, list[tuple[float, int]]] = {track_id: [] for track_id in unresolved}
        candidates_by_person: dict[int, list[tuple[float, int]]] = {}
        for track_id, bbox in unresolved.items():
            for person_id, record in self._records.items():
                if person_id in used_person_ids or record.last_seen_frame >= frame_index:
                    continue
                score = self._candidate_score(record, bbox, frame_index)
                if score is None:
                    continue
                candidates_by_track[track_id].append((score, person_id))
                candidates_by_person.setdefault(person_id, []).append((score, track_id))
        for candidates in candidates_by_track.values():
            candidates.sort()
        for candidates in candidates_by_person.values():
            candidates.sort()

        accepted: dict[int, int] = {}
        ambiguous_tracks: set[int] = set()
        for track_id, candidates in candidates_by_track.items():
            if not candidates:
                continue
            best_score, person_id = candidates[0]
            if not self._has_clear_winner(candidates, self.config.ambiguity_margin):
                ambiguous_tracks.add(track_id)
                continue
            person_candidates = candidates_by_person[person_id]
            if person_candidates[0] != (best_score, track_id) or not self._has_clear_winner(
                person_candidates, self.config.ambiguity_margin
            ):
                ambiguous_tracks.add(track_id)
                continue
            accepted[track_id] = person_id
        self._ambiguous_match_rejections += len(ambiguous_tracks)

        for track_id, bbox in unresolved.items():
            person_id = accepted.get(track_id)
            if person_id is None:
                person_id = self._create_person(track_id, bbox, frame_index)
            else:
                self._bind(person_id, track_id, bbox, frame_index, recovered=True)
            resolved[track_id] = person_id
            self._dormant_track_to_person.pop(track_id, None)

        self._expire_stale_records(frame_index)
        return dict(sorted(resolved.items()))

    def person_id_for_track(self, track_id: int) -> int | None:
        """Return the active stream-local identity for a local tracker ID."""

        return self._active_track_to_person.get(track_id)

    @staticmethod
    def display_label(person_id: int | None) -> str:
        return f"P{person_id:04d}" if person_id is not None else "P?"

    def get_statistics(self) -> dict[str, object]:
        """Expose bounded, non-biometric telemetry for the current session."""

        active = [
            {
                "track_id": track_id,
                "person_id": person_id,
                "person_label": self.display_label(person_id),
                "first_seen_frame": self._records[person_id].first_seen_frame,
                "tracker_recoveries": self._records[person_id].tracker_recoveries,
            }
            for track_id, person_id in sorted(self._active_track_to_person.items())
            if person_id in self._records
        ]
        return {
            "enabled": self.config.enabled,
            "scope": "stream_session",
            "association_policy": "conservative_geometry_motion_one_to_one",
            "uses_biometric_reid": False,
            "active_person_count": len(active),
            "unique_person_count": self._created_person_count,
            "active_track_to_person": {
                str(item["track_id"]): item["person_id"]
                for item in active
            },
            "active": active,
            "recovered_track_bindings": self._recovered_track_count,
            "ambiguous_match_rejections": self._ambiguous_match_rejections,
            "continuity_breaks": self._continuity_breaks,
            "expired_person_records": self._expired_person_count,
            "retained_inactive_person_records": max(0, len(self._records) - len(active)),
        }
