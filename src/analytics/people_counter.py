from __future__ import annotations

from src.tracking.track_state import FinalizedTrack, TrackStateManager


class PeopleCounter:
    """Reports active tracks and compact, finalized local-track aggregates for one stream."""

    def __init__(self, track_state_manager: TrackStateManager, min_active_frames: int = 15) -> None:
        self.tsm = track_state_manager
        self.min_active_frames = min_active_frames
        self.reset()

    def reset(self) -> None:
        self.confirmed_track_ids: set[int] = set()
        self._active_gender_by_track: dict[int, str] = {}
        self._finalized_confirmed_count = 0
        self._finalized_gender_counts = {"female": 0, "male": 0, "unknown": 0}

    def _snapshot_gender(self, track_id: int) -> None:
        gender, _ = self.tsm.get_gender(track_id)
        # Do not overwrite a known label with a transient unknown result.
        if gender != "unknown" or track_id not in self._active_gender_by_track:
            self._active_gender_by_track[track_id] = gender

    def update(self, active_track_ids: list[int]) -> None:
        for track_id in active_track_ids:
            if self.tsm.is_stable(track_id, self.min_active_frames):
                self.confirmed_track_ids.add(track_id)
                self._snapshot_gender(track_id)

    def consume_finalized(self, finalized_tracks: list[FinalizedTrack]) -> None:
        """Move finalized confirmed tracks into numeric aggregates and release their ID state."""
        for summary in finalized_tracks:
            track_id = summary.track_id
            if track_id not in self.confirmed_track_ids:
                continue
            self.confirmed_track_ids.remove(track_id)
            self._finalized_confirmed_count += 1
            gender = summary.gender
            if gender == "unknown":
                gender = self._active_gender_by_track.get(track_id, "unknown")
            self._finalized_gender_counts[gender] += 1
            self._active_gender_by_track.pop(track_id, None)

    def get_statistics(self, active_track_ids: list[int]) -> dict:
        for track_id in set(active_track_ids) & self.confirmed_track_ids:
            self._snapshot_gender(track_id)

        active_gender_counts = {"female": 0, "male": 0, "unknown": 0}
        for track_id in self.confirmed_track_ids:
            active_gender_counts[self._active_gender_by_track.get(track_id, "unknown")] += 1

        gender_counts = {
            gender: self._finalized_gender_counts[gender] + active_gender_counts[gender]
            for gender in active_gender_counts
        }
        known_count = gender_counts["female"] + gender_counts["male"]
        unique_count = self._finalized_confirmed_count + len(self.confirmed_track_ids)
        return {
            "current_count": len(set(active_track_ids)),
            "confirmed_active_count": len(set(active_track_ids) & self.confirmed_track_ids),
            "unique_track_count": unique_count,
            "gender_counts": gender_counts,
            "ratios": {
                "female": gender_counts["female"] / unique_count if unique_count else 0.0,
                "male": gender_counts["male"] / unique_count if unique_count else 0.0,
                "unknown": gender_counts["unknown"] / unique_count if unique_count else 0.0,
            },
            "known_gender_ratios": {
                "female": gender_counts["female"] / known_count if known_count else 0.0,
                "male": gender_counts["male"] / known_count if known_count else 0.0,
            },
            "gender_coverage": known_count / unique_count if unique_count else 0.0,
        }
