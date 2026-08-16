from __future__ import annotations

from src.tracking.track_state import FinalizedTrack, TrackStateManager


Point = tuple[float, float]


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a: Point, b: Point, c: Point, d: Point, epsilon: float = 1e-6) -> bool:
    """Return true for a proper segment intersection; collinear motion is ignored."""
    ab_c = _cross(a, b, c)
    ab_d = _cross(a, b, d)
    cd_a = _cross(c, d, a)
    cd_b = _cross(c, d, b)
    return (ab_c * ab_d < -epsilon) and (cd_a * cd_b < -epsilon)


class LineCrossingCounter:
    """Counts a confirmed track when its foot point geometrically crosses a virtual line."""

    def __init__(self, track_state_manager: TrackStateManager, min_active_frames: int = 15) -> None:
        self.tsm = track_state_manager
        self.min_active_frames = min_active_frames
        self.reset()

    def reset(self) -> None:
        self.in_count = 0
        self.out_count = 0
        self.gender_counts_in = {"female": 0, "male": 0, "unknown": 0}
        self.gender_counts_out = {"female": 0, "male": 0, "unknown": 0}
        self._counted_directions: dict[str, set[int]] = {"IN": set(), "OUT": set()}
        self._pending_directions: dict[int, set[str]] = {}

    @staticmethod
    def _direction(line_start: Point, line_end: Point, previous: Point, current: Point) -> str | None:
        movement_cross = (line_end[0] - line_start[0]) * (current[1] - previous[1]) - (
            line_end[1] - line_start[1]
        ) * (current[0] - previous[0])
        if abs(movement_cross) < 1e-6:
            return None
        # With a horizontal left-to-right line, downward movement is IN.
        return "IN" if movement_cross > 0 else "OUT"

    def _commit(self, track_id: int, direction: str) -> None:
        if track_id in self._counted_directions[direction]:
            return
        gender, _ = self.tsm.get_gender(track_id)
        self._counted_directions[direction].add(track_id)
        if direction == "IN":
            self.in_count += 1
            self.gender_counts_in[gender] += 1
        else:
            self.out_count += 1
            self.gender_counts_out[gender] += 1

    def update(self, active_track_ids: list[int], line_start: Point, line_end: Point) -> None:
        active_set = set(active_track_ids)
        for track_id in active_track_ids:
            trajectory = self.tsm.get_trajectory(track_id)
            if len(trajectory) < 2:
                continue
            previous, current = trajectory[-2:]
            if _segments_intersect(previous, current, line_start, line_end):
                direction = self._direction(line_start, line_end, previous, current)
                if direction:
                    self._pending_directions.setdefault(track_id, set()).add(direction)

            if self.tsm.is_stable(track_id, self.min_active_frames):
                for direction in self._pending_directions.pop(track_id, set()):
                    self._commit(track_id, direction)

        # A non-confirmed track that vanished never becomes a counted event.
        self._pending_directions = {
            track_id: directions
            for track_id, directions in self._pending_directions.items()
            if track_id in active_set
        }

    def consume_finalized(self, finalized_tracks: list[FinalizedTrack]) -> None:
        """Release per-ID deduplication state after the tracker can no longer revive that ID."""
        for summary in finalized_tracks:
            track_id = summary.track_id
            self._pending_directions.pop(track_id, None)
            self._counted_directions["IN"].discard(track_id)
            self._counted_directions["OUT"].discard(track_id)

    def get_counts(self) -> dict:
        return {
            "in": self.in_count,
            "out": self.out_count,
            "gender_in": self.gender_counts_in.copy(),
            "gender_out": self.gender_counts_out.copy(),
        }
