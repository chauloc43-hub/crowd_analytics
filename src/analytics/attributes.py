"""Lightweight summaries for the existing face-first/body-fallback attribute branch."""

from __future__ import annotations

from typing import Iterable

from src.tracking.track_state import TrackStateManager


class AttributeBranch:
    """Expose current visual-presentation evidence without treating it as identity data."""

    def __init__(self, track_state_manager: TrackStateManager) -> None:
        self.tsm = track_state_manager

    def summarize(self, active_track_ids: Iterable[int]) -> dict[str, object]:
        source_counts = {"face": 0, "body": 0, "unknown": 0}
        labels = {"female": 0, "male": 0, "unknown": 0}
        for track_id in active_track_ids:
            label, _confidence = self.tsm.get_gender(track_id)
            source = self.tsm.get_gender_source(track_id)
            labels[label] += 1
            source_counts[source if source in source_counts else "unknown"] += 1
        known = labels["female"] + labels["male"]
        total = sum(labels.values())
        return {
            "visual_presentation": {
                "labels": labels,
                "source_counts": source_counts,
                "coverage": round(known / total, 4) if total else 0.0,
                "note": "Binary visual-presentation estimates; not self-identified gender or identity.",
            }
        }
