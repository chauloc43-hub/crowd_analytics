"""Data contracts for one batched body-classifier forward pass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BodyGenderCandidate:
    """One clipped person crop selected by the body-fallback scheduler."""

    track_id: int
    crop: np.ndarray
    person_width: int
    person_height: int


@dataclass(frozen=True)
class BodyGenderEvidence:
    """Calibrated logits and confidence mapped back to the source track."""

    track_id: int
    logits: np.ndarray
    gender_confidence: float
    person_width: int
    person_height: int


def map_body_gender_batch(
    candidates: list[BodyGenderCandidate], logits: np.ndarray, confidences: np.ndarray
) -> list[BodyGenderEvidence]:
    """Validate one-to-one body batch mapping before any track state is updated."""
    if not candidates:
        if np.asarray(logits).size or np.asarray(confidences).size:
            raise ValueError("Empty body candidate batches must not contain classifier outputs.")
        return []
    track_ids = [candidate.track_id for candidate in candidates]
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("A body gender batch cannot contain duplicate track IDs.")
    logits = np.asarray(logits, dtype=np.float32)
    confidences = np.asarray(confidences, dtype=np.float32)
    if logits.shape != (len(candidates), 2):
        raise ValueError("Body gender model returned an unexpected batch-logit shape.")
    if confidences.shape != (len(candidates),):
        raise ValueError("Body gender model returned an unexpected confidence-batch shape.")
    return [
        BodyGenderEvidence(
            track_id=candidate.track_id,
            logits=logits[index],
            gender_confidence=float(confidences[index]),
            person_width=candidate.person_width,
            person_height=candidate.person_height,
        )
        for index, candidate in enumerate(candidates)
    ]
