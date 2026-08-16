"""Data contracts for mapping one batched gender forward pass back to tracker IDs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceQuality:
    """Non-image quality telemetry captured for one YuNet face candidate.

    The crop itself deliberately stays out of telemetry.  These primitive
    signals are sufficient to audit why a face inference was accepted or
    rejected without retaining biometric image data.
    """

    detection_confidence: float
    face_width: int
    blur_variance: float
    brightness: float
    landmarks_valid: bool
    alignment_mode: str
    accepted: bool
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenderCandidate:
    """One valid face crop ready to join a classifier batch."""

    track_id: int
    crop: np.ndarray
    face_detection_confidence: float
    face_width: int
    quality: FaceQuality | None = None


@dataclass(frozen=True)
class GenderEvidence:
    """One row of a classifier batch mapped back to its source tracker ID."""

    track_id: int
    logits: np.ndarray
    gender_confidence: float
    face_detection_confidence: float
    face_width: int
    quality: FaceQuality | None = None


def map_gender_batch(
    candidates: list[GenderCandidate], logits: np.ndarray, confidences: np.ndarray
) -> list[GenderEvidence]:
    """Validate batched output and preserve the one-to-one candidate/track mapping."""
    if not candidates:
        if np.asarray(logits).size or np.asarray(confidences).size:
            raise ValueError("Empty candidate batches must not contain classifier outputs.")
        return []
    track_ids = [candidate.track_id for candidate in candidates]
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("A gender batch cannot contain duplicate track IDs.")
    logits = np.asarray(logits, dtype=np.float32)
    confidences = np.asarray(confidences, dtype=np.float32)
    if logits.shape != (len(candidates), 2):
        raise ValueError("Gender model returned an unexpected batch-logit shape.")
    if confidences.shape != (len(candidates),):
        raise ValueError("Gender model returned an unexpected confidence-batch shape.")
    return [
        GenderEvidence(
            track_id=candidate.track_id,
            logits=logits[index],
            gender_confidence=float(confidences[index]),
            face_detection_confidence=candidate.face_detection_confidence,
            face_width=candidate.face_width,
            quality=candidate.quality,
        )
        for index, candidate in enumerate(candidates)
    ]
