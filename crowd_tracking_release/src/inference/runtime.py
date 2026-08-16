"""Model-owning runtime for one active Ultralytics tracking stream.

Ultralytics attaches a persistent tracker to the YOLO predictor used by ``model.track``.
Until detection and tracking are split into separate adapters, one ModelRuntime must therefore
serve only one active webcam/video stream. The analytics state remains outside this module.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.inference.body_gender_batch import BodyGenderCandidate, BodyGenderEvidence, map_body_gender_batch
from src.models.gender_classifier import GENDER_LABELS, IMAGE_NET_MEAN, IMAGE_NET_STD, GenderClassifier
from src.models.body_gender_classifier import BODY_MODEL_ARCHITECTURE, BODY_MODEL_ROLE, BodyGenderClassifier
from src.inference.gender_batch import FaceQuality, GenderCandidate, GenderEvidence, map_gender_batch
from src.tracking.tracker_config import (
    PERSIST_TRACKER_ACROSS_FRAMES,
    TrackerProfile,
    ensure_tracker_backend_available,
    reset_attached_trackers,
)


BBox = tuple[int, int, int, int]

_FACE_ALIGNMENT_MODES = frozenset({"eye_rotation", "raw_bbox", "five_point_similarity"})
_FIVE_POINT_TEMPLATE = np.asarray(
    [[0.30, 0.35], [0.70, 0.35], [0.50, 0.54], [0.36, 0.75], [0.64, 0.75]], dtype=np.float32
)


@dataclass(frozen=True)
class DetectorRecoverySettings:
    """Bounded high-detail fallback used after detector dropouts.

    The ordinary detector input confidence is deliberately left untouched by
    this policy: FastTracker needs its complete low-score association band.
    A recovery pass therefore spends a small, bounded number of frames at a
    larger input size instead of accepting noisier detections at a lower score
    threshold.
    """

    enabled: bool
    empty_frames_before_boost: int
    boosted_imgsz: int | None
    boost_frames: int
    cooldown_frames: int
    small_person_max_height_px: int | None
    small_person_frames_before_boost: int


@dataclass(frozen=True)
class DetectorSettings:
    """Validated detector-to-tracker settings for the active stream.

    ``tracker_input_confidence`` is deliberately named differently from a
    detector evaluation threshold. FastTracker needs low-score detections
    below ``track_high_thresh`` for association, so using a reporting
    threshold here silently changes tracker behaviour.
    """

    tracker_input_confidence: float
    evaluation_confidence: float | None
    iou_threshold: float
    imgsz: int | None
    use_half: bool
    end2end: bool | None
    max_det: int
    telemetry_enabled: bool
    telemetry_sample_every_n_frames: int
    require_full_low_score_recovery: bool
    recovery: DetectorRecoverySettings


def _probability_value(value: object, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"person_detector.{name} must be a numeric probability.")
    number = float(value)
    lower_bound = 0.0 if allow_zero else 0.0
    if not lower_bound <= number <= 1.0 or (not allow_zero and number <= 0.0):
        qualifier = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"person_detector.{name} must be in {qualifier}.")
    return number


def _integer_value(value: object, name: str, *, minimum: int = 0) -> int:
    """Validate an integer configuration value without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = f"at least {minimum}" if minimum else "zero or positive"
        raise ValueError(f"person_detector.{name} must be an integer {qualifier}.")
    return value


def resolve_detector_settings(detector: Mapping[str, object]) -> DetectorSettings:
    """Read the compact detector contract while accepting legacy YAML profiles.

    Existing configurations used ``confidence_threshold`` for the value sent
    to ``YOLO.track``.  New profiles use ``tracker_input_confidence`` so the
    distinction from detector-only reporting is explicit.  Keeping the legacy
    fallback avoids breaking an older exported YAML.
    """

    model_path = detector.get("model_path")
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("person_detector.model_path must be a non-empty string.")

    tracker_confidence = _probability_value(
        detector.get("tracker_input_confidence", detector.get("confidence_threshold", 0.10)),
        "tracker_input_confidence",
    )
    raw_evaluation_confidence = detector.get("evaluation_confidence")
    evaluation_confidence = (
        None
        if raw_evaluation_confidence is None
        else _probability_value(raw_evaluation_confidence, "evaluation_confidence", allow_zero=True)
    )
    iou_threshold = _probability_value(detector.get("iou_threshold", 0.60), "iou_threshold")

    raw_imgsz = detector.get("imgsz")
    if raw_imgsz is None:
        imgsz = None
    elif isinstance(raw_imgsz, bool) or not isinstance(raw_imgsz, int) or raw_imgsz < 32:
        raise ValueError("person_detector.imgsz must be an integer of at least 32 when provided.")
    else:
        imgsz = raw_imgsz

    raw_use_half = detector.get("use_half", False)
    if not isinstance(raw_use_half, bool):
        raise ValueError("person_detector.use_half must be boolean.")
    raw_end2end = detector.get("end2end")
    if raw_end2end is not None and not isinstance(raw_end2end, bool):
        raise ValueError("person_detector.end2end must be boolean or null.")
    raw_max_det = detector.get("max_det", 300)
    if isinstance(raw_max_det, bool) or not isinstance(raw_max_det, int) or raw_max_det < 1:
        raise ValueError("person_detector.max_det must be a positive integer.")
    raw_telemetry = detector.get("telemetry_enabled", False)
    if not isinstance(raw_telemetry, bool):
        raise ValueError("person_detector.telemetry_enabled must be boolean.")

    telemetry_sample_every_n_frames = _integer_value(
        detector.get("telemetry_sample_every_n_frames", 1),
        "telemetry_sample_every_n_frames",
        minimum=1,
    )
    raw_low_score_guard = detector.get("require_full_low_score_recovery", False)
    if not isinstance(raw_low_score_guard, bool):
        raise ValueError("person_detector.require_full_low_score_recovery must be boolean.")

    raw_recovery = detector.get("recovery", {})
    if raw_recovery is None:
        raw_recovery = {}
    if not isinstance(raw_recovery, Mapping):
        raise ValueError("person_detector.recovery must be a mapping when provided.")
    recovery_enabled = raw_recovery.get("enabled", False)
    if not isinstance(recovery_enabled, bool):
        raise ValueError("person_detector.recovery.enabled must be boolean.")
    empty_frames_before_boost = _integer_value(
        raw_recovery.get("empty_frames_before_boost", 2),
        "recovery.empty_frames_before_boost",
        minimum=1,
    )
    boost_frames = _integer_value(raw_recovery.get("boost_frames", 3), "recovery.boost_frames", minimum=1)
    cooldown_frames = _integer_value(raw_recovery.get("cooldown_frames", 30), "recovery.cooldown_frames")
    raw_boosted_imgsz = raw_recovery.get("boosted_imgsz")
    if raw_boosted_imgsz is None:
        boosted_imgsz = None
    elif isinstance(raw_boosted_imgsz, bool) or not isinstance(raw_boosted_imgsz, int) or raw_boosted_imgsz < 32:
        raise ValueError("person_detector.recovery.boosted_imgsz must be an integer of at least 32 when provided.")
    else:
        boosted_imgsz = raw_boosted_imgsz
    raw_small_height = raw_recovery.get("small_person_max_height_px")
    if raw_small_height is None:
        small_person_max_height_px = None
    else:
        small_person_max_height_px = _integer_value(
            raw_small_height,
            "recovery.small_person_max_height_px",
            minimum=1,
        )
    small_person_frames_before_boost = _integer_value(
        raw_recovery.get("small_person_frames_before_boost", 0),
        "recovery.small_person_frames_before_boost",
    )
    if small_person_max_height_px is None and small_person_frames_before_boost:
        raise ValueError(
            "person_detector.recovery.small_person_frames_before_boost requires "
            "recovery.small_person_max_height_px."
        )
    if small_person_max_height_px is not None and small_person_frames_before_boost < 1:
        raise ValueError(
            "person_detector.recovery.small_person_frames_before_boost must be at least 1 when "
            "recovery.small_person_max_height_px is configured."
        )
    if recovery_enabled:
        if imgsz is None:
            raise ValueError("person_detector.recovery.enabled requires an explicit person_detector.imgsz.")
        if boosted_imgsz is None or boosted_imgsz <= imgsz:
            raise ValueError(
                "person_detector.recovery.boosted_imgsz must be greater than person_detector.imgsz when recovery is enabled."
            )

    return DetectorSettings(
        tracker_input_confidence=tracker_confidence,
        evaluation_confidence=evaluation_confidence,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        use_half=raw_use_half,
        end2end=raw_end2end,
        max_det=raw_max_det,
        telemetry_enabled=raw_telemetry,
        telemetry_sample_every_n_frames=telemetry_sample_every_n_frames,
        require_full_low_score_recovery=raw_low_score_guard,
        recovery=DetectorRecoverySettings(
            enabled=recovery_enabled,
            empty_frames_before_boost=empty_frames_before_boost,
            boosted_imgsz=boosted_imgsz,
            boost_frames=boost_frames,
            cooldown_frames=cooldown_frames,
            small_person_max_height_px=small_person_max_height_px,
            small_person_frames_before_boost=small_person_frames_before_boost,
        ),
    )


class _ResizePad:
    """Match the 256x128 aspect-preserving RGB preprocessing used during body training."""

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        source_width, source_height = image.size
        scale = min(self.width / source_width, self.height / source_height)
        resized_width = max(1, round(source_width * scale))
        resized_height = max(1, round(source_height * scale))
        resized = transforms.functional.resize(image, [resized_height, resized_width], antialias=True)
        canvas = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        canvas.paste(resized, ((self.width - resized_width) // 2, (self.height - resized_height) // 2))
        return canvas


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _clip_bbox(bbox: Sequence[float], frame_width: int, frame_height: int) -> BBox | None:
    x1, y1, x2, y2 = bbox
    clipped = (max(0, int(x1)), max(0, int(y1)), min(frame_width, int(x2)), min(frame_height, int(y2)))
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def _align_face(image: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_eye = landmarks[:2]
    right_eye = landmarks[2:4]
    angle = float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
    center = tuple(((left_eye + right_eye) / 2.0).tolist())
    transform = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(
        image,
        transform,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(128, 128, 128),
    )
    return aligned, transform


def _rotate_face_box(box: tuple[int, int, int, int], transform: np.ndarray, shape: tuple[int, int]) -> BBox:
    x, y, width, height = box
    corners = np.array([[[x, y], [x + width, y], [x + width, y + height], [x, y + height]]], dtype=np.float32)
    transformed = cv2.transform(corners, transform)[0]
    x1, y1 = np.floor(transformed.min(axis=0)).astype(int)
    x2, y2 = np.ceil(transformed.max(axis=0)).astype(int)
    frame_height, frame_width = shape
    return max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)


@dataclass(frozen=True)
class FaceCropSettings:
    """Validated, model-agnostic crop policy for the face gender branch."""

    mode: str = "eye_rotation"
    padding_fraction: float = 0.30
    similarity_output_size: int = 224
    minimum_blur_variance: float | None = None
    minimum_brightness: float | None = None
    maximum_brightness: float | None = None
    require_valid_landmarks: bool = True


@dataclass(frozen=True)
class FaceExtractionResult:
    """Outcome of one YuNet extraction attempt, with no retained face pixels."""

    candidate: GenderCandidate | None
    status: str
    quality: FaceQuality | None = None


def _finite_number(value: object, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"face_detector.{name} must be numeric.")
    number = float(value)
    if not np.isfinite(number) or (minimum is not None and number < minimum) or (maximum is not None and number > maximum):
        bounds = []
        if minimum is not None:
            bounds.append(f">= {minimum}")
        if maximum is not None:
            bounds.append(f"<= {maximum}")
        raise ValueError(f"face_detector.{name} must be finite and {' and '.join(bounds)}.")
    return number


def _parse_face_crop_settings(face_config: Mapping[str, object]) -> FaceCropSettings:
    """Read new crop options while preserving the historical runtime defaults."""

    raw_alignment = face_config.get("alignment", {})
    if raw_alignment is None:
        raw_alignment = {}
    if not isinstance(raw_alignment, Mapping):
        raise ValueError("face_detector.alignment must be a mapping when provided.")
    mode = str(raw_alignment.get("mode", face_config.get("crop_mode", "eye_rotation"))).strip().lower()
    if mode not in _FACE_ALIGNMENT_MODES:
        raise ValueError(f"face_detector.alignment.mode must be one of {sorted(_FACE_ALIGNMENT_MODES)}.")
    padding = _finite_number(
        raw_alignment.get("padding_fraction", face_config.get("crop_padding_fraction", 0.30)),
        "alignment.padding_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    raw_output_size = raw_alignment.get("output_size", face_config.get("similarity_output_size", 224))
    if isinstance(raw_output_size, bool) or not isinstance(raw_output_size, int) or raw_output_size < 32:
        raise ValueError("face_detector.alignment.output_size must be an integer of at least 32.")
    raw_quality = face_config.get("quality", {})
    if raw_quality is None:
        raw_quality = {}
    if not isinstance(raw_quality, Mapping):
        raise ValueError("face_detector.quality must be a mapping when provided.")

    def optional_threshold(key: str, *, minimum: float = 0.0, maximum: float | None = None) -> float | None:
        value = raw_quality.get(key)
        return None if value is None else _finite_number(value, f"quality.{key}", minimum=minimum, maximum=maximum)

    minimum_brightness = optional_threshold("minimum_brightness", maximum=255.0)
    maximum_brightness = optional_threshold("maximum_brightness", maximum=255.0)
    if minimum_brightness is not None and maximum_brightness is not None and minimum_brightness > maximum_brightness:
        raise ValueError("face_detector.quality.minimum_brightness cannot exceed maximum_brightness.")
    require_landmarks = raw_quality.get("require_valid_landmarks", True)
    if not isinstance(require_landmarks, bool):
        raise ValueError("face_detector.quality.require_valid_landmarks must be boolean.")
    return FaceCropSettings(
        mode=mode,
        padding_fraction=padding,
        similarity_output_size=raw_output_size,
        minimum_blur_variance=optional_threshold("minimum_blur_variance"),
        minimum_brightness=minimum_brightness,
        maximum_brightness=maximum_brightness,
        require_valid_landmarks=require_landmarks,
    )


def _landmarks_are_valid(landmarks: np.ndarray) -> bool:
    values = np.asarray(landmarks, dtype=np.float32)
    if values.size != 10:
        return False
    points = values.reshape(5, 2)
    if not np.isfinite(points).all():
        return False
    eye_distance = float(np.linalg.norm(points[1] - points[0]))
    return eye_distance >= 2.0 and len({(round(float(x), 3), round(float(y), 3)) for x, y in points}) >= 4


def _expand_face_box(box: tuple[int, int, int, int], padding_fraction: float, shape: tuple[int, int]) -> BBox:
    x, y, width, height = box
    padding_x, padding_y = int(width * padding_fraction), int(height * padding_fraction)
    frame_height, frame_width = shape
    return max(0, x - padding_x), max(0, y - padding_y), min(frame_width, x + width + padding_x), min(frame_height, y + height + padding_y)


def _five_point_similarity_crop(
    image: np.ndarray, landmarks: np.ndarray, settings: FaceCropSettings
) -> np.ndarray | None:
    """Warp YuNet's five landmarks into a deterministic square BGR face crop."""

    source = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    # More requested padding moves canonical landmarks toward the output centre,
    # leaving contextual pixels around the face while retaining the same output size.
    template_scale = 1.0 / (1.0 + 2.0 * settings.padding_fraction)
    target = 0.5 + (_FIVE_POINT_TEMPLATE - 0.5) * template_scale
    target *= float(settings.similarity_output_size - 1)
    transform, _ = cv2.estimateAffinePartial2D(source, target.astype(np.float32), method=cv2.LMEDS)
    if transform is None or not np.isfinite(transform).all():
        return None
    return cv2.warpAffine(
        image,
        transform,
        (settings.similarity_output_size, settings.similarity_output_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(128, 128, 128),
    )


def _crop_face(
    image: np.ndarray,
    face_box: tuple[int, int, int, int],
    landmarks: np.ndarray,
    settings: FaceCropSettings,
) -> np.ndarray | None:
    """Return a BGR crop for one configured policy without changing pixel order."""

    if settings.mode == "five_point_similarity":
        return _five_point_similarity_crop(image, landmarks, settings)
    if settings.mode == "raw_bbox":
        x1, y1, x2, y2 = _expand_face_box(face_box, settings.padding_fraction, image.shape[:2])
        return image[y1:y2, x1:x2]
    aligned, transform = _align_face(image, landmarks)
    x1, y1, x2, y2 = _rotate_face_box(face_box, transform, aligned.shape[:2])
    expanded = _expand_face_box((x1, y1, x2 - x1, y2 - y1), settings.padding_fraction, aligned.shape[:2])
    return aligned[expanded[1] : expanded[3], expanded[0] : expanded[2]]


def _evaluate_face_quality(
    crop: np.ndarray,
    *,
    detection_confidence: float,
    face_width: int,
    landmarks_valid: bool,
    alignment_mode: str,
    settings: FaceCropSettings,
) -> FaceQuality:
    """Measure quality primitives and evaluate only explicitly configured gates."""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    reasons: list[str] = []
    if settings.require_valid_landmarks and not landmarks_valid:
        reasons.append("invalid_landmarks")
    if settings.minimum_blur_variance is not None and blur_variance < settings.minimum_blur_variance:
        reasons.append("low_blur_variance")
    if settings.minimum_brightness is not None and brightness < settings.minimum_brightness:
        reasons.append("underexposed")
    if settings.maximum_brightness is not None and brightness > settings.maximum_brightness:
        reasons.append("overexposed")
    return FaceQuality(
        detection_confidence=float(detection_confidence),
        face_width=int(face_width),
        blur_variance=blur_variance,
        brightness=brightness,
        landmarks_valid=landmarks_valid,
        alignment_mode=alignment_mode,
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
    )


class ModelRuntime:
    """Loads model weights once and exposes detector/face/gender inference primitives."""

    def __init__(
        self,
        pipeline_config: dict[str, Any],
        project_root: Path,
        gender_model_path: Path | None,
        device: torch.device,
        tracker_profile: TrackerProfile,
    ) -> None:
        self.config = pipeline_config
        self.project_root = project_root
        self.gender_model_path = gender_model_path
        self.device = device
        self.tracker_profile = tracker_profile
        router = dict(self.config.get("attribute_router", {}) or {})
        self.attribute_router_mode = str(router.get("mode", "face_first")).strip().lower()
        self.face_enabled = self.attribute_router_mode != "body_only"
        runtime_config = self.config.get("runtime", {})
        self.use_mixed_precision = bool(runtime_config.get("use_mixed_precision", False)) and self.device.type == "cuda"
        self.use_horizontal_tta = False
        self.yunet = None
        self.gender_model = None
        self.gender_transform = None
        # Detector execution mode and max_det are applied while Ultralytics
        # creates its predictor. They therefore cannot be safely changed on a
        # live model after the first track call.
        self._detector_mode_lock: tuple[bool | None, int] | None = None
        self._effective_end2end: bool | None = None
        self._detector_callback_registered = False
        self._reset_detector_telemetry()
        self._init_yolo()
        if self.face_enabled:
            self._init_yunet()
            self._init_gender_classifier()
        self._init_body_gender_classifier()

    def _init_yolo(self) -> None:
        configured_model = Path(self.config["person_detector"]["model_path"])
        resolved_model = _resolve_path(configured_model, self.project_root)
        if not resolved_model.is_file():
            raise FileNotFoundError(
                "Person detector checkpoint not found: "
                f"{resolved_model}. Provision and verify the configured local asset with "
                "`python tools/prepare_production_assets.py` before starting the pipeline."
            )
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Install ultralytics before running the pipeline.") from error
        ensure_tracker_backend_available(self.tracker_profile.tracker_type)
        self._detector_settings()
        # Passing only a verified absolute/local path prevents Ultralytics from
        # treating a bare official model name as a download request.
        self.yolo = YOLO(str(resolved_model))
        # This callback is registered before Ultralytics lazily adds its tracker
        # callback on the first ``track`` call. It can therefore inspect the raw
        # detector output without an extra YOLO inference.
        self._register_detector_telemetry_callback()

    def _detector_settings(self) -> DetectorSettings:
        return resolve_detector_settings(self.config["person_detector"])

    def _register_detector_telemetry_callback(self) -> None:
        if self._detector_callback_registered:
            return
        add_callback = getattr(self.yolo, "add_callback", None)
        if not callable(add_callback):
            raise RuntimeError("The selected Ultralytics YOLO object does not support inference callbacks.")
        add_callback("on_predict_postprocess_end", self._capture_pre_tracker_detections)
        self._detector_callback_registered = True

    def _reset_detector_telemetry(self) -> None:
        """Drop bounded per-stream detector diagnostics together with tracker state."""

        self._last_pre_tracker_detections: dict[str, object] | None = None
        self._last_detector_frame: dict[str, object] = {}
        # This lightweight raw-output summary is populated before the tracker
        # callback. It lets the recovery policy respond to a genuine detector
        # dropout instead of confusing an unconfirmed/lost track with a miss.
        self._last_raw_detection_count: int | None = None
        self._last_raw_largest_height_px: float | None = None
        self._detector_callback_frames = 0
        self._detector_telemetry_totals: dict[str, int] = {
            "frames": 0,
            "pre_tracker_detections": 0,
            "pre_tracker_cap_reached_frames": 0,
            "score_at_or_below_track_low": 0,
            "score_low_association_band": 0,
            "score_high_association_band": 0,
            "score_new_track_eligible": 0,
            "height_lt_50": 0,
            "height_50_to_99": 0,
            "height_100_to_199": 0,
            "height_ge_200": 0,
        }
        self._recovery_consecutive_empty_frames = 0
        self._recovery_consecutive_small_person_frames = 0
        self._recovery_boost_frames_remaining = 0
        self._recovery_cooldown_frames_remaining = 0
        self._recovery_last_frame_used_boost = False
        self._recovery_last_requested_imgsz: int | None = None
        self._recovery_last_trigger_reason: str | None = None
        self._detector_recovery_totals: dict[str, int] = {
            "frames": 0,
            "raw_candidate_frames": 0,
            "empty_raw_candidate_frames": 0,
            "small_person_trigger_frames": 0,
            "boost_activations": 0,
            "boosted_frames": 0,
            "boosted_frames_with_candidates": 0,
            "callback_missing_frames": 0,
        }

    @staticmethod
    def _tensor_to_numpy(value: object) -> np.ndarray:
        """Convert a Tensor/array-like result field to CPU NumPy exactly once."""

        detached = value.detach() if hasattr(value, "detach") else value
        if hasattr(detached, "float"):
            detached = detached.float()
        if hasattr(detached, "cpu"):
            detached = detached.cpu()
        if hasattr(detached, "numpy"):
            detached = detached.numpy()
        return np.asarray(detached)

    def _capture_pre_tracker_detections(self, predictor: object) -> None:
        """Capture aggregate raw detections before Ultralytics replaces them with tracks.

        The data is deliberately bounded: counts, score bands, and size bands
        only. No image crop, raw box coordinate, or local identifier is
        retained. It can be enabled for operational diagnostics without
        running a second detector pass and remains disabled by default.
        """

        settings = self._detector_settings()
        if not settings.telemetry_enabled and not settings.recovery.enabled:
            return

        boxes_list = [getattr(result, "boxes", None) for result in (getattr(predictor, "results", ()) or ())]
        boxes_list = [boxes for boxes in boxes_list if boxes is not None]
        raw_detection_count = sum(len(boxes) for boxes in boxes_list)
        self._last_raw_detection_count = raw_detection_count
        self._detector_callback_frames += 1
        telemetry_sampled = settings.telemetry_enabled and (
            (self._detector_callback_frames - 1) % settings.telemetry_sample_every_n_frames == 0
        )
        needs_height_for_recovery = settings.recovery.small_person_max_height_px is not None
        if not telemetry_sampled and not needs_height_for_recovery:
            return

        score_arrays: list[np.ndarray] = []
        coordinate_arrays: list[np.ndarray] = []
        for boxes in boxes_list:
            if len(boxes) == 0:
                continue
            if telemetry_sampled:
                score_arrays.append(self._tensor_to_numpy(boxes.conf).reshape(-1))
            coordinate_arrays.append(self._tensor_to_numpy(boxes.xyxy).reshape(-1, 4))
        coordinates = np.concatenate(coordinate_arrays) if coordinate_arrays else np.empty((0, 4), dtype=np.float32)
        heights = coordinates[:, 3] - coordinates[:, 1] if len(coordinates) else np.empty((0,), dtype=np.float32)
        self._last_raw_largest_height_px = float(np.max(heights)) if len(heights) else None
        if not telemetry_sampled:
            return

        scores = np.concatenate(score_arrays) if score_arrays else np.empty((0,), dtype=np.float32)

        tracker_values = self.tracker_profile.values
        track_low = float(tracker_values.get("track_low_thresh", 0.0))
        track_high = float(tracker_values.get("track_high_thresh", 1.0))
        new_track = float(tracker_values.get("new_track_thresh", track_high))
        score_counts = {
            "at_or_below_track_low": int(np.count_nonzero(scores <= track_low)),
            "low_association_band": int(np.count_nonzero((scores > track_low) & (scores < track_high))),
            "high_association_band": int(np.count_nonzero(scores >= track_high)),
            "new_track_eligible": int(np.count_nonzero(scores >= new_track)),
        }
        height_counts = {
            "lt_50": int(np.count_nonzero(heights < 50)),
            "50_to_99": int(np.count_nonzero((heights >= 50) & (heights < 100))),
            "100_to_199": int(np.count_nonzero((heights >= 100) & (heights < 200))),
            "ge_200": int(np.count_nonzero(heights >= 200)),
        }
        frame_summary = {
            "pre_tracker_detection_count": int(len(scores)),
            "pre_tracker_cap_reached": int(len(scores)) >= settings.max_det,
            "score_bands": score_counts,
            "height_bins": height_counts,
        }
        self._last_pre_tracker_detections = frame_summary
        totals = self._detector_telemetry_totals
        totals["frames"] += 1
        totals["pre_tracker_detections"] += int(len(scores))
        totals["pre_tracker_cap_reached_frames"] += int(frame_summary["pre_tracker_cap_reached"])
        for name, value in score_counts.items():
            totals[f"score_{name}"] += value
        for name, value in height_counts.items():
            totals[f"height_{name}"] += value

    def _validate_detector_tracker_contract(self, settings: DetectorSettings) -> None:
        """Reject accidental loss of FastTracker's low-score recovery band.

        Raising is intentionally opt-in at the YAML level for backwards
        compatibility with old saved profiles. The selected live FastTracker
        profile enables it, and a configuration must explicitly opt out before
        it can raise the detector input confidence to/above ``track_low``.
        """

        if not settings.require_full_low_score_recovery:
            return
        tracker_low = float(self.tracker_profile.values["track_low_thresh"])
        if settings.tracker_input_confidence >= tracker_low:
            raise ValueError(
                "person_detector.require_full_low_score_recovery=true requires "
                "tracker_input_confidence below the tracker track_low_thresh "
                f"({tracker_low:.3f}); set require_full_low_score_recovery=false only with an intentional configuration change."
            )

    def _requested_detector_imgsz(self, settings: DetectorSettings) -> int | None:
        """Return this frame's detector size and consume one bounded boost slot."""

        self._recovery_last_frame_used_boost = False
        self._recovery_last_requested_imgsz = settings.imgsz
        recovery = settings.recovery
        if recovery.enabled and self._recovery_boost_frames_remaining > 0:
            self._recovery_boost_frames_remaining -= 1
            self._recovery_last_frame_used_boost = True
            self._recovery_last_requested_imgsz = recovery.boosted_imgsz
            return recovery.boosted_imgsz
        return settings.imgsz

    def _start_detector_recovery(self, settings: DetectorSettings, reason: str) -> None:
        """Schedule, rather than immediately recurse into, a bounded high-detail retry."""

        recovery = settings.recovery
        if not recovery.enabled or self._recovery_cooldown_frames_remaining > 0:
            return
        self._recovery_boost_frames_remaining = recovery.boost_frames
        self._recovery_consecutive_empty_frames = 0
        self._recovery_consecutive_small_person_frames = 0
        self._recovery_last_trigger_reason = reason
        self._detector_recovery_totals["boost_activations"] += 1

    def _update_detector_recovery(self, settings: DetectorSettings) -> None:
        """Update the next-frame recovery state from a pre-tracker detector summary."""

        recovery = settings.recovery
        if not recovery.enabled:
            return
        totals = self._detector_recovery_totals
        totals["frames"] += 1
        raw_detection_count = self._last_raw_detection_count
        if raw_detection_count is None:
            totals["callback_missing_frames"] += 1
            if self._recovery_last_frame_used_boost and self._recovery_boost_frames_remaining == 0:
                self._recovery_cooldown_frames_remaining = recovery.cooldown_frames
            return

        if raw_detection_count > 0:
            totals["raw_candidate_frames"] += 1
        else:
            totals["empty_raw_candidate_frames"] += 1

        if self._recovery_last_frame_used_boost:
            totals["boosted_frames"] += 1
            if raw_detection_count > 0:
                # This is evidence that a boosted frame found candidates, not
                # proof that the boost caused the recovery. Keep the metric
                # factual and leave effectiveness assessment to operators.
                totals["boosted_frames_with_candidates"] += 1
                self._recovery_boost_frames_remaining = 0
                self._recovery_consecutive_empty_frames = 0
                self._recovery_consecutive_small_person_frames = 0
                self._recovery_cooldown_frames_remaining = recovery.cooldown_frames
            elif self._recovery_boost_frames_remaining == 0:
                self._recovery_cooldown_frames_remaining = recovery.cooldown_frames
            return

        if self._recovery_cooldown_frames_remaining > 0:
            self._recovery_cooldown_frames_remaining -= 1
            if raw_detection_count > 0:
                self._recovery_consecutive_empty_frames = 0
                self._recovery_consecutive_small_person_frames = 0
            return

        if raw_detection_count == 0:
            self._recovery_consecutive_empty_frames += 1
            self._recovery_consecutive_small_person_frames = 0
            if self._recovery_consecutive_empty_frames >= recovery.empty_frames_before_boost:
                self._start_detector_recovery(settings, "empty_raw_detections")
            return

        self._recovery_consecutive_empty_frames = 0
        small_height = recovery.small_person_max_height_px
        if small_height is None or self._last_raw_largest_height_px is None:
            self._recovery_consecutive_small_person_frames = 0
            return
        if self._last_raw_largest_height_px <= small_height:
            self._recovery_consecutive_small_person_frames += 1
            totals["small_person_trigger_frames"] += 1
            if self._recovery_consecutive_small_person_frames >= recovery.small_person_frames_before_boost:
                self._start_detector_recovery(settings, "small_person_detections")
        else:
            self._recovery_consecutive_small_person_frames = 0

    def _track_kwargs(self, frame: np.ndarray) -> tuple[dict[str, Any], DetectorSettings]:
        settings = self._detector_settings()
        self._validate_detector_tracker_contract(settings)
        requested_mode = (settings.end2end, settings.max_det)
        if self._detector_mode_lock is not None and requested_mode != self._detector_mode_lock:
            raise RuntimeError(
                "person_detector.end2end and max_det are fixed after the first YOLO.track call. "
                "Create a fresh CrowdGenderPipeline before changing detector execution mode."
            )
        track_kwargs: dict[str, Any] = {
            "source": frame,
            "persist": PERSIST_TRACKER_ACROSS_FRAMES,
            "classes": [0],
            "conf": settings.tracker_input_confidence,
            "iou": settings.iou_threshold,
            "max_det": settings.max_det,
            "tracker": str(self.tracker_profile.path),
            "verbose": False,
        }
        if settings.end2end is not None:
            track_kwargs["end2end"] = settings.end2end
        requested_imgsz = self._requested_detector_imgsz(settings)
        if requested_imgsz is not None:
            track_kwargs["imgsz"] = requested_imgsz
        if self.device.type == "cuda":
            track_kwargs["device"] = 0
            track_kwargs["half"] = settings.use_half
        return track_kwargs, settings

    def _capture_effective_end2end(self, requested: bool | None) -> None:
        predictor = getattr(self.yolo, "predictor", None)
        model = getattr(predictor, "model", None)
        candidates = (model, getattr(model, "model", None))
        effective: bool | None = None
        for candidate in candidates:
            value = getattr(candidate, "end2end", None)
            if isinstance(value, (bool, np.bool_)):
                effective = bool(value)
                break
        self._effective_end2end = effective
        if requested is not None and effective is not None and effective != requested:
            raise RuntimeError(
                f"YOLO predictor mode mismatch: requested end2end={requested}, but effective end2end={effective}."
            )

    def detector_statistics(self) -> dict[str, object]:
        """Return reproducible detector/tracker-interface metadata for reports."""

        settings = self._detector_settings()
        settings_payload = asdict(settings)
        settings_payload["model_path"] = str(self.config["person_detector"]["model_path"])
        tracker_low = float(self.tracker_profile.values["track_low_thresh"])
        low_band_complete = settings.tracker_input_confidence < tracker_low
        return {
            "settings": settings_payload,
            "effective_end2end": self._effective_end2end,
            "mode_locked": self._detector_mode_lock is not None,
            "iou_effective": self._effective_end2end is not True,
            "tracker_low_threshold": tracker_low,
            "full_low_score_band_available": low_band_complete,
            "full_low_score_band_required": settings.require_full_low_score_recovery,
            "low_score_band_note": (
                None
                if low_band_complete
                else "tracker_input_confidence is at/above the tracker track_low_thresh; some low-score rescue detections are filtered first."
            ),
            "last_frame": dict(self._last_detector_frame),
            "recovery": {
                "enabled": settings.recovery.enabled,
                "last_requested_imgsz": self._recovery_last_requested_imgsz,
                "last_frame_used_boost": self._recovery_last_frame_used_boost,
                "last_trigger_reason": self._recovery_last_trigger_reason,
                "boost_frames_remaining": self._recovery_boost_frames_remaining,
                "cooldown_frames_remaining": self._recovery_cooldown_frames_remaining,
                "consecutive_empty_frames": self._recovery_consecutive_empty_frames,
                "consecutive_small_person_frames": self._recovery_consecutive_small_person_frames,
                "last_raw_detection_count": self._last_raw_detection_count,
                "last_raw_largest_height_px": self._last_raw_largest_height_px,
                "totals": dict(self._detector_recovery_totals),
                "note": (
                    "A boosted frame with candidates is an observed signal, not causal proof that high-resolution recovery improved detection."
                    if settings.recovery.enabled
                    else None
                ),
            },
            "telemetry": {
                "enabled": settings.telemetry_enabled,
                "sample_every_n_frames": settings.telemetry_sample_every_n_frames,
                "callback_frames": self._detector_callback_frames,
                "last_pre_tracker": self._last_pre_tracker_detections,
                "totals": dict(self._detector_telemetry_totals),
            },
        }

    def _init_yunet(self) -> None:
        face_config = self.config["face_detector"]
        model_path = _resolve_path(face_config["model_path"], self.project_root)
        if not model_path.is_file():
            raise FileNotFoundError(
                "Face detector checkpoint not found: "
                f"{model_path}. Provision the configured local asset with "
                "`python tools/prepare_production_assets.py` before starting the pipeline."
            )
        self.yunet = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(320, 320),
            score_threshold=face_config["confidence_threshold"],
            nms_threshold=0.3,
            top_k=100,
        )

    def _init_gender_classifier(self) -> None:
        if self.gender_model_path is None:
            raise ValueError("A face gender checkpoint is required unless attribute_router.mode=body_only.")
        try:
            checkpoint = torch.load(self.gender_model_path, map_location=self.device, weights_only=True)
        except TypeError:  # PyTorch before weights_only was introduced.
            checkpoint = torch.load(self.gender_model_path, map_location=self.device)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Gender checkpoint must contain a model_state_dict entry.")
        labels = tuple(checkpoint.get("labels", GENDER_LABELS))
        if labels != GENDER_LABELS:
            raise ValueError(f"Checkpoint labels must be {GENDER_LABELS}; received {labels}.")
        self.gender_model = GenderClassifier(pretrained=False).to(self.device)
        self.gender_model.load_state_dict(checkpoint["model_state_dict"])
        self.gender_model.eval()
        input_size = int(checkpoint.get("input_size", self.config["gender_classifier"]["input_size"]))
        normalization = checkpoint.get("normalization", {})
        self.gender_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=normalization.get("mean", IMAGE_NET_MEAN),
                    std=normalization.get("std", IMAGE_NET_STD),
                ),
            ]
        )
        self.use_horizontal_tta = bool(self.config["gender_classifier"].get("use_horizontal_tta", False))

    def _init_body_gender_classifier(self) -> None:
        """Load the optional PA-100K body checkpoint without changing the face-model contract."""
        body_config = self.config.get("body_gender_classifier", {})
        self.body_enabled = bool(body_config.get("enabled", False))
        self.body_model: BodyGenderClassifier | None = None
        self.body_transform = None
        self.body_temperature = 1.0
        if not self.body_enabled:
            return

        model_path = _resolve_path(body_config["model_path"], self.project_root)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Body gender checkpoint not found: {model_path}. Provision the configured local asset with "
                "`python tools/prepare_production_assets.py` before starting the pipeline."
            )
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:  # PyTorch before weights_only was introduced.
            checkpoint = torch.load(model_path, map_location=self.device)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Body gender checkpoint must contain a model_state_dict entry.")
        if checkpoint.get("model_role") != BODY_MODEL_ROLE:
            raise ValueError(f"Body checkpoint model_role must be {BODY_MODEL_ROLE!r}.")
        if checkpoint.get("model_architecture") != BODY_MODEL_ARCHITECTURE:
            raise ValueError(f"Body checkpoint architecture must be {BODY_MODEL_ARCHITECTURE!r}.")
        labels = tuple(checkpoint.get("labels", ()))
        if labels != GENDER_LABELS:
            raise ValueError(f"Body checkpoint labels must be {GENDER_LABELS}; received {labels}.")
        if checkpoint.get("resize_mode") != "aspect_preserving_letterbox":
            raise ValueError("Body checkpoint must use aspect_preserving_letterbox preprocessing.")

        input_height = int(checkpoint.get("input_height", 0))
        input_width = int(checkpoint.get("input_width", 0))
        if input_height < 1 or input_width < 1:
            raise ValueError("Body checkpoint must define positive input_height and input_width.")
        normalization = checkpoint.get("normalization", {})
        temperature = float(checkpoint.get("temperature", 1.0))
        if temperature <= 0.0:
            raise ValueError("Body checkpoint temperature must be positive.")
        # The live threshold is explicit in YAML so both runtime and temporal state use
        # the same calibrated acceptance policy. The checkpoint value remains provenance.
        threshold = float(body_config["confidence_threshold"])
        if not 0.5 < threshold < 1.0:
            raise ValueError("Body confidence_threshold must be in (0.5, 1.0).")

        self.body_model = BodyGenderClassifier(pretrained=False).to(self.device)
        self.body_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.body_model.eval()
        self.body_transform = transforms.Compose(
            [
                _ResizePad(input_height, input_width),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=normalization.get("mean", IMAGE_NET_MEAN),
                    std=normalization.get("std", IMAGE_NET_STD),
                ),
            ]
        )
        self.body_temperature = temperature
        if bool(self.config.get("runtime", {}).get("warmup_body_classifier", True)):
            self._warmup_body_classifier(input_height, input_width)

    def _warmup_body_classifier(self, input_height: int, input_width: int) -> None:
        """Pay CUDA's first-forward cost while the stream is still initializing.

        The body fallback is intentionally infrequent. Without this warm-up its first
        valid person crop can pause an otherwise live webcam feed for several seconds.
        Warm the configured maximum batch shape so the first scheduled fallback does
        not become a latency spike.
        """
        if self.body_model is None:
            return
        body_config = self.config.get("body_gender_classifier", {})
        maximum_batch_size = max(1, int(body_config.get("max_body_tracks_per_frame", 1)))
        # CUDA/cuDNN can initialize a separate execution path for batch 1 and the
        # configured maximum. The scheduler regularly emits either shape, so warm both.
        for batch_size in sorted({1, maximum_batch_size}):
            warmup_batch = torch.zeros((batch_size, 3, input_height, input_width), device=self.device)
            with torch.inference_mode(), self._autocast_context():
                self.body_model(warmup_batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reset_tracker(self) -> None:
        reset_attached_trackers(self.yolo)
        self._reset_detector_telemetry()

    def warmup(self, frame_shape: tuple[int, int] = (480, 640)) -> None:
        """Initialize live inference paths before a real webcam frame is counted.

        ``YOLO.track`` must see ``persist=True`` on its first call, so warm-up goes through
        ``active_tracks`` and then resets the attached tracker.  YuNet and the face classifier
        are also touched once to avoid a first-person latency spike during a presentation.
        """
        height, width = frame_shape
        if height < 32 or width < 32:
            raise ValueError("warmup frame dimensions must be at least 32 pixels")
        blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.active_tracks(blank_frame)
        if self.face_enabled:
            if self.yunet is None or self.gender_model is None:
                raise RuntimeError("Face runtime was enabled but failed to initialize.")
            self.yunet.setInputSize((128, 128))
            self.yunet.detect(np.zeros((128, 128, 3), dtype=np.uint8))
            face_size = int(self.config["gender_classifier"]["input_size"])
            face_batch = torch.zeros((1, 3, face_size, face_size), device=self.device)
            with torch.inference_mode(), self._autocast_context():
                self.gender_model(face_batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.reset_tracker()

    def active_tracks(self, frame: np.ndarray) -> dict[int, BBox]:
        track_kwargs, settings = self._track_kwargs(frame)
        self._last_pre_tracker_detections = None
        self._last_raw_detection_count = None
        self._last_raw_largest_height_px = None
        started = perf_counter()
        results = self.yolo.track(**track_kwargs)
        track_elapsed_ms = (perf_counter() - started) * 1_000
        self._detector_mode_lock = (settings.end2end, settings.max_det)
        self._capture_effective_end2end(settings.end2end)
        self._update_detector_recovery(settings)
        active_tracks: dict[int, BBox] = {}
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            self._record_post_tracker_summary(results, active_track_count=0, outer_track_ms=track_elapsed_ms)
            return active_tracks
        frame_height, frame_width = frame.shape[:2]
        coordinates = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        for track_id, bbox in zip(ids, coordinates):
            clipped = _clip_bbox(bbox, frame_width, frame_height)
            if clipped is not None:
                active_tracks[int(track_id)] = clipped
        self._record_post_tracker_summary(results, active_track_count=len(active_tracks), outer_track_ms=track_elapsed_ms)
        return active_tracks

    def _record_post_tracker_summary(
        self, results: Sequence[object], *, active_track_count: int, outer_track_ms: float
    ) -> None:
        """Expose a transparent, approximate detector/tracker timing split.

        Ultralytics returns detector timing inside ``Results.speed`` but its
        tracker callback runs inside the same public ``track`` call. The
        residual is labelled approximate rather than being misrepresented as a
        pure association measurement.
        """

        result = results[0] if results else None
        boxes = getattr(result, "boxes", None)
        post_tracker_boxes = len(boxes) if boxes is not None else 0
        speed = getattr(result, "speed", {}) or {}
        detector_speed_ms = sum(
            float(speed.get(name, 0.0) or 0.0) for name in ("preprocess", "inference", "postprocess")
        )
        self._last_detector_frame = {
            "outer_track_ms": round(outer_track_ms, 3),
            "detector_speed_ms": round(detector_speed_ms, 3),
            "association_wrapper_ms_approx": round(max(0.0, outer_track_ms - detector_speed_ms), 3),
            "post_tracker_box_count": int(post_tracker_boxes),
            "usable_active_track_count": int(active_track_count),
        }

    @staticmethod
    def _choose_face(faces: np.ndarray, roi_width: int) -> np.ndarray:
        """Favor a confident face close to the center of the current person ROI."""

        def score(face: np.ndarray) -> float:
            center_x = face[0] + face[2] / 2.0
            center_bonus = max(0.0, 1.0 - abs(center_x - roi_width / 2.0) / max(1.0, roi_width / 2.0))
            return float(face[14]) * (0.7 + 0.3 * center_bonus)

        return max(faces, key=score)

    def extract_gender_candidate_result(self, frame: np.ndarray, track_id: int, bbox: BBox) -> FaceExtractionResult:
        """Run YuNet and return an explicit no-crop diagnostic outcome.

        New callers can distinguish a detector miss from a quality-policy
        rejection.  ``extract_gender_candidate`` remains as the compatible
        crop-only API used by the existing scheduler.
        """
        if not self.face_enabled or self.yunet is None:
            raise RuntimeError("Face candidate extraction was requested while the face branch is disabled.")
        face_config = self.config["face_detector"]
        settings = _parse_face_crop_settings(face_config)
        x1, y1, x2, y2 = bbox
        person_height = y2 - y1
        if person_height < int(face_config.get("minimum_person_height", 1)):
            return FaceExtractionResult(None, "person_too_small")
        top_y2 = min(y2, y1 + int(person_height * face_config["upper_person_ratio"]))
        candidate_rois = [frame[y1:top_y2, x1:x2]]
        if bool(face_config.get("full_person_fallback", True)) and top_y2 < y2:
            candidate_rois.append(frame[y1:y2, x1:x2])

        face_roi = None
        faces = None
        for roi in candidate_rois:
            if roi.shape[0] <= 10 or roi.shape[1] <= 10:
                continue
            self.yunet.setInputSize((roi.shape[1], roi.shape[0]))
            _, detected = self.yunet.detect(roi)
            if detected is not None and len(detected):
                face_roi, faces = roi, detected
                break
        if face_roi is None or faces is None:
            return FaceExtractionResult(None, "no_face")

        face = self._choose_face(faces, face_roi.shape[1])
        face_x, face_y, face_width, face_height = map(int, face[:4])
        if face_width < face_config["minimum_width"] or face_height <= 0:
            return FaceExtractionResult(None, "face_too_small")
        landmarks = np.asarray(face[4:14], dtype=np.float32)
        landmarks_valid = _landmarks_are_valid(landmarks)
        if not landmarks_valid and settings.require_valid_landmarks:
            quality = FaceQuality(
                detection_confidence=float(face[14]),
                face_width=face_width,
                blur_variance=0.0,
                brightness=0.0,
                landmarks_valid=False,
                alignment_mode=settings.mode,
                accepted=False,
                rejection_reasons=("invalid_landmarks",),
            )
            return FaceExtractionResult(None, "quality_rejected", quality)
        crop = _crop_face(face_roi, (face_x, face_y, face_width, face_height), landmarks, settings)
        if crop is None or crop.size == 0:
            return FaceExtractionResult(None, "empty_crop")
        quality = _evaluate_face_quality(
            crop,
            detection_confidence=float(face[14]),
            face_width=face_width,
            landmarks_valid=landmarks_valid,
            alignment_mode=settings.mode,
            settings=settings,
        )
        if not quality.accepted:
            return FaceExtractionResult(None, "quality_rejected", quality)
        candidate = GenderCandidate(
            track_id=track_id,
            crop=crop,
            face_detection_confidence=float(face[14]),
            face_width=face_width,
            quality=quality,
        )
        return FaceExtractionResult(candidate, "candidate", quality)

    def extract_gender_candidate(self, frame: np.ndarray, track_id: int, bbox: BBox) -> GenderCandidate | None:
        """Compatibility crop-only API; use ``extract_gender_candidate_result`` for diagnostics."""

        return self.extract_gender_candidate_result(frame, track_id, bbox).candidate

    def _autocast_context(self):
        if self.use_mixed_precision:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def classify_gender_batch(self, candidates: list[GenderCandidate]) -> list[GenderEvidence]:
        """Classify all valid face crops with one forward pass."""
        if not candidates:
            return []
        if not self.face_enabled or self.gender_model is None or self.gender_transform is None:
            raise RuntimeError("Face classification was requested while the face branch is disabled.")
        tensors = [self.gender_transform(Image.fromarray(cv2.cvtColor(candidate.crop, cv2.COLOR_BGR2RGB))) for candidate in candidates]
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode(), self._autocast_context():
            logits = self.gender_model(batch)
            if self.use_horizontal_tta:
                logits = (logits + self.gender_model(torch.flip(batch, dims=[3]))) / 2.0
            probabilities = torch.softmax(logits, dim=1)
        logits_np = logits.float().cpu().numpy()
        confidences = probabilities.max(dim=1).values.float().cpu().numpy()
        return map_gender_batch(candidates, logits_np, confidences)

    def extract_body_gender_candidate(
        self, frame: np.ndarray, track_id: int, bbox: BBox
    ) -> BodyGenderCandidate | None:
        """Create one body crop only for a track that already missed face detection."""
        if not self.body_enabled:
            return None
        body_config = self.config["body_gender_classifier"]
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        if width < int(body_config.get("minimum_person_width", 1)):
            return None
        if height < int(body_config.get("minimum_person_height", 1)):
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return BodyGenderCandidate(
            track_id=track_id,
            crop=crop,
            person_width=width,
            person_height=height,
        )

    def classify_body_gender_batch(self, candidates: list[BodyGenderCandidate]) -> list[BodyGenderEvidence]:
        """Classify one bounded body-fallback batch with calibrated confidence."""
        if not candidates:
            return []
        if not self.body_enabled or self.body_model is None or self.body_transform is None:
            raise RuntimeError("Body gender inference was requested while the body branch is disabled.")
        tensors = [
            self.body_transform(Image.fromarray(cv2.cvtColor(candidate.crop, cv2.COLOR_BGR2RGB)))
            for candidate in candidates
        ]
        batch = torch.stack(tensors).to(self.device, non_blocking=self.device.type == "cuda")
        with torch.inference_mode(), self._autocast_context():
            calibrated_logits = self.body_model(batch) / self.body_temperature
            probabilities = torch.softmax(calibrated_logits, dim=1)
        logits_np = calibrated_logits.float().cpu().numpy()
        confidences = probabilities.max(dim=1).values.float().cpu().numpy()
        return map_body_gender_batch(candidates, logits_np, confidences)
