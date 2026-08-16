from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from pathlib import Path

import cv2
import gradio as gr

from src.inference.pipeline import CrowdGenderPipeline
from src.inference.live_stream import LiveFrameProcessor
from src.inference.video_io import run_ffmpeg


PROJECT_ROOT = Path(__file__).resolve().parent
# The interactive app uses the classroom/room profile by default. Set
# PIPELINE_CONFIG to a calibrated deployment profile when a different camera
# installation should be used.
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline-classroom-template.yaml"
CONFIG_PATH = Path(os.getenv("PIPELINE_CONFIG", DEFAULT_CONFIG_PATH))
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "gender_classifier"
    / "face_gender_classifier_mobilenet_v3_large.pth"
)
# Each live stream owns a YOLO tracker and three model objects. Keep this demo bounded so
# reconnects or multiple browser tabs cannot retain GPU models indefinitely.
# Send a bounded webcam cadence so mobile/tunnel uploads do not overwhelm the
# inference worker or accumulate network latency. Override per deployment with
# LIVE_STREAM_EVERY_SECONDS when a faster local stream is available.
LIVE_STREAM_EVERY_SECONDS = max(0.01, float(os.getenv("LIVE_STREAM_EVERY_SECONDS", "0.25")))


@dataclass
class LivePipelineEntry:
    """One persistent tracker plus its capacity-one newest-frame worker."""

    processor: LiveFrameProcessor
    last_used: float


LIVE_PIPELINES: dict[str, LivePipelineEntry] = {}
LIVE_PIPELINES_LOCK = RLock()
LIVE_PIPELINE_MAX_SESSIONS = max(1, int(os.getenv("LIVE_PIPELINE_MAX_SESSIONS", "1")))
LIVE_PIPELINE_TTL_SECONDS = max(0.0, float(os.getenv("LIVE_PIPELINE_TTL_SECONDS", "600")))

# Upload analysis is also stateful and uses the same GPU as live webcam
# inference. Keep one warm instance for repeated clips, but never let an upload
# construct a second model set beside a live session.
UPLOAD_PIPELINE: CrowdGenderPipeline | None = None
UPLOAD_PIPELINE_LOCK = RLock()
PIPELINE_OWNERSHIP_LOCK = RLock()
UPLOAD_JOB_ACTIVE = False


def _model_path() -> str:
    """Return the explicit local production checkpoint path.

    Model acquisition is an operator/deployment step, not an application
    request side effect. ``ModelRuntime`` reports a clear missing-file error
    if this asset was not provisioned before startup.
    """

    return str(Path(os.getenv("GENDER_MODEL_PATH", str(DEFAULT_MODEL_PATH))))


def _new_pipeline() -> CrowdGenderPipeline:
    pipeline = CrowdGenderPipeline(
        config_path=str(CONFIG_PATH),
        model_path=_model_path(),
    )
    pipeline.warmup()
    return pipeline


def _begin_upload_job() -> None:
    """Claim the one-GPU upload slot before loading or reusing its pipeline."""

    global UPLOAD_JOB_ACTIVE
    with PIPELINE_OWNERSHIP_LOCK:
        with LIVE_PIPELINES_LOCK:
            if LIVE_PIPELINES:
                raise RuntimeError("Close the active live session before analyzing an uploaded video.")
        if UPLOAD_JOB_ACTIVE:
            raise RuntimeError("Another uploaded video is already being analyzed.")
        UPLOAD_JOB_ACTIVE = True


def _upload_pipeline_for_job() -> CrowdGenderPipeline:
    """Return the warm upload pipeline, resetting only stream-local state."""

    global UPLOAD_PIPELINE
    with UPLOAD_PIPELINE_LOCK:
        if UPLOAD_PIPELINE is None:
            UPLOAD_PIPELINE = _new_pipeline()
        else:
            UPLOAD_PIPELINE.reset()
        return UPLOAD_PIPELINE


def _finish_upload_job(pipeline: CrowdGenderPipeline | None) -> None:
    """Reset the pooled upload stream and release its ownership claim."""

    global UPLOAD_JOB_ACTIVE
    if pipeline is not None:
        with suppress(Exception):
            pipeline.reset()
    with PIPELINE_OWNERSHIP_LOCK:
        UPLOAD_JOB_ACTIVE = False


def _run_ffmpeg(arguments: list[str]) -> None:
    run_ffmpeg(arguments[1:])


def _open_h264_writer(
    path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter | None:
    """Open a browser-compatible H.264 writer when the OpenCV backend supports it."""

    width, height = frame_size
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (width, height),
    )
    if writer.isOpened():
        return writer
    writer.release()
    return None


def _report(stats: dict) -> str:
    crowd = stats.get("crowd", {})
    identity = stats.get("identity", {})
    crossing = stats.get("crossing", {})
    attributes = stats.get("attributes", {}).get("visual_presentation", {})
    trajectory = stats.get("trajectory", {})
    spatial = stats.get("spatial", {})
    density = spatial.get("density", {})
    zones = spatial.get("zones", {})
    space = stats.get("space", {})
    occupancy = space.get("occupancy", {})
    physical_density = space.get("physical_density", {})
    distribution = space.get("crowd_distribution", {})
    history = stats.get("history", {})
    flow = history.get("flow", {})
    classroom = stats.get("classroom", {})
    classroom_room = classroom.get("room", {})
    classroom_layout = classroom.get("layout", {})
    classroom_seats = classroom.get("seats", {})
    classroom_aisles = classroom.get("aisles", {})
    runtime = stats.get("runtime", {})
    detector = runtime.get("detector", {})
    detector_settings = detector.get("settings", {})
    router = runtime.get("attribute_router", {})
    route_counts = router.get("route_counts", {})
    face_extraction = router.get("face_extraction_status_counts", {})
    face_quality = router.get("face_quality", {})
    evidence_states = router.get("active_face_evidence_states", {})
    fallback_reasons = router.get("body_fallback_reason_counts", {})
    timing = runtime.get("timing_ms", {})
    live_stream = runtime.get("live_stream", {})
    gender = crowd.get("gender_counts", {})
    ratios = crowd.get("ratios", {})
    sources = attributes.get("source_counts", {})
    zone_summary = " | ".join(
        f"{name}: {state.get('current_count', 0)} ({state.get('density_per_100k_reference_pixels', 0):.1f}/100k)"
        for name, state in zones.items()
    ) or "No camera zones configured"
    if occupancy.get("calibrated"):
        occupancy_summary = (
            f"{occupancy.get('percentage', 0.0):.1f}% "
            f"({occupancy.get('confirmed_active_count', 0)}/{occupancy.get('capacity', 0)})"
        )
    else:
        occupancy_summary = "not calibrated"
    if physical_density.get("calibrated"):
        physical_density_summary = f"{physical_density.get('people_per_m2', 0.0):.3f} confirmed people/m²"
    else:
        physical_density_summary = "not calibrated"
    distribution_summary = " | ".join(
        f"{item.get('name', 'zone')}: {item.get('percentage', 0.0):.1f}%"
        for item in distribution.get("zones", [])
    )
    outside = distribution.get("outside_zones", {})
    if distribution.get("denominator_confirmed_count", 0):
        distribution_summary = (
            f"{distribution_summary} | outside: {outside.get('percentage', 0.0):.1f}%"
            if distribution_summary
            else f"outside: {outside.get('percentage', 0.0):.1f}%"
        )
    else:
        distribution_summary = "no confirmed tracks"
    flow_windows = flow.get("windows", [])
    flow_60 = next((item for item in flow_windows if item.get("window_seconds") == 60.0), None)
    flow_summary = (
        f"60s IN {flow_60.get('in_per_minute', 0.0):.1f}/min | "
        f"OUT {flow_60.get('out_per_minute', 0.0):.1f}/min"
        if flow_60 is not None
        else "waiting for flow window"
    )
    classroom_status = str(classroom.get("status", "not_configured"))
    if classroom_status == "ready":
        maximum_capacity = classroom_room.get("maximum_capacity")
        capacity_summary = (
            f"{classroom_room.get('current_people', 0)}/{maximum_capacity}"
            if maximum_capacity is not None
            else f"{classroom_room.get('current_people', 0)} (formal capacity not set)"
        )
        seat_utilization = classroom_seats.get("utilization")
        seat_summary = (
            f"{classroom_seats.get('occupied_seats', 0)}/{classroom_seats.get('enabled_seats', 0)} occupied"
            + (f" ({float(seat_utilization) * 100:.1f}%)" if isinstance(seat_utilization, (int, float)) else "")
        )
        classroom_density = classroom_room.get("people_per_m2")
        density_suffix = (
            f" | density {float(classroom_density):.3f} people/m²"
            if isinstance(classroom_density, (int, float))
            else ""
        )
        classroom_summary = (
            f"{classroom_room.get('name', 'configured room')} | people {capacity_summary} | seats {seat_summary}"
            f"{density_suffix}"
        )
    elif classroom_status == "geometry_required":
        configured_seats = classroom_seats.get("enabled_seats")
        configured_rows = classroom_layout.get("rows")
        layout_name = classroom_layout.get("template", "layout")
        classroom_summary = (
            f"{layout_name} | {configured_rows} rows | {configured_seats} seats configured; "
            "camera seat polygons are required before occupancy is measured"
        )
    elif classroom_status == "room_configured":
        classroom_summary = "room profile is configured; select a session layout to measure seats"
    else:
        classroom_summary = "layout not configured"
    aisle_items = classroom_aisles.get("items", [])
    aisle_summary = " | ".join(
        f"{item.get('name', 'aisle')}: {item.get('current_people', 0)}"
        for item in aisle_items
        if isinstance(item, dict)
    ) or "not configured"
    classroom_lines = (
        f"<p><b>Classroom:</b> {classroom_summary}</p>"
        f"<p><b>Aisles:</b> {aisle_summary}</p>"
        if classroom.get("enabled")
        else ""
    )
    mean_face_width = face_quality.get("means", {}).get("face_width")
    face_width_summary = f"{float(mean_face_width):.1f}" if isinstance(mean_face_width, (int, float)) else "—"
    detector_name = Path(str(detector_settings.get("model_path", "configured detector"))).name
    effective_end2end = detector.get("effective_end2end")
    detector_mode = "E2E" if effective_end2end is True else "NMS" if effective_end2end is False else "model default"
    if isinstance(live_stream, dict) and live_stream:
        live_worker = live_stream.get("worker_processing_ms", {})
        live_queue = live_stream.get("queue_wait_ms", {})
        live_end_to_end = live_stream.get("end_to_end_ms", {})
        live_stream_lines = f"""
      <p><b>Live cadence:</b> {live_stream.get('configured_cadence_ms', 0):.0f} ms, newest-frame queue (capacity {live_stream.get('queue_capacity', 1)}) &nbsp;
         ingress {live_stream.get('frames_received', 0)} | processed {live_stream.get('frames_processed', 0)} | pending {live_stream.get('pending_frames', 0)}</p>
      <p><b>Live drops:</b> replaced before inference {live_stream.get('frames_dropped_replaced', 0)} | reset/shutdown {live_stream.get('frames_dropped_reset', 0) + live_stream.get('frames_dropped_shutdown', 0)} &nbsp;
         <b>Live p95:</b> queue {live_queue.get('p95', 0):.0f} ms | worker {live_worker.get('p95', 0):.0f} ms | end-to-end {live_end_to_end.get('p95', 0):.0f} ms</p>
        """
    else:
        live_stream_lines = ""
    return f"""
    <div style="padding: 16px; border: 1px solid #335; border-radius: 10px; background: #10131a; color: #f3f6ff">
      <h3 style="margin-top: 0">Crowd analysis</h3>
      <p><b>Tracker:</b> {runtime.get('tracker_type', 'unknown')}</p>
      <p><b>Detector:</b> {detector_name} ({detector_mode}) &nbsp;
         input conf {detector_settings.get('tracker_input_confidence', '—')} &nbsp;
         imgsz {detector_settings.get('imgsz', 'auto')}</p>
      <p><b>Active tracks:</b> {crowd.get('current_count', 0)}</p>
      <p><b>Confirmed active tracks:</b> {crowd.get('confirmed_active_count', 0)}</p>
      <p><b>Unique local tracks:</b> {crowd.get('unique_track_count', 0)}</p>
      <p><b>Session person IDs:</b> {identity.get('active_person_count', 0)} active / {identity.get('unique_person_count', 0)} unique &nbsp;
         <b>Recovered ID bindings:</b> {identity.get('recovered_track_bindings', 0)} &nbsp;
         <b>Ambiguous matches rejected:</b> {identity.get('ambiguous_match_rejections', 0)}</p>
      <p><b>Female:</b> {gender.get('female', 0)} ({ratios.get('female', 0) * 100:.1f}%) &nbsp;
         <b>Male:</b> {gender.get('male', 0)} ({ratios.get('male', 0) * 100:.1f}%)</p>
      <p><b>Unknown:</b> {gender.get('unknown', 0)} ({ratios.get('unknown', 0) * 100:.1f}%)</p>
      <p><b>Gender coverage:</b> {crowd.get('gender_coverage', 0) * 100:.1f}% &nbsp;
         <b>Face batch:</b> {runtime.get('gender_batch_size', 0)} &nbsp;
         <b>Body batch:</b> {runtime.get('body_gender_batch_size', 0)}</p>
      <p><b>Attribute source:</b> face {sources.get('face', 0)} | body {sources.get('body', 0)} | unknown {sources.get('unknown', 0)}</p>
      <p><b>Attribute router:</b> {router.get('mode', 'face_first')} &nbsp;
         face {route_counts.get('face', 0)} | face-to-body {route_counts.get('face_then_body', 0)} |
         body {route_counts.get('body', 0)} | skipped {route_counts.get('unknown', 0)}</p>
      <p><b>Face diagnostics:</b> candidate {face_extraction.get('candidate', 0)} | no face {face_extraction.get('no_face', 0)} |
         quality rejected {face_extraction.get('quality_rejected', 0)} | uncertain {evidence_states.get('uncertain_after_minimum_observations', 0)} |
         Body after Face uncertainty {fallback_reasons.get('face_uncertain_after_minimum_observations', 0)}</p>
      <p><b>Face crop:</b> accepted {face_quality.get('accepted', 0)}/{face_quality.get('samples', 0)} quality samples &nbsp;
         mean width {face_width_summary} px</p>
      <p><b>Motion:</b> {trajectory.get('moving_count', 0)} moving | {trajectory.get('stationary_count', 0)} stationary | mean {trajectory.get('mean_speed_reference_px_per_second', 0):.1f} reference px/s</p>
      <p><b>Screen density:</b> {density.get('density_per_100k_reference_pixels', 0):.2f} confirmed tracks / 100k reference px</p>
      <p><b>Occupancy:</b> {occupancy_summary} &nbsp; <b>Physical density:</b> {physical_density_summary}</p>
      <p><b>Distribution:</b> {distribution_summary}</p>
      <p><b>Zones:</b> {zone_summary}</p>
      {classroom_lines}
      <p><b>Peak / average confirmed:</b> {history.get('peak_confirmed_occupancy', 0)} / {history.get('average_confirmed_occupancy', 0):.1f} &nbsp; <b>Flow:</b> {flow_summary}</p>
      <p><b>Processing:</b> {runtime.get('processing_fps', 0):.1f} FPS &nbsp;
         <b>p95:</b> {timing.get('p95', 0):.0f} ms</p>
      {live_stream_lines}
      <p><b>Geometric line crossings:</b> IN {crossing.get('in', 0)} | OUT {crossing.get('out', 0)}</p>
    </div>
    """


def process_video_clip(video_path: str | None, progress=gr.Progress()):
    if not video_path:
        return None, "<p>Please upload a video first.</p>"
    pipeline: CrowdGenderPipeline | None = None
    capture: cv2.VideoCapture | None = None
    writer: cv2.VideoWriter | None = None
    upload_claimed = False
    try:
        _begin_upload_job()
        upload_claimed = True
        pipeline = _upload_pipeline_for_job()
        job_dir = Path(tempfile.mkdtemp(prefix="crowd_gender_"))
        normalized_input = job_dir / "input.mp4"
        raw_output = job_dir / "annotated_raw.mp4"
        browser_output = job_dir / "annotated.mp4"
        progress(0.02, desc="Opening video")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            capture = None
            progress(0.02, desc="Normalizing unsupported video")
            _run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(normalized_input),
                ]
            )
            capture = cv2.VideoCapture(str(normalized_input))
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the uploaded video after optional normalization.")
        total_frames = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError("Video dimensions are invalid.")

        last_stats, frame_number = {}, 0
        direct_h264 = False
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotated, last_stats = pipeline.process_frame(frame, timestamp_seconds=frame_number / fps)
            if writer is None:
                output_height, output_width = annotated.shape[:2]
                writer = _open_h264_writer(browser_output, fps, (output_width, output_height))
                direct_h264 = writer is not None
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(raw_output),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (output_width, output_height),
                    )
                if not writer.isOpened():
                    raise RuntimeError("OpenCV could not create the output video.")
            writer.write(annotated)
            frame_number += 1
            progress(min(0.95, frame_number / total_frames), desc=f"Processing frame {frame_number}/{total_frames}")
        if writer is not None:
            writer.release()
            writer = None
        if frame_number == 0:
            raise RuntimeError("No frames were decoded from the video.")
        if not direct_h264:
            progress(0.97, desc="Encoding browser-compatible video")
            _run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(raw_output),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    str(browser_output),
                ]
            )
        return str(browser_output), _report(last_stats)
    except Exception as error:
        return None, f"<p style='color:#d55'><b>Processing failed:</b> {error}</p>"
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if upload_claimed:
            _finish_upload_job(pipeline)


def _session_key(request: gr.Request | None) -> str:
    return getattr(request, "session_hash", None) or "local"


def _release_live_pipeline(key: str) -> None:
    with LIVE_PIPELINES_LOCK:
        entry = LIVE_PIPELINES.pop(key, None)
    if entry is not None:
        entry.processor.close()


def _evict_inactive_live_pipelines(active_key: str) -> None:
    """Free tracker/model ownership from stale or surplus webcam sessions."""
    now = monotonic()
    evicted: list[LivePipelineEntry] = []
    with LIVE_PIPELINES_LOCK:
        if LIVE_PIPELINE_TTL_SECONDS:
            stale_keys = [
                key
                for key, entry in LIVE_PIPELINES.items()
                if key != active_key and now - entry.last_used >= LIVE_PIPELINE_TTL_SECONDS
            ]
            for key in stale_keys:
                evicted.append(LIVE_PIPELINES.pop(key))
        while len(LIVE_PIPELINES) >= LIVE_PIPELINE_MAX_SESSIONS and active_key not in LIVE_PIPELINES:
            oldest_key = min(LIVE_PIPELINES, key=lambda key: LIVE_PIPELINES[key].last_used)
            evicted.append(LIVE_PIPELINES.pop(oldest_key))
    for entry in evicted:
        entry.processor.close()


def _live_processor_for_session(key: str) -> LiveFrameProcessor:
    """Return the persistent per-session tracker without letting sessions share it."""

    with PIPELINE_OWNERSHIP_LOCK:
        if UPLOAD_JOB_ACTIVE:
            raise RuntimeError("An uploaded video is being analyzed; live webcam will resume when it finishes.")
        while True:
            _evict_inactive_live_pipelines(key)
            with LIVE_PIPELINES_LOCK:
                entry = LIVE_PIPELINES.get(key)
                if entry is not None:
                    entry.last_used = monotonic()
                    return entry.processor
                if len(LIVE_PIPELINES) < LIVE_PIPELINE_MAX_SESSIONS:
                    # Keep creation serialized: model warm-up uses the same GPU as
                    # the worker and two simultaneous first callbacks must not
                    # construct two trackers for one configured session limit.
                    processor = LiveFrameProcessor(
                        _new_pipeline(),
                        cadence_seconds=LIVE_STREAM_EVERY_SECONDS,
                    )
                    LIVE_PIPELINES[key] = LivePipelineEntry(processor=processor, last_used=monotonic())
                    return processor
            # Another callback filled the bounded session pool after the eviction
            # check. Retry so it can be evicted and closed outside the dict lock.


def _with_live_telemetry(stats: dict, telemetry: dict) -> dict:
    output = dict(stats)
    runtime = dict(stats.get("runtime", {}))
    runtime["live_stream"] = telemetry
    output["runtime"] = runtime
    return output


def process_live_frame(frame, request: gr.Request | None = None):
    if frame is None:
        # A streaming callback can briefly receive ``None`` while the browser
        # starts/stops the camera. Keep the last rendered frame/report instead
        # of asking Gradio to clear both outputs, which causes a visible flash.
        return gr.skip(), gr.skip()
    try:
        key = _session_key(request)
        processor = _live_processor_for_session(key)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        processor.submit(bgr, submitted_at=monotonic())
        result = processor.latest_result()
        telemetry = processor.telemetry()
        if result is None:
            # Inference runs on a private worker and may not have completed by
            # the time this ingress callback returns. Display the current raw
            # webcam frame and leave the existing report untouched; returning
            # ``None`` here makes Gradio clear the image and produces the
            # recurring "waiting" flicker reported by the UI.
            if telemetry.get("last_error"):
                return frame, f"<p style='color:#d55'><b>Live processing failed:</b> {telemetry['last_error']}</p>"
            return frame, gr.skip()
        stats = _with_live_telemetry(result.stats, telemetry)
        return cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB), _report(stats)
    except Exception as error:
        # Keep the camera image visible while surfacing the error; clearing the
        # image on every failed callback creates the same flash as the startup
        # waiting path.
        return frame, f"<p style='color:#d55'><b>Live processing failed:</b> {error}</p>"


def reset_live(request: gr.Request | None = None):
    key = _session_key(request)
    with LIVE_PIPELINES_LOCK:
        entry = LIVE_PIPELINES.get(key)
        if entry is not None:
            entry.last_used = monotonic()
    if entry is not None:
        entry.processor.reset()
    return None, "<p>Live session state reset. The loaded models remain warm for the next webcam frame.</p>"


with gr.Blocks(title="Crowd Gender Analytics") as demo:
    gr.Markdown(
        "# Crowd analytics\n"
        "Person detection, track-based statistics, geometric line-crossing and binary image-label inference. "
        "Results are estimates; do not use them for consequential decisions."
    )
    with gr.Tab("Upload video"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Input video")
                process_button = gr.Button("Analyze", variant="primary")
            with gr.Column():
                video_output = gr.Video(label="Annotated output")
                video_report = gr.HTML("<p>No video has been processed.</p>")
        process_button.click(process_video_clip, inputs=video_input, outputs=[video_output, video_report])
    with gr.Tab("Live webcam"):
        with gr.Row():
            webcam_input = gr.Image(
                sources=["webcam"],
                streaming=True,
                type="numpy",
                label="Camera",
                webcam_constraints={
                    "video": {
                        "width": {"ideal": 640, "max": 640},
                        "height": {"ideal": 480, "max": 480},
                        "facingMode": {"ideal": "environment"},
                    },
                    "audio": False,
                },
            )
            webcam_output = gr.Image(label="Annotated frame")
        live_report = gr.HTML("<p>Start the webcam to begin.</p>")
        reset_button = gr.Button("Reset live session")
        webcam_input.stream(
            process_live_frame,
            inputs=webcam_input,
            outputs=[webcam_output, live_report],
            stream_every=LIVE_STREAM_EVERY_SECONDS,
            trigger_mode="always_last",
            show_progress="hidden",
            # Ingress returns immediately to the browser. LiveFrameProcessor
            # is the only owner of model execution and accepts one newest
            # pending frame, so Gradio's global queue cannot build a stale
            # webcam backlog or overlap a persistent FastTracker call.
            queue=False,
            concurrency_limit=None,
        )
        reset_button.click(reset_live, outputs=[webcam_output, live_report])


if __name__ == "__main__":
    # Upload jobs remain queued. Webcam events opt out above and use their own
    # per-session bounded latest-frame mailbox.
    demo.queue(default_concurrency_limit=1, max_size=4).launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
