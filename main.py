from __future__ import annotations

import argparse
import csv
import logging
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median, quantiles

import cv2
import numpy as np

from analysis_profiles import ANALYSIS_PROFILES, get_analysis_profile
from anchor_metrics import ANCHOR_GROUPS, AnchorVelocity, AnchorVelocityTracker, smooth_series
from bar_anchor import (
    BarMeasurementGate,
    BarAnchorState,
    BarAnchorTracker,
    Point2D,
)
from biomech_angles import compute_ipf_flags
from calibration import create_calibration_from_plate_diameter, estimate_plate_diameter_pixels
from detect_objects import (
    BeigePlateDetector,
    ColorPlateDetector,
    Detection,
    MultiColorPlateDetector,
    YoloObjectDetector,
)
from io_video import (
    Frame,
    OutputMode,
    VideoMetadata,
    VideoReader,
    measure_video_fps,
    process_video_two_pass,
    read_video_metadata,
    resolve_output_geometry,
)
from inference import InferenceTransform
from kinematics import BarMeasurement, ReconstructedBarSample, reconstruct_bar_kinematics
from load_estimation import LoadEstimate
from mask_cache import MaskFrameCache
from metrics import (
    EXERCISE_DISPLACEMENT_DEFAULTS,
    BiomechanicsConfig,
    BiomechanicsEngine,
    KinematicSample,
    get_exercise_profile,
)
from pose import HybridPoseEstimator, PoseResult, YoloPoseEstimator, create_pose_estimator, refine_pose_with_mask
from rep_review import RepDecision, decide_rep_validations
from reporting import AnalysisReport, RepReportBuilder, write_csv_report, write_json_report
from render_overlay import OverlayConfig, OverlayRenderer
from segmentation import SegmentationResult, create_segmenter, select_subject_mask
from technique import TechniqueAssessment, TechniqueMonitor
from track import PointTracker
from validation_outputs import create_validation_run, save_validation_screenshots
from video_export import convert_to_mobile_mp4, make_mobile_compatible_in_place


logger = logging.getLogger("powerai")
DEFAULT_BAR_OBJECT_MODEL = Path(__file__).with_name("models") / "powerai_bar_detector.pt"
BarPathPoint = tuple[float, float] | None


@dataclass
class FrameRecord:
    """Per-frame result cached in pass 1 and completed after kinematic replay."""

    pose: PoseResult
    raw_pose: PoseResult
    detections: list[Detection]
    sample: KinematicSample | None = None
    technique: TechniqueAssessment | None = None
    bar_anchor: BarAnchorState | None = None
    load_estimate: LoadEstimate | None = None
    measurement: BarMeasurement | None = None
    depth_ok: bool | None = None
    lockout_ok: bool | None = None
    anchor_velocities: list[AnchorVelocity] = field(default_factory=list)
    bar_path: list[BarPathPoint] = field(default_factory=list)
    mask_cached: bool = False


def estimate_plate_diameter_from_video(input_path: Path, max_frames: int = 120) -> float | None:
    detector = ColorPlateDetector(expected_diameter_pixels=None)
    observations: list[float] = []

    with VideoReader(input_path) as reader:
        for frame_index, frame in reader.frames():
            if frame_index >= max_frames:
                break

            diameter_pixels = estimate_plate_diameter_pixels(detector.detect(frame))
            if diameter_pixels is not None and _plate_diameter_observation_is_plausible(
                diameter_pixels,
                frame,
            ):
                observations.append(diameter_pixels)

    if len(observations) < 5:
        return None

    filtered = _reject_outlier_observations(observations)
    if len(filtered) < 3:
        filtered = observations

    return float(median(filtered))


def estimate_plate_diameter_from_tracker(
    input_path: Path,
    object_detector: object | None,
    fallback_object_detector: object | None,
    fps: float,
    max_frames: int = 90,
) -> float | None:
    """Estimate the Olympic plate diameter (in raw pixels) from the stabilized anchor
    rectangle -- the same tight box the overlay draws on the plate. This is robust to
    the colour detector over-merging the plate with the background, and works whether
    plates come from the trained YOLO model or the heuristic colour fallback."""
    tracker = BarAnchorTracker(fps=fps if fps > 0 else 30.0)
    observations: list[float] = []

    with VideoReader(input_path) as reader:
        for frame_index, frame in reader.frames():
            if frame_index >= max_frames:
                break

            detections = object_detector.detect(frame) if object_detector is not None else []
            if not detections and fallback_object_detector is not None:
                detections = fallback_object_detector.detect(frame)

            state = tracker.update(frame, detections)
            if state.source != "detection" or state.rect is None:
                continue

            diameter = max(state.rect.width, state.rect.height)
            if diameter > 0 and _plate_diameter_observation_is_plausible(diameter, frame):
                observations.append(diameter)

    if len(observations) < 4:
        return None

    filtered = _reject_outlier_observations(observations)
    if len(filtered) < 3:
        filtered = observations

    return float(median(filtered))


def estimate_plate_diameter_from_model_boxes(
    input_path: Path,
    object_detector: YoloObjectDetector,
    fps: float,
    max_frames: int = 90,
) -> float | None:
    """Estimate plate diameter from raw trained-model plate boxes while using the
    anchor tracker only to choose the bar plate. This keeps metric calibration off
    the display/anchor scale and avoids changing BarAnchorConfig.refined_plate_rect_scale."""
    tracker = BarAnchorTracker(fps=fps if fps > 0 else 30.0)
    observations: list[float] = []

    with VideoReader(input_path) as reader:
        for frame_index, frame in reader.frames():
            if frame_index >= max_frames:
                break

            detections = object_detector.detect(frame)
            if not detections:
                continue

            state = tracker.update(frame, detections)
            if state.source != "detection":
                continue

            plate = _nearest_plate_detection_to_anchor(state, detections)
            if plate is None:
                continue

            diameter = max(plate.width, plate.height)
            if diameter > 0 and _plate_diameter_observation_is_plausible(diameter, frame):
                observations.append(diameter)

    if len(observations) < 4:
        return None

    filtered = _reject_outlier_observations(observations)
    if len(filtered) < 3:
        filtered = observations

    return float(median(filtered))


def _nearest_plate_detection_to_anchor(
    state: BarAnchorState,
    detections: list[Detection],
) -> Detection | None:
    if state.point is None:
        return None

    plates = [detection for detection in detections if detection.label == "plate"]
    if not plates:
        return None

    return min(
        plates,
        key=lambda detection: (
            (detection.center[0] - state.point.x) ** 2
            + (detection.center[1] - state.point.y) ** 2
        ),
    )


def _bar_path_horizontal_drift_cm(
    bar_path: list[BarPathPoint],
    meters_per_pixel: float,
) -> float | None:
    visible_points = [point for point in bar_path if point is not None]
    if len(visible_points) < 2:
        return None

    xs = [point[0] for point in visible_points]
    return (max(xs) - min(xs)) * meters_per_pixel * 100.0


def _manual_load_estimate(load_kg: float | None) -> LoadEstimate | None:
    if load_kg is None:
        return None
    side_weight = max(0.0, (float(load_kg) - 20.0) / 2.0)
    return LoadEstimate(
        total_kg=float(load_kg),
        side_weight_kg=side_weight,
        colors=("manual",),
        confidence=1.0,
    )


def _strict_ipf_gate(value: bool | None, strict: bool) -> bool:
    if value is None:
        return not strict
    return value


def _compute_ipf_flags_with_pose_fallback(
    exercise: str,
    pose: PoseResult | None,
    fallback_pose: PoseResult | None,
) -> tuple[bool | None, bool | None]:
    depth_ok, lockout_ok = compute_ipf_flags(exercise, pose)
    if depth_ok is not None and lockout_ok is not None:
        return depth_ok, lockout_ok

    fallback_depth, fallback_lockout = compute_ipf_flags(exercise, fallback_pose)
    return (
        depth_ok if depth_ok is not None else fallback_depth,
        lockout_ok if lockout_ok is not None else fallback_lockout,
    )


def _warp_mask_with_optical_flow(
    previous_frame: Frame | None,
    current_frame: Frame,
    mask: np.ndarray | None,
) -> np.ndarray | None:
    if previous_frame is None or mask is None:
        return mask
    if previous_frame.shape[:2] != current_frame.shape[:2]:
        return mask

    if mask.shape[:2] != current_frame.shape[:2]:
        mask = cv2.resize(mask, (current_frame.shape[1], current_frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    height, width = mask.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (grid_x - flow[:, :, 0]).astype(np.float32)
    map_y = (grid_y - flow[:, :, 1]).astype(np.float32)
    return cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _processing_scale_from_video(
    input_path: Path,
    target_resolution: int,
    output_mode: OutputMode = "source",
) -> float:
    if output_mode == "source" and target_resolution <= 0:
        return 1.0

    with VideoReader(input_path) as reader:
        for _, frame in reader.frames():
            height, width = frame.shape[:2]
            if output_mode == "portrait-720":
                return min(720.0 / max(1, width), 1280.0 / max(1, height))
            larger = max(width, height)
            smaller = min(width, height)
            if larger <= 0 or larger <= target_resolution:
                return 1.0
            scale = target_resolution / larger
            min_smaller = target_resolution * 0.55
            if smaller * scale < min_smaller:
                return min_smaller / smaller
            return scale

    return 1.0


def _plate_diameter_observation_is_plausible(diameter_pixels: float, frame: Frame) -> bool:
    frame_height, frame_width = frame.shape[:2]
    frame_limit = min(frame_width, frame_height)
    return 18.0 <= diameter_pixels <= frame_limit * 0.92


def _reject_outlier_observations(observations: list[float]) -> list[float]:
    if len(observations) < 4:
        return list(observations)

    quartiles = quantiles(observations, n=4)
    q1, q3 = quartiles[0], quartiles[2]
    iqr = q3 - q1
    if iqr <= 0:
        return list(observations)

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return [value for value in observations if lower <= value <= upper]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analizador de video PowerNZ v1.")
    parser.add_argument("--input", required=True, type=Path, help="Path to the input video.")
    parser.add_argument("--output", required=True, type=Path, help="Path to save the analyzed video.")
    parser.add_argument(
        "--pose-backend",
        choices=["auto", "mediapipe", "yolo"],
        default="auto",
        help="Pose backend. Auto locks the athlete with YOLO and uses MediaPipe when it agrees.",
    )
    parser.add_argument(
        "--pose-model",
        default=None,
        type=Path,
        help="Pose model path. For MediaPipe Tasks use pose_landmarker_lite.task.",
    )
    parser.add_argument(
        "--object-model",
        default=None,
        type=Path,
        help="YOLO model with plate and bar_hub classes. Defaults to models/powerai_bar_detector.pt when present.",
    )
    parser.add_argument(
        "--disable-trained-object-model",
        action="store_true",
        help="Do not auto-load models/powerai_bar_detector.pt; use the heuristic fallback unless --object-model is provided.",
    )
    parser.add_argument(
        "--object-confidence",
        default=0.25,
        type=float,
        help="Minimum confidence for the trained bar detector (plate/bar_hub). Raise it for fewer false positives.",
    )
    parser.add_argument(
        "--plate-diameter-px",
        default=None,
        type=float,
        help="Manual Olympic plate diameter in pixels; used only for manual calibration or auto fallback.",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=["auto", "manual"],
        default="auto",
        help="Auto estimates the plate size from trained detections; manual requires --plate-diameter-px.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(ANALYSIS_PROFILES),
        default="balanced",
        help="Inference preset. Balanced keeps the 720p overlay while reducing model work.",
    )
    parser.add_argument(
        "--velocity-window-seconds",
        default=None,
        type=float,
        help="Seconds visible in the synchronized velocity graph. Defaults to the selected profile.",
    )
    parser.add_argument(
        "--max-frames",
        default=0,
        type=int,
        help="Maximum number of frames to process. Use 0 for the full video.",
    )
    parser.add_argument(
        "--disable-plate-heuristic",
        action="store_true",
        help="Deprecated compatibility flag. Keeps the color-based plate fallback disabled.",
    )
    parser.add_argument(
        "--enable-plate-heuristic",
        action="store_true",
        help="Enable the legacy color-based plate fallback. Disabled by default when a trained model is available.",
    )
    parser.add_argument(
        "--disable-auto-calibration",
        action="store_true",
        help="Use --plate-diameter-px exactly instead of estimating plate size from the video.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        type=Path,
        help="Optional path to save a JSON analysis report.",
    )
    parser.add_argument(
        "--report-csv",
        default=None,
        type=Path,
        help="Optional path to save a CSV report with one row per completed rep.",
    )
    parser.add_argument(
        "--mobile-output",
        default=None,
        type=Path,
        help="Optional H.264 MP4 output for phones/social apps. Requires ffmpeg.",
    )
    parser.add_argument(
        "--no-mobile-conversion",
        action="store_true",
        help="Keep the raw OpenCV MP4 without automatic H.264 conversion.",
    )
    parser.add_argument(
        "--mobile-max-dimension",
        default=0,
        type=int,
        help="Maximum width/height for mobile MP4 conversion. Defaults to 0 to keep original size.",
    )
    parser.add_argument(
        "--segmentation-backend",
        choices=["auto", "none", "mediapipe-mask", "mediapipe", "yolo-seg", "yolo", "pose-hull"],
        default="auto",
        help="Subject silhouette backend. Auto falls back to a pose-based hull.",
    )
    parser.add_argument(
        "--segmentation-model",
        default=None,
        type=Path,
        help="YOLO segmentation model with 'athlete' class for subject mask.",
    )
    parser.add_argument(
        "--hub-confidence-threshold",
        default=0.28,
        type=float,
        help="Minimum hub confidence to add a point to the bar trajectory (default: 0.28).",
    )
    parser.add_argument(
        "--measurement-requires-hub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a reliable bar_hub detection before calculating velocity (default: true).",
    )
    parser.add_argument(
        "--show-unmeasured-anchor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the locked plate even when the hub is not reliable enough to measure (default: true).",
    )
    parser.add_argument(
        "--plate-box-style",
        choices=["full", "corners"],
        default="full",
        help="How to draw the plate anchor box. Default: full.",
    )
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        help="Force BoT-SORT/ByteTrack temporal tracking when the object detector supports it.",
    )
    parser.add_argument(
        "--disable-object-tracking",
        action="store_true",
        help="Disable YOLO temporal tracking for bar/plate detections.",
    )
    parser.add_argument(
        "--tracker-config",
        default=None,
        type=Path,
        help="Optional YAML tracker config (e.g. bytetrack.yaml).",
    )
    parser.add_argument(
        "--debug-anchor",
        action="store_true",
        help="Draw anchor confidence, source and missing-frame diagnostics.",
    )
    parser.add_argument(
        "--exercise",
        choices=["deadlift", "squat", "bench"],
        default="deadlift",
        help=(
            "Lift being analyzed. Drives rep detection (deadlift goes up first; squat/bench "
            "go down first) and the technique cues. Default: deadlift."
        ),
    )
    parser.add_argument(
        "--view-mode",
        choices=["auto", "lateral", "diagonal"],
        default="auto",
        help="Deprecated/no-op. Kept for compatibility; the best visible plate anchor is always used.",
    )
    parser.add_argument(
        "--athlete-lock",
        choices=["auto", "off"],
        default="auto",
        help="Keep the subject mask locked to the primary athlete.",
    )
    parser.add_argument(
        "--use-grabcut",
        action="store_true",
        help="Enable GrabCut refinement for silhouette segmentation (slower but higher quality).",
    )
    parser.add_argument(
        "--max-resolution",
        default=0,
        type=int,
        help=(
            "Maximum source-mode output width/height in pixels (longest side). Ignored by the default "
            "portrait-720 output unless --output-format source is used."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=["portrait-720", "source"],
        default="portrait-720",
        help="Output geometry. Default portrait-720 exports a 720x1280 9:16 canvas without cropping.",
    )
    parser.add_argument(
        "--velocity-chart",
        choices=["bar", "multi"],
        default="bar",
        help="Bottom chart content. Default bar keeps the graph focused on bar speed.",
    )
    parser.add_argument(
        "--body-velocity-display",
        choices=["compact", "off"],
        default="compact",
        help="Show compact on-body joint velocity tags outside the bottom chart.",
    )
    parser.add_argument(
        "--velocity-loss-threshold",
        default=20.0,
        type=float,
        help="Percent velocity loss from the best rep that triggers a report warning.",
    )
    parser.add_argument(
        "--load-kg",
        default=None,
        type=float,
        help="Peso total manual de la barra en kg. Si no se indica, no se muestra carga.",
    )
    parser.add_argument(
        "--strict-ipf-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require pose evidence for IPF depth/lockout when that rule applies. "
            "Default: true; use --no-strict-ipf-validation only for debugging bar-only clips."
        ),
    )
    parser.add_argument(
        "--min-rep-displacement-m",
        default=None,
        type=float,
        help="Minimum vertical bar displacement in meters to validate a rep. Defaults per --exercise.",
    )
    parser.add_argument(
        "--min-rep-frames",
        default=18,
        type=int,
        help="Minimum number of frames a rep must span to be validated.",
    )
    parser.add_argument(
        "--min-rep-gap-frames",
        default=8,
        type=int,
        help="Minimum frames between two validated reps to keep them separate.",
    )
    parser.add_argument(
        "--validation-run-label",
        default=None,
        help="Save validation outputs under outputs/validation/runs/<timestamp>_<label>.",
    )
    parser.add_argument(
        "--save-validation-screenshots",
        action="store_true",
        help="Extract representative screenshots into the validation run screenshots folder.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Defaults to INFO.",
    )
    return parser


def _anchor_point_is_usable(
    point: Point2D | None,
    confidence: float,
    source: str,
    missing_frames: int,
) -> bool:
    if point is None or source in {"lost", "pose_seed", "wrist"}:
        return False
    if confidence < 0.22:
        return False
    if source == "hold" and missing_frames > 8:
        return False
    return True


def _sample_point_from_single_anchor(state: BarAnchorState) -> Point2D | None:
    return BarMeasurementGate().point_for_measurement(state)


def _bar_height_hint_from_pose(pose: PoseResult | None) -> Point2D | None:
    """Midpoint of the visible wrists: an approximation of where the bar is held.
    Used to keep the anchor on the lifting plate instead of one on the floor."""
    if pose is None or not pose.detected or not pose.keypoints:
        return None

    wrists = [
        keypoint
        for keypoint in pose.keypoints
        if keypoint.name in {"left_wrist", "right_wrist"} and keypoint.visibility >= 0.35
    ]
    if not wrists:
        return None

    center_x = sum(keypoint.x for keypoint in wrists) / len(wrists)
    center_y = sum(keypoint.y for keypoint in wrists) / len(wrists)
    return Point2D(center_x, center_y)


def _reset_visible_motion_history(
    bar_path: list[BarPathPoint],
    anchor_velocity_history: dict[str, list[float]],
    velocity_frame_history: list[int],
) -> None:
    bar_path.clear()
    velocity_frame_history.clear()
    for values in anchor_velocity_history.values():
        values.clear()


def _append_visible_motion_history(
    anchor_velocity_history: dict[str, list[float]],
    velocity_frame_history: list[int],
    anchor_velocities: list[AnchorVelocity],
    sample: KinematicSample,
    frame_index: int,
) -> None:
    current_anchor_velocities = {
        anchor.name: anchor.velocity_mps
        for anchor in anchor_velocities
    }
    for anchor_name, _ in ANCHOR_GROUPS:
        anchor_velocity_history[anchor_name].append(
            current_anchor_velocities.get(anchor_name, float("nan"))
        )
    anchor_velocity_history["bar"].append(sample.smoothed_velocity_mps)
    velocity_frame_history.append(frame_index)


def _filter_detections_near_bar(
    detections: list[Detection],
    pose: PoseResult | None,
    frame_shape: tuple[int, ...],
) -> list[Detection]:
    """Keep only plate/bar detections plausibly on the bar (a band around the wrists),
    rejecting plates on the floor or in the background. This is the strongest runtime
    lever for reliable disc/bar detection and, therefore, reliable velocity. Returns the
    input unchanged when there is no reliable wrist info or filtering would drop every
    plate, so the anchor is never starved."""
    if pose is None or not pose.detected or not pose.keypoints:
        return detections

    wrists = [
        keypoint
        for keypoint in pose.keypoints
        if keypoint.name in {"left_wrist", "right_wrist"} and keypoint.visibility >= 0.4
    ]
    if not wrists:
        return detections

    wrist_y = sum(kp.y for kp in wrists) / len(wrists)
    wrist_xs = [kp.x for kp in wrists]
    wrist_cx = sum(wrist_xs) / len(wrist_xs)
    wrist_span = (max(wrist_xs) - min(wrist_xs)) if len(wrist_xs) > 1 else 0.0

    height, width = frame_shape[:2]
    y_margin = max(48.0, min(height * 0.13, 170.0))
    x_margin = max(width * 0.24, wrist_span * 1.05 + width * 0.08)
    wrist_band_plates = []
    for detection in detections:
        if detection.label != "plate":
            continue
        center_x, center_y = detection.center
        if abs(center_y - wrist_y) <= y_margin and abs(center_x - wrist_cx) <= x_margin:
            wrist_band_plates.append(detection)

    def _hub_is_near_kept_plate(detection: Detection) -> bool:
        center_x, center_y = detection.center
        for plate in wrist_band_plates:
            plate_x, plate_y = plate.center
            plate_size = max(plate.width, plate.height)
            if abs(center_x - plate_x) <= plate_size * 0.85 and abs(center_y - plate_y) <= plate_size * 0.48:
                return True
        return False

    bar_labels = {"plate", "barbell", "bar_hub", "bar_sleeve"}
    kept: list[Detection] = []
    kept_bar = 0
    for detection in detections:
        if detection.label not in bar_labels:
            kept.append(detection)
            continue
        center_x, center_y = detection.center
        near_wrist_band = abs(center_y - wrist_y) <= y_margin and abs(center_x - wrist_cx) <= x_margin
        near_kept_plate = detection.label in {"barbell", "bar_hub", "bar_sleeve"} and _hub_is_near_kept_plate(detection)
        if near_wrist_band or near_kept_plate:
            kept.append(detection)
            kept_bar += 1

    return kept if kept_bar > 0 else detections


def _resolve_object_model_path(
    requested_model: Path | None,
    disable_default_model: bool,
) -> Path | None:
    if requested_model is not None:
        return requested_model
    if disable_default_model:
        return None
    return DEFAULT_BAR_OBJECT_MODEL if DEFAULT_BAR_OBJECT_MODEL.exists() else None


def _object_tracking_enabled(args: argparse.Namespace, object_detector: object) -> bool:
    if args.disable_object_tracking:
        return False
    supports_tracking = isinstance(object_detector, YoloObjectDetector)
    return supports_tracking and (args.enable_tracking or args.object_model is not None or DEFAULT_BAR_OBJECT_MODEL.exists())


def _create_heuristic_object_detector(calibration_diameter_px: float | None) -> MultiColorPlateDetector:
    red_detector = ColorPlateDetector(expected_diameter_pixels=calibration_diameter_px)
    beige_detector = BeigePlateDetector(expected_diameter_pixels=calibration_diameter_px)
    return MultiColorPlateDetector(
        red_detector=red_detector,
        beige_detector=beige_detector,
        expected_diameter_pixels=calibration_diameter_px,
    )


def _plate_heuristic_enabled(
    disable_plate_heuristic: bool,
    enable_plate_heuristic: bool,
    trained_detector: object | None,
) -> bool:
    if disable_plate_heuristic:
        return False
    return enable_plate_heuristic or trained_detector is None


def _announce_object_detector(
    object_model_path: Path | None,
    trained_detector: "YoloObjectDetector | None",
    object_detector: object | None,
) -> None:
    """Print which bar detector is active, so dropping in a trained model is easy to confirm."""
    if trained_detector is not None:
        names = trained_detector.model_names
        classes = ", ".join(names[index] for index in sorted(names)) or "(sin nombres)"
        print(f"Detector de barra: modelo entrenado {object_model_path} (clases: {classes})")
        canonical = {label.lower() for label in names.values()}
        if not ({"plate", "bar_hub", "barbell"} & canonical):
            print(
                "  AVISO: el modelo no expone clases 'plate'/'bar_hub'. "
                "Reentrena con esas clases o el tracking de barra no funcionara."
            )
    elif object_detector is not None:
        print(
            "Detector de barra: heuristica de color "
            "(no hay modelo entrenado en models/powerai_bar_detector.pt)"
        )
    else:
        print("Detector de barra: desactivado")


def _box_area(box: tuple[float, float, float, float] | None) -> float:
    if box is None:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _should_refresh_athlete_lock(
    current_box: tuple[float, float, float, float] | None,
    previous_box: tuple[float, float, float, float] | None,
) -> bool:
    if current_box is None:
        return False
    if previous_box is None:
        return True
    previous_area = _box_area(previous_box)
    current_area = _box_area(current_box)
    if previous_area <= 0.0:
        return True
    return current_area >= previous_area * 0.55


def _legacy_main() -> None:
    args = build_parser().parse_args()
    if args.load_kg is not None and args.load_kg <= 0:
        raise SystemExit("--load-kg debe ser mayor que 0.")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    validation_paths = (
        create_validation_run(args.validation_run_label)
        if args.validation_run_label is not None
        else None
    )
    output_path = (
        validation_paths.videos_dir / f"{validation_paths.run_id}.mp4"
        if validation_paths is not None
        else args.output
    )
    report_json_path = args.report_json
    report_csv_path = args.report_csv
    if validation_paths is not None:
        report_json_path = report_json_path or validation_paths.reports_dir / "analysis_report.json"
        report_csv_path = report_csv_path or validation_paths.reports_dir / "reps.csv"

    input_metadata = read_video_metadata(args.input)
    measured_fps = measure_video_fps(args.input)
    fps_source = "reported"
    if (
        measured_fps is not None
        and abs(measured_fps - input_metadata.fps) > 0.5
    ):
        input_metadata = VideoMetadata(
            width=input_metadata.width,
            height=input_metadata.height,
            fps=measured_fps,
            frame_count=input_metadata.frame_count,
            codec=input_metadata.codec,
        )
        fps_source = "measured"
    logger.debug(
        "FPS resolution: reported=%.3f measured=%s source=%s",
        input_metadata.fps,
        f"{measured_fps:.3f}" if measured_fps is not None else "n/a",
        fps_source,
    )

    object_model_path = _resolve_object_model_path(
        args.object_model,
        args.disable_trained_object_model,
    )

    calibration_diameter_px = args.plate_diameter_px
    auto_calibration_diameter_px: float | None = None
    processing_scale = _processing_scale_from_video(args.input, args.max_resolution, args.output_format)

    # Cargo primero el detector YOLO entrenado para que tambien pueda guiar la autocalibracion.
    # Si falta o falla el modelo, bajo al respaldo configurado en vez de romper el analisis.
    trained_detector: YoloObjectDetector | None = None
    if object_model_path is not None:
        try:
            trained_detector = YoloObjectDetector(
                model_path=object_model_path,
                confidence_threshold=args.object_confidence,
            )
        except Exception as exc:  # noqa: BLE001 - any load error should fall back, not crash
            print(f"No pude cargar el modelo de barra {object_model_path}: {exc}")
            print("  Continuo con la heuristica de color como respaldo.")
            object_model_path = None

    heuristic_enabled = _plate_heuristic_enabled(
        args.disable_plate_heuristic,
        args.enable_plate_heuristic,
        trained_detector,
    )

    if not args.disable_auto_calibration:
        # Use a colour detector with no diameter constraint for the calibration pass so
        # it can measure the real plate before we know its size.
        calibration_fallback = _create_heuristic_object_detector(None) if heuristic_enabled else None
        if trained_detector is not None:
            auto_calibration_diameter_px = estimate_plate_diameter_from_model_boxes(
                args.input,
                trained_detector,
                fps=input_metadata.fps,
            )
        if auto_calibration_diameter_px is None and (
            trained_detector is not None or calibration_fallback is not None
        ):
            auto_calibration_diameter_px = estimate_plate_diameter_from_tracker(
                args.input,
                trained_detector,
                calibration_fallback,
                fps=input_metadata.fps,
            )
        if auto_calibration_diameter_px is None and heuristic_enabled:
            auto_calibration_diameter_px = estimate_plate_diameter_from_video(args.input)

        if auto_calibration_diameter_px is not None:
            calibration_diameter_px = auto_calibration_diameter_px * processing_scale
            logger.debug(
                "Auto-calibration overrode manual diameter: %.1fpx -> %.1fpx (scale %.3f)",
                args.plate_diameter_px,
                calibration_diameter_px,
                processing_scale,
            )

    calibration = create_calibration_from_plate_diameter(calibration_diameter_px)
    pose_estimator = create_pose_estimator(
        backend=args.pose_backend,
        model_path=args.pose_model,
        fps=input_metadata.fps,
    )

    fallback_object_detector = None
    object_detector_name = "none"
    if trained_detector is not None:
        object_detector = trained_detector
        if heuristic_enabled:
            fallback_object_detector = _create_heuristic_object_detector(None)
            object_detector_name = f"yolo+heuristic:{object_model_path}"
        else:
            object_detector_name = f"yolo:{object_model_path}"
    elif heuristic_enabled:
        object_detector = _create_heuristic_object_detector(calibration_diameter_px)
        object_detector_name = "heuristic"
    else:
        object_detector = None

    use_object_tracking = (
        object_detector is not None
        and _object_tracking_enabled(args, object_detector)
    )

    _announce_object_detector(object_model_path, trained_detector, object_detector)

    tracker = PointTracker(frequency_hz=input_metadata.fps, min_cutoff=1.4, beta=0.12)
    bar_anchor_tracker = BarAnchorTracker(fps=input_metadata.fps)
    anchor_tracker = AnchorVelocityTracker(fps=input_metadata.fps, calibration=calibration)
    exercise_profile = get_exercise_profile(args.exercise)
    min_disp_default, max_disp_default = EXERCISE_DISPLACEMENT_DEFAULTS[args.exercise]
    min_rep_displacement_m = (
        args.min_rep_displacement_m if args.min_rep_displacement_m is not None else min_disp_default
    )
    biomechanics_config = BiomechanicsConfig(
        min_rep_displacement_m=min_rep_displacement_m,
        max_reasonable_rep_displacement_m=max_disp_default,
        min_rep_frames=args.min_rep_frames,
        min_gap_between_completed_reps_frames=args.min_rep_gap_frames,
    )
    engine = BiomechanicsEngine(
        fps=input_metadata.fps, config=biomechanics_config, profile=exercise_profile
    )
    renderer = OverlayRenderer(
        OverlayConfig(
            plate_box_style=args.plate_box_style,
            velocity_chart_mode=args.velocity_chart,
            body_velocity_display=args.body_velocity_display,
        )
    )
    measurement_gate = BarMeasurementGate(
        requires_hub=args.measurement_requires_hub,
        hub_confidence_threshold=args.hub_confidence_threshold,
    )
    analysis_segmenter = create_segmenter(
        backend=args.segmentation_backend,
        model_path=args.segmentation_model,
        athlete_lock=args.athlete_lock,
        use_grabcut=args.use_grabcut,
    )
    render_segmenter = create_segmenter(
        backend=args.segmentation_backend,
        model_path=args.segmentation_model,
        athlete_lock=args.athlete_lock,
        use_grabcut=args.use_grabcut,
    )
    technique_monitor = TechniqueMonitor(stable_frames=12, exercise=args.exercise)
    report_builder = RepReportBuilder(
        fps=input_metadata.fps,
        velocity_loss_threshold_percent=args.velocity_loss_threshold,
    )
    manual_load_estimate = _manual_load_estimate(args.load_kg)
    bar_path: list[BarPathPoint] = []
    anchor_velocity_history: dict[str, list[float]] = {
        label: []
        for label, _ in ANCHOR_GROUPS
    }
    anchor_velocity_history["bar"] = []
    velocity_frame_history: list[int] = []
    stats = {
        "frames": 0,
        "tracked_frames": 0,
        "object_frames": 0,
        "hub_reliable_frames": 0,
    }
    tracking_state = {
        "athlete_locked": False,
        "lock_refresh_frame": 0,
        "athlete_box": None,
        "last_anchor_point": None,
        "last_rep_index": 0,
        "last_state": "reposo",
    }
    anchor_diagnostics: list[dict[str, object]] = []
    # Segmentation is the most expensive per-frame model; the silhouette is temporally
    # smoothed, so refreshing it every other frame halves that cost with no visible lag.
    seg_stride = 2

    # --- Two-pass pipeline ---------------------------------------------------------
    # Pass 1 (analyze_frame) runs all pose/bar inference and the rep FSM, caching a
    # FrameRecord per frame. on_analysis_complete then knows the whole clip: total reps,
    # the full (globally smoothed) velocity series and a stable chart scale. Pass 2
    # (render_frame) re-decodes the video and draws the overlay from the cache, so the rep
    # counter can show "done/total" and the chart no longer rescales/jitters frame to frame.
    timeline: dict[int, FrameRecord] = {}
    analysis: dict[str, object] = {
        "total_reps": 0,
        "rep_reports": [],
        "lockout_frames": [],
        "smoothed_history": {},
        "chart_max_abs": None,
    }
    analysis_seg_state: dict[str, object] = {"last_mask": None, "last_seg_frame": None}

    def analyze_frame(frame: Frame, frame_index: int) -> None:
        nonlocal tracker
        if isinstance(pose_estimator, YoloPoseEstimator):
            hub = bar_anchor_tracker.state.point
            if hub is not None:
                pose_estimator.set_preferred_bar_center((hub.x, hub.y))
            else:
                pose_estimator.set_preferred_bar_center(None)

        raw_pose = pose_estimator.estimate(frame)

        if frame_index % seg_stride == 0 or analysis_seg_state["last_mask"] is None:
            athlete_mask_for_pose = analysis_segmenter.segment(frame, raw_pose).mask
        else:
            athlete_mask_for_pose = _warp_mask_with_optical_flow(
                analysis_seg_state["last_seg_frame"],
                frame,
                analysis_seg_state["last_mask"],
            )
        analysis_seg_state["last_mask"] = athlete_mask_for_pose
        analysis_seg_state["last_seg_frame"] = frame.copy()
        pose = refine_pose_with_mask(raw_pose, athlete_mask_for_pose)

        if isinstance(pose_estimator, YoloPoseEstimator) and pose is not None and pose.detected:
            person_box = pose_estimator.get_person_box()
            previous_box = tracking_state["athlete_box"]
            if _should_refresh_athlete_lock(person_box, previous_box):
                if not tracking_state["athlete_locked"]:
                    pose_estimator.lock_to_person(person_box)
                    tracking_state["athlete_locked"] = True
                    tracking_state["lock_refresh_frame"] = frame_index
                    tracking_state["athlete_box"] = person_box
                elif frame_index - tracking_state["lock_refresh_frame"] >= 30:
                    pose_estimator.lock_to_person(person_box)
                    tracking_state["lock_refresh_frame"] = frame_index
                    tracking_state["athlete_box"] = person_box

        if use_object_tracking and isinstance(object_detector, YoloObjectDetector):
            detections = object_detector.detect_with_tracking(frame, tracker_config=args.tracker_config)
        elif object_detector is not None:
            detections = object_detector.detect(frame)
        else:
            detections = []
        if not detections and fallback_object_detector is not None:
            detections = fallback_object_detector.detect(frame)
        # Reject floor/background plates: only keep bar detections near the wrists.
        detections = _filter_detections_near_bar(detections, pose, frame.shape)

        wrist_point = _bar_height_hint_from_pose(pose)
        bar_anchor_tracker.set_pose_hint(wrist_point)
        bar_anchor = bar_anchor_tracker.update(frame, detections)
        load_estimate = manual_load_estimate
        raw_point = measurement_gate.point_for_measurement(bar_anchor)
        sample_point = raw_point
        previous_anchor = tracking_state["last_anchor_point"]
        if raw_point is None:
            if previous_anchor is not None:
                tracker = PointTracker(frequency_hz=input_metadata.fps, min_cutoff=1.4, beta=0.12)
            tracking_state["last_anchor_point"] = None
        elif isinstance(previous_anchor, Point2D):
            jump = ((raw_point.x - previous_anchor.x) ** 2 + (raw_point.y - previous_anchor.y) ** 2) ** 0.5
            max_anchor_jump = max(90.0, calibration_diameter_px * 0.32)
            if jump > max_anchor_jump:
                # Reset the velocity filter and skip this sample on a teleport, but
                # keep the existing trajectory with an explicit break.
                tracker = PointTracker(frequency_hz=input_metadata.fps, min_cutoff=1.4, beta=0.12)
                tracking_state["last_anchor_point"] = raw_point
                sample_point = None
                if bar_path and bar_path[-1] is not None:
                    bar_path.append(None)
        if raw_point is not None:
            tracking_state["last_anchor_point"] = raw_point

        tracked = tracker.update(sample_point, frame_index)
        stats["tracked_frames"] += int(sample_point is not None)
        stats["hub_reliable_frames"] += int(sample_point is not None)

        sample: KinematicSample | None = None
        technique: TechniqueAssessment | None = None
        anchor_velocities = []
        hub_confidence = bar_anchor.hub_confidence
        plate_confidence = bar_anchor.plate_confidence
        tracking_source = bar_anchor.source
        if sample_point is not None and tracked.filtered is not None and tracked.is_valid:
            vertical_position_m = -tracked.filtered.y * calibration.meters_per_pixel
            # IPF gating from pose angles. In v1 strict mode, "pose can't tell" means
            # no valid IPF evidence; bar-only fallback remains available only for debug.
            depth_ok_raw, lockout_ok_raw = _compute_ipf_flags_with_pose_fallback(
                args.exercise,
                pose,
                raw_pose,
            )
            sample = engine.update(
                frame_index,
                vertical_position_m,
                hub_confidence=hub_confidence,
                plate_confidence=plate_confidence,
                tracking_source=tracking_source,
                depth_ok=_strict_ipf_gate(depth_ok_raw, args.strict_ipf_validation),
                lockout_ok=_strict_ipf_gate(lockout_ok_raw, args.strict_ipf_validation),
            )
            tracking_state["last_rep_index"] = sample.rep_index
            tracking_state["last_state"] = sample.state
            technique = technique_monitor.update(
                pose=pose,
                detections=detections,
                calibration=calibration,
                bar_velocity_mps=sample.smoothed_velocity_mps,
                lift_state=sample.state,
                rep_index=sample.rep_index,
            )
            report_builder.add_sample(sample)
            anchor_velocities = anchor_tracker.update(pose)
            path_point = measurement_gate.point_for_measurement(bar_anchor)
            if path_point is not None:
                last_pt = next((point for point in reversed(bar_path) if point is not None), None)
                max_horizontal = max(30.0, calibration_diameter_px * 0.18)
                if last_pt is None or abs(path_point.x - last_pt[0]) <= max_horizontal:
                    bar_path.append((path_point.x, path_point.y))
                else:
                    if bar_path and bar_path[-1] is not None:
                        bar_path.append(None)
                    bar_path.append((path_point.x, path_point.y))
            elif bar_path and bar_path[-1] is not None:
                bar_path.append(None)
            _append_visible_motion_history(
                anchor_velocity_history,
                velocity_frame_history,
                anchor_velocities,
                sample,
                frame_index,
            )
        elif bar_path and bar_path[-1] is not None:
            bar_path.append(None)

        stats["frames"] += 1
        has_plate_anchor = bar_anchor.rect is not None and bar_anchor.source != "lost"
        stats["object_frames"] += int(has_plate_anchor)
        if validation_paths is not None or args.debug_anchor:
            anchor_diagnostics.append(
                {
                    "frame": frame_index,
                    "source": bar_anchor.source,
                    "plate_confidence": round(float(bar_anchor.plate_confidence), 4),
                    "hub_confidence": round(float(bar_anchor.hub_confidence), 4),
                    "measurement_confidence": round(float(bar_anchor.measurement_confidence), 4),
                    "measurable": bool(bar_anchor.measurable and raw_point is not None),
                    "missing_frames": int(bar_anchor.missing_frames),
                    "plate_x": None if bar_anchor.rect is None else round(float(bar_anchor.rect.center.x), 2),
                    "plate_y": None if bar_anchor.rect is None else round(float(bar_anchor.rect.center.y), 2),
                    "hub_x": None if bar_anchor.measurement_point is None else round(float(bar_anchor.measurement_point.x), 2),
                    "hub_y": None if bar_anchor.measurement_point is None else round(float(bar_anchor.measurement_point.y), 2),
                }
            )

        timeline[frame_index] = FrameRecord(
            pose=pose,
            detections=detections,
            sample=sample,
            technique=technique,
            bar_anchor=bar_anchor if (args.show_unmeasured_anchor or bar_anchor.measurable) else None,
            load_estimate=load_estimate,
            anchor_velocities=anchor_velocities,
            bar_path=list(bar_path),
            history_len=len(velocity_frame_history),
        )

    def on_analysis_complete(frame_count: int) -> None:
        engine.finalize(frame_count)
        validated = engine.validated_reps
        analysis["total_reps"] = len(validated)
        analysis["rep_reports"] = [report_builder.build_rep_report(rep) for rep in validated]
        # Counter ticks up when each accepted rep reaches its lockout (IPF), not at liftoff.
        analysis["lockout_frames"] = sorted(rep.lockout_frame for rep in validated)
        smoothed = {key: smooth_series(values) for key, values in anchor_velocity_history.items()}
        analysis["smoothed_history"] = smoothed
        finite_abs = [
            abs(value)
            for series in smoothed.values()
            for value in series
            if np.isfinite(value)
        ]
        analysis["chart_max_abs"] = (
            float(max(0.75, np.percentile(finite_abs, 95))) if finite_abs else 0.75
        )

    seg_state: dict[str, object] = {"last_mask": None, "last_seg_frame": None}

    def render_frame(frame: Frame, frame_index: int) -> Frame:
        record = timeline.get(frame_index)
        if record is None:
            return frame

        if frame_index % seg_stride == 0 or seg_state["last_mask"] is None:
            subject_mask = render_segmenter.segment(frame, record.pose).mask
        else:
            subject_mask = _warp_mask_with_optical_flow(
                seg_state["last_seg_frame"],
                frame,
                seg_state["last_mask"],
            )
        seg_state["last_mask"] = subject_mask
        seg_state["last_seg_frame"] = frame.copy()

        history_len = record.history_len
        smoothed_history = analysis["smoothed_history"]  # type: ignore[assignment]
        sliced_history = {
            key: values[:history_len] for key, values in smoothed_history.items()
        }
        reps_done = bisect_right(analysis["lockout_frames"], frame_index)  # type: ignore[arg-type]

        return renderer.render(
            frame=frame,
            pose=record.pose,
            detections=record.detections,
            sample=record.sample,
            completed_reps=reps_done,
            total_reps=analysis["total_reps"],  # type: ignore[arg-type]
            technique=record.technique,
            bar_path=record.bar_path,
            anchor_velocity_history=sliced_history,
            velocity_frame_history=velocity_frame_history[:history_len],
            chart_max_abs=analysis["chart_max_abs"],  # type: ignore[arg-type]
            anchor_velocities=record.anchor_velocities,
            rep_reports=analysis["rep_reports"],  # type: ignore[arg-type]
            bar_anchor=record.bar_anchor,
            subject_mask=subject_mask,
            load_estimate=record.load_estimate,
            bar_drift_cm=_bar_path_horizontal_drift_cm(
                record.bar_path,
                calibration.meters_per_pixel,
            ),
            debug_anchor=args.debug_anchor,
        )

    try:
        metadata = process_video_two_pass(
            args.input,
            output_path,
            analyze_frame,
            render_frame,
            on_analysis_complete=on_analysis_complete,
            max_frames=args.max_frames,
            target_resolution=args.max_resolution,
            output_mode=args.output_format,
        )
    finally:
        pose_estimator.close()
        analysis_segmenter.close()
        render_segmenter.close()

    mobile_conversion_warning: str | None = None
    if not args.no_mobile_conversion:
        try:
            make_mobile_compatible_in_place(output_path, max_dimension=args.mobile_max_dimension)
        except RuntimeError as exc:
            mobile_conversion_warning = str(exc)
        except Exception as exc:
            mobile_conversion_warning = f"Could not convert output to mobile-compatible MP4: {exc}"

    print("Analisis PowerNZ completado.")
    print(f"Input resolution: {metadata.width}x{metadata.height}")
    print(f"Input FPS: {metadata.fps:.2f} ({fps_source})")
    print(f"Frames processed: {stats['frames']}")
    print(f"Frames tracked: {stats['tracked_frames']}")
    print(f"Frames with barbell/plate: {stats['object_frames']}")
    print(f"Frames with reliable hub: {stats['hub_reliable_frames']}")
    print(f"Completed reps: {len(engine.validated_reps)}")
    print(f"Object detector: {object_detector_name}")
    if isinstance(object_detector, YoloObjectDetector):
        print(f"Object classes: {object_detector.model_names}")
        print(f"Object tracking: {'on' if use_object_tracking else 'off'}")
    print(f"Calibration plate diameter: {calibration_diameter_px:.1f}px")
    if auto_calibration_diameter_px is not None:
        print(f"Auto-calibration observed: {auto_calibration_diameter_px:.1f}px")
        if processing_scale != 1.0:
            print(f"Auto-calibration scale: {processing_scale:.3f}")
    print(f"Output: {output_path}")
    if validation_paths is not None:
        print(f"Validation run: {validation_paths.root}")
    if mobile_conversion_warning is None and not args.no_mobile_conversion:
        print("Output format: mobile-compatible H.264 MP4")
    elif mobile_conversion_warning is not None:
        print(f"Mobile conversion skipped: {mobile_conversion_warning}")

    rep_reports = [
        report_builder.build_rep_report(rep)
        for rep in engine.validated_reps
    ]
    analysis_report = AnalysisReport(
        input_path=str(args.input),
        output_path=str(output_path),
        fps=metadata.fps,
        frame_count=metadata.frame_count,
        tracked_frames=stats["tracked_frames"],
        object_frames=stats["object_frames"],
        completed_reps=len(rep_reports),
        reps=rep_reports,
        hub_reliable_frames_pct=(
            stats["hub_reliable_frames"] / max(1, stats["tracked_frames"]) * 100
        ),
    )

    if report_json_path is not None:
        write_json_report(analysis_report, report_json_path)
        print(f"JSON report: {report_json_path}")

    if report_csv_path is not None:
        write_csv_report(rep_reports, report_csv_path)
        print(f"CSV report: {report_csv_path}")

    if validation_paths is not None and args.save_validation_screenshots:
        screenshots = save_validation_screenshots(
            video_path=output_path,
            screenshots_dir=validation_paths.screenshots_dir,
            label=validation_paths.run_id,
        )
        print(f"Validation screenshots: {len(screenshots)}")

    if validation_paths is not None and anchor_diagnostics:
        diagnostics_path = validation_paths.reports_dir / "anchor_diagnostics.csv"
        with diagnostics_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(anchor_diagnostics[0].keys()))
            writer.writeheader()
            writer.writerows(anchor_diagnostics)
        print(f"Anchor diagnostics: {diagnostics_path}")

    if args.mobile_output is not None:
        try:
            convert_to_mobile_mp4(
                output_path,
                args.mobile_output,
                max_dimension=args.mobile_max_dimension,
            )
            print(f"Mobile MP4: {args.mobile_output}")
        except RuntimeError as exc:
            print(f"Mobile MP4 skipped: {exc}")


def main() -> None:
    args = build_parser().parse_args()
    profile = get_analysis_profile(args.profile)
    velocity_window_seconds = args.velocity_window_seconds or profile.velocity_window_seconds
    if args.load_kg is not None and args.load_kg <= 0:
        raise SystemExit("--load-kg debe ser mayor que 0.")
    if args.calibration_mode == "manual" and (args.plate_diameter_px is None or args.plate_diameter_px <= 0):
        raise SystemExit("La calibración manual requiere --plate-diameter-px mayor que 0.")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    validation_paths = (
        create_validation_run(args.validation_run_label)
        if args.validation_run_label is not None
        else None
    )
    output_path = (
        validation_paths.videos_dir / f"{validation_paths.run_id}.mp4"
        if validation_paths is not None
        else args.output
    )
    report_json_path = args.report_json
    report_csv_path = args.report_csv
    if validation_paths is not None:
        report_json_path = report_json_path or validation_paths.reports_dir / "analysis_report.json"
        report_csv_path = report_csv_path or validation_paths.reports_dir / "reps.csv"

    input_metadata = read_video_metadata(args.input)
    measured_fps = measure_video_fps(args.input)
    fps_source = "reported"
    if measured_fps is not None and abs(measured_fps - input_metadata.fps) > 0.5:
        input_metadata = VideoMetadata(
            width=input_metadata.width,
            height=input_metadata.height,
            fps=measured_fps,
            frame_count=input_metadata.frame_count,
            codec=input_metadata.codec,
        )
        fps_source = "measured"
    output_geometry = resolve_output_geometry(
        input_metadata,
        target_resolution=args.max_resolution,
        output_mode=args.output_format,
    )
    inference_transform = InferenceTransform.from_output_geometry(
        output_geometry,
        profile.inference_max_side,
    )

    object_model_path = _resolve_object_model_path(args.object_model, args.disable_trained_object_model)
    trained_detector: YoloObjectDetector | None = None
    if object_model_path is not None:
        try:
            trained_detector = YoloObjectDetector(
                model_path=object_model_path,
                confidence_threshold=args.object_confidence,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"No pude cargar el modelo de barra {object_model_path}: {exc}")
            object_model_path = None

    heuristic_enabled = _plate_heuristic_enabled(
        args.disable_plate_heuristic,
        args.enable_plate_heuristic,
        trained_detector,
    )
    processing_scale = _processing_scale_from_video(args.input, args.max_resolution, args.output_format)
    auto_calibration_diameter_px: float | None = None
    manual_diameter = args.plate_diameter_px
    auto_requested = args.calibration_mode == "auto" and not args.disable_auto_calibration
    if auto_requested:
        calibration_fallback = _create_heuristic_object_detector(None) if heuristic_enabled else None
        if trained_detector is not None:
            auto_calibration_diameter_px = estimate_plate_diameter_from_model_boxes(
                args.input, trained_detector, fps=input_metadata.fps
            )
        if auto_calibration_diameter_px is None and (trained_detector is not None or calibration_fallback is not None):
            auto_calibration_diameter_px = estimate_plate_diameter_from_tracker(
                args.input,
                trained_detector,
                calibration_fallback,
                fps=input_metadata.fps,
            )
        if auto_calibration_diameter_px is None and heuristic_enabled:
            auto_calibration_diameter_px = estimate_plate_diameter_from_video(args.input)

    if auto_calibration_diameter_px is not None:
        calibration_diameter_px: float | None = auto_calibration_diameter_px * processing_scale
    elif manual_diameter is not None and manual_diameter > 0:
        calibration_diameter_px = manual_diameter
    else:
        calibration_diameter_px = None
    metric_enabled = calibration_diameter_px is not None
    # A non-metric fallback keeps visual tracking available.  It is never used to
    # emit speed, ROM, or repetitions when calibration was not established.
    calibration = create_calibration_from_plate_diameter(calibration_diameter_px or 120.0)

    fallback_object_detector = None
    object_detector_name = "none"
    if trained_detector is not None:
        object_detector = trained_detector
        if heuristic_enabled:
            fallback_object_detector = _create_heuristic_object_detector(None)
            object_detector_name = f"yolo+heuristic:{object_model_path}"
        else:
            object_detector_name = f"yolo:{object_model_path}"
    elif heuristic_enabled:
        object_detector = _create_heuristic_object_detector(calibration_diameter_px)
        object_detector_name = "heuristic"
    else:
        object_detector = None
    use_object_tracking = object_detector is not None and _object_tracking_enabled(args, object_detector)
    _announce_object_detector(object_model_path, trained_detector, object_detector)

    pose_estimator = create_pose_estimator(
        backend=args.pose_backend,
        model_path=args.pose_model,
        fps=input_metadata.fps,
    )
    analysis_segmenter = create_segmenter(
        backend=args.segmentation_backend,
        model_path=args.segmentation_model,
        athlete_lock=args.athlete_lock,
        use_grabcut=args.use_grabcut,
    )
    bar_anchor_tracker = BarAnchorTracker(fps=input_metadata.fps)
    measurement_gate = BarMeasurementGate(
        requires_hub=args.measurement_requires_hub,
        hub_confidence_threshold=args.hub_confidence_threshold,
    )
    exercise_profile = get_exercise_profile(args.exercise)
    min_disp_default, max_disp_default = EXERCISE_DISPLACEMENT_DEFAULTS[args.exercise]
    biomechanics_config = BiomechanicsConfig(
        min_rep_displacement_m=(
            args.min_rep_displacement_m if args.min_rep_displacement_m is not None else min_disp_default
        ),
        max_reasonable_rep_displacement_m=max_disp_default,
        min_rep_frames=args.min_rep_frames,
        min_gap_between_completed_reps_frames=args.min_rep_gap_frames,
    )
    engine = BiomechanicsEngine(fps=input_metadata.fps, config=biomechanics_config, profile=exercise_profile)
    renderer = OverlayRenderer(
        OverlayConfig(
            plate_box_style=args.plate_box_style,
            velocity_chart_mode=args.velocity_chart,
            body_velocity_display=args.body_velocity_display,
            velocity_window_seconds=velocity_window_seconds,
        )
    )
    manual_load_estimate = _manual_load_estimate(args.load_kg)
    timeline: dict[int, FrameRecord] = {}
    analysis: dict[str, object] = {
        "total_reps": 0,
        "rep_reports": [],
        "accepted_reports": [],
        "lockout_frames": [],
        "velocity_history": {"bar": []},
        "velocity_frame_history": [],
        "chart_max_abs": 0.75,
        "decisions": [],
    }
    stats = {"frames": 0, "tracked_frames": 0, "object_frames": 0, "hub_reliable_frames": 0}
    tracking_state: dict[str, object] = {"athlete_locked": False, "lock_refresh_frame": 0, "athlete_box": None}
    anchor_diagnostics: list[dict[str, object]] = []
    seg_state: dict[str, object] = {"last_mask": None, "last_seg_frame": None}

    def _set_pose_bar_hint() -> None:
        if not isinstance(pose_estimator, (YoloPoseEstimator, HybridPoseEstimator)):
            return
        hub = bar_anchor_tracker.state.point
        if hub is None:
            pose_estimator.set_preferred_bar_center(None)
            return
        pose_estimator.set_preferred_bar_center(inference_transform.point_to_inference(hub.x, hub.y))

    with MaskFrameCache() as mask_cache:
        def analyze_frame(frame: Frame, frame_index: int) -> None:
            _set_pose_bar_hint()
            inference_frame = inference_transform.prepare(frame)
            raw_pose_small = pose_estimator.estimate(inference_frame)

            if frame_index % profile.segmentation_stride == 0 or seg_state["last_mask"] is None:
                yolo_mask = analysis_segmenter.segment(inference_frame, raw_pose_small)
                candidates = [yolo_mask]
                if raw_pose_small.segmentation_mask is not None:
                    candidates.append(
                        SegmentationResult(
                            mask=raw_pose_small.segmentation_mask,
                            backend="mediapipe-mask",
                            confidence=0.80,
                        )
                    )
                small_mask = select_subject_mask(candidates, raw_pose_small).mask
                athlete_mask = inference_transform.mask_to_output(small_mask)
            else:
                athlete_mask = _warp_mask_with_optical_flow(
                    seg_state["last_seg_frame"], frame, seg_state["last_mask"]
                )
            seg_state["last_mask"] = athlete_mask
            seg_state["last_seg_frame"] = frame.copy()
            if athlete_mask is not None:
                mask_cache.put(frame_index, athlete_mask)

            raw_pose = inference_transform.pose_to_output(raw_pose_small)
            pose = refine_pose_with_mask(raw_pose, athlete_mask)
            if isinstance(pose_estimator, (YoloPoseEstimator, HybridPoseEstimator)) and pose.detected:
                person_box = pose_estimator.get_person_box()
                if person_box is not None:
                    x1, y1 = inference_transform.point_to_output(person_box[0], person_box[1])
                    x2, y2 = inference_transform.point_to_output(person_box[2], person_box[3])
                    person_box_output = (x1, y1, x2, y2)
                    previous_box = tracking_state["athlete_box"]
                    if _should_refresh_athlete_lock(person_box_output, previous_box if isinstance(previous_box, tuple) else None):
                        if not tracking_state["athlete_locked"] or frame_index - int(tracking_state["lock_refresh_frame"]) >= 30:
                            pose_estimator.lock_to_person(person_box)
                            tracking_state["athlete_locked"] = True
                            tracking_state["lock_refresh_frame"] = frame_index
                            tracking_state["athlete_box"] = person_box_output

            if use_object_tracking and isinstance(object_detector, YoloObjectDetector):
                small_detections = object_detector.detect_with_tracking(inference_frame, tracker_config=args.tracker_config)
            elif object_detector is not None:
                small_detections = object_detector.detect(inference_frame)
            else:
                small_detections = []
            if not small_detections and fallback_object_detector is not None:
                small_detections = fallback_object_detector.detect(inference_frame)
            detections = inference_transform.detections_to_output(small_detections)
            detections = _filter_detections_near_bar(detections, pose, frame.shape)
            bar_anchor_tracker.set_pose_hint(_bar_height_hint_from_pose(pose))
            bar_anchor = bar_anchor_tracker.update(frame, detections)
            raw_point = measurement_gate.point_for_measurement(bar_anchor) if metric_enabled else None
            depth_ok, lockout_ok = _compute_ipf_flags_with_pose_fallback(args.exercise, pose, raw_pose)
            measurement = BarMeasurement(
                frame_index=frame_index,
                time_seconds=frame_index / input_metadata.fps,
                point=raw_point,
                meters_per_pixel=calibration.meters_per_pixel,
                confidence=bar_anchor.measurement_confidence,
                measurable=raw_point is not None and metric_enabled,
            )
            stats["frames"] += 1
            stats["tracked_frames"] += int(measurement.measurable)
            stats["hub_reliable_frames"] += int(measurement.measurable)
            stats["object_frames"] += int(bar_anchor.rect is not None and bar_anchor.source != "lost")
            record = FrameRecord(
                pose=pose,
                raw_pose=raw_pose,
                detections=detections,
                bar_anchor=bar_anchor if (args.show_unmeasured_anchor or bar_anchor.measurable) else None,
                load_estimate=manual_load_estimate,
                measurement=measurement,
                depth_ok=depth_ok,
                lockout_ok=lockout_ok,
                mask_cached=athlete_mask is not None,
            )
            timeline[frame_index] = record
            if validation_paths is not None or args.debug_anchor:
                anchor_diagnostics.append(
                    {
                        "frame": frame_index,
                        "source": bar_anchor.source,
                        "plate_confidence": round(float(bar_anchor.plate_confidence), 4),
                        "hub_confidence": round(float(bar_anchor.hub_confidence), 4),
                        "measurement_confidence": round(float(bar_anchor.measurement_confidence), 4),
                        "measurable": bool(measurement.measurable),
                        "missing_frames": int(bar_anchor.missing_frames),
                    }
                )

        def on_analysis_complete(frame_count: int) -> None:
            measurements = [
                timeline[index].measurement
                for index in range(frame_count)
                if index in timeline and timeline[index].measurement is not None
            ]
            reconstructed = reconstruct_bar_kinematics(
                [measurement for measurement in measurements if measurement is not None],
                fps=input_metadata.fps,
            )
            reconstructed_by_frame: dict[int, ReconstructedBarSample] = {
                sample.frame_index: sample for sample in reconstructed
            }
            anchor_tracker = AnchorVelocityTracker(fps=input_metadata.fps, calibration=calibration)
            technique_monitor = TechniqueMonitor(stable_frames=12, exercise=args.exercise)
            report_builder = RepReportBuilder(
                fps=input_metadata.fps,
                velocity_loss_threshold_percent=args.velocity_loss_threshold,
            )
            histories: dict[str, list[float]] = {label: [] for label, _ in ANCHOR_GROUPS}
            histories["bar"] = []
            path: list[BarPathPoint] = []
            evidence: dict[int, tuple[bool | None, bool | None]] = {}
            max_horizontal = max(30.0, (calibration_diameter_px or 120.0) * 0.18)
            last_path_point: tuple[float, float] | None = None

            for frame_index in range(frame_count):
                record = timeline.get(frame_index)
                reconstructed_sample = reconstructed_by_frame.get(frame_index)
                evidence[frame_index] = (
                    record.depth_ok if record is not None else None,
                    record.lockout_ok if record is not None else None,
                )
                anchor_values = {label: float("nan") for label, _ in ANCHOR_GROUPS}
                if record is not None and reconstructed_sample is not None and reconstructed_sample.valid:
                    anchor = record.bar_anchor
                    sample = engine.update_reconstructed(
                        frame_index=frame_index,
                        position_m=float(reconstructed_sample.position_m),
                        velocity_mps=float(reconstructed_sample.velocity_mps),
                        hub_confidence=anchor.hub_confidence if anchor is not None else 0.0,
                        plate_confidence=anchor.plate_confidence if anchor is not None else 0.0,
                        tracking_source=anchor.source if anchor is not None else "offline",
                    )
                    record.sample = sample
                    record.technique = technique_monitor.update(
                        pose=record.pose,
                        detections=record.detections,
                        calibration=calibration,
                        bar_velocity_mps=sample.smoothed_velocity_mps,
                        lift_state=sample.state,
                        rep_index=sample.rep_index,
                    )
                    record.anchor_velocities = anchor_tracker.update(record.pose)
                    for value in record.anchor_velocities:
                        anchor_values[value.name] = value.velocity_mps
                    report_builder.add_sample(sample)
                    point = reconstructed_sample.point
                    if point is not None and (
                        last_path_point is None or abs(point.x - last_path_point[0]) <= max_horizontal
                    ):
                        path.append((point.x, point.y))
                        last_path_point = (point.x, point.y)
                    elif path and path[-1] is not None:
                        path.append(None)
                        last_path_point = None
                elif path and path[-1] is not None:
                    path.append(None)
                    last_path_point = None

                for label, _ in ANCHOR_GROUPS:
                    histories[label].append(anchor_values[label])
                histories["bar"].append(
                    float(reconstructed_sample.velocity_mps)
                    if reconstructed_sample is not None and reconstructed_sample.valid and reconstructed_sample.velocity_mps is not None
                    else float("nan")
                )
                if record is not None:
                    record.bar_path = list(path)

            engine.finalize(frame_count)
            decisions = decide_rep_validations(
                engine.validate_reps(), evidence, args.exercise, strict=args.strict_ipf_validation
            )
            reports = [
                report_builder.build_rep_report(
                    decision.rep,
                    validation_status=decision.status,
                    validation_reason=decision.reason,
                )
                for decision in decisions
            ]
            accepted_reports = [report for report in reports if report.validation_status == "accepted"]
            analysis["decisions"] = decisions
            analysis["rep_reports"] = reports
            analysis["accepted_reports"] = accepted_reports
            analysis["total_reps"] = len(accepted_reports)
            analysis["lockout_frames"] = sorted(
                decision.rep.lockout_frame for decision in decisions if decision.status == "accepted"
            )
            analysis["velocity_history"] = histories
            analysis["velocity_frame_history"] = list(range(frame_count))
            finite_values = [
                abs(value)
                for values in histories.values()
                for value in values
                if np.isfinite(value)
            ]
            analysis["chart_max_abs"] = float(max(0.75, np.percentile(finite_values, 95))) if finite_values else 0.75

        def render_frame(frame: Frame, frame_index: int) -> Frame:
            record = timeline.get(frame_index)
            if record is None:
                return frame
            histories = analysis["velocity_history"]  # type: ignore[assignment]
            sliced_history = {key: values[: frame_index + 1] for key, values in histories.items()}
            frame_history = analysis["velocity_frame_history"][: frame_index + 1]  # type: ignore[index]
            reps_done = bisect_right(analysis["lockout_frames"], frame_index)  # type: ignore[arg-type]
            return renderer.render(
                frame=frame,
                pose=record.pose,
                detections=record.detections,
                sample=record.sample,
                completed_reps=reps_done,
                total_reps=analysis["total_reps"],  # type: ignore[arg-type]
                technique=record.technique,
                bar_path=record.bar_path,
                anchor_velocity_history=sliced_history,
                velocity_frame_history=frame_history,
                chart_max_abs=analysis["chart_max_abs"],  # type: ignore[arg-type]
                video_fps=input_metadata.fps,
                anchor_velocities=record.anchor_velocities,
                rep_reports=analysis["accepted_reports"],  # type: ignore[arg-type]
                bar_anchor=record.bar_anchor,
                subject_mask=mask_cache.get(frame_index),
                load_estimate=record.load_estimate,
                bar_drift_cm=_bar_path_horizontal_drift_cm(record.bar_path, calibration.meters_per_pixel),
                debug_anchor=args.debug_anchor,
            )

        def on_progress(phase: str, current: int, total: int) -> None:
            print(f"PROGRESS {phase} {current} {total}", flush=True)

        try:
            metadata = process_video_two_pass(
                args.input,
                output_path,
                analyze_frame,
                render_frame,
                on_analysis_complete=on_analysis_complete,
                max_frames=args.max_frames,
                target_resolution=args.max_resolution,
                output_mode=args.output_format,
                progress_callback=on_progress,
            )
        finally:
            pose_estimator.close()
            analysis_segmenter.close()

    mobile_conversion_warning: str | None = None
    if not args.no_mobile_conversion:
        try:
            print("PROGRESS exporting 0 1", flush=True)
            make_mobile_compatible_in_place(output_path, max_dimension=args.mobile_max_dimension)
            print("PROGRESS exporting 1 1", flush=True)
        except RuntimeError as exc:
            mobile_conversion_warning = str(exc)
        except Exception as exc:  # noqa: BLE001
            mobile_conversion_warning = f"Could not convert output to mobile-compatible MP4: {exc}"

    decisions: list[RepDecision] = analysis["decisions"]  # type: ignore[assignment]
    rep_reports = analysis["rep_reports"]  # type: ignore[assignment]
    analysis_report = AnalysisReport(
        input_path=str(args.input),
        output_path=str(output_path),
        fps=metadata.fps,
        frame_count=metadata.frame_count,
        tracked_frames=stats["tracked_frames"],
        object_frames=stats["object_frames"],
        completed_reps=int(analysis["total_reps"]),
        reps=rep_reports,
        hub_reliable_frames_pct=(stats["hub_reliable_frames"] / max(1, stats["frames"]) * 100),
        reviewed_reps=sum(decision.status == "review" for decision in decisions),
        rejected_reps=sum(decision.status == "rejected" for decision in decisions),
    )
    if report_json_path is not None:
        write_json_report(analysis_report, report_json_path)
        print(f"JSON report: {report_json_path}")
    if report_csv_path is not None:
        write_csv_report(rep_reports, report_csv_path)
        print(f"CSV report: {report_csv_path}")
    if validation_paths is not None and args.save_validation_screenshots:
        screenshots = save_validation_screenshots(output_path, validation_paths.screenshots_dir, validation_paths.run_id)
        print(f"Validation screenshots: {len(screenshots)}")
    if validation_paths is not None and anchor_diagnostics:
        diagnostics_path = validation_paths.reports_dir / "anchor_diagnostics.csv"
        with diagnostics_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(anchor_diagnostics[0].keys()))
            writer.writeheader()
            writer.writerows(anchor_diagnostics)

    print("Analisis PowerNZ completado.")
    print(f"Perfil: {profile.name}")
    print(f"Input FPS: {metadata.fps:.2f} ({fps_source})")
    print(f"Frames processed: {stats['frames']}")
    print(f"Completed reps: {analysis['total_reps']}")
    print(f"Reps requiring review: {analysis_report.reviewed_reps}")
    print(f"Object detector: {object_detector_name}")
    if calibration_diameter_px is None:
        print("Calibration: unavailable; speed and rep metrics were not emitted.")
    else:
        print(f"Calibration plate diameter: {calibration_diameter_px:.1f}px")
    print(f"Output: {output_path}")
    if mobile_conversion_warning is not None:
        print(f"Mobile conversion skipped: {mobile_conversion_warning}")


if __name__ == "__main__":
    main()
