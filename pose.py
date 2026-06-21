from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Literal, Protocol

import cv2
import numpy as np

from io_video import Frame


PoseBackendName = Literal["mediapipe", "yolo", "auto"]
DEFAULT_YOLO_POSE_MODEL = Path(__file__).with_name("yolov8n-pose.pt")
DEFAULT_MEDIAPIPE_POSE_MODEL = Path(__file__).with_name("models") / "pose_landmarker_lite.task"


@dataclass(frozen=True)
class PoseKeypoint:
    name: str
    x: float
    y: float
    visibility: float


@dataclass(frozen=True)
class PoseResult:
    keypoints: list[PoseKeypoint]
    backend: PoseBackendName
    detected: bool
    segmentation_mask: np.ndarray | None = None
    source: str = ""
    confidence: float = 0.0
    person_box: tuple[float, float, float, float] | None = None


class PoseEstimator(Protocol):
    def estimate(self, frame: Frame) -> PoseResult:
        ...

    def close(self) -> None:
        ...


def refine_pose_with_mask(
    pose: PoseResult,
    mask: np.ndarray | None,
    mask_threshold: int = 32,
    search_radius_pixels: int | None = None,
) -> PoseResult:
    """Use the athlete mask as a guard rail for pose landmarks.

    Landmarks already inside the athlete silhouette are kept. Landmarks just outside
    the silhouette are snapped to the nearest local mask pixel with slightly reduced
    confidence. Landmarks far from the mask keep their coordinates but their
    visibility is reduced so angle gates can fall back instead of trusting a
    background point.
    """
    if mask is None or pose is None or not pose.detected or not pose.keypoints:
        return pose

    if mask.ndim == 3:
        mask = mask[:, :, 0] if mask.shape[2] == 1 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    binary = mask > mask_threshold
    if not np.any(binary):
        return pose

    height, width = binary.shape[:2]
    radius = search_radius_pixels
    if radius is None:
        radius = max(8, int(round(max(width, height) * 0.014)))

    refined: list[PoseKeypoint] = []
    for keypoint in pose.keypoints:
        x = int(round(keypoint.x))
        y = int(round(keypoint.y))
        if 0 <= x < width and 0 <= y < height and binary[y, x]:
            refined.append(keypoint)
            continue

        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(width, x + radius + 1)
        y2 = min(height, y + radius + 1)
        if x1 < x2 and y1 < y2:
            ys, xs = np.where(binary[y1:y2, x1:x2])
            if len(xs) > 0:
                abs_xs = xs + x1
                abs_ys = ys + y1
                distances = (abs_xs - keypoint.x) ** 2 + (abs_ys - keypoint.y) ** 2
                best = int(np.argmin(distances))
                refined.append(
                    PoseKeypoint(
                        name=keypoint.name,
                        x=float(abs_xs[best]),
                        y=float(abs_ys[best]),
                        visibility=float(keypoint.visibility * 0.85),
                    )
                )
                continue

        refined.append(
            PoseKeypoint(
                name=keypoint.name,
                x=keypoint.x,
                y=keypoint.y,
                visibility=float(keypoint.visibility * 0.25),
            )
        )

    return PoseResult(
        keypoints=refined,
        backend=pose.backend,
        detected=pose.detected,
        segmentation_mask=pose.segmentation_mask,
        source=pose.source,
        confidence=pose.confidence,
        person_box=pose.person_box,
    )


MEDIAPIPE_KEYPOINT_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


YOLO_POSE_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


class MediaPipePoseEstimator:
    def __init__(
        self,
        model_path: str | Path | None = None,
        fps: float = 30.0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mode = "legacy"
        self._timestamp_ms = 0
        self._timestamp_step_ms = 1000.0 / fps if fps > 0 else 1000.0 / 30.0

        try:
            mp_pose = _load_mediapipe_pose_module()
            self._pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except ImportError as legacy_error:
            self._mode = "tasks"
            self._pose = _create_mediapipe_tasks_landmarker(
                model_path=model_path,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                legacy_error=legacy_error,
            )

    def estimate(self, frame: Frame) -> PoseResult:
        if self._mode == "tasks":
            return self._estimate_with_tasks(frame)

        return self._estimate_with_legacy(frame)

    def _estimate_with_legacy(self, frame: Frame) -> PoseResult:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb_frame)

        if result.pose_landmarks is None:
            return PoseResult(keypoints=[], backend="mediapipe", detected=False)

        height, width = frame.shape[:2]
        keypoints = [
            PoseKeypoint(
                name=MEDIAPIPE_KEYPOINT_NAMES[index],
                x=float(landmark.x * width),
                y=float(landmark.y * height),
                visibility=float(landmark.visibility),
            )
            for index, landmark in enumerate(result.pose_landmarks.landmark)
        ]

        segmentation_mask = (
            (result.segmentation_mask * 255.0).clip(0, 255).astype(np.uint8)
            if result.segmentation_mask is not None
            else None
        )

        return PoseResult(
            keypoints=keypoints,
            backend="mediapipe",
            detected=True,
            segmentation_mask=segmentation_mask,
            source="mediapipe-legacy",
            confidence=_pose_confidence(keypoints),
        )

    def _estimate_with_tasks(self, frame: Frame) -> PoseResult:
        import mediapipe as mp

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._pose.detect_for_video(mp_image, self._timestamp_ms)
        self._timestamp_ms = int(round(self._timestamp_ms + self._timestamp_step_ms))

        if not result.pose_landmarks:
            return PoseResult(keypoints=[], backend="mediapipe", detected=False)

        height, width = frame.shape[:2]
        landmarks = result.pose_landmarks[0]
        keypoints = [
            PoseKeypoint(
                name=MEDIAPIPE_KEYPOINT_NAMES[index],
                x=float(landmark.x * width),
                y=float(landmark.y * height),
                visibility=float(getattr(landmark, "visibility", 1.0)),
            )
            for index, landmark in enumerate(landmarks)
            if index < len(MEDIAPIPE_KEYPOINT_NAMES)
        ]

        return PoseResult(
            keypoints=keypoints,
            backend="mediapipe",
            detected=True,
            segmentation_mask=_mediapipe_tasks_segmentation_mask(result),
            source="mediapipe-tasks",
            confidence=_pose_confidence(keypoints),
        )

    def close(self) -> None:
        self._pose.close()


class YoloPoseEstimator:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_YOLO_POSE_MODEL,
        max_center_jump_pixels: float = 250.0,
        lock_max_jump_pixels: float = 350.0,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required for the yolo pose backend. "
                "Install it with: pip install ultralytics"
            ) from exc

        self._model = YOLO(str(model_path))
        self._max_center_jump_pixels = max_center_jump_pixels
        self._lock_max_jump_pixels = lock_max_jump_pixels
        self._last_center: tuple[float, float] | None = None
        self._last_person_box: tuple[float, float, float, float] | None = None
        self._preferred_bar_center: tuple[float, float] | None = None
        self._locked_person_box: tuple[float, float, float, float] | None = None

    def set_preferred_bar_center(self, center: tuple[float, float] | None) -> None:
        self._preferred_bar_center = center

    def lock_to_person(self, box: tuple[float, float, float, float] | None) -> None:
        self._locked_person_box = box

    def get_person_box(self) -> tuple[float, float, float, float] | None:
        return self._last_person_box

    def estimate(self, frame: Frame) -> PoseResult:
        logger_level = logging.getLogger("ultralytics").level
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        try:
            results = self._model.predict(frame, verbose=False)
        finally:
            logging.getLogger("ultralytics").setLevel(logger_level)
        if not results or results[0].keypoints is None:
            return PoseResult(keypoints=[], backend="yolo", detected=False)

        keypoints_xy = results[0].keypoints.xy
        keypoints_conf = results[0].keypoints.conf
        if keypoints_xy is None or len(keypoints_xy) == 0:
            return PoseResult(keypoints=[], backend="yolo", detected=False)

        person_index = _select_pose_person_index(
            results[0],
            previous_center=self._last_center,
            max_jump_pixels=self._max_center_jump_pixels,
            preferred_center=self._preferred_bar_center,
            locked_box=self._locked_person_box,
            lock_max_jump_pixels=self._lock_max_jump_pixels,
        )
        xy = keypoints_xy[person_index].cpu().numpy()
        conf = keypoints_conf[person_index].cpu().numpy() if keypoints_conf is not None else np.ones(len(xy))

        keypoints = [
            PoseKeypoint(
                name=YOLO_POSE_KEYPOINT_NAMES[index],
                x=float(point[0]),
                y=float(point[1]),
                visibility=float(conf[index]),
            )
            for index, point in enumerate(xy)
            if index < len(YOLO_POSE_KEYPOINT_NAMES)
        ]

        self._last_center = _person_center(results[0], person_index)
        self._last_person_box = _person_box(results[0], person_index)

        person_box = _person_box(results[0], person_index)
        return PoseResult(
            keypoints=keypoints,
            backend="yolo",
            detected=True,
            source="yolo",
            confidence=_pose_confidence(keypoints),
            person_box=person_box,
        )

    def close(self) -> None:
        return None


class HybridPoseEstimator:
    """Use YOLO to preserve athlete identity and MediaPipe for denser pose/masks.

    MediaPipe is deliberately accepted only when enough of its landmarks land in
    the YOLO-selected athlete box.  This prevents a nearby spectator from taking
    over the silhouette in multi-person clips while retaining MediaPipe's 33
    landmarks and segmentation when it agrees with the locked athlete.
    """

    def __init__(
        self,
        yolo_model_path: str | Path = DEFAULT_YOLO_POSE_MODEL,
        mediapipe_model_path: str | Path | None = None,
        fps: float = 30.0,
    ) -> None:
        self._yolo = YoloPoseEstimator(model_path=yolo_model_path)
        self._mediapipe: MediaPipePoseEstimator | None = None
        model_path = mediapipe_model_path or DEFAULT_MEDIAPIPE_POSE_MODEL
        try:
            self._mediapipe = MediaPipePoseEstimator(model_path=model_path, fps=fps)
        except Exception as exc:  # A robust auto mode must still work without the optional task model.
            logging.getLogger(__name__).warning("MediaPipe unavailable; auto pose falls back to YOLO: %s", exc)

    def set_preferred_bar_center(self, center: tuple[float, float] | None) -> None:
        self._yolo.set_preferred_bar_center(center)

    def lock_to_person(self, box: tuple[float, float, float, float] | None) -> None:
        self._yolo.lock_to_person(box)

    def get_person_box(self) -> tuple[float, float, float, float] | None:
        return self._yolo.get_person_box()

    def estimate(self, frame: Frame) -> PoseResult:
        yolo_pose = self._yolo.estimate(frame)
        if self._mediapipe is None:
            return yolo_pose

        try:
            mediapipe_pose = self._mediapipe.estimate(frame)
        except Exception as exc:  # Do not make a long export fail for one MediaPipe frame.
            logging.getLogger(__name__).warning("MediaPipe frame failed; keeping YOLO pose: %s", exc)
            return yolo_pose

        return fuse_pose_results(yolo_pose, mediapipe_pose)

    def close(self) -> None:
        self._yolo.close()
        if self._mediapipe is not None:
            self._mediapipe.close()


def fuse_pose_results(yolo_pose: PoseResult, mediapipe_pose: PoseResult) -> PoseResult:
    """Return the safest fused pose without mixing athletes between detectors."""
    if not yolo_pose.detected:
        return mediapipe_pose
    if not mediapipe_pose.detected:
        return yolo_pose

    person_box = yolo_pose.person_box
    if person_box is not None and not _pose_matches_person_box(mediapipe_pose, person_box):
        return yolo_pose

    by_name = {keypoint.name: keypoint for keypoint in yolo_pose.keypoints}
    for keypoint in mediapipe_pose.keypoints:
        previous = by_name.get(keypoint.name)
        if previous is None or keypoint.visibility >= previous.visibility * 0.88:
            by_name[keypoint.name] = keypoint

    keypoints = list(by_name.values())
    return PoseResult(
        keypoints=keypoints,
        backend="auto",
        detected=True,
        segmentation_mask=mediapipe_pose.segmentation_mask,
        source="hybrid",
        confidence=max(yolo_pose.confidence, mediapipe_pose.confidence),
        person_box=person_box,
    )


def _pose_matches_person_box(
    pose: PoseResult,
    person_box: tuple[float, float, float, float],
) -> bool:
    x1, y1, x2, y2 = person_box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = width * 0.18
    pad_y = height * 0.18
    visible = [keypoint for keypoint in pose.keypoints if keypoint.visibility >= 0.35]
    if len(visible) < 5:
        return False
    inside = sum(
        x1 - pad_x <= keypoint.x <= x2 + pad_x and y1 - pad_y <= keypoint.y <= y2 + pad_y
        for keypoint in visible
    )
    return inside / len(visible) >= 0.70


def _pose_confidence(keypoints: list[PoseKeypoint]) -> float:
    visible = [keypoint.visibility for keypoint in keypoints if keypoint.visibility > 0]
    return float(sum(visible) / len(visible)) if visible else 0.0


def _largest_pose_person_index(result: object) -> int:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        return 0

    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return int(np.argmax(areas))


def _person_center(result: object, person_index: int) -> tuple[float, float] | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) <= person_index:
        return None

    xyxy = boxes.xyxy[person_index].cpu().numpy()
    if len(xyxy) < 4:
        return None

    center_x = float((xyxy[0] + xyxy[2]) / 2.0)
    center_y = float((xyxy[1] + xyxy[3]) / 2.0)
    return center_x, center_y


def _person_box(result: object, person_index: int) -> tuple[float, float, float, float] | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) <= person_index:
        return None

    xyxy = boxes.xyxy[person_index].cpu().numpy()
    if len(xyxy) < 4:
        return None

    return float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])


def _bar_owner_index(
    xyxy: np.ndarray,
    centers_x: np.ndarray,
    centers_y: np.ndarray,
    previous_center: tuple[float, float] | None,
    preferred_center: tuple[float, float],
) -> int | None:
    """Index of the person whose (padded) box encloses the bar hub, or None.

    When several enclose it, prefer the one closest to the previously tracked
    athlete for stability, else the one whose center is nearest the hub.
    """
    px, py = preferred_center
    pad_x = (xyxy[:, 2] - xyxy[:, 0]) * 0.30
    pad_y = (xyxy[:, 3] - xyxy[:, 1]) * 0.30
    contains = (
        (xyxy[:, 0] - pad_x <= px)
        & (px <= xyxy[:, 2] + pad_x)
        & (xyxy[:, 1] - pad_y <= py)
        & (py <= xyxy[:, 3] + pad_y)
    )
    owner_indices = np.where(contains)[0]
    if owner_indices.size == 0:
        return None
    if owner_indices.size == 1:
        return int(owner_indices[0])

    if previous_center is not None:
        prev_x, prev_y = previous_center
        distances = np.hypot(centers_x[owner_indices] - prev_x, centers_y[owner_indices] - prev_y)
    else:
        distances = np.hypot(centers_x[owner_indices] - px, centers_y[owner_indices] - py)
    return int(owner_indices[int(np.argmin(distances))])


def _select_pose_person_index(
    result: object,
    previous_center: tuple[float, float] | None,
    max_jump_pixels: float,
    preferred_center: tuple[float, float] | None = None,
    locked_box: tuple[float, float, float, float] | None = None,
    lock_max_jump_pixels: float = 350.0,
) -> int:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        return 0

    xyxy = boxes.xyxy.cpu().numpy()
    if len(xyxy) == 1:
        return 0

    centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    centers_y = (xyxy[:, 1] + xyxy[:, 3]) / 2.0

    # The athlete is the person holding the bar: strongly prefer whoever's box
    # encloses the bar hub. This rejects bystanders standing nearby (a common
    # failure where the upright bystander wins on size/continuity).
    if preferred_center is not None:
        owner_index = _bar_owner_index(xyxy, centers_x, centers_y, previous_center, preferred_center)
        if owner_index is not None:
            return owner_index

    if locked_box is not None:
        lx1, ly1, lx2, ly2 = locked_box
        locked_cx = (lx1 + lx2) / 2.0
        locked_cy = (ly1 + ly2) / 2.0
        locked_half_w = (lx2 - lx1) / 2.0
        locked_half_h = (ly2 - ly1) / 2.0
        box_centers_x = centers_x
        box_centers_y = centers_y
        box_half_w = (xyxy[:, 2] - xyxy[:, 0]) / 2.0
        box_half_h = (xyxy[:, 3] - xyxy[:, 1]) / 2.0
        center_distance = np.hypot(box_centers_x - locked_cx, box_centers_y - locked_cy)
        size_ratio_w = box_half_w / max(1.0, locked_half_w)
        size_ratio_h = box_half_h / max(1.0, locked_half_h)
        size_score = np.maximum(size_ratio_w, size_ratio_h)
        in_range = center_distance <= lock_max_jump_pixels
        size_acceptable = (size_score >= 0.4) & (size_score <= 2.5)
        if in_range.any() and size_acceptable.any():
            best_match_score = -center_distance / max(1.0, lock_max_jump_pixels)
            best_match_score = np.where(in_range & size_acceptable, best_match_score, -1.0)
            best_index = int(np.argmax(best_match_score))
            if best_match_score[best_index] > -1.0:
                return best_index

    if previous_center is not None:
        prev_x, prev_y = previous_center
        distances = np.hypot(centers_x - prev_x, centers_y - prev_y)
        closest_index = int(np.argmin(distances))

        if distances[closest_index] <= max_jump_pixels:
            return closest_index

    if preferred_center is not None:
        pref_x, pref_y = preferred_center
        dist_to_bar = np.hypot(centers_x - pref_x, centers_y - pref_y)
        closest_to_bar = int(np.argmin(dist_to_bar))
        max_bar_distance = max_jump_pixels * 2.5
        if dist_to_bar[closest_to_bar] <= max_bar_distance:
            return closest_to_bar

    return _largest_pose_person_index(result)


def _load_mediapipe_pose_module() -> object:
    import_errors: list[str] = []

    for module_name in ("mediapipe.solutions.pose", "mediapipe.python.solutions.pose"):
        try:
            return import_module(module_name)
        except ImportError as exc:
            import_errors.append(f"{module_name}: {exc}")

    try:
        import mediapipe as mp

        solutions = getattr(mp, "solutions", None)
        pose = getattr(solutions, "pose", None) if solutions is not None else None
        if pose is not None:
            return pose
    except ImportError as exc:
        import_errors.append(f"mediapipe: {exc}")

    details = "\n".join(import_errors)
    raise ImportError(
        "MediaPipe is installed, but this Python environment does not expose the "
        "legacy Pose API used by Phase 2. Try reinstalling a compatible build with:\n"
        "python -m pip uninstall -y mediapipe\n"
        "python -m pip install mediapipe\n\n"
        f"Import attempts:\n{details}"
    )


def _create_mediapipe_tasks_landmarker(
    model_path: str | Path | None,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    legacy_error: ImportError,
) -> object:
    if model_path is None:
        raise ImportError(
            "This MediaPipe install does not expose the legacy Pose API. "
            "Use the modern MediaPipe Tasks API by passing a pose landmarker model:\n"
            "python test_phase_2.py --input video_prueba.mp4 "
            "--output outputs/phase_2_test.mp4 "
            "--pose-model models/pose_landmarker_lite.task\n\n"
            "Download example model:\n"
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task\n\n"
            f"Original legacy import error:\n{legacy_error}"
        )

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"MediaPipe Tasks pose model not found: {model_path}\n"
            "Download pose_landmarker_lite.task and pass its path with --pose-model."
        )

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError("MediaPipe is required. Install it with: pip install mediapipe") from exc

    try:
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=True,
        )
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)
    except AttributeError as exc:
        raise ImportError(
            "This MediaPipe install exposes neither legacy Pose nor MediaPipe Tasks Pose. "
            "Try reinstalling MediaPipe, or use YOLO pose with: "
            "python test_phase_2.py --input video_prueba.mp4 --pose-backend yolo"
        ) from exc


def _mediapipe_tasks_segmentation_mask(result: object) -> np.ndarray | None:
    masks = getattr(result, "segmentation_masks", None)
    if not masks:
        return None

    mask = masks[0]
    if hasattr(mask, "numpy_view"):
        array = mask.numpy_view()
    else:
        array = np.asarray(mask)

    return (array * 255.0).clip(0, 255).astype(np.uint8)


def create_pose_estimator(
    backend: PoseBackendName = "auto",
    model_path: str | Path | None = None,
    fps: float = 30.0,
) -> PoseEstimator:
    if backend == "auto":
        return HybridPoseEstimator(
            yolo_model_path=DEFAULT_YOLO_POSE_MODEL,
            mediapipe_model_path=model_path,
            fps=fps,
        )

    if backend == "mediapipe":
        return MediaPipePoseEstimator(model_path=model_path, fps=fps)

    if backend == "yolo":
        return YoloPoseEstimator(model_path=model_path or DEFAULT_YOLO_POSE_MODEL)

    raise ValueError(f"Unsupported pose backend: {backend}")
