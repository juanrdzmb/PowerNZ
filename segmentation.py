from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from io_video import Frame
from pose import PoseKeypoint, PoseResult


SegmentationBackendName = Literal["auto", "none", "mediapipe-mask", "mediapipe", "yolo-seg", "yolo", "pose-hull"]
AthleteLockMode = Literal["auto", "off"]


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray | None
    backend: str
    confidence: float


def select_subject_mask(
    candidates: list[SegmentationResult],
    pose: PoseResult,
    previous_mask: np.ndarray | None = None,
) -> SegmentationResult:
    """Choose the mask that best belongs to the currently locked athlete.

    The score rewards confident masks that contain the pose landmarks and remain
    continuous with the preceding frame.  It selects rather than blends masks,
    avoiding the visible silhouette lag caused by temporal alpha mixing.
    """
    usable = [candidate for candidate in candidates if candidate.mask is not None and np.any(candidate.mask)]
    if not usable:
        return SegmentationResult(mask=None, backend="none", confidence=0.0)

    pose_points = [
        (int(keypoint.x), int(keypoint.y))
        for keypoint in pose.keypoints
        if keypoint.visibility >= 0.35
    ]
    best: SegmentationResult | None = None
    best_score = -1.0
    for candidate in usable:
        assert candidate.mask is not None
        mask = candidate.mask
        if mask.ndim == 3:
            mask = mask[:, :, 0] if mask.shape[2] == 1 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        height, width = mask.shape[:2]
        contained = sum(
            1
            for x, y in pose_points
            if 0 <= x < width and 0 <= y < height and mask[y, x] > 32
        )
        containment = contained / max(1, len(pose_points))
        continuity = 1.0
        if previous_mask is not None and previous_mask.shape == mask.shape:
            intersection = np.count_nonzero((previous_mask > 32) & (mask > 32))
            union = np.count_nonzero((previous_mask > 32) | (mask > 32))
            continuity = 0.55 + 0.45 * (intersection / union if union else 0.0)
        score = candidate.confidence * (0.45 + 0.55 * containment) * continuity
        if score > best_score:
            best_score = score
            best = SegmentationResult(mask=mask, backend=candidate.backend, confidence=candidate.confidence)
    assert best is not None
    return best


class Segmenter:
    def segment(self, frame: Frame, pose: PoseResult) -> SegmentationResult:
        ...

    def close(self) -> None:
        ...


class NoopSegmenter:
    def segment(self, frame: Frame, pose: PoseResult) -> SegmentationResult:
        return SegmentationResult(mask=None, backend="none", confidence=0.0)

    def close(self) -> None:
        return None


class MediaPipeMaskSegmenter:
    def __init__(self, athlete_lock: AthleteLockMode = "auto") -> None:
        self._fallback = PoseHullSegmenter(athlete_lock=athlete_lock)
        self._previous_mask: np.ndarray | None = None

    def segment(self, frame: Frame, pose: PoseResult) -> SegmentationResult:
        if pose.segmentation_mask is None:
            return self._fallback.segment(frame, pose)

        mask = pose.segmentation_mask
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        mask = cv2.GaussianBlur(mask.astype(np.uint8), (15, 15), 0)
        if self._previous_mask is not None and self._previous_mask.shape == mask.shape:
            mask = cv2.addWeighted(mask, 0.82, self._previous_mask, 0.18, 0).astype(np.uint8)

        self._previous_mask = mask
        return SegmentationResult(mask=mask, backend="mediapipe-mask", confidence=0.80)

    def close(self) -> None:
        self._previous_mask = None
        self._fallback.close()


class PoseHullSegmenter:
    def __init__(
        self,
        visibility_threshold: float = 0.35,
        use_grabcut: bool = False,
        grabcut_stride_frames: int = 30,
        athlete_lock: AthleteLockMode = "auto",
    ) -> None:
        self._visibility_threshold = visibility_threshold
        self._use_grabcut = use_grabcut
        self._grabcut_stride_frames = max(1, grabcut_stride_frames)
        self._athlete_lock = athlete_lock
        self._previous_mask: np.ndarray | None = None
        self._previous_bbox: tuple[int, int, int, int] | None = None
        self._locked_pose_bbox: tuple[int, int, int, int] | None = None
        self._missing_frames = 0
        self._max_mask_hold_frames = 120
        self._frame_index = 0

    def segment(self, frame: Frame, pose: PoseResult) -> SegmentationResult:
        self._frame_index += 1
        visible = self._visible_keypoints(pose)
        if len(visible) < 4:
            self._missing_frames += 1
            mask = self._previous_mask if self._missing_frames <= self._max_mask_hold_frames else None
            return SegmentationResult(mask=mask, backend="pose-hull", confidence=0.20)

        height, width = frame.shape[:2]
        pose_bbox = self._pose_bbox(visible)
        if self._should_reject_pose_bbox(pose_bbox, frame_width=width, frame_height=height):
            self._missing_frames += 1
            mask = self._previous_mask if self._previous_mask is not None else None
            return SegmentationResult(mask=mask, backend="pose-hull", confidence=0.30)

        self._locked_pose_bbox = pose_bbox
        seed_mask = np.zeros((height, width), dtype=np.uint8)

        limb_radius = max(10, int(max(width, height) * 0.010))
        torso_radius = max(16, int(max(width, height) * 0.017))
        joint_radius = max(10, int(max(width, height) * 0.010))

        self._draw_torso(seed_mask, visible, torso_radius)
        for start_name, end_name in _BODY_EDGES:
            self._draw_limb(seed_mask, visible, start_name, end_name, limb_radius)

        for keypoint in visible.values():
            radius = torso_radius if keypoint.name in _TORSO_KEYPOINTS else joint_radius
            cv2.circle(seed_mask, (int(keypoint.x), int(keypoint.y)), radius, 255, -1, cv2.LINE_AA)

        kernel_size = max(5, int(max(width, height) * 0.004))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        should_refine = (
            self._use_grabcut
            and (
                self._previous_mask is None
                or self._frame_index % self._grabcut_stride_frames == 0
            )
        )
        mask = self._refine_with_grabcut(frame, seed_mask, visible) if should_refine else seed_mask
        mask = self._largest_component_overlapping_seed(mask, seed_mask)
        mask = self._stabilize_mask(mask, seed_mask)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)

        if self._previous_mask is not None and self._previous_mask.shape == mask.shape:
            mask = cv2.addWeighted(mask, 0.78, self._previous_mask, 0.22, 0)

        self._previous_mask = mask
        self._previous_bbox = self._mask_bbox(mask)
        self._missing_frames = 0
        return SegmentationResult(mask=mask, backend="pose-hull", confidence=0.55)

    def close(self) -> None:
        self._previous_mask = None
        self._previous_bbox = None
        self._locked_pose_bbox = None
        self._missing_frames = 0
        self._frame_index = 0

    def _visible_keypoints(self, pose: PoseResult) -> dict[str, PoseKeypoint]:
        return {
            keypoint.name: keypoint
            for keypoint in pose.keypoints
            if keypoint.visibility >= self._visibility_threshold
        }

    def _should_reject_pose_bbox(
        self,
        pose_bbox: tuple[int, int, int, int] | None,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        if self._athlete_lock == "off" or pose_bbox is None or self._locked_pose_bbox is None:
            return False

        previous = self._locked_pose_bbox
        previous_area = max(1, (previous[2] - previous[0]) * (previous[3] - previous[1]))
        current_area = max(1, (pose_bbox[2] - pose_bbox[0]) * (pose_bbox[3] - pose_bbox[1]))
        area_ratio = current_area / previous_area
        center_jump = self._bbox_center_distance(pose_bbox, previous)
        max_jump = max(frame_width, frame_height) * 0.18

        if self._missing_frames > self._max_mask_hold_frames:
            return False

        return area_ratio < 0.35 or area_ratio > 2.2 or center_jump > max_jump

    @staticmethod
    def _draw_limb(
        mask: np.ndarray,
        visible: dict[str, PoseKeypoint],
        start_name: str,
        end_name: str,
        radius: int,
    ) -> None:
        start = visible.get(start_name)
        end = visible.get(end_name)
        if start is None or end is None:
            return

        cv2.line(
            mask,
            (int(start.x), int(start.y)),
            (int(end.x), int(end.y)),
            255,
            radius * 2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_torso(mask: np.ndarray, visible: dict[str, PoseKeypoint], radius: int) -> None:
        names = ("left_shoulder", "right_shoulder", "right_hip", "left_hip")
        points = [visible.get(name) for name in names]
        if any(point is None for point in points):
            return

        polygon = np.array(
            [(int(point.x), int(point.y)) for point in points if point is not None],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(mask, polygon, 255, cv2.LINE_AA)
        cv2.polylines(mask, [polygon], True, 255, radius, cv2.LINE_AA)

    @staticmethod
    def _draw_body_envelope(
        mask: np.ndarray,
        visible: dict[str, PoseKeypoint],
        frame_width: int,
        frame_height: int,
    ) -> None:
        left_shoulder = visible.get("left_shoulder")
        right_shoulder = visible.get("right_shoulder")
        left_hip = visible.get("left_hip")
        right_hip = visible.get("right_hip")
        left_ankle = visible.get("left_ankle")
        right_ankle = visible.get("right_ankle")
        left_knee = visible.get("left_knee")
        right_knee = visible.get("right_knee")
        nose = visible.get("nose")

        polygon: list[tuple[int, int]] = []

        if left_ankle is not None and right_ankle is not None and left_knee is not None and right_knee is not None:
            polygon.append((int(left_ankle.x), int(left_ankle.y)))
            polygon.append((int(left_knee.x), int(left_knee.y)))
            polygon.append((int(left_hip.x), int(left_hip.y)))
            polygon.append((int(right_hip.x), int(right_hip.y)))
            polygon.append((int(right_knee.x), int(right_knee.y)))
            polygon.append((int(right_ankle.x), int(right_ankle.y)))
            polygon.append((int(right_ankle.x + (right_ankle.x - right_knee.x) * 0.3), int(right_ankle.y)))
        elif left_ankle is not None:
            polygon.append((int(left_ankle.x), int(left_ankle.y)))
            if left_knee is not None:
                polygon.append((int(left_knee.x), int(left_knee.y)))

        if right_shoulder is not None:
            polygon.append((int(right_shoulder.x), int(right_shoulder.y)))
        if nose is not None:
            polygon.append((int(nose.x), int(nose.y)))
        elif left_shoulder is not None and right_shoulder is not None:
            polygon.append(
                (
                    int((left_shoulder.x + right_shoulder.x) / 2.0),
                    int(min(left_shoulder.y, right_shoulder.y) - 20),
                )
            )
        if left_shoulder is not None:
            polygon.append((int(left_shoulder.x), int(left_shoulder.y)))

        if len(polygon) < 3:
            return

        polygon = [
            (
                min(max(0, x), frame_width - 1),
                min(max(0, y), frame_height - 1),
            )
            for x, y in polygon
        ]
        array = np.asarray(polygon, dtype=np.int32)
        cv2.fillConvexPoly(mask, array, 255, cv2.LINE_AA)

    @staticmethod
    def _refine_with_grabcut(
        frame: Frame,
        seed_mask: np.ndarray,
        visible: dict[str, PoseKeypoint],
        downscale: float = 0.5,
    ) -> np.ndarray:
        if not np.any(seed_mask):
            return seed_mask

        orig_h, orig_w = frame.shape[:2]
        if downscale < 1.0:
            small_w = max(64, int(orig_w * downscale))
            small_h = max(64, int(orig_h * downscale))
            small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
            small_seed = cv2.resize(seed_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        else:
            small_frame = frame
            small_seed = seed_mask

        if not np.any(small_seed):
            return seed_mask

        full_points = np.array([(keypoint.x, keypoint.y) for keypoint in visible.values()], dtype=np.float32)
        points = full_points.copy()
        if downscale < 1.0:
            points = points * downscale
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        pad_x = max(12, int(w * 0.45))
        pad_y = max(12, int(h * 0.28))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(small_frame.shape[1] - 1, x + w + pad_x)
        y2 = min(small_frame.shape[0] - 1, y + h + pad_y)
        if x2 <= x1 or y2 <= y1:
            return seed_mask

        grabcut_mask = np.full(small_seed.shape, cv2.GC_BGD, dtype=np.uint8)
        grabcut_mask[y1:y2, x1:x2] = cv2.GC_PR_BGD

        probable_fg = cv2.dilate(
            small_seed,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
            iterations=1,
        )
        definite_fg = cv2.erode(
            small_seed,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        grabcut_mask[probable_fg > 0] = cv2.GC_PR_FGD
        grabcut_mask[definite_fg > 0] = cv2.GC_FGD

        try:
            bg_model = np.zeros((1, 65), np.float64)
            fg_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(small_frame, grabcut_mask, None, bg_model, fg_model, 1, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return seed_mask

        refined_small = np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)

        if downscale < 1.0:
            refined = cv2.resize(refined_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            probable_fg_full = cv2.dilate(
                seed_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
                iterations=1,
            )
        else:
            refined = refined_small
            probable_fg_full = probable_fg

        overlap = cv2.bitwise_and(refined, probable_fg_full)
        refined_count = np.count_nonzero(refined)
        seed_count = np.count_nonzero(seed_mask)
        if np.count_nonzero(overlap) < seed_count * 0.60:
            return seed_mask

        if refined_count > seed_count * 3.4:
            return seed_mask

        refined = cv2.bitwise_and(refined, _pose_allowed_region(seed_mask.shape, full_points))
        if np.count_nonzero(refined) < seed_count * 0.45:
            return seed_mask

        return cv2.bitwise_or(refined, seed_mask)

    def _stabilize_mask(self, mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
        if self._previous_mask is None or self._previous_mask.shape != mask.shape:
            return mask

        bbox = self._mask_bbox(mask)
        previous_bbox = self._previous_bbox
        if bbox is None or previous_bbox is None:
            return mask

        area = max(1, np.count_nonzero(mask))
        previous_area = max(1, np.count_nonzero(self._previous_mask))
        area_ratio = area / previous_area
        center_jump = self._bbox_center_distance(bbox, previous_bbox)
        max_jump = max(mask.shape) * 0.10

        if area_ratio > 1.65 or area_ratio < 0.45 or center_jump > max_jump:
            return cv2.addWeighted(seed_mask, 0.72, self._previous_mask, 0.28, 0).astype(np.uint8)

        return mask

    @staticmethod
    def _largest_component_overlapping_seed(mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count <= 1:
            return mask

        best_label = 0
        best_score = 0.0
        seed_binary = seed_mask > 0
        for label in range(1, count):
            component = labels == label
            overlap = int(np.count_nonzero(component & seed_binary))
            if overlap <= 0:
                continue

            area = int(stats[label, cv2.CC_STAT_AREA])
            score = overlap + area * 0.08
            if score > best_score:
                best_score = score
                best_label = label

        if best_label == 0:
            return seed_mask

        return np.where(labels == best_label, 255, 0).astype(np.uint8)

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        ys, xs = np.where(mask > 8)
        if len(xs) == 0 or len(ys) == 0:
            return None

        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def _pose_bbox(visible: dict[str, PoseKeypoint]) -> tuple[int, int, int, int] | None:
        if not visible:
            return None

        xs = [keypoint.x for keypoint in visible.values()]
        ys = [keypoint.y for keypoint in visible.values()]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    @staticmethod
    def _bbox_center_distance(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        first_x = (first[0] + first[2]) / 2.0
        first_y = (first[1] + first[3]) / 2.0
        second_x = (second[0] + second[2]) / 2.0
        second_y = (second[1] + second[3]) / 2.0
        return float(np.hypot(first_x - second_x, first_y - second_y))


def _pose_allowed_region(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    height, width = shape
    if points.size == 0:
        return np.zeros(shape, dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    pad_x = max(28, int(w * 0.50))
    pad_y = max(28, int(h * 0.22))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width - 1, x + w + pad_x)
    y2 = min(height - 1, y + h + pad_y)

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


_BODY_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

_TORSO_KEYPOINTS = {
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
}


class YoloSegmentationSegmenter:
    _ATHLETE_CLASSES = {"athlete", "person"}
    _REJECT_CLASSES = {"background_person"}

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        athlete_lock: AthleteLockMode = "auto",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required for YOLO segmentation. Install it with: pip install ultralytics"
            ) from exc

        self._model = YOLO(str(model_path))
        names = {str(label).lower() for label in getattr(self._model, "names", {}).values()}
        if not (names & self._ATHLETE_CLASSES):
            raise ValueError(
                f"Segmentation model {model_path} must expose an athlete/person class; got {sorted(names)}"
            )
        task = str(getattr(self._model, "task", "") or "").lower()
        if task and task != "segment":
            raise ValueError(f"Segmentation model {model_path} must be a segment model; got task={task!r}")
        self._confidence_threshold = confidence_threshold
        self._fallback = PoseHullSegmenter(athlete_lock=athlete_lock)
        self._previous_bbox: tuple[int, int, int, int] | None = None
        self._previous_mask: np.ndarray | None = None
        self._athlete_lock = athlete_lock
        self._athlete_track_id: int | None = None
        self._track_id_missing_count: int = 0

    def segment(self, frame: Frame, pose: PoseResult) -> SegmentationResult:
        try:
            results = self._model.track(frame, persist=True, verbose=False)
        except Exception:
            results = self._model.predict(frame, verbose=False)
        if not results or getattr(results[0], "masks", None) is None:
            return self._fallback.segment(frame, pose)

        result = results[0]
        masks = result.masks
        boxes = result.boxes
        if masks is None or masks.data is None or boxes is None or len(boxes) == 0:
            return self._fallback.segment(frame, pose)

        best_index = None
        best_score = 0.0
        names = result.names
        frame_h, frame_w = frame.shape[:2]
        track_ids = getattr(boxes, "id", None)

        for index, box in enumerate(boxes):
            confidence = float(box.conf[0].item())
            class_id = int(box.cls[0].item())
            label = str(names[class_id]).lower()
            if label in self._REJECT_CLASSES:
                continue
            if label not in self._ATHLETE_CLASSES or confidence < self._confidence_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            bbox = (int(x1), int(y1), int(x2), int(y2))
            area = max(0.0, (x2 - x1) * (y2 - y1))
            continuity = self._continuity_score(bbox, frame_w, frame_h)
            is_athlete = label == "athlete"
            class_bonus = 1.5 if is_athlete else 1.0

            tid = (
                int(track_ids[index].item())
                if track_ids is not None and len(track_ids) > index
                else None
            )
            id_bonus = self._track_id_bonus(tid)
            # Tie the silhouette to the same person the skeleton tracks: strongly
            # prefer the instance whose box contains the visible pose keypoints.
            pose_bonus = self._pose_overlap_score(bbox, pose)
            score = confidence * area * continuity * class_bonus * id_bonus * pose_bonus
            if score > best_score:
                best_score = score
                best_index = index

        if best_index is None:
            self._track_id_missing_count += 1
            if self._track_id_missing_count > 30:
                self._athlete_track_id = None
            return self._fallback.segment(frame, pose)

        self._track_id_missing_count = 0
        if track_ids is not None and len(track_ids) > best_index:
            self._athlete_track_id = int(track_ids[best_index].item())

        mask = masks.data[best_index].cpu().numpy()
        mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
        mask = self._postprocess_mask(mask, pose, frame)
        self._previous_bbox = self._mask_bbox(mask)
        backend_label = "yolo-seg" if self._athlete_track_id is not None else "yolo"
        return SegmentationResult(mask=mask, backend=backend_label, confidence=0.85)

    def _postprocess_mask(
        self,
        soft_mask: np.ndarray,
        pose: PoseResult,
        frame: Frame,
    ) -> np.ndarray:
        """Turn the raw YOLO mask into a clean, body-hugging silhouette: threshold,
        close holes, drop stray blobs, keep the component over the athlete, then
        feather and lightly smooth across time to remove edge flicker."""
        frame_h, frame_w = frame.shape[:2]
        binary = (soft_mask >= 0.45).astype(np.uint8)
        kernel_size = max(3, int(round(min(frame_w, frame_h) * 0.006)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = self._largest_component_for_pose(binary, pose)
        binary = self._refine_mask_edge(binary, frame)
        binary = self._largest_component_for_pose(binary, pose)

        mask = (binary * 255).astype(np.uint8)
        # Mild temporal blend reduces edge shimmer without lagging fast movement.
        if self._previous_mask is not None and self._previous_mask.shape == mask.shape:
            mask = cv2.addWeighted(mask, 0.7, self._previous_mask, 0.3, 0)
        self._previous_mask = mask

        feather = max(3, int(round(min(frame_w, frame_h) * 0.006)))
        if feather % 2 == 0:
            feather += 1
        return cv2.GaussianBlur(mask, (feather, feather), 0)

    @staticmethod
    def _refine_mask_edge(binary: np.ndarray, frame: Frame) -> np.ndarray:
        if not np.any(binary):
            return binary

        height, width = binary.shape[:2]
        original_area = max(1, int(np.count_nonzero(binary)))
        ximgproc = getattr(cv2, "ximgproc", None)
        guided_filter = getattr(ximgproc, "guidedFilter", None) if ximgproc is not None else None

        if guided_filter is not None:
            try:
                guide = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                source = binary.astype(np.float32)
                radius = max(8, int(round(width * 0.01)))
                filtered = guided_filter(guide=guide, src=source, radius=radius, eps=1e-2)
                refined = (filtered >= 0.48).astype(np.uint8)
                refined_area = int(np.count_nonzero(refined))
                if original_area * 0.55 <= refined_area <= original_area * 1.55:
                    return refined
            except (cv2.error, TypeError):
                pass

        kernel_size = max(5, int(round(min(width, height) * 0.011)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        refined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)

    @staticmethod
    def _largest_component_for_pose(binary: np.ndarray, pose: PoseResult) -> np.ndarray:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count <= 2:
            return binary

        points = (
            [(int(kp.x), int(kp.y)) for kp in pose.keypoints if kp.visibility >= 0.3]
            if pose is not None and pose.detected and pose.keypoints
            else []
        )
        height, width = binary.shape
        best_label = 0
        best_score = -1.0
        for label in range(1, count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            overlap = 0
            for px, py in points:
                if 0 <= px < width and 0 <= py < height and labels[py, px] == label:
                    overlap += 1
            score = area + overlap * area  # pose overlap dominates ties
            if score > best_score:
                best_score = score
                best_label = label

        if best_label == 0:
            return binary
        return (labels == best_label).astype(np.uint8)

    def close(self) -> None:
        self._previous_bbox = None
        self._previous_mask = None
        self._athlete_track_id = None
        self._track_id_missing_count = 0

    def _track_id_bonus(self, track_id: int | None) -> float:
        if self._athlete_track_id is None or track_id is None:
            return 1.0
        return 1.3 if track_id == self._athlete_track_id else 0.7

    @staticmethod
    def _pose_overlap_score(bbox: tuple[int, int, int, int], pose: PoseResult) -> float:
        if pose is None or not pose.detected or not pose.keypoints:
            return 1.0

        x1, y1, x2, y2 = bbox
        points = [(kp.x, kp.y) for kp in pose.keypoints if kp.visibility >= 0.3]
        if not points:
            return 1.0

        inside = sum(1 for px, py in points if x1 <= px <= x2 and y1 <= py <= y2)
        fraction = inside / len(points)
        return 0.35 + 1.65 * fraction

    def _continuity_score(self, bbox: tuple[int, int, int, int], frame_width: int, frame_height: int) -> float:
        if self._athlete_lock == "off" or self._previous_bbox is None:
            return 1.0

        jump = PoseHullSegmenter._bbox_center_distance(bbox, self._previous_bbox)
        max_jump = max(frame_width, frame_height) * 0.18
        return max(0.25, 1.0 - min(1.0, jump / max(1.0, max_jump)))

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        return PoseHullSegmenter._mask_bbox(mask)


# Pretrained instance-segmentation models that expose the COCO "person" class,
# in descending quality order. The small ("s") model hugs the body noticeably
# better than nano; nano stays as a lighter fallback. Ultralytics auto-downloads
# these by name on first use; a local copy under models/ takes priority (offline).
DEFAULT_PERSON_SEG_MODELS = ("yolo11s-seg.pt", "yolo11n-seg.pt", "yolov8n-seg.pt")


def _person_seg_model_candidates(model_path: str | Path | None) -> list[str]:
    if model_path is not None:
        return [str(model_path)]

    candidates: list[str] = []
    models_dir = Path(__file__).with_name("models")
    athlete_model = models_dir / "powerai_athlete_seg.pt"
    if athlete_model.exists():
        candidates.append(str(athlete_model))
    for name in DEFAULT_PERSON_SEG_MODELS:
        local = models_dir / name
        if local.exists():
            candidates.append(str(local))
        # Bare name triggers Ultralytics' automatic download when no local copy exists.
        candidates.append(name)
    return candidates


def _build_person_seg_segmenter(
    model_path: str | Path | None,
    athlete_lock: AthleteLockMode,
) -> YoloSegmentationSegmenter | None:
    for candidate in _person_seg_model_candidates(model_path):
        try:
            return YoloSegmentationSegmenter(candidate, athlete_lock=athlete_lock)
        except Exception:
            continue
    return None


def create_segmenter(
    backend: SegmentationBackendName = "auto",
    model_path: str | Path | None = None,
    athlete_lock: AthleteLockMode = "auto",
    use_grabcut: bool = False,
) -> Segmenter:
    if backend == "none":
        return NoopSegmenter()

    if backend == "pose-hull":
        return PoseHullSegmenter(athlete_lock=athlete_lock, use_grabcut=use_grabcut)

    if backend in {"mediapipe", "mediapipe-mask"}:
        return MediaPipeMaskSegmenter(athlete_lock=athlete_lock)

    # "auto", "yolo" y "yolo-seg" usan segmentacion real para que la silueta
    # abrace al atleta. Sin modelo explicito, pruebo un modelo generico de persona
    # y despues el casco de pose si no hay pesos disponibles.
    if backend in {"auto", "yolo", "yolo-seg"}:
        segmenter = _build_person_seg_segmenter(model_path, athlete_lock)
        if segmenter is not None:
            return segmenter
        return PoseHullSegmenter(athlete_lock=athlete_lock, use_grabcut=use_grabcut)

    raise ValueError(f"Unsupported segmentation backend: {backend}")
