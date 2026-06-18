from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from io_video import Frame


DEFAULT_TARGET_CLASSES = frozenset({"plate", "barbell", "bar_hub"})
_LEGACY_CLASS_ALIASES = {"barbell": "bar_hub"}


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    color: str = "unknown"

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class TrackedDetection(Detection):
    track_id: int | None = None

    def to_detection(self) -> Detection:
        return Detection(
            label=self.label,
            confidence=self.confidence,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            color=self.color,
        )


def _is_initial_edge_clipped_plate(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
) -> bool:
    touches_left_or_right = x1 <= 1.0 or x2 >= frame_width - 2.0
    touches_top_or_bottom = y1 <= 1.0 or y2 >= frame_height - 2.0
    width_ratio = (x2 - x1) / max(1.0, frame_width)
    height_ratio = (y2 - y1) / max(1.0, frame_height)
    center_x_ratio = ((x1 + x2) / 2.0) / max(1.0, frame_width)
    center_y_ratio = ((y1 + y2) / 2.0) / max(1.0, frame_height)
    return width_ratio > 0.88 or height_ratio > 0.65 or (
        touches_left_or_right and touches_top_or_bottom
    ) or (
        touches_left_or_right and center_y_ratio > 0.90
    ) or (
        touches_left_or_right and (center_x_ratio < 0.18 or center_x_ratio > 0.82)
    )


class YoloObjectDetector:
    def __init__(
        self,
        model_path: str | Path,
        target_classes: Iterable[str] = DEFAULT_TARGET_CLASSES,
        confidence_threshold: float = 0.25,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required for object detection. "
                "Install it with: pip install ultralytics"
            ) from exc

        self._model = YOLO(str(model_path))
        self._target_classes = self._build_target_class_set(target_classes)
        self._confidence_threshold = confidence_threshold

    @property
    def model_names(self) -> dict[int, str]:
        names = getattr(self._model, "names", {})
        return {int(index): str(label) for index, label in dict(names).items()}

    @staticmethod
    def _build_target_class_set(target_classes: Iterable[str]) -> set[str]:
        resolved: set[str] = set()
        for label in target_classes:
            canonical = label.lower()
            alias = _LEGACY_CLASS_ALIASES.get(canonical)
            resolved.add(canonical)
            if alias is not None:
                resolved.add(alias)
        return resolved

    def detect(self, frame: Frame) -> list[Detection]:
        logger_level = logging.getLogger("ultralytics").level
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        try:
            results = self._model.predict(frame, verbose=False)
        finally:
            logging.getLogger("ultralytics").setLevel(logger_level)
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        names = result.names
        detections: list[Detection] = []

        for box in result.boxes:
            confidence = float(box.conf[0].item())
            if confidence < self._confidence_threshold:
                continue

            class_id = int(box.cls[0].item())
            label = str(names[class_id]).lower()
            alias_label = _LEGACY_CLASS_ALIASES.get(label, label)
            if label not in self._target_classes and alias_label not in self._target_classes:
                continue

            resolved_label = alias_label if alias_label in self._target_classes else label

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            detections.append(
                Detection(
                    label=resolved_label,
                    confidence=confidence,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )

        return detections

    def detect_with_tracking(
        self,
        frame: Frame,
        tracker_config: str | Path | None = None,
    ) -> list[TrackedDetection]:
        logger_level = logging.getLogger("ultralytics").level
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        try:
            results = self._model.track(
                frame,
                persist=True,
                tracker=str(tracker_config or "bytetrack.yaml"),
                verbose=False,
            )
        except Exception:
            logging.getLogger("ultralytics").setLevel(logger_level)
            results = self._model.predict(frame, verbose=False)
        finally:
            logging.getLogger("ultralytics").setLevel(logger_level)
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        names = result.names
        detections: list[TrackedDetection] = []
        track_ids = getattr(result.boxes, "id", None)

        for idx, box in enumerate(result.boxes):
            confidence = float(box.conf[0].item())
            if confidence < self._confidence_threshold:
                continue

            class_id = int(box.cls[0].item())
            label = str(names[class_id]).lower()
            alias_label = _LEGACY_CLASS_ALIASES.get(label, label)
            if label not in self._target_classes and alias_label not in self._target_classes:
                continue

            resolved_label = alias_label if alias_label in self._target_classes else label

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            track_id = (
                int(track_ids[idx].item())
                if track_ids is not None and len(track_ids) > idx
                else None
            )
            detections.append(
                TrackedDetection(
                    label=resolved_label,
                    confidence=confidence,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    track_id=track_id,
                )
            )

        return detections


class ColorPlateDetector:
    def __init__(
        self,
        min_area_pixels: float = 0.0,
        min_circularity: float = 0.18,
        max_missing_frames: int = 12,
        max_center_jump_ratio: float = 0.22,
        expected_diameter_pixels: float | None = None,
        center_smoothing_alpha: float = 0.28,
        center_deadband_pixels: float = 8.0,
    ) -> None:
        if min_area_pixels > 0:
            self._min_area_pixels = min_area_pixels
        elif expected_diameter_pixels is not None and expected_diameter_pixels > 0:
            self._min_area_pixels = 3.14159 * (expected_diameter_pixels * 0.35) ** 2
        else:
            self._min_area_pixels = 8_000.0
        self._min_circularity = min_circularity
        self._max_missing_frames = max_missing_frames
        self._max_center_jump_ratio = max_center_jump_ratio
        self._expected_diameter_pixels = expected_diameter_pixels
        self._center_smoothing_alpha = center_smoothing_alpha
        self._center_deadband_pixels = center_deadband_pixels
        self._last_plate: Detection | None = None
        self._missing_frames = 0

    def detect(self, frame: Frame) -> list[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        red_low = cv2.inRange(hsv, np.array([0, 80, 35]), np.array([14, 255, 255]))
        red_high = cv2.inRange(hsv, np.array([160, 80, 35]), np.array([179, 255, 255]))
        orange = cv2.inRange(hsv, np.array([8, 90, 45]), np.array([28, 255, 255]))
        mask = cv2.bitwise_or(cv2.bitwise_or(red_low, red_high), orange)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_height, frame_width = frame.shape[:2]
        candidates: list[tuple[float, Detection]] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self._min_area_pixels:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self._min_circularity:
                continue

            (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
            diameter = radius * 2.0
            if diameter <= 0:
                continue

            if not self._diameter_is_plausible(diameter):
                continue

            center_x = float(circle_x)
            center_y = float(circle_y)
            lower_body_bias = center_y / frame_height
            if self._last_plate is None and lower_body_bias < 0.28:
                continue

            x1 = max(0.0, center_x - radius)
            y1 = max(0.0, center_y - radius)
            x2 = min(float(frame_width - 1), center_x + radius)
            y2 = min(float(frame_height - 1), center_y + radius)
            if self._last_plate is None and _is_initial_edge_clipped_plate(x1, y1, x2, y2, frame_width, frame_height):
                continue
            confidence = min(0.99, 0.45 + circularity * 0.5)
            plate = Detection(
                label="plate",
                confidence=float(confidence),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                color="red",
            )
            candidates.append((self._score_candidate(plate, area, frame_width, frame_height), plate))

        if not candidates:
            return self._reuse_last_detection()

        plate = self._select_plate(candidates, frame_width, frame_height)
        if plate is None:
            return self._reuse_last_detection()

        plate = self._stabilize_plate_center(plate, frame_width, frame_height)
        self._last_plate = plate
        self._missing_frames = 0

        hub_radius = max(12.0, min(plate.width, plate.height) * 0.12)
        center_x, center_y = plate.center
        barbell = Detection(
            label="barbell",
            confidence=plate.confidence,
            x1=center_x - hub_radius,
            y1=center_y - hub_radius,
            x2=center_x + hub_radius,
            y2=center_y + hub_radius,
            color=plate.color,
        )
        return [plate, barbell]

    def _stabilize_plate_center(
        self,
        plate: Detection,
        frame_width: int,
        frame_height: int,
    ) -> Detection:
        if self._last_plate is None:
            return plate

        last_x, last_y = self._last_plate.center
        current_x, current_y = plate.center
        distance = float(np.hypot(current_x - last_x, current_y - last_y))

        if distance < self._center_deadband_pixels:
            center_x, center_y = last_x, last_y
        else:
            alpha = self._center_smoothing_alpha
            center_x = alpha * current_x + (1.0 - alpha) * last_x
            center_y = alpha * current_y + (1.0 - alpha) * last_y

        radius = max(plate.width, plate.height) / 2.0
        return Detection(
            label=plate.label,
            confidence=plate.confidence,
            x1=max(0.0, center_x - radius),
            y1=max(0.0, center_y - radius),
            x2=min(float(frame_width - 1), center_x + radius),
            y2=min(float(frame_height - 1), center_y + radius),
            color=plate.color,
        )

    def _select_plate(
        self,
        candidates: list[tuple[float, Detection]],
        frame_width: int,
        frame_height: int,
    ) -> Detection | None:
        if self._last_plate is None:
            return max(candidates, key=lambda item: item[0])[1]

        last_center_x, last_center_y = self._last_plate.center
        max_jump_pixels = max(frame_width, frame_height) * self._max_center_jump_ratio
        last_size = max(self._last_plate.width, self._last_plate.height)
        valid_candidates: list[tuple[float, Detection]] = []

        for score, candidate in candidates:
            center_x, center_y = candidate.center
            distance = float(np.hypot(center_x - last_center_x, center_y - last_center_y))
            size = max(candidate.width, candidate.height)
            size_ratio = size / last_size if last_size > 0 else 1.0

            if distance > max_jump_pixels:
                continue

            if size_ratio < 0.55 or size_ratio > 1.65:
                continue

            continuity_bonus = max(0.0, 1.0 - distance / max_jump_pixels)
            valid_candidates.append((score * (1.0 + continuity_bonus), candidate))

        if not valid_candidates:
            return None

        return max(valid_candidates, key=lambda item: item[0])[1]

    def _diameter_is_plausible(self, diameter: float) -> bool:
        if self._expected_diameter_pixels is None:
            return True

        return (
            self._expected_diameter_pixels * 0.62
            <= diameter
            <= self._expected_diameter_pixels * 1.38
        )

    def _score_candidate(
        self,
        plate: Detection,
        area: float,
        frame_width: int,
        frame_height: int,
    ) -> float:
        center_x, center_y = plate.center
        lower_body_bias = center_y / frame_height
        center_bias = 1.0 - min(1.0, abs(center_x - frame_width * 0.5) / (frame_width * 0.5))
        size = max(plate.width, plate.height)
        size_bias = min(2.0, size / max(1.0, frame_width * 0.25))
        return area * (0.7 + lower_body_bias) * (0.8 + 0.2 * center_bias) * (0.8 + size_bias)

    def _reuse_last_detection(self) -> list[Detection]:
        if self._last_plate is None:
            return []

        self._missing_frames += 1
        if self._missing_frames > self._max_missing_frames:
            self._last_plate = None
            return []

        plate = Detection(
            label="plate",
            confidence=max(0.25, self._last_plate.confidence * 0.85),
            x1=self._last_plate.x1,
            y1=self._last_plate.y1,
            x2=self._last_plate.x2,
            y2=self._last_plate.y2,
        )
        self._last_plate = plate
        center_x, center_y = plate.center
        hub_radius = max(12.0, min(plate.width, plate.height) * 0.12)
        barbell = Detection(
            label="barbell",
            confidence=plate.confidence,
            x1=center_x - hub_radius,
            y1=center_y - hub_radius,
            x2=center_x + hub_radius,
            y2=center_y + hub_radius,
            color="red",
        )
        return [plate, barbell]


class BeigePlateDetector:
    def __init__(
        self,
        min_area_pixels: float = 0.0,
        min_circularity: float = 0.18,
        max_missing_frames: int = 12,
        max_center_jump_ratio: float = 0.22,
        expected_diameter_pixels: float | None = None,
        center_smoothing_alpha: float = 0.28,
        center_deadband_pixels: float = 8.0,
    ) -> None:
        if min_area_pixels > 0:
            self._min_area_pixels = min_area_pixels
        elif expected_diameter_pixels is not None and expected_diameter_pixels > 0:
            self._min_area_pixels = 3.14159 * (expected_diameter_pixels * 0.35) ** 2
        else:
            self._min_area_pixels = 8_000.0
        self._min_circularity = min_circularity
        self._max_missing_frames = max_missing_frames
        self._max_center_jump_ratio = max_center_jump_ratio
        self._expected_diameter_pixels = expected_diameter_pixels
        self._center_smoothing_alpha = center_smoothing_alpha
        self._center_deadband_pixels = center_deadband_pixels
        self._last_plate: Detection | None = None
        self._missing_frames = 0

    def detect(self, frame: Frame) -> list[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        beige = cv2.inRange(hsv, np.array([10, 20, 85]), np.array([45, 155, 245]))
        light_yellow = cv2.inRange(hsv, np.array([20, 30, 105]), np.array([60, 150, 250]))
        olive = cv2.inRange(hsv, np.array([35, 25, 55]), np.array([85, 175, 215]))
        mask = cv2.bitwise_or(cv2.bitwise_or(beige, light_yellow), olive)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_height, frame_width = frame.shape[:2]
        candidates: list[tuple[float, Detection]] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self._min_area_pixels:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self._min_circularity:
                continue

            (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
            diameter = radius * 2.0
            if diameter <= 0:
                continue

            if not self._diameter_is_plausible(diameter):
                continue

            center_x = float(circle_x)
            center_y = float(circle_y)
            lower_body_bias = center_y / frame_height
            if self._last_plate is None and lower_body_bias < 0.28:
                continue

            x1 = max(0.0, center_x - radius)
            y1 = max(0.0, center_y - radius)
            x2 = min(float(frame_width - 1), center_x + radius)
            y2 = min(float(frame_height - 1), center_y + radius)
            if self._last_plate is None and _is_initial_edge_clipped_plate(x1, y1, x2, y2, frame_width, frame_height):
                continue
            confidence = min(0.99, 0.40 + circularity * 0.5)
            plate = Detection(
                label="plate",
                confidence=float(confidence),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                color="beige",
            )
            candidates.append((self._score_candidate(plate, area, frame_width, frame_height), plate))

        if not candidates:
            return self._reuse_last_detection()

        plate = self._select_plate(candidates, frame_width, frame_height)
        if plate is None:
            return self._reuse_last_detection()

        plate = self._stabilize_plate_center(plate, frame_width, frame_height)
        self._last_plate = plate
        self._missing_frames = 0

        hub_radius = max(12.0, min(plate.width, plate.height) * 0.12)
        center_x, center_y = plate.center
        barbell = Detection(
            label="barbell",
            confidence=plate.confidence,
            x1=center_x - hub_radius,
            y1=center_y - hub_radius,
            x2=center_x + hub_radius,
            y2=center_y + hub_radius,
            color="beige",
        )

        return [plate, barbell]

    def _stabilize_plate_center(
        self,
        plate: Detection,
        frame_width: int,
        frame_height: int,
    ) -> Detection:
        if self._last_plate is None:
            return plate

        last_x, last_y = self._last_plate.center
        current_x, current_y = plate.center
        distance = float(np.hypot(current_x - last_x, current_y - last_y))

        if distance < self._center_deadband_pixels:
            center_x, center_y = last_x, last_y
        else:
            alpha = self._center_smoothing_alpha
            center_x = alpha * current_x + (1.0 - alpha) * last_x
            center_y = alpha * current_y + (1.0 - alpha) * last_y

        radius = max(plate.width, plate.height) / 2.0
        return Detection(
            label=plate.label,
            confidence=plate.confidence,
            x1=max(0.0, center_x - radius),
            y1=max(0.0, center_y - radius),
            x2=min(float(frame_width - 1), center_x + radius),
            y2=min(float(frame_height - 1), center_y + radius),
            color=plate.color,
        )

    def _select_plate(
        self,
        candidates: list[tuple[float, Detection]],
        frame_width: int,
        frame_height: int,
    ) -> Detection | None:
        if self._last_plate is None:
            return max(candidates, key=lambda item: item[0])[1]

        last_center_x, last_center_y = self._last_plate.center
        max_jump_pixels = max(frame_width, frame_height) * self._max_center_jump_ratio
        last_size = max(self._last_plate.width, self._last_plate.height)
        valid_candidates: list[tuple[float, Detection]] = []

        for score, candidate in candidates:
            center_x, center_y = candidate.center
            distance = float(np.hypot(center_x - last_center_x, center_y - last_center_y))
            size = max(candidate.width, candidate.height)
            size_ratio = size / last_size if last_size > 0 else 1.0

            if distance > max_jump_pixels:
                continue

            if size_ratio < 0.55 or size_ratio > 1.65:
                continue

            continuity_bonus = max(0.0, 1.0 - distance / max_jump_pixels)
            valid_candidates.append((score * (1.0 + continuity_bonus), candidate))

        if not valid_candidates:
            return None

        return max(valid_candidates, key=lambda item: item[0])[1]

    def _diameter_is_plausible(self, diameter: float) -> bool:
        if self._expected_diameter_pixels is None:
            return True

        return (
            self._expected_diameter_pixels * 0.55
            <= diameter
            <= self._expected_diameter_pixels * 1.55
        )

    def _score_candidate(
        self,
        plate: Detection,
        area: float,
        frame_width: int,
        frame_height: int,
    ) -> float:
        center_x, center_y = plate.center
        lower_body_bias = center_y / frame_height
        center_bias = 1.0 - min(1.0, abs(center_x - frame_width * 0.5) / (frame_width * 0.5))
        size = max(plate.width, plate.height)
        size_bias = min(2.0, size / max(1.0, frame_width * 0.25))
        return area * (0.7 + lower_body_bias) * (0.8 + 0.2 * center_bias) * (0.8 + size_bias)

    def _reuse_last_detection(self) -> list[Detection]:
        if self._last_plate is None:
            return []

        self._missing_frames += 1
        if self._missing_frames > self._max_missing_frames:
            self._last_plate = None
            return []

        plate = Detection(
            label="plate",
            confidence=max(0.25, self._last_plate.confidence * 0.85),
            x1=self._last_plate.x1,
            y1=self._last_plate.y1,
            x2=self._last_plate.x2,
            y2=self._last_plate.y2,
            color="beige",
        )
        self._last_plate = plate
        center_x, center_y = plate.center
        hub_radius = max(12.0, min(plate.width, plate.height) * 0.12)
        barbell = Detection(
            label="barbell",
            confidence=plate.confidence,
            x1=center_x - hub_radius,
            y1=center_y - hub_radius,
            x2=center_x + hub_radius,
            y2=center_y + hub_radius,
            color="beige",
        )
        return [plate, barbell]


class MultiColorPlateDetector:
    def __init__(
        self,
        red_detector: "ColorPlateDetector | None" = None,
        beige_detector: BeigePlateDetector | None = None,
        expected_diameter_pixels: float | None = None,
    ) -> None:
        self._red_detector = red_detector or ColorPlateDetector(
            expected_diameter_pixels=expected_diameter_pixels
        )
        self._beige_detector = beige_detector or BeigePlateDetector(
            expected_diameter_pixels=expected_diameter_pixels
        )

    def detect(self, frame: Frame) -> list[Detection]:
        red_detections = self._red_detector.detect(frame)
        beige_detections = self._beige_detector.detect(frame)

        result: list[Detection] = []

        beige_plates = [d for d in beige_detections if d.label == "plate"]
        red_plates = [d for d in red_detections if d.label == "plate"]

        chosen_beige = self._select_one_per_color(beige_plates, color="beige")
        chosen_red = self._select_one_per_color(red_plates, color="red")

        if chosen_beige is not None and chosen_red is not None:
            distance = float(
                np.hypot(
                    chosen_beige.center[0] - chosen_red.center[0],
                    chosen_beige.center[1] - chosen_red.center[1],
                )
            )
            if distance < 60.0:
                kept = chosen_beige if chosen_beige.confidence >= chosen_red.confidence else chosen_red
                result.append(kept)
            else:
                result.append(chosen_beige)
                result.append(chosen_red)
        elif chosen_beige is not None:
            result.append(chosen_beige)
        elif chosen_red is not None:
            result.append(chosen_red)

        selected_centers = [plate.center for plate in result if plate.label == "plate"]
        unknown_plates = [
            plate
            for plate in [*beige_plates, *red_plates]
            if plate.color == "unknown"
            and not _near_any_center(plate.center, selected_centers, max(40.0, max(plate.width, plate.height) * 0.35))
        ]
        for plate in self._select_missing_unknown_plates(result, unknown_plates):
            result.append(plate)

        for detection in beige_detections:
            if detection.label == "barbell":
                result.append(detection)
        for detection in red_detections:
            if detection.label == "barbell" and not any(
                existing.label == "barbell" and existing.color == detection.color
                for existing in result
            ):
                result.append(detection)

        return result

    @staticmethod
    def _select_one_per_color(
        plates: list[Detection],
        color: str,
    ) -> Detection | None:
        tagged = [plate for plate in plates if plate.color == color]
        if not tagged:
            return None
        return max(tagged, key=lambda detection: detection.confidence)

    @staticmethod
    def _select_missing_unknown_plates(
        selected: list[Detection],
        unknown_plates: list[Detection],
    ) -> list[Detection]:
        selected_plates = [plate for plate in selected if plate.label == "plate"]
        if len(selected_plates) >= 2 or not unknown_plates:
            return []

        if not selected_plates:
            return sorted(unknown_plates, key=lambda detection: detection.center[0])[:2]

        selected_x = selected_plates[0].center[0]
        return [
            max(
                unknown_plates,
                key=lambda detection: (
                    abs(detection.center[0] - selected_x),
                    detection.confidence,
                ),
            )
        ]


def _near_any_center(
    center: tuple[float, float],
    centers: list[tuple[float, float]],
    max_distance: float,
) -> bool:
    return any(
        float(np.hypot(center[0] - other[0], center[1] - other[1])) <= max_distance
        for other in centers
    )
