from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from detect_objects import Detection, TrackedDetection
from io_video import Frame
from track import OneEuroFilter, Point2D

if TYPE_CHECKING:
    from pose import PoseKeypoint, PoseResult


AnchorSource = Literal[
    "detection",
    "prediction",
    "optical_flow",
    "template",
    "hold",
    "pose_seed",
    "wrist",
    "lost",
]


@dataclass(frozen=True)
class AnchorRect:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Point2D:
        return Point2D((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class BarAnchorState:
    point: Point2D | None
    rect: AnchorRect | None
    confidence: float
    missing_frames: int
    locked: bool
    source: AnchorSource
    hub_detected: bool = False
    track_id: int | None = None
    hub_rect: AnchorRect | None = None
    plate_confidence: float = 0.0
    hub_confidence: float = 0.0
    display_rect: AnchorRect | None = None
    measurement_point: Point2D | None = None
    measurement_confidence: float = 0.0
    measurable: bool = False


@dataclass(frozen=True)
class AnchorObservation:
    point: Point2D
    rect: AnchorRect
    confidence: float
    label: str
    hub_rect: AnchorRect | None = None
    plate_confidence: float = 0.0
    hub_confidence: float = 0.0
    track_id: int | None = None
    display_rect: AnchorRect | None = None


@dataclass(frozen=True)
class BarAnchorConfig:
    max_missing_frames: int = 900
    reacquire_after_frames: int = 14
    max_detection_jump_ratio: float = 0.42
    rect_padding_ratio: float = 0.015
    refined_plate_rect_scale: float = 0.72
    min_rect_size_pixels: float = 150.0
    plate_rect_fallback_scale: float = 0.86
    center_deadband_pixels: float = 4.0
    detection_confidence_floor: float = 0.30
    optical_flow_confidence_decay: float = 0.92
    template_confidence_decay: float = 0.88
    hold_confidence_decay: float = 0.94
    prediction_confidence_decay: float = 0.90
    kalman_prediction_frames: int = 8
    kalman_process_noise: float = 0.035
    kalman_measurement_noise: float = 7.5
    template_match_threshold: float = 0.36
    rect_center_deadband_pixels: float = 3.0
    rect_size_smoothing_alpha: float = 0.12
    locked_switch_distance_ratio: float = 0.28
    vertical_reacquire_x_ratio: float = 0.36
    vertical_reacquire_y_ratio: float = 1.55
    vertical_reacquire_pose_y_ratio: float = 0.55
    hub_confidence_threshold: float = 0.35
    plate_confidence_threshold: float = 0.30
    hub_confidence_multiplier: float = 1.0
    hold_confidence_multiplier: float = 0.35


class BarMeasurementGate:
    """Strictly separates a visible plate lock from a metric bar/hub sample."""

    def __init__(self, requires_hub: bool = True, hub_confidence_threshold: float = 0.35) -> None:
        self._requires_hub = requires_hub
        self._hub_confidence_threshold = hub_confidence_threshold

    def point_for_measurement(self, state: BarAnchorState) -> Point2D | None:
        if state.point is None or state.source in {"lost", "pose_seed", "wrist"}:
            return None

        if self._requires_hub:
            if not state.measurable:
                return None
            if state.measurement_point is None:
                return None
            if state.measurement_confidence < self._hub_confidence_threshold:
                return None
            return state.measurement_point

        confidence = state.measurement_confidence if state.measurement_point is not None else state.confidence
        if confidence < 0.22:
            return None
        return state.measurement_point or state.point


class _BarAnchorKalmanFilter:
    def __init__(self, fps: float, config: BarAnchorConfig) -> None:
        self._dt = 1.0 / fps
        self._filter = cv2.KalmanFilter(4, 2)
        self._filter.transitionMatrix = np.array(
            [
                [1.0, 0.0, self._dt, 0.0],
                [0.0, 1.0, 0.0, self._dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self._filter.measurementMatrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        process_noise = np.diag([1.0, 1.0, 18.0, 18.0]).astype(np.float32)
        self._filter.processNoiseCov = process_noise * config.kalman_process_noise
        self._filter.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * config.kalman_measurement_noise
        )
        self._filter.errorCovPost = np.eye(4, dtype=np.float32)
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def reset(self) -> None:
        self._filter.statePre = np.zeros((4, 1), dtype=np.float32)
        self._filter.statePost = np.zeros((4, 1), dtype=np.float32)
        self._filter.errorCovPost = np.eye(4, dtype=np.float32)
        self._initialized = False

    def predict(self) -> Point2D | None:
        if not self._initialized:
            return None

        prediction = self._filter.predict()
        return Point2D(float(prediction[0][0]), float(prediction[1][0]))

    def correct(self, point: Point2D) -> Point2D:
        measurement = np.array([[point.x], [point.y]], dtype=np.float32)
        if not self._initialized:
            self._filter.statePre = np.array(
                [[point.x], [point.y], [0.0], [0.0]],
                dtype=np.float32,
            )
            self._filter.statePost = self._filter.statePre.copy()
            self._initialized = True
            return point

        corrected = self._filter.correct(measurement)
        return Point2D(float(corrected[0][0]), float(corrected[1][0]))


class BarAnchorTracker:
    def __init__(self, fps: float, config: BarAnchorConfig = BarAnchorConfig()) -> None:
        self._fps = fps
        self._config = config
        # Point filters: responsive enough to follow the bar through a fast pull
        # without lagging, while the beta term still suppresses jitter at rest.
        self._x_filter = OneEuroFilter(frequency_hz=fps, min_cutoff=1.1, beta=0.05)
        self._y_filter = OneEuroFilter(frequency_hz=fps, min_cutoff=1.1, beta=0.05)
        # Rect filters: track the plate tightly but stay a touch smoother than the point.
        self._rect_x_filter = OneEuroFilter(frequency_hz=fps, min_cutoff=0.8, beta=0.015)
        self._rect_y_filter = OneEuroFilter(frequency_hz=fps, min_cutoff=0.8, beta=0.015)
        self._last_state = BarAnchorState(None, None, 0.0, 0, False, "lost")
        self._previous_gray: np.ndarray | None = None
        self._previous_point: np.ndarray | None = None
        self._template: np.ndarray | None = None
        self._template_half_size = 34
        self._pose_hint: Point2D | None = None
        self._kalman = _BarAnchorKalmanFilter(fps, config)
        self._predicted_point: Point2D | None = None
        self._last_detection_point: Point2D | None = None
        self._last_detection_velocity: Point2D | None = None

    def _reset_motion_filters(self) -> None:
        self._x_filter = OneEuroFilter(frequency_hz=self._fps, min_cutoff=1.1, beta=0.05)
        self._y_filter = OneEuroFilter(frequency_hz=self._fps, min_cutoff=1.1, beta=0.05)
        self._rect_x_filter = OneEuroFilter(frequency_hz=self._fps, min_cutoff=0.8, beta=0.015)
        self._rect_y_filter = OneEuroFilter(frequency_hz=self._fps, min_cutoff=0.8, beta=0.015)
        self._kalman.reset()

    @property
    def state(self) -> BarAnchorState:
        return self._last_state

    def set_pose_hint(self, point: Point2D | None) -> None:
        """Hint of where the bar is held (e.g. wrist midpoint). Biases the anchor
        toward the plate at bar height instead of a plate resting on the floor."""
        self._pose_hint = point

    def seed_from_pose(self, frame: Frame, pose: "PoseResult") -> bool:
        if self._last_state.locked and self._last_state.source != "lost":
            return False

        if pose is None or not pose.detected or pose.keypoints is None:
            return False

        visibility_floor = 0.5
        wrists: list["PoseKeypoint"] = []
        for keypoint in pose.keypoints:
            if keypoint.name in {"left_wrist", "right_wrist"} and keypoint.visibility >= visibility_floor:
                wrists.append(keypoint)

        if len(wrists) < 2:
            return False

        center_x = sum(kp.x for kp in wrists) / len(wrists)
        center_y = sum(kp.y for kp in wrists) / len(wrists)
        confidence = min(0.55, float(min(kp.visibility for kp in wrists)) * 0.7 + 0.15)

        point = self._correct_anchor_point(
            Point2D(self._x_filter.apply(center_x), self._y_filter.apply(center_y))
        )
        size = max(self._config.min_rect_size_pixels, 200.0)
        frame_height, frame_width = frame.shape[:2]
        rect = self._make_rect(point, size, frame_width, frame_height)

        self._last_state = BarAnchorState(
            point=point,
            rect=rect,
            confidence=float(confidence),
            missing_frames=0,
            locked=True,
            source="pose_seed",
            hub_detected=False,
            hub_rect=None,
            plate_confidence=0.0,
            hub_confidence=0.0,
            display_rect=rect,
            measurement_point=None,
            measurement_confidence=0.0,
            measurable=False,
        )
        self._remember_frame(frame, point, "pose_seed")
        return True

    def update(self, frame: Frame, detections: list[Detection]) -> BarAnchorState:
        frame_height, frame_width = frame.shape[:2]
        self._predicted_point = self._kalman.predict()
        observation = self._select_observation(frame, detections, frame_width, frame_height)

        if observation is not None:
            state = self._state_from_observation(observation, frame_width, frame_height)
        else:
            flow_point = (
                self._track_with_optical_flow(frame)
                if self._last_state.missing_frames < self._config.max_missing_frames
                else None
            )
            if flow_point is not None and self._last_state.rect is not None:
                state = self._state_from_flow(flow_point, frame_width, frame_height)
            else:
                template_point = (
                    self._track_with_template(frame)
                    if self._last_state.missing_frames < self._config.max_missing_frames
                    else None
                )
                if template_point is not None and self._last_state.rect is not None:
                    state = self._state_from_template(template_point, frame_width, frame_height)
                else:
                    prediction_point = self._usable_prediction_point()
                    if prediction_point is not None and self._last_state.rect is not None:
                        state = self._state_from_prediction(
                            prediction_point,
                            frame_width,
                            frame_height,
                        )
                    else:
                        state = self._state_from_hold()

        self._remember_frame(frame, state.point, state.source)
        self._last_state = state
        return state

    def _select_observation(
        self,
        frame: Frame,
        detections: list[Detection],
        frame_width: int,
        frame_height: int,
    ) -> AnchorObservation | None:
        candidates = self._build_observations(frame, detections)
        if not candidates:
            return None

        if self._last_state.point is None or self._last_state.rect is None:
            return max(
                candidates,
                key=lambda obs: self._initial_observation_score(obs)
                * self._pose_hint_bonus(obs, frame_width, frame_height),
            )

        max_jump = max(frame_width, frame_height) * self._config.max_detection_jump_ratio
        if self._last_state.track_id is not None and self._last_state.source != "lost":
            same_track_candidates = [
                candidate
                for candidate in candidates
                if candidate.track_id == self._last_state.track_id
            ]
            if same_track_candidates:
                candidates = same_track_candidates

        if self._last_state.rect is not None and self._last_state.source != "lost":
            locked_max_jump = max(
                36.0,
                max(self._last_state.rect.width, self._last_state.rect.height)
                * self._config.locked_switch_distance_ratio,
            )
            nearby_candidates = [
                candidate
                for candidate in candidates
                if self._observation_distance_from_state(candidate) <= locked_max_jump
                or self._candidate_matches_locked_plate(candidate)
            ]
            # A reliable wrist hint can break a stale lock. If the bar (held at the wrists)
            # has moved away from the locked spot -- e.g. the lifter stood up while a static
            # floor plate kept the lock -- also admit detections near the wrists so the
            # scoring below (which rewards the pose hint) can switch onto the real bar plate.
            if self._pose_hint is not None:
                hint_radius = max(locked_max_jump, max(frame_width, frame_height) * 0.33)
                for candidate in candidates:
                    if candidate in nearby_candidates:
                        continue
                    distance_to_hint = float(
                        np.hypot(
                            candidate.point.x - self._pose_hint.x,
                            candidate.point.y - self._pose_hint.y,
                        )
                    )
                    if distance_to_hint <= hint_radius:
                        nearby_candidates.append(candidate)
            if nearby_candidates:
                candidates = nearby_candidates
            elif self._last_state.missing_frames < self._config.reacquire_after_frames:
                reacquire_candidates = self._vertical_reacquire_candidates(
                    candidates,
                    frame_width,
                    frame_height,
                )
                if reacquire_candidates:
                    candidates = reacquire_candidates
                else:
                    # Locked, but the only detections are far (the other disc or a floor
                    # plate). Hold the current position via optical flow instead of jumping;
                    # only allow a free re-acquire after a longer loss.
                    return None

        valid: list[tuple[float, AnchorObservation]] = []

        for candidate in candidates:
            distance = self._observation_distance_from_state(candidate)
            if (
                self._last_state.missing_frames < self._config.reacquire_after_frames
                and distance > max_jump
            ):
                continue

            continuity = max(0.0, 1.0 - distance / max(1.0, max_jump))
            is_hub = candidate.label in {"bar_hub", "bar_sleeve"}
            label_bonus = 1.5 if is_hub else (1.2 if candidate.label == "plate" else 1.0)
            hint_bonus = self._pose_hint_bonus(candidate, frame_width, frame_height)
            valid.append(
                (candidate.confidence * label_bonus * (1.0 + continuity) * hint_bonus, candidate)
            )

        if not valid:
            return None

        return max(valid, key=lambda item: item[0])[1]

    @staticmethod
    def _observation_distance_from_point(observation: AnchorObservation, point: Point2D) -> float:
        distances = [
            float(np.hypot(observation.point.x - point.x, observation.point.y - point.y)),
            float(np.hypot(observation.rect.center.x - point.x, observation.rect.center.y - point.y)),
        ]
        if observation.display_rect is not None:
            center = observation.display_rect.center
            distances.append(float(np.hypot(center.x - point.x, center.y - point.y)))
        return min(distances)

    def _observation_distance_from_state(self, observation: AnchorObservation) -> float:
        points: list[Point2D] = []
        if self._last_state.point is not None:
            points.append(self._last_state.point)
        if self._last_state.rect is not None:
            points.append(self._last_state.rect.center)
        if self._last_state.display_rect is not None:
            points.append(self._last_state.display_rect.center)
        if not points:
            return 0.0
        return min(self._observation_distance_from_point(observation, point) for point in points)

    def _candidate_matches_locked_plate(self, observation: AnchorObservation) -> bool:
        if self._last_state.rect is None:
            return False
        previous = self._last_state.display_rect or self._last_state.rect
        current = observation.display_rect or observation.rect
        previous_center = previous.center
        current_center = current.center
        last_size = max(previous.width, previous.height, 1.0)
        dx = abs(current_center.x - previous_center.x)
        dy = abs(current_center.y - previous_center.y)
        return dx <= max(28.0, last_size * 0.38) and dy <= max(48.0, last_size * 0.95)

    def _pose_hint_bonus(
        self,
        observation: AnchorObservation,
        frame_width: int,
        frame_height: int,
    ) -> float:
        if self._pose_hint is None:
            return 1.0

        scale = max(1.0, max(frame_width, frame_height) * 0.22)
        dy = abs(observation.point.y - self._pose_hint.y)
        dx = abs(observation.point.x - self._pose_hint.x)
        vertical = max(0.15, 1.0 - dy / scale)
        horizontal = max(0.55, 1.0 - dx / (scale * 2.2))
        return vertical * horizontal

    def _vertical_reacquire_candidates(
        self,
        candidates: list[AnchorObservation],
        frame_width: int,
        frame_height: int,
    ) -> list[AnchorObservation]:
        if self._last_state.point is None or self._last_state.rect is None:
            return []

        last = self._last_state.point
        last_size = max(self._last_state.rect.width, self._last_state.rect.height)
        max_dx = max(24.0, last_size * self._config.vertical_reacquire_x_ratio)
        max_dy = max(48.0, last_size * self._config.vertical_reacquire_y_ratio)
        pose_max_dy = max(40.0, last_size * self._config.vertical_reacquire_pose_y_ratio)

        vertical: list[AnchorObservation] = []
        for candidate in candidates:
            if candidate.hub_confidence < self._config.hub_confidence_threshold:
                continue
            dx = abs(candidate.point.x - last.x)
            dy = abs(candidate.point.y - last.y)
            if dx > max_dx or dy > max_dy:
                continue
            if self._pose_hint is not None:
                pose_dx = abs(candidate.point.x - self._pose_hint.x)
                pose_dy = abs(candidate.point.y - self._pose_hint.y)
                pose_band_x = max(max_dx, frame_width * 0.10)
                if pose_dx > pose_band_x or pose_dy > pose_max_dy:
                    continue
            vertical.append(candidate)

        return vertical

    def _build_observations(self, frame: Frame, detections: list[Detection]) -> list[AnchorObservation]:
        frame_height, frame_width = frame.shape[:2]
        usable = [
            detection
            for detection in detections
            if detection.label in {"barbell", "plate", "bar_hub", "bar_sleeve"}
            and detection.confidence >= self._config.detection_confidence_floor
        ]
        bar_hubs = [d for d in usable if d.label == "bar_hub"]
        bar_sleeves = [d for d in usable if d.label == "bar_sleeve"]
        plates = [d for d in usable if d.label == "plate"]
        barbells = [d for d in usable if d.label == "barbell"]
        hub_like = bar_hubs + barbells + bar_sleeves
        observations: list[AnchorObservation] = []

        for plate in plates:
            plate_center = Point2D(*plate.center)
            plate_rect = self._refined_plate_rect_from_frame(frame, plate)
            display_rect = self._raw_plate_rect_from_detection(plate, frame_width, frame_height)
            nearby_hub = self._nearest_hub(plate, hub_like)
            if nearby_hub is not None:
                point = Point2D(*nearby_hub.center)
                hub_confidence = nearby_hub.confidence * self._config.hub_confidence_multiplier
                confidence = max(plate.confidence, hub_confidence)
                label = "bar_hub" if nearby_hub.label in {"bar_hub", "barbell"} else "bar_sleeve"
                hub_rect = self._hub_rect_from_detection(nearby_hub)
                track_id = getattr(plate, "track_id", None) or getattr(nearby_hub, "track_id", None)
            else:
                point = plate_rect.center
                hub_confidence = 0.0
                confidence = plate.confidence
                label = "plate"
                hub_rect = None
                track_id = getattr(plate, "track_id", None)
            observations.append(
                AnchorObservation(
                    point=point,
                    rect=plate_rect,
                    confidence=confidence,
                    label=label,
                    hub_rect=hub_rect,
                    plate_confidence=plate.confidence,
                    hub_confidence=hub_confidence,
                    track_id=track_id,
                    display_rect=display_rect,
                )
            )

        if observations:
            return observations

        if self._last_state.rect is None or self._last_state.source == "lost":
            return observations

        expanded_last_rect = self._expanded_rect(self._last_state.rect, frame_width, frame_height, 0.55)
        for hub in hub_like:
            point = Point2D(*hub.center)
            if not self._point_inside_rect(point, expanded_last_rect):
                continue
            rect = self._shift_last_rect(point, frame_width, frame_height)
            observations.append(
                AnchorObservation(
                    point=point,
                    rect=rect,
                    confidence=hub.confidence * self._config.hub_confidence_multiplier,
                    label="bar_hub" if hub.label in {"bar_hub", "barbell"} else "bar_sleeve",
                    hub_rect=self._hub_rect_from_detection(hub),
                    plate_confidence=self._last_state.plate_confidence * 0.85,
                    hub_confidence=hub.confidence,
                    track_id=getattr(hub, "track_id", self._last_state.track_id),
                    display_rect=self._shift_last_display_rect(point, frame_width, frame_height),
                )
            )

        return observations

    @staticmethod
    def _nearest_hub(plate: Detection, hubs: list[Detection]) -> Detection | None:
        if not hubs:
            return None

        plate_x, plate_y = plate.center
        # The trained model labels the visible sleeve/hub, which often sits near
        # the plate edge in side-view videos, and can sit just outside the disc box
        # when the visible hub is the exposed sleeve next to the loaded plate.
        plate_size = max(plate.width, plate.height)
        max_inside_distance = plate_size * 0.62
        pad_x = plate.width * 0.36
        pad_y = plate.height * 0.30
        lateral_reach = max(24.0, plate.width * 0.78, plate_size * 0.45)
        vertical_band = max(12.0, plate.height * 0.34)
        nearby: list[tuple[float, Detection]] = []
        for hub in hubs:
            hub_x, hub_y = hub.center
            in_padded_plate = (
                plate.x1 - pad_x <= hub_x <= plate.x2 + pad_x
                and plate.y1 - pad_y <= hub_y <= plate.y2 + pad_y
            )
            distance = float(np.hypot(hub_x - plate_x, hub_y - plate_y))
            if in_padded_plate and distance <= max_inside_distance:
                nearby.append((distance, hub))
                continue

            outside_gap = max(plate.x1 - hub_x, hub_x - plate.x2, 0.0)
            lateral_to_edge = outside_gap > 0.0 and outside_gap <= lateral_reach
            vertically_aligned = abs(hub_y - plate_y) <= vertical_band
            if not (lateral_to_edge and vertically_aligned):
                continue

            distance = float(np.hypot(hub_x - plate_x, hub_y - plate_y))
            if distance <= max(plate.width * 1.30, plate_size * 0.95):
                edge_score = outside_gap * 0.65 + abs(hub_y - plate_y) * 1.35
                nearby.append((edge_score, hub))

        if not nearby:
            return None

        return min(nearby, key=lambda item: item[0])[1]

    @staticmethod
    def _point_inside_rect(point: Point2D, rect: AnchorRect) -> bool:
        return rect.x1 <= point.x <= rect.x2 and rect.y1 <= point.y <= rect.y2

    @staticmethod
    def _initial_observation_score(observation: AnchorObservation) -> float:
        is_hub = observation.label in {"bar_hub", "bar_sleeve"}
        label_bonus = 1.5 if is_hub else (1.2 if observation.label == "plate" else 1.0)
        size_bonus = max(observation.rect.width, observation.rect.height) / 100.0
        return observation.confidence * label_bonus + size_bonus

    def _state_from_observation(
        self,
        observation: AnchorObservation,
        frame_width: int,
        frame_height: int,
    ) -> BarAnchorState:
        center_x = observation.point.x
        center_y = observation.point.y
        snap_to_detection = self._should_snap_to_vertical_reacquire(
            observation,
            frame_width,
            frame_height,
        )
        if self._last_state.point is not None:
            delta = float(np.hypot(center_x - self._last_state.point.x, center_y - self._last_state.point.y))
            if not snap_to_detection and delta < self._config.center_deadband_pixels:
                center_x = self._last_state.point.x
                center_y = self._last_state.point.y

        measured = Point2D(center_x, center_y)
        use_measured_detection = (
            snap_to_detection
            or (
                observation.display_rect is not None
                and observation.plate_confidence >= self._config.plate_confidence_threshold
            )
        )
        if snap_to_detection:
            self._reset_motion_filters()
        predicted = self._correct_anchor_point(measured)
        filtered = Point2D(
            self._x_filter.apply(predicted.x),
            self._y_filter.apply(predicted.y),
        )
        point = measured if use_measured_detection else filtered
        rect = self._stabilize_observation_rect(observation.rect, point, frame_width, frame_height)
        is_hub = observation.label in {"bar_hub", "bar_sleeve"}
        measurement_confidence = (
            float(observation.hub_confidence)
            if is_hub and observation.hub_confidence >= self._config.hub_confidence_threshold
            else 0.0
        )
        measurable = measurement_confidence >= self._config.hub_confidence_threshold
        track_id = getattr(observation, "track_id", None)
        self._remember_detection_motion(measured)
        return BarAnchorState(
            point=point,
            rect=rect,
            confidence=float(min(0.99, observation.confidence)),
            missing_frames=0,
            locked=True,
            source="detection",
            hub_detected=is_hub,
            track_id=track_id,
            hub_rect=observation.hub_rect if measurable else None,
            plate_confidence=float(observation.plate_confidence),
            hub_confidence=float(observation.hub_confidence),
            display_rect=observation.display_rect or rect,
            measurement_point=point if measurable else None,
            measurement_confidence=measurement_confidence,
            measurable=measurable,
        )

    def _remember_detection_motion(self, point: Point2D) -> None:
        if self._last_detection_point is not None:
            self._last_detection_velocity = Point2D(
                point.x - self._last_detection_point.x,
                point.y - self._last_detection_point.y,
            )
        self._last_detection_point = point

    def _should_snap_to_vertical_reacquire(
        self,
        observation: AnchorObservation,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        if self._last_state.point is None or self._last_state.rect is None:
            return False
        if observation.hub_confidence < self._config.hub_confidence_threshold:
            return False

        last = self._last_state.point
        last_size = max(self._last_state.rect.width, self._last_state.rect.height)
        dx = abs(observation.point.x - last.x)
        dy = abs(observation.point.y - last.y)
        locked_max_jump = max(36.0, last_size * self._config.locked_switch_distance_ratio)
        if dy <= locked_max_jump:
            return False
        if dx > max(24.0, last_size * self._config.vertical_reacquire_x_ratio):
            return False
        if dy > max(48.0, last_size * self._config.vertical_reacquire_y_ratio):
            return False
        if self._pose_hint is not None:
            pose_dx = abs(observation.point.x - self._pose_hint.x)
            pose_dy = abs(observation.point.y - self._pose_hint.y)
            if pose_dx > max(24.0, frame_width * 0.10):
                return False
            if pose_dy > max(40.0, last_size * self._config.vertical_reacquire_pose_y_ratio):
                return False
        return True

    def _stabilize_observation_rect(
        self,
        observation_rect: AnchorRect,
        point: Point2D,
        frame_width: int,
        frame_height: int,
    ) -> AnchorRect:
        observed_center = observation_rect.center
        center_x = observed_center.x
        center_y = observed_center.y
        width = observation_rect.width
        height = observation_rect.height

        if self._last_state.rect is not None and self._last_state.source != "lost":
            last_rect = self._last_state.rect
            last_center = last_rect.center
            if self._last_state.point is not None:
                predicted_rect = self._shift_last_rect(point, frame_width, frame_height)
                predicted_center = predicted_rect.center
            else:
                predicted_center = last_center

            center_delta = float(np.hypot(center_x - predicted_center.x, center_y - predicted_center.y))
            if center_delta < max(self._config.rect_center_deadband_pixels, max(last_rect.width, last_rect.height) * 0.10):
                center_x = predicted_center.x
                center_y = predicted_center.y
            else:
                correction_alpha = 0.08
                center_x = predicted_center.x * (1.0 - correction_alpha) + center_x * correction_alpha
                center_y = predicted_center.y * (1.0 - correction_alpha) + center_y * correction_alpha

            alpha = self._config.rect_size_smoothing_alpha
            last_width = max(1.0, last_rect.width)
            last_height = max(1.0, last_rect.height)
            width_ratio = width / last_width
            height_ratio = height / last_height
            if 0.72 <= width_ratio <= 1.38:
                width = last_width * (1.0 - alpha) + width * alpha
            else:
                width = last_width
            if 0.72 <= height_ratio <= 1.38:
                height = last_height * (1.0 - alpha) + height * alpha
            else:
                height = last_height

        filtered_center = Point2D(
            self._rect_x_filter.apply(center_x),
            self._rect_y_filter.apply(center_y),
        )
        rect = AnchorRect(
            x1=filtered_center.x - width / 2.0,
            y1=filtered_center.y - height / 2.0,
            x2=filtered_center.x + width / 2.0,
            y2=filtered_center.y + height / 2.0,
        )
        return self._clamp_rect(rect, frame_width, frame_height)

    def _state_from_flow(
        self,
        point: Point2D,
        frame_width: int,
        frame_height: int,
    ) -> BarAnchorState:
        predicted = self._correct_anchor_point(point)
        filtered = Point2D(self._x_filter.apply(predicted.x), self._y_filter.apply(predicted.y))
        rect = self._shift_last_rect(filtered, frame_width, frame_height)
        display_rect = self._shift_last_display_rect(filtered, frame_width, frame_height)
        confidence = self._last_state.confidence * self._config.optical_flow_confidence_decay
        return BarAnchorState(
            point=filtered,
            rect=rect,
            confidence=float(max(0.18, confidence)),
            missing_frames=self._last_state.missing_frames + 1,
            locked=True,
            source="optical_flow",
            hub_detected=False,
            track_id=self._last_state.track_id,
            hub_rect=None,
            plate_confidence=self._last_state.plate_confidence * self._config.optical_flow_confidence_decay,
            hub_confidence=0.0,
            display_rect=display_rect,
            measurement_point=None,
            measurement_confidence=0.0,
            measurable=False,
        )

    def _state_from_template(
        self,
        point: Point2D,
        frame_width: int,
        frame_height: int,
    ) -> BarAnchorState:
        predicted = self._correct_anchor_point(point)
        filtered = Point2D(self._x_filter.apply(predicted.x), self._y_filter.apply(predicted.y))
        rect = self._shift_last_rect(filtered, frame_width, frame_height)
        display_rect = self._shift_last_display_rect(filtered, frame_width, frame_height)
        confidence = self._last_state.confidence * self._config.template_confidence_decay
        return BarAnchorState(
            point=filtered,
            rect=rect,
            confidence=float(max(0.12, confidence)),
            missing_frames=self._last_state.missing_frames + 1,
            locked=True,
            source="template",
            hub_detected=False,
            track_id=self._last_state.track_id,
            hub_rect=None,
            plate_confidence=self._last_state.plate_confidence * self._config.template_confidence_decay,
            hub_confidence=0.0,
            display_rect=display_rect,
            measurement_point=None,
            measurement_confidence=0.0,
            measurable=False,
        )

    def _state_from_prediction(
        self,
        point: Point2D,
        frame_width: int,
        frame_height: int,
    ) -> BarAnchorState:
        filtered = Point2D(self._x_filter.apply(point.x), self._y_filter.apply(point.y))
        rect = self._shift_last_rect(filtered, frame_width, frame_height)
        display_rect = self._shift_last_display_rect(filtered, frame_width, frame_height)
        confidence = self._last_state.confidence * self._config.prediction_confidence_decay
        return BarAnchorState(
            point=filtered,
            rect=rect,
            confidence=float(max(0.10, confidence)),
            missing_frames=self._last_state.missing_frames + 1,
            locked=True,
            source="prediction",
            hub_detected=False,
            track_id=self._last_state.track_id,
            hub_rect=None,
            plate_confidence=self._last_state.plate_confidence * self._config.prediction_confidence_decay,
            hub_confidence=0.0,
            display_rect=display_rect,
            measurement_point=None,
            measurement_confidence=0.0,
            measurable=False,
        )

    def _state_from_hold(self) -> BarAnchorState:
        if (
            self._last_state.point is None
            or self._last_state.rect is None
            or self._last_state.missing_frames >= self._config.max_missing_frames
        ):
            self._kalman.reset()
            return BarAnchorState(None, None, 0.0, self._last_state.missing_frames + 1, False, "lost")

        confidence = float(max(0.08, self._last_state.confidence * self._config.hold_confidence_decay))
        return BarAnchorState(
            point=self._last_state.point,
            rect=self._last_state.rect,
            confidence=confidence * self._config.hold_confidence_multiplier,
            missing_frames=self._last_state.missing_frames + 1,
            locked=True,
            source="hold",
            hub_detected=False,
            track_id=self._last_state.track_id,
            hub_rect=None,
            plate_confidence=self._last_state.plate_confidence * self._config.hold_confidence_decay,
            hub_confidence=0.0,
            display_rect=self._last_state.display_rect,
            measurement_point=None,
            measurement_confidence=0.0,
            measurable=False,
        )

    def _correct_anchor_point(self, point: Point2D) -> Point2D:
        if self._last_state.source == "lost" or self._last_state.point is None:
            self._kalman.reset()
        return self._kalman.correct(point)

    def _usable_prediction_point(self) -> Point2D | None:
        if (
            self._last_state.point is None
            or self._last_state.missing_frames >= self._config.kalman_prediction_frames
        ):
            return None

        max_step = max(
            40.0,
            max(
                self._last_state.rect.width if self._last_state.rect is not None else 0.0,
                self._last_state.rect.height if self._last_state.rect is not None else 0.0,
            )
            * 0.30,
        )

        if self._last_detection_velocity is not None:
            simple_prediction = Point2D(
                self._last_state.point.x + self._last_detection_velocity.x,
                self._last_state.point.y + self._last_detection_velocity.y,
            )
            simple_step = float(
                np.hypot(
                    simple_prediction.x - self._last_state.point.x,
                    simple_prediction.y - self._last_state.point.y,
                )
            )
            if simple_step <= max_step:
                return simple_prediction

        if self._predicted_point is None or not self._kalman.initialized:
            return None

        step = float(
            np.hypot(
                self._predicted_point.x - self._last_state.point.x,
                self._predicted_point.y - self._last_state.point.y,
            )
        )
        if step > max_step:
            return None

        return self._predicted_point

    def _track_with_optical_flow(self, frame: Frame) -> Point2D | None:
        if self._previous_gray is None or self._previous_point is None or self._last_state.point is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            gray,
            self._previous_point,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_points is None or status is None or int(status[0][0]) != 1:
            return None

        if errors is not None and float(errors[0][0]) > 35.0:
            return None

        x, y = next_points[0][0]
        return Point2D(float(x), float(y))

    def _track_with_template(self, frame: Frame) -> Point2D | None:
        if self._template is None or self._last_state.point is None or self._last_state.rect is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        search = self._expanded_rect(self._last_state.rect, frame.shape[1], frame.shape[0], 0.45)
        x1, y1, x2, y2 = int(search.x1), int(search.y1), int(search.x2), int(search.y2)
        search_image = gray[y1:y2, x1:x2]
        if (
            search_image.shape[0] < self._template.shape[0]
            or search_image.shape[1] < self._template.shape[1]
        ):
            return None

        result = cv2.matchTemplate(search_image, self._template, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, max_location = cv2.minMaxLoc(result)
        if max_value < self._config.template_match_threshold:
            return None

        center_x = x1 + max_location[0] + self._template.shape[1] / 2.0
        center_y = y1 + max_location[1] + self._template.shape[0] / 2.0
        return Point2D(float(center_x), float(center_y))

    def _remember_frame(self, frame: Frame, point: Point2D | None, source: AnchorSource) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._previous_gray = gray
        if point is None:
            self._previous_point = None
            return

        self._previous_point = np.array([[[point.x, point.y]]], dtype=np.float32)
        if source in {"detection", "template", "optical_flow", "pose_seed"}:
            template = self._extract_template(gray, point)
            if template is not None:
                self._template = template

    def _extract_template(self, gray: np.ndarray, point: Point2D) -> np.ndarray | None:
        half = self._template_half_size
        x1 = int(round(point.x - half))
        y1 = int(round(point.y - half))
        x2 = int(round(point.x + half))
        y2 = int(round(point.y + half))
        if x1 < 0 or y1 < 0 or x2 > gray.shape[1] or y2 > gray.shape[0]:
            return None

        patch = gray[y1:y2, x1:x2]
        if patch.size == 0 or float(np.std(patch)) < 4.0:
            return None

        return patch.copy()

    def _plate_rect_from_detection(self, detection: Detection) -> AnchorRect:
        pad_x = detection.width * self._config.rect_padding_ratio
        pad_y = detection.height * self._config.rect_padding_ratio
        rect = AnchorRect(
            x1=detection.x1 - pad_x,
            y1=detection.y1 - pad_y,
            x2=detection.x2 + pad_x,
            y2=detection.y2 + pad_y,
        )
        return self._scaled_rect(rect, self._config.plate_rect_fallback_scale)

    @staticmethod
    def _raw_plate_rect_from_detection(
        detection: Detection,
        frame_width: int,
        frame_height: int,
    ) -> AnchorRect:
        return BarAnchorTracker._clamp_rect(
            AnchorRect(detection.x1, detection.y1, detection.x2, detection.y2),
            frame_width,
            frame_height,
        )

    def _refined_plate_rect_from_frame(self, frame: Frame, detection: Detection) -> AnchorRect:
        fallback = self._plate_rect_from_detection(detection)
        frame_height, frame_width = frame.shape[:2]
        search = self._clamp_rect(
            AnchorRect(
                detection.x1 - detection.width * 0.10,
                detection.y1 - detection.height * 0.10,
                detection.x2 + detection.width * 0.10,
                detection.y2 + detection.height * 0.10,
            ),
            frame_width,
            frame_height,
        )
        x1, y1, x2, y2 = int(search.x1), int(search.y1), int(search.x2), int(search.y2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return fallback

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = self._plate_color_mask(hsv, detection.color)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return fallback

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        min_area = max(250.0, detection.width * detection.height * 0.08)
        if area < min_area:
            return fallback

        (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
        if radius <= 0:
            return fallback
        max_refined_diameter = max(detection.width, detection.height) * 1.15
        if radius * 2.0 > max_refined_diameter:
            return fallback

        center_x = x1 + float(circle_x)
        center_y = y1 + float(circle_y)
        pad = max(3.0, radius * self._config.rect_padding_ratio)
        refined = AnchorRect(
            x1=center_x - radius - pad,
            y1=center_y - radius - pad,
            x2=center_x + radius + pad,
            y2=center_y + radius + pad,
        )
        return self._clamp_rect(
            self._scaled_rect(refined, self._config.refined_plate_rect_scale),
            frame_width,
            frame_height,
        )

    @staticmethod
    def _plate_color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
        if color == "beige":
            beige = cv2.inRange(hsv, np.array([10, 20, 85]), np.array([45, 155, 245]))
            light_yellow = cv2.inRange(hsv, np.array([20, 30, 105]), np.array([60, 150, 250]))
            olive = cv2.inRange(hsv, np.array([35, 25, 55]), np.array([85, 175, 215]))
            return cv2.bitwise_or(cv2.bitwise_or(beige, light_yellow), olive)

        red_low = cv2.inRange(hsv, np.array([0, 85, 35]), np.array([14, 255, 255]))
        red_high = cv2.inRange(hsv, np.array([170, 85, 35]), np.array([179, 255, 255]))
        orange = cv2.inRange(hsv, np.array([8, 80, 45]), np.array([28, 255, 255]))
        return cv2.bitwise_or(cv2.bitwise_or(red_low, red_high), orange)

    def _hub_rect_from_detection(self, detection: Detection) -> AnchorRect:
        point = Point2D(*detection.center)
        size = max(18.0, min(72.0, max(detection.width, detection.height) * 1.5))
        return AnchorRect(
            x1=point.x - size / 2.0,
            y1=point.y - size / 2.0,
            x2=point.x + size / 2.0,
            y2=point.y + size / 2.0,
        )

    @staticmethod
    def _make_rect(point: Point2D, size: float, frame_width: int, frame_height: int) -> AnchorRect:
        half = size / 2.0
        return AnchorRect(
            x1=max(0.0, point.x - half),
            y1=max(0.0, point.y - half),
            x2=min(float(frame_width - 1), point.x + half),
            y2=min(float(frame_height - 1), point.y + half),
        )

    def _move_rect_to_point(
        self,
        rect: AnchorRect,
        old_point: Point2D,
        new_point: Point2D,
        frame_width: int,
        frame_height: int,
    ) -> AnchorRect:
        dx = new_point.x - old_point.x
        dy = new_point.y - old_point.y
        return self._clamp_rect(
            AnchorRect(rect.x1 + dx, rect.y1 + dy, rect.x2 + dx, rect.y2 + dy),
            frame_width,
            frame_height,
        )

    def _shift_last_rect(self, point: Point2D, frame_width: int, frame_height: int) -> AnchorRect:
        if self._last_state.point is None or self._last_state.rect is None:
            size = self._config.min_rect_size_pixels
            return self._make_rect(point, size, frame_width, frame_height)

        return self._move_rect_to_point(
            self._last_state.rect,
            self._last_state.point,
            point,
            frame_width,
            frame_height,
        )

    def _shift_last_display_rect(
        self,
        point: Point2D,
        frame_width: int,
        frame_height: int,
    ) -> AnchorRect | None:
        if self._last_state.point is None or self._last_state.display_rect is None:
            return None

        return self._move_rect_to_point(
            self._last_state.display_rect,
            self._last_state.point,
            point,
            frame_width,
            frame_height,
        )

    @staticmethod
    def _clamp_rect(rect: AnchorRect, frame_width: int, frame_height: int) -> AnchorRect:
        width = rect.width
        height = rect.height
        x1 = min(max(0.0, rect.x1), max(0.0, frame_width - width))
        y1 = min(max(0.0, rect.y1), max(0.0, frame_height - height))
        return AnchorRect(
            x1=x1,
            y1=y1,
            x2=min(float(frame_width - 1), x1 + width),
            y2=min(float(frame_height - 1), y1 + height),
        )

    @staticmethod
    def _expanded_rect(rect: AnchorRect, frame_width: int, frame_height: int, ratio: float) -> AnchorRect:
        pad_x = rect.width * ratio
        pad_y = rect.height * ratio
        return AnchorRect(
            x1=max(0.0, rect.x1 - pad_x),
            y1=max(0.0, rect.y1 - pad_y),
            x2=min(float(frame_width - 1), rect.x2 + pad_x),
            y2=min(float(frame_height - 1), rect.y2 + pad_y),
        )

    @staticmethod
    def _scaled_rect(rect: AnchorRect, scale: float) -> AnchorRect:
        center = rect.center
        half_width = rect.width * scale / 2.0
        half_height = rect.height * scale / 2.0
        return AnchorRect(
            x1=center.x - half_width,
            y1=center.y - half_height,
            x2=center.x + half_width,
            y2=center.y + half_height,
        )
