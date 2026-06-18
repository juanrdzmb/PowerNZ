from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from detect_objects import Detection


OLYMPIC_PLATE_DIAMETER_METERS = 0.45


@dataclass(frozen=True)
class CalibrationConfig:
    plate_diameter_meters: float = OLYMPIC_PLATE_DIAMETER_METERS
    min_plate_confidence: float = 0.35


@dataclass(frozen=True)
class SpatialCalibration:
    meters_per_pixel: float
    plate_diameter_pixels: float
    plate_diameter_meters: float = OLYMPIC_PLATE_DIAMETER_METERS

    @property
    def pixels_per_meter(self) -> float:
        return 1.0 / self.meters_per_pixel

    def pixels_to_meters(self, value_pixels: float) -> float:
        return value_pixels * self.meters_per_pixel

    def meters_to_pixels(self, value_meters: float) -> float:
        return value_meters / self.meters_per_pixel


def create_calibration_from_plate_diameter(
    plate_diameter_pixels: float,
    plate_diameter_meters: float = OLYMPIC_PLATE_DIAMETER_METERS,
) -> SpatialCalibration:
    if plate_diameter_pixels <= 0:
        raise ValueError("Plate diameter in pixels must be greater than zero.")

    if plate_diameter_meters <= 0:
        raise ValueError("Plate diameter in meters must be greater than zero.")

    return SpatialCalibration(
        meters_per_pixel=plate_diameter_meters / plate_diameter_pixels,
        plate_diameter_pixels=plate_diameter_pixels,
        plate_diameter_meters=plate_diameter_meters,
    )


def estimate_plate_diameter_pixels(
    detections: list[Detection],
    config: CalibrationConfig = CalibrationConfig(),
) -> float | None:
    plate_sizes = [
        (detection.width + detection.height) / 2.0
        for detection in detections
        if detection.label == "plate"
        and detection.confidence >= config.min_plate_confidence
        and detection.width > 0
        and detection.height > 0
    ]

    if not plate_sizes:
        return None

    return float(median(plate_sizes))


class CalibrationEstimator:
    def __init__(
        self,
        config: CalibrationConfig = CalibrationConfig(),
        min_observations: int = 5,
    ) -> None:
        self._config = config
        self._min_observations = min_observations
        self._diameter_observations: list[float] = []
        self._calibration: SpatialCalibration | None = None

    @property
    def calibration(self) -> SpatialCalibration | None:
        return self._calibration

    def update(self, detections: list[Detection]) -> SpatialCalibration | None:
        diameter_pixels = estimate_plate_diameter_pixels(detections, self._config)
        if diameter_pixels is None:
            return self._calibration

        self._diameter_observations.append(diameter_pixels)

        if len(self._diameter_observations) >= self._min_observations:
            stable_diameter = float(median(self._diameter_observations))
            self._calibration = create_calibration_from_plate_diameter(
                plate_diameter_pixels=stable_diameter,
                plate_diameter_meters=self._config.plate_diameter_meters,
            )

        return self._calibration
