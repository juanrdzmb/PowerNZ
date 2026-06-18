from __future__ import annotations

from dataclasses import dataclass
from math import exp, hypot, pi


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class TrackedPoint:
    raw: Point2D | None
    filtered: Point2D | None
    frame_index: int
    missing_frames: int
    is_valid: bool


class LowPassFilter:
    def __init__(self) -> None:
        self._initialized = False
        self._value = 0.0

    @property
    def value(self) -> float:
        return self._value

    def apply(self, value: float, alpha: float) -> float:
        if not self._initialized:
            self._value = value
            self._initialized = True
            return value

        self._value = alpha * value + (1.0 - alpha) * self._value
        return self._value


class OneEuroFilter:
    def __init__(
        self,
        frequency_hz: float,
        min_cutoff: float = 1.0,
        beta: float = 0.02,
        derivative_cutoff: float = 1.0,
    ) -> None:
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be greater than zero.")

        self._frequency_hz = frequency_hz
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._derivative_cutoff = derivative_cutoff
        self._value_filter = LowPassFilter()
        self._derivative_filter = LowPassFilter()
        self._last_raw_value: float | None = None

    def apply(self, value: float) -> float:
        derivative = 0.0
        if self._last_raw_value is not None:
            derivative = (value - self._last_raw_value) * self._frequency_hz

        self._last_raw_value = value
        filtered_derivative = self._derivative_filter.apply(
            derivative,
            self._alpha(self._derivative_cutoff),
        )
        cutoff = self._min_cutoff + self._beta * abs(filtered_derivative)
        return self._value_filter.apply(value, self._alpha(cutoff))

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * pi * cutoff)
        te = 1.0 / self._frequency_hz
        return 1.0 / (1.0 + tau / te)

    def set_min_cutoff(self, min_cutoff: float) -> None:
        if min_cutoff <= 0:
            raise ValueError("min_cutoff must be greater than zero.")
        self._min_cutoff = min_cutoff


class PointTracker:
    def __init__(
        self,
        frequency_hz: float,
        max_missing_frames: int = 10,
        min_cutoff: float = 1.0,
        beta: float = 0.02,
        deadband_pixels: float = 1.5,
    ) -> None:
        self._x_filter = OneEuroFilter(
            frequency_hz=frequency_hz,
            min_cutoff=min_cutoff,
            beta=beta,
        )
        self._y_filter = OneEuroFilter(
            frequency_hz=frequency_hz,
            min_cutoff=min_cutoff,
            beta=beta,
        )
        self._max_missing_frames = max_missing_frames
        self._deadband_pixels = deadband_pixels
        self._missing_frames = 0
        self._last_filtered: Point2D | None = None

    def update(self, point: Point2D | None, frame_index: int) -> TrackedPoint:
        if point is None:
            self._missing_frames += 1
            is_valid = (
                self._last_filtered is not None
                and self._missing_frames <= self._max_missing_frames
            )
            return TrackedPoint(
                raw=None,
                filtered=self._last_filtered if is_valid else None,
                frame_index=frame_index,
                missing_frames=self._missing_frames,
                is_valid=is_valid,
            )

        self._missing_frames = 0
        if (
            self._last_filtered is not None
            and hypot(point.x - self._last_filtered.x, point.y - self._last_filtered.y)
            < self._deadband_pixels
        ):
            return TrackedPoint(
                raw=point,
                filtered=self._last_filtered,
                frame_index=frame_index,
                missing_frames=0,
                is_valid=True,
            )

        filtered = Point2D(
            x=self._x_filter.apply(point.x),
            y=self._y_filter.apply(point.y),
        )
        self._last_filtered = filtered

        return TrackedPoint(
            raw=point,
            filtered=filtered,
            frame_index=frame_index,
            missing_frames=0,
            is_valid=True,
        )
