"""Offline, timestamp-aligned reconstruction of barbell kinematics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from track import Point2D


@dataclass(frozen=True)
class BarMeasurement:
    frame_index: int
    time_seconds: float
    point: Point2D | None
    meters_per_pixel: float
    confidence: float
    measurable: bool


@dataclass(frozen=True)
class ReconstructedBarSample:
    frame_index: int
    time_seconds: float
    point: Point2D | None
    position_m: float | None
    velocity_mps: float | None
    observed: bool
    valid: bool


def reconstruct_bar_kinematics(
    measurements: list[BarMeasurement],
    *,
    fps: float,
    max_gap_frames: int = 6,
    window_seconds: float = 0.30,
    max_speed_mps: float = 3.0,
) -> list[ReconstructedBarSample]:
    """Reconstruct a low-noise, zero-phase trajectory from hub observations.

    Small detector gaps are filled only to support the centred local fit. They
    remain ``observed=False``, so no velocity is displayed or used for a missing
    hub frame. Long gaps and implausible teleports break the trajectory.
    """
    if not measurements:
        return []

    ordered = sorted(measurements, key=lambda item: item.frame_index)
    frames = np.array([item.frame_index for item in ordered], dtype=int)
    times = np.array([item.time_seconds for item in ordered], dtype=float)
    positions = np.full(len(ordered), np.nan, dtype=float)
    xs = np.full(len(ordered), np.nan, dtype=float)
    ys = np.full(len(ordered), np.nan, dtype=float)
    observed = np.zeros(len(ordered), dtype=bool)

    for index, measurement in enumerate(ordered):
        if (
            measurement.measurable
            and measurement.point is not None
            and measurement.confidence >= 0.28
            and measurement.meters_per_pixel > 0
        ):
            positions[index] = -measurement.point.y * measurement.meters_per_pixel
            xs[index] = measurement.point.x
            ys[index] = measurement.point.y
            observed[index] = True

    _reject_teleports(positions, times, observed, max_speed_mps)
    _reject_local_spikes(positions, times, observed)
    observed &= np.isfinite(positions)
    _interpolate_short_gaps(positions, max_gap_frames)
    _interpolate_short_gaps(xs, max_gap_frames)
    _interpolate_short_gaps(ys, max_gap_frames)

    radius = max(2, int(round(window_seconds * max(1.0, fps) / 2.0)))
    velocities = _local_polynomial_velocity(positions, times, radius)
    smoothed_positions = _local_polynomial_position(positions, times, radius)
    smoothed_xs = _local_polynomial_position(xs, times, max(2, radius // 2))
    smoothed_ys = _local_polynomial_position(ys, times, max(2, radius // 2))

    result: list[ReconstructedBarSample] = []
    for index, measurement in enumerate(ordered):
        is_observed = bool(observed[index])
        valid = bool(is_observed and np.isfinite(velocities[index]) and np.isfinite(smoothed_positions[index]))
        point = (
            Point2D(float(smoothed_xs[index]), float(smoothed_ys[index]))
            if valid and np.isfinite(smoothed_xs[index]) and np.isfinite(smoothed_ys[index])
            else None
        )
        result.append(
            ReconstructedBarSample(
                frame_index=int(frames[index]),
                time_seconds=float(times[index]),
                point=point,
                position_m=float(smoothed_positions[index]) if valid else None,
                velocity_mps=float(velocities[index]) if valid else None,
                observed=is_observed,
                valid=valid,
            )
        )
    return result


def _reject_teleports(
    positions: np.ndarray,
    times: np.ndarray,
    observed: np.ndarray,
    max_speed_mps: float,
) -> None:
    previous: int | None = None
    for index in range(len(positions)):
        if not observed[index]:
            continue
        if previous is not None:
            dt = times[index] - times[previous]
            speed = abs(positions[index] - positions[previous]) / max(dt, 1e-6)
            if speed > max_speed_mps:
                positions[index] = np.nan
                observed[index] = False
                continue
        previous = index


def _reject_local_spikes(
    positions: np.ndarray,
    times: np.ndarray,
    observed: np.ndarray,
    min_residual_m: float = 0.018,
) -> None:
    """Remove isolated detector jumps that are fast but not quite teleports.

    A hub box can jump a few centimetres for one frame and still remain below the
    absolute speed limit.  Differentiating that point creates the characteristic
    false down/up spike.  A genuine bar path is locally smooth at video frame rates,
    so compare each observation with the time-interpolated position of its immediate
    neighbours and reject only isolated residuals above 18 mm.
    """
    rejected: list[int] = []
    for index in range(1, len(positions) - 1):
        if not (observed[index - 1] and observed[index] and observed[index + 1]):
            continue
        total_dt = times[index + 1] - times[index - 1]
        if total_dt <= 0:
            continue
        fraction = (times[index] - times[index - 1]) / total_dt
        expected = positions[index - 1] + fraction * (positions[index + 1] - positions[index - 1])
        residual = abs(positions[index] - expected)
        neighbour_motion = abs(positions[index + 1] - positions[index - 1])
        threshold = max(min_residual_m, neighbour_motion * 0.65)
        if residual > threshold:
            rejected.append(index)

    for index in rejected:
        positions[index] = np.nan
        observed[index] = False


def _interpolate_short_gaps(values: np.ndarray, max_gap_frames: int) -> None:
    index = 0
    while index < len(values):
        if np.isfinite(values[index]):
            index += 1
            continue
        start = index
        while index < len(values) and not np.isfinite(values[index]):
            index += 1
        end = index
        if start == 0 or end >= len(values) or end - start > max_gap_frames:
            continue
        left = values[start - 1]
        right = values[end]
        for fill_index in range(start, end):
            fraction = (fill_index - start + 1) / (end - start + 1)
            values[fill_index] = left + (right - left) * fraction


def _local_polynomial_velocity(values: np.ndarray, times: np.ndarray, radius: int) -> np.ndarray:
    velocity = np.full(len(values), np.nan, dtype=float)
    for index in range(len(values)):
        if not np.isfinite(values[index]):
            continue
        low = max(0, index - radius)
        high = min(len(values), index + radius + 1)
        valid = np.isfinite(values[low:high]) & np.isfinite(times[low:high])
        if np.count_nonzero(valid) < 3:
            continue
        local_times = times[low:high][valid] - times[index]
        local_values = values[low:high][valid]
        degree = 2 if len(local_values) >= 5 else 1
        coefficients = np.polyfit(local_times, local_values, degree)
        velocity[index] = coefficients[-2]
    return velocity


def _local_polynomial_position(values: np.ndarray, times: np.ndarray, radius: int) -> np.ndarray:
    smoothed = np.full(len(values), np.nan, dtype=float)
    for index in range(len(values)):
        if not np.isfinite(values[index]):
            continue
        low = max(0, index - radius)
        high = min(len(values), index + radius + 1)
        valid = np.isfinite(values[low:high]) & np.isfinite(times[low:high])
        if np.count_nonzero(valid) < 3:
            smoothed[index] = values[index]
            continue
        local_times = times[low:high][valid] - times[index]
        local_values = values[low:high][valid]
        degree = 2 if len(local_values) >= 5 else 1
        smoothed[index] = np.polyval(np.polyfit(local_times, local_values, degree), 0.0)
    return smoothed
