from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from metrics import CompletedRep, KinematicSample


@dataclass(frozen=True)
class RepReport:
    rep_index: int
    start_frame: int
    lockout_frame: int
    end_frame: int
    duration_seconds: float
    concentric_seconds: float
    eccentric_seconds: float
    rom_m: float
    mean_concentric_velocity_mps: float
    peak_velocity_mps: float
    velocity_loss_from_best_percent: float
    velocity_loss_from_previous_percent: float
    velocity_loss_warning: bool
    tracking_confidence_mean: float = 0.0
    hub_confidence_mean: float = 0.0
    plate_confidence_mean: float = 0.0
    tracking_source_pct: dict[str, float] | None = None
    validation_status: str = "accepted"
    validation_reason: str = "accepted"


@dataclass(frozen=True)
class AnalysisReport:
    input_path: str
    output_path: str
    fps: float
    frame_count: int
    tracked_frames: int
    object_frames: int
    completed_reps: int
    reps: list[RepReport]
    hub_reliable_frames_pct: float = 0.0
    reviewed_reps: int = 0
    rejected_reps: int = 0


class RepReportBuilder:
    def __init__(self, fps: float, velocity_loss_threshold_percent: float = 20.0) -> None:
        self._fps = fps
        self._velocity_loss_threshold_percent = velocity_loss_threshold_percent
        self._samples: list[KinematicSample] = []
        self._built_reports: list[RepReport] = []

    def add_sample(self, sample: KinematicSample) -> None:
        self._samples.append(sample)

    def build_rep_report(
        self,
        rep: CompletedRep,
        validation_status: str = "accepted",
        validation_reason: str = "accepted",
    ) -> RepReport:
        for existing in self._built_reports:
            if existing.rep_index == rep.rep_index:
                return existing

        rep_samples = [
            sample
            for sample in self._samples
            if rep.start_frame <= sample.frame_index <= rep.end_frame
        ]
        concentric_samples = [
            sample
            for sample in rep_samples
            if rep.start_frame <= sample.frame_index <= rep.lockout_frame
        ]

        positive_velocities = [
            sample.smoothed_velocity_mps
            for sample in concentric_samples
            if sample.smoothed_velocity_mps > 0
        ]
        mean_concentric_velocity = (
            sum(positive_velocities) / len(positive_velocities)
            if positive_velocities
            else 0.0
        )

        movement_start_frame = (
            rep.eccentric_start_frame
            if rep.eccentric_start_frame is not None
            else rep.start_frame
        )
        duration_seconds = max(0.0, (rep.end_frame - movement_start_frame) / self._fps)
        concentric_seconds = max(0.0, (rep.lockout_frame - rep.start_frame) / self._fps)
        if rep.eccentric_start_frame is not None:
            eccentric_seconds = max(0.0, (rep.start_frame - rep.eccentric_start_frame) / self._fps)
        else:
            eccentric_seconds = max(0.0, (rep.end_frame - rep.lockout_frame) / self._fps)

        accepted_history = [
            report.mean_concentric_velocity_mps
            for report in self._built_reports
            if report.validation_status == "accepted"
        ]
        best_velocity = max(
            accepted_history + [mean_concentric_velocity],
            default=mean_concentric_velocity,
        )
        previous_velocity = (
            accepted_history[-1]
            if accepted_history
            else mean_concentric_velocity
        )
        loss_from_best = _velocity_loss_percent(mean_concentric_velocity, best_velocity)
        loss_from_previous = _velocity_loss_percent(mean_concentric_velocity, previous_velocity)
        warning = loss_from_best >= self._velocity_loss_threshold_percent

        report = RepReport(
            rep_index=rep.rep_index,
            start_frame=rep.start_frame,
            lockout_frame=rep.lockout_frame,
            end_frame=rep.end_frame,
            duration_seconds=duration_seconds,
            concentric_seconds=concentric_seconds,
            eccentric_seconds=eccentric_seconds,
            rom_m=rep.displacement_m,
            mean_concentric_velocity_mps=mean_concentric_velocity,
            peak_velocity_mps=rep.peak_velocity_mps,
            velocity_loss_from_best_percent=loss_from_best,
            velocity_loss_from_previous_percent=loss_from_previous,
            velocity_loss_warning=warning,
            tracking_confidence_mean=self._compute_tracking_confidence_mean(rep_samples),
            hub_confidence_mean=self._compute_confidence_mean(rep_samples, "hub"),
            plate_confidence_mean=self._compute_confidence_mean(rep_samples, "plate"),
            tracking_source_pct=self._compute_source_pct(rep_samples),
            validation_status=validation_status,
            validation_reason=validation_reason,
        )
        self._built_reports.append(report)

        return report

    @staticmethod
    def _compute_confidence_mean(samples: list[KinematicSample], field: str) -> float:
        values = [
            getattr(s, f"{field}_confidence", 0.0)
            for s in samples
        ]
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _compute_tracking_confidence_mean(samples: list[KinematicSample]) -> float:
        if not samples:
            return 0.0
        values = [
            max(sample.hub_confidence, sample.plate_confidence)
            for sample in samples
        ]
        return sum(values) / len(values)

    @staticmethod
    def _compute_source_pct(samples: list[KinematicSample]) -> dict[str, float]:
        counts: dict[str, int] = {}
        for s in samples:
            src = s.tracking_source or "unknown"
            counts[src] = counts.get(src, 0) + 1
        total = max(1, sum(counts.values()))
        return {src: round(cnt / total * 100, 1) for src, cnt in counts.items()}


def write_json_report(report: AnalysisReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(asdict(report), file, indent=2)


def write_csv_report(reps: list[RepReport], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rep_index",
        "start_frame",
        "lockout_frame",
        "end_frame",
        "duration_seconds",
        "concentric_seconds",
        "eccentric_seconds",
        "rom_m",
        "mean_concentric_velocity_mps",
        "peak_velocity_mps",
        "velocity_loss_from_best_percent",
        "velocity_loss_from_previous_percent",
        "velocity_loss_warning",
        "tracking_confidence_mean",
        "hub_confidence_mean",
        "plate_confidence_mean",
        "tracking_source_pct",
        "validation_status",
        "validation_reason",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rep in reps:
            row = asdict(rep)
            row["tracking_source_pct"] = json.dumps(row.get("tracking_source_pct") or {}, sort_keys=True)
            writer.writerow(row)


def _velocity_loss_percent(current_velocity: float, reference_velocity: float) -> float:
    if reference_velocity <= 0:
        return 0.0

    return max(0.0, (reference_velocity - current_velocity) / reference_velocity * 100.0)
