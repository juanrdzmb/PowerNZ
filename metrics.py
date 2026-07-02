from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from track import OneEuroFilter


LiftState = Literal["reposo", "inicio", "tirón", "bloqueo", "bajada"]
ExerciseName = Literal["deadlift", "squat", "bench"]
MovementOrder = Literal["concentric_first", "eccentric_first"]


@dataclass(frozen=True)
class BiomechanicsConfig:
    upward_velocity_threshold_mps: float = 0.15
    downward_velocity_threshold_mps: float = -0.15
    lockout_velocity_threshold_mps: float = 0.20
    rest_velocity_threshold_mps: float = 0.15
    min_rep_displacement_m: float = 0.18
    min_rep_frames: int = 18
    lockout_hold_frames: int = 4
    rest_hold_frames: int = 4
    position_filter_min_cutoff: float = 1.0
    position_filter_beta: float = 0.015
    velocity_filter_min_cutoff: float = 1.1
    velocity_filter_beta: float = 0.02
    velocity_deadband_mps: float = 0.025
    max_abs_velocity_mps: float = 2.5
    max_reasonable_rep_displacement_m: float = 0.75
    min_gap_between_completed_reps_frames: int = 8
    velocity_outlier_window: int = 30
    velocity_outlier_mad_factor: float = 3.0


@dataclass(frozen=True)
class ExerciseProfile:
    """How a lift moves, so rep detection knows whether the bar goes up first
    (deadlift) or down first (squat/bench)."""

    name: ExerciseName
    movement_order: MovementOrder


EXERCISE_PROFILES: dict[str, ExerciseProfile] = {
    "deadlift": ExerciseProfile("deadlift", "concentric_first"),
    "squat": ExerciseProfile("squat", "eccentric_first"),
    "bench": ExerciseProfile("bench", "eccentric_first"),
}

# Per-exercise (min_rep_displacement_m, max_reasonable_rep_displacement_m) defaults.
# The bar travels different vertical ranges in each lift.
EXERCISE_DISPLACEMENT_DEFAULTS: dict[str, tuple[float, float]] = {
    "deadlift": (0.18, 0.75),
    "squat": (0.20, 1.00),
    "bench": (0.10, 0.60),
}


def get_exercise_profile(name: str) -> ExerciseProfile:
    try:
        return EXERCISE_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown exercise '{name}'. Use one of {sorted(EXERCISE_PROFILES)}.") from exc


@dataclass(frozen=True)
class KinematicSample:
    frame_index: int
    time_seconds: float
    position_m: float
    velocity_mps: float
    smoothed_velocity_mps: float
    state: LiftState
    rep_index: int
    rep_displacement_m: float
    hub_confidence: float = 0.0
    plate_confidence: float = 0.0
    tracking_source: str = "unknown"
    raw_velocity_mps: float = 0.0


@dataclass(frozen=True)
class CompletedRep:
    rep_index: int
    start_frame: int
    lockout_frame: int
    end_frame: int
    displacement_m: float
    peak_velocity_mps: float
    eccentric_start_frame: int | None = None

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class RepValidation:
    rep: CompletedRep
    accepted: bool
    reason: str


class VelocityEstimator:
    def __init__(
        self,
        fps: float,
        min_cutoff: float = 1.2,
        beta: float = 0.04,
        deadband_mps: float = 0.025,
        max_abs_velocity_mps: float = 4.0,
        extra_smoothing: float = 1.0,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be greater than zero.")

        if extra_smoothing < 0:
            raise ValueError("extra_smoothing must be non-negative.")

        self._fps = fps
        self._previous_position_m: float | None = None
        self._previous_time_seconds: float | None = None
        self._deadband_mps = deadband_mps
        self._max_abs_velocity_mps = max_abs_velocity_mps
        self._velocity_filter = OneEuroFilter(
            frequency_hz=fps,
            min_cutoff=max(0.05, min_cutoff * extra_smoothing),
            beta=beta,
        )

    def update(self, position_m: float, frame_index: int) -> tuple[float, float]:
        time_seconds = frame_index / self._fps

        if self._previous_position_m is None or self._previous_time_seconds is None:
            raw_velocity = 0.0
        else:
            dt = time_seconds - self._previous_time_seconds
            raw_velocity = 0.0 if dt <= 0 else (position_m - self._previous_position_m) / dt
            raw_velocity = max(
                -self._max_abs_velocity_mps,
                min(self._max_abs_velocity_mps, raw_velocity),
            )
            if abs(raw_velocity) < self._deadband_mps:
                raw_velocity = 0.0

        self._previous_position_m = position_m
        self._previous_time_seconds = time_seconds

        smoothed_velocity = self._velocity_filter.apply(raw_velocity)
        if abs(smoothed_velocity) < self._deadband_mps:
            smoothed_velocity = 0.0
        return raw_velocity, smoothed_velocity

    def set_min_cutoff(self, min_cutoff: float) -> None:
        self._velocity_filter.set_min_cutoff(min_cutoff)


class LiftStateMachine:
    def __init__(self, config: BiomechanicsConfig = BiomechanicsConfig()) -> None:
        self._config = config
        self._state: LiftState = "reposo"
        self._rep_index = 0
        self._active_start_frame: int | None = None
        self._active_start_position_m: float | None = None
        self._lockout_frame: int | None = None
        self._peak_velocity_mps = 0.0
        self._max_displacement_m = 0.0
        self._lockout_hold_count = 0
        self._rest_hold_count = 0
        self._bottom_position_m: float | None = None
        self._downward_before_lockout = False
        self._downward_before_lockout_frames = 0
        self.completed_reps: list[CompletedRep] = []

    @property
    def state(self) -> LiftState:
        return self._state

    @property
    def rep_index(self) -> int:
        return self._rep_index

    def update(
        self,
        frame_index: int,
        position_m: float,
        smoothed_velocity_mps: float,
        depth_ok: bool = True,
        lockout_ok: bool = True,
    ) -> tuple[LiftState, int, float]:
        if self._state == "reposo":
            self._update_bottom_position(position_m)
            self._update_reposo(frame_index, position_m, smoothed_velocity_mps)
        elif self._state == "inicio":
            self._update_inicio(position_m, smoothed_velocity_mps)
        elif self._state == "tirón":
            self._update_tiron(frame_index, position_m, smoothed_velocity_mps, lockout_ok)
        elif self._state == "bloqueo":
            self._update_bloqueo(smoothed_velocity_mps)
        elif self._state == "bajada":
            self._update_bajada(frame_index, position_m, smoothed_velocity_mps)

        return self._state, self._rep_index, self._current_displacement(position_m)

    def _update_reposo(
        self,
        frame_index: int,
        position_m: float,
        smoothed_velocity_mps: float,
    ) -> None:
        if smoothed_velocity_mps <= self._config.upward_velocity_threshold_mps:
            return

        self._state = "inicio"
        self._rep_index += 1
        self._active_start_frame = frame_index
        self._active_start_position_m = (
            self._bottom_position_m
            if self._bottom_position_m is not None
            else position_m
        )
        self._lockout_frame = None
        self._peak_velocity_mps = smoothed_velocity_mps
        self._max_displacement_m = 0.0
        self._lockout_hold_count = 0
        self._rest_hold_count = 0
        self._downward_before_lockout = False
        self._downward_before_lockout_frames = 0

    def _update_inicio(self, position_m: float, smoothed_velocity_mps: float) -> None:
        displacement = self._current_displacement(position_m)
        self._max_displacement_m = max(self._max_displacement_m, displacement)
        self._peak_velocity_mps = max(self._peak_velocity_mps, smoothed_velocity_mps)
        if smoothed_velocity_mps > self._config.upward_velocity_threshold_mps:
            self._state = "tirón"

    def _update_tiron(
        self,
        frame_index: int,
        position_m: float,
        smoothed_velocity_mps: float,
        lockout_ok: bool = True,
    ) -> None:
        self._peak_velocity_mps = max(self._peak_velocity_mps, smoothed_velocity_mps)
        displacement = self._current_displacement(position_m)
        self._max_displacement_m = max(self._max_displacement_m, displacement)
        descent_tolerance_m = max(0.04, self._config.min_rep_displacement_m * 0.16)
        downward_loss_m = self._max_displacement_m - displacement
        if self._lockout_frame is None and smoothed_velocity_mps < self._config.downward_velocity_threshold_mps:
            if downward_loss_m >= descent_tolerance_m - 1e-9:
                self._downward_before_lockout_frames += 1
            if self._downward_before_lockout_frames >= 2:
                self._downward_before_lockout = True
        elif downward_loss_m < descent_tolerance_m * 0.5:
            self._downward_before_lockout_frames = 0

        near_still = abs(smoothed_velocity_mps) <= self._config.lockout_velocity_threshold_mps
        enough_range = displacement >= self._config.min_rep_displacement_m
        # Reject false lockouts from a brief velocity dip early in the pull:
        # the concentric phase must have lasted at least half a min rep before we
        # trust a "still" reading as a real lockout.
        concentric_frames = frame_index - (
            self._active_start_frame
            if self._active_start_frame is not None
            else frame_index
        )
        mature_pull = concentric_frames >= max(1, self._config.min_rep_frames // 2)

        # IPF lockout plus a mature pull: only treat the top as valid when the joints
        # are extended and the velocity dip is not just an early false pause.
        if near_still and enough_range and mature_pull and lockout_ok and not self._downward_before_lockout:
            self._lockout_hold_count += 1
        else:
            self._lockout_hold_count = 0

        if self._lockout_hold_count >= self._config.lockout_hold_frames:
            self._state = "bloqueo"
            self._lockout_frame = frame_index

    def _update_bloqueo(self, smoothed_velocity_mps: float) -> None:
        if smoothed_velocity_mps < self._config.downward_velocity_threshold_mps:
            self._state = "bajada"
            self._rest_hold_count = 0

    def _update_bajada(
        self,
        frame_index: int,
        position_m: float,
        smoothed_velocity_mps: float,
    ) -> None:
        close_to_start = self._current_displacement(position_m) <= self._config.min_rep_displacement_m * 0.5
        slow_enough = abs(smoothed_velocity_mps) <= self._config.rest_velocity_threshold_mps
        starts_next_rep = smoothed_velocity_mps >= self._config.upward_velocity_threshold_mps

        # Completing the rep as soon as the next ascent begins lets multi-rep sets
        # (no full rest between reps) count each rep instead of collapsing to one.
        if starts_next_rep and self._max_displacement_m >= self._config.min_rep_displacement_m:
            self._complete_active_rep(frame_index)
            self._update_bottom_position(position_m)
            self._update_reposo(frame_index, position_m, smoothed_velocity_mps)
            return

        if close_to_start and slow_enough:
            self._rest_hold_count += 1
        else:
            self._rest_hold_count = 0

        if self._rest_hold_count < self._config.rest_hold_frames:
            return

        self._complete_active_rep(frame_index)
        self._state = "reposo"
        self._update_bottom_position(position_m)

    def finalize(self, frame_index: int) -> None:
        if self._state in {"reposo", "inicio"}:
            return

        if not self._can_complete_active_rep(frame_index):
            return

        self._complete_active_rep(frame_index)

    def _complete_active_rep(self, frame_index: int) -> None:
        if not self._can_complete_active_rep(frame_index):
            self._reset_active_rep()
            return

        self.completed_reps.append(
            CompletedRep(
                rep_index=self._rep_index,
                start_frame=(
                    self._active_start_frame
                    if self._active_start_frame is not None
                    else frame_index
                ),
                lockout_frame=(
                    self._lockout_frame
                    if self._lockout_frame is not None
                    else frame_index
                ),
                end_frame=frame_index,
                displacement_m=self._max_displacement_m,
                peak_velocity_mps=self._peak_velocity_mps,
            )
        )
        self._reset_active_rep()

    def _can_complete_active_rep(self, frame_index: int) -> bool:
        if self._active_start_frame is None:
            return False

        if self._lockout_frame is None:
            return False
        if self._downward_before_lockout:
            return False

        rep_frames = frame_index - self._active_start_frame
        if rep_frames < self._config.min_rep_frames:
            return False

        return self._max_displacement_m >= self._config.min_rep_displacement_m

    def _reset_active_rep(self) -> None:
        self._state = "reposo"
        self._active_start_frame = None
        self._active_start_position_m = None
        self._lockout_frame = None
        self._peak_velocity_mps = 0.0
        self._max_displacement_m = 0.0
        self._lockout_hold_count = 0
        self._rest_hold_count = 0
        self._downward_before_lockout = False
        self._downward_before_lockout_frames = 0

    def _current_displacement(self, position_m: float) -> float:
        if self._active_start_position_m is None:
            return 0.0

        return max(0.0, position_m - self._active_start_position_m)

    def _update_bottom_position(self, position_m: float) -> None:
        if self._bottom_position_m is None:
            self._bottom_position_m = position_m
            return

        self._bottom_position_m = min(self._bottom_position_m, position_m)


class EccentricFirstStateMachine:
    """Rep detection for lifts that start at the top and descend first (squat, bench).

    Cycle: reposo(top) -> bajada(eccentric) -> tiron(concentric ascent) -> bloqueo(top).
    Velocity metrics are measured on the concentric ascent, so it emits the same
    CompletedRep/KinematicSample shape as the deadlift machine and the rest of the
    pipeline (graph, telemetry, reporting) works unchanged. Up is positive velocity.
    """

    def __init__(self, config: BiomechanicsConfig = BiomechanicsConfig()) -> None:
        self._config = config
        self._state: LiftState = "reposo"
        self._rep_index = 0
        self._top_position_m: float | None = None
        self._bottom_position_m: float | None = None
        self._active_start_frame: int | None = None  # ascent start (bottom)
        self._eccentric_start_frame: int | None = None
        self._active_start_position_m: float | None = None
        self._lockout_frame: int | None = None  # ascent end (top)
        self._peak_velocity_mps = 0.0
        self._max_displacement_m = 0.0
        self._lockout_hold_count = 0
        self._depth_reached = False  # parallel reached during the current descent
        self._downward_during_ascent = False
        self._downward_during_ascent_frames = 0
        self.completed_reps: list[CompletedRep] = []

    @property
    def state(self) -> LiftState:
        return self._state

    @property
    def rep_index(self) -> int:
        return self._rep_index

    def update(
        self,
        frame_index: int,
        position_m: float,
        smoothed_velocity_mps: float,
        depth_ok: bool = True,
        lockout_ok: bool = True,
    ) -> tuple[LiftState, int, float]:
        if self._state == "reposo":
            self._update_top_position(position_m)
            self._update_reposo(frame_index, position_m, smoothed_velocity_mps)
        elif self._state == "bajada":
            self._update_bajada(frame_index, position_m, smoothed_velocity_mps, depth_ok)
        elif self._state == "tirón":
            self._update_tiron(frame_index, position_m, smoothed_velocity_mps, lockout_ok)
        elif self._state == "bloqueo":
            self._update_bloqueo(frame_index, position_m, smoothed_velocity_mps)

        return self._state, self._rep_index, self._current_displacement(position_m)

    def _update_reposo(
        self,
        frame_index: int,
        position_m: float,
        smoothed_velocity_mps: float,
    ) -> None:
        # Threshold is negative; descend once the bar moves down fast enough.
        if smoothed_velocity_mps >= self._config.downward_velocity_threshold_mps:
            return
        self._state = "bajada"
        self._eccentric_start_frame = frame_index
        self._bottom_position_m = position_m
        self._lockout_hold_count = 0
        self._depth_reached = False
        self._downward_during_ascent = False
        self._downward_during_ascent_frames = 0

    def _update_bajada(
        self,
        frame_index: int,
        position_m: float,
        smoothed_velocity_mps: float,
        depth_ok: bool = True,
    ) -> None:
        self._bottom_position_m = (
            position_m if self._bottom_position_m is None else min(self._bottom_position_m, position_m)
        )
        # IPF depth: remember once the lifter has reached parallel during this descent.
        # depth_ok defaults True when pose is unavailable, preserving bar-only behaviour.
        if depth_ok:
            self._depth_reached = True
        if smoothed_velocity_mps <= self._config.upward_velocity_threshold_mps:
            return  # still going down / not yet ascending

        top = self._top_position_m if self._top_position_m is not None else position_m
        bottom = self._bottom_position_m if self._bottom_position_m is not None else position_m
        depth = top - bottom
        if depth < self._config.min_rep_displacement_m * 0.5 or not self._depth_reached:
            self._state = "reposo"  # shallow dip / not deep enough, not a real rep
            self._eccentric_start_frame = None
            return

        self._rep_index += 1
        self._state = "tirón"
        self._active_start_frame = frame_index
        self._active_start_position_m = bottom
        self._lockout_frame = None
        self._peak_velocity_mps = smoothed_velocity_mps
        self._max_displacement_m = 0.0
        self._lockout_hold_count = 0
        self._downward_during_ascent = False
        self._downward_during_ascent_frames = 0

    def _update_tiron(
        self,
        frame_index: int,
        position_m: float,
        smoothed_velocity_mps: float,
        lockout_ok: bool = True,
    ) -> None:
        self._peak_velocity_mps = max(self._peak_velocity_mps, smoothed_velocity_mps)
        displacement = self._current_displacement(position_m)
        self._max_displacement_m = max(self._max_displacement_m, displacement)
        descent_tolerance_m = max(0.035, self._config.min_rep_displacement_m * 0.16)
        downward_loss_m = self._max_displacement_m - displacement
        if self._lockout_frame is None and smoothed_velocity_mps < self._config.downward_velocity_threshold_mps:
            if downward_loss_m >= descent_tolerance_m - 1e-9:
                self._downward_during_ascent_frames += 1
            if self._downward_during_ascent_frames >= 2:
                self._downward_during_ascent = True
        elif downward_loss_m < descent_tolerance_m * 0.5:
            self._downward_during_ascent_frames = 0

        near_still = abs(smoothed_velocity_mps) <= self._config.lockout_velocity_threshold_mps
        enough_range = displacement >= self._config.min_rep_displacement_m
        concentric_frames = frame_index - (
            self._active_start_frame
            if self._active_start_frame is not None
            else frame_index
        )
        mature_pull = concentric_frames >= max(1, self._config.min_rep_frames // 2)

        # IPF lockout at the top plus a mature ascent: squat standing erect / bench arms
        # extended, without counting a brief early stall as lockout.
        if near_still and enough_range and mature_pull and lockout_ok and not self._downward_during_ascent:
            self._lockout_hold_count += 1
        else:
            self._lockout_hold_count = 0

        if self._lockout_hold_count >= self._config.lockout_hold_frames:
            self._state = "bloqueo"
            self._lockout_frame = frame_index

    def _update_bloqueo(
        self,
        frame_index: int,
        position_m: float,
        smoothed_velocity_mps: float,
    ) -> None:
        self._complete_active_rep(frame_index)
        self._top_position_m = position_m
        if smoothed_velocity_mps < self._config.downward_velocity_threshold_mps:
            self._state = "bajada"
            self._bottom_position_m = position_m
            self._depth_reached = False  # fresh depth gate for the next rep
        else:
            self._state = "reposo"

    def finalize(self, frame_index: int) -> None:
        if self._state in {"reposo", "bajada"}:
            return
        if not self._can_complete_active_rep(frame_index):
            return
        self._complete_active_rep(frame_index)

    def _complete_active_rep(self, frame_index: int) -> None:
        if not self._can_complete_active_rep(frame_index):
            self._reset_active_rep()
            return

        self.completed_reps.append(
            CompletedRep(
                rep_index=self._rep_index,
                start_frame=(
                    self._active_start_frame
                    if self._active_start_frame is not None
                    else frame_index
                ),
                lockout_frame=(
                    self._lockout_frame
                    if self._lockout_frame is not None
                    else frame_index
                ),
                end_frame=frame_index,
                displacement_m=self._max_displacement_m,
                peak_velocity_mps=self._peak_velocity_mps,
                eccentric_start_frame=self._eccentric_start_frame,
            )
        )
        self._reset_active_rep()

    def _can_complete_active_rep(self, frame_index: int) -> bool:
        if self._active_start_frame is None:
            return False
        if self._lockout_frame is None:
            return False
        if self._downward_during_ascent:
            return False
        if frame_index - self._active_start_frame < self._config.min_rep_frames:
            return False
        return self._max_displacement_m >= self._config.min_rep_displacement_m

    def _reset_active_rep(self) -> None:
        self._active_start_frame = None
        self._eccentric_start_frame = None
        self._active_start_position_m = None
        self._lockout_frame = None
        self._peak_velocity_mps = 0.0
        self._max_displacement_m = 0.0
        self._lockout_hold_count = 0
        self._downward_during_ascent = False
        self._downward_during_ascent_frames = 0

    def _current_displacement(self, position_m: float) -> float:
        if self._active_start_position_m is None:
            return 0.0
        return max(0.0, position_m - self._active_start_position_m)

    def _update_top_position(self, position_m: float) -> None:
        if self._top_position_m is None:
            self._top_position_m = position_m
            return
        self._top_position_m = max(self._top_position_m, position_m)


class BiomechanicsEngine:
    def __init__(
        self,
        fps: float,
        config: BiomechanicsConfig = BiomechanicsConfig(),
        profile: ExerciseProfile | None = None,
    ) -> None:
        self._fps = fps
        self._config = config
        self._profile = profile or EXERCISE_PROFILES["deadlift"]
        self._velocity_estimator = VelocityEstimator(
            fps=fps,
            min_cutoff=config.velocity_filter_min_cutoff,
            beta=config.velocity_filter_beta,
            deadband_mps=config.velocity_deadband_mps,
            max_abs_velocity_mps=config.max_abs_velocity_mps,
            extra_smoothing=1.0,
        )
        self._position_filter = OneEuroFilter(
            frequency_hz=fps,
            min_cutoff=config.position_filter_min_cutoff,
            beta=config.position_filter_beta,
        )
        if self._profile.movement_order == "eccentric_first":
            self._state_machine: LiftStateMachine | EccentricFirstStateMachine = (
                EccentricFirstStateMachine(config)
            )
        else:
            self._state_machine = LiftStateMachine(config)
        self._velocity_window: list[float] = []
        self._last_filtered_position_m: float | None = None
        self._last_smoothed_velocity_mps: float = 0.0
        self._concentric_peak_position_m: float | None = None

    @property
    def completed_reps(self) -> list[CompletedRep]:
        return self._state_machine.completed_reps

    @property
    def validated_reps(self) -> list[CompletedRep]:
        return [
            validation.rep
            for validation in self.validate_reps()
            if validation.accepted
        ]

    def validate_reps(self) -> list[RepValidation]:
        validations: list[RepValidation] = []
        last_accepted_end_frame: int | None = None

        for rep in self.completed_reps:
            if rep.displacement_m > self._config.max_reasonable_rep_displacement_m:
                validations.append(
                    RepValidation(
                        rep=rep,
                        accepted=False,
                        reason="rejected: displacement too large for a single rep",
                    )
                )
                continue

            if (
                last_accepted_end_frame is not None
                and rep.start_frame - last_accepted_end_frame
                < self._config.min_gap_between_completed_reps_frames
            ):
                validations.append(
                    RepValidation(
                        rep=rep,
                        accepted=False,
                        reason="rejected: starts too close to previous accepted rep",
                    )
                )
                continue

            validations.append(
                RepValidation(rep=rep, accepted=True, reason="accepted")
            )
            last_accepted_end_frame = rep.end_frame

        return validations

    def update(
        self,
        frame_index: int,
        vertical_position_m: float,
        hub_confidence: float = 0.0,
        plate_confidence: float = 0.0,
        tracking_source: str = "unknown",
        depth_ok: bool = True,
        lockout_ok: bool = True,
    ) -> KinematicSample:
        filtered_position_m = self._position_filter.apply(vertical_position_m)

        if self._is_position_outlier(filtered_position_m):
            velocity_mps = 0.0
            smoothed_velocity_mps = self._last_smoothed_velocity_mps
        else:
            velocity_mps, smoothed_velocity_mps = self._velocity_estimator.update(
                position_m=filtered_position_m,
                frame_index=frame_index,
            )
            self._last_smoothed_velocity_mps = smoothed_velocity_mps

        self._last_filtered_position_m = filtered_position_m
        self._record_velocity(smoothed_velocity_mps)

        state, rep_index, rep_displacement_m = self._state_machine.update(
            frame_index=frame_index,
            position_m=filtered_position_m,
            smoothed_velocity_mps=smoothed_velocity_mps,
            depth_ok=depth_ok,
            lockout_ok=lockout_ok,
        )

        return KinematicSample(
            frame_index=frame_index,
            time_seconds=frame_index / self._fps,
            position_m=filtered_position_m,
            velocity_mps=velocity_mps,
            smoothed_velocity_mps=smoothed_velocity_mps,
            state=state,
            rep_index=rep_index,
            rep_displacement_m=rep_displacement_m,
            hub_confidence=hub_confidence,
            plate_confidence=plate_confidence,
            tracking_source=tracking_source,
            raw_velocity_mps=velocity_mps,
        )

    def update_reconstructed(
        self,
        frame_index: int,
        position_m: float,
        velocity_mps: float,
        hub_confidence: float = 0.0,
        plate_confidence: float = 0.0,
        tracking_source: str = "offline",
    ) -> KinematicSample:
        """Replay a completed, zero-phase trajectory through the rep FSM.

        Technical evidence is deliberately evaluated after the mechanical replay
        so unknown camera geometry produces a review rather than silently
        preventing a candidate from being reported at all.
        """
        raw_velocity_mps = velocity_mps
        velocity_mps = self._stabilize_reconstructed_velocity(position_m, velocity_mps)
        previous_state = self._state_machine.state
        self._last_filtered_position_m = position_m
        self._last_smoothed_velocity_mps = velocity_mps
        self._record_velocity(velocity_mps)
        state, rep_index, rep_displacement_m = self._state_machine.update(
            frame_index=frame_index,
            position_m=position_m,
            smoothed_velocity_mps=velocity_mps,
            depth_ok=True,
            lockout_ok=True,
        )
        if state in {"inicio", "tirón"}:
            if previous_state not in {"inicio", "tirón"} or self._concentric_peak_position_m is None:
                self._concentric_peak_position_m = position_m
            else:
                self._concentric_peak_position_m = max(self._concentric_peak_position_m, position_m)
        elif previous_state in {"inicio", "tirón"}:
            self._concentric_peak_position_m = None
        return KinematicSample(
            frame_index=frame_index,
            time_seconds=frame_index / self._fps,
            position_m=position_m,
            velocity_mps=velocity_mps,
            smoothed_velocity_mps=velocity_mps,
            state=state,
            rep_index=rep_index,
            rep_displacement_m=rep_displacement_m,
            hub_confidence=hub_confidence,
            plate_confidence=plate_confidence,
            tracking_source=tracking_source,
            raw_velocity_mps=raw_velocity_mps,
        )

    def _stabilize_reconstructed_velocity(self, position_m: float, velocity_mps: float) -> float:
        """Suppress tiny sign reversals while the bar is still at its highest point.

        This does not hide a real descent: once position has fallen by a meaningful
        amount the negative velocity is passed through.  It only removes the common
        one/two-frame detector wobble that made the graph dip before lockout.
        """
        if abs(velocity_mps) < self._config.velocity_deadband_mps:
            return 0.0
        if (
            self._state_machine.state == "bloqueo"
            and abs(velocity_mps) <= self._config.lockout_velocity_threshold_mps
        ):
            return 0.0
        if self._state_machine.state not in {"inicio", "tirón"} or velocity_mps >= 0:
            return velocity_mps
        peak = self._concentric_peak_position_m
        if peak is None:
            return velocity_mps
        reversal_tolerance_m = max(0.015, self._config.min_rep_displacement_m * 0.06)
        if peak - position_m < reversal_tolerance_m:
            return 0.0
        return velocity_mps

    def finalize(self, frame_index: int) -> None:
        self._state_machine.finalize(frame_index)

    def set_velocity_smoothing(self, extra_smoothing: float) -> None:
        if extra_smoothing < 0:
            raise ValueError("extra_smoothing must be non-negative.")

        self._velocity_estimator.set_min_cutoff(
            max(0.05, self._config.velocity_filter_min_cutoff * extra_smoothing)
        )

    def _is_velocity_outlier(self, value: float) -> bool:
        if len(self._velocity_window) < 10:
            return False
        window = list(self._velocity_window)
        sorted_window = sorted(window)
        median = sorted_window[len(sorted_window) // 2]
        deviations = [abs(v - median) for v in window]
        sorted_deviations = sorted(deviations)
        mad = sorted_deviations[len(sorted_deviations) // 2]
        if mad < 1e-6:
            return abs(value - median) > 0.5
        return abs(value - median) > self._config.velocity_outlier_mad_factor * mad

    def _is_position_outlier(self, position_m: float) -> bool:
        if self._last_filtered_position_m is None:
            return False
        max_jump_m = self._config.max_abs_velocity_mps / self._fps
        return abs(position_m - self._last_filtered_position_m) > max_jump_m

    def _record_velocity(self, value: float) -> None:
        self._velocity_window.append(value)
        if len(self._velocity_window) > self._config.velocity_outlier_window:
            self._velocity_window = self._velocity_window[-self._config.velocity_outlier_window :]
