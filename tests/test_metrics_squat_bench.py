"""Exercise-aware rep detection for eccentric-first lifts (squat, bench)."""

from __future__ import annotations

import numpy as np

from metrics import (
    EXERCISE_DISPLACEMENT_DEFAULTS,
    BiomechanicsConfig,
    BiomechanicsEngine,
    EccentricFirstStateMachine,
    LiftStateMachine,
    get_exercise_profile,
)


def _run(positions: list[float], exercise: str, fps: float = 30.0) -> BiomechanicsEngine:
    profile = get_exercise_profile(exercise)
    min_d, max_d = EXERCISE_DISPLACEMENT_DEFAULTS[exercise]
    config = BiomechanicsConfig(min_rep_displacement_m=min_d, max_reasonable_rep_displacement_m=max_d)
    engine = BiomechanicsEngine(fps=fps, config=config, profile=profile)
    samples = []
    for index, position in enumerate(positions):
        samples.append(engine.update(frame_index=index, vertical_position_m=float(position)))
    engine.finalize(len(positions))
    engine._samples = samples  # type: ignore[attr-defined]
    return engine


def _run_with_gates(
    positions: list[float],
    exercise: str,
    depth_ok: list[bool] | bool = True,
    lockout_ok: list[bool] | bool = True,
    fps: float = 30.0,
) -> BiomechanicsEngine:
    profile = get_exercise_profile(exercise)
    min_d, max_d = EXERCISE_DISPLACEMENT_DEFAULTS[exercise]
    config = BiomechanicsConfig(min_rep_displacement_m=min_d, max_reasonable_rep_displacement_m=max_d)
    engine = BiomechanicsEngine(fps=fps, config=config, profile=profile)
    for index, position in enumerate(positions):
        depth_flag = depth_ok[index] if isinstance(depth_ok, list) else depth_ok
        lockout_flag = lockout_ok[index] if isinstance(lockout_ok, list) else lockout_ok
        engine.update(
            frame_index=index,
            vertical_position_m=float(position),
            depth_ok=depth_flag,
            lockout_ok=lockout_flag,
        )
    engine.finalize(len(positions))
    return engine


def _down_then_up(top: float, bottom: float, rest: int = 12, span: int = 22, hold: int = 16) -> list[float]:
    return (
        [top] * rest
        + list(np.linspace(top, bottom, span))
        + list(np.linspace(bottom, top, span))
        + [top] * hold
    )


def test_engine_selects_eccentric_machine_for_squat_and_bench():
    assert isinstance(BiomechanicsEngine(30.0, profile=get_exercise_profile("squat"))._state_machine, EccentricFirstStateMachine)
    assert isinstance(BiomechanicsEngine(30.0, profile=get_exercise_profile("bench"))._state_machine, EccentricFirstStateMachine)
    assert isinstance(BiomechanicsEngine(30.0, profile=get_exercise_profile("deadlift"))._state_machine, LiftStateMachine)
    assert isinstance(BiomechanicsEngine(30.0)._state_machine, LiftStateMachine)  # default deadlift


def test_squat_detects_one_rep_with_positive_concentric_velocity():
    engine = _run(_down_then_up(top=1.0, bottom=0.45), exercise="squat")
    reps = engine.validated_reps
    assert len(reps) == 1
    rep = reps[0]
    assert 0.45 <= rep.displacement_m <= 0.65  # ~0.55 m ROM
    assert rep.peak_velocity_mps > 0.2  # concentric (ascending) velocity is positive

    # The ascending phase must report upward (positive) smoothed velocity.
    ascending = [s for s in engine._samples if s.state == "tirón"]
    assert ascending and max(s.smoothed_velocity_mps for s in ascending) > 0.2


def test_bench_detects_one_rep_with_smaller_rom():
    engine = _run(_down_then_up(top=1.2, bottom=0.95), exercise="bench")
    reps = engine.validated_reps
    assert len(reps) == 1
    assert 0.18 <= reps[0].displacement_m <= 0.35  # ~0.25 m bench ROM


def test_bench_rejects_partial_without_enough_bar_range():
    positions = _down_then_up(top=1.2, bottom=1.17)
    engine = _run_with_gates(positions, exercise="bench", depth_ok=True, lockout_ok=True)

    assert engine.validated_reps == []


def test_bench_rejects_rep_without_bottom_depth():
    positions = _down_then_up(top=1.2, bottom=0.95)
    engine = _run_with_gates(positions, exercise="bench", depth_ok=False, lockout_ok=True)

    assert engine.validated_reps == []


def test_bench_rejects_rep_without_elbow_lockout():
    positions = _down_then_up(top=1.2, bottom=0.95)
    engine = _run_with_gates(positions, exercise="bench", depth_ok=True, lockout_ok=False)

    assert engine.validated_reps == []


def test_bench_counts_full_rep_with_depth_and_lockout():
    positions = _down_then_up(top=1.2, bottom=0.95)
    lockout_flags = [position >= 1.18 for position in positions]
    engine = _run_with_gates(
        positions,
        exercise="bench",
        depth_ok=True,
        lockout_ok=lockout_flags,
    )

    assert len(engine.validated_reps) == 1


def test_two_squat_reps_are_counted():
    # Real reps have a clear pause at the top between them; give the filters time to settle.
    cycle = _down_then_up(top=1.0, bottom=0.45, rest=18, span=22, hold=24)
    engine = _run(cycle + cycle, exercise="squat")
    assert len(engine.validated_reps) == 2


def test_squat_rejects_depth_without_final_lockout():
    positions = _down_then_up(top=1.0, bottom=0.45)
    engine = _run_with_gates(positions, exercise="squat", depth_ok=True, lockout_ok=False)

    assert engine.validated_reps == []
