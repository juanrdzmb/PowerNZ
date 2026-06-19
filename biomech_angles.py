"""Joint-angle helpers for IPF-style rep validation.

The rep state machines in ``metrics.py`` only see the bar position/velocity, which
cannot tell a deep squat from a quarter squat or a real lockout from a partial pull.
These helpers turn the pose keypoints into the two gating booleans the FSM consumes:

- ``depth_ok``: the lifter has reached the required depth (squat parallel, knee <= ~90deg).
- ``lockout_ok``: the lifter is in a valid lockout (deadlift/squat standing erect, bench
  arms extended).

When the keypoints are not reliable enough to decide (low visibility, frontal view),
the corresponding flag is returned as ``None``. The default runtime treats that as
"not enough evidence" so it does not invent a valid IPF rep.
"""

from __future__ import annotations

from math import acos, degrees, hypot

from pose import PoseKeypoint, PoseResult


_MIN_VISIBILITY = 0.4

# Angle thresholds (degrees). Deliberately lenient so noisy 2D keypoints from a side
# view do not reject genuine reps.
SQUAT_PARALLEL_KNEE_MAX = 100.0   # knee angle at/below parallel (smaller = deeper)
LOCKOUT_KNEE_MIN = 165.0          # knee nearly straight when standing
LOCKOUT_HIP_MIN = 160.0           # hip nearly straight when standing
BENCH_ELBOW_LOCKOUT_MIN = 158.0   # arms extended at the top of a bench press
BENCH_BOTTOM_ELBOW_MAX = 145.0     # elbow flexed enough to be a real bench bottom


def _visible(keypoints: dict[str, PoseKeypoint], name: str) -> PoseKeypoint | None:
    keypoint = keypoints.get(name)
    if keypoint is None or keypoint.visibility < _MIN_VISIBILITY:
        return None
    return keypoint


def _joint_angle(
    a: PoseKeypoint | None,
    b: PoseKeypoint | None,
    c: PoseKeypoint | None,
) -> float | None:
    """Interior angle at vertex ``b`` formed by segments b->a and b->c, in degrees."""
    if a is None or b is None or c is None:
        return None

    bax, bay = a.x - b.x, a.y - b.y
    bcx, bcy = c.x - b.x, c.y - b.y
    length_a = hypot(bax, bay)
    length_c = hypot(bcx, bcy)
    if length_a <= 0 or length_c <= 0:
        return None

    cosine = (bax * bcx + bay * bcy) / (length_a * length_c)
    return float(degrees(acos(max(-1.0, min(1.0, cosine)))))


def _best_side_angle(
    keypoints: dict[str, PoseKeypoint],
    proximal: str,
    vertex: str,
    distal: str,
) -> float | None:
    """Compute the joint angle on whichever body side has all three keypoints visible.
    Prefer the side whose weakest keypoint is most confident."""
    best: tuple[float, float] | None = None  # (min_visibility, angle)
    for side in ("left", "right"):
        a = _visible(keypoints, f"{side}_{proximal}")
        b = _visible(keypoints, f"{side}_{vertex}")
        c = _visible(keypoints, f"{side}_{distal}")
        angle = _joint_angle(a, b, c)
        if angle is None:
            continue
        confidence = min(a.visibility, b.visibility, c.visibility)  # type: ignore[union-attr]
        if best is None or confidence > best[0]:
            best = (confidence, angle)
    return None if best is None else best[1]


def knee_angle_deg(keypoints: dict[str, PoseKeypoint]) -> float | None:
    """Knee flexion angle (hip-knee-ankle). ~180 standing, ~90 at squat parallel."""
    return _best_side_angle(keypoints, "hip", "knee", "ankle")


def hip_angle_deg(keypoints: dict[str, PoseKeypoint]) -> float | None:
    """Hip angle (shoulder-hip-knee). ~180 standing erect, smaller when hinged."""
    return _best_side_angle(keypoints, "shoulder", "hip", "knee")


def elbow_angle_deg(keypoints: dict[str, PoseKeypoint]) -> float | None:
    """Elbow angle (shoulder-elbow-wrist). ~180 when the press is locked out."""
    return _best_side_angle(keypoints, "shoulder", "elbow", "wrist")


def bench_elbow_depth_ok(keypoints: dict[str, PoseKeypoint]) -> bool | None:
    """Approximate the IPF bench depth rule with visible elbows and shoulders.

    In video coordinates, y grows downward. A valid bottom position requires the
    visible elbow joint to be level with or below its shoulder line. If both sides
    are visible, both must satisfy the rule; if only one side is visible, use that
    side conservatively instead of falling back to the bar path.
    """
    observed: list[bool] = []
    for side in ("left", "right"):
        shoulder = _visible(keypoints, f"{side}_shoulder")
        elbow = _visible(keypoints, f"{side}_elbow")
        wrist = _visible(keypoints, f"{side}_wrist")
        angle = _joint_angle(shoulder, elbow, wrist)
        if shoulder is None or elbow is None or wrist is None or angle is None:
            continue
        upper_arm_length = hypot(elbow.x - shoulder.x, elbow.y - shoulder.y)
        tolerance = max(6.0, upper_arm_length * 0.08)
        observed.append(elbow.y + tolerance >= shoulder.y and angle <= BENCH_BOTTOM_ELBOW_MAX)
    if not observed:
        return None
    return all(observed)


def _keypoints_by_name(pose: PoseResult | None) -> dict[str, PoseKeypoint]:
    if pose is None or not pose.detected or not pose.keypoints:
        return {}
    return {
        keypoint.name: keypoint
        for keypoint in pose.keypoints
        if keypoint.visibility >= _MIN_VISIBILITY
    }


def compute_ipf_flags(
    exercise: str,
    pose: PoseResult | None,
) -> tuple[bool | None, bool | None]:
    """Return ``(depth_ok, lockout_ok)`` for this frame, or ``None`` for either flag when
    the pose cannot decide (caller should then fall back to the bar heuristic).

    - squat: ``depth_ok`` when the knee reaches parallel; ``lockout_ok`` when knee and hip
      are extended at the top.
    - deadlift: no depth gate (``None``); ``lockout_ok`` when knee and hip are extended.
    - bench: ``depth_ok`` when the elbow has reached the shoulder line; ``lockout_ok`` when
      the elbows are extended.
    """
    keypoints = _keypoints_by_name(pose)
    if not keypoints:
        return None, None

    if exercise == "squat":
        knee = knee_angle_deg(keypoints)
        hip = hip_angle_deg(keypoints)
        depth_ok = None if knee is None else knee <= SQUAT_PARALLEL_KNEE_MAX
        if knee is None or hip is None:
            lockout_ok: bool | None = None
        else:
            lockout_ok = knee >= LOCKOUT_KNEE_MIN and hip >= LOCKOUT_HIP_MIN
        return depth_ok, lockout_ok

    if exercise == "deadlift":
        knee = knee_angle_deg(keypoints)
        hip = hip_angle_deg(keypoints)
        if knee is None or hip is None:
            return None, None
        return None, knee >= LOCKOUT_KNEE_MIN and hip >= LOCKOUT_HIP_MIN

    if exercise == "bench":
        elbow = elbow_angle_deg(keypoints)
        return (
            bench_elbow_depth_ok(keypoints),
            None if elbow is None else elbow >= BENCH_ELBOW_LOCKOUT_MIN,
        )

    return None, None
