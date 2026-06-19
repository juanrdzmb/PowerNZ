"""Technique-evidence decisions for completed bar-path candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from metrics import CompletedRep, RepValidation


RepDecisionStatus = Literal["accepted", "review", "rejected"]


@dataclass(frozen=True)
class RepDecision:
    rep: CompletedRep
    status: RepDecisionStatus
    reason: str

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


def decide_rep_validations(
    base_validations: list[RepValidation],
    evidence: dict[int, tuple[bool | None, bool | None]],
    exercise: str,
    *,
    strict: bool = True,
    lockout_window_frames: int = 6,
) -> list[RepDecision]:
    """Turn mechanical candidates into accepted/review/rejected decisions.

    A false technical signal is a rejection.  Missing evidence is deliberately a
    review in strict mode rather than a guessed valid lift.
    """
    decisions: list[RepDecision] = []
    for validation in base_validations:
        rep = validation.rep
        if not validation.accepted:
            decisions.append(RepDecision(rep, "rejected", validation.reason))
            continue

        depth_required = exercise in {"squat", "bench"}
        depth_values = [
            evidence.get(frame_index, (None, None))[0]
            for frame_index in range(rep.start_frame, rep.lockout_frame + 1)
        ]
        lockout_values = [
            evidence.get(frame_index, (None, None))[1]
            for frame_index in range(
                max(rep.start_frame, rep.lockout_frame - lockout_window_frames),
                rep.lockout_frame + 1,
            )
        ]

        if depth_required:
            depth_decision = _gate_decision(depth_values, "profundidad", strict)
            if depth_decision is not None:
                decisions.append(RepDecision(rep, *depth_decision))
                continue
        lockout_decision = _gate_decision(lockout_values, "bloqueo", strict)
        if lockout_decision is not None:
            decisions.append(RepDecision(rep, *lockout_decision))
            continue
        decisions.append(RepDecision(rep, "accepted", "evidencia técnica suficiente"))
    return decisions


def _gate_decision(
    values: list[bool | None],
    name: str,
    strict: bool,
) -> tuple[RepDecisionStatus, str] | None:
    if any(value is True for value in values):
        return None
    if any(value is None for value in values):
        if strict:
            return "review", f"requiere revisión: evidencia insuficiente de {name}"
        return None
    return "rejected", f"rechazada: {name} no confirmado"
