from metrics import CompletedRep, RepValidation
from rep_review import decide_rep_validations


def _validation() -> RepValidation:
    return RepValidation(
        rep=CompletedRep(1, start_frame=10, lockout_frame=20, end_frame=24, displacement_m=0.4, peak_velocity_mps=0.7),
        accepted=True,
        reason="accepted",
    )


def test_missing_lockout_evidence_requires_review() -> None:
    decisions = decide_rep_validations([_validation()], {index: (True, None) for index in range(10, 25)}, "squat")
    assert decisions[0].status == "review"


def test_false_depth_rejects_candidate() -> None:
    decisions = decide_rep_validations([_validation()], {index: (False, True) for index in range(10, 25)}, "bench")
    assert decisions[0].status == "rejected"


def test_full_evidence_accepts_candidate() -> None:
    evidence = {index: (True, True) for index in range(10, 25)}
    assert decide_rep_validations([_validation()], evidence, "squat")[0].status == "accepted"
