from metrics import CompletedRep, KinematicSample
from reporting import RepReportBuilder, write_csv_report


def test_rep_report_builder_calculates_rep_values() -> None:
    builder = RepReportBuilder(fps=10.0)
    builder.add_sample(KinematicSample(0, 0.0, 0.0, 0.0, 0.0, "tirón", 1, 0.0))
    builder.add_sample(KinematicSample(1, 0.1, 0.1, 1.0, 0.8, "tirón", 1, 0.1))
    builder.add_sample(KinematicSample(2, 0.2, 0.2, 1.0, 0.6, "bloqueo", 1, 0.2))
    builder.add_sample(KinematicSample(3, 0.3, 0.1, -1.0, -0.4, "bajada", 1, 0.1))

    report = builder.build_rep_report(CompletedRep(1, 0, 2, 3, 0.2, 0.8))

    assert report.rep_index == 1
    assert report.duration_seconds == 0.3
    assert report.concentric_seconds == 0.2
    assert report.eccentric_seconds == 0.1
    assert report.mean_concentric_velocity_mps == 0.7
    assert report.peak_velocity_mps == 0.8
    assert report.velocity_loss_from_best_percent == 0.0
    assert report.velocity_loss_from_previous_percent == 0.0
    assert report.velocity_loss_warning is False


def test_rep_report_builder_flags_velocity_loss() -> None:
    builder = RepReportBuilder(fps=10.0, velocity_loss_threshold_percent=20.0)
    samples = [
        KinematicSample(0, 0.0, 0.0, 0.0, 1.0, "tirón", 1, 0.0),
        KinematicSample(1, 0.1, 0.1, 0.0, 1.0, "tirón", 1, 0.1),
        KinematicSample(2, 0.2, 0.2, 0.0, 0.5, "tirón", 2, 0.0),
        KinematicSample(3, 0.3, 0.3, 0.0, 0.5, "tirón", 2, 0.1),
    ]
    for sample in samples:
        builder.add_sample(sample)

    first = builder.build_rep_report(CompletedRep(1, 0, 1, 1, 0.2, 1.0))
    second = builder.build_rep_report(CompletedRep(2, 2, 3, 3, 0.2, 0.5))

    assert first.velocity_loss_warning is False
    assert second.velocity_loss_from_best_percent == 50.0
    assert second.velocity_loss_from_previous_percent == 50.0
    assert second.velocity_loss_warning is True


def test_rep_report_builder_caches_existing_reports() -> None:
    builder = RepReportBuilder(fps=10.0)
    builder.add_sample(KinematicSample(0, 0.0, 0.0, 0.0, 1.0, "tirón", 1, 0.0))
    builder.add_sample(KinematicSample(1, 0.1, 0.1, 0.0, 1.0, "tirón", 1, 0.1))

    rep = CompletedRep(1, 0, 1, 1, 0.2, 1.0)
    first = builder.build_rep_report(rep)
    builder._samples.clear()
    second = builder.build_rep_report(rep)

    assert first is second


def test_rep_report_includes_tracking_confidence_and_writes_csv(tmp_path) -> None:
    builder = RepReportBuilder(fps=10.0)
    builder.add_sample(
        KinematicSample(
            0,
            0.0,
            0.0,
            0.0,
            0.4,
            "tirón",
            1,
            0.0,
            hub_confidence=0.8,
            plate_confidence=0.9,
            tracking_source="detection",
        )
    )

    report = builder.build_rep_report(CompletedRep(1, 0, 0, 0, 0.2, 0.4))
    output_path = tmp_path / "reps.csv"
    write_csv_report([report], output_path)

    assert report.tracking_confidence_mean == 0.9
    assert report.hub_confidence_mean == 0.8
    assert report.plate_confidence_mean == 0.9
    assert "tracking_source_pct" in output_path.read_text(encoding="utf-8")


def test_rep_report_tracking_confidence_uses_plate_when_hub_is_missing() -> None:
    builder = RepReportBuilder(fps=10.0)
    builder.add_sample(
        KinematicSample(
            0,
            0.0,
            0.0,
            0.0,
            0.4,
            "tirón",
            1,
            0.0,
            hub_confidence=0.0,
            plate_confidence=0.7,
            tracking_source="detection",
        )
    )

    report = builder.build_rep_report(CompletedRep(1, 0, 0, 0, 0.2, 0.4))

    assert report.tracking_confidence_mean == 0.7
    assert report.hub_confidence_mean == 0.0
    assert report.plate_confidence_mean == 0.7
