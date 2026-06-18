from pathlib import Path

from validation_outputs import create_validation_run


def test_create_validation_run_builds_expected_structure(tmp_path: Path) -> None:
    run = create_validation_run("My Check", base_dir=tmp_path / "runs")

    assert run.run_id.endswith("_my_check")
    assert run.videos_dir.exists()
    assert run.screenshots_dir.exists()
    assert run.reports_dir.exists()
    assert run.videos_dir.parent == run.root
    assert run.screenshots_dir.parent == run.root
    assert run.reports_dir.parent == run.root
