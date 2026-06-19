from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ValidationSummary:
    models_zip: str
    detector_model: str
    athlete_model: str | None
    smoke_video: str | None
    smoke_output: str | None
    smoke_report: str | None
    frames_processed: int | None
    frames_tracked: int | None
    completed_reps: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate downloaded Kaggle models without integrating them into PowerNZ."
    )
    parser.add_argument(
        "--models-zip",
        type=Path,
        default=None,
        help="Path to PowerNZ_trained_models_v1.zip. Defaults to the newest matching ZIP in Downloads.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="PowerNZ project root.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "work" / "validation",
        help="Validation work folder inside the isolated kit.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path.home() / "Documents" / "entrenamiento" / "peso_muerto_8.mp4",
        help="Smoke-test video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=90,
        help="Maximum frames for the smoke test. Use 0 for full video.",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=720,
        help="Maximum output dimension for the smoke test.",
    )
    parser.add_argument(
        "--plate-diameter-px",
        type=float,
        default=120.0,
        help="Fallback plate diameter for the smoke test.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Only check that models exist and load with Ultralytics.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    kit_root = Path(__file__).resolve().parent
    models_zip = args.models_zip or find_latest_downloaded_zip()
    work_dir = args.work_dir.resolve()
    _reset_validation_dir(work_dir, kit_root)

    extracted_dir = work_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    extract_zip(models_zip, extracted_dir)

    detector = find_model(extracted_dir, "PowerNZ_bar_detector.pt")
    athlete = find_optional_model(extracted_dir, "PowerNZ_athlete_seg.pt")
    load_yolo_model(detector)
    if athlete is not None:
        load_yolo_model(athlete)

    smoke_output: Path | None = None
    smoke_report: Path | None = None
    smoke_data: dict[str, object] = {}
    if not args.skip_smoke:
        smoke_output, smoke_report = run_smoke(
            project_root=args.project_root.resolve(),
            work_dir=work_dir,
            video=args.video.resolve(),
            detector=detector,
            athlete=athlete,
            max_frames=args.max_frames,
            max_resolution=args.max_resolution,
            plate_diameter_px=args.plate_diameter_px,
        )
        smoke_data = read_smoke_report(smoke_report)

    summary = ValidationSummary(
        models_zip=str(models_zip),
        detector_model=str(detector),
        athlete_model=str(athlete) if athlete else None,
        smoke_video=str(args.video) if not args.skip_smoke else None,
        smoke_output=str(smoke_output) if smoke_output else None,
        smoke_report=str(smoke_report) if smoke_report else None,
        frames_processed=_int_or_none(smoke_data.get("frames_processed")),
        frames_tracked=_int_or_none(smoke_data.get("frames_tracked")),
        completed_reps=_int_or_none(smoke_data.get("completed_reps")),
    )
    summary_path = work_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    print(json.dumps(asdict(summary), indent=2))
    print(f"\nValidation summary: {summary_path}")


def find_latest_downloaded_zip() -> Path:
    candidates: list[Path] = []
    for directory in (Path.home() / "Downloads", Path.cwd()):
        if directory.exists():
            candidates.extend(directory.glob("PowerNZ_trained_models*.zip"))
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "No PowerNZ_trained_models*.zip found. Pass --models-zip with the Kaggle download."
        )
    return candidates[0]


def extract_zip(models_zip: Path, output_dir: Path) -> None:
    if not models_zip.exists():
        raise FileNotFoundError(f"Models ZIP not found: {models_zip}")
    with zipfile.ZipFile(models_zip) as archive:
        archive.extractall(output_dir)


def find_model(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Missing required model {filename} inside {root}")
    return matches[0]


def find_optional_model(root: Path, filename: str) -> Path | None:
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def load_yolo_model(model_path: Path) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics first: pip install ultralytics") from exc
    YOLO(str(model_path))
    print(f"Loaded model: {model_path}")


def run_smoke(
    *,
    project_root: Path,
    work_dir: Path,
    video: Path,
    detector: Path,
    athlete: Path | None,
    max_frames: int,
    max_resolution: int,
    plate_diameter_px: float,
) -> tuple[Path, Path]:
    if not video.exists():
        raise FileNotFoundError(f"Smoke video not found: {video}")
    output_path = work_dir / "smoke_PowerNZ_models.mp4"
    report_path = work_dir / "smoke_PowerNZ_models.json"
    command = [
        sys.executable,
        str(project_root / "main.py"),
        "--input",
        str(video),
        "--output",
        str(output_path),
        "--pose-backend",
        "yolo",
        "--object-model",
        str(detector),
        "--plate-diameter-px",
        str(plate_diameter_px),
        "--max-frames",
        str(max_frames),
        "--max-resolution",
        str(max_resolution),
        "--report-json",
        str(report_path),
        "--no-mobile-conversion",
    ]
    if athlete is not None:
        command.extend(["--segmentation-backend", "yolo-seg", "--segmentation-model", str(athlete)])

    subprocess.run(command, cwd=project_root, check=True)
    return output_path, report_path


def read_smoke_report(report_path: Path) -> dict[str, object]:
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


def _reset_validation_dir(work_dir: Path, kit_root: Path) -> None:
    allowed_root = (kit_root / "work").resolve()
    resolved = work_dir.resolve()
    if not resolved.is_relative_to(allowed_root):
        raise RuntimeError(f"Refusing to clear outside {allowed_root}: {resolved}")
    shutil.rmtree(resolved, ignore_errors=True)
    resolved.mkdir(parents=True, exist_ok=True)


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
