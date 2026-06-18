from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_MODELS = (
    Path("models/powerai_bar_detector.pt"),
    Path("models/powerai_athlete_seg.pt"),
)


@dataclass(frozen=True)
class ExportResult:
    model_path: Path
    format_name: str
    export_path: Path | None
    size_mb: float | None
    latency_ms: float | None
    status: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export PowerAI YOLO models for mobile/edge targets."
    )
    parser.add_argument(
        "--model",
        dest="models",
        nargs="+",
        type=Path,
        default=list(DEFAULT_MODELS),
        help="Model paths to export.",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        nargs="+",
        default=["onnx", "tflite", "coreml"],
        help="Ultralytics export formats, e.g. onnx tflite coreml.",
    )
    parser.add_argument(
        "--imgsz",
        default=720,
        type=int,
        help="Export and benchmark image size.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for export/benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("models/exported"),
        type=Path,
        help="Directory for copied export artifacts.",
    )
    parser.add_argument(
        "--benchmark-runs",
        default=5,
        type=int,
        help="Number of PyTorch predict runs used for rough latency.",
    )
    return parser


def export_model(
    model_path: Path,
    formats: list[str],
    output_dir: Path,
    imgsz: int,
    device: str,
    benchmark_runs: int,
) -> list[ExportResult]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics is required for model export.") from exc

    if not model_path.exists():
        return [
            ExportResult(model_path, fmt, None, None, None, "missing model")
            for fmt in formats
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    latency_ms = _benchmark_predict_ms(model, imgsz=imgsz, runs=benchmark_runs, device=device)
    results: list[ExportResult] = []

    for fmt in formats:
        try:
            exported = Path(model.export(format=fmt, imgsz=imgsz, device=device))
            copied = output_dir / f"{model_path.stem}.{_extension_for_format(fmt, exported)}"
            if exported.exists() and exported.resolve() != copied.resolve():
                if exported.is_dir():
                    if copied.exists():
                        shutil.rmtree(copied)
                    shutil.copytree(exported, copied)
                else:
                    shutil.copy2(exported, copied)
            else:
                copied = exported
            size_mb = _artifact_size_bytes(copied) / (1024 * 1024)
            results.append(ExportResult(model_path, fmt, copied, size_mb, latency_ms, "ok"))
        except Exception as exc:
            results.append(ExportResult(model_path, fmt, None, None, latency_ms, f"failed: {exc}"))

    return results


def _benchmark_predict_ms(model: object, imgsz: int, runs: int, device: str) -> float | None:
    if runs <= 0:
        return None

    frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    timings: list[float] = []
    for index in range(runs + 1):
        start = time.perf_counter()
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if index > 0:
            timings.append(elapsed_ms)
    return sum(timings) / len(timings) if timings else None


def _artifact_size_bytes(path: Path) -> int:
    if path.is_dir():
        return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    return path.stat().st_size


def _extension_for_format(format_name: str, exported: Path) -> str:
    if exported.is_dir():
        return exported.suffix.lstrip(".") or format_name
    return exported.suffix.lstrip(".") or format_name


def main() -> None:
    args = build_parser().parse_args()
    all_results: list[ExportResult] = []
    for model_path in args.models:
        all_results.extend(
            export_model(
                model_path=model_path,
                formats=args.formats,
                output_dir=args.output_dir,
                imgsz=args.imgsz,
                device=args.device,
                benchmark_runs=args.benchmark_runs,
            )
        )

    for result in all_results:
        size = "n/a" if result.size_mb is None else f"{result.size_mb:.1f} MB"
        latency = "n/a" if result.latency_ms is None else f"{result.latency_ms:.1f} ms"
        export_path = "n/a" if result.export_path is None else str(result.export_path)
        print(
            f"{result.model_path} [{result.format_name}]: "
            f"{result.status}; size={size}; pytorch_latency={latency}; output={export_path}"
        )


if __name__ == "__main__":
    main()
