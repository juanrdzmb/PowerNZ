from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2


VIDEO_PATTERNS = {
    "deadlift": ("peso_muerto_*.mp4",),
    "squat": ("sentadilla*.mp4", "Sentadilla*.mp4", "Sentadila*.mp4"),
    "bench": ("banca*.mp4", "bench*.mp4", "press_banca*.mp4", "Press_banca*.mp4"),
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass(frozen=True)
class ExtractedFrame:
    filename: str
    source_video: str
    source_path: str
    frame_index: int
    timestamp_s: float
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class DatasetStats:
    exercise: str
    work_root: str
    videos: int
    frames: int
    labels: int
    masks: int
    zip_path: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare an isolated PowerAI dataset package for Kaggle training."
    )
    parser.add_argument(
        "--stage",
        choices=("frames", "package"),
        required=True,
        help="'frames' extracts and zips frames. 'package' zips corrected labels/masks.",
    )
    parser.add_argument(
        "--exercise",
        choices=tuple(VIDEO_PATTERNS),
        default="deadlift",
        help="Exercise dataset to prepare. Default: deadlift.",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path.home() / "Documents" / "entrenamiento",
        help="Folder containing training videos. Default: ~/Documents/entrenamiento.",
    )
    parser.add_argument(
        "--any-video-name",
        action="store_true",
        help="Use every video file in --videos-dir instead of exercise filename patterns.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="Output work folder. Default: training_cloud_kit/work/<exercise>_v1.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=180,
        help="Maximum frames sampled from each video. Default: 180.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=0,
        help="Extract every Nth frame. 0 uses adaptive sampling per video.",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=1280,
        help="Resize frames so the longest side is at most this value. 0 keeps original size.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality for extracted frames. Default: 92.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear generated work folders before extracting frames.",
    )
    parser.add_argument(
        "--allow-empty-labels",
        action="store_true",
        help="Allow packaging without labels. Useful only for debugging.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    kit_root = Path(__file__).resolve().parent
    work_root = _resolve_work_root(args.work_root, kit_root, args.exercise)
    work_root_explicit = args.work_root is not None

    if args.stage == "frames":
        stats = prepare_frame_package(
            exercise=args.exercise,
            videos_dir=args.videos_dir,
            work_root=work_root,
            kit_root=kit_root,
            work_root_explicit=work_root_explicit,
            any_video_name=args.any_video_name,
            max_frames_per_video=args.max_frames_per_video,
            stride=args.stride,
            max_dimension=args.max_dimension,
            jpeg_quality=args.jpeg_quality,
            overwrite=args.overwrite,
        )
    else:
        stats = package_corrected_dataset(
            exercise=args.exercise,
            work_root=work_root,
            allow_empty_labels=args.allow_empty_labels,
        )

    print(json.dumps(asdict(stats), indent=2))
    if stats.zip_path:
        print(f"\nReady to upload to Kaggle: {stats.zip_path}")


def prepare_frame_package(
    *,
    exercise: str,
    videos_dir: Path,
    work_root: Path,
    kit_root: Path,
    work_root_explicit: bool,
    any_video_name: bool,
    max_frames_per_video: int,
    stride: int,
    max_dimension: int,
    jpeg_quality: int,
    overwrite: bool,
) -> DatasetStats:
    videos = discover_videos(videos_dir, exercise, any_video_name=any_video_name)
    if not videos:
        patterns = ", ".join(VIDEO_PATTERNS[exercise])
        raise FileNotFoundError(f"No videos found in {videos_dir} for patterns: {patterns}")

    if overwrite:
        _clear_generated_work(work_root, kit_root, work_root_explicit=work_root_explicit)
    elif _has_images(work_root / "frames"):
        raise RuntimeError(
            f"Frames already exist in {work_root / 'frames'}. "
            "Use --overwrite to regenerate them."
        )

    _ensure_dataset_dirs(work_root)
    frames = extract_frames(
        videos=videos,
        frames_dir=work_root / "frames",
        max_frames_per_video=max_frames_per_video,
        stride=stride,
        max_dimension=max_dimension,
        jpeg_quality=jpeg_quality,
    )
    _write_manifest(work_root / "manifest.csv", frames)
    _write_dataset_files(work_root=work_root, exercise=exercise, frames=len(frames), videos=videos)
    zip_path = work_root / f"powerai_{exercise}_v1_frames.zip"
    _zip_dataset(
        root=work_root,
        zip_path=zip_path,
        include_labels=False,
        include_masks=False,
    )
    return _stats(exercise=exercise, work_root=work_root, videos=len(videos), zip_path=zip_path)


def package_corrected_dataset(
    *,
    exercise: str,
    work_root: Path,
    allow_empty_labels: bool,
) -> DatasetStats:
    _ensure_dataset_dirs(work_root)
    labels = _count_files(work_root / "labels", {".txt"})
    if labels == 0 and not allow_empty_labels:
        raise RuntimeError(
            f"No labels found in {work_root / 'labels'}. "
            "Download/correct labels first, or pass --allow-empty-labels for a dry package."
        )

    frame_count = _count_files(work_root / "frames", IMAGE_EXTENSIONS)
    if frame_count == 0:
        raise RuntimeError(f"No frames found in {work_root / 'frames'}. Run --stage frames first.")

    _write_dataset_files(work_root=work_root, exercise=exercise, frames=frame_count, videos=[])
    zip_path = work_root / f"powerai_{exercise}_v1_corrected.zip"
    _zip_dataset(
        root=work_root,
        zip_path=zip_path,
        include_labels=True,
        include_masks=_count_files(work_root / "masks", {".txt"}) > 0,
    )
    return _stats(exercise=exercise, work_root=work_root, videos=0, zip_path=zip_path)


def discover_videos(videos_dir: Path, exercise: str, *, any_video_name: bool = False) -> list[Path]:
    videos: list[Path] = []
    if any_video_name:
        videos.extend(path for path in videos_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)
    else:
        for pattern in VIDEO_PATTERNS[exercise]:
            videos.extend(sorted(videos_dir.glob(pattern)))
    return sorted({path.resolve() for path in videos if path.is_file()})


def extract_frames(
    *,
    videos: list[Path],
    frames_dir: Path,
    max_frames_per_video: int,
    stride: int,
    max_dimension: int,
    jpeg_quality: int,
) -> list[ExtractedFrame]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[ExtractedFrame] = []
    for video_path in videos:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        sampling_stride = stride if stride > 0 else max(1, total_frames // max_frames_per_video)
        saved_from_video = 0

        for frame_index in range(0, total_frames, sampling_stride):
            if saved_from_video >= max_frames_per_video:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue

            frame = _resize_to_max_dimension(frame, max_dimension)
            height, width = frame.shape[:2]
            filename = f"{video_path.stem}_frame_{frame_index:06d}.jpg"
            output_path = frames_dir / filename
            cv2.imwrite(
                str(output_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, jpeg_quality))],
            )
            extracted.append(
                ExtractedFrame(
                    filename=filename,
                    source_video=video_path.name,
                    source_path=str(video_path),
                    frame_index=frame_index,
                    timestamp_s=frame_index / fps if fps > 0 else 0.0,
                    width=width,
                    height=height,
                    fps=fps,
                )
            )
            saved_from_video += 1

        capture.release()
        print(
            f"{video_path.name}: {saved_from_video} frames "
            f"(stride={sampling_stride}, fps={fps:.2f})"
        )

    return extracted


def _resolve_work_root(work_root: Path | None, kit_root: Path, exercise: str) -> Path:
    if work_root is None:
        return (kit_root / "work" / f"{exercise}_v1").resolve()
    return work_root.resolve()


def _ensure_dataset_dirs(work_root: Path) -> None:
    for name in ("frames", "labels", "masks", "review"):
        (work_root / name).mkdir(parents=True, exist_ok=True)


def _clear_generated_work(work_root: Path, kit_root: Path, *, work_root_explicit: bool = False) -> None:
    allowed_root = (kit_root / "work").resolve()
    resolved = work_root.resolve()
    # Only the generated subfolders and packaging artifacts below are ever removed;
    # source videos sitting next to them are never touched. An explicitly provided
    # --work-root is trusted (e.g. per-exercise folders under Documents/Entrenamiento),
    # otherwise stay inside training_cloud_kit/work as a guardrail.
    if not work_root_explicit and not resolved.is_relative_to(allowed_root):
        raise RuntimeError(f"Refusing to clear outside {allowed_root}: {resolved}")

    for name in ("frames", "labels", "masks", "review"):
        shutil.rmtree(resolved / name, ignore_errors=True)
    for pattern in ("*.zip", "manifest.csv", "training_manifest.json", "dataset_*.yaml", "README_DATASET.md"):
        for path in resolved.glob(pattern):
            if path.is_file():
                path.unlink()


def _resize_to_max_dimension(frame, max_dimension: int):
    if max_dimension <= 0:
        return frame
    height, width = frame.shape[:2]
    longest = max(width, height)
    if longest <= max_dimension:
        return frame
    scale = max_dimension / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _write_manifest(path: Path, frames: list[ExtractedFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ExtractedFrame.__dataclass_fields__))
        writer.writeheader()
        for frame in frames:
            writer.writerow(asdict(frame))


def _write_dataset_files(*, work_root: Path, exercise: str, frames: int, videos: list[Path]) -> None:
    (work_root / "dataset_bar.yaml").write_text(
        "\n".join(
            [
                "path: .",
                "train: frames",
                "val: frames",
                "nc: 2",
                "names:",
                "  0: plate",
                "  1: bar_hub",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (work_root / "dataset_athlete.yaml").write_text(
        "\n".join(
            [
                "path: .",
                "train: frames",
                "val: frames",
                "nc: 2",
                "names:",
                "  0: athlete",
                "  1: background_person",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "dataset": f"powerai_{exercise}_v1",
        "exercise": exercise,
        "frames": frames,
        "source_videos": [str(path) for path in videos],
        "bar_classes": {"0": "plate", "1": "bar_hub"},
        "athlete_classes": {"0": "athlete", "1": "background_person"},
        "notes": [
            "Only label plates and hubs that belong to the loaded bar.",
            "Do not label plates on the floor or in the background.",
            "Keep negative frames when confusing objects appear.",
        ],
    }
    (work_root / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (work_root / "README_DATASET.md").write_text(
        "\n".join(
            [
                f"# PowerAI {exercise} v1 dataset",
                "",
                "Classes for detector:",
                "",
                "- 0: plate",
                "- 1: bar_hub",
                "",
                "Correction rules:",
                "",
                "- Label only the plate and hub on the athlete's loaded bar.",
                "- Leave floor/background plates unlabeled.",
                "- Keep hard negative frames in the dataset.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _zip_dataset(
    *,
    root: Path,
    zip_path: Path,
    include_labels: bool,
    include_masks: bool,
) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_root_name = root.name
    include_dirs = ["frames"]
    if include_labels:
        include_dirs.append("labels")
    if include_masks:
        include_dirs.append("masks")

    metadata_files = [
        "dataset_bar.yaml",
        "dataset_athlete.yaml",
        "manifest.csv",
        "training_manifest.json",
        "README_DATASET.md",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for dirname in include_dirs:
            directory = root / dirname
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(dataset_root_name) / path.relative_to(root))
        for filename in metadata_files:
            path = root / filename
            if path.exists():
                archive.write(path, Path(dataset_root_name) / filename)


def _stats(*, exercise: str, work_root: Path, videos: int, zip_path: Path | None) -> DatasetStats:
    return DatasetStats(
        exercise=exercise,
        work_root=str(work_root),
        videos=videos,
        frames=_count_files(work_root / "frames", IMAGE_EXTENSIONS),
        labels=_count_files(work_root / "labels", {".txt"}),
        masks=_count_files(work_root / "masks", {".txt"}),
        zip_path=str(zip_path) if zip_path else None,
    )


def _count_files(directory: Path, extensions: set[str]) -> int:
    if not directory.exists():
        return 0
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def _has_images(directory: Path) -> bool:
    return _count_files(directory, IMAGE_EXTENSIONS) > 0


if __name__ == "__main__":
    main()
