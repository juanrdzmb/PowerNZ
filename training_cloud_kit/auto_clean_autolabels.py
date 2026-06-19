from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAMES = {0: "plate", 1: "bar_hub"}


@dataclass(frozen=True)
class Box:
    cls: int
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def left(self) -> float:
        return self.cx - self.w / 2.0

    @property
    def right(self) -> float:
        return self.cx + self.w / 2.0

    @property
    def top(self) -> float:
        return self.cy - self.h / 2.0

    @property
    def bottom(self) -> float:
        return self.cy + self.h / 2.0


@dataclass(frozen=True)
class CleanStats:
    frames: int
    labels_in: int
    labels_out: int
    frames_with_labels: int
    frames_emptied: int
    synthetic_hubs: int
    output_zip: str
    preview_dir: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import and automatically clean Kaggle YOLO-World labels for PowerNZ."
    )
    parser.add_argument(
        "--review-zip",
        type=Path,
        default=None,
        help="PowerNZ_autolabel_review.zip. Defaults to the newest matching ZIP in Downloads.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(__file__).resolve().parent / "work" / "deadlift_v1",
        help="PowerNZ cloud kit work folder.",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path.home() / "Documents" / "entrenamiento",
        help="Training videos folder used if frames must be regenerated.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=180,
        help="Frames per video when regenerating missing frames.",
    )
    parser.add_argument(
        "--prepare-frames-if-missing",
        action="store_true",
        help="Regenerate frames with prepare_PowerNZ_cloud_dataset.py if work frames are missing.",
    )
    parser.add_argument(
        "--max-previews",
        type=int,
        default=160,
        help="Maximum cleaned preview images to write.",
    )
    parser.add_argument(
        "--min-plate-area",
        type=float,
        default=0.002,
        help="Minimum normalized plate box area.",
    )
    parser.add_argument(
        "--max-plate-area",
        type=float,
        default=0.45,
        help="Maximum normalized plate box area.",
    )
    parser.add_argument(
        "--max-plate-aspect-ratio",
        type=float,
        default=1.45,
        help="Maximum plate width/height or height/width ratio.",
    )
    parser.add_argument(
        "--max-plate-width-ratio",
        type=float,
        default=0.55,
        help="Maximum plate width as a fraction of image width.",
    )
    parser.add_argument(
        "--min-hub-area",
        type=float,
        default=0.00003,
        help="Minimum normalized hub box area.",
    )
    parser.add_argument(
        "--max-hub-area",
        type=float,
        default=0.08,
        help="Maximum normalized hub box area.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    kit_root = Path(__file__).resolve().parent
    work_root = args.work_root.resolve()
    review_zip = args.review_zip or find_latest_review_zip()

    ensure_frames(
        work_root=work_root,
        kit_root=kit_root,
        videos_dir=args.videos_dir,
        max_frames_per_video=args.max_frames_per_video,
        prepare_if_missing=args.prepare_frames_if_missing,
    )
    imported_labels = import_review_labels(review_zip=review_zip, work_root=work_root)
    stats = clean_labels(
        work_root=work_root,
        imported_labels_dir=imported_labels,
        max_previews=args.max_previews,
        min_plate_area=args.min_plate_area,
        max_plate_area=args.max_plate_area,
        max_plate_aspect_ratio=args.max_plate_aspect_ratio,
        max_plate_width_ratio=args.max_plate_width_ratio,
        min_hub_area=args.min_hub_area,
        max_hub_area=args.max_hub_area,
    )
    package_dataset(kit_root=kit_root, work_root=work_root)

    final_zip = work_root / "PowerNZ_deadlift_v1_corrected.zip"
    final_stats = CleanStats(
        frames=stats.frames,
        labels_in=stats.labels_in,
        labels_out=stats.labels_out,
        frames_with_labels=stats.frames_with_labels,
        frames_emptied=stats.frames_emptied,
        synthetic_hubs=stats.synthetic_hubs,
        output_zip=str(final_zip),
        preview_dir=stats.preview_dir,
    )
    summary_path = work_root / "auto_clean_summary.json"
    summary_path.write_text(json.dumps(asdict(final_stats), indent=2), encoding="utf-8")
    print(json.dumps(asdict(final_stats), indent=2))
    print(f"\nCorrected dataset ready for Kaggle: {final_zip}")


def find_latest_review_zip() -> Path:
    candidates: list[Path] = []
    for directory in (Path.home() / "Downloads", Path.cwd()):
        if directory.exists():
            candidates.extend(directory.glob("*autolabel*review*.zip"))
            candidates.extend(directory.glob("PowerNZ_autolabel_review*.zip"))
    candidates = sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "No PowerNZ_autolabel_review.zip found. Download it from Kaggle or pass --review-zip."
        )
    return candidates[0]


def ensure_frames(
    *,
    work_root: Path,
    kit_root: Path,
    videos_dir: Path,
    max_frames_per_video: int,
    prepare_if_missing: bool,
) -> None:
    frames_dir = work_root / "frames"
    if _count_images(frames_dir) > 0:
        return
    if not prepare_if_missing:
        raise RuntimeError(
            f"No frames found in {frames_dir}. "
            "Run prepare_PowerNZ_cloud_dataset.py --stage frames first, "
            "or rerun this script with --prepare-frames-if-missing."
        )

    subprocess.run(
        [
            sys.executable,
            str(kit_root / "prepare_PowerNZ_cloud_dataset.py"),
            "--stage",
            "frames",
            "--videos-dir",
            str(videos_dir),
            "--max-frames-per-video",
            str(max_frames_per_video),
            "--overwrite",
        ],
        check=True,
        cwd=kit_root.parent,
    )


def import_review_labels(*, review_zip: Path, work_root: Path) -> Path:
    if not review_zip.exists():
        raise FileNotFoundError(f"Review ZIP not found: {review_zip}")

    extract_dir = work_root / "review_import"
    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(review_zip) as archive:
        archive.extractall(extract_dir)

    label_roots = [
        path
        for path in extract_dir.rglob("labels")
        if path.is_dir() and any(path.glob("*.txt"))
    ]
    if not label_roots:
        raise RuntimeError(f"No labels folder with .txt files found inside {review_zip}")

    label_roots.sort(key=lambda path: sum(1 for _ in path.glob("*.txt")), reverse=True)
    imported = work_root / "labels_autolabel_raw"
    shutil.rmtree(imported, ignore_errors=True)
    imported.mkdir(parents=True, exist_ok=True)
    for label_path in label_roots[0].glob("*.txt"):
        shutil.copy2(label_path, imported / label_path.name)
    return imported


def clean_labels(
    *,
    work_root: Path,
    imported_labels_dir: Path,
    max_previews: int,
    min_plate_area: float,
    max_plate_area: float,
    max_plate_aspect_ratio: float,
    max_plate_width_ratio: float,
    min_hub_area: float,
    max_hub_area: float,
) -> CleanStats:
    frames_dir = work_root / "frames"
    labels_dir = work_root / "labels"
    preview_dir = work_root / "review" / "auto_cleaned_previews"
    labels_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(preview_dir, ignore_errors=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    frames = _iter_images(frames_dir)
    labels_in = 0
    labels_out = 0
    frames_with_labels = 0
    frames_emptied = 0
    synthetic_hubs = 0
    previews_written = 0

    for frame_path in frames:
        image = cv2.imread(str(frame_path))
        if image is None:
            continue

        source_label = imported_labels_dir / f"{frame_path.stem}.txt"
        boxes = read_boxes(source_label)
        labels_in += len(boxes)
        cleaned, added_synthetic_hub = select_clean_boxes(
            boxes,
            min_plate_area=min_plate_area,
            max_plate_area=max_plate_area,
            max_plate_aspect_ratio=max_plate_aspect_ratio,
            max_plate_width_ratio=max_plate_width_ratio,
            min_hub_area=min_hub_area,
            max_hub_area=max_hub_area,
        )
        synthetic_hubs += int(added_synthetic_hub)

        label_path = labels_dir / f"{frame_path.stem}.txt"
        label_path.write_text("\n".join(format_box(box) for box in cleaned), encoding="utf-8")
        labels_out += len(cleaned)
        if cleaned:
            frames_with_labels += 1
        elif boxes:
            frames_emptied += 1

        if previews_written < max_previews and (cleaned or boxes):
            preview = draw_preview(image, raw_boxes=boxes, clean_boxes=cleaned)
            cv2.imwrite(str(preview_dir / frame_path.name), preview)
            previews_written += 1

    return CleanStats(
        frames=len(frames),
        labels_in=labels_in,
        labels_out=labels_out,
        frames_with_labels=frames_with_labels,
        frames_emptied=frames_emptied,
        synthetic_hubs=synthetic_hubs,
        output_zip="",
        preview_dir=str(preview_dir),
    )


def select_clean_boxes(
    boxes: list[Box],
    *,
    min_plate_area: float,
    max_plate_area: float,
    max_plate_aspect_ratio: float,
    max_plate_width_ratio: float,
    min_hub_area: float,
    max_hub_area: float,
) -> tuple[list[Box], bool]:
    plates = [
        box
        for box in boxes
        if box.cls == 0
        and min_plate_area <= box.area <= max_plate_area
        and max(box.w / max(0.000001, box.h), box.h / max(0.000001, box.w)) <= max_plate_aspect_ratio
        and box.w <= max_plate_width_ratio
        and box.h <= 0.72
        and 0.01 <= box.left
        and box.right <= 0.99
        and 0.01 <= box.top
        and box.bottom <= 0.995
    ]
    hubs = [
        box
        for box in boxes
        if box.cls == 1
        and min_hub_area <= box.area <= max_hub_area
        and 0.0 <= box.left
        and box.right <= 1.0
        and 0.0 <= box.top
        and box.bottom <= 1.0
    ]
    if not plates:
        return [], False

    plate = max(plates, key=lambda candidate: score_plate(candidate, hubs))
    hub = choose_hub_for_plate(plate, hubs)
    synthetic = False
    if hub is None:
        hub = synthesize_hub_from_plate(plate)
        synthetic = True

    return [plate, hub], synthetic


def score_plate(plate: Box, hubs: list[Box]) -> float:
    size_score = min(1.0, plate.area / 0.08)
    center_score = 1.0 - min(1.0, abs(plate.cx - 0.5) * 1.1)
    floor_penalty = max(0.0, plate.bottom - 0.93) * 4.0
    hub_score = 0.0
    if hubs:
        nearest = min(_center_distance(plate, hub) for hub in hubs)
        hub_score = max(0.0, 1.0 - nearest / max(plate.w, plate.h, 0.01))
    return size_score + center_score + hub_score * 2.0 - floor_penalty


def choose_hub_for_plate(plate: Box, hubs: list[Box]) -> Box | None:
    if not hubs:
        return None
    max_distance = max(plate.w, plate.h) * 0.85
    candidates = [
        hub
        for hub in hubs
        if _center_distance(plate, hub) <= max_distance
        and plate.left - plate.w * 0.15 <= hub.cx <= plate.right + plate.w * 0.15
        and plate.top - plate.h * 0.15 <= hub.cy <= plate.bottom + plate.h * 0.15
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda hub: _center_distance(plate, hub))


def synthesize_hub_from_plate(plate: Box) -> Box:
    hub_size = max(0.018, min(0.12, min(plate.w, plate.h) * 0.20))
    return Box(cls=1, cx=plate.cx, cy=plate.cy, w=hub_size, h=hub_size)


def package_dataset(*, kit_root: Path, work_root: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(kit_root / "prepare_PowerNZ_cloud_dataset.py"),
            "--stage",
            "package",
            "--work-root",
            str(work_root),
        ],
        check=True,
        cwd=kit_root.parent,
    )


def read_boxes(path: Path) -> list[Box]:
    if not path.exists():
        return []
    boxes: list[Box] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(value) for value in parts[1:])
        except ValueError:
            continue
        if cls not in CLASS_NAMES:
            continue
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, w, h)):
            continue
        if w <= 0.0 or h <= 0.0:
            continue
        boxes.append(Box(cls=cls, cx=cx, cy=cy, w=w, h=h))
    return boxes


def format_box(box: Box) -> str:
    return f"{box.cls} {box.cx:.6f} {box.cy:.6f} {box.w:.6f} {box.h:.6f}"


def draw_preview(image, *, raw_boxes: list[Box], clean_boxes: list[Box]):
    preview = image.copy()
    height, width = preview.shape[:2]
    for box in raw_boxes:
        draw_box(preview, box, width=width, height=height, color=(80, 80, 80), prefix="raw")
    for box in clean_boxes:
        color = (0, 220, 255) if box.cls == 0 else (80, 255, 120)
        draw_box(preview, box, width=width, height=height, color=color, prefix="clean")
    return preview


def draw_box(image, box: Box, *, width: int, height: int, color: tuple[int, int, int], prefix: str) -> None:
    x1 = int(max(0.0, box.left) * width)
    y1 = int(max(0.0, box.top) * height)
    x2 = int(min(1.0, box.right) * width)
    y2 = int(min(1.0, box.bottom) * height)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    label = f"{prefix}:{CLASS_NAMES[box.cls]}"
    cv2.putText(image, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def _iter_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _count_images(directory: Path) -> int:
    return len(_iter_images(directory))


def _center_distance(a: Box, b: Box) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


if __name__ == "__main__":
    main()
