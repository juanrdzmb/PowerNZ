from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
WORLD_CLASSES = (
    "barbell weight plate",
    "bumper plate",
    "weight plate",
    "olympic plate",
    "barbell collar sleeve",
    "barbell sleeve",
)
WORLD_CLASS_TO_TARGET = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 1,
}
DETECTOR_NAMES = {0: "plate", 1: "bar_hub"}
ATHLETE_NAMES = {0: "athlete", 1: "background_person"}


@dataclass(frozen=True)
class AutolabelStats:
    images: int
    labeled_images: int
    labels: int
    skipped_large: int
    skipped_low_confidence: int
    review_zip: str


@dataclass(frozen=True)
class PreparedDataset:
    yaml_path: str
    train_count: int
    val_count: int
    skipped_count: int


@dataclass(frozen=True)
class TrainingOutputs:
    detector_model: str | None
    athlete_model: str | None
    output_zip: str


@dataclass(frozen=True)
class CleanedDataset:
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
        description="Run PowerNZ auto-labeling and training inside Kaggle."
    )
    parser.add_argument(
        "--mode",
        choices=("autolabel", "clean", "autolabel-clean", "train", "all", "autolabel-clean-train"),
        required=True,
        help=(
            "'autolabel' creates labels/review. 'clean' cleans existing labels. "
            "'autolabel-clean' does both and writes a corrected ZIP. "
            "'train' trains from corrected labels. 'all' does autolabel then train. "
            "'autolabel-clean-train' does the full automatic path."
        ),
    )
    parser.add_argument(
        "--dataset-zip",
        type=Path,
        default=None,
        help="Dataset ZIP path. If omitted, a ZIP or folder with frames/ is discovered under /kaggle/input.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset folder that contains frames/. Use this when Kaggle shows an extracted dataset folder.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/kaggle/working/PowerNZ_cloud_training"),
        help="Kaggle working directory.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Training device. Use 0 for Kaggle GPU or cpu for debugging.",
    )
    parser.add_argument(
        "--autolabel-model",
        default="yolov8x-worldv2.pt",
        help="YOLO-World model for first-pass labels.",
    )
    parser.add_argument(
        "--autolabel-confidence",
        type=float,
        default=0.25,
        help="Minimum confidence for YOLO-World labels.",
    )
    parser.add_argument(
        "--autolabel-imgsz",
        type=int,
        default=960,
        help="Image size for YOLO-World inference.",
    )
    parser.add_argument(
        "--max-box-area-ratio",
        type=float,
        default=0.45,
        help="Discard auto-label boxes larger than this fraction of image area.",
    )
    parser.add_argument(
        "--max-plate-aspect-ratio",
        type=float,
        default=1.45,
        help="Discard YOLO-World plate boxes that are too elongated to be a single disc.",
    )
    parser.add_argument(
        "--max-plate-width-ratio",
        type=float,
        default=0.55,
        help="Discard YOLO-World plate boxes wider than this fraction of the image.",
    )
    parser.add_argument(
        "--disable-color-fallback",
        action="store_true",
        help="Disable HSV color fallback labels for colored plates.",
    )
    parser.add_argument(
        "--save-empty-labels",
        action="store_true",
        help="Write empty label files for images with no accepted auto-labels.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=80,
        help="Maximum label preview images to create.",
    )
    parser.add_argument(
        "--include-frames-in-review",
        action="store_true",
        help="Include frames in the auto-label review ZIP. This makes the ZIP much larger.",
    )
    parser.add_argument(
        "--cleaned-output-zip",
        type=Path,
        default=Path("/kaggle/working/PowerNZ_deadlift_v1_corrected.zip"),
        help="Corrected dataset ZIP written by clean/autolabel-clean modes.",
    )
    parser.add_argument(
        "--min-plate-area",
        type=float,
        default=0.002,
        help="Minimum normalized plate box area kept by automatic cleaning.",
    )
    parser.add_argument(
        "--max-plate-area",
        type=float,
        default=0.45,
        help="Maximum normalized plate box area kept by automatic cleaning.",
    )
    parser.add_argument(
        "--min-hub-area",
        type=float,
        default=0.00003,
        help="Minimum normalized hub box area kept by automatic cleaning.",
    )
    parser.add_argument(
        "--max-hub-area",
        type=float,
        default=0.08,
        help="Maximum normalized hub box area kept by automatic cleaning.",
    )
    parser.add_argument(
        "--detector-base-model",
        default="yolo11n.pt",
        help="Base YOLO detector checkpoint.",
    )
    parser.add_argument(
        "--detector-epochs",
        type=int,
        default=120,
        help="Detector training epochs.",
    )
    parser.add_argument(
        "--detector-imgsz",
        type=int,
        default=960,
        help="Detector training image size.",
    )
    parser.add_argument(
        "--detector-batch",
        type=int,
        default=8,
        help="Detector batch size. Lower to 4 if Kaggle runs out of memory.",
    )
    parser.add_argument(
        "--athlete-base-model",
        default="yolo11s-seg.pt",
        help="Base YOLO segmentation checkpoint.",
    )
    parser.add_argument(
        "--athlete-epochs",
        type=int,
        default=100,
        help="Athlete segmentation training epochs.",
    )
    parser.add_argument(
        "--athlete-imgsz",
        type=int,
        default=768,
        help="Athlete segmentation image size.",
    )
    parser.add_argument(
        "--athlete-batch",
        type=int,
        default=4,
        help="Athlete segmentation batch size.",
    )
    parser.add_argument(
        "--skip-athlete-seg",
        action="store_true",
        help="Train only the bar detector even when masks are present.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic train/val split seed.",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=Path("/kaggle/working/PowerNZ_trained_models_v1.zip"),
        help="Final model ZIP path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = materialize_dataset_source(
        dataset_zip=args.dataset_zip,
        dataset_root=args.dataset_root,
        output_dir=args.work_dir / "dataset",
    )

    result: object
    if args.mode == "autolabel":
        result = run_autolabel(args=args, dataset_root=dataset_root)
    elif args.mode == "clean":
        result = run_cleaning(args=args, dataset_root=dataset_root)
    elif args.mode == "autolabel-clean":
        run_autolabel(args=args, dataset_root=dataset_root)
        result = run_cleaning(args=args, dataset_root=dataset_root)
    elif args.mode == "train":
        result = run_training(args=args, dataset_root=dataset_root)
    elif args.mode == "all":
        run_autolabel(args=args, dataset_root=dataset_root)
        result = run_training(args=args, dataset_root=dataset_root)
    else:
        run_autolabel(args=args, dataset_root=dataset_root)
        run_cleaning(args=args, dataset_root=dataset_root)
        result = run_training(args=args, dataset_root=dataset_root)

    print(json.dumps(asdict(result), indent=2))


def materialize_dataset_source(
    *,
    dataset_zip: Path | None,
    dataset_root: Path | None,
    output_dir: Path,
) -> Path:
    if dataset_zip is not None:
        return extract_dataset_zip(dataset_zip, output_dir)

    discovered = dataset_root or find_input_dataset_source()
    if discovered.is_file() and discovered.suffix.lower() == ".zip":
        return extract_dataset_zip(discovered, output_dir)
    return copy_dataset_root(discovered, output_dir)


def find_input_dataset_source() -> Path:
    input_root = Path("/kaggle/input")
    zips = sorted(input_root.rglob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    if zips:
        print(f"Using dataset ZIP: {zips[0]}")
        return zips[0]

    dataset_roots = find_dataset_roots(input_root)
    if not dataset_roots:
        raise FileNotFoundError(
            "No ZIP file or folder with frames/ was found under /kaggle/input. "
            "Add the PowerNZ frames dataset, usually PowerNZ_deadlift_v1_frames.zip, "
            "then attach that dataset to this notebook with Add Data."
        )

    dataset_roots.sort(
        key=lambda root: (_count_files(root / "frames", IMAGE_EXTENSIONS), root.stat().st_mtime),
        reverse=True,
    )
    print(f"Using extracted dataset folder: {dataset_roots[0]}")
    return dataset_roots[0]


def extract_dataset_zip(zip_path: Path, extract_dir: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset ZIP not found: {zip_path}")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    dataset_roots = [path for path in extract_dir.rglob("frames") if path.is_dir()]
    if not dataset_roots:
        raise RuntimeError(f"No frames directory found after extracting {zip_path}")
    dataset_root = dataset_roots[0].parent
    print(f"Dataset root: {dataset_root}")
    return dataset_root


def copy_dataset_root(source: Path, output_dir: Path) -> Path:
    source_root = resolve_dataset_root(source)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    destination = output_dir / source_root.name
    shutil.copytree(
        source_root,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "runs", "*.pt"),
    )
    (destination / "labels").mkdir(parents=True, exist_ok=True)
    (destination / "masks").mkdir(parents=True, exist_ok=True)
    print(f"Copied dataset folder to writable Kaggle work dir: {destination}")
    return destination


def resolve_dataset_root(source: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Dataset folder not found: {source}")
    if source.is_file():
        raise ValueError(f"Expected a folder with frames/, got file: {source}")
    if (source / "frames").is_dir():
        return source

    candidates = find_dataset_roots(source)
    if not candidates:
        raise RuntimeError(f"No frames directory found under {source}")
    candidates.sort(
        key=lambda root: (_count_files(root / "frames", IMAGE_EXTENSIONS), root.stat().st_mtime),
        reverse=True,
    )
    return candidates[0]


def find_dataset_roots(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        frames_dir.parent
        for frames_dir in root.rglob("frames")
        if frames_dir.is_dir() and _count_files(frames_dir, IMAGE_EXTENSIONS) > 0
    ]


def run_autolabel(*, args: argparse.Namespace, dataset_root: Path) -> AutolabelStats:
    try:
        from ultralytics import YOLOWorld
    except ImportError as exc:
        raise ImportError("Install ultralytics first: pip install ultralytics") from exc

    frames_dir = dataset_root / "frames"
    labels_dir = dataset_root / "labels"
    review_dir = args.work_dir / "autolabel_review"
    labels_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    images = _iter_images(frames_dir)
    if not images:
        raise RuntimeError(f"No images found in {frames_dir}")

    model = YOLOWorld(args.autolabel_model)
    model.set_classes(list(WORLD_CLASSES))

    labeled_images = 0
    label_count = 0
    skipped_large = 0
    skipped_low_confidence = 0

    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        result = model.predict(frame, imgsz=args.autolabel_imgsz, verbose=False)[0]
        lines: list[str] = []

        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for box in boxes:
                confidence = float(box.conf[0].item())
                if confidence < args.autolabel_confidence:
                    skipped_low_confidence += 1
                    continue
                class_id = int(box.cls[0].item())
                target_class_id = WORLD_CLASS_TO_TARGET.get(class_id)
                if target_class_id not in DETECTOR_NAMES:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                if not is_plausible_world_box(
                    target_class_id,
                    x1,
                    y1,
                    x2,
                    y2,
                    width,
                    height,
                    max_box_area_ratio=args.max_box_area_ratio,
                    max_plate_aspect_ratio=args.max_plate_aspect_ratio,
                    max_plate_width_ratio=args.max_plate_width_ratio,
                ):
                    skipped_large += 1
                    continue
                line = _to_yolo_box_line(target_class_id, x1, y1, x2, y2, width, height)
                if line:
                    lines.append(line)

        if not args.disable_color_fallback:
            lines.extend(color_fallback_label_lines(frame, existing_lines=lines))

        label_path = labels_dir / f"{image_path.stem}.txt"
        if lines or args.save_empty_labels:
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        elif label_path.exists():
            label_path.unlink()

        if lines:
            labeled_images += 1
            label_count += len(lines)

    preview_count = create_label_previews(
        frames_dir=frames_dir,
        labels_dir=labels_dir,
        output_dir=review_dir / "previews",
        class_names=DETECTOR_NAMES,
        max_images=args.preview_count,
    )
    stats = {
        "images": len(images),
        "labeled_images": labeled_images,
        "labels": label_count,
        "skipped_large": skipped_large,
        "skipped_low_confidence": skipped_low_confidence,
        "preview_images": preview_count,
        "classes": DETECTOR_NAMES,
    }
    (review_dir / "autolabel_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (review_dir / "README_AUTOLABEL.md").write_text(
        "\n".join(
            [
                "# PowerNZ auto-label review",
                "",
                "Copy the labels folder back into training_cloud_kit/work/deadlift_v1/labels.",
                "Review previews before training. Delete or fix labels that point to floor/background plates.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    review_zip = args.work_dir / "PowerNZ_autolabel_review.zip"
    zip_autolabel_review(
        zip_path=review_zip,
        labels_dir=labels_dir,
        review_dir=review_dir,
        frames_dir=frames_dir if args.include_frames_in_review else None,
    )
    return AutolabelStats(
        images=len(images),
        labeled_images=labeled_images,
        labels=label_count,
        skipped_large=skipped_large,
        skipped_low_confidence=skipped_low_confidence,
        review_zip=str(review_zip),
    )


def run_training(*, args: argparse.Namespace, dataset_root: Path) -> TrainingOutputs:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics first: pip install ultralytics") from exc

    models_dir = args.work_dir / "models"
    runs_dir = args.work_dir / "runs"
    prepared_dir = args.work_dir / "prepared"
    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    bar_prepared = prepare_yolo_dataset(
        frames_dir=dataset_root / "frames",
        labels_dir=dataset_root / "labels",
        output_root=prepared_dir / "bar_detector",
        yaml_path=prepared_dir / "bar_detector.yaml",
        class_names=DETECTOR_NAMES,
        val_ratio=args.val_ratio,
        seed=args.seed,
        segmentation=False,
    )
    detector_model_path = train_detector(args, bar_prepared, models_dir, runs_dir)

    athlete_model_path: Path | None = None
    masks_dir = dataset_root / "masks"
    if not args.skip_athlete_seg and _count_label_files(masks_dir) > 0:
        athlete_prepared = prepare_yolo_dataset(
            frames_dir=dataset_root / "frames",
            labels_dir=masks_dir,
            output_root=prepared_dir / "athlete_seg",
            yaml_path=prepared_dir / "athlete_seg.yaml",
            class_names=ATHLETE_NAMES,
            val_ratio=args.val_ratio,
            seed=args.seed,
            segmentation=True,
        )
        athlete_model_path = train_athlete_seg(args, athlete_prepared, models_dir, runs_dir)
    else:
        print("Skipping athlete segmentation training: no masks found or --skip-athlete-seg was used.")

    summary = {
        "bar_dataset": asdict(bar_prepared),
        "detector_model": str(detector_model_path) if detector_model_path else None,
        "athlete_model": str(athlete_model_path) if athlete_model_path else None,
    }
    (args.work_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    zip_training_outputs(args.output_zip, models_dir=models_dir, runs_dir=runs_dir, work_dir=args.work_dir)
    return TrainingOutputs(
        detector_model=str(detector_model_path) if detector_model_path else None,
        athlete_model=str(athlete_model_path) if athlete_model_path else None,
        output_zip=str(args.output_zip),
    )


def run_cleaning(*, args: argparse.Namespace, dataset_root: Path) -> CleanedDataset:
    frames_dir = dataset_root / "frames"
    labels_dir = dataset_root / "labels"
    masks_dir = dataset_root / "masks"
    review_dir = args.work_dir / "auto_cleaned_review"
    preview_dir = review_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    frames = _iter_images(frames_dir)
    if not frames:
        raise RuntimeError(f"No images found in {frames_dir}")
    if not labels_dir.exists() or _count_label_files(labels_dir) == 0:
        raise RuntimeError(
            f"No labels found in {labels_dir}. Run --mode autolabel-clean or autolabel before clean."
        )

    cleaned_dir = args.work_dir / "cleaned_dataset"
    if cleaned_dir.exists():
        shutil.rmtree(cleaned_dir)
    for dirname in ("frames", "labels", "masks", "review"):
        (cleaned_dir / dirname).mkdir(parents=True, exist_ok=True)

    labels_in = 0
    labels_out = 0
    frames_with_labels = 0
    frames_emptied = 0
    synthetic_hubs = 0
    previews_written = 0

    for image_path in frames:
        shutil.copy2(image_path, cleaned_dir / "frames" / image_path.name)
        mask_path = masks_dir / f"{image_path.stem}.txt"
        if mask_path.exists():
            shutil.copy2(mask_path, cleaned_dir / "masks" / mask_path.name)

        raw_boxes = read_boxes(labels_dir / f"{image_path.stem}.txt")
        labels_in += len(raw_boxes)
        clean_boxes, added_synthetic_hub = select_clean_boxes(
            raw_boxes,
            min_plate_area=args.min_plate_area,
            max_plate_area=args.max_plate_area,
            max_plate_aspect_ratio=args.max_plate_aspect_ratio,
            max_plate_width_ratio=args.max_plate_width_ratio,
            min_hub_area=args.min_hub_area,
            max_hub_area=args.max_hub_area,
        )
        synthetic_hubs += int(added_synthetic_hub)
        labels_out += len(clean_boxes)
        if clean_boxes:
            frames_with_labels += 1
        elif raw_boxes:
            frames_emptied += 1

        (cleaned_dir / "labels" / f"{image_path.stem}.txt").write_text(
            "\n".join(format_box(box) for box in clean_boxes),
            encoding="utf-8",
        )

        if previews_written < args.preview_count and (raw_boxes or clean_boxes):
            frame = cv2.imread(str(image_path))
            if frame is not None:
                preview = draw_cleaning_preview(frame, raw_boxes=raw_boxes, clean_boxes=clean_boxes)
                preview_path = preview_dir / image_path.name
                cv2.imwrite(str(preview_path), preview)
                shutil.copy2(preview_path, cleaned_dir / "review" / image_path.name)
                previews_written += 1

    write_kaggle_dataset_metadata(cleaned_dir, dataset_root=dataset_root)
    zip_cleaned_dataset(root=cleaned_dir, zip_path=args.cleaned_output_zip)

    stats = CleanedDataset(
        frames=len(frames),
        labels_in=labels_in,
        labels_out=labels_out,
        frames_with_labels=frames_with_labels,
        frames_emptied=frames_emptied,
        synthetic_hubs=synthetic_hubs,
        output_zip=str(args.cleaned_output_zip),
        preview_dir=str(preview_dir),
    )
    (args.work_dir / "auto_clean_summary.json").write_text(
        json.dumps(asdict(stats), indent=2),
        encoding="utf-8",
    )
    print(f"Corrected dataset ZIP: {args.cleaned_output_zip}")
    return stats


def prepare_yolo_dataset(
    *,
    frames_dir: Path,
    labels_dir: Path,
    output_root: Path,
    yaml_path: Path,
    class_names: dict[int, str],
    val_ratio: float,
    seed: int,
    segmentation: bool,
) -> PreparedDataset:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Missing frames directory: {frames_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    items: list[tuple[Path, list[str]]] = []
    skipped = 0
    for image_path in _iter_images(frames_dir):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            skipped += 1
            continue
        labels = _read_valid_labels(
            label_path,
            class_count=len(class_names),
            segmentation=segmentation,
        )
        if not labels:
            skipped += 1
            continue
        items.append((image_path, labels))

    if not items:
        raise RuntimeError(f"No usable labels found in {labels_dir}")

    rng = random.Random(seed)
    rng.shuffle(items)
    val_count = int(round(len(items) * val_ratio))
    if len(items) > 1:
        val_count = max(1, min(len(items) - 1, val_count))
    else:
        val_count = 0

    val_items = items[:val_count]
    train_items = items[val_count:]
    _reset_prepared_dirs(output_root)
    _copy_split(train_items, output_root=output_root, split="train")
    _copy_split(val_items, output_root=output_root, split="val")
    write_yolo_yaml(yaml_path=yaml_path, dataset_root=output_root, class_names=class_names)
    return PreparedDataset(
        yaml_path=str(yaml_path),
        train_count=len(train_items),
        val_count=len(val_items),
        skipped_count=skipped,
    )


def train_detector(
    args: argparse.Namespace,
    prepared: PreparedDataset,
    models_dir: Path,
    runs_dir: Path,
) -> Path | None:
    from ultralytics import YOLO

    model = YOLO(args.detector_base_model)
    model.train(
        data=prepared.yaml_path,
        epochs=args.detector_epochs,
        imgsz=args.detector_imgsz,
        batch=args.detector_batch,
        device=args.device,
        project=str(runs_dir / "detect"),
        name="PowerNZ_bar_detector",
        exist_ok=True,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
    )
    best = runs_dir / "detect" / "PowerNZ_bar_detector" / "weights" / "best.pt"
    if not best.exists():
        print(f"Detector finished, but best.pt was not found at {best}")
        return None
    output = models_dir / "PowerNZ_bar_detector.pt"
    shutil.copy2(best, output)
    print(f"Detector model saved: {output}")
    return output


def train_athlete_seg(
    args: argparse.Namespace,
    prepared: PreparedDataset,
    models_dir: Path,
    runs_dir: Path,
) -> Path | None:
    from ultralytics import YOLO

    model = YOLO(args.athlete_base_model)
    model.train(
        data=prepared.yaml_path,
        epochs=args.athlete_epochs,
        imgsz=args.athlete_imgsz,
        batch=args.athlete_batch,
        device=args.device,
        project=str(runs_dir / "segment"),
        name="PowerNZ_athlete_seg",
        exist_ok=True,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
    )
    best = runs_dir / "segment" / "PowerNZ_athlete_seg" / "weights" / "best.pt"
    if not best.exists():
        print(f"Athlete segmentation finished, but best.pt was not found at {best}")
        return None
    output = models_dir / "PowerNZ_athlete_seg.pt"
    shutil.copy2(best, output)
    print(f"Athlete segmentation model saved: {output}")
    return output


def create_label_previews(
    *,
    frames_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    class_names: dict[int, str],
    max_images: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for image_path in _iter_images(frames_dir):
        if written >= max_images:
            break
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        labels = _read_valid_labels(label_path, class_count=len(class_names), segmentation=False)
        if not labels:
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        for line in labels:
            cls, x, y, w, h = _parse_box_label(line)
            x1 = int((x - w / 2.0) * width)
            y1 = int((y - h / 2.0) * height)
            x2 = int((x + w / 2.0) * width)
            y2 = int((y + h / 2.0) * height)
            color = (0, 220, 255) if cls == 0 else (80, 255, 120)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
            label = class_names.get(cls, str(cls))
            cv2.putText(
                frame,
                label,
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(output_dir / image_path.name), frame)
        written += 1
    return written


def zip_autolabel_review(
    *,
    zip_path: Path,
    labels_dir: Path,
    review_dir: Path,
    frames_dir: Path | None,
) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory, arc_root in ((labels_dir, "autolabel_review/labels"), (review_dir, "autolabel_review")):
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(arc_root) / path.relative_to(directory))
        if frames_dir is not None:
            for path in sorted(frames_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, Path("autolabel_review/frames") / path.relative_to(frames_dir))


def zip_training_outputs(*, zip_path: Path, models_dir: Path, runs_dir: Path, work_dir: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory, arc_root in ((models_dir, "models"), (runs_dir, "runs")):
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(arc_root) / path.relative_to(directory))
        summary = work_dir / "training_summary.json"
        if summary.exists():
            archive.write(summary, "training_summary.json")


def zip_cleaned_dataset(*, root: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_root_name = root.name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for dirname in ("frames", "labels", "masks", "review"):
            directory = root / dirname
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(dataset_root_name) / path.relative_to(root))
        for filename in (
            "dataset_bar.yaml",
            "dataset_athlete.yaml",
            "training_manifest.json",
            "README_DATASET.md",
        ):
            path = root / filename
            if path.exists():
                archive.write(path, Path(dataset_root_name) / filename)


def write_kaggle_dataset_metadata(cleaned_dir: Path, *, dataset_root: Path) -> None:
    (cleaned_dir / "dataset_bar.yaml").write_text(
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
    (cleaned_dir / "dataset_athlete.yaml").write_text(
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
        "dataset": "PowerNZ_deadlift_v1_cleaned",
        "source_dataset_root": str(dataset_root),
        "bar_classes": {"0": "plate", "1": "bar_hub"},
        "athlete_classes": {"0": "athlete", "1": "background_person"},
        "notes": [
            "Auto-cleaned in Kaggle after YOLO-World autolabeling.",
            "Only the best plate + bar_hub pair is kept per frame.",
            "Review preview images before final training when possible.",
        ],
    }
    (cleaned_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (cleaned_dir / "README_DATASET.md").write_text(
        "\n".join(
            [
                "# PowerNZ deadlift v1 cleaned dataset",
                "",
                "This dataset was auto-labeled and auto-cleaned in Kaggle.",
                "",
                "Detector classes:",
                "",
                "- 0: plate",
                "- 1: bar_hub",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_yolo_yaml(*, yaml_path: Path, dataset_root: Path, class_names: dict[int, str]) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    names_lines = [f"  {class_id}: {name}" for class_id, name in sorted(class_names.items())]
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(class_names)}",
                "names:",
                *names_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _iter_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _read_valid_labels(label_path: Path, *, class_count: int, segmentation: bool) -> list[str]:
    labels: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if segmentation:
            if len(parts) < 7:
                continue
        elif len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        if cls < 0 or cls >= class_count:
            continue
        if not all(0.0 <= value <= 1.0 for value in values):
            continue
        if not segmentation:
            _, _, w, h = values
            if w <= 0.0 or h <= 0.0:
                continue
        labels.append(" ".join([str(cls), *[f"{value:.6f}" for value in values]]))
    return labels


class CleanBox:
    def __init__(self, cls: int, cx: float, cy: float, w: float, h: float) -> None:
        self.cls = cls
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h

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


def read_boxes(label_path: Path) -> list[CleanBox]:
    if not label_path.exists():
        return []
    boxes: list[CleanBox] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(value) for value in parts[1:])
        except ValueError:
            continue
        if cls not in DETECTOR_NAMES:
            continue
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, w, h)):
            continue
        if w <= 0.0 or h <= 0.0:
            continue
        boxes.append(CleanBox(cls=cls, cx=cx, cy=cy, w=w, h=h))
    return boxes


def select_clean_boxes(
    boxes: list[CleanBox],
    *,
    min_plate_area: float,
    max_plate_area: float,
    max_plate_aspect_ratio: float,
    max_plate_width_ratio: float,
    min_hub_area: float,
    max_hub_area: float,
) -> tuple[list[CleanBox], bool]:
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


def score_plate(plate: CleanBox, hubs: list[CleanBox]) -> float:
    size_score = min(1.0, plate.area / 0.08)
    center_score = 1.0 - min(1.0, abs(plate.cx - 0.5) * 1.1)
    floor_penalty = max(0.0, plate.bottom - 0.93) * 4.0
    hub_score = 0.0
    if hubs:
        nearest = min(_center_distance(plate, hub) for hub in hubs)
        hub_score = max(0.0, 1.0 - nearest / max(plate.w, plate.h, 0.01))
    return size_score + center_score + hub_score * 2.0 - floor_penalty


def choose_hub_for_plate(plate: CleanBox, hubs: list[CleanBox]) -> CleanBox | None:
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


def synthesize_hub_from_plate(plate: CleanBox) -> CleanBox:
    hub_size = max(0.018, min(0.12, min(plate.w, plate.h) * 0.20))
    return CleanBox(cls=1, cx=plate.cx, cy=plate.cy, w=hub_size, h=hub_size)


def format_box(box: CleanBox) -> str:
    return f"{box.cls} {box.cx:.6f} {box.cy:.6f} {box.w:.6f} {box.h:.6f}"


def draw_cleaning_preview(frame, *, raw_boxes: list[CleanBox], clean_boxes: list[CleanBox]):
    preview = frame.copy()
    height, width = preview.shape[:2]
    for box in raw_boxes:
        draw_clean_box(preview, box, width=width, height=height, color=(90, 90, 90), prefix="raw")
    for box in clean_boxes:
        color = (0, 220, 255) if box.cls == 0 else (80, 255, 120)
        draw_clean_box(preview, box, width=width, height=height, color=color, prefix="clean")
    return preview


def draw_clean_box(
    frame,
    box: CleanBox,
    *,
    width: int,
    height: int,
    color: tuple[int, int, int],
    prefix: str,
) -> None:
    x1 = int(max(0.0, box.left) * width)
    y1 = int(max(0.0, box.top) * height)
    x2 = int(min(1.0, box.right) * width)
    y2 = int(min(1.0, box.bottom) * height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    label = f"{prefix}:{DETECTOR_NAMES[box.cls]}"
    cv2.putText(frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def _center_distance(a: CleanBox, b: CleanBox) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


def _parse_box_label(line: str) -> tuple[int, float, float, float, float]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Invalid box label: {line}")
    return int(float(parts[0])), *(float(value) for value in parts[1:])


def _reset_prepared_dirs(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    for relative in ("images/train", "images/val", "labels/train", "labels/val"):
        (output_root / relative).mkdir(parents=True, exist_ok=True)


def _copy_split(items: list[tuple[Path, list[str]]], *, output_root: Path, split: str) -> None:
    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split
    for image_path, labels in items:
        shutil.copy2(image_path, image_dir / image_path.name)
        (label_dir / f"{image_path.stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")


def _box_area_ratio(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> float:
    box_w = max(0.0, min(float(width), x2) - max(0.0, x1))
    box_h = max(0.0, min(float(height), y2) - max(0.0, y1))
    return (box_w * box_h) / max(1.0, float(width * height))


def is_plausible_world_box(
    class_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    *,
    max_box_area_ratio: float,
    max_plate_aspect_ratio: float,
    max_plate_width_ratio: float,
) -> bool:
    box_w = max(0.0, min(float(width), x2) - max(0.0, x1))
    box_h = max(0.0, min(float(height), y2) - max(0.0, y1))
    if box_w <= 2.0 or box_h <= 2.0:
        return False
    area_ratio = (box_w * box_h) / max(1.0, float(width * height))
    if area_ratio > max_box_area_ratio:
        return False

    normalized_w = box_w / max(1.0, float(width))
    normalized_h = box_h / max(1.0, float(height))
    aspect = max(box_w / box_h, box_h / box_w)
    if class_id == 0:
        if aspect > max_plate_aspect_ratio:
            return False
        if normalized_w > max_plate_width_ratio or normalized_h > 0.72:
            return False
    else:
        if aspect > 2.20:
            return False
        if area_ratio > 0.12:
            return False
    return True


def color_fallback_label_lines(frame, *, existing_lines: list[str]) -> list[str]:
    height, width = frame.shape[:2]
    existing_boxes = [box for line in existing_lines for box in [_box_from_line(line)] if box is not None]
    existing_plates = [box for box in existing_boxes if box.cls == 0]
    fallback_boxes = detect_colored_plate_boxes(frame)
    lines: list[str] = []

    for plate in fallback_boxes:
        if any(_center_distance(plate, existing) <= max(plate.w, plate.h) * 0.35 for existing in existing_plates):
            continue
        lines.append(format_box(plate))
        lines.append(format_box(synthesize_hub_from_plate(plate)))
        break
    return lines


def detect_colored_plate_boxes(frame) -> list[CleanBox]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, (0, 70, 40), (14, 255, 255)),
        cv2.inRange(hsv, (160, 70, 40), (179, 255, 255)),
        cv2.inRange(hsv, (15, 55, 70), (38, 255, 255)),
        cv2.inRange(hsv, (38, 35, 45), (88, 220, 230)),
        cv2.inRange(hsv, (90, 45, 45), (130, 255, 255)),
    ]
    mask = masks[0]
    for extra in masks[1:]:
        mask = cv2.bitwise_or(mask, extra)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = frame.shape[:2]
    candidates: list[tuple[float, CleanBox]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < max(80.0, width * height * 0.0015):
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue
        circularity = 4.0 * 3.14159265 * area / (perimeter * perimeter)
        if circularity < 0.12:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w / max(1.0, h), h / max(1.0, w))
        if aspect > 1.65:
            continue
        if w / max(1.0, width) > 0.55 or h / max(1.0, height) > 0.72:
            continue
        if y + h >= height * 0.985:
            continue

        pad = max(4.0, min(w, h) * 0.10)
        x1 = max(0.0, x - pad)
        y1 = max(0.0, y - pad)
        x2 = min(float(width - 1), x + w + pad)
        y2 = min(float(height - 1), y + h + pad)
        line = _to_yolo_box_line(0, x1, y1, x2, y2, width, height)
        box = _box_from_line(line) if line else None
        if box is None:
            continue
        score = area * (0.70 + box.cy) * (1.0 + min(1.0, circularity))
        candidates.append((score, box))

    return [box for _, box in sorted(candidates, key=lambda item: item[0], reverse=True)]


def _box_from_line(line: str | None) -> CleanBox | None:
    if not line:
        return None
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(value) for value in parts[1:])
    except ValueError:
        return None
    if cls not in DETECTOR_NAMES:
        return None
    if not all(0.0 <= value <= 1.0 for value in (cx, cy, w, h)):
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    return CleanBox(cls=cls, cx=cx, cy=cy, w=w, h=h)


def _to_yolo_box_line(
    class_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> str | None:
    left = max(0.0, min(float(width), x1))
    top = max(0.0, min(float(height), y1))
    right = max(0.0, min(float(width), x2))
    bottom = max(0.0, min(float(height), y2))
    if right <= left or bottom <= top:
        return None

    box_w = (right - left) / width
    box_h = (bottom - top) / height
    center_x = ((left + right) / 2.0) / width
    center_y = ((top + bottom) / 2.0) / height
    return f"{class_id} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}"


def _count_label_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.glob("*.txt") if path.is_file())


def _count_files(directory: Path, extensions: set[str]) -> int:
    if not directory.exists():
        return 0
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


if __name__ == "__main__":
    main()
