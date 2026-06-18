from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WORLD_CLASSES = ("barbell weight plate", "barbell collar sleeve")


@dataclass(frozen=True)
class AutolabelStats:
    images: int
    labeled_images: int
    labels: int
    skipped_large: int
    skipped_low_confidence: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto-label barbell plates and hubs with YOLO-World."
    )
    parser.add_argument(
        "--frames-dir",
        default=Path("datasets/training/frames"),
        type=Path,
        help="Directory containing extracted training frames.",
    )
    parser.add_argument(
        "--labels-dir",
        default=Path("datasets/training/labels"),
        type=Path,
        help="Directory where YOLO label txt files will be written.",
    )
    parser.add_argument(
        "--model",
        default="yolov8x-worldv2.pt",
        help="YOLO-World checkpoint name/path.",
    )
    parser.add_argument(
        "--confidence",
        default=0.25,
        type=float,
        help="Minimum detection confidence.",
    )
    parser.add_argument(
        "--max-box-area-ratio",
        default=0.80,
        type=float,
        help="Discard boxes larger than this fraction of the frame area.",
    )
    parser.add_argument(
        "--imgsz",
        default=960,
        type=int,
        help="Inference image size.",
    )
    parser.add_argument(
        "--save-empty",
        action="store_true",
        help="Write empty label files for frames with no accepted labels.",
    )
    return parser


def autolabel_frames(
    *,
    frames_dir: Path,
    labels_dir: Path,
    model_name: str,
    confidence: float,
    max_box_area_ratio: float,
    imgsz: int,
    save_empty: bool = False,
) -> AutolabelStats:
    try:
        from ultralytics import YOLOWorld
    except ImportError as exc:
        raise ImportError(
            "Ultralytics with YOLOWorld is required. Install it with: pip install ultralytics"
        ) from exc

    frames = _iter_images(frames_dir)
    if not frames:
        raise RuntimeError(f"No image frames found under {frames_dir}")

    labels_dir.mkdir(parents=True, exist_ok=True)
    model = YOLOWorld(model_name)
    model.set_classes(list(WORLD_CLASSES))

    labeled_images = 0
    label_count = 0
    skipped_large = 0
    skipped_low_confidence = 0

    for image_path in frames:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        height, width = frame.shape[:2]
        result = model.predict(frame, imgsz=imgsz, verbose=False)[0]
        lines: list[str] = []

        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for box in boxes:
                box_confidence = float(box.conf[0].item())
                if box_confidence < confidence:
                    skipped_low_confidence += 1
                    continue

                class_id = int(box.cls[0].item())
                if class_id not in {0, 1}:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                box_area_ratio = _box_area_ratio(x1, y1, x2, y2, width, height)
                if box_area_ratio > max_box_area_ratio:
                    skipped_large += 1
                    continue

                line = _to_yolo_line(class_id, x1, y1, x2, y2, width, height)
                if line is not None:
                    lines.append(line)

        label_path = labels_dir / f"{image_path.stem}.txt"
        if lines or save_empty:
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        elif label_path.exists():
            label_path.unlink()

        if lines:
            labeled_images += 1
            label_count += len(lines)

    return AutolabelStats(
        images=len(frames),
        labeled_images=labeled_images,
        labels=label_count,
        skipped_large=skipped_large,
        skipped_low_confidence=skipped_low_confidence,
    )


def _iter_images(directory: Path) -> list[Path]:
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


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


def _to_yolo_line(
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


def main() -> None:
    args = build_parser().parse_args()
    stats = autolabel_frames(
        frames_dir=args.frames_dir,
        labels_dir=args.labels_dir,
        model_name=args.model,
        confidence=args.confidence,
        max_box_area_ratio=args.max_box_area_ratio,
        imgsz=args.imgsz,
        save_empty=args.save_empty,
    )
    print(
        "Auto-label complete: "
        f"{stats.labeled_images}/{stats.images} images, "
        f"{stats.labels} labels, "
        f"{stats.skipped_low_confidence} low-confidence skipped, "
        f"{stats.skipped_large} large boxes skipped."
    )


if __name__ == "__main__":
    main()
