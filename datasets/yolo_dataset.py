from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class PreparedYoloDataset:
    output_root: Path
    train_count: int
    val_count: int
    skipped_count: int


def materialize_yolo_detection_dataset(
    *,
    source_root: Path,
    output_root: Path,
    class_count: int,
    val_ratio: float = 0.2,
    seed: int = 42,
    repair_labels: bool = True,
) -> PreparedYoloDataset:
    frames_dir = source_root / "frames"
    labels_dir = source_root / "labels"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Missing frames directory: {frames_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    labeled_images: list[tuple[Path, list[str]]] = []
    skipped_count = 0
    for image_path in sorted(_iter_images(frames_dir)):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            skipped_count += 1
            continue

        labels = _read_repaired_labels(label_path, class_count=class_count, repair=repair_labels)
        if not labels:
            skipped_count += 1
            continue
        labeled_images.append((image_path, labels))

    if not labeled_images:
        raise RuntimeError(f"No usable labeled images found under {source_root}")

    rng = random.Random(seed)
    rng.shuffle(labeled_images)
    val_count = int(round(len(labeled_images) * val_ratio))
    if len(labeled_images) > 1:
        val_count = max(1, min(len(labeled_images) - 1, val_count))
    else:
        val_count = 0

    val_items = labeled_images[:val_count]
    train_items = labeled_images[val_count:]

    _reset_split_dirs(output_root)
    _copy_split(train_items, output_root=output_root, split="train")
    _copy_split(val_items, output_root=output_root, split="val")

    return PreparedYoloDataset(
        output_root=output_root,
        train_count=len(train_items),
        val_count=len(val_items),
        skipped_count=skipped_count,
    )


def create_label_previews(
    *,
    source_root: Path,
    output_dir: Path,
    class_names: list[str],
    max_images: int = 40,
) -> int:
    frames_dir = source_root / "frames"
    labels_dir = source_root / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for image_path in sorted(_iter_images(frames_dir)):
        if written >= max_images:
            break
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        labels = _read_repaired_labels(label_path, class_count=len(class_names), repair=True)
        if not labels:
            continue

        height, width = frame.shape[:2]
        for line in labels:
            cls, x, y, w, h = _parse_label_line(line)
            x1 = int((x - w / 2.0) * width)
            y1 = int((y - h / 2.0) * height)
            x2 = int((x + w / 2.0) * width)
            y2 = int((y + h / 2.0) * height)
            color = (0, 220, 255) if cls == 0 else (80, 255, 120)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
            label = class_names[cls] if cls < len(class_names) else str(cls)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        cv2.imwrite(str(output_dir / image_path.name), frame)
        written += 1

    return written


def _iter_images(directory: Path) -> list[Path]:
    return [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _reset_split_dirs(output_root: Path) -> None:
    for relative in ("images/train", "images/val", "labels/train", "labels/val"):
        split_dir = output_root / relative
        split_dir.mkdir(parents=True, exist_ok=True)
        for child in split_dir.iterdir():
            if child.is_file():
                child.unlink()


def _copy_split(items: list[tuple[Path, list[str]]], *, output_root: Path, split: str) -> None:
    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split
    for image_path, labels in items:
        shutil.copy2(image_path, image_dir / image_path.name)
        (label_dir / f"{image_path.stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")


def _read_repaired_labels(label_path: Path, *, class_count: int, repair: bool) -> list[str]:
    labels: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            cls, x, y, w, h = _parse_label_line(raw_line)
        except ValueError:
            continue
        if cls < 0 or cls >= class_count:
            continue

        repaired = _repair_box(x, y, w, h) if repair else (x, y, w, h)
        if repaired is None:
            continue
        x, y, w, h = repaired
        if not all(0.0 <= value <= 1.0 for value in (x, y, w, h)):
            continue
        if w <= 0.0 or h <= 0.0:
            continue
        labels.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return labels


def _parse_label_line(line: str) -> tuple[int, float, float, float, float]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO label line: {line}")
    cls = int(float(parts[0]))
    x, y, w, h = (float(value) for value in parts[1:])
    return cls, x, y, w, h


def _repair_box(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float] | None:
    if w <= 0.0 or h <= 0.0:
        return None

    x1 = max(0.0, x - w / 2.0)
    y1 = max(0.0, y - h / 2.0)
    x2 = min(1.0, x + w / 2.0)
    y2 = min(1.0, y + h / 2.0)
    if x2 <= x1 or y2 <= y1:
        return None

    repaired_w = x2 - x1
    repaired_h = y2 - y1
    repaired_x = x1 + repaired_w / 2.0
    repaired_y = y1 + repaired_h / 2.0
    return repaired_x, repaired_y, repaired_w, repaired_h
