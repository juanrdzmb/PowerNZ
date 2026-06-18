from __future__ import annotations

import argparse
from pathlib import Path

from datasets.yolo_dataset import create_label_previews, materialize_yolo_detection_dataset

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "Ultralytics is required for training. Install it with: pip install ultralytics"
    ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a custom YOLO detector for plate and bar_hub."
    )
    parser.add_argument(
        "--data",
        default="datasets/dataset_bar_2cls.yaml",
        type=Path,
        help="Dataset YAML config (default: datasets/dataset_bar_2cls.yaml).",
    )
    parser.add_argument(
        "--source-root",
        default="datasets/training",
        type=Path,
        help="Flat annotated source dataset with frames/ and labels/ folders.",
    )
    parser.add_argument(
        "--prepared-root",
        default="datasets/training_yolo/bar_2cls",
        type=Path,
        help="Prepared YOLO dataset root with train/val splits.",
    )
    parser.add_argument(
        "--val-ratio",
        default=0.2,
        type=float,
        help="Validation split ratio used when preparing the YOLO dataset.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Deterministic split seed.",
    )
    parser.add_argument(
        "--skip-label-preview",
        action="store_true",
        help="Skip automatic label preview images under outputs/validation/dataset_label_review.",
    )
    parser.add_argument(
        "--base-model",
        default="yolov8n.pt",
        help="Base YOLO model checkpoint (default: yolov8n.pt, auto-downloaded).",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--imgsz",
        default=640,
        type=int,
        help="Input image size (default: 640).",
    )
    parser.add_argument(
        "--batch",
        default=16,
        type=int,
        help="Batch size (default: 16).",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device (0 for GPU 0, cpu for CPU, mps for Apple).",
    )
    parser.add_argument(
        "--output",
        default="models/powerai_bar_detector.pt",
        type=Path,
        help="Output model path (default: models/powerai_bar_detector.pt).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    prepared = materialize_yolo_detection_dataset(
        source_root=args.source_root,
        output_root=args.prepared_root,
        class_count=2,
        val_ratio=args.val_ratio,
        seed=args.seed,
        repair_labels=True,
    )
    print(
        "Prepared YOLO dataset: "
        f"{prepared.train_count} train, {prepared.val_count} val, "
        f"{prepared.skipped_count} skipped -> {prepared.output_root}"
    )
    if not args.skip_label_preview:
        preview_count = create_label_previews(
            source_root=args.source_root,
            output_dir=Path("outputs/validation/dataset_label_review/bar_detector"),
            class_names=["plate", "bar_hub"],
        )
        print(f"Label preview images: {preview_count}")

    model = YOLO(args.base_model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name="powerai_bar_detector",
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.export(format="pt", imgsz=args.imgsz)
    best_pt = Path("runs/detect/powerai_bar_detector/weights/best.pt")
    if best_pt.exists():
        import shutil
        shutil.copy2(str(best_pt), str(output_path))
        print(f"Model saved to: {output_path}")
    else:
        print("Training finished. Run export manually or check runs/detect/powerai_bar_detector/")

    print("Done. Use with: --object-model models/powerai_bar_detector.pt")


if __name__ == "__main__":
    main()
