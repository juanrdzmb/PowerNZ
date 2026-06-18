from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "Ultralytics is required for training. Install it with: pip install ultralytics"
    ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a custom YOLO segmentation model for athlete/background_person."
    )
    parser.add_argument(
        "--data",
        default="datasets/dataset_athlete.yaml",
        type=Path,
        help="Dataset YAML config (default: datasets/dataset_athlete.yaml).",
    )
    parser.add_argument(
        "--base-model",
        default="yolov8n-seg.pt",
        help="Base YOLO segmentation checkpoint (default: yolov8n-seg.pt, auto-downloaded).",
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
        default="models/powerai_athlete_seg.pt",
        type=Path,
        help="Output model path (default: models/powerai_athlete_seg.pt).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    model = YOLO(args.base_model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name="powerai_athlete_seg",
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
    best_pt = Path("runs/segment/powerai_athlete_seg/weights/best.pt")
    if best_pt.exists():
        import shutil
        shutil.copy2(str(best_pt), str(output_path))
        print(f"Model saved to: {output_path}")
    else:
        print("Training finished. Run export manually or check runs/segment/powerai_athlete_seg/")

    print("Done. Use with: --segmentation-model models/powerai_athlete_seg.pt")


if __name__ == "__main__":
    main()
