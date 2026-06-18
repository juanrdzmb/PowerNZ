from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from detect_objects import (
    BeigePlateDetector,
    ColorPlateDetector,
    Detection,
    MultiColorPlateDetector,
)


CLASS_IDS = {"plate": 0, "bar_hub": 1, "bar_sleeve": 2, "bar_shaft": 3}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto-annotate frames using existing color/ellipse plate detectors."
    )
    parser.add_argument(
        "--frames-dir",
        default=Path("datasets/training/frames"),
        type=Path,
        help="Directory with extracted frames.",
    )
    parser.add_argument(
        "--labels-dir",
        default=Path("datasets/training/labels"),
        type=Path,
        help="Directory to save YOLO-format labels.",
    )
    return parser


def detection_to_yolo(det: Detection, img_w: int, img_h: int, class_id: int) -> str:
    cx = (det.x1 + det.x2) / 2.0 / img_w
    cy = (det.y1 + det.y2) / 2.0 / img_h
    bw = det.width / img_w
    bh = det.height / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main() -> None:
    args = build_parser().parse_args()
    frames_dir = Path(args.frames_dir)
    labels_dir = Path(args.labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    if not frames:
        print(f"No frames found in {frames_dir}")
        return

    red_detector = ColorPlateDetector(expected_diameter_pixels=None)
    beige_detector = BeigePlateDetector(expected_diameter_pixels=None)
    multi_detector = MultiColorPlateDetector(
        red_detector=red_detector,
        beige_detector=beige_detector,
        expected_diameter_pixels=None,
    )

    total_plates = 0
    total_hubs = 0
    labeled = 0

    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        img_h, img_w = frame.shape[:2]

        detections = multi_detector.detect(frame)
        if not detections:
            continue

        lines = []
        for det in detections:
            if det.label == "plate":
                lines.append(detection_to_yolo(det, img_w, img_h, CLASS_IDS["plate"]))
                total_plates += 1
            elif det.label == "barbell":
                lines.append(detection_to_yolo(det, img_w, img_h, CLASS_IDS["bar_hub"]))
                total_hubs += 1

        if lines:
            label_path = labels_dir / f"{frame_path.stem}.txt"
            label_path.write_text("\n".join(lines))
            labeled += 1

    print(f"Auto-annotated {labeled}/{len(frames)} frames")
    print(f"  Plates: {total_plates} | Hubs (from barbell): {total_hubs}")
    print(f"  Labels saved to: {labels_dir}")

    if labeled == 0:
        print("\nWARNING: No detections found. The color detectors may need plate_diameter_px.")
        print("Try running with subjects wearing visible plates (red/beige) on a bright background.")


if __name__ == "__main__":
    main()
