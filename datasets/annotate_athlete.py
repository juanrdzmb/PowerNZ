from __future__ import annotations

import argparse
from pathlib import Path

import cv2

ATHLETE_CLASSES = ["athlete", "background_person"]
CLASS_COLORS = [
    (0, 255, 0),    # athlete: green
    (0, 0, 255),    # background_person: red
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive polygon annotation tool for YOLO-seg athlete dataset."
    )
    parser.add_argument(
        "--frames-dir",
        default=Path("datasets/training/frames"),
        type=Path,
        help="Directory with extracted frames.",
    )
    parser.add_argument(
        "--masks-dir",
        default=Path("datasets/training/masks"),
        type=Path,
        help="Directory to save YOLO-seg .txt mask labels.",
    )
    parser.add_argument(
        "--sam2-model",
        default=None,
        type=Path,
        help="Optional SAM2 model path for semi-automatic mask generation.",
    )
    return parser


def points_to_yolo_seg_line(points: list[tuple[int, int]], class_id: int, img_w: int, img_h: int) -> str:
    norm_points = []
    for x, y in points:
        norm_points.append(f"{x / img_w:.6f}")
        norm_points.append(f"{y / img_h:.6f}")
    return f"{class_id} " + " ".join(norm_points)


def main() -> None:
    args = build_parser().parse_args()

    frames_dir = Path(args.frames_dir)
    masks_dir = Path(args.masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    if not frames_dir.exists():
        print(f"Frames directory not found: {frames_dir}")
        print("Run extract_frames.py first.")
        return

    frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not frames:
        print(f"No JPEG/PNG frames found in: {frames_dir}")
        return

    print(f"Found {len(frames)} frames.")
    print(f"Classes: {ATHLETE_CLASSES}")
    print("\nControls:")
    print("  Left click  : add polygon point")
    print("  Enter       : finish current polygon (prompt for class)")
    print("  c           : clear polygons on current frame")
    print("  n/p         : next/previous frame (saves current)")
    print("  q           : quit")
    print("  +           : skip frame")
    print("  z           : undo last point")
    print()

    idx = 0
    window_name = "Athlete Seg Annotator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame = cv2.imread(str(frames[idx]))
    if frame is None:
        print("Failed to load first frame.")
        return

    img_h, img_w = frame.shape[:2]
    display = frame.copy()

    current_polygon: list[tuple[int, int]] = []
    polygons: list[tuple[int, list[tuple[int, int]]]] = []  # (class_id, points)

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_polygon, display
        if event == cv2.EVENT_LBUTTONDOWN:
            current_polygon.append((x, y))
            temp = frame.copy()
            draw_all(temp)
            cv2.polylines(temp, [np.array(current_polygon, dtype=np.int32)], False, (0, 255, 255), 2)
            for px, py in current_polygon:
                cv2.circle(temp, (px, py), 4, (0, 255, 255), -1)
            cv2.imshow(window_name, temp)

    import numpy as np
    cv2.setMouseCallback(window_name, mouse_callback)

    def draw_all(img):
        for cls_id, pts in polygons:
            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            if len(pts) >= 3:
                cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], (*color, 64))
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], True, color, 2)
        info_lines = [
            f"Frame: {idx+1}/{len(frames)}  {frames[idx].name}",
            f"Polygons: {len(polygons)} | Current: {len(current_polygon)}pts",
            "Left=add pt | Enter=finish | c=clear | n/p=nav | q=quit",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(img, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def save_mask(frame_path: Path) -> None:
        label_path = masks_dir / f"{frame_path.stem}.txt"
        lines = []
        for cls_id, pts in polygons:
            if len(pts) < 3:
                continue
            lines.append(points_to_yolo_seg_line(pts, cls_id, img_w, img_h))
        label_path.write_text("\n".join(lines))
        print(f"  Saved: {label_path} ({len(lines)} polygons)")

    def load_mask(frame_path: Path) -> None:
        label_path = masks_dir / f"{frame_path.stem}.txt"
        polygons.clear()
        if not label_path.exists():
            return
        for line in label_path.read_text().strip().splitlines():
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 7:  # class_id + at least 3 points (6 coords)
                continue
            try:
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                pts = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * img_w)
                    y = int(coords[i + 1] * img_h)
                    pts.append((x, y))
                polygons.append((cls_id, pts))
            except (ValueError, IndexError):
                continue

    load_mask(frames[idx])
    draw_all(display)
    cv2.imshow(window_name, display)

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            save_mask(frames[idx])
            break
        elif key == 13:  # Enter
            if len(current_polygon) >= 3:
                print(f"Class for new polygon? (a=athlete, b=background_person): ", end="")
                # Use a small popup approach via key:
                cls_chosen = 0
                while True:
                    display = frame.copy()
                    draw_all(display)
                    for cls_id, pts in polygons:
                        color = CLASS_COLORS[cls_id]
                        if len(pts) >= 3:
                            cv2.fillPoly(display, [np.array(pts, dtype=np.int32)], (*color, 48))
                        cv2.polylines(display, [np.array(pts, dtype=np.int32)], True, color, 2)
                    cv2.polylines(display, [np.array(current_polygon, dtype=np.int32)], True, (255, 255, 0), 2)
                    cv2.putText(display, "Press 'a' for athlete, 'b' for background_person", (10, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    cv2.imshow(window_name, display)
                    k = cv2.waitKey(0) & 0xFF
                    if k == ord("a"):
                        cls_chosen = 0
                        break
                    elif k == ord("b"):
                        cls_chosen = 1
                        break
                    elif k == ord("q") or k == 27:
                        cls_chosen = -1
                        break
                if cls_chosen >= 0:
                    polygons.append((cls_chosen, list(current_polygon)))
                current_polygon.clear()
            else:
                current_polygon.clear()
            display = frame.copy()
            draw_all(display)
            cv2.imshow(window_name, display)
        elif key == ord("n"):
            if current_polygon:
                current_polygon.clear()
            save_mask(frames[idx])
            idx = min(idx + 1, len(frames) - 1)
            frame = cv2.imread(str(frames[idx]))
            if frame is None:
                continue
            img_h, img_w = frame.shape[:2]
            load_mask(frames[idx])
            display = frame.copy()
            draw_all(display)
            cv2.imshow(window_name, display)
        elif key == ord("p"):
            if current_polygon:
                current_polygon.clear()
            save_mask(frames[idx])
            idx = max(0, idx - 1)
            frame = cv2.imread(str(frames[idx]))
            if frame is None:
                continue
            img_h, img_w = frame.shape[:2]
            load_mask(frames[idx])
            display = frame.copy()
            draw_all(display)
            cv2.imshow(window_name, display)
        elif key == ord("+"):
            idx = min(idx + 1, len(frames) - 1)
            frame = cv2.imread(str(frames[idx]))
            if frame is None:
                continue
            img_h, img_w = frame.shape[:2]
            load_mask(frames[idx])
            display = frame.copy()
            draw_all(display)
            cv2.imshow(window_name, display)
        elif key == ord("c"):
            current_polygon.clear()
            polygons.clear()
            display = frame.copy()
            draw_all(display)
            cv2.imshow(window_name, display)
        elif key == ord("z"):
            if current_polygon:
                current_polygon.pop()
                display = frame.copy()
                draw_all(display)
                cv2.polylines(display, [np.array(current_polygon, dtype=np.int32)], False, (0, 255, 255), 2)
                for px, py in current_polygon:
                    cv2.circle(display, (px, py), 4, (0, 255, 255), -1)
                cv2.imshow(window_name, display)
        elif key in {8, 127}:  # backspace
            if polygons:
                polygons.pop()
                display = frame.copy()
                draw_all(display)
                cv2.imshow(window_name, display)

    cv2.destroyAllWindows()
    print("Annotation session ended.")


if __name__ == "__main__":
    main()
