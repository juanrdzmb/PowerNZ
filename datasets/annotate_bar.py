from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


CLASS_NAMES = ["plate", "bar_hub"]
CLASS_COLORS = [
    (0, 255, 0),    # plate: green
    (0, 0, 255),    # bar_hub: red
    (255, 0, 0),    # bar_sleeve: blue
    (0, 255, 255),  # bar_shaft: yellow
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive bounding-box annotation tool for YOLO detection dataset."
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
        help="Directory to save YOLO-format .txt labels.",
    )
    parser.add_argument(
        "--classes",
        default=",".join(CLASS_NAMES),
        help="Comma-separated class names (default: plate,bar_hub).",
    )
    return parser


class AnnotationState:
    def __init__(self, class_names: list[str], labels_dir: Path) -> None:
        self.class_names = class_names
        self.labels_dir = labels_dir
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.current_class_idx = 0
        self.drawing = False
        self.start_point = (0, 0)
        self.end_point = (0, 0)
        self.annotations: list[tuple[str, float, float, float, float]] = []

    def active_class(self) -> str:
        return self.class_names[self.current_class_idx]

    def active_color(self) -> tuple[int, int, int]:
        idx = self.current_class_idx % len(CLASS_COLORS)
        return CLASS_COLORS[idx]


def main() -> None:
    args = build_parser().parse_args()

    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    frames_dir = Path(args.frames_dir)
    labels_dir = Path(args.labels_dir)

    if not frames_dir.exists():
        print(f"Frames directory not found: {frames_dir}")
        print("Run extract_frames.py first.")
        return

    frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not frames:
        print(f"No JPEG/PNG frames found in: {frames_dir}")
        return

    print(f"Found {len(frames)} frames.")
    print(f"Classes: {class_names}")
    print("\nControls:")
    print("  Mouse drag  : draw bounding box for current class")
    class_help = ", ".join(
        f"{number + 1}={name}" for number, name in enumerate(class_names[:9])
    )
    print(f"  1-9         : select class ({class_help})")
    print("  c           : clear annotations on current frame")
    print("  n           : next frame (saves current)")
    print("  p           : previous frame")
    print("  q           : quit")
    print("  +           : skip frame (no labels, go to next)")
    print()

    idx = 0
    state = AnnotationState(class_names, labels_dir)
    window_name = "Bar Annotator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    frame = cv2.imread(str(frames[idx]))
    if frame is None:
        print("Failed to load first frame.")
        return

    h, w = frame.shape[:2]
    display = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal state
        if event == cv2.EVENT_LBUTTONDOWN:
            state.drawing = True
            state.start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
            state.end_point = (x, y)
            temp = frame.copy()
            for cls, x1, y1, x2, y2 in state.annotations:
                ci = class_names.index(cls)
                cv2.rectangle(temp, (int(x1), int(y1)), (int(x2), int(y2)), CLASS_COLORS[ci % len(CLASS_COLORS)], 2)
                cv2.putText(temp, cls, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[ci % len(CLASS_COLORS)], 1)
            cv2.rectangle(temp, state.start_point, state.end_point, state.active_color(), 2)
            cv2.imshow(window_name, temp)
        elif event == cv2.EVENT_LBUTTONUP:
            state.drawing = False
            x1 = min(state.start_point[0], x)
            y1 = min(state.start_point[1], y)
            x2 = max(state.start_point[0], x)
            y2 = max(state.start_point[1], y)
            if x2 - x1 > 3 and y2 - y1 > 3:
                state.annotations.append((state.active_class(), float(x1), float(y1), float(x2), float(y2)))

    cv2.setMouseCallback(window_name, mouse_callback)

    def save_labels(frame_path: Path) -> None:
        label_path = state.labels_dir / f"{frame_path.stem}.txt"
        lines = []
        for cls_name, x1, y1, x2, y2 in state.annotations:
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            cls_id = class_names.index(cls_name)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        label_path.write_text("\n".join(lines))
        print(f"  Saved: {label_path} ({len(lines)} boxes)")

    def draw_annotations():
        nonlocal display
        display = frame.copy()
        for cls_name, x1, y1, x2, y2 in state.annotations:
            ci = class_names.index(cls_name)
            cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), CLASS_COLORS[ci % len(CLASS_COLORS)], 2)
            cv2.putText(display, cls_name, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[ci % len(CLASS_COLORS)], 1)
        info_lines = [
            f"Frame: {idx+1}/{len(frames)}  {frames[idx].name}",
            f"Class: {state.active_class()} ({state.current_class_idx + 1}/{len(class_names)})",
            f"Boxes: {len(state.annotations)}",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(display, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def load_annotations(frame_path: Path) -> None:
        label_path = state.labels_dir / f"{frame_path.stem}.txt"
        state.annotations.clear()
        if not label_path.exists():
            return
        for line in label_path.read_text().strip().splitlines():
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = (cx - bw / 2.0) * w
                y1 = (cy - bh / 2.0) * h
                x2 = (cx + bw / 2.0) * w
                y2 = (cy + bh / 2.0) * h
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
                state.annotations.append((cls_name, x1, y1, x2, y2))
            except (ValueError, IndexError):
                continue

    load_annotations(frames[idx])
    draw_annotations()
    cv2.imshow(window_name, display)

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            save_labels(frames[idx])
            break
        elif key == ord("n"):
            save_labels(frames[idx])
            idx = min(idx + 1, len(frames) - 1)
            frame = cv2.imread(str(frames[idx]))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            load_annotations(frames[idx])
            draw_annotations()
            cv2.imshow(window_name, display)
        elif key == ord("p"):
            save_labels(frames[idx])
            idx = max(0, idx - 1)
            frame = cv2.imread(str(frames[idx]))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            load_annotations(frames[idx])
            draw_annotations()
            cv2.imshow(window_name, display)
        elif key == ord("+"):
            idx = min(idx + 1, len(frames) - 1)
            frame = cv2.imread(str(frames[idx]))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            load_annotations(frames[idx])
            draw_annotations()
            cv2.imshow(window_name, display)
        elif key == ord("c"):
            state.annotations.clear()
            draw_annotations()
            cv2.imshow(window_name, display)
        elif ord("1") <= key <= ord(str(len(class_names))):
            state.current_class_idx = key - ord("1")
            draw_annotations()
            cv2.imshow(window_name, display)
        elif key in {8, 127}:  # backspace / delete
            if state.annotations:
                state.annotations.pop()
                draw_annotations()
                cv2.imshow(window_name, display)

    cv2.destroyAllWindows()
    print("Annotation session ended.")


if __name__ == "__main__":
    main()
