from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract representative frames from videos for dataset annotation."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        nargs="+",
        help="Input video paths.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("datasets/training/frames"),
        type=Path,
        help="Output directory for extracted JPEG frames.",
    )
    parser.add_argument(
        "--stride",
        default=0,
        type=int,
        help="Extract every Nth frame. 0 = adaptive sampling (target ~150-300 frames).",
    )
    parser.add_argument(
        "--max-frames",
        default=300,
        type=int,
        help="Maximum total frames to extract per video (default: 300).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_counter = 0

    for video_path in args.input:
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Skipping missing video: {video_path}")
            continue

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            print(f"Cannot open video: {video_path}")
            continue

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        stride = args.stride
        if stride <= 0:
            stride = max(1, total_frames // args.max_frames)

        video_name = video_path.stem
        saved_from_video = 0

        for frame_idx in range(0, total_frames, stride):
            if saved_from_video >= args.max_frames:
                break

            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok:
                continue

            filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame)
            global_counter += 1
            saved_from_video += 1

        capture.release()
        print(f"{video_path.name}: extracted {saved_from_video} frames (stride={stride})")

    print(f"\nTotal extracted: {global_counter} frames -> {output_dir}")


if __name__ == "__main__":
    main()
