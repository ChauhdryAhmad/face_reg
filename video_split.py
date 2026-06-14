#!/usr/bin/env python3
"""Split a video into fixed-length chunks.

Usage examples:
  python3 video_split.py inputs/clip2.mp4 --seconds 10
  python3 video_split.py inputs/clip2.mp4 --seconds 30 --output-dir output/chunks --overwrite

The script uses OpenCV so it can run with the project's existing Python
dependencies. It writes video-only chunks; audio tracks are not preserved.
"""
import argparse
import sys
from pathlib import Path

import cv2


def parse_seconds(value: str) -> float:
    """Parse and validate a positive chunk length in seconds."""
    try:
        seconds = float(value)
    except ValueError as exc:
        raise ValueError(f"invalid seconds value: {value}") from exc

    if seconds <= 0:
        raise ValueError("seconds must be greater than 0")
    return seconds


def make_default_output_dir(input_path: Path) -> Path:
    return input_path.parent / f"{input_path.stem}_chunks"


def make_chunk_path(output_dir: Path, stem: str, chunk_index: int) -> Path:
    return output_dir / f"{stem}_chunk_{chunk_index:03d}.mp4"


def create_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not create output video: {output_path}")
    return writer


def split_video(input_path: Path, seconds: float, output_dir: Path, overwrite: bool = False) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"input is not a file: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("could not read video metadata")

    frames_per_chunk = max(1, round(fps * seconds))
    chunk_index = 0
    frame_index = 0
    writer = None
    current_output = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % frames_per_chunk == 0:
                if writer is not None:
                    writer.release()
                    print(f"Wrote {current_output}")

                current_output = make_chunk_path(output_dir, input_path.stem, chunk_index)
                if current_output.exists() and not overwrite:
                    raise FileExistsError(f"output already exists: {current_output}")

                writer = create_writer(current_output, fps, width, height)
                chunk_index += 1

            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if frame_index == 0:
        raise RuntimeError("input video did not contain any frames")

    if current_output is not None:
        print(f"Wrote {current_output}")
    print(f"Done. Split {frame_index} frames into {chunk_index} chunk(s) in {output_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Split a video into fixed-length chunks")
    parser.add_argument("input", help="input video path")
    parser.add_argument("--seconds", required=True, help="chunk length in seconds")
    parser.add_argument(
        "--output-dir",
        help="directory for chunks. default: <input folder>/<input name>_chunks",
    )
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing chunk files")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else make_default_output_dir(input_path)

    try:
        split_video(input_path, parse_seconds(args.seconds), output_dir, overwrite=args.overwrite)
    except Exception as exc:
        print("Error:", exc, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
