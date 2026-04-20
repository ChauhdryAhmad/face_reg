#!/usr/bin/env python3
"""Trim a video between start and end times and save to an output file.

Usage examples:
  python3 video_trim.py inputs/clip2.mp4 --start 0 --end 5
  python3 video_trim.py inputs/clip2.mp4 --start 00:00:10 --end 00:00:20 --output out.mp4 --overwrite

This script uses ffmpeg. It first attempts a fast stream-copy trim and
falls back to re-encoding if that fails.
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_time(t: str) -> float:
    """Parse time strings in seconds or HH:MM:SS / MM:SS format into seconds (float)."""
    if t is None:
        raise ValueError("time string is required")
    t = str(t)
    # If it's a pure number, treat as seconds
    try:
        return float(t)
    except ValueError:
        pass

    parts = t.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    if len(parts) == 1:
        return parts[0]
    raise ValueError(f"unrecognized time format: {t}")


def run_ffmpeg(cmd: list) -> subprocess.CompletedProcess:
    """Run ffmpeg command and return CompletedProcess."""
    # print a friendly preview for debugging
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def trim_video(input_path: Path, start: float, end: float, output_path: Path, overwrite: bool = False) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")
    if end <= start:
        raise ValueError("end time must be greater than start time")

    duration = end - start
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try fast stream-copy first
    overwrite_flag = "-y" if overwrite else "-n"
    # Use -ss before -i for fast seeking; -t uses duration
    cmd_copy = [
        "ffmpeg",
        overwrite_flag,
        "-ss",
        str(start),
        "-i",
        str(input_path),
        "-t",
        str(duration),
        "-c",
        "copy",
        str(output_path),
    ]

    cp = run_ffmpeg(cmd_copy)
    if cp.returncode == 0:
        print(f"Wrote (stream-copy) {output_path}")
        return

    print("Stream-copy failed, retrying with re-encode...")

    # Fallback: re-encode for accurate trimming
    cmd_reencode = [
        "ffmpeg",
        overwrite_flag,
        "-ss",
        str(start),
        "-i",
        str(input_path),
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(output_path),
    ]

    cp2 = run_ffmpeg(cmd_reencode)
    if cp2.returncode != 0:
        print(cp2.stdout.decode(errors="ignore"), file=sys.stderr)
        raise RuntimeError("ffmpeg failed to produce trimmed file")
    print(f"Wrote (re-encoded) {output_path}")


def make_default_output(input_path: Path, start: float, end: float) -> Path:
    stem = input_path.stem
    ext = input_path.suffix or ".mp4"
    out_name = f"{stem}_trim_{int(start)}_{int(end)}{ext}"
    return input_path.parent.joinpath("outputs").joinpath(out_name)


def main(argv=None):
    p = argparse.ArgumentParser(description="Trim a video using ffmpeg")
    p.add_argument("input", help="input video path")
    p.add_argument("--start", required=True, help="start time (seconds or HH:MM:SS)")
    p.add_argument("--end", required=True, help="end time (seconds or HH:MM:SS)")
    p.add_argument("--output", help="output path (optional). default: ./outputs/<input>_trim_<start>_<end>.ext")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing output")
    args = p.parse_args(argv)

    input_path = Path(args.input)
    start = parse_time(args.start)
    end = parse_time(args.end)
    output_path = Path(args.output) if args.output else make_default_output(input_path, start, end)

    try:
        trim_video(input_path, start, end, output_path, overwrite=args.overwrite)
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
