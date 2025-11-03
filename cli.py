from __future__ import annotations

import argparse
from pathlib import Path

from videonarrate.config import Config
from videonarrate.pipeline import process_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="videonarrate",
        description="Convert videos into structured, time-aware narratives and JSON events.",
    )
    p.add_argument("--input", required=True, help="Path to input video file")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--fps", type=int, default=8, help="Sampling FPS")
    p.add_argument("--window", type=float, default=2.5, help="Window size in seconds")
    p.add_argument("--stride", type=float, default=1.0, help="Stride in seconds")
    p.add_argument("--detector", default="yolov8-seg", choices=["yolov8-seg", "mock"], help="Detector backend")
    p.add_argument("--min-det-conf", type=float, default=0.25, help="Minimum detection confidence")
    p.add_argument("--max-seconds", type=float, default=None, help="Max seconds to process (optional)")
    p.add_argument("--allow-labels", default=None, help="Comma-separated labels to keep (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    allow_labels = None
    if args.allow_labels:
        allow_labels = {s.strip() for s in args.allow_labels.split(',') if s.strip()}

    cfg = Config(
        fps=args.fps,
        window=args.window,
        stride=args.stride,
        detector=args.detector,
        min_det_conf=args.min_det_conf,
        max_seconds=args.max_seconds,
        allowed_labels=allow_labels,
    )
    process_video(args.input, args.out, cfg)


if __name__ == "__main__":
    main()
