from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set


@dataclass
class Config:
    fps: int = 8
    window: float = 2.5
    stride: float = 1.0
    detector: str = "yolov8-seg"  # or "mock"
    tracker: str = "simple"        # or external like "bytetrack" if available
    max_tracks: int = 128
    min_det_conf: float = 0.25
    compose_with_vlm: bool = False
    # Optional cap on processing time (in seconds) to avoid long runs
    max_seconds: float | None = None
    # Optional allow-list of labels to keep from detector
    allowed_labels: Optional[Set[str]] = None
