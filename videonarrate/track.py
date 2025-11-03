from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .schemas import Entity, BBox


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = a.w * a.h
    b_area = b.w * b.h
    union = a_area + b_area - inter + 1e-6
    return inter / union


@dataclass
class Track:
    id: int
    label: str
    bbox: BBox
    score: float
    history: List[Tuple[int, float, BBox]] = field(default_factory=list)  # (frame_idx, t, bbox)
    alive: bool = True

    def update(self, frame_idx: int, t: float, bbox: BBox, score: float):
        self.bbox = bbox
        self.score = score
        self.history.append((frame_idx, t, bbox))


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._last_seen: Dict[int, int] = {}

    def step(self, frame_idx: int, t: float, detections: List[Entity]) -> List[Track]:
        assigned: Dict[int, int] = {}  # det_idx -> track_id
        # Try to match detections to existing tracks by IoU
        for det_idx, det in enumerate(detections):
            best_iou = 0.0
            best_tid = None
            for tid, tr in self.tracks.items():
                if not tr.alive:
                    continue
                if tr.label != det.label:
                    continue
                ov = iou(tr.bbox, det.bbox)
                if ov > self.iou_threshold and ov > best_iou:
                    best_iou = ov
                    best_tid = tid
            if best_tid is not None:
                assigned[det_idx] = best_tid

        # Update matched tracks
        for det_idx, tid in assigned.items():
            det = detections[det_idx]
            tr = self.tracks[tid]
            tr.update(frame_idx, t, det.bbox, det.score)
            self._last_seen[tid] = frame_idx

        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx in assigned:
                continue
            tid = self._next_id
            self._next_id += 1
            tr = Track(id=tid, label=det.label, bbox=det.bbox, score=det.score)
            tr.update(frame_idx, t, det.bbox, det.score)
            self.tracks[tid] = tr
            self._last_seen[tid] = frame_idx

        # Age out old tracks
        to_remove = []
        for tid, last_idx in self._last_seen.items():
            if frame_idx - last_idx > self.max_age:
                self.tracks[tid].alive = False
                to_remove.append(tid)
        for tid in to_remove:
            self._last_seen.pop(tid, None)

        return [tr for tr in self.tracks.values() if tr.alive]

