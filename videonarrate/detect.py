from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .schemas import Entity, BBox


@dataclass
class Detection:
    bbox: BBox
    score: float
    label: str


class Detector:
    def __init__(self, name: str = "yolov8-seg", min_conf: float = 0.25):
        self.name = name
        self.min_conf = min_conf
        self._impl = None
        if name == "yolov8-seg":
            try:
                from ultralytics import YOLO  # type: ignore
                self._impl = YOLO("yolov8s-seg.pt")
            except Exception:
                self._impl = None
        elif name == "mock":
            self._impl = None

    def infer(self, frame, next_entity_id_start: int = 1) -> List[Entity]:
        if self._impl is not None and self.name == "yolov8-seg":
            try:
                results = self._impl.predict(frame, verbose=False)
                ents: List[Entity] = []
                eid = next_entity_id_start
                for r in results:
                    # r.boxes.xywh, r.boxes.conf, r.boxes.cls
                    boxes = getattr(r.boxes, "xywh", [])
                    confs = getattr(r.boxes, "conf", [])
                    clses = getattr(r.boxes, "cls", [])
                    for i in range(len(boxes)):
                        try:
                            x, y, w, h = [float(v) for v in boxes[i]]
                            score = float(confs[i])
                            if score < self.min_conf:
                                continue
                            label_idx = int(clses[i])
                            label = r.names.get(label_idx, str(label_idx)) if hasattr(r, "names") else str(label_idx)
                            ents.append(Entity(id=eid, label=label, bbox=BBox(x, y, w, h), score=score))
                            eid += 1
                        except Exception:
                            continue
                return ents
            except Exception:
                pass
        # Fallback: no detections
        return []

