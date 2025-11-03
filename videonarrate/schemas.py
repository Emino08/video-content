from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float

    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


@dataclass
class Mask:
    # For simplicity store RLE or polygon in data; optional
    data: Optional[Any] = None


@dataclass
class Entity:
    id: int
    label: str
    bbox: BBox
    score: float = 1.0
    mask: Optional[Mask] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Motion:
    direction: Optional[str] = None  # e.g., "N->S", "E->W"
    speed: Optional[float] = None     # pixels/sec or m/s if calibrated
    accel: Optional[float] = None     # change in speed per sec


@dataclass
class Action:
    label: str
    confidence: float
    source_model: Optional[str] = None


@dataclass
class Interaction:
    type: str
    confidence: float


@dataclass
class Scene:
    tags: List[str]
    confidence: float = 1.0


@dataclass
class Provenance:
    frames: Tuple[int, int]
    models: Dict[str, str] = field(default_factory=dict)


@dataclass
class Event:
    start: float
    end: float
    subjects: List[Entity]
    objects: List[Entity] = field(default_factory=list)
    action: Optional[Action] = None
    interaction: Optional[Interaction] = None
    motion: Optional[Motion] = None
    intent_hypothesis: Optional[Dict[str, Any]] = None
    scene: Optional[Scene] = None
    provenance: Optional[Provenance] = None
    caption: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        def convert(o):
            if hasattr(o, "to_dict"):
                return o.to_dict()
            if hasattr(o, "__dict__") or hasattr(o, "__dataclass_fields__"):
                return asdict(o)
            return o

        d = asdict(self)
        # Fix nested dataclasses inside lists
        d["subjects"] = [asdict(s) for s in self.subjects]
        d["objects"] = [asdict(o) for o in self.objects]
        if self.provenance is not None:
            d["provenance"] = asdict(self.provenance)
        return d


@dataclass
class CaptionLine:
    t_start: float
    t_end: float
    text: str
    confidence: float = 1.0


@dataclass
class Summary:
    scenes: List[Dict[str, Any]] = field(default_factory=list)

