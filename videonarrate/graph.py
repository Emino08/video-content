from __future__ import annotations

from typing import List, Tuple

from .track import Track
from .schemas import Interaction


def infer_interactions(tracks: List[Track]) -> List[Tuple[int, int, Interaction]]:
    """
    Very simple interaction inference: if two tracks are close and relative speed suggests
    approach/avoid/yield. Returns list of (subject_id, object_id, Interaction).
    """
    pairs: List[Tuple[int, int, Interaction]] = []
    def center(b):
        c = (b.x + b.w/2.0, b.y + b.h/2.0)
        return c
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            ti, tj = tracks[i], tracks[j]
            ci = center(ti.bbox)
            cj = center(tj.bbox)
            dx = cj[0] - ci[0]
            dy = cj[1] - ci[1]
            dist2 = dx*dx + dy*dy
            # Close if centers within 2x average bbox size
            thresh = ((ti.bbox.w + tj.bbox.w + ti.bbox.h + tj.bbox.h) / 4.0)
            if dist2 < (2.0 * thresh) ** 2:
                # Naive: if one is person and the other is vehicle, assume potential yield
                labels = {ti.label, tj.label}
                if "person" in labels and ("car" in labels or "truck" in labels or "bus" in labels or "vehicle" in labels):
                    pairs.append((ti.id, tj.id, Interaction(type="yielding/passing", confidence=0.6)))
                else:
                    pairs.append((ti.id, tj.id, Interaction(type="nearby", confidence=0.5)))
    return pairs

