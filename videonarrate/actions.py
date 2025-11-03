from __future__ import annotations

from typing import Optional

from .track import Track
from .motion import summarize_motion
from .schemas import Action


def heuristic_action(track: Track, label_hint: Optional[str] = None) -> Optional[Action]:
    """
    Lightweight action recognizer: infer basic actions from motion stats.
    - stationary -> "standing" / "stopped"
    - slow movement -> "walking" / "moving"
    - fast movement -> "running" / "driving"
    """
    speed, accel, _ = summarize_motion(track)
    lbl = label_hint or track.label
    if speed < 2.0:
        action = "standing" if lbl == "person" else "stopped"
        conf = 0.55
    elif speed < 20.0:
        action = "walking" if lbl == "person" else "moving"
        conf = 0.65
    else:
        action = "running" if lbl == "person" else "driving"
        conf = 0.7

    # Adjust confidence based on accel magnitude (more dynamic -> stronger signal)
    conf = min(0.9, max(0.5, conf + min(0.2, abs(accel) / 30.0)))
    return Action(label=action, confidence=conf, source_model="heuristic")

