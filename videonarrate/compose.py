from __future__ import annotations

from typing import List, Dict, Any

from .schemas import Event, CaptionLine, Scene, Motion, Action, Entity, BBox, Provenance


def _direction_phrase(direction: str) -> str:
    mapping = {
        "N": "north",
        "S": "south",
        "E": "east",
        "W": "west",
        "N->E": "northeast",
        "N->W": "northwest",
        "S->E": "southeast",
        "S->W": "southwest",
        "stationary": "in place",
    }
    return mapping.get(direction, direction)


def caption_from_event(ev: Event) -> str:
    subj = ev.subjects[0] if ev.subjects else None
    obj = ev.objects[0] if ev.objects else None
    parts: List[str] = []
    if subj is not None:
        name = f"a {subj.label}"
        if ev.action is not None:
            verb = ev.action.label
        else:
            verb = "moves"
        if ev.motion and ev.motion.direction and ev.motion.direction != "stationary":
            dir_phrase = _direction_phrase(ev.motion.direction)
            parts.append(f"{name} {verb} towards the {dir_phrase}")
        else:
            parts.append(f"{name} {verb}")
    if obj is not None:
        if subj is not None:
            parts.append(f"near a {obj.label}")
        else:
            parts.append(f"a {obj.label} is visible")
    if not parts:
        parts.append("A scene unfolds")
    return " ".join(parts) + "."


def compose_captions(events: List[Event]) -> List[CaptionLine]:
    caps: List[CaptionLine] = []
    for ev in events:
        text = ev.caption or caption_from_event(ev)
        caps.append(CaptionLine(t_start=ev.start, t_end=ev.end, text=text, confidence=1.0))
    return caps


def summarize_events(events: List[Event]) -> Dict[str, Any]:
    labels: Dict[str, int] = {}
    for ev in events:
        for e in (ev.subjects + ev.objects):
            labels[e.label] = labels.get(e.label, 0) + 1
    top = sorted(labels.items(), key=lambda x: x[1], reverse=True)[:5]
    return {
        "entities": [{"label": k, "count": v} for k, v in top],
        "events_count": len(events),
    }

