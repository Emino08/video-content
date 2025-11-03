from __future__ import annotations

from typing import List, Tuple

from .track import Track


def _velocity(track: Track) -> List[Tuple[float, float, float]]:
    """Return list of (t, vx, vy) using bbox center deltas per second."""
    v = []
    hist = track.history
    for i in range(1, len(hist)):
        _, t0, b0 = hist[i - 1]
        _, t1, b1 = hist[i]
        dt = max(1e-6, t1 - t0)
        cx0, cy0 = b0.center()
        cx1, cy1 = b1.center()
        vx = (cx1 - cx0) / dt
        vy = (cy1 - cy0) / dt
        v.append((t1, vx, vy))
    return v


def direction_from_velocity(vx: float, vy: float) -> str:
    # Map to coarse compass directions
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        return "stationary"
    horiz = "E" if vx > 0 else "W"
    vert = "S" if vy > 0 else "N"  # screen coords: y increases downward
    if abs(vx) > 1.5 * abs(vy):
        return f"{horiz}"
    if abs(vy) > 1.5 * abs(vx):
        return f"{vert}"
    return f"{vert}->{horiz}"


def summarize_motion(track: Track) -> Tuple[float, float, str]:
    """Return (speed, accel, direction) simple summary using last 5 steps."""
    v = _velocity(track)
    if not v:
        return 0.0, 0.0, "stationary"
    tail = v[-5:]
    speeds = [(vx * vx + vy * vy) ** 0.5 for _, vx, vy in tail]
    speed = sum(speeds) / len(speeds)
    accel = 0.0
    if len(speeds) >= 2:
        dt = max(1e-6, tail[-1][0] - tail[0][0])
        accel = (speeds[-1] - speeds[0]) / dt
    direction = direction_from_velocity(tail[-1][1], tail[-1][2])
    return speed, accel, direction

