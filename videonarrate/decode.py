from __future__ import annotations

from typing import Iterator, Tuple


def decode_video_cv2(path: str, fps: int) -> Iterator[Tuple[int, float, "Frame"]]:
    """
    Yields (frame_index, timestamp_seconds, frame_bgr) at approximately the given fps.
    Falls back to yielding no frames if OpenCV is unavailable or the video cannot be opened,
    allowing the pipeline to run with empty outputs (per README mock-detector path).
    """
    try:
        import cv2  # type: ignore
    except Exception:
        # No OpenCV: yield nothing (graceful no-op)
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        # Unopenable source: yield nothing
        return

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval_native = int(max(1, round(native_fps / max(1, fps))))

    idx = 0
    out_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval_native == 0:
            t = out_idx / max(1, fps)
            yield (idx, t, frame)
            out_idx += 1
        idx += 1

    cap.release()
