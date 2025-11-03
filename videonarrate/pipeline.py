from __future__ import annotations

from pathlib import Path
from typing import List

from .config import Config
from .decode import decode_video_cv2
from .detect import Detector
from .track import SimpleTracker, Track
from .motion import summarize_motion
from .actions import heuristic_action
from .graph import infer_interactions
from .schemas import Event, Entity, Motion, Action, Provenance, Scene, Summary


def process_video(input_path: str, out_dir: str, cfg: Config) -> None:
    out_dir_p = Path(out_dir)

    # Initialize components
    detector = Detector(name=cfg.detector, min_conf=cfg.min_det_conf)
    tracker = SimpleTracker()

    events: List[Event] = []

    # Sliding window buffers
    window_s = cfg.window
    stride_s = cfg.stride
    window_frames: List[int] = []
    window_time: List[float] = []
    window_tracks: List[List[Track]] = []

    # Decode and process
    for frame_idx, t, frame in decode_video_cv2(input_path, cfg.fps):
        # Optional cap on processing time
        if cfg.max_seconds is not None and t > cfg.max_seconds:
            break
        ents = detector.infer(frame, next_entity_id_start=1)
        # Optional label filtering before tracking
        if cfg.allowed_labels:
            ents = [e for e in ents if e.label in cfg.allowed_labels]
        tracks = tracker.step(frame_idx, t, ents)

        window_frames.append(frame_idx)
        window_time.append(t)
        window_tracks.append(tracks)

        # When window is full, analyze and emit events
        if window_time and (t - window_time[0] >= window_s):
            start_t = window_time[0]
            end_t = window_time[-1]

            # Choose representative tracks (last snapshot)
            cur_tracks = window_tracks[-1] if window_tracks else []

            for tr in cur_tracks:
                speed, accel, direction = summarize_motion(tr)
                motion = Motion(direction=direction, speed=speed, accel=accel)
                action = heuristic_action(tr)
                ev = Event(
                    start=start_t,
                    end=end_t,
                    subjects=[Entity(id=tr.id, label=tr.label, bbox=tr.bbox, score=tr.score)],
                    action=action,
                    motion=motion,
                    provenance=Provenance(frames=(window_frames[0], window_frames[-1]), models={"detector": detector.name}),
                )
                events.append(ev)

            # Simple interactions
            for sid, oid, inter in infer_interactions(cur_tracks):
                s_track = next((tr for tr in cur_tracks if tr.id == sid), None)
                o_track = next((tr for tr in cur_tracks if tr.id == oid), None)
                if not s_track or not o_track:
                    continue
                ev = Event(
                    start=start_t,
                    end=end_t,
                    subjects=[Entity(id=s_track.id, label=s_track.label, bbox=s_track.bbox, score=s_track.score)],
                    objects=[Entity(id=o_track.id, label=o_track.label, bbox=o_track.bbox, score=o_track.score)],
                    interaction=inter,
                    provenance=Provenance(frames=(window_frames[0], window_frames[-1]), models={"detector": detector.name}),
                )
                events.append(ev)

            # Slide window
            # Remove frames until window meets stride
            while window_time and (window_time[-1] - window_time[0] >= stride_s):
                window_time.pop(0)
                window_frames.pop(0)
                window_tracks.pop(0)

    # Write outputs
    from .compose import compose_captions, summarize_events
    from .io import write_jsonl, write_srt, write_summary

    out_dir_p.mkdir(parents=True, exist_ok=True)
    write_jsonl(events, out_dir_p / "events.jsonl")
    captions = compose_captions(events)
    write_srt(captions, out_dir_p / "captions.srt")
    summary_data = summarize_events(events)
    write_summary(Summary(scenes=[summary_data]), out_dir_p / "summary.json")
