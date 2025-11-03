from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .schemas import Event, CaptionLine, Summary


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(events: Iterable[Event], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")


def write_summary(summary: Summary, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary.__dict__, f, ensure_ascii=False, indent=2)


def _format_ts(seconds: float) -> str:
    # For SRT: HH:MM:SS,mmm
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(captions: List[CaptionLine], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, c in enumerate(captions, start=1):
            f.write(f"{idx}\n")
            f.write(f"{_format_ts(c.t_start)} --> {_format_ts(c.t_end)}\n")
            f.write(c.text.strip() + "\n\n")


def write_vtt(captions: List[CaptionLine], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for c in captions:
            # VTT timestamp format: HH:MM:SS.mmm
            def vtt_ts(t: float) -> str:
                ms = int(round((t - int(t)) * 1000))
                s = int(t) % 60
                m = (int(t) // 60) % 60
                h = int(t) // 3600
                return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

            f.write(f"{vtt_ts(c.t_start)} --> {vtt_ts(c.t_end)}\n")
            f.write(c.text.strip() + "\n\n")

