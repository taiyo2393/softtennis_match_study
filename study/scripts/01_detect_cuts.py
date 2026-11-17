"""Create conservative cut candidates from a video without decoding full 4K frames to disk.

This script detects visual discontinuities only.  It does not decide whether a
candidate is a replay or whether the following camera is suitable for analysis.
Those decisions remain in the segment-review stage.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path


TIME_RE = re.compile(r"pts_time:([0-9.]+)")
SCORE_RE = re.compile(r"lavfi\.scene_score=([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=15, help="scan fps; not analysis fps")
    parser.add_argument("--long-side", type=int, default=960)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--min-gap-sec", type=float, default=0.4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be in (0, 1)")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH")

    # fps and scale happen before scene scoring, so no proxy file is created.
    vf = (
        f"fps={args.fps},scale={args.long_side}:-2,"
        f"select='gt(scene\\,{args.threshold})',metadata=print"
    )
    command = [
        "ffmpeg", "-hide_banner", "-nostdin", "-i", str(args.input),
        "-an", "-vf", vf, "-f", "null", "-",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        raise RuntimeError(f"ffmpeg failed with exit code {completed.returncode}")

    candidates: list[tuple[float, float]] = []
    pending_time: float | None = None
    for line in completed.stderr.splitlines():
        time_match = TIME_RE.search(line)
        if time_match:
            pending_time = float(time_match.group(1))
            continue
        score_match = SCORE_RE.search(line)
        if score_match and pending_time is not None:
            score = float(score_match.group(1))
            if not candidates or pending_time - candidates[-1][0] >= args.min_gap_sec:
                candidates.append((pending_time, score))
            elif score > candidates[-1][1]:
                candidates[-1] = (pending_time, score)
            pending_time = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "video_id", "candidate_id", "timestamp_sec", "source_frame_60fps_approx",
            "scene_score", "scan_fps", "scan_long_side_px", "threshold",
            "review_status", "review_note",
        ])
        for number, (timestamp, score) in enumerate(candidates, start=1):
            writer.writerow([
                args.video_id, f"{args.video_id}_cut_{number:04d}", f"{timestamp:.6f}",
                round(timestamp * 60000 / 1001), f"{score:.6f}", args.fps,
                args.long_side, args.threshold, "unreviewed", "",
            ])
    print(f"Wrote {len(candidates)} cut candidates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
