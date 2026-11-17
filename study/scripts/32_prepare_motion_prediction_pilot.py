"""Create a reproducible 20-window pilot queue for motion-prediction events.

This is a screening queue, not a ground-truth event detector.  It samples
fixed-main-camera play windows and makes review clips.  The reviewer must still
identify a rear-court groundstroke and verify its impact frame.
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--min-duration-sec", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    if not args.input.is_file():
        raise SystemExit(f"Input video not found: {args.input}")

    project = Path(__file__).resolve().parents[1]
    segment_dir = project / "data" / "segments"
    source = read(segment_dir / f"{args.video_id}_reviewed_play_windows.csv")
    eligible = [
        row for row in source
        if row["end_status"] == "human_confirmed_departure_after_play"
        and float(row["duration_sec"]) >= args.min_duration_sec
    ]
    if len(eligible) < args.limit:
        raise SystemExit(f"Only {len(eligible)} eligible play windows; requested {args.limit}")

    rng = random.Random(args.seed)
    selected = rng.sample(eligible, args.limit)
    selected.sort(key=lambda row: float(row["source_start_timestamp_sec"]))
    existing_path = segment_dir / f"{args.video_id}_motion_prediction_pilot_queue.csv"
    previous = {row["candidate_id"]: row for row in read(existing_path)} if existing_path.exists() else {}
    clip_dir = segment_dir / "motion_prediction_pilot" / args.video_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "candidate_id", "video_id", "play_window_id", "source_start_timestamp_sec",
        "source_end_timestamp_sec", "duration_sec", "selection_seed", "selection_rank",
        "screening_status", "screening_note",
    ]
    rows: list[dict[str, str]] = []
    for rank, source_row in enumerate(selected, start=1):
        candidate_id = f"{args.video_id}_motion_pilot_{rank:03d}"
        row = {
            "candidate_id": candidate_id,
            "video_id": args.video_id,
            "play_window_id": source_row["play_window_id"],
            "source_start_timestamp_sec": source_row["source_start_timestamp_sec"],
            "source_end_timestamp_sec": source_row["source_end_timestamp_sec"],
            "duration_sec": source_row["duration_sec"],
            "selection_seed": str(args.seed),
            "selection_rank": str(rank),
            "screening_status": previous.get(candidate_id, {}).get("screening_status", "unreviewed"),
            "screening_note": previous.get(candidate_id, {}).get("screening_note", ""),
        }
        rows.append(row)
        clip = clip_dir / f"{candidate_id}.mp4"
        if clip.exists() and not args.overwrite:
            continue
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", row["source_start_timestamp_sec"], "-i", str(args.input),
            "-t", row["duration_sec"], "-map", "0:v:0", "-map", "0:a?",
            "-vf", "scale=1920:-2", "-r", "60000/1001",
            "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-movflags", "+faststart", str(clip),
        ], check=True)
    write(existing_path, fields, rows)
    print(f"Prepared {len(rows)} reproducibly selected pilot windows: {existing_path}")
    print(f"Review clips: {clip_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
