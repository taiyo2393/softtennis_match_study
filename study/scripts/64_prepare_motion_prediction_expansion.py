"""Create a separate reproducible extension batch without touching pilot data.

The selected windows are fixed-main-camera play windows that have not already
been screened in ``annotations/motion_prediction_events.csv``.  This only
creates review clips; it does not determine an impact, a direction, or a
front-player trajectory.
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
    parser.add_argument("--batch-id", required=True, help="For example: expand_01")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--min-duration-sec", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    if not args.input.is_file():
        raise SystemExit(f"Input video not found: {args.input}")
    if not args.batch_id.replace("_", "").isalnum():
        raise SystemExit("--batch-id may use letters, numbers, and underscores only")

    project = Path(__file__).resolve().parents[1]
    segment_dir = project / "data" / "segments"
    source = read(segment_dir / f"{args.video_id}_reviewed_play_windows.csv")
    events_path = project / "annotations" / "motion_prediction_events.csv"
    used_play_windows = {row["play_window_id"] for row in read(events_path) if row.get("play_window_id")}
    eligible = [
        row for row in source
        if row["end_status"] == "human_confirmed_departure_after_play"
        and float(row["duration_sec"]) >= args.min_duration_sec
        and row["play_window_id"] not in used_play_windows
    ]
    if len(eligible) < args.limit:
        raise SystemExit(
            f"Only {len(eligible)} unused eligible windows; requested {args.limit}. "
            "Use a smaller limit or review another source video."
        )

    batch_stem = f"{args.video_id}_motion_prediction_{args.batch_id}"
    queue_path = segment_dir / f"{batch_stem}_queue.csv"
    clip_dir = segment_dir / f"motion_prediction_{args.batch_id}" / args.video_id
    if queue_path.exists() and not args.overwrite:
        raise SystemExit(f"Batch queue already exists: {queue_path}. Use another --batch-id; do not overwrite reviewed work.")
    clip_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    selected = rng.sample(eligible, args.limit)
    selected.sort(key=lambda row: float(row["source_start_timestamp_sec"]))
    fields = [
        "candidate_id", "video_id", "play_window_id", "source_start_timestamp_sec",
        "source_end_timestamp_sec", "duration_sec", "selection_seed", "selection_rank",
        "screening_status", "screening_note",
    ]
    rows: list[dict[str, str]] = []
    for rank, source_row in enumerate(selected, start=1):
        candidate_id = f"{args.video_id}_motion_{args.batch_id}_{rank:03d}"
        row = {
            "candidate_id": candidate_id,
            "video_id": args.video_id,
            "play_window_id": source_row["play_window_id"],
            "source_start_timestamp_sec": source_row["source_start_timestamp_sec"],
            "source_end_timestamp_sec": source_row["source_end_timestamp_sec"],
            "duration_sec": source_row["duration_sec"],
            "selection_seed": str(args.seed),
            "selection_rank": str(rank),
            "screening_status": "unreviewed",
            "screening_note": "",
        }
        rows.append(row)
        clip = clip_dir / f"{candidate_id}.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", row["source_start_timestamp_sec"], "-i", str(args.input),
            "-t", row["duration_sec"], "-map", "0:v:0", "-map", "0:a?",
            "-vf", "scale=1920:-2", "-r", "60000/1001",
            "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-movflags", "+faststart", str(clip),
        ], check=True)
    write(queue_path, fields, rows)
    print(f"Prepared {len(rows)} unused windows: {queue_path}")
    print(f"Review clips: {clip_dir}")
    print("Next: run script 37 with --queue-path and --clip-dir, then script 45 and script 46 with those paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
