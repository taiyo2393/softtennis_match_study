"""Create 60-fps local clips to correct ambiguous serve-start labels.

The previous field was entered while the UI described a motion start.  It is
retained as a navigation anchor; this script creates a separate, frame-level
valid-serve-impact label and never overwrites the earlier review.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


POSITIVE = {"complete_valid_rally", "complete_rally_after_first_serve_fault"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--before-sec", type=float, default=1.5)
    parser.add_argument("--after-sec", type=float, default=3.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    directory = project / "data" / "segments"
    source = [r for r in read(directory / f"{args.video_id}_unresolved_start_review.csv") if r["review_status"] in POSITIVE]
    review_path = directory / f"{args.video_id}_serve_impact_review.csv"
    old = {r["window_id"]: r for r in read(review_path)} if review_path.exists() else {}
    clip_dir = directory / "serve_impact_review" / args.video_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    output = []
    for row in source:
        anchor = float(row["human_rally_start_timestamp_sec"])
        clip_start = max(0.0, anchor - args.before_sec)
        clip_end = anchor + args.after_sec
        previous = old.get(row["window_id"], {})
        output.append({
            "video_id": args.video_id, "window_id": row["window_id"],
            "legacy_motion_start_timestamp_sec": f"{anchor:.6f}",
            "clip_start_timestamp_sec": f"{clip_start:.6f}", "clip_end_timestamp_sec": f"{clip_end:.6f}",
            "impact_review_status": previous.get("impact_review_status", "unreviewed"),
            "human_valid_serve_impact_timestamp_sec": previous.get("human_valid_serve_impact_timestamp_sec", ""),
            "impact_review_note": previous.get("impact_review_note", ""),
        })
        clip = clip_dir / f"{row['window_id']}.mp4"
        if clip.exists() and not args.overwrite:
            continue
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{clip_start:.6f}", "-i", str(args.input),
            "-t", f"{args.before_sec + args.after_sec:.6f}", "-map", "0:v:0", "-map", "0:a?",
            "-vf", "scale=1920:-2", "-r", "60000/1001", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-movflags", "+faststart", str(clip),
        ], check=True)
    fields = list(output[0]) if output else []
    with review_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(output)
    print(f"Prepared {len(output)} 60-fps serve-impact review clips in {clip_dir}")


if __name__ == "__main__":
    main()
