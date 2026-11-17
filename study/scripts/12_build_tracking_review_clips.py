"""Create exact short H.264 clips for interval-level tracking review."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--segment-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); project = Path(__file__).resolve().parents[1]
    stem = f"{args.video_id}_{args.segment_id}_person_tracking_pilot"
    tracking_dir = project / "data" / "tracking"
    source = tracking_dir / f"{stem}_browser.mp4"
    if not source.exists(): source = tracking_dir / f"{stem}.mp4"
    review = project / "annotations" / f"{args.video_id}_{args.segment_id}_tracking_review.csv"
    params = json.loads((tracking_dir / f"{stem}.json").read_text(encoding="utf-8"))
    with review.open(newline="", encoding="utf-8") as file: rows = list(csv.DictReader(file))
    output_dir = tracking_dir / "review_clips" / f"{args.video_id}_{args.segment_id}"; output_dir.mkdir(parents=True, exist_ok=True)
    base = float(params["start_timestamp_sec"])
    created = 0
    for row in rows:
        output = output_dir / f"{row['chunk_id']}.mp4"
        if output.exists() and not args.overwrite: continue
        start = float(row["start_timestamp_sec"]) - base
        duration = float(row["end_timestamp_sec"]) - float(row["start_timestamp_sec"])
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.6f}", "-i", str(source),
                        "-t", f"{duration:.6f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], check=True)
        created += 1
    print(f"Created {created} clips in {output_dir}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
