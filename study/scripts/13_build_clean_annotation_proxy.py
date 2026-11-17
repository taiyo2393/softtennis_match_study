"""Build a clean 1080p60 proxy for baseline manual annotation."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--segment-id", required=True)
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]
    with (project / "data" / "segments" / "analysis_ready_main_camera_segments.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    row = next((item for item in rows if item["segment_id"] == args.segment_id), None)
    if row is None: raise SystemExit(f"Segment not found: {args.segment_id}")
    start, end = float(row["start_timestamp_sec"]), float(row["end_timestamp_sec"])
    output_dir = project / "data" / "proxy" / "annotation"; output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.video_id}_{args.segment_id}_clean_1080p60.mp4"
    # -ss after input prioritizes temporal correspondence to the source video.
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(args.input), "-ss", f"{start:.6f}",
                    "-t", f"{end-start:.6f}", "-map", "0:v:0", "-map", "0:a?", "-vf", "scale=1920:-2",
                    "-r", "60000/1001", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-movflags", "+faststart", str(output)], check=True)
    print(output); return 0


if __name__ == "__main__": raise SystemExit(main())
