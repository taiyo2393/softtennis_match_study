"""Create short low-resolution clips around main-camera departure candidates."""

from __future__ import annotations

import argparse, csv, subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True); parser.add_argument("--video-id", required=True)
    parser.add_argument("--before-sec", type=float, default=3.0); parser.add_argument("--after-sec", type=float, default=3.0)
    parser.add_argument("--boundary-id", action="append", default=[], help="Optional ID(s) to regenerate")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]; segments = project / "data" / "segments"
    with (segments / f"{args.video_id}_rally_boundary_candidates.csv").open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["candidate_type"] == "main_camera_departure_candidate"]
    if args.boundary_id:
        wanted = set(args.boundary_id); rows = [row for row in rows if row["boundary_id"] in wanted]
        missing = wanted - {row["boundary_id"] for row in rows}
        if missing: raise SystemExit(f"Unknown departure boundary IDs: {sorted(missing)}")
    out_dir = project / "data" / "segments" / "boundary_review" / args.video_id; out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for row in rows:
        output = out_dir / f"{row['boundary_id']}.mp4"
        if output.exists() and not args.overwrite: continue
        start = max(0.0, float(row["timestamp_sec"]) - args.before_sec); duration = args.before_sec + args.after_sec
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.6f}", "-i", str(args.input),
                        "-t", f"{duration:.6f}", "-map", "0:v:0", "-map", "0:a?", "-vf", "scale=960:-2,drawbox=x=0:y=0:w=iw:h=ih:color=red@0.95:t=12:enable='between(t,2.8,3.2)'", "-r", "15", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(output)], check=True)
        made += 1
    print(f"Created {made} boundary-review clips in {out_dir}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
