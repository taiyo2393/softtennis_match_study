"""Build bounded review videos and audio-cue proposals for unresolved starts.

Audio proposals are only navigation aids.  They are not asserted to be racket
impacts and cannot confirm a rally start without visual review.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    directory = project / "data" / "segments"
    windows = [r for r in read(directory / f"{args.video_id}_rally_window_candidates.csv") if r["window_status"] == "excluded_or_unresolved"]
    onsets = read(directory / f"{args.video_id}_audio_onset_candidates.csv")
    output_dir = directory / "unresolved_start_review" / args.video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    review_rows = []
    for row in windows:
        start = float(row["source_start_timestamp_sec"])
        end = float(row["source_end_timestamp_sec"])
        # Candidate sounds with the greatest detector scores: navigational hints only.
        sounds = [o for o in onsets if start <= float(o["timestamp_sec"]) <= end]
        sounds.sort(key=lambda o: float(o["onset_score"]), reverse=True)
        hints = ";".join(f"{float(o['timestamp_sec']):.3f}" for o in sounds[:5])
        review_rows.append({
            "video_id": args.video_id, "window_id": row["window_id"],
            "source_start_timestamp_sec": f"{start:.6f}", "source_end_timestamp_sec": f"{end:.6f}",
            "duration_sec": f"{end-start:.6f}", "audio_cue_times_sec": hints,
            "review_status": "unreviewed", "human_rally_start_timestamp_sec": "", "review_note": "",
        })
        output = output_dir / f"{row['window_id']}.mp4"
        if output.exists() and not args.overwrite:
            continue
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.6f}",
            "-i", str(args.input), "-t", f"{end-start:.6f}", "-map", "0:v:0", "-map", "0:a?",
            "-vf", "scale=960:-2", "-r", "15", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-movflags", "+faststart", str(output),
        ], check=True)
    fields = list(review_rows[0]) if review_rows else ["video_id", "window_id", "source_start_timestamp_sec", "source_end_timestamp_sec", "duration_sec", "audio_cue_times_sec", "review_status", "human_rally_start_timestamp_sec", "review_note"]
    with (directory / f"{args.video_id}_unresolved_start_review.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"Prepared {len(review_rows)} unresolved-start reviews in {output_dir}")


if __name__ == "__main__":
    main()
