"""Run the fixed MS-TrackNetV3 candidate generator on every clip in a queue.

Each clip is independent. Existing predictions for the specified named run are
retained unless ``--overwrite`` is supplied, making interrupted batch runs
safe to resume.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument(
        "--queue-path", type=Path, default=None,
        help="CSV queue to process. Defaults to the original pilot queue.",
    )
    parser.add_argument(
        "--clip-dir", type=Path, default=None,
        help="Directory containing candidate clips. Defaults to the original pilot clip directory.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--median-samples", type=int, default=120)
    parser.add_argument(
        "--run-name", default="default",
        help="Named BallTrack condition. It must match the --run-name used later by script 45.",
    )
    parser.add_argument(
        "--background-mode", choices=("median", "first_frame"), default="median",
        help="The released model's median background is the default; first_frame is diagnostic only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    queue = args.queue_path or (project / "data" / "segments" / f"{args.video_id}_motion_prediction_pilot_queue.csv")
    queue = queue.resolve()
    clip_dir = args.clip_dir or (project / "data" / "segments" / "motion_prediction_pilot" / args.video_id)
    clip_dir = clip_dir.resolve()
    if not queue.is_file():
        raise SystemExit(f"Queue not found: {queue}")
    if not clip_dir.is_dir():
        raise SystemExit(f"Clip directory not found: {clip_dir}")
    runner = project / "scripts" / "34_run_racketvision_balltrack_pilot.py"
    with queue.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    completed = 0
    for index, row in enumerate(rows, start=1):
        candidate_id = row["candidate_id"]
        candidate_dir = project / "data" / "balltrack" / "racketvision" / args.video_id / candidate_id
        output = candidate_dir / "balltrack_predictions.csv" if args.run_name == "default" else candidate_dir / "runs" / args.run_name / "balltrack_predictions.csv"
        if output.is_file() and not args.overwrite:
            print(f"[{index}/{len(rows)}] Skip existing: {candidate_id}", flush=True)
            completed += 1
            continue
        print(f"[{index}/{len(rows)}] Run: {candidate_id}", flush=True)
        command = [
            sys.executable, str(runner), "--video-id", args.video_id,
            "--candidate-id", candidate_id, "--device", args.device,
            "--threshold", str(args.threshold), "--median-samples", str(args.median_samples),
            "--queue-path", str(queue), "--clip-dir", str(clip_dir),
            "--run-name", args.run_name, "--background-mode", args.background_mode,
        ]
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)
        completed += 1
    print(f"Completed {completed}/{len(rows)} clips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
