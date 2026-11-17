"""Evaluate only validation clips at fixed MS-TrackNetV3 threshold settings.

The test split is intentionally never processed by this script, so selecting a
threshold cannot overwrite or tune on its held-out predictions.
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
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    labels_path = project / "annotations" / f"balltrack_groundtruth_{args.video_id}_pilot.csv"
    with labels_path.open(newline="", encoding="utf-8") as file:
        validation = sorted({row["candidate_id"] for row in csv.DictReader(file) if row["evaluation_split"] == "validation"})
    if not validation:
        raise SystemExit("No validation rows found")
    runner = project / "scripts" / "34_run_racketvision_balltrack_pilot.py"
    scorer = project / "scripts" / "40_score_balltrack_groundtruth.py"
    for threshold in (0.30, 0.40, 0.50, 0.60, 0.70):
        run_name = f"threshold{int(threshold * 100):03d}_median120"
        for index, candidate_id in enumerate(validation, start=1):
            output = (project / "data" / "balltrack" / "racketvision" / args.video_id / candidate_id /
                      "runs" / run_name / "balltrack_predictions.csv")
            if output.is_file() and not args.overwrite:
                print(f"[{run_name} {index}/{len(validation)}] Skip {candidate_id}", flush=True)
                continue
            command = [sys.executable, str(runner), "--video-id", args.video_id,
                       "--candidate-id", candidate_id, "--device", args.device,
                       "--threshold", str(threshold), "--median-samples", "120",
                       "--run-name", run_name]
            if args.overwrite:
                command.append("--overwrite")
            print(f"[{run_name} {index}/{len(validation)}] Run {candidate_id}", flush=True)
            subprocess.run(command, check=True)
        subprocess.run([sys.executable, str(scorer), "--video-id", args.video_id,
                        "--split", "validation", "--run-name", run_name], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
