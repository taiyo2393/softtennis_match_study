"""Run the frozen validation-selected threshold once on the held-out test clips."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    labels_path = project / "annotations" / f"balltrack_groundtruth_{args.video_id}_pilot.csv"
    with labels_path.open(newline="", encoding="utf-8") as file:
        test = sorted({row["candidate_id"] for row in csv.DictReader(file) if row["evaluation_split"] == "test"})
    if not test:
        raise SystemExit("No test rows found")
    run_name = f"threshold{int(args.threshold * 100):03d}_median120"
    runner = project / "scripts" / "34_run_racketvision_balltrack_pilot.py"
    for index, candidate_id in enumerate(test, start=1):
        output = (project / "data" / "balltrack" / "racketvision" / args.video_id / candidate_id /
                  "runs" / run_name / "balltrack_predictions.csv")
        if output.is_file():
            print(f"[{run_name} test {index}/{len(test)}] Skip {candidate_id}", flush=True)
            continue
        print(f"[{run_name} test {index}/{len(test)}] Run {candidate_id}", flush=True)
        subprocess.run([
            sys.executable, str(runner), "--video-id", args.video_id,
            "--candidate-id", candidate_id, "--device", args.device,
            "--threshold", str(args.threshold), "--median-samples", "120",
            "--run-name", run_name,
        ], check=True)
    scorer = project / "scripts" / "40_score_balltrack_groundtruth.py"
    subprocess.run([
        sys.executable, str(scorer), "--video-id", args.video_id,
        "--split", "test", "--run-name", run_name,
    ], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
