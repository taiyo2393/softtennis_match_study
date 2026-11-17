"""Run a compact, reproducible BallTrack threshold/background ablation.

This script deliberately reports candidate coverage and confidence, not ball
tracking accuracy.  Accuracy requires frame-level manually verified ball
positions, which must be evaluated separately on a held-out sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def prediction_path(project: Path, video_id: str, candidate_id: str, run_name: str) -> Path:
    root = project / "data" / "balltrack" / "racketvision" / video_id / candidate_id
    return root / "runs" / run_name / "balltrack_predictions.csv"


def summarise(path: Path, run_name: str, threshold: float, background_mode: str, median_samples: int) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    visible = [row for row in rows if int(row["visible"]) == 1]
    confidences = [float(row["confidence"]) for row in visible]
    runs = 0
    preceding_visible = False
    for row in rows:
        now_visible = int(row["visible"]) == 1
        if now_visible and not preceding_visible:
            runs += 1
        preceding_visible = now_visible
    return {
        "run_name": run_name,
        "threshold": threshold,
        "background_mode": background_mode,
        "median_samples_requested": "all" if median_samples == 0 else median_samples,
        "frames": len(rows),
        "candidate_frames": len(visible),
        "candidate_rate": round(len(visible) / len(rows), 6) if rows else "",
        "median_candidate_confidence": round(sorted(confidences)[len(confidences) // 2], 6) if confidences else "",
        "visible_runs": runs,
        "prediction_csv": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batchsize", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    runner = project / "scripts" / "34_run_racketvision_balltrack_pilot.py"

    # One-factor-at-a-time around the initial 0.50 / 120-frame setting.
    # 'all' tests the effect of estimating the static median from every clip
    # frame; it does not make this a different trained model.
    conditions = [
        ("threshold030_median120", 0.30, "median", 120),
        ("threshold040_median120", 0.40, "median", 120),
        ("threshold050_median120", 0.50, "median", 120),
        ("threshold060_median120", 0.60, "median", 120),
        ("threshold070_median120", 0.70, "median", 120),
        ("threshold050_median030", 0.50, "median", 30),
        ("threshold050_medianall", 0.50, "median", 0),
        ("threshold050_firstframe", 0.50, "first_frame", 1),
    ]
    summaries: list[dict[str, object]] = []
    for run_name, threshold, background_mode, median_samples in conditions:
        output = prediction_path(project, args.video_id, args.candidate_id, run_name)
        if not output.is_file() or args.overwrite:
            command = [
                sys.executable, str(runner), "--video-id", args.video_id,
                "--candidate-id", args.candidate_id, "--device", args.device,
                "--batchsize", str(args.batchsize), "--threshold", str(threshold),
                "--background-mode", background_mode, "--median-samples", str(median_samples),
                "--run-name", run_name,
            ]
            if args.overwrite:
                command.append("--overwrite")
            subprocess.run(command, check=True)
        summaries.append(summarise(output, run_name, threshold, background_mode, median_samples))

    analysis_dir = project / "analysis" / "balltrack_ablation"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / f"{args.video_id}_{args.candidate_id}_candidate_summary.csv"
    fields = list(summaries[0])
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    metadata = {
        "video_id": args.video_id,
        "candidate_id": args.candidate_id,
        "design": "one-factor-at-a-time around threshold=0.50 and 120 median samples",
        "interpretation": (
            "candidate_rate, confidence, and visible_runs are operational outputs only. "
            "They are not precision, recall, MDE, or impact-detection accuracy. "
            "Use a separately labelled held-out frame sample for accuracy evaluation."
        ),
        "conditions": summaries,
    }
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote ablation summary: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
