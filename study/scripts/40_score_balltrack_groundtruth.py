"""Score BallTrack candidates against blinded frame-level ground truth.

This reports presence detection separately from localization.  A model cannot
hide a wrong point by counting it solely as a detected ball.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--split", choices=("validation", "test", "all"), default="all")
    parser.add_argument(
        "--run-name", default="default",
        help="BallTrack run to score. 'default' uses the initially stored prediction columns.",
    )
    parser.add_argument("--tolerance-px", type=float, default=16.0)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    labels = read(project / "annotations" / f"balltrack_groundtruth_{args.video_id}_pilot.csv")
    labels = [row for row in labels if args.split == "all" or row["evaluation_split"] == args.split]
    predicted: dict[tuple[str, int], dict[str, str]] = {}
    if args.run_name != "default":
        for candidate_id in sorted({row["candidate_id"] for row in labels}):
            path = (project / "data" / "balltrack" / "racketvision" / args.video_id / candidate_id /
                    "runs" / args.run_name / "balltrack_predictions.csv")
            if not path.is_file():
                raise SystemExit(f"Missing requested prediction run: {path}")
            predicted.update({(candidate_id, int(row["local_frame"])): row for row in read(path)})

    counts = {"total": len(labels), "evaluable": 0, "unreviewed": 0, "uncertain": 0,
              "presence_tp": 0, "presence_fp": 0, "presence_fn": 0, "presence_tn": 0,
              "localized_tp": 0, "localized_fp": 0, "localized_fn": 0, "localized_tn": 0,
              "wrong_localization": 0}
    errors: list[float] = []
    per_sample: list[dict[str, object]] = []
    for row in labels:
        state = row["truth_visibility"]
        if state in ("unreviewed", "uncertain"):
            counts[state] += 1
            continue
        counts["evaluable"] += 1
        true_visible = state == "visible"
        prediction = predicted.get((row["candidate_id"], int(row["local_frame"])), row)
        predicted_visible = int(prediction["visible"] if args.run_name != "default" else row["prediction_visible"]) == 1
        distance: float | None = None
        localized = False
        if true_visible and predicted_visible:
            x = prediction["x_px"] if args.run_name != "default" else row["prediction_x_px"]
            y = prediction["y_px"] if args.run_name != "default" else row["prediction_y_px"]
            distance = math.hypot(float(row["truth_x_px"]) - float(x), float(row["truth_y_px"]) - float(y))
            errors.append(distance)
            localized = distance <= args.tolerance_px

        if true_visible and predicted_visible: counts["presence_tp"] += 1
        elif true_visible: counts["presence_fn"] += 1
        elif predicted_visible: counts["presence_fp"] += 1
        else: counts["presence_tn"] += 1

        # Strict localization treats an incorrect coordinate as both a missed
        # true point and a spurious predicted point.
        if true_visible and predicted_visible and localized: counts["localized_tp"] += 1
        elif true_visible and predicted_visible:
            counts["localized_fp"] += 1; counts["localized_fn"] += 1; counts["wrong_localization"] += 1
        elif true_visible: counts["localized_fn"] += 1
        elif predicted_visible: counts["localized_fp"] += 1
        else: counts["localized_tn"] += 1
        per_sample.append({"sample_id": row["sample_id"], "evaluation_split": row["evaluation_split"],
                           "truth_visibility": state, "prediction_visible": int(predicted_visible),
                           "distance_px": round(distance, 4) if distance is not None else "",
                           "localized_within_tolerance": int(localized)})

    presence_precision = ratio(counts["presence_tp"], counts["presence_tp"] + counts["presence_fp"])
    presence_recall = ratio(counts["presence_tp"], counts["presence_tp"] + counts["presence_fn"])
    localized_precision = ratio(counts["localized_tp"], counts["localized_tp"] + counts["localized_fp"])
    localized_recall = ratio(counts["localized_tp"], counts["localized_tp"] + counts["localized_fn"])
    metrics = {
        "presence_precision": presence_precision, "presence_recall": presence_recall,
        "localized_precision": localized_precision, "localized_recall": localized_recall,
        "mean_distance_px_when_both_visible": round(sum(errors) / len(errors), 4) if errors else None,
        "median_distance_px_when_both_visible": round(sorted(errors)[len(errors) // 2], 4) if errors else None,
    }
    output_dir = project / "analysis" / "balltrack_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.split}_{args.run_name}"
    (output_dir / f"{args.video_id}_{suffix}_summary.json").write_text(json.dumps({
        "model": "RacketVision_MS-TrackNetV3_background_median_seq4", "run_name": args.run_name,
        "tolerance_px": args.tolerance_px,
        "important": "Scores exclude unreviewed/uncertain labels. Do not use test results to select the threshold.",
        "counts": counts, "metrics": metrics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / f"{args.video_id}_{suffix}_per_sample.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_sample[0]) if per_sample else ["sample_id"])
        writer.writeheader(); writer.writerows(per_sample)
    print(json.dumps({"counts": counts, "metrics": metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
