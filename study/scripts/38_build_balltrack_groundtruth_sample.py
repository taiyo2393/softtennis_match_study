"""Create a blinded, time-stratified frame sample for BallTrack evaluation.

Predictions are retained in the CSV for later scoring, but the review UI must
not show them while the reviewer records the ground truth.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "sample_id", "evaluation_split", "candidate_id", "play_window_id", "sample_order",
    "local_frame", "source_frame", "source_timestamp_sec", "sampling_method",
    "prediction_visible", "prediction_x_px", "prediction_y_px", "prediction_confidence",
    "truth_visibility", "truth_x_px", "truth_y_px", "reviewer_id", "label_confidence", "notes",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evenly_spaced_indices(frame_count: int, samples: int) -> list[int]:
    if frame_count < samples:
        raise ValueError(f"Clip has {frame_count} frames but needs {samples} samples")
    # One frame from each equal temporal bin; deterministic and repeatable.
    return [min(frame_count - 1, int((index + 0.5) * frame_count / samples)) for index in range(samples)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--samples-per-clip", type=int, default=15)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    queue = read(project / "data" / "segments" / f"{args.video_id}_motion_prediction_pilot_queue.csv")
    output = project / "annotations" / f"balltrack_groundtruth_{args.video_id}_pilot.csv"
    existing = {row["sample_id"]: row for row in read(output)} if output.is_file() else {}
    rows: list[dict[str, str]] = []

    for queue_index, item in enumerate(queue):
        candidate_id = item["candidate_id"]
        prediction_path = project / "data" / "balltrack" / "racketvision" / args.video_id / candidate_id / "balltrack_predictions.csv"
        if not prediction_path.is_file():
            raise SystemExit(f"Missing BallTrack prediction: {prediction_path}")
        predictions = read(prediction_path)
        by_frame = {int(row["local_frame"]): row for row in predictions}
        for order, frame in enumerate(evenly_spaced_indices(len(predictions), args.samples_per_clip), start=1):
            prediction = by_frame[frame]
            sample_id = f"{candidate_id}_frame_{frame:06d}"
            prior = existing.get(sample_id, {})
            row = {
                "sample_id": sample_id,
                "evaluation_split": "validation" if queue_index < len(queue) // 2 else "test",
                "candidate_id": candidate_id,
                "play_window_id": item["play_window_id"],
                "sample_order": str(order),
                "local_frame": str(frame),
                "source_frame": prediction["source_frame"],
                "source_timestamp_sec": prediction["source_timestamp_sec"],
                "sampling_method": f"time_stratified_{args.samples_per_clip}_per_clip",
                # Store for later scoring, but never expose during ground-truth review.
                "prediction_visible": prediction["visible"],
                "prediction_x_px": prediction["x_px"],
                "prediction_y_px": prediction["y_px"],
                "prediction_confidence": prediction["confidence"],
                "truth_visibility": prior.get("truth_visibility", "unreviewed"),
                "truth_x_px": prior.get("truth_x_px", ""),
                "truth_y_px": prior.get("truth_y_px", ""),
                "reviewer_id": prior.get("reviewer_id", ""),
                "label_confidence": prior.get("label_confidence", ""),
                "notes": prior.get("notes", ""),
            }
            rows.append(row)
    write(output, rows)
    print(f"Wrote {len(rows)} blinded evaluation samples: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
