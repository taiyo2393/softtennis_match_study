"""Create human-review rows for pre-impact ball locations of included events.

BallTrack values are displayed only as AI candidates. The human review fields
remain blank until a reviewer confirms the ball centre on the exact frame.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FPS = 60000 / 1001
INPUT_OFFSETS = (-0.50, -0.30, -0.10)
FIELDS = [
    "sample_id", "event_id", "video_id", "candidate_id", "input_offset_sec",
    "source_frame", "source_timestamp_sec", "local_frame",
    "ai_visible", "ai_x_px", "ai_y_px", "ai_confidence", "ai_model",
    "human_visibility", "human_x_px", "human_y_px", "review_method", "label_confidence",
    "reviewer_id", "notes",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    events = [
        row for row in read(project / "annotations" / "motion_prediction_events.csv")
        if row["video_id"] == args.video_id and row["inclusion_status"] == "included"
    ]
    usable_tracks = {
        row["event_id"] for row in read(project / "annotations" / "motion_prediction_track_review.csv")
        if row["tracking_status"] == "usable"
    }
    events = [row for row in events if row["event_id"] in usable_tracks]
    queue = {
        row["candidate_id"]: row
        for row in read(project / "data" / "segments" / f"{args.video_id}_motion_prediction_pilot_queue.csv")
    }
    output = project / "annotations" / f"front_motion_ball_inputs_{args.video_id}.csv"
    existing = {row["sample_id"]: row for row in read(output)} if output.is_file() and not args.overwrite else {}
    rows: list[dict[str, str]] = []
    for event in events:
        event_id = event["event_id"]
        candidate_id = event_id.removesuffix("_event_001")
        if candidate_id not in queue:
            raise RuntimeError(f"No queue record for {event_id}")
        predictions_path = project / "data" / "balltrack" / "racketvision" / args.video_id / candidate_id / "balltrack_predictions.csv"
        predictions = read(predictions_path) if predictions_path.is_file() else []
        by_source_frame = {int(row["source_frame"]): row for row in predictions}
        impact_frame = int(event["impact_source_frame"])
        for offset in INPUT_OFFSETS:
            source_frame = round(impact_frame + offset * FPS)
            local_frame = round(source_frame - float(queue[candidate_id]["source_start_timestamp_sec"]) * FPS)
            candidate = by_source_frame.get(source_frame)
            sample_id = f"{event_id}_m{abs(round(offset * 1000)):03d}"
            row = {
                "sample_id": sample_id, "event_id": event_id, "video_id": args.video_id,
                "candidate_id": candidate_id, "input_offset_sec": f"{offset:.2f}",
                "source_frame": str(source_frame), "source_timestamp_sec": f"{source_frame / FPS:.6f}",
                "local_frame": str(local_frame),
                "ai_visible": candidate.get("visible", "") if candidate else "",
                "ai_x_px": candidate.get("x_px", "") if candidate else "",
                "ai_y_px": candidate.get("y_px", "") if candidate else "",
                "ai_confidence": candidate.get("confidence", "") if candidate else "",
                "ai_model": candidate.get("model_name", "") if candidate else "",
                "human_visibility": "unreviewed", "human_x_px": "", "human_y_px": "", "review_method": "",
                "label_confidence": "", "reviewer_id": "", "notes": "",
            }
            if sample_id in existing:
                row.update({key: existing[sample_id].get(key, "") for key in FIELDS if key.startswith("human_") or key in {"label_confidence", "reviewer_id", "notes"}})
            rows.append(row)
    write(output, rows)
    print(f"Created {len(rows)} ball-input review rows for {len(events)} included events: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
