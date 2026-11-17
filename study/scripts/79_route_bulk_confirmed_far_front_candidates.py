"""Record a human bulk role confirmation and create far-front direction jobs.

This is intentionally not an automatic role classifier.  It records the
annotator's explicit confirmation that every queued ambiguous near-side hitter
was the rear player, then creates a separate direction-label queue.  The
resulting far-front cohort remains distinct from the near-front cohort.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROLE_FIELDS = [
    "candidate_id", "source_candidate_id", "video_id", "impact_id", "impact_source_frame",
    "impact_timestamp_sec", "target_front_camera_end", "near_hitter_role_at_impact",
    "review_status", "role_source", "routing_source", "routing_note", "review_note",
    "annotator_id", "role_definition",
]
DIRECTION_FIELDS = [
    "candidate_id", "source_candidate_id", "video_id", "impact_id",
    "impact_source_frame", "impact_timestamp_sec", "source_clip_path",
    "impact_local_frame", "target_front_camera_end", "routing_source",
    "routing_note",
]
BULK_NOTE = "ユーザー一括確認：全件、手前後衛が打球"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--role-review", type=Path, required=True)
    parser.add_argument("--direction-queue", type=Path, required=True)
    parser.add_argument("--direction-cohort-id", default="far_front_direction_02")
    args = parser.parse_args()
    if not args.queue.is_file():
        raise SystemExit(f"Role-resolution queue not found: {args.queue}")
    candidates = read(args.queue)
    if not candidates:
        raise SystemExit("Role-resolution queue is empty")
    if len({row["impact_id"] for row in candidates}) != len(candidates):
        raise SystemExit("Duplicate impact_id in role-resolution queue")

    role_rows: list[dict[str, str]] = []
    direction_rows: list[dict[str, str]] = []
    for rank, candidate in enumerate(candidates, start=1):
        role_rows.append({
            "candidate_id": candidate["candidate_id"],
            "source_candidate_id": candidate["source_candidate_id"],
            "video_id": candidate["video_id"],
            "impact_id": candidate["impact_id"],
            "impact_source_frame": candidate["impact_source_frame"],
            "impact_timestamp_sec": candidate["impact_timestamp_sec"],
            "target_front_camera_end": "far",
            "near_hitter_role_at_impact": "near_rear",
            "review_status": "confirmed",
            "role_source": "human_bulk_confirmation",
            "routing_source": candidate["routing_source"],
            "routing_note": candidate["routing_note"],
            "review_note": BULK_NOTE,
            "annotator_id": "human_video_review",
            "role_definition": "near_side_striker_relative_depth_at_impact_v1",
        })
        direction_rows.append({
            "candidate_id": f"{candidate['video_id']}_{args.direction_cohort_id}_{rank:03d}",
            "source_candidate_id": candidate["source_candidate_id"],
            "video_id": candidate["video_id"],
            "impact_id": candidate["impact_id"],
            "impact_source_frame": candidate["impact_source_frame"],
            "impact_timestamp_sec": candidate["impact_timestamp_sec"],
            "source_clip_path": candidate["source_clip_path"],
            "impact_local_frame": candidate["impact_local_frame"],
            "target_front_camera_end": "far",
            "routing_source": "human_bulk_role_confirmation",
            "routing_note": BULK_NOTE,
        })
    write(args.role_review, ROLE_FIELDS, role_rows)
    write(args.direction_queue, DIRECTION_FIELDS, direction_rows)
    print(
        f"Recorded {len(role_rows)} human-confirmed near-rear roles and wrote "
        f"{len(direction_rows)} far-front direction candidates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
