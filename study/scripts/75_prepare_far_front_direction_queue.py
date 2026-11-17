"""Route already reviewed near-rear impacts to a separate far-front cohort.

The near-front review must not be relabelled or overwritten.  This script uses
only rows whose human exclusion note explicitly says that the *near rear*
player hit the ball.  They become candidates for a new analysis in which the
far-court front player is the target.  The original 1080p clips are reused.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROUTING_NOTE = "手前後衛がボールを打っている"
FIELDS = [
    "candidate_id", "source_candidate_id", "video_id", "impact_id",
    "impact_source_frame", "impact_timestamp_sec", "source_clip_path",
    "impact_local_frame", "target_front_camera_end", "routing_source",
    "routing_note",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--near-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-id", default="far_front_direction_01")
    args = parser.parse_args()
    if not args.near_review.is_file():
        raise SystemExit(f"Near-front review not found: {args.near_review}")
    project = Path(__file__).resolve().parents[1]
    selected = [
        row for row in read(args.near_review)
        if row.get("review_status") == "excluded" and row.get("review_note") == ROUTING_NOTE
    ]
    if not selected:
        raise SystemExit("No human-confirmed near-rear hitter rows found")
    selected.sort(key=lambda row: int(row["impact_source_frame"]))
    rows: list[dict[str, str]] = []
    for rank, row in enumerate(selected, start=1):
        source_id = row["candidate_id"]
        clips = list((project / "data" / "segments").glob(f"full_match_*/{row['video_id']}/{source_id}.mp4"))
        if len(clips) != 1:
            raise SystemExit(f"Expected exactly one reusable clip for {source_id}; found {len(clips)}")
        # The source candidate was produced from a queue whose clip-local
        # impact index is fixed at 12 for the 0.2 s pre-window (59.94 fps).
        # Derive it from its queue instead of assuming a new timing label.
        queue_matches = list((project / "data" / "segments").glob(f"{row['video_id']}_full_match_*_queue.csv"))
        queue_row = None
        for queue_path in queue_matches:
            queue_row = next((item for item in read(queue_path) if item.get("candidate_id") == source_id), None)
            if queue_row:
                break
        if queue_row is None:
            raise SystemExit(f"Original queue entry not found for {source_id}")
        rows.append({
            "candidate_id": f"{row['video_id']}_{args.cohort_id}_{rank:03d}",
            "source_candidate_id": source_id,
            "video_id": row["video_id"], "impact_id": row["impact_id"],
            "impact_source_frame": row["impact_source_frame"],
            "impact_timestamp_sec": row["impact_timestamp_sec"],
            "source_clip_path": str(clips[0].resolve()),
            "impact_local_frame": queue_row["impact_local_frame"],
            "target_front_camera_end": "far",
            "routing_source": "human_near_front_review",
            "routing_note": ROUTING_NOTE,
        })
    if len({row["impact_id"] for row in rows}) != len(rows):
        raise SystemExit("Duplicate impact_id in far-front candidate queue")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} routed far-front candidates: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
