"""Merge non-overlapping far-front direction reviews without relabelling rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "candidate_id", "source_candidate_id", "video_id", "impact_id", "impact_source_frame",
    "impact_timestamp_sec", "target_front_camera_end", "actual_direction_target_front_view",
    "review_status", "direction_source", "routing_source", "routing_note", "review_note",
    "annotator_id", "label_definition", "source_review_batch",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, str]] = []
    seen_impact: set[str] = set()
    seen_candidate: set[str] = set()
    for path in args.input:
        if not path.is_file():
            raise SystemExit(f"Input review not found: {path}")
        for row in read(path):
            impact_id = row.get("impact_id", "")
            candidate_id = row.get("candidate_id", "")
            if not impact_id or not candidate_id:
                raise SystemExit(f"Missing impact_id or candidate_id in {path}")
            if impact_id in seen_impact or candidate_id in seen_candidate:
                raise SystemExit(f"Duplicate impact/candidate across review batches: {impact_id} / {candidate_id}")
            seen_impact.add(impact_id)
            seen_candidate.add(candidate_id)
            rows.append({**{field: row.get(field, "") for field in FIELDS}, "source_review_batch": path.name})
    rows.sort(key=lambda row: int(row["impact_source_frame"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    confirmed = sum(row["review_status"] == "confirmed" for row in rows)
    print(f"Wrote {len(rows)} rows ({confirmed} confirmed) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
