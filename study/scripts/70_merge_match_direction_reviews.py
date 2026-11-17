"""Create one auditable, canonical direction review from completed batches.

The reviewer batches are deliberately kept immutable.  This script validates
that completed rows are unique by impact_id, then concatenates them into a
single input for downstream tracking and pose extraction.  It does not infer
or change any human labels.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED = {
    "candidate_id", "impact_id", "video_id", "impact_source_frame",
    "actual_direction_front_view", "review_status",
}
VALID = {
    ("confirmed", "front_left"),
    ("confirmed", "front_right"),
    ("excluded", "excluded"),
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = REQUIRED.difference(fields)
        if missing:
            raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
        return fields, list(reader)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="Completed front-direction review CSV; repeat per batch.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fields: list[str] | None = None
    merged: list[dict[str, str]] = []
    seen_impacts: set[str] = set()
    for source in args.input:
        if not source.is_file():
            raise SystemExit(f"Input CSV not found: {source}")
        source_fields, rows = read_csv(source)
        if fields is None:
            fields = source_fields + ["source_review_file"]
        elif source_fields != fields[:-1]:
            raise SystemExit(f"Header differs from first input: {source}")
        for row in rows:
            status = row["review_status"]
            direction = row["actual_direction_front_view"]
            if (status, direction) not in VALID:
                raise SystemExit(
                    f"{source}, {row['candidate_id']}: incomplete or invalid label "
                    f"({status}, {direction})"
                )
            impact_id = row["impact_id"]
            if impact_id in seen_impacts:
                raise SystemExit(f"Duplicate impact_id across batches: {impact_id}")
            seen_impacts.add(impact_id)
            row["source_review_file"] = source.name
            merged.append(row)

    if not merged or fields is None:
        raise SystemExit("No review rows were supplied")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)

    confirmed = sum(row["review_status"] == "confirmed" for row in merged)
    print(f"Wrote {args.output}: {len(merged)} rows, {confirmed} confirmed, "
          f"{len(merged) - confirmed} excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
