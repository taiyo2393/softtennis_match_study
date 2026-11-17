"""Restore reviewed rally-start fields after a boundary-candidate rebuild.

Matches rows by end_boundary_id so changed window numbering cannot attach a
human decision to a different candidate.  New endings remain unreviewed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = ["start_confirmation", "human_start_timestamp_sec", "start_review_note"]


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--from-file", required=True, type=Path)
    args = parser.parse_args()

    directory = Path(__file__).resolve().parents[1] / "data" / "segments"
    target = directory / f"{args.video_id}_rally_window_candidates.csv"
    fields, current = read(target)
    _, prior = read(args.from_file)
    prior_by_end = {row["end_boundary_id"]: row for row in prior}
    restored = 0
    for row in current:
        old = prior_by_end.get(row["end_boundary_id"])
        if not old or old.get("start_confirmation") == "unreviewed":
            continue
        for field in FIELDS:
            row[field] = old.get(field, "")
        row["window_status"] = (
            "complete_boundary" if row["start_confirmation"] == "confirmed"
            else "excluded_or_unresolved" if row["start_confirmation"] == "not_rally_start"
            else "needs_start_confirmation"
        )
        restored += 1
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(current)
    print(f"Restored {restored} reviewed start decisions; {len(current)-restored} rows remain unreviewed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
