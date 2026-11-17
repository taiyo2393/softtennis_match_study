"""Merge reviewed windows while preserving how each start was established.

This output is for subsequent event screening.  It is deliberately not a
trial table: a target hitter, impact, front player and direction still need
their own labels.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[1] / "data" / "segments"
    windows = read(directory / f"{args.video_id}_rally_window_candidates.csv")
    unresolved = {row["window_id"]: row for row in read(directory / f"{args.video_id}_unresolved_start_review.csv")}
    output = []
    exclusions = []
    for row in windows:
        start_method = ""
        valid_serve_impact = ""
        serve_context = ""
        note = ""
        if row["window_status"] == "complete_boundary":
            # This only confirms that a pre-rally fixed-camera view is present.
            start_method = "camera_entry_confirmed_not_impact"
            start = row["source_start_timestamp_sec"]
            serve_context = "not_yet_labeled"
        else:
            review = unresolved.get(row["window_id"])
            if not review:
                exclusions.append({"window_id": row["window_id"], "reason": "no_unresolved_review"}); continue
            if review["review_status"] not in ("complete_valid_rally", "complete_rally_after_first_serve_fault"):
                exclusions.append({"window_id": row["window_id"], "reason": review["review_status"]}); continue
            start_method = "manual_valid_serve_impact"
            valid_serve_impact = review["human_rally_start_timestamp_sec"]
            start = valid_serve_impact
            serve_context = review["review_status"]
            note = review["review_note"]
        end = row["source_end_timestamp_sec"]
        output.append({
            "play_window_id": f"{args.video_id}_play_{len(output)+1:03d}", "video_id": args.video_id,
            "source_window_id": row["window_id"], "source_start_timestamp_sec": start,
            "source_end_timestamp_sec": end, "duration_sec": f"{float(end)-float(start):.6f}",
            "start_establishment_method": start_method,
            "valid_serve_impact_timestamp_sec": valid_serve_impact,
            "serve_context": serve_context, "end_boundary_id": row["end_boundary_id"],
            "end_status": "human_confirmed_departure_after_play", "review_note": note,
            "restriction": "Not a trial_id. Screen target hitter, target front player, impact, direction and visibility separately.",
        })
    fields = list(output[0]) if output else []
    with (directory / f"{args.video_id}_reviewed_play_windows.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(output)
    with (directory / f"{args.video_id}_play_window_exclusions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["window_id", "reason"]); writer.writeheader(); writer.writerows(exclusions)
    exact = sum(row["start_establishment_method"] == "manual_valid_serve_impact" for row in output)
    print(f"Wrote {len(output)} reviewed play windows ({exact} with manually marked valid-serve impacts); excluded {len(exclusions)}.")


if __name__ == "__main__":
    main()
