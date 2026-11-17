"""Merge original usable tracks and one or more manually re-seeded repairs.

The source review files remain immutable.  The resulting ``final`` batch is a
canonical input for pose extraction and explicitly records whether a row came
from an automatic seed or a human-selected re-seed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def field_union(rows: list[dict[str, str]]) -> list[str]:
    """Preserve original order while retaining provenance-only repair columns."""
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields


def usable_ids(path: Path) -> set[str]:
    return {row["event_id"] for row in read(path) if row.get("tracking_status") == "usable"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-tracks", type=Path, required=True)
    parser.add_argument("--original-summary", type=Path, required=True)
    parser.add_argument("--original-review", type=Path, required=True)
    parser.add_argument("--repair-tracks", type=Path, action="append", required=True)
    parser.add_argument("--repair-summary", type=Path, action="append", required=True)
    parser.add_argument("--repair-review", type=Path, action="append", required=True)
    parser.add_argument("--reseed-selection", type=Path, action="append", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--model-stem", default="yolo11m")
    args = parser.parse_args()
    repair_counts = {
        len(args.repair_tracks), len(args.repair_summary),
        len(args.repair_review), len(args.reseed_selection),
    }
    if len(repair_counts) != 1:
        raise SystemExit(
            "Each repair input option must occur the same number of times: "
            "--repair-tracks, --repair-summary, --repair-review, and --reseed-selection"
        )
    repair_batches = list(zip(
        args.repair_tracks, args.repair_summary,
        args.repair_review, args.reseed_selection,
        strict=True,
    ))
    for path in [
        args.original_tracks, args.original_summary, args.original_review,
        *(path for batch in repair_batches for path in batch),
    ]:
        if not path.is_file():
            raise SystemExit(f"Required input not found: {path}")

    original_ids = usable_ids(args.original_review)
    repaired_ids: set[str] = set()
    reseed_ranks: dict[str, str] = {}
    repair_tracks: list[dict[str, str]] = []
    repair_summaries: list[dict[str, str]] = []
    for repair_track_path, repair_summary_path, repair_review_path, reseed_selection_path in repair_batches:
        batch_ids = usable_ids(repair_review_path)
        if original_ids.intersection(batch_ids) or repaired_ids.intersection(batch_ids):
            raise SystemExit("An event is usable in more than one input batch")
        batch_ranks = {
            row["event_id"]: row["selected_candidate_rank"]
            for row in read(reseed_selection_path)
            if row.get("review_status") == "confirmed"
        }
        if not batch_ids.issubset(batch_ranks):
            raise SystemExit("Every repaired usable event must have a confirmed manual re-seed")
        repaired_ids.update(batch_ids)
        reseed_ranks.update(batch_ranks)
        repair_tracks.extend(row for row in read(repair_track_path) if row["event_id"] in batch_ids)
        repair_summaries.extend(row for row in read(repair_summary_path) if row["event_id"] in batch_ids)
    if not repaired_ids.issubset(reseed_ranks):
        raise SystemExit("Every repaired usable event must have a confirmed manual re-seed")

    original_tracks = [row for row in read(args.original_tracks) if row["event_id"] in original_ids]
    original_summaries = [row for row in read(args.original_summary) if row["event_id"] in original_ids]
    event_ids = {row["event_id"] for row in original_summaries + repair_summaries}
    if event_ids != original_ids.union(repaired_ids):
        raise SystemExit("Track summaries do not cover every audited usable event")

    for row in original_tracks:
        row["tracking_source"] = "automatic_seed_audited"
        row["reseed_candidate_rank"] = ""
    for row in repair_tracks:
        row["tracking_source"] = "human_reseed_audited"
        row["reseed_candidate_rank"] = reseed_ranks[row["event_id"]]
    for row in original_summaries:
        row["tracking_source"] = "automatic_seed_audited"
        row["reseed_candidate_rank"] = ""
    for row in repair_summaries:
        row["tracking_source"] = "human_reseed_audited"
        row["reseed_candidate_rank"] = reseed_ranks[row["event_id"]]

    project = Path(__file__).resolve().parents[1]
    track_dir = project / "data" / "tracking" / "front_target_highres"
    annotation_dir = project / "annotations"
    tracks = original_tracks + repair_tracks
    summaries = original_summaries + repair_summaries
    tracks.sort(key=lambda row: (int(row["source_frame"]), row["event_id"]))
    summaries.sort(key=lambda row: int(row["impact_source_frame"]))
    write(track_dir / f"{args.output_prefix}_tracks_{args.model_stem}.csv", field_union(tracks), tracks)
    write(track_dir / f"{args.output_prefix}_summary_{args.model_stem}.csv", field_union(summaries), summaries)
    review_rows = []
    for summary in summaries:
        repaired = summary["event_id"] in repaired_ids
        review_rows.append({
            "event_id": summary["event_id"], "video_id": "match1", "model": summary["model"],
            "tracking_status": "usable", "review_note": "",
            "tracking_source": summary["tracking_source"],
            "reseed_candidate_rank": summary["reseed_candidate_rank"],
        })
    write(
        annotation_dir / f"{args.output_prefix}_track_review_{args.model_stem}.csv",
        list(review_rows[0]), review_rows,
    )
    print(
        f"Merged {len(event_ids)} audited usable events: {len(original_ids)} automatic + "
        f"{len(repaired_ids)} re-seeded across {len(repair_batches)} repair batch(es)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
