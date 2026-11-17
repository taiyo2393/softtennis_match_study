"""Build conservative analysis-ready main-camera intervals from reviewed cuts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--duration-sec", required=True, type=float)
    parser.add_argument("--fps", type=float, default=60000 / 1001)
    parser.add_argument("--margin-sec", type=float, default=1.0)
    parser.add_argument("--min-duration-sec", type=float, default=2.0)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    project = Path(__file__).resolve().parents[1]
    segment_dir = project / "data" / "segments"
    reviews = read_rows(segment_dir / f"{args.video_id}_transition_review.csv")
    if not reviews:
        raise ValueError("No reviewed transitions found")
    reviews.sort(key=lambda row: float(row["timestamp_sec"]))
    if any(row["before_view"] == "unreviewed" or row["after_view"] == "unreviewed" for row in reviews):
        raise ValueError("All transitions must be reviewed before generating segments")

    # One interval exists before the first cut, one after the last, and one
    # between each pair. Internal intervals are accepted only when the two
    # independent side labels agree that it is the main camera.
    boundaries = [0.0] + [float(row["timestamp_sec"]) for row in reviews] + [args.duration_sec]
    main_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    for index in range(len(boundaries) - 1):
        raw_start, raw_end = boundaries[index], boundaries[index + 1]
        if index == 0:
            labels = [reviews[0]["before_view"]]
        elif index == len(boundaries) - 2:
            labels = [reviews[-1]["after_view"]]
        else:
            labels = [reviews[index - 1]["after_view"], reviews[index]["before_view"]]
        agreed_main = all(label == "main_camera" for label in labels)
        start = raw_start + (args.margin_sec if index > 0 else 0.0)
        end = raw_end - (args.margin_sec if index < len(boundaries) - 2 else 0.0)
        duration = end - start
        label_text = " / ".join(labels)
        if agreed_main and duration >= args.min_duration_sec:
            segment_id = f"{args.video_id}_main_{len(main_rows) + 1:03d}"
            main_rows.append({
                "video_id": args.video_id,
                "segment_id": segment_id,
                "start_timestamp_sec": f"{start:.6f}",
                "end_timestamp_sec": f"{end:.6f}",
                "start_source_frame": round(start * args.fps),
                "end_source_frame": round(end * args.fps),
                "scene_type": "main_camera",
                "cut_confidence": "human_reviewed",
                "main_camera_confidence": "1.0",
                "usable_for_analysis": "true",
                "human_checked": "true",
                "notes": f"labels={label_text}; edge_margin_sec={args.margin_sec}",
            })
        else:
            reason = "not_main_camera" if not agreed_main else "too_short_after_edge_margin"
            rejected_rows.append({
                "video_id": args.video_id,
                "raw_start_sec": f"{raw_start:.6f}",
                "raw_end_sec": f"{raw_end:.6f}",
                "labels": label_text,
                "reason": reason,
            })

    output = segment_dir / "main_camera_segments.csv"
    write_rows(output, [
        "video_id", "segment_id", "start_timestamp_sec", "end_timestamp_sec",
        "start_source_frame", "end_source_frame", "scene_type", "cut_confidence",
        "main_camera_confidence", "usable_for_analysis", "human_checked", "notes",
    ], main_rows)
    write_rows(segment_dir / f"{args.video_id}_nonmain_or_conflict_intervals.csv", [
        "video_id", "raw_start_sec", "raw_end_sec", "labels", "reason",
    ], rejected_rows)
    print(f"Wrote {len(main_rows)} main-camera intervals; rejected {len(rejected_rows)} intervals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
