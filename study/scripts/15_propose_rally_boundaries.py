"""Propose, but never confirm, rally boundaries from main-camera departures.

The input is a fixed-view signal, not a rally label.  In replay-insertion
broadcasts, leaving the fixed court view is often near a rally end; in an
unedited recording it is merely a weak boundary cue.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file: return list(csv.DictReader(file))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--max-background-difference", type=float, default=0.05)
    parser.add_argument("--min-geometry-score", type=float, default=0.95)
    parser.add_argument("--cut-match-window-sec", type=float, default=2.0)
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]; directory = project / "data" / "segments"
    background = {row["timestamp_sec"]: row for row in read(directory / f"{args.video_id}_fixed_view_scores.csv")}
    geometry = {row["timestamp_sec"]: row for row in read(directory / f"{args.video_id}_court_geometry_scores.csv")}
    samples = []
    for stamp in sorted(set(background) & set(geometry), key=float):
        bg, geo = float(background[stamp]["background_difference"]), float(geometry[stamp]["court_geometry_score"])
        samples.append((float(stamp), bg <= args.max_background_difference and geo >= args.min_geometry_score, bg, geo))
    if len(samples) < 2: raise SystemExit("Need aligned fixed-view and geometry score files")
    step = samples[1][0] - samples[0][0]
    cuts = [float(row["timestamp_sec"]) for row in read(directory / f"{args.video_id}_cut_candidates.csv")]
    rows: list[dict[str, object]] = []
    number = 1
    def add(kind: str, timestamp: float, bg: float, geo: float) -> None:
        nonlocal number
        nearest = min(cuts, key=lambda value: abs(value - timestamp), default=None)
        cut_gap = abs(nearest - timestamp) if nearest is not None else None
        has_cut = cut_gap is not None and cut_gap <= args.cut_match_window_sec
        # This is a candidate priority only. It never asserts that a rally ended.
        priority = "high" if has_cut else "medium"
        rows.append({"boundary_id": f"{args.video_id}_boundary_{number:04d}", "video_id": args.video_id,
                     "timestamp_sec": f"{timestamp:.6f}", "source_frame": round(timestamp * 60000 / 1001),
                     "candidate_type": kind, "signal_main_view_change": "true", "signal_visual_cut": str(has_cut).lower(),
                     "nearest_cut_timestamp_sec": "" if nearest is None else f"{nearest:.6f}",
                     "cut_gap_sec": "" if cut_gap is None else f"{cut_gap:.6f}",
                     "background_difference": f"{bg:.6f}", "court_geometry_score": f"{geo:.6f}",
                     "review_priority": priority, "human_status": "unreviewed", "human_boundary_type": "",
                     "human_timestamp_sec": "", "review_note": ""})
        number += 1
    previous_time, previous_main, previous_bg, previous_geo = samples[0]
    if previous_main: add("main_camera_enter_candidate", previous_time, previous_bg, previous_geo)
    for timestamp, is_main, bg, geo in samples[1:]:
        if is_main != previous_main:
            # Do not trim a second off the boundary. Keep the source timestamp and
            # hand the surrounding interval to the reviewer.
            add("main_camera_enter_candidate" if is_main else "main_camera_departure_candidate", timestamp, bg, geo)
        previous_time, previous_main, previous_bg, previous_geo = timestamp, is_main, bg, geo
    fields = ["boundary_id", "video_id", "timestamp_sec", "source_frame", "candidate_type", "signal_main_view_change", "signal_visual_cut", "nearest_cut_timestamp_sec", "cut_gap_sec", "background_difference", "court_geometry_score", "review_priority", "human_status", "human_boundary_type", "human_timestamp_sec", "review_note"]
    output = directory / f"{args.video_id}_rally_boundary_candidates.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    departures = sum(row["candidate_type"] == "main_camera_departure_candidate" for row in rows)
    print(f"Wrote {len(rows)} view-transition candidates ({departures} departure candidates) to {output}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
