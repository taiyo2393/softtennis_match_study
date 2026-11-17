"""Require both fixed court geometry and a clean, non-transition background."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--duration-sec", type=float, required=True)
    parser.add_argument("--max-background-difference", type=float, default=0.05)
    parser.add_argument("--min-geometry-score", type=float, default=0.95)
    parser.add_argument("--edge-margin-sec", type=float, default=1.0)
    parser.add_argument("--min-duration-sec", type=float, default=2.0)
    return parser.parse_args()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file: return list(csv.DictReader(file))


def main() -> int:
    args = parse_args(); project = Path(__file__).resolve().parents[1]; directory = project / "data" / "segments"
    background = {row["timestamp_sec"]: row for row in read(directory / f"{args.video_id}_fixed_view_scores.csv")}
    geometry = {row["timestamp_sec"]: row for row in read(directory / f"{args.video_id}_court_geometry_scores.csv")}
    timestamps = sorted(set(background) & set(geometry), key=float)
    samples = []
    for timestamp in timestamps:
        bg, geo = float(background[timestamp]["background_difference"]), float(geometry[timestamp]["court_geometry_score"])
        samples.append((float(timestamp), bg, geo, bg <= args.max_background_difference and geo >= args.min_geometry_score))
    step = samples[1][0] - samples[0][0]
    segments = []; start = None; confidences = []
    def finish(end: float) -> None:
        nonlocal start, confidences
        if start is None: return
        usable_start, usable_end = start + args.edge_margin_sec, min(args.duration_sec, end + step) - args.edge_margin_sec
        if usable_end - usable_start >= args.min_duration_sec:
            number = len(segments) + 1
            segments.append({"video_id": args.video_id, "segment_id": f"{args.video_id}_ready_{number:03d}", "start_timestamp_sec": f"{usable_start:.6f}", "end_timestamp_sec": f"{usable_end:.6f}", "start_source_frame": round(usable_start * 60000 / 1001), "end_source_frame": round(usable_end * 60000 / 1001), "scene_type": "analysis_ready_main_camera", "cut_confidence": "geometry_and_background", "main_camera_confidence": f"{sum(confidences)/len(confidences):.6f}", "usable_for_analysis": "true", "human_checked": "false", "notes": f"background<={args.max_background_difference}; geometry>={args.min_geometry_score}; edge_margin_sec={args.edge_margin_sec}"})
        start, confidences = None, []
    for timestamp, bg, geo, passed in samples:
        if passed:
            if start is None: start = timestamp
            confidences.append((1 - bg + geo) / 2)
        else: finish(timestamp - step)
    finish(args.duration_sec - step)
    fields = ["video_id", "segment_id", "start_timestamp_sec", "end_timestamp_sec", "start_source_frame", "end_source_frame", "scene_type", "cut_confidence", "main_camera_confidence", "usable_for_analysis", "human_checked", "notes"]
    with (directory / "analysis_ready_main_camera_segments.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(segments)
    print(f"Wrote {len(segments)} analysis-ready intervals from {len(samples)} samples")


if __name__ == "__main__": main()
