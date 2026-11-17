"""Measure pre-impact target-front lateral box motion at fixed time points.

The detector's box-bottom centre is a provisional lateral-motion proxy, not an
anatomical foot landmark.  It is used here to make the timing/label pipeline
auditable before replacing it with pose-derived ankle positions.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


FPS = 60000 / 1001


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def median_position(rows: dict[int, dict[str, str]], center_frame: int, radius: int) -> tuple[float | None, int]:
    values = [
        float(row["foot_x_px"])
        for frame in range(center_frame - radius, center_frame + radius + 1)
        if (row := rows.get(frame)) and row.get("visible") == "1" and row.get("foot_x_px") not in ("", None)
    ]
    return (statistics.median(values), len(values)) if values else (None, 0)


def movement(displacement: float, threshold: float) -> str:
    if displacement >= threshold:
        return "front_right"
    if displacement <= -threshold:
        return "front_left"
    return "stay"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--model-stem", default="yolo11m")
    parser.add_argument("--timepoints-sec", default="-0.5,-0.3,-0.2,-0.1")
    parser.add_argument("--motion-window-ms", type=float, default=100.0)
    parser.add_argument("--median-radius-frames", type=int, default=1)
    parser.add_argument("--movement-threshold-px", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path for a threshold-sensitivity run")
    args = parser.parse_args()
    if args.motion_window_ms <= 0 or args.median_radius_frames < 0 or args.movement_threshold_px <= 0:
        raise SystemExit("Motion window, median radius, and threshold must be positive")
    try:
        timepoints = [float(value) for value in args.timepoints_sec.split(",")]
    except ValueError as error:
        raise SystemExit("--timepoints-sec must be comma-separated decimal seconds") from error
    if any(value >= 0 for value in timepoints):
        raise SystemExit("All time points must precede impact (< 0)")
    project = Path(__file__).resolve().parents[1]
    labels = {
        row["event_id"]: row
        for row in read(project / "annotations" / f"shot_direction_review_{args.video_id}.csv")
        if row["review_status"] == "confirmed" and row["actual_direction_front_view"] in ("front_left", "front_right")
    }
    events = {row["event_id"]: row for row in read(project / "annotations" / "motion_prediction_events.csv")}
    track_path = project / "data" / "tracking" / "front_target_highres" / f"{args.video_id}_target_front_tracks_{args.model_stem}.csv"
    tracks: dict[str, dict[int, dict[str, str]]] = {}
    for row in read(track_path):
        if row["event_id"] in labels:
            tracks.setdefault(row["event_id"], {})[int(row["source_frame"])] = row
    window_frames = round(args.motion_window_ms / 1000 * FPS)
    results: list[dict[str, object]] = []
    for event_id, label in labels.items():
        event = events.get(event_id)
        if not event or not event.get("impact_source_frame") or event_id not in tracks:
            continue
        impact = int(event["impact_source_frame"])
        for timepoint in timepoints:
            end_frame = impact + round(timepoint * FPS)
            start_frame = end_frame - window_frames
            start_x, start_count = median_position(tracks[event_id], start_frame, args.median_radius_frames)
            end_x, end_count = median_position(tracks[event_id], end_frame, args.median_radius_frames)
            if start_x is None or end_x is None:
                direction, displacement, matches = "missing", "", ""
            else:
                displacement_value = end_x - start_x
                direction = movement(displacement_value, args.movement_threshold_px)
                displacement = f"{displacement_value:.3f}"
                matches = "" if direction == "stay" else str(int(direction == label["actual_direction_front_view"]))
            results.append({
                "event_id": event_id, "video_id": args.video_id,
                "impact_source_frame": impact, "actual_direction_front_view": label["actual_direction_front_view"],
                "timepoint_sec": f"{timepoint:.3f}", "motion_window_ms": f"{args.motion_window_ms:.1f}",
                "motion_start_source_frame": start_frame, "motion_end_source_frame": end_frame,
                "start_median_samples": start_count, "end_median_samples": end_count,
                "start_box_bottom_center_x_px": "" if start_x is None else f"{start_x:.3f}",
                "end_box_bottom_center_x_px": "" if end_x is None else f"{end_x:.3f}",
                "lateral_displacement_px": displacement, "movement_direction_front_view": direction,
                "direction_matches_actual": matches,
                "coordinate_definition": "detector_box_bottom_center_x_px_not_anatomical_foot",
            })
    if not results:
        raise SystemExit("No confirmed labels with matching target-front tracks")
    output_dir = project / "analysis"; output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or output_dir / f"{args.video_id}_front_motion_timepoints_boxproxy.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    metadata = {
        "video_id": args.video_id, "events": len(labels), "timepoints_sec": timepoints,
        "nominal_fps": FPS, "motion_window_ms": args.motion_window_ms,
        "motion_window_frames": window_frames, "median_radius_frames": args.median_radius_frames,
        "movement_threshold_px": args.movement_threshold_px,
        "direction_convention": "match1 audited: screen left/right equals target-front left/right",
        "coordinate_definition": "detector box bottom centre; preliminary proxy, not an anatomical foot point",
        "leakage_guard": "displacement uses only frames at or before each negative time point; post-impact direction is outcome label only",
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} event-timepoint rows: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
