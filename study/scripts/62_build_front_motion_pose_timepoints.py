"""Measure pre-impact lateral motion from human-audited target-front ankle poses."""

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


def median_position(rows: dict[int, dict[str, str]], center: int, radius: int) -> tuple[float | None, int]:
    values = [
        float(row["ankle_midpoint_x_px"])
        for frame in range(center - radius, center + radius + 1)
        if (row := rows.get(frame)) and row["ankle_pair_visible"] == "1" and row["ankle_midpoint_x_px"]
    ]
    return (statistics.median(values), len(values)) if values else (None, 0)


def direction(displacement: float, threshold: float) -> str:
    if displacement >= threshold:
        return "front_right"
    if displacement <= -threshold:
        return "front_left"
    return "stay"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--pose-stem", default="match1_target_front_pose_yolo11m")
    parser.add_argument("--timepoints-sec", default="-0.5,-0.3,-0.2,-0.1")
    parser.add_argument("--motion-window-ms", type=float, default=100.0)
    parser.add_argument("--median-radius-frames", type=int, default=1)
    parser.add_argument("--movement-threshold-px", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.motion_window_ms <= 0 or args.median_radius_frames < 0 or args.movement_threshold_px <= 0:
        raise SystemExit("Motion window, median radius, and threshold must be positive")
    try:
        timepoints = [float(value) for value in args.timepoints_sec.split(",")]
    except ValueError as error:
        raise SystemExit("--timepoints-sec must be comma-separated decimal seconds") from error
    if any(value >= 0 for value in timepoints):
        raise SystemExit("All time points must precede impact")
    project = Path(__file__).resolve().parents[1]
    labels = {row["event_id"]: row for row in read(project / "annotations" / f"shot_direction_review_{args.video_id}.csv") if row["review_status"] == "confirmed" and row["actual_direction_front_view"] in ("front_left", "front_right")}
    usable_pose = {row["event_id"] for row in read(project / "annotations" / f"target_front_pose_review_{args.video_id}_yolo11m.csv") if row["pose_status"] == "usable"}
    events = {row["event_id"]: row for row in read(project / "annotations" / "motion_prediction_events.csv")}
    pose_by_event: dict[str, dict[int, dict[str, str]]] = {}
    for row in read(project / "data" / "pose" / "target_front" / f"{args.pose_stem}.csv"):
        if row["event_id"] in labels and row["event_id"] in usable_pose:
            pose_by_event.setdefault(row["event_id"], {})[int(row["source_frame"])] = row
    window = round(args.motion_window_ms / 1000 * FPS)
    results: list[dict[str, object]] = []
    for event_id in sorted(pose_by_event):
        event, label = events.get(event_id), labels[event_id]
        if not event or not event["impact_source_frame"]:
            continue
        impact = int(event["impact_source_frame"])
        for timepoint in timepoints:
            end = impact + round(timepoint * FPS); start = end - window
            start_x, start_count = median_position(pose_by_event[event_id], start, args.median_radius_frames)
            end_x, end_count = median_position(pose_by_event[event_id], end, args.median_radius_frames)
            if start_x is None or end_x is None:
                displacement, observed, matches = "", "missing", ""
            else:
                displacement_value = end_x - start_x
                observed = direction(displacement_value, args.movement_threshold_px)
                displacement = f"{displacement_value:.3f}"
                matches = "" if observed == "stay" else str(int(observed == label["actual_direction_front_view"]))
            results.append({
                "event_id": event_id, "video_id": args.video_id, "impact_source_frame": impact,
                "actual_direction_front_view": label["actual_direction_front_view"], "timepoint_sec": f"{timepoint:.3f}",
                "motion_window_ms": f"{args.motion_window_ms:.1f}", "motion_start_source_frame": start, "motion_end_source_frame": end,
                "start_median_samples": start_count, "end_median_samples": end_count,
                "start_ankle_midpoint_x_px": "" if start_x is None else f"{start_x:.3f}", "end_ankle_midpoint_x_px": "" if end_x is None else f"{end_x:.3f}",
                "lateral_displacement_px": displacement, "movement_direction_front_view": observed, "direction_matches_actual": matches,
                "coordinate_definition": "audited_yolo11m_pose_bilateral_ankle_midpoint_x_px",
            })
    if not results:
        raise SystemExit("No usable pose events with confirmed shot-direction labels")
    output_dir = project / "analysis"; output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or output_dir / f"{args.video_id}_front_motion_timepoints_pose.csv"; output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    output.with_suffix(".metadata.json").write_text(json.dumps({
        "video_id": args.video_id, "events": len(pose_by_event), "timepoints_sec": timepoints,
        "nominal_fps": FPS, "motion_window_ms": args.motion_window_ms, "motion_window_frames": window,
        "median_radius_frames": args.median_radius_frames, "movement_threshold_px": args.movement_threshold_px,
        "threshold_rationale": "provisional 15px threshold exceeds the observed 95th percentile adjacent-frame ankle-midpoint displacement (12.78px), without using final-direction agreement",
        "coordinate_definition": "human-audited bilateral ankle midpoint from YOLO11m-pose",
        "leakage_guard": "each displacement window ends before impact; actual post-impact direction is used only as a comparison label",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} pose event-timepoint rows: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
