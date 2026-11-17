"""Build an auditable, pre-impact 2-D front-player motion table.

This joins only (1) human-confirmed target-front-view shot-direction labels
and (2) human-audited usable target-front ankle poses.  It does not create a
motion-direction label, infer player-centred left/right from screen x, or fit
a predictive model.  All coordinates remain fixed-camera 2-D screen values.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


FPS = 60000 / 1001


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def median_x(rows_by_frame: dict[int, dict[str, str]], center: int, radius: int) -> tuple[float | None, int]:
    values = [
        float(row["ankle_midpoint_x_px"])
        for frame in range(center - radius, center + radius + 1)
        if (row := rows_by_frame.get(frame))
        and row.get("ankle_pair_visible") == "1"
        and row.get("ankle_midpoint_x_px")
    ]
    return (statistics.median(values), len(values)) if values else (None, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--direction-review", type=Path, required=True)
    parser.add_argument("--pose-review", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--timepoints-sec", default="-0.5,-0.3,-0.2,-0.1")
    parser.add_argument("--motion-window-ms", type=float, default=100.0)
    parser.add_argument("--median-radius-frames", type=int, default=1)
    parser.add_argument("--screen-width-px", type=float, default=3840.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not path.is_file() for path in (args.direction_review, args.pose_review, args.pose)):
        raise SystemExit("Direction-review, pose-review, and pose CSVs must all exist")
    if args.motion_window_ms <= 0 or args.median_radius_frames < 0 or args.screen_width_px <= 0:
        raise SystemExit("Motion window, median radius, and screen width must be positive")
    try:
        timepoints = [float(value) for value in args.timepoints_sec.split(",")]
    except ValueError as error:
        raise SystemExit("--timepoints-sec must be comma-separated decimal seconds") from error
    if not timepoints or any(value >= 0 for value in timepoints):
        raise SystemExit("Every time point must be strictly before impact")

    # The original near-front review used ``actual_direction_front_view``;
    # the later far-front review made the same perspective explicit as
    # ``actual_direction_target_front_view``.  Normalise only the column name
    # here.  Both values are human labels in the target front player's view.
    labels: dict[str, dict[str, str]] = {}
    for source_row in read(args.direction_review):
        direction = (
            source_row.get("actual_direction_target_front_view")
            or source_row.get("actual_direction_front_view")
        )
        if (
            source_row.get("video_id") == args.video_id
            and source_row.get("review_status") == "confirmed"
            and direction in ("front_left", "front_right")
        ):
            row = dict(source_row)
            row["actual_direction_front_view"] = direction
            labels[row["candidate_id"]] = row
    pose_usable = {
        row["event_id"]
        for row in read(args.pose_review)
        if row.get("video_id") == args.video_id and row.get("pose_status") == "usable"
    }
    poses_by_event: dict[str, dict[int, dict[str, str]]] = {}
    for row in read(args.pose):
        event_id = row.get("event_id", "")
        if event_id in labels and event_id in pose_usable:
            poses_by_event.setdefault(event_id, {})[int(row["source_frame"])] = row

    event_ids = sorted(set(labels) & set(pose_usable) & set(poses_by_event))
    if not event_ids:
        raise SystemExit("No events have both a confirmed direction and usable audited pose")
    motion_frames = round(args.motion_window_ms / 1000 * FPS)
    fields = [
        "event_id", "video_id", "impact_source_frame", "actual_direction_front_view",
        "timepoint_sec", "timepoint_source_frame", "position_median_x_px",
        "position_median_x_screen_fraction", "position_samples", "impact_median_x_px",
        "relative_to_impact_x_px", "motion_start_source_frame", "motion_end_source_frame",
        "motion_start_median_x_px", "motion_end_median_x_px", "motion_start_samples",
        "motion_end_samples", "lateral_displacement_px", "lateral_displacement_screen_fraction",
        "lateral_velocity_px_per_sec", "feature_status", "coordinate_definition",
    ]
    output_rows: list[dict[str, object]] = []
    for event_id in event_ids:
        label = labels[event_id]
        impact = int(label["impact_source_frame"])
        rows_by_frame = poses_by_event[event_id]
        impact_x, _ = median_x(rows_by_frame, impact, args.median_radius_frames)
        for timepoint in timepoints:
            point_frame = impact + round(timepoint * FPS)
            end_frame = point_frame
            start_frame = end_frame - motion_frames
            point_x, point_count = median_x(rows_by_frame, point_frame, args.median_radius_frames)
            start_x, start_count = median_x(rows_by_frame, start_frame, args.median_radius_frames)
            end_x, end_count = median_x(rows_by_frame, end_frame, args.median_radius_frames)
            if point_x is None or start_x is None or end_x is None:
                displacement: float | None = None
                status = "missing_pose"
            else:
                displacement = end_x - start_x
                status = "usable"
            output_rows.append({
                "event_id": event_id,
                "video_id": args.video_id,
                "impact_source_frame": impact,
                "actual_direction_front_view": label["actual_direction_front_view"],
                "timepoint_sec": f"{timepoint:.3f}",
                "timepoint_source_frame": point_frame,
                "position_median_x_px": "" if point_x is None else f"{point_x:.3f}",
                "position_median_x_screen_fraction": "" if point_x is None else f"{point_x / args.screen_width_px:.8f}",
                "position_samples": point_count,
                "impact_median_x_px": "" if impact_x is None else f"{impact_x:.3f}",
                "relative_to_impact_x_px": "" if point_x is None or impact_x is None else f"{point_x - impact_x:.3f}",
                "motion_start_source_frame": start_frame,
                "motion_end_source_frame": end_frame,
                "motion_start_median_x_px": "" if start_x is None else f"{start_x:.3f}",
                "motion_end_median_x_px": "" if end_x is None else f"{end_x:.3f}",
                "motion_start_samples": start_count,
                "motion_end_samples": end_count,
                "lateral_displacement_px": "" if displacement is None else f"{displacement:.3f}",
                "lateral_displacement_screen_fraction": "" if displacement is None else f"{displacement / args.screen_width_px:.8f}",
                "lateral_velocity_px_per_sec": "" if displacement is None else f"{displacement / (motion_frames / FPS):.3f}",
                "feature_status": status,
                "coordinate_definition": "fixed_camera_2d_screen_x_px; not court metres or player-centred left/right",
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    metadata = {
        "video_id": args.video_id,
        "event_count": len(event_ids),
        "event_outcome_counts": dict(Counter(labels[event_id]["actual_direction_front_view"] for event_id in event_ids)),
        "timepoints_sec": timepoints,
        "nominal_fps": FPS,
        "motion_window_ms": args.motion_window_ms,
        "motion_window_frames": motion_frames,
        "median_radius_frames": args.median_radius_frames,
        "direction_review": str(args.direction_review),
        "pose_review": str(args.pose_review),
        "pose": str(args.pose),
        "inference_guard": (
            "This file contains only fixed-camera 2-D pre-impact features. "
            "It is an exploratory single-match pilot and does not identify a causal effect."
        ),
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    usable = sum(row["feature_status"] == "usable" for row in output_rows)
    print(f"Wrote {len(output_rows)} event-timepoint rows from {len(event_ids)} events; usable rows {usable}/{len(output_rows)}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
