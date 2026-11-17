"""Transform audited ankle poses into a fixed-camera normalized court plane.

Unlike script 69, this joins the reviewed near and far target-front events in
one table.  It uses a manually calibrated homography for one fixed main-camera
view.  The result can compare lateral displacement across near/far ends, but
remains a single-match descriptive dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


FPS = 60000 / 1001


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def median_coordinate(
    rows_by_frame: dict[int, dict[str, float]], center: int, radius: int, coordinate: str
) -> tuple[float | None, int]:
    values = [
        row[coordinate]
        for frame in range(center - radius, center + radius + 1)
        if (row := rows_by_frame.get(frame)) is not None
    ]
    return (statistics.median(values), len(values)) if values else (None, 0)


def transform_ankle_pair(row: dict[str, str], transform: np.ndarray) -> tuple[float, float] | None:
    if row.get("ankle_pair_visible") != "1":
        return None
    try:
        ankles = np.float32([[
            [float(row["left_ankle_x_px"]), float(row["left_ankle_y_px"])],
            [float(row["right_ankle_x_px"]), float(row["right_ankle_y_px"])],
        ]])
    except (KeyError, ValueError):
        return None
    transformed = cv2.perspectiveTransform(ankles, transform)[0]
    # Transform each ankle first, then take their midpoint.  A projective
    # transform is not linear in image coordinates, so transforming the
    # screen-space midpoint would be a different (and less defensible) value.
    midpoint = transformed.mean(axis=0)
    return float(midpoint[0]), float(midpoint[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--direction-review", type=Path, action="append", required=True)
    parser.add_argument("--pose-review", type=Path, action="append", required=True)
    parser.add_argument("--pose", type=Path, action="append", required=True)
    parser.add_argument("--identity-review", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--timepoints-sec", default="-0.5,-0.3,-0.2,-0.1")
    parser.add_argument("--motion-window-ms", type=float, default=100.0)
    parser.add_argument("--median-radius-frames", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [
        *args.direction_review, *args.pose_review, *args.pose,
        args.identity_review, args.calibration,
    ]
    if any(not path.is_file() for path in inputs):
        missing = [str(path) for path in inputs if not path.is_file()]
        raise SystemExit(f"Required input not found: {missing}")
    if args.motion_window_ms <= 0 or args.median_radius_frames < 0:
        raise SystemExit("Motion window must be positive and median radius non-negative")
    try:
        timepoints = [float(value) for value in args.timepoints_sec.split(",")]
    except ValueError as error:
        raise SystemExit("--timepoints-sec must be comma-separated decimals") from error
    if not timepoints or any(value >= 0 for value in timepoints):
        raise SystemExit("Every timepoint must precede impact")

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if calibration.get("validation", {}).get("status") != "usable":
        raise SystemExit("Calibration is not usable; correct the held-out net-point check first")
    try:
        transform = np.asarray(calibration["image_to_court_homography"], dtype=np.float32)
    except (KeyError, ValueError) as error:
        raise SystemExit("Calibration has no valid image_to_court_homography") from error
    if transform.shape != (3, 3):
        raise SystemExit("Calibration homography must be 3x3")

    identities = {
        row["event_id"]: row
        for row in read(args.identity_review)
        if (
            row.get("video_id") == args.video_id
            and row.get("identity_status") == "confirmed"
            and row.get("target_front_player_id")
            and row.get("target_front_camera_end") in ("near", "far")
        )
    }
    if not identities:
        raise SystemExit("No confirmed target-front identities")

    labels: dict[str, dict[str, str]] = {}
    for path in args.direction_review:
        for source in read(path):
            direction = source.get("actual_direction_target_front_view") or source.get("actual_direction_front_view")
            event_id = source.get("candidate_id", "")
            if (
                source.get("video_id") == args.video_id
                and source.get("review_status") == "confirmed"
                and event_id in identities
                and direction in ("front_left", "front_right")
            ):
                existing = labels.get(event_id)
                if existing and existing["actual_direction_front_view"] != direction:
                    raise SystemExit(f"Conflicting direction labels for {event_id}")
                labels[event_id] = {
                    "impact_source_frame": source["impact_source_frame"],
                    "actual_direction_front_view": direction,
                }
    pose_usable: set[str] = set()
    for path in args.pose_review:
        pose_usable.update(
            row["event_id"] for row in read(path)
            if row.get("video_id") == args.video_id and row.get("pose_status") == "usable"
        )
    poses_by_event: dict[str, dict[int, dict[str, float]]] = {}
    for path in args.pose:
        for row in read(path):
            event_id = row.get("event_id", "")
            if event_id not in labels or event_id not in pose_usable:
                continue
            court = transform_ankle_pair(row, transform)
            if court is None:
                continue
            poses_by_event.setdefault(event_id, {})[int(row["source_frame"])] = {
                "court_x": court[0], "court_y": court[1],
            }
    event_ids = sorted(set(identities) & set(labels) & pose_usable & set(poses_by_event))
    if not event_ids:
        raise SystemExit("No events jointly have identity, direction, usable pose, and court coordinates")

    motion_frames = round(args.motion_window_ms / 1000 * FPS)
    fields = [
        "event_id", "video_id", "target_front_player_id", "target_front_camera_end",
        "impact_source_frame", "actual_direction_front_view", "timepoint_sec", "timepoint_source_frame",
        "court_x_median", "court_y_median", "position_samples", "impact_court_x_median",
        "relative_to_impact_court_x", "motion_start_source_frame", "motion_end_source_frame",
        "motion_start_court_x_median", "motion_end_court_x_median", "motion_start_samples",
        "motion_end_samples", "lateral_displacement_court_width", "lateral_velocity_court_width_per_sec",
        "feature_status", "coordinate_definition",
    ]
    output_rows: list[dict[str, object]] = []
    for event_id in event_ids:
        identity = identities[event_id]
        label = labels[event_id]
        impact = int(label["impact_source_frame"])
        rows_by_frame = poses_by_event[event_id]
        impact_x, _ = median_coordinate(rows_by_frame, impact, args.median_radius_frames, "court_x")
        for timepoint in timepoints:
            point_frame = impact + round(timepoint * FPS)
            start_frame = point_frame - motion_frames
            point_x, point_n = median_coordinate(rows_by_frame, point_frame, args.median_radius_frames, "court_x")
            point_y, _ = median_coordinate(rows_by_frame, point_frame, args.median_radius_frames, "court_y")
            start_x, start_n = median_coordinate(rows_by_frame, start_frame, args.median_radius_frames, "court_x")
            end_x, end_n = median_coordinate(rows_by_frame, point_frame, args.median_radius_frames, "court_x")
            if point_x is None or start_x is None or end_x is None:
                displacement: float | None = None
                status = "missing_pose"
            else:
                displacement = end_x - start_x
                status = "usable"
            output_rows.append({
                "event_id": event_id,
                "video_id": args.video_id,
                "target_front_player_id": identity["target_front_player_id"],
                "target_front_camera_end": identity["target_front_camera_end"],
                "impact_source_frame": impact,
                "actual_direction_front_view": label["actual_direction_front_view"],
                "timepoint_sec": f"{timepoint:.3f}",
                "timepoint_source_frame": point_frame,
                "court_x_median": "" if point_x is None else f"{point_x:.6f}",
                "court_y_median": "" if point_y is None else f"{point_y:.6f}",
                "position_samples": point_n,
                "impact_court_x_median": "" if impact_x is None else f"{impact_x:.6f}",
                "relative_to_impact_court_x": "" if point_x is None or impact_x is None else f"{point_x - impact_x:.6f}",
                "motion_start_source_frame": start_frame,
                "motion_end_source_frame": point_frame,
                "motion_start_court_x_median": "" if start_x is None else f"{start_x:.6f}",
                "motion_end_court_x_median": "" if end_x is None else f"{end_x:.6f}",
                "motion_start_samples": start_n,
                "motion_end_samples": end_n,
                "lateral_displacement_court_width": "" if displacement is None else f"{displacement:.6f}",
                "lateral_velocity_court_width_per_sec": "" if displacement is None else f"{displacement / (motion_frames / FPS):.6f}",
                "feature_status": status,
                "coordinate_definition": (
                    "fixed_main_camera image-to-normalized-court homography; x=0 screen-left sideline, "
                    "x=1 screen-right sideline; not metres"
                ),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(output_rows)
    usable = sum(row["feature_status"] == "usable" for row in output_rows)
    args.output.with_suffix(".metadata.json").write_text(json.dumps({
        "video_id": args.video_id,
        "events": len(event_ids),
        "event_outcome_counts": dict(Counter(labels[event_id]["actual_direction_front_view"] for event_id in event_ids)),
        "timepoints_sec": timepoints,
        "motion_window_ms": args.motion_window_ms,
        "motion_window_frames": motion_frames,
        "median_radius_frames": args.median_radius_frames,
        "calibration": str(args.calibration),
        "calibration_validation": calibration["validation"],
        "direction_reviews": [str(path) for path in args.direction_review],
        "pose_reviews": [str(path) for path in args.pose_review],
        "poses": [str(path) for path in args.pose],
        "identity_review": str(args.identity_review),
        "analysis_guard": (
            "This single-match table applies one manually validated fixed-camera homography. "
            "It enables near/far pooling for lateral court-width displacement, but not causal, "
            "anticipation, or player-independent claims."
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(output_rows)} rows from {len(event_ids)} events; usable {usable}/{len(output_rows)}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
