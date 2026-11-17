"""Extract target-front ankle landmarks inside audited high-resolution tracks.

The pose model sees only a padded crop around the already audited target-front
box.  It is therefore a pose proposal for that player, not a four-person pose
tracker.  Missing ankles remain missing and are never interpolated.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
from pathlib import Path

import cv2


LEFT_ANKLE, RIGHT_ANKLE = 15, 16  # COCO keypoint indices


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def padded_crop(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box; box_width, box_height = x2 - x1, y2 - y1
    return (
        max(0, round(x1 - .55 * box_width)), max(0, round(y1 - .18 * box_height)),
        min(width, round(x2 + .55 * box_width)), min(height, round(y2 + .22 * box_height)),
    )


def select_pose(result: object, target_box: tuple[float, float, float, float], offset_x: int, offset_y: int) -> tuple[int | None, float]:
    if result.boxes is None or len(result.boxes) == 0:
        return None, 0.0
    scored: list[tuple[float, int]] = []
    for index, box in enumerate(result.boxes.xyxy.cpu().tolist()):
        restored = (box[0] + offset_x, box[1] + offset_y, box[2] + offset_x, box[3] + offset_y)
        score = iou(restored, target_box)
        scored.append((score, index))
    score, index = max(scored, default=(0.0, None))
    return index, score


def blank_row(event_id: str, video_id: str, source_frame: int, reason: str) -> dict[str, object]:
    return {"event_id": event_id, "video_id": video_id, "source_frame": source_frame, "pose_detected": 0, "selection_iou": "", "left_ankle_x_px": "", "left_ankle_y_px": "", "left_ankle_confidence": "", "right_ankle_x_px": "", "right_ankle_y_px": "", "right_ankle_confidence": "", "ankle_midpoint_x_px": "", "ankle_pair_visible": 0, "missing_reason": reason}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-stem", default="yolo11m")
    parser.add_argument("--event-id", default=None, help="Run one audited event as a smoke test")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=.15)
    parser.add_argument("--ankle-confidence", type=float, default=.25)
    parser.add_argument("--output-stem", default=None)
    parser.add_argument(
        "--track-prefix", default=None,
        help="Separate high-resolution track batch prefix made by script 53.",
    )
    args = parser.parse_args()
    if not args.input.is_file() or not args.model.is_file():
        raise SystemExit("--input and --model must be existing files")
    project = Path(__file__).resolve().parents[1]
    os.environ.setdefault("YOLO_CONFIG_DIR", str(project / ".ultralytics"))
    from ultralytics import YOLO  # noqa: PLC0415

    if args.track_prefix:
        review_path = project / "annotations" / f"{args.track_prefix}_track_review_{args.model_stem}.csv"
        track_path = project / "data" / "tracking" / "front_target_highres" / f"{args.track_prefix}_tracks_{args.model_stem}.csv"
    else:
        review_path = project / "annotations" / f"highres_front_track_review_{args.video_id}_{args.model_stem}.csv"
        track_path = project / "data" / "tracking" / "front_target_highres" / f"{args.video_id}_target_front_tracks_{args.model_stem}.csv"
    if not review_path.is_file() or not track_path.is_file():
        raise SystemExit("Audited target-front review or track CSV is missing")
    review = {row["event_id"] for row in read(review_path) if row["tracking_status"] == "usable"}
    tracks = [row for row in read(track_path) if row["event_id"] in review]
    if args.event_id:
        tracks = [row for row in tracks if row["event_id"] == args.event_id]
    if not tracks:
        raise SystemExit("No usable audited target-front tracks found")
    capture = cv2.VideoCapture(str(args.input)); width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps = capture.get(cv2.CAP_PROP_FPS)
    if width <= 0 or height <= 0 or fps <= 0:
        raise SystemExit("Could not read video metadata")
    model = YOLO(str(args.model))
    rows: list[dict[str, object]] = []
    # Different impact windows can overlap.  Keep both event/frame records:
    # the same decoded source frame may legitimately contribute to two events.
    for track in sorted(tracks, key=lambda row: (int(row["source_frame"]), row["event_id"])):
        source_frame = int(track["source_frame"]); event_id = track["event_id"]
        if track["visible"] != "1":
            rows.append(blank_row(event_id, args.video_id, source_frame, "tracker_missing")); continue
        target = tuple(float(track[key]) for key in ("x1_px", "y1_px", "x2_px", "y2_px"))
        x1, y1, x2, y2 = padded_crop(target, width, height)
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame); ok, frame = capture.read()
        if not ok:
            rows.append(blank_row(event_id, args.video_id, source_frame, "source_frame_unreadable")); continue
        result = model.predict(frame[y1:y2, x1:x2], classes=[0], conf=args.confidence, imgsz=args.imgsz, verbose=False)[0]
        index, selection_iou = select_pose(result, target, x1, y1)
        if index is None or result.keypoints is None:
            rows.append(blank_row(event_id, args.video_id, source_frame, "pose_missing")); continue
        points = result.keypoints.xy.cpu().tolist()[index]
        confidences = result.keypoints.conf.cpu().tolist()[index]
        left, right = points[LEFT_ANKLE], points[RIGHT_ANKLE]
        left_conf, right_conf = confidences[LEFT_ANKLE], confidences[RIGHT_ANKLE]
        left_visible = left_conf >= args.ankle_confidence; right_visible = right_conf >= args.ankle_confidence
        midpoint = (left[0] + right[0]) / 2 + x1 if left_visible and right_visible else None
        rows.append({
            "event_id": event_id, "video_id": args.video_id, "source_frame": source_frame, "pose_detected": 1,
            "selection_iou": f"{selection_iou:.4f}",
            "left_ankle_x_px": f"{left[0] + x1:.3f}" if left_visible else "", "left_ankle_y_px": f"{left[1] + y1:.3f}" if left_visible else "", "left_ankle_confidence": f"{left_conf:.4f}",
            "right_ankle_x_px": f"{right[0] + x1:.3f}" if right_visible else "", "right_ankle_y_px": f"{right[1] + y1:.3f}" if right_visible else "", "right_ankle_confidence": f"{right_conf:.4f}",
            "ankle_midpoint_x_px": "" if midpoint is None else f"{midpoint:.3f}", "ankle_pair_visible": int(midpoint is not None),
            "missing_reason": "" if midpoint is not None else "one_or_both_ankles_low_confidence",
        })
    capture.release()
    output_dir = project / "data" / "pose" / "target_front"; output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem or f"{args.video_id}_target_front_pose_{args.model.stem}"
    output = output_dir / f"{stem}.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    paired = sum(int(row["ankle_pair_visible"]) for row in rows)
    print(f"Wrote {len(rows)} pose rows; ankle-pair coverage {paired}/{len(rows)} ({paired / len(rows):.1%}): {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
