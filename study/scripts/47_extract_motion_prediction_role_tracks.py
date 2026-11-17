"""Extract short-window player tracks for human-confirmed motion-prediction events.

The fixed broadcast view also detects line judges and ball persons as ``person``.
It therefore proposes four player-role tracks from camera-specific depth bands,
rather than requiring exactly four detections.  The output remains a candidate
trajectory and must be visually audited before it is used for prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


ROLES = ("far_rear", "far_front", "near_front", "near_rear")
COLORS = {"far_rear": (255, 180, 0), "far_front": (0, 220, 0), "near_front": (0, 160, 255), "near_rear": (255, 0, 180)}
COURT_VICINITY = np.array([[0.02, 0.90], [0.98, 0.90], [0.68, 0.51], [0.32, 0.51]], dtype=np.float32)
# Valid only for match1's fixed, end-on main-camera geometry.  The ranges are
# intentionally broad and are a proposal rule, not a claim of player identity.
ROLE_Y_BANDS = {
    "near_rear": (0.68, 0.93),
    "near_front": (0.60, 0.70),
    "far_front": (0.58, 0.64),
    "far_rear": (0.52, 0.60),
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def in_court(x: float, y: float, width: int, height: int) -> bool:
    polygon = (COURT_VICINITY * np.array([width, height])).astype(np.float32)
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0


def detections(result: object, scale: float, width: int, height: int) -> list[dict[str, float | int]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None:
        return []
    found: list[dict[str, float | int]] = []
    for track_id, box, confidence in zip(boxes.id.int().cpu().tolist(), boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()):
        x1, y1, x2, y2 = [float(value) / scale for value in box]
        foot_x, foot_y = (x1 + x2) / 2, y2
        if in_court(foot_x * scale, foot_y * scale, round(width * scale), round(height * scale)):
            found.append({"track_id": int(track_id), "x1": x1, "y1": y1, "x2": x2, "y2": y2, "foot_x": foot_x, "foot_y": foot_y, "confidence": float(confidence)})
    return found


def role_map(found: list[dict[str, float | int]], height: int) -> dict[str, int]:
    """Propose one ID for each fixed-camera depth slot.

    Staff detections can outnumber players even inside the coarse court mask.
    Select the highest-confidence unused ID in each expected depth band and
    leave the event unassigned if any band has no candidate.  A later visual
    review accepts/rejects the proposal; this function does not establish
    ground-truth roles.
    """
    unique: dict[int, dict[str, float | int]] = {}
    for row in found:
        track_id = int(row["track_id"])
        if track_id not in unique or float(row["confidence"]) > float(unique[track_id]["confidence"]):
            unique[track_id] = row
    mapping: dict[str, int] = {}
    used: set[int] = set()
    # Start nearest to farthest where the bands overlap, so the nearer player
    # occupies its more specific slot before the far-rear fallback is chosen.
    for role in ("near_rear", "near_front", "far_front", "far_rear"):
        low, high = ROLE_Y_BANDS[role]
        candidates = [
            row for row in unique.values()
            if int(row["track_id"]) not in used and low <= float(row["foot_y"]) / height <= high
        ]
        if not candidates:
            return {}
        chosen = max(candidates, key=lambda row: float(row["confidence"]))
        track_id = int(chosen["track_id"])
        mapping[role] = track_id
        used.add(track_id)
    return {role: mapping[role] for role in ROLES}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input video not found: {args.input}")
    project = Path(__file__).resolve().parents[1]
    os.environ.setdefault("YOLO_CONFIG_DIR", str(project / ".ultralytics"))
    from ultralytics import YOLO  # noqa: PLC0415

    events_path = project / "annotations" / "motion_prediction_events.csv"
    events = [row for row in read(events_path) if row["video_id"] == args.video_id and row["inclusion_status"] == "included"]
    if not events:
        raise SystemExit("No included motion-prediction events. Confirm events in script 46 first.")
    model_path = args.model or project / "models" / "yolo11n.pt"
    if not model_path.is_file():
        raise SystemExit(f"Missing model: {model_path}")

    capture = cv2.VideoCapture(str(args.input))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise SystemExit("Could not read input video metadata")
    model = YOLO(str(model_path))
    track_dir = project / "data" / "tracking" / "motion_prediction_events"
    preview_dir = track_dir / "previews"; preview_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    output_csv = track_dir / f"{args.video_id}_event_role_tracks.csv"

    for event in events:
        event_id = event["event_id"]
        impact_frame = int(event["impact_source_frame"])
        start_frame = round((float(event["impact_timestamp_sec"]) - 0.5) * fps)
        end_frame = round((float(event["impact_timestamp_sec"]) + 0.3) * fps)
        capture = cv2.VideoCapture(str(args.input)); capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        temporary = preview_dir / f"{event_id}_temporary.avi"
        preview = preview_dir / f"{event_id}.mp4"
        if temporary.exists(): temporary.unlink()
        writer: cv2.VideoWriter | None = None
        mapping: dict[str, int] = {}
        impact_found: list[dict[str, float | int]] = []
        found_by_frame: dict[int, list[dict[str, float | int]]] = {}
        detections_per_role = {role: 0 for role in ROLES}
        local_index = 0
        # Keep all candidate detections before assigning labels.  The mapping
        # is defined at impact, so writing rows in this first pass would omit
        # every pre-impact observation.
        for source_frame in range(start_frame, end_frame + 1):
            ok, frame = capture.read()
            if not ok:
                break
            scale = min(1.0, args.max_width / frame.shape[1])
            resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
            result = model.track(resized, persist=True, classes=[0], conf=args.confidence, imgsz=1280, tracker="bytetrack.yaml", verbose=False)[0]
            found = detections(result, scale, width, height)
            found_by_frame[source_frame] = found
            if source_frame == impact_frame:
                impact_found = found
                mapping = role_map(found, height)
            local_index += 1
        capture.release()
        inverse = {track_id: role for role, track_id in mapping.items()}
        for source_frame, found in found_by_frame.items():
            for item in found:
                track_id = int(item["track_id"])
                role = inverse.get(track_id)
                if not role:
                    continue
                detections_per_role[role] += 1
                rows.append({"event_id": event_id, "video_id": args.video_id, "source_frame": source_frame, "timestamp_sec": f"{source_frame / fps:.6f}", "relative_to_impact_sec": f"{source_frame / fps - float(event['impact_timestamp_sec']):.6f}", "role": role, "track_id": track_id, "foot_x_px": f"{float(item['foot_x']):.3f}", "foot_y_px": f"{float(item['foot_y']):.3f}", "foot_x_norm": f"{float(item['foot_x']) / width:.8f}", "foot_y_norm": f"{float(item['foot_y']) / height:.8f}", "confidence": f"{float(item['confidence']):.6f}", "assignment_basis": "match1_fixed_camera_depth_band_proposal", "review_status": "unreviewed"})
        # Render in a second pass so labels are visible before t=0 as well.
        capture = cv2.VideoCapture(str(args.input)); capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for source_frame in range(start_frame, end_frame + 1):
            ok, frame = capture.read()
            if not ok:
                break
            annotated = frame.copy()
            for item in found_by_frame.get(source_frame, []):
                track_id = int(item["track_id"])
                role = inverse.get(track_id)
                if not role:
                    continue
                x1, y1, x2, y2 = (round(float(item[key])) for key in ("x1", "y1", "x2", "y2"))
                color = COLORS[role]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
                cv2.putText(annotated, f"{role} / id {track_id}", (x1, max(28, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            cv2.putText(annotated, f"{event_id} / source frame {source_frame} / rel {source_frame / fps - float(event['impact_timestamp_sec']):+.3f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            if writer is None:
                preview_width = 1280; preview_height = round(annotated.shape[0] * preview_width / annotated.shape[1])
                writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"MJPG"), fps, (preview_width, preview_height))
            writer.write(cv2.resize(annotated, (preview_width, preview_height), interpolation=cv2.INTER_AREA))
        capture.release()
        if writer is not None:
            writer.release()
            if preview.exists() and args.overwrite: preview.unlink()
            if not preview.exists():
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(temporary), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(preview)], check=True)
            temporary.unlink(missing_ok=True)
        target_role = event["target_front_player_id"]
        target_track = mapping.get(target_role, "")
        summaries.append({"event_id": event_id, "impact_source_frame": impact_frame, "window_start_frame": start_frame, "window_end_frame": end_frame, "frames_read": local_index, "impact_detection_count": len(impact_found), "role_assignment_status": "assigned" if mapping else "needs_review_not_exactly_four_players", "role_track_ids": json.dumps(mapping, ensure_ascii=False), "target_front_role": target_role, "target_front_track_id": target_track, "target_front_observation_rate": f"{detections_per_role.get(target_role, 0) / max(1, local_index):.4f}", "preview_path": str(preview)})
        print(f"{event_id}: {summaries[-1]['role_assignment_status']}, target observations={summaries[-1]['target_front_observation_rate']}")

    fields = ["event_id", "video_id", "source_frame", "timestamp_sec", "relative_to_impact_sec", "role", "track_id", "foot_x_px", "foot_y_px", "foot_x_norm", "foot_y_norm", "confidence", "assignment_basis", "review_status"]
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary_path = track_dir / f"{args.video_id}_event_role_track_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    print(output_csv); print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
