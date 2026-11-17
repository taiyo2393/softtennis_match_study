"""Run a limited person-tracking pilot on one analysis-ready fixed-camera segment.

This creates *candidate* player tracks only. It does not identify the target
front player and does not make these coordinates scientific measurements.
Review track-ID swaps and missed players before using any row for Pose.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


# Expanded, match1-specific court vicinity. A player's foot can appear below the
# visible baseline, hence this is wider than the line-detection trapezoid.
COURT_VICINITY = np.array([[0.02, 0.90], [0.98, 0.90], [0.68, 0.51], [0.32, 0.51]], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--segment-id", default=None, help="Default: longest analysis-ready interval")
    parser.add_argument("--screening-segment-id", default=None,
                        help="Use one human-confirmed boundary window from reviewed_screening_segments.csv.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--processing-fps", type=float, default=15.0,
                        help="Pilot-only sampling rate; never use this for the final timing analysis.")
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--disable-court-filter", action="store_true",
                        help="Diagnostic only. Includes spectators and officials, so not appropriate as the default.")
    return parser.parse_args()


def read_segments(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return [row for row in csv.DictReader(file) if row["usable_for_analysis"].lower() == "true"]


def select_segment(rows: list[dict[str, str]], requested: str | None) -> dict[str, str]:
    if requested:
        found = [row for row in rows if row["segment_id"] == requested]
        if not found:
            raise SystemExit(f"segment_id not found: {requested}")
        return found[0]
    return max(rows, key=lambda row: float(row["end_timestamp_sec"]) - float(row["start_timestamp_sec"]))


def select_screening_segment(path: Path, requested: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    found = [row for row in rows if row["screening_segment_id"] == requested]
    if not found:
        raise SystemExit(f"screening_segment_id not found: {requested}")
    row = found[0]
    return {
        "segment_id": row["screening_segment_id"],
        "start_timestamp_sec": row["source_start_timestamp_sec"],
        "end_timestamp_sec": row["source_end_timestamp_sec"],
    }


def in_court_vicinity(x: float, y: float, width: int, height: int) -> bool:
    polygon = (COURT_VICINITY * np.array([width, height])).astype(np.float32)
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0


def main() -> int:
    args = parse_args()
    project = Path(__file__).resolve().parents[1]
    os.environ.setdefault("YOLO_CONFIG_DIR", str(project / ".ultralytics"))
    from ultralytics import YOLO  # Import after the project-local config directory is set.

    if args.segment_id and args.screening_segment_id:
        raise SystemExit("Use either --segment-id or --screening-segment-id, not both.")
    if args.screening_segment_id:
        segment = select_screening_segment(project / "data" / "segments" / f"{args.video_id}_reviewed_screening_segments.csv", args.screening_segment_id)
    else:
        segments = read_segments(project / "data" / "segments" / "analysis_ready_main_camera_segments.csv")
        segment = select_segment(segments, args.segment_id)
    start_sec, end_sec = float(segment["start_timestamp_sec"]), float(segment["end_timestamp_sec"])
    model_path = args.model or project / "models" / "yolo11n.pt"
    if not model_path.exists():
        raise SystemExit(f"Missing model: {model_path}. Download it explicitly before tracking.")

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open: {args.input}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        raise SystemExit("Cannot read source FPS")
    source_width, source_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame, end_frame = round(start_sec * source_fps), round(end_sec * source_fps)
    stride = max(1, round(source_fps / args.processing_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    tracking_dir = project / "data" / "tracking"; tracking_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.video_id}_{segment['segment_id']}_person_tracking_pilot"
    csv_path, video_path = tracking_dir / f"{stem}.csv", tracking_dir / f"{stem}.mp4"
    temporary_video_path = tracking_dir / f"{stem}_temporary.mp4"
    params_path = tracking_dir / f"{stem}.json"
    model = YOLO(str(model_path))
    writer = None
    fields = ["video_id", "segment_id", "source_timestamp_sec", "source_frame", "processed_frame_index", "track_id",
              "class_name", "confidence", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px",
              "center_x_px", "center_y_px", "detection_width_px", "detection_height_px", "review_status", "notes"]
    processed, rows_written = 0, 0
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        out = csv.DictWriter(file, fieldnames=fields); out.writeheader()
        for source_frame in range(start_frame, end_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if (source_frame - start_frame) % stride:
                continue
            scale = min(1.0, args.max_width / frame.shape[1])
            detection_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
            result = model.track(detection_frame, persist=True, classes=[0], conf=args.confidence,
                                 imgsz=1280, tracker="bytetrack.yaml", verbose=False)[0]
            annotated = detection_frame.copy()
            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.int().cpu().tolist()
                boxes = result.boxes.xyxy.cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()
                for track_id, box, confidence in zip(ids, boxes, confidences):
                    foot_x, foot_y = (box[0] + box[2]) / 2, box[3]
                    if not args.disable_court_filter and not in_court_vicinity(foot_x, foot_y, detection_frame.shape[1], detection_frame.shape[0]):
                        continue
                    cv2.rectangle(annotated, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])), (255, 80, 0), 3)
                    cv2.putText(annotated, f"id:{track_id} person {confidence:.2f}", (round(box[0]), max(24, round(box[1])-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 80, 0), 2, cv2.LINE_AA)
                    x1, y1, x2, y2 = [value / scale for value in box]
                    out.writerow({"video_id": args.video_id, "segment_id": segment["segment_id"],
                                  "source_timestamp_sec": f"{source_frame / source_fps:.6f}", "source_frame": source_frame,
                                  "processed_frame_index": processed, "track_id": track_id, "class_name": "person",
                                  "confidence": f"{confidence:.6f}", "bbox_x1_px": f"{x1:.3f}", "bbox_y1_px": f"{y1:.3f}",
                                  "bbox_x2_px": f"{x2:.3f}", "bbox_y2_px": f"{y2:.3f}",
                                  "center_x_px": f"{(x1+x2)/2:.3f}", "center_y_px": f"{(y1+y2)/2:.3f}",
                                  "detection_width_px": source_width, "detection_height_px": source_height,
                                  "review_status": "unreviewed", "notes": ""})
                    rows_written += 1
            if writer is None:
                preview_width = 960
                preview_height = round(annotated.shape[0] * preview_width / annotated.shape[1])
                writer = cv2.VideoWriter(str(temporary_video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.processing_fps,
                                         (preview_width, preview_height))
            preview = cv2.resize(annotated, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
            cv2.putText(preview, f"source {source_frame} / {source_frame / source_fps:.3f}s", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            writer.write(preview)
            processed += 1
    cap.release()
    if writer is not None:
        writer.release()
        # Chromium-based browsers reliably seek H.264 MP4, unlike OpenCV's mp4v output.
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(temporary_video_path),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path)], check=True)
        temporary_video_path.unlink()
    params_path.write_text(json.dumps({"purpose": "candidate person tracking pilot; not final analysis data",
        "video_id": args.video_id, "segment_id": segment["segment_id"], "start_timestamp_sec": start_sec,
        "end_timestamp_sec": end_sec, "source_fps": source_fps, "processing_fps": args.processing_fps,
        "stride_source_frames": stride, "source_resolution": [source_width, source_height], "model": str(model_path),
        "court_filter": ("foot point within expanded match1 court vicinity" if not args.disable_court_filter else "disabled"),
        "court_vicinity_normalized_polygon": COURT_VICINITY.tolist(), "rows_written": rows_written,
        "review_required": ["target-front-player selection", "track-ID swaps", "missed detections"]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Segment: {segment['segment_id']} ({start_sec:.3f}-{end_sec:.3f}s)")
    print(f"Wrote {rows_written} candidate boxes across {processed} processed frames")
    print(csv_path); print(video_path); print(params_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
