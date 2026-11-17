"""Render short pose-audit videos for target-front ankle landmarks."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import cv2


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--pose-stem", default="match1_target_front_pose_yolo11m")
    parser.add_argument("--track-model-stem", default="yolo11m")
    parser.add_argument(
        "--track-prefix", default=None,
        help="Separate high-resolution track batch prefix made by script 53.",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    pose_path = project / "data" / "pose" / "target_front" / f"{args.pose_stem}.csv"
    track_prefix = args.track_prefix or f"{args.video_id}_target_front"
    track_path = project / "data" / "tracking" / "front_target_highres" / f"{track_prefix}_tracks_{args.track_model_stem}.csv"
    if not args.input.is_file() or not pose_path.is_file() or not track_path.is_file():
        raise SystemExit("Input video, pose CSV, or target-track CSV is missing")
    poses_by_event: dict[str, list[dict[str, str]]] = {}
    for row in read(pose_path):
        poses_by_event.setdefault(row["event_id"], []).append(row)
    track_by_key = {(row["event_id"], int(row["source_frame"])): row for row in read(track_path)}
    output_dir = project / "data" / "pose" / "target_front" / "previews"; output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.input)); fps = capture.get(cv2.CAP_PROP_FPS); width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise SystemExit("Could not read input video")
    target_width = 1280; target_height = round(height * target_width / width)
    for event_id, poses in poses_by_event.items():
        poses.sort(key=lambda row: int(row["source_frame"])); start, end = int(poses[0]["source_frame"]), int(poses[-1]["source_frame"])
        by_frame = {int(row["source_frame"]): row for row in poses}
        temporary = output_dir / f"{event_id}.avi"; output = output_dir / f"{event_id}.mp4"; temporary.unlink(missing_ok=True)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"MJPG"), fps, (target_width, target_height))
        for frame_number in range(start, end + 1):
            ok, frame = capture.read()
            if not ok:
                break
            pose = by_frame[frame_number]; track = track_by_key.get((event_id, frame_number))
            if track and track["visible"] == "1":
                x1, y1, x2, y2 = (round(float(track[key])) for key in ("x1_px", "y1_px", "x2_px", "y2_px"))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 5)
            if pose["ankle_pair_visible"] == "1":
                left = (round(float(pose["left_ankle_x_px"])), round(float(pose["left_ankle_y_px"])))
                right = (round(float(pose["right_ankle_x_px"])), round(float(pose["right_ankle_y_px"])))
                midpoint = (round(float(pose["ankle_midpoint_x_px"])), round((left[1] + right[1]) / 2))
                cv2.circle(frame, left, 15, (255, 80, 255), 4, cv2.LINE_AA); cv2.circle(frame, right, 15, (255, 80, 255), 4, cv2.LINE_AA)
                cv2.line(frame, left, right, (0, 220, 0), 4, cv2.LINE_AA); cv2.circle(frame, midpoint, 12, (0, 220, 0), -1, cv2.LINE_AA)
                note = "ankle pair usable"
            else:
                note = f"ankle pair missing: {pose['missing_reason']}"
            cv2.putText(frame, f"{event_id} / F{frame_number} / {note}", (28, 50), cv2.FONT_HERSHEY_SIMPLEX, .85, (255, 255, 255), 3, cv2.LINE_AA)
            writer.write(cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA))
        writer.release()
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(temporary), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], check=True)
        temporary.unlink(missing_ok=True)
        print(output)
    capture.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
