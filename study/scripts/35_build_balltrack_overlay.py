"""Render RacketVision BallTrack candidates over one pilot clip for visual audit."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    clip = project / "data" / "segments" / "motion_prediction_pilot" / args.video_id / f"{args.candidate_id}.mp4"
    candidate_dir = project / "data" / "balltrack" / "racketvision" / args.video_id / args.candidate_id
    output_dir = candidate_dir if args.run_name == "default" else candidate_dir / "runs" / args.run_name
    prediction_path = output_dir / "balltrack_predictions.csv"
    output = output_dir / "balltrack_overlay.mp4"
    temporary = output_dir / "balltrack_overlay_temp.avi"
    if output.exists() and not args.overwrite:
        print(f"Overlay already exists: {output}")
        return 0
    if not clip.is_file() or not prediction_path.is_file():
        raise SystemExit("Run script 34 first; pilot clip or prediction CSV is missing.")
    predictions = {int(row["local_frame"]): row for row in csv.DictReader(prediction_path.open(encoding="utf-8"))}
    capture = cv2.VideoCapture(str(clip))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Cannot create temporary overlay video")
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        row = predictions.get(index)
        if row and int(row["visible"]):
            x, y = int(row["x_px"]), int(row["y_px"])
            confidence = float(row["confidence"])
            cv2.circle(frame, (x, y), 12, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"BallTrack {confidence:.2f}", (x + 14, max(26, y - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "BallTrack: no candidate", (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"frame {index}", (22, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        index += 1
    capture.release()
    writer.release()
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(temporary), "-i", str(clip),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-movflags", "+faststart", str(output),
    ], check=True)
    temporary.unlink(missing_ok=True)
    print(f"Wrote {index} overlay frames: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
