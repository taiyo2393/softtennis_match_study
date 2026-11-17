"""Score whether the required fixed-camera court-line geometry is visible."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np


# Normalized trapezoid enclosing the playable court in match1's fixed wide view.
# It intentionally excludes score graphics, stands, and most player action.
COURT_POLYGON = np.array([[0.07, 0.78], [0.93, 0.77], [0.64, 0.56], [0.36, 0.56]], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--reference-sec", type=float, required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def shape(width: int) -> tuple[int, int]:
    return round(width * 9 / 16), width


def decode_one(video: Path, timestamp: float, width: int) -> np.ndarray:
    height, width = shape(width)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{timestamp:.6f}", "-i", str(video), "-frames:v", "1", "-vf", f"scale={width}:{height}", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    result = subprocess.run(command, capture_output=True, check=True)
    return np.frombuffer(result.stdout, np.uint8).reshape(height, width, 3)


def court_mask(height: int, width: int) -> np.ndarray:
    points = (COURT_POLYGON * np.array([width, height])).astype(np.int32)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, points, 255)
    return mask


def white_mask(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Court lines are bright and low saturation; the ROI removes most white text.
    result = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([180, 95, 255]))
    return cv2.bitwise_and(result, roi)


def make_line_template(reference: np.ndarray, roi: np.ndarray) -> np.ndarray:
    binary = white_mask(reference, roi)
    edges = cv2.Canny(binary, 40, 120)
    detected = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=55, maxLineGap=25)
    template = np.zeros_like(roi)
    if detected is None:
        raise RuntimeError("No court lines detected in reference frame")
    for x1, y1, x2, y2 in detected.reshape(-1, 4):
        length = math.hypot(x2 - x1, y2 - y1)
        if length >= 55:
            cv2.line(template, (x1, y1), (x2, y2), 255, thickness=5)
    template = cv2.bitwise_and(template, roi)
    if cv2.countNonZero(template) < 1000:
        raise RuntimeError("Court-line template is too small")
    return template


def geometry_score(image: np.ndarray, template: np.ndarray, roi: np.ndarray) -> float:
    candidate = white_mask(image, roi)
    candidate = cv2.dilate(candidate, np.ones((9, 9), np.uint8))
    return cv2.countNonZero(cv2.bitwise_and(template, candidate)) / cv2.countNonZero(template)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    args = parse_args()
    project = Path(__file__).resolve().parents[1]
    segment_dir = project / "data" / "segments"
    reference = decode_one(args.input, args.reference_sec, args.width)
    height, width = shape(args.width)
    roi = court_mask(height, width)
    template = make_line_template(reference, roi)
    overlay = reference.copy(); overlay[template > 0] = (0, 0, 255)
    cv2.imwrite(str(segment_dir / f"{args.video_id}_court_line_template.jpg"), overlay)

    if args.audit_only:
        audit_path = segment_dir / f"{args.video_id}_fixed_view_audit.csv"
        with audit_path.open(newline="", encoding="utf-8") as file: audit_rows = list(csv.DictReader(file))
        rows = []
        for row in audit_rows:
            image = cv2.imread(str(segment_dir / "audit" / args.video_id / row["image_filename"]))
            rows.append({**row, "court_geometry_score": f"{geometry_score(image, template, roi):.6f}"})
        write_csv(segment_dir / f"{args.video_id}_court_geometry_audit.csv", list(rows[0]), rows)
        print(f"Scored {len(rows)} audit images")
        return 0

    bytes_per_frame = height * width * 3
    process = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(args.input), "-an", "-vf", f"fps={args.sample_fps},scale={width}:{height}", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    rows = []; index = 0
    assert process.stdout is not None
    while raw := process.stdout.read(bytes_per_frame):
        if len(raw) != bytes_per_frame: raise RuntimeError("Incomplete FFmpeg frame")
        image = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
        rows.append({"video_id": args.video_id, "timestamp_sec": f"{index / args.sample_fps:.6f}", "court_geometry_score": f"{geometry_score(image, template, roi):.6f}"})
        index += 1
    if process.wait() != 0: raise RuntimeError(process.stderr.read().decode("utf-8", errors="replace"))
    write_csv(segment_dir / f"{args.video_id}_court_geometry_scores.csv", ["video_id", "timestamp_sec", "court_geometry_score"], rows)
    print(f"Scored {len(rows)} frames")
    return 0


if __name__ == "__main__":
    main()
