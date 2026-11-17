"""Keep intervals matching one fixed main-camera view; reject all other views.

The reference is one known-good frame from the fixed, wide court camera.  Each
sampled frame is compared only on stable background regions, not on players.
This is deliberately a camera-view classifier, not a replay classifier.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--duration-sec", required=True, type=float)
    parser.add_argument("--reference-sec", required=True, type=float)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--difference-threshold", type=float, default=0.12)
    parser.add_argument("--edge-margin-sec", type=float, default=1.0)
    parser.add_argument("--min-duration-sec", type=float, default=2.0)
    return parser.parse_args()


def frame_shape(width: int) -> tuple[int, int, int]:
    height = round(width * 9 / 16)
    return height, width, 3


def run_ffmpeg(command: list[str]) -> bytes:
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    return completed.stdout


def decode_reference(video: Path, timestamp: float, width: int) -> np.ndarray:
    height, _, channels = frame_shape(width)
    raw = run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{timestamp:.6f}",
        "-i", str(video), "-frames:v", "1", "-vf", f"scale={width}:{height}",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ])
    expected = height * width * channels
    if len(raw) != expected:
        raise RuntimeError(f"Reference decode has {len(raw)} bytes; expected {expected}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, channels).astype(np.float32)


def background_mask(height: int, width: int) -> np.ndarray:
    """Use mostly stands and court edges; central player action is excluded."""
    mask = np.zeros((height, width), dtype=bool)
    mask[: int(height * 0.28), :] = True             # upper stands / roof
    mask[int(height * 0.28): int(height * 0.82), : int(width * 0.14)] = True
    mask[int(height * 0.28): int(height * 0.82), int(width * 0.86):] = True
    mask[int(height * 0.82):, :] = True              # foreground court shape
    return mask


def normalized_difference(frame: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    # Correct global exposure per RGB channel.  This keeps camera geometry and
    # background color while reducing false differences from broadcast exposure.
    frame_shift = frame[mask].mean(axis=0)
    ref_shift = reference[mask].mean(axis=0)
    adjusted = np.clip(frame - frame_shift + ref_shift, 0, 255)
    return float(np.abs(adjusted[mask] - reference[mask]).mean() / 255.0)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH")
    if not 0 < args.difference_threshold < 1:
        raise ValueError("difference threshold must be in (0, 1)")

    height, width, channels = frame_shape(args.width)
    bytes_per_frame = height * width * channels
    reference = decode_reference(args.input, args.reference_sec, width)
    mask = background_mask(height, width)
    vf = f"fps={args.sample_fps},scale={width}:{height}"
    process = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(args.input),
        "-an", "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    samples: list[dict[str, object]] = []
    sample_index = 0
    assert process.stdout is not None
    while True:
        raw = process.stdout.read(bytes_per_frame)
        if not raw:
            break
        if len(raw) != bytes_per_frame:
            raise RuntimeError("Incomplete raw frame from ffmpeg")
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, channels).astype(np.float32)
        difference = normalized_difference(frame, reference, mask)
        timestamp = sample_index / args.sample_fps
        samples.append({
            "video_id": args.video_id,
            "timestamp_sec": f"{timestamp:.6f}",
            "background_difference": f"{difference:.6f}",
            "proposed_view": "main_camera" if difference <= args.difference_threshold else "other_camera",
        })
        sample_index += 1
    stderr = process.stderr.read() if process.stderr else b""
    if process.wait() != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))

    project = Path(__file__).resolve().parents[1]
    segment_dir = project / "data" / "segments"
    write_csv(segment_dir / f"{args.video_id}_fixed_view_scores.csv", [
        "video_id", "timestamp_sec", "background_difference", "proposed_view",
    ], samples)

    # Adjacent matching samples form a candidate main-camera interval.  The
    # fixed margin removes the cross-fade / residual frames at its boundaries.
    main_segments: list[dict[str, object]] = []
    run_start: float | None = None
    run_scores: list[float] = []
    step = 1.0 / args.sample_fps
    def finish_run(run_end: float) -> None:
        nonlocal run_start, run_scores
        if run_start is None:
            return
        start = run_start + args.edge_margin_sec
        end = min(args.duration_sec, run_end + step) - args.edge_margin_sec
        if end - start >= args.min_duration_sec:
            number = len(main_segments) + 1
            main_segments.append({
                "video_id": args.video_id,
                "segment_id": f"{args.video_id}_main_{number:03d}",
                "start_timestamp_sec": f"{start:.6f}",
                "end_timestamp_sec": f"{end:.6f}",
                "start_source_frame": round(start * 60000 / 1001),
                "end_source_frame": round(end * 60000 / 1001),
                "scene_type": "main_camera",
                "cut_confidence": "fixed_view_match",
                "main_camera_confidence": f"{1 - float(np.mean(run_scores)):.6f}",
                "usable_for_analysis": "true",
                "human_checked": "false",
                "notes": f"reference_sec={args.reference_sec}; threshold={args.difference_threshold}; edge_margin_sec={args.edge_margin_sec}",
            })
        run_start, run_scores = None, []

    for sample in samples:
        timestamp = float(sample["timestamp_sec"])
        if sample["proposed_view"] == "main_camera":
            if run_start is None:
                run_start = timestamp
            run_scores.append(float(sample["background_difference"]))
        else:
            finish_run(timestamp - step)
    finish_run(args.duration_sec - step)

    write_csv(segment_dir / "main_camera_segments.csv", [
        "video_id", "segment_id", "start_timestamp_sec", "end_timestamp_sec",
        "start_source_frame", "end_source_frame", "scene_type", "cut_confidence",
        "main_camera_confidence", "usable_for_analysis", "human_checked", "notes",
    ], main_segments)
    print(f"Scored {len(samples)} samples; wrote {len(main_segments)} main-camera intervals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
