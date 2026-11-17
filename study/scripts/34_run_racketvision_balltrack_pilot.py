"""Run RacketVision BallTrack on one local fixed-camera pilot clip.

This is an AI candidate generator.  It never confirms a ball contact or an
impact; it writes per-frame ball candidates with confidence for later review.
The third-party RacketVision checkout and its Python environment stay isolated
under ``third_party/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def extract_frames(video: Path, frame_dir: Path) -> tuple[float, int, int, int]:
    """Extract each source frame once; return fps, count, width, height."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Cannot read video metadata: {video}")
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        path = frame_dir / f"{index:06d}.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write frame: {path}")
        index += 1
    capture.release()
    if index == 0:
        raise RuntimeError(f"No frames extracted: {video}")
    return fps, index, width, height


def median_background(frame_paths: list[Path], output: Path, sample_count: int = 120) -> int:
    """Build the model-required static-scene median from uniformly sampled frames."""
    effective_count = len(frame_paths) if sample_count == 0 else min(sample_count, len(frame_paths))
    if effective_count < 1:
        raise ValueError("sample_count must be positive, or 0 to use all clip frames")
    indices = np.linspace(0, len(frame_paths) - 1, effective_count, dtype=int)
    sampled = []
    for index in indices:
        frame = cv2.imread(str(frame_paths[index]))
        if frame is None:
            raise RuntimeError(f"Cannot read extracted frame: {frame_paths[index]}")
        sampled.append(frame)
    np.savez_compressed(output, median=np.median(np.stack(sampled, axis=0), axis=0).astype(np.uint8))
    return effective_count


def write_background_reference(frame_paths: list[Path], output: Path, mode: str, sample_count: int) -> int:
    """Write the required background array and return its effective sample count.

    ``first_frame`` is a diagnostic static-reference condition only.  The
    released model was trained with a median background, so it is not a fair
    architecture-level 'no background' baseline.
    """
    if mode == "median":
        return median_background(frame_paths, output, sample_count)
    if mode == "first_frame":
        frame = cv2.imread(str(frame_paths[0]))
        if frame is None:
            raise RuntimeError(f"Cannot read extracted frame: {frame_paths[0]}")
        np.savez_compressed(output, median=frame)
        return 1
    raise ValueError(f"Unsupported background mode: {mode}")


def validate_run_name(name: str) -> str:
    """Keep a run name inside the candidate directory."""
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in name):
        raise ValueError("--run-name may use only letters, digits, '_' and '-'")
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--queue-path", type=Path, default=None,
        help="CSV queue containing candidate_id and source_start_timestamp_sec. Defaults to the original pilot queue.",
    )
    parser.add_argument(
        "--clip-dir", type=Path, default=None,
        help="Directory containing <candidate_id>.mp4. Defaults to the original pilot clip directory.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--batchsize", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--median-samples", type=int, default=120,
        help="Uniformly sampled frames for the static median background; 0 uses every clip frame.",
    )
    parser.add_argument(
        "--background-mode", choices=("median", "first_frame"), default="median",
        help="Static reference supplied to the released model. first_frame is diagnostic only.",
    )
    parser.add_argument(
        "--run-name", default="default",
        help="Name for an ablation run. 'default' preserves the original output location.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    queue_path = args.queue_path or (project / "data" / "segments" / f"{args.video_id}_motion_prediction_pilot_queue.csv")
    queue_path = queue_path.resolve()
    rows = {row["candidate_id"]: row for row in read_csv(queue_path)}
    if args.candidate_id not in rows:
        raise SystemExit(f"Candidate not in queue: {args.candidate_id}")
    queue = rows[args.candidate_id]
    clip_dir = args.clip_dir or (project / "data" / "segments" / "motion_prediction_pilot" / args.video_id)
    clip = clip_dir.resolve() / f"{args.candidate_id}.mp4"
    if not clip.is_file():
        raise SystemExit(f"Pilot clip not found: {clip}")

    third_party = project / "third_party" / "RacketVision"
    balltrack = third_party / "source" / "BallTrack"
    checkpoint = balltrack / "checkpoints" / "balltrack_best.pth"
    if not checkpoint.is_file():
        raise SystemExit(f"RacketVision BallTrack checkpoint not found: {checkpoint}")
    sys.path.insert(0, str(balltrack))
    from inference import BallInferencer  # noqa: PLC0415

    run_name = validate_run_name(args.run_name)
    candidate_dir = project / "data" / "balltrack" / "racketvision" / args.video_id / args.candidate_id
    output_dir = candidate_dir if run_name == "default" else candidate_dir / "runs" / run_name
    frame_dir = candidate_dir / "frames"
    median_path = output_dir / "median.npz"
    prediction_path = output_dir / "balltrack_predictions.csv"
    metadata_path = output_dir / "run_metadata.json"
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = sorted(frame_dir.glob("*.jpg"))
    if existing_frames:
        capture = cv2.VideoCapture(str(clip))
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if len(existing_frames) != count:
            raise RuntimeError("Cached frames do not match the clip. Re-run default with --overwrite after removing only this candidate output.")
    else:
        fps, count, width, height = extract_frames(clip, frame_dir)
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if len(frame_paths) != count:
        raise RuntimeError("Extracted frame count does not match the frame file count")
    median_count = write_background_reference(frame_paths, median_path, args.background_mode, args.median_samples)

    inferencer = BallInferencer(
        cfg_path=str(balltrack / "configs" / "tracknetv3_base.py"),
        ckpt_path=str(checkpoint),
        device=args.device,
        thre=args.threshold,
        batchsize=args.batchsize,
    )
    predictions = inferencer([str(path) for path in frame_paths], str(median_path))
    clip_start = float(queue["source_start_timestamp_sec"])
    fields = [
        "candidate_id", "local_frame", "source_frame", "source_timestamp_sec",
        "x_px", "y_px", "visible", "confidence", "model_name", "threshold", "background_mode", "median_samples",
    ]
    with prediction_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for prediction in predictions:
            local_frame = int(prediction["Frame"])
            source_timestamp = clip_start + local_frame / fps
            writer.writerow({
                "candidate_id": args.candidate_id,
                "local_frame": local_frame,
                "source_frame": round(source_timestamp * fps),
                "source_timestamp_sec": f"{source_timestamp:.6f}",
                "x_px": prediction["X"], "y_px": prediction["Y"],
                "visible": prediction["Visibility"], "confidence": prediction["Confidence"],
                "model_name": "RacketVision_MS-TrackNetV3_background_median_seq4",
                "threshold": args.threshold, "background_mode": args.background_mode, "median_samples": median_count,
            })
    metadata_path.write_text(json.dumps({
        "candidate_id": args.candidate_id, "clip": str(clip), "clip_start_timestamp_sec": clip_start,
        "fps": fps, "frame_count": count, "width": width, "height": height,
        "device_requested": args.device, "device_used": str(inferencer.device),
        "checkpoint": str(checkpoint), "threshold": args.threshold,
        "run_name": run_name, "background_mode": args.background_mode,
        "median_samples_requested": args.median_samples,
        "median_samples_effective": median_count,
        "purpose": "AI ball-candidate generation only; not impact ground truth",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    visible = sum(int(row["Visibility"]) for row in predictions)
    print(f"Wrote {len(predictions)} rows ({visible} visible) to {prediction_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
