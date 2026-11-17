"""Propose, but never confirm, possible ball-contact times from BallTrack paths.

Candidates are local reversals in the image vertical coordinate.  They include
player contacts, bounces, and occasional tracking artifacts; the purpose is to
reduce review time, not to create impact labels automatically.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def interpolate_short_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill only gaps bracketed by observations within ``max_gap``."""
    result = values.astype(float).copy()
    observed = np.flatnonzero(np.isfinite(result))
    for left, right in zip(observed[:-1], observed[1:]):
        gap = right - left - 1
        if 0 < gap <= max_gap:
            result[left + 1:right] = np.linspace(result[left], result[right], gap + 2)[1:-1]
    return result


def candidates_for_clip(rows: list[dict[str, str]], look: int, max_gap: int, min_speed: float, min_spacing: int) -> list[dict[str, object]]:
    frame_count = max(int(row["local_frame"]) for row in rows) + 1
    y = np.full(frame_count, np.nan)
    x = np.full(frame_count, np.nan)
    confidence = np.zeros(frame_count)
    for row in rows:
        frame = int(row["local_frame"])
        if int(row["visible"]):
            x[frame] = float(row["x_px"])
            y[frame] = float(row["y_px"])
            confidence[frame] = float(row["confidence"])
    y_fill = interpolate_short_gaps(y, max_gap)
    x_fill = interpolate_short_gaps(x, max_gap)
    raw: list[dict[str, object]] = []
    for frame in range(look, frame_count - look):
        before = y_fill[frame - look:frame + 1]
        after = y_fill[frame:frame + look + 1]
        if not (np.isfinite(before).all() and np.isfinite(after).all()):
            continue
        vy_before = float(np.median(np.diff(before)))
        vy_after = float(np.median(np.diff(after)))
        if vy_before * vy_after >= 0 or min(abs(vy_before), abs(vy_after)) < min_speed:
            continue
        observed = np.isfinite(y[max(0, frame - look):min(frame_count, frame + look + 1)])
        coverage = float(observed.mean())
        # Reversal strength is intentionally simple and does not distinguish a bounce from a hit.
        raw.append({
            "local_frame": frame,
            "x_px": float(x_fill[frame]),
            "y_px": float(y_fill[frame]),
            "vy_before_px_per_frame": vy_before,
            "vy_after_px_per_frame": vy_after,
            "candidate_score": min(abs(vy_before), abs(vy_after)) * coverage,
            "local_coverage": coverage,
            "mean_confidence": float(confidence[max(0, frame - look):frame + look + 1][observed].mean()) if observed.any() else 0.0,
        })
    # Non-maximum suppression: retain the strongest proposal in a short time neighborhood.
    selected: list[dict[str, object]] = []
    for row in sorted(raw, key=lambda row: float(row["candidate_score"]), reverse=True):
        if all(abs(int(row["local_frame"]) - int(old["local_frame"])) > min_spacing for old in selected):
            selected.append(row)
    return sorted(selected, key=lambda row: int(row["local_frame"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument(
        "--queue-path", type=Path, default=None,
        help="CSV queue to inspect. Defaults to the original pilot queue.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Where to write unverified proposals. Defaults to the original pilot proposal CSV.",
    )
    parser.add_argument("--run-name", default="threshold030_median120")
    parser.add_argument("--look-frames", type=int, default=6)
    parser.add_argument("--max-gap-frames", type=int, default=6)
    parser.add_argument("--min-speed-px-per-frame", type=float, default=2.0)
    parser.add_argument("--min-spacing-frames", type=int, default=12)
    args = parser.parse_args()
    if args.look_frames < 2 or args.max_gap_frames < 0 or args.min_spacing_frames < 1:
        raise SystemExit("Invalid temporal parameters")

    project = Path(__file__).resolve().parents[1]
    queue_path = args.queue_path or (project / "data" / "segments" / f"{args.video_id}_motion_prediction_pilot_queue.csv")
    queue_path = queue_path.resolve()
    output = args.output or (project / "analysis" / f"{args.video_id}_motion_prediction_impact_candidates.csv")
    output = output.resolve()
    if not queue_path.is_file():
        raise SystemExit(f"Queue not found: {queue_path}")
    fields = [
        "candidate_id", "run_name", "proposal_type", "local_frame", "source_frame", "source_timestamp_sec",
        "video_offset_sec", "x_px", "y_px", "vy_before_px_per_frame", "vy_after_px_per_frame",
        "candidate_score", "local_coverage", "mean_confidence", "human_decision", "human_note",
    ]
    results: list[dict[str, object]] = []
    for queue_row in read(queue_path):
        candidate_id = queue_row["candidate_id"]
        path = (project / "data" / "balltrack" / "racketvision" / args.video_id / candidate_id /
                "runs" / args.run_name / "balltrack_predictions.csv")
        if not path.is_file():
            print(f"Skip missing BallTrack run: {candidate_id}")
            continue
        predictions = read(path)
        fps = 60000 / 1001
        for row in candidates_for_clip(predictions, args.look_frames, args.max_gap_frames,
                                       args.min_speed_px_per_frame, args.min_spacing_frames):
            local_frame = int(row["local_frame"])
            results.append({
                "candidate_id": candidate_id, "run_name": args.run_name,
                "proposal_type": "ball_image_y_reversal_unverified", "local_frame": local_frame,
                "source_frame": next(item["source_frame"] for item in predictions if int(item["local_frame"]) == local_frame),
                "source_timestamp_sec": next(item["source_timestamp_sec"] for item in predictions if int(item["local_frame"]) == local_frame),
                "video_offset_sec": f"{local_frame / fps:.6f}",
                "x_px": f"{float(row['x_px']):.3f}", "y_px": f"{float(row['y_px']):.3f}",
                "vy_before_px_per_frame": f"{float(row['vy_before_px_per_frame']):.4f}",
                "vy_after_px_per_frame": f"{float(row['vy_after_px_per_frame']):.4f}",
                "candidate_score": f"{float(row['candidate_score']):.4f}",
                "local_coverage": f"{float(row['local_coverage']):.4f}",
                "mean_confidence": f"{float(row['mean_confidence']):.4f}",
                "human_decision": "unreviewed", "human_note": "",
            })
    results.sort(key=lambda row: (str(row["candidate_id"]), int(row["local_frame"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(results)
    print(f"Wrote {len(results)} unverified proposals: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
