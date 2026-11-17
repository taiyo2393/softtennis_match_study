"""Generate screen-left/right proposals from post-impact BallTrack trajectories.

This script is intentionally limited to the frozen pilot created by script 66.
It records a post-impact *outcome* candidate in camera coordinates.  It does
not infer a player-facing left/right label, does not decide a target front
player, and does not confirm any direction.  A human reviewer must confirm
the direction before the candidate generator can be evaluated or scaled.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
from pathlib import Path

import cv2
import numpy as np


FIELDS = [
    "candidate_id", "pilot_id", "video_id", "impact_id", "impact_source_frame",
    "impact_timestamp_sec", "impact_local_frame", "ready_segment_id", "play_window_id",
    "post_start_local_frame", "post_end_local_frame", "visible_points",
    "selected_run_points", "run_start_local_frame", "run_end_local_frame",
    "run_dx_px", "run_dy_px", "median_step_dx_px", "dx_sign_agreement",
    "median_confidence", "ai_screen_direction", "proposal_status", "proposal_reason",
    "preview_path",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def split_runs(points: list[dict[str, float]], max_gap: int, max_step_per_frame: float) -> list[list[dict[str, float]]]:
    runs: list[list[dict[str, float]]] = []
    for point in points:
        if not runs:
            runs.append([point])
            continue
        previous = runs[-1][-1]
        gap = point["local_frame"] - previous["local_frame"]
        distance = ((point["x"] - previous["x"]) ** 2 + (point["y"] - previous["y"]) ** 2) ** 0.5
        if gap <= max_gap and distance <= max_step_per_frame * gap:
            runs[-1].append(point)
        else:
            runs.append([point])
    return runs


def summarize(run: list[dict[str, float]]) -> dict[str, float]:
    if not run:
        return {"run_dx_px": 0.0, "run_dy_px": 0.0, "median_step_dx_px": 0.0, "sign_agreement": 0.0}
    edge = max(1, len(run) // 3)
    start_x = statistics.median(point["x"] for point in run[:edge])
    end_x = statistics.median(point["x"] for point in run[-edge:])
    start_y = statistics.median(point["y"] for point in run[:edge])
    end_y = statistics.median(point["y"] for point in run[-edge:])
    steps = [
        (after["x"] - before["x"]) / (after["local_frame"] - before["local_frame"])
        for before, after in zip(run, run[1:])
        if after["local_frame"] != before["local_frame"]
    ]
    dx = end_x - start_x
    nonzero = [step for step in steps if abs(step) >= 0.2]
    agreement = 0.0 if not nonzero or dx == 0 else sum((step > 0) == (dx > 0) for step in nonzero) / len(nonzero)
    return {
        "run_dx_px": dx,
        "run_dy_px": end_y - start_y,
        "median_step_dx_px": statistics.median(steps) if steps else 0.0,
        "sign_agreement": agreement,
    }


def make_proposal(run: list[dict[str, float]], stats: dict[str, float], min_points: int, min_dx: float) -> tuple[str, str, str]:
    if len(run) < min_points:
        return "", "needs_human", f"coherent run has only {len(run)} points (< {min_points})"
    if abs(stats["run_dx_px"]) < min_dx:
        return "", "needs_human", f"coherent x displacement {stats['run_dx_px']:.1f}px is below {min_dx:.1f}px"
    direction = "screen_right" if stats["run_dx_px"] > 0 else "screen_left"
    if len(run) >= 4 and stats["sign_agreement"] >= 0.75:
        return direction, "high_confidence", "long coherent post-impact run"
    return direction, "low_confidence", "short or internally variable post-impact run"


def write_preview(clip: Path, output: Path, impact_local: int, points: list[dict[str, float]], selected: list[dict[str, float]], direction: str, status: str) -> None:
    """Render an audit video; green is selected, gray is unselected BallTrack."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".avi")
    temporary.unlink(missing_ok=True)
    capture = cv2.VideoCapture(str(clip))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create preview: {temporary}")
    all_by_frame = {round(point["local_frame"]): point for point in points}
    selected_frames = {round(point["local_frame"]) for point in selected}
    history: list[tuple[int, int]] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        point = all_by_frame.get(frame_index)
        if point:
            color = (0, 220, 0) if frame_index in selected_frames else (155, 155, 155)
            location = (round(point["x"]), round(point["y"]))
            cv2.circle(frame, location, 10, color, 2, cv2.LINE_AA)
            if frame_index in selected_frames:
                history.append(location)
        if len(history) > 1:
            cv2.polylines(frame, [np.asarray(history)], False, (0, 220, 0), 3, cv2.LINE_AA)
        relative = (frame_index - impact_local) / fps
        cv2.putText(frame, f"impact {relative:+.3f}s / AI {direction or 'no direction'} ({status})", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "green: selected run / gray: other BallTrack candidates", (24, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        frame_index += 1
    capture.release()
    writer.release()
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(temporary), "-i", str(clip),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-movflags", "+faststart", str(output),
    ], check=True)
    temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--pilot-id", default="direction_pilot_01")
    parser.add_argument("--post-start-sec", type=float, default=0.017)
    parser.add_argument("--post-end-sec", type=float, default=0.45)
    parser.add_argument("--max-gap-frames", type=int, default=3)
    parser.add_argument("--max-step-px-per-frame", type=float, default=70.0)
    parser.add_argument("--min-run-points", type=int, default=3)
    parser.add_argument("--min-direction-dx-px", type=float, default=12.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.post_start_sec < 0 or args.post_end_sec <= args.post_start_sec:
        raise SystemExit("Post-impact window must satisfy 0 <= start < end")
    project = Path(__file__).resolve().parents[1]
    queue_path = project / "data" / "segments" / f"{args.video_id}_full_match_{args.pilot_id}_queue.csv"
    clip_dir = project / "data" / "segments" / f"full_match_{args.pilot_id}" / args.video_id
    output_dir = project / "data" / "balltrack" / "direction_proposals" / f"full_match_{args.pilot_id}" / args.video_id
    output_path = output_dir / "screen_direction_candidates.csv"
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Candidate table exists: {output_path}. Use --overwrite only before human review begins.")
    queue = read_csv(queue_path)
    results: list[dict[str, object]] = []
    for position, event in enumerate(queue, start=1):
        candidate_id = event["candidate_id"]
        clip = clip_dir / f"{candidate_id}.mp4"
        prediction_path = project / "data" / "balltrack" / "racketvision" / args.video_id / candidate_id / "balltrack_predictions.csv"
        if not clip.is_file() or not prediction_path.is_file():
            raise SystemExit(f"Missing clip or BallTrack output for {candidate_id}")
        capture = cv2.VideoCapture(str(clip))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if fps <= 0:
            raise SystemExit(f"Cannot read clip FPS: {clip}")
        impact_local = int(event["impact_local_frame"])
        lower = impact_local + round(args.post_start_sec * fps)
        upper = min(frame_count - 1, impact_local + round(args.post_end_sec * fps))
        points = [
            {"local_frame": float(row["local_frame"]), "x": float(row["x_px"]), "y": float(row["y_px"]), "confidence": float(row["confidence"])}
            for row in read_csv(prediction_path)
            if row["visible"] == "1" and lower <= int(row["local_frame"]) <= upper
        ]
        runs = split_runs(points, args.max_gap_frames, args.max_step_px_per_frame)
        selected = max(runs, key=lambda run: (len(run), abs(summarize(run)["run_dx_px"])), default=[])
        stats = summarize(selected)
        direction, status, reason = make_proposal(selected, stats, args.min_run_points, args.min_direction_dx_px)
        preview = output_dir / "previews" / f"{candidate_id}.mp4"
        write_preview(clip, preview, impact_local, points, selected, direction, status)
        results.append({
            "candidate_id": candidate_id, "pilot_id": args.pilot_id, "video_id": args.video_id,
            "impact_id": event["impact_id"], "impact_source_frame": event["impact_source_frame"],
            "impact_timestamp_sec": event["impact_timestamp_sec"], "impact_local_frame": impact_local,
            "ready_segment_id": event["ready_segment_id"], "play_window_id": event["play_window_id"],
            "post_start_local_frame": lower, "post_end_local_frame": upper,
            "visible_points": len(points), "selected_run_points": len(selected),
            "run_start_local_frame": "" if not selected else round(selected[0]["local_frame"]),
            "run_end_local_frame": "" if not selected else round(selected[-1]["local_frame"]),
            "run_dx_px": f"{stats['run_dx_px']:.3f}", "run_dy_px": f"{stats['run_dy_px']:.3f}",
            "median_step_dx_px": f"{stats['median_step_dx_px']:.3f}", "dx_sign_agreement": f"{stats['sign_agreement']:.3f}",
            "median_confidence": "" if not selected else f"{statistics.median(point['confidence'] for point in selected):.3f}",
            "ai_screen_direction": direction, "proposal_status": status, "proposal_reason": reason,
            "preview_path": str(preview),
        })
        print(f"[{position}/{len(queue)}] {candidate_id}: {direction or 'no direction'} / {status}")
    write_csv(output_path, results)
    print(f"Wrote {len(results)} candidate rows: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
