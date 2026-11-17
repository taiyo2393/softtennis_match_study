"""Propose post-impact shot-direction candidates from BallTrack detections.

Only frames after the human-confirmed impact are used.  This produces a
post-impact outcome-label candidate, not an input to pre-impact front-motion
measurement.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
from pathlib import Path

import cv2
import numpy as np


FPS = 60000 / 1001
# The human review confirmed the orientation for match1: the target front
# player's left/right labels agree with image left/right.  Do not reuse this
# convention for a new camera without an orientation audit.
FRONT_VIEW_FROM_SCREEN = {"screen_left": "front_left", "screen_right": "front_right"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def candidate_id(event_id: str) -> str:
    suffix = "_event_001"
    if not event_id.endswith(suffix):
        raise ValueError(f"Cannot recover candidate id from {event_id}")
    return event_id[: -len(suffix)]


def split_runs(points: list[dict[str, float]], max_gap: int, max_step_per_frame: float) -> list[list[dict[str, float]]]:
    """Split sparse candidates where a temporal/spatial jump suggests an ID switch."""
    runs: list[list[dict[str, float]]] = []
    for point in points:
        if not runs:
            runs.append([point]); continue
        previous = runs[-1][-1]
        gap = point["local_frame"] - previous["local_frame"]
        distance = ((point["x"] - previous["x"]) ** 2 + (point["y"] - previous["y"]) ** 2) ** 0.5
        if gap <= max_gap and distance <= max_step_per_frame * gap:
            runs[-1].append(point)
        else:
            runs.append([point])
    return runs


def direction_summary(run: list[dict[str, float]]) -> dict[str, float]:
    """Use medians of first/last thirds rather than a single detected point."""
    if not run:
        return {"run_dx_px": 0.0, "run_dy_px": 0.0, "median_step_dx_px": 0.0, "sign_agreement": 0.0}
    edge = max(1, len(run) // 3)
    start_x = statistics.median(point["x"] for point in run[:edge])
    end_x = statistics.median(point["x"] for point in run[-edge:])
    start_y = statistics.median(point["y"] for point in run[:edge])
    end_y = statistics.median(point["y"] for point in run[-edge:])
    steps = []
    for before, after in zip(run, run[1:]):
        gap = after["local_frame"] - before["local_frame"]
        if gap:
            steps.append((after["x"] - before["x"]) / gap)
    dx = end_x - start_x
    median_step = statistics.median(steps) if steps else 0.0
    nonzero = [step for step in steps if abs(step) >= 0.2]
    agreement = 0.0 if not nonzero or dx == 0 else sum((step > 0) == (dx > 0) for step in nonzero) / len(nonzero)
    return {"run_dx_px": dx, "run_dy_px": end_y - start_y, "median_step_dx_px": median_step, "sign_agreement": agreement}


def make_proposal(run: list[dict[str, float]], summary: dict[str, float], min_points: int, min_dx: float) -> tuple[str, str, str]:
    if len(run) < min_points:
        return "", "needs_human", f"coherent run has only {len(run)} points (< {min_points})"
    dx = summary["run_dx_px"]
    if abs(dx) < min_dx:
        return "", "needs_human", f"coherent x displacement {dx:.1f}px is below {min_dx:.1f}px"
    direction = "screen_right" if dx > 0 else "screen_left"
    if len(run) >= 4 and summary["sign_agreement"] >= 0.75:
        return direction, "high_confidence", "long coherent post-impact run"
    return direction, "low_confidence", "short or internally variable post-impact run"


def write_preview(project: Path, clip: Path, event: dict[str, str], impact_local: int, all_points: list[dict[str, float]], selected: list[dict[str, float]], direction: str, status: str, pre_frames: int, post_frames: int) -> Path:
    output_dir = project / "data" / "balltrack" / "direction_proposals" / event["video_id"] / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f"{event['event_id']}.avi"
    output = output_dir / f"{event['event_id']}.mp4"
    temporary.unlink(missing_ok=True)
    capture = cv2.VideoCapture(str(clip))
    fps = capture.get(cv2.CAP_PROP_FPS); width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)); count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start, end = max(0, impact_local - pre_frames), min(count - 1, impact_local + post_frames)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    selected_frames = {int(point["local_frame"]) for point in selected}
    points_by_frame = {int(point["local_frame"]): point for point in all_points}
    history: list[tuple[int, int]] = []
    for local_frame in range(start, end + 1):
        ok, frame = capture.read()
        if not ok:
            break
        point = points_by_frame.get(local_frame)
        if point:
            color = (0, 220, 0) if local_frame in selected_frames else (160, 160, 160)
            xy = (round(point["x"]), round(point["y"]))
            cv2.circle(frame, xy, 11, color, 2, cv2.LINE_AA)
            if local_frame in selected_frames:
                history.append(xy)
        if len(history) > 1:
            cv2.polylines(frame, [np.array(history)], False, (0, 220, 0), 3, cv2.LINE_AA)
        relative = (local_frame - impact_local) / fps
        title = f"impact {relative:+.3f}s / AI {direction or 'no direction'} ({status})"
        cv2.putText(frame, title, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "green: selected BallTrack run / gray: other candidates", (20, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    capture.release(); writer.release()
    duration = (end - start + 1) / fps
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(temporary),
        "-ss", f"{start / fps:.6f}", "-t", f"{duration:.6f}", "-i", str(clip),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-movflags", "+faststart", str(output),
    ], check=True)
    temporary.unlink(missing_ok=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--post-start-sec", type=float, default=0.017)
    parser.add_argument("--post-end-sec", type=float, default=0.45)
    parser.add_argument("--max-gap-frames", type=int, default=3)
    parser.add_argument("--max-step-px-per-frame", type=float, default=70.0)
    parser.add_argument("--min-run-points", type=int, default=3)
    parser.add_argument("--min-direction-dx-px", type=float, default=12.0)
    args = parser.parse_args()
    if args.post_start_sec < 0 or args.post_end_sec <= args.post_start_sec:
        raise SystemExit("Post-impact window must satisfy 0 <= start < end")
    project = Path(__file__).resolve().parents[1]
    event_by_id = {row["event_id"]: row for row in read(project / "annotations" / "motion_prediction_events.csv") if row["video_id"] == args.video_id and row["inclusion_status"] == "included"}
    usable = [row["event_id"] for row in read(project / "annotations" / f"highres_front_track_review_{args.video_id}_yolo11m.csv") if row["tracking_status"] == "usable" and row["event_id"] in event_by_id]
    queue = {row["candidate_id"]: row for row in read(project / "data" / "segments" / f"{args.video_id}_motion_prediction_pilot_queue.csv")}
    results: list[dict[str, object]] = []
    for event_id in usable:
        event = event_by_id[event_id]; candidate = candidate_id(event_id)
        clip = project / "data" / "segments" / "motion_prediction_pilot" / args.video_id / f"{candidate}.mp4"
        prediction_path = project / "data" / "balltrack" / "racketvision" / args.video_id / candidate / "balltrack_predictions.csv"
        if not clip.is_file() or not prediction_path.is_file() or candidate not in queue:
            raise SystemExit(f"Missing clip, queue row, or BallTrack output for {event_id}")
        capture = cv2.VideoCapture(str(clip)); clip_fps = capture.get(cv2.CAP_PROP_FPS); capture.release()
        if clip_fps <= 0:
            raise SystemExit(f"Cannot read FPS: {clip}")
        impact_local = round((float(event["impact_timestamp_sec"]) - float(queue[candidate]["source_start_timestamp_sec"])) * clip_fps)
        lower = impact_local + round(args.post_start_sec * clip_fps); upper = impact_local + round(args.post_end_sec * clip_fps)
        points = [
            {"local_frame": float(row["local_frame"]), "x": float(row["x_px"]), "y": float(row["y_px"]), "confidence": float(row["confidence"])}
            for row in read(prediction_path)
            if row["visible"] == "1" and lower <= int(row["local_frame"]) <= upper
        ]
        runs = split_runs(points, args.max_gap_frames, args.max_step_px_per_frame)
        selected = max(runs, key=lambda run: (len(run), abs(direction_summary(run)["run_dx_px"])), default=[])
        summary = direction_summary(selected)
        direction, status, reason = make_proposal(selected, summary, args.min_run_points, args.min_direction_dx_px)
        front_direction = FRONT_VIEW_FROM_SCREEN.get(direction, "")
        preview = write_preview(project, clip, event, impact_local, points, selected, direction, status, round(.15 * clip_fps), round(.50 * clip_fps))
        results.append({
            "event_id": event_id, "video_id": args.video_id, "candidate_id": candidate,
            "impact_source_frame": event["impact_source_frame"], "impact_local_frame": impact_local,
            "post_start_local_frame": lower, "post_end_local_frame": upper,
            "visible_points": len(points), "selected_run_points": len(selected),
            "run_start_local_frame": "" if not selected else int(selected[0]["local_frame"]), "run_end_local_frame": "" if not selected else int(selected[-1]["local_frame"]),
            "run_dx_px": f"{summary['run_dx_px']:.3f}", "run_dy_px": f"{summary['run_dy_px']:.3f}", "median_step_dx_px": f"{summary['median_step_dx_px']:.3f}", "dx_sign_agreement": f"{summary['sign_agreement']:.3f}",
            "median_confidence": "" if not selected else f"{statistics.median(point['confidence'] for point in selected):.3f}",
            "ai_screen_direction": direction, "ai_front_view_proposal": front_direction,
            "proposal_status": status, "proposal_reason": reason, "preview_path": str(preview),
        })
        print(f"{event_id}: {direction or 'no direction'} / {status} ({len(selected)} coherent points)")
    output_dir = project / "data" / "balltrack" / "direction_proposals"; output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.video_id}_shot_direction_candidates.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
