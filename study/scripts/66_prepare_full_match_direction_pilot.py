"""Prepare a frozen pilot batch for AI-assisted post-impact direction labels.

The input is the human-confirmed full-match rear-impact table.  Every saved
impact is retained in an audit registry, but only impacts whose complete
short review window falls inside both an automatically screened fixed-main-
camera interval and a reviewed play window are eligible for this pilot.

This script does not infer or confirm a direction.  It creates short 1080p
clips for a reproducible random pilot so that BallTrack direction proposals
can be evaluated before any 295-event bulk run is attempted.
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
from pathlib import Path

import cv2


REGISTRY_FIELDS = [
    "impact_id", "video_id", "impact_source_frame", "impact_timestamp_sec",
    "ready_segment_id", "play_window_id", "serve_context", "end_status",
    "review_start_source_frame", "review_end_source_frame", "review_start_timestamp_sec",
    "review_end_timestamp_sec", "auto_screen_status", "auto_screen_reason",
]
QUEUE_FIELDS = [
    "candidate_id", "pilot_id", "video_id", "impact_id", "impact_source_frame",
    "impact_timestamp_sec", "source_start_frame", "source_end_frame",
    "source_start_timestamp_sec", "source_end_timestamp_sec", "duration_sec",
    "impact_local_frame", "ready_segment_id", "play_window_id", "selection_seed",
    "selection_rank", "screening_status", "screening_note",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def containing_interval(rows: list[dict[str, str]], start: float, end: float, start_key: str, end_key: str) -> dict[str, str] | None:
    """Return an interval containing the complete review window, if any."""
    return next(
        (
            row for row in rows
            if float(row[start_key]) <= start and end <= float(row[end_key])
        ),
        None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original complete-match video")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--pilot-id", default="direction_pilot_01")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--pre-sec", type=float, default=0.20)
    parser.add_argument("--post-sec", type=float, default=0.60)
    parser.add_argument(
        "--include-previously-reviewed", action="store_true",
        help="Allow impacts already present in a front-view direction-review CSV. Normally they are excluded.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or args.pre_sec < 0 or args.post_sec <= 0:
        raise SystemExit("--limit must be positive; --pre-sec must be >= 0; --post-sec must be > 0")
    if not args.pilot_id.replace("_", "").isalnum():
        raise SystemExit("--pilot-id may use letters, digits, and underscores only")
    source_path = args.input.resolve()
    if not source_path.is_file():
        raise SystemExit(f"Input video not found: {source_path}")

    capture = cv2.VideoCapture(str(source_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frame_count <= 0:
        raise SystemExit(f"Cannot read video metadata: {source_path}")

    project = Path(__file__).resolve().parents[1]
    impact_path = project / "annotations" / f"{args.video_id}_full_match_rear_impact_review.csv"
    ready_path = project / "data" / "segments" / "analysis_ready_main_camera_segments.csv"
    play_path = project / "data" / "segments" / f"{args.video_id}_reviewed_play_windows.csv"
    impacts = [
        row for row in read_csv(impact_path)
        if row.get("video_id") == args.video_id
        and row.get("annotation_status") == "human_confirmed_rear_impact"
    ]
    ready = [row for row in read_csv(ready_path) if row.get("video_id") == args.video_id]
    plays = [row for row in read_csv(play_path) if row.get("video_id") == args.video_id]
    if not impacts:
        raise SystemExit("No human-confirmed rear-impact rows found.")

    pre_frames = round(args.pre_sec * fps)
    post_frames = round(args.post_sec * fps)
    reviewed_impact_ids: set[str] = set()
    if not args.include_previously_reviewed:
        for review_path in (project / "annotations").glob(
            f"{args.video_id}_full_match_*_front_direction_blind_review.csv"
        ):
            reviewed_impact_ids.update(
                row.get("impact_id", "") for row in read_csv(review_path) if row.get("impact_id")
            )

    registry: list[dict[str, object]] = []
    eligible: list[dict[str, object]] = []
    for impact in sorted(impacts, key=lambda row: int(row["impact_source_frame"])):
        impact_frame = int(impact["impact_source_frame"])
        start_frame = max(0, impact_frame - pre_frames)
        end_frame = min(frame_count - 1, impact_frame + post_frames)
        start_sec, end_sec = start_frame / fps, end_frame / fps
        ready_hit = containing_interval(ready, start_sec, end_sec, "start_timestamp_sec", "end_timestamp_sec")
        play_hit = containing_interval(plays, start_sec, end_sec, "source_start_timestamp_sec", "source_end_timestamp_sec")
        if ready_hit and play_hit:
            status, reason = "auto_direction_candidate", "complete review window in fixed-main-camera and reviewed play intervals"
        elif not ready_hit and not play_hit:
            status, reason = "outside_auto_screen", "review window is outside both automatic fixed-camera and reviewed-play intervals"
        elif not ready_hit:
            status, reason = "outside_auto_screen", "review window is outside automatic fixed-main-camera interval"
        else:
            status, reason = "outside_auto_screen", "review window is outside reviewed-play interval"
        row = {
            "impact_id": impact["impact_id"], "video_id": args.video_id,
            "impact_source_frame": impact_frame, "impact_timestamp_sec": impact["impact_timestamp_sec"],
            "ready_segment_id": ready_hit["segment_id"] if ready_hit else "",
            "play_window_id": play_hit["play_window_id"] if play_hit else "",
            "serve_context": play_hit["serve_context"] if play_hit else "",
            "end_status": play_hit["end_status"] if play_hit else "",
            "review_start_source_frame": start_frame, "review_end_source_frame": end_frame,
            "review_start_timestamp_sec": f"{start_sec:.6f}", "review_end_timestamp_sec": f"{end_sec:.6f}",
            "auto_screen_status": status, "auto_screen_reason": reason,
        }
        registry.append(row)
        if status == "auto_direction_candidate" and row["impact_id"] not in reviewed_impact_ids:
            eligible.append(row)

    registry_path = project / "analysis" / f"{args.video_id}_full_match_impact_registry.csv"
    write_csv(registry_path, REGISTRY_FIELDS, registry)
    if len(eligible) < args.limit:
        raise SystemExit(f"Only {len(eligible)} auto-screened eligible impacts; requested {args.limit}.")

    queue_path = project / "data" / "segments" / f"{args.video_id}_full_match_{args.pilot_id}_queue.csv"
    clip_dir = project / "data" / "segments" / f"full_match_{args.pilot_id}" / args.video_id
    if queue_path.exists() and not args.overwrite:
        raise SystemExit(f"Pilot queue already exists: {queue_path}. Use another --pilot-id; do not overwrite reviewed work.")
    clip_dir.mkdir(parents=True, exist_ok=True)
    chosen = random.Random(args.seed).sample(eligible, args.limit)
    chosen.sort(key=lambda row: int(row["impact_source_frame"]))
    queue: list[dict[str, object]] = []
    for rank, row in enumerate(chosen, start=1):
        start_frame = int(row["review_start_source_frame"])
        end_frame = int(row["review_end_source_frame"])
        candidate_id = f"{args.video_id}_{args.pilot_id}_{rank:03d}"
        queue_row = {
            "candidate_id": candidate_id, "pilot_id": args.pilot_id, "video_id": args.video_id,
            "impact_id": row["impact_id"], "impact_source_frame": row["impact_source_frame"],
            "impact_timestamp_sec": row["impact_timestamp_sec"], "source_start_frame": start_frame,
            "source_end_frame": end_frame, "source_start_timestamp_sec": row["review_start_timestamp_sec"],
            "source_end_timestamp_sec": row["review_end_timestamp_sec"],
            "duration_sec": f"{(end_frame - start_frame + 1) / fps:.6f}",
            "impact_local_frame": int(row["impact_source_frame"]) - start_frame,
            "ready_segment_id": row["ready_segment_id"], "play_window_id": row["play_window_id"],
            "selection_seed": args.seed, "selection_rank": rank,
            "screening_status": "unreviewed", "screening_note": "",
        }
        queue.append(queue_row)
        clip = clip_dir / f"{candidate_id}.mp4"
        if not clip.is_file() or args.overwrite:
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", queue_row["source_start_timestamp_sec"], "-i", str(source_path),
                "-t", queue_row["duration_sec"], "-map", "0:v:0", "-map", "0:a?",
                "-vf", "scale=1920:-2", "-r", "60000/1001",
                "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-movflags", "+faststart", str(clip),
            ], check=True)
    write_csv(queue_path, QUEUE_FIELDS, queue)
    print(f"Saved complete impact registry ({len(registry)} rows): {registry_path}")
    print(
        f"Auto-screen eligible after excluding {len(reviewed_impact_ids)} previously reviewed impacts: "
        f"{len(eligible)}; frozen pilot selected: {len(queue)}"
    )
    print(f"Pilot queue: {queue_path}")
    print(f"Pilot clips: {clip_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
