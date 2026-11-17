"""Track one human-confirmed target front player at high resolution.

The older role-tracking pilot resized the whole court and assigned four player
roles from depth bands.  This script instead detects persons only in the
near-front court band, seeds one target at the human-confirmed impact frame,
and associates that single target independently forward and backward.  Missing
observations remain missing; no coordinates are interpolated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path

import cv2


FPS = 60000 / 1001
# Match1 fixed-camera proposals. They select a seed only and must be audited
# before use on a new match/camera. Far-side values deliberately do not reuse
# the near-side range: the far target is much smaller and nearer the net.
NEAR_SEED_FOOT_Y_RANGE = (0.56, 0.73)
NEAR_SEED_FOOT_X_RANGE = (0.00, 1.00)
NEAR_FRONT_ROI_Y_RANGE = (0.47, 0.80)
FAR_SEED_FOOT_Y_RANGE = (0.46, 0.59)
FAR_SEED_FOOT_X_RANGE = (0.25, 0.75)
FAR_FRONT_ROI_Y_RANGE = (0.42, 0.60)
COLORS = {"target": (0, 210, 255), "candidate": (110, 110, 110)}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def person_detections(model: object, frame: object, width: int, height: int, confidence: float, imgsz: int, roi_y_range: tuple[float, float]) -> list[dict[str, float]]:
    """Detect people in a high-resolution front-court crop and restore coordinates."""
    y0, y1 = round(height * roi_y_range[0]), round(height * roi_y_range[1])
    crop = frame[y0:y1, :]
    result = model.predict(crop, classes=[0], conf=confidence, imgsz=imgsz, verbose=False)[0]
    found: list[dict[str, float]] = []
    for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
        x1, top, x2, bottom = [float(value) for value in box]
        bottom += y0; top += y0
        found.append({
            "x1": x1, "y1": top, "x2": x2, "y2": bottom,
            "cx": (x1 + x2) / 2, "cy": (top + bottom) / 2,
            "foot_x": (x1 + x2) / 2, "foot_y": bottom, "confidence": float(score),
        })
    return found


def seed_target(candidates: list[dict[str, float]], width: int, height: int, seed_y_range: tuple[float, float], seed_x_range: tuple[float, float], seed_strategy: str) -> dict[str, float] | None:
    eligible = [
        row for row in candidates
        if seed_y_range[0] <= row["foot_y"] / height <= seed_y_range[1]
        and seed_x_range[0] <= row["foot_x"] / width <= seed_x_range[1]
    ]
    if seed_strategy == "nearest_net":
        # In this fixed-camera view, a larger image y is closer to the net on
        # the far court. This operationalizes the far-front role at impact.
        return max(eligible, key=lambda row: (row["foot_y"], row["confidence"]), default=None)
    return max(eligible, key=lambda row: row["confidence"], default=None)


def iou(a: dict[str, float], b: dict[str, float]) -> float:
    left, top = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    right, bottom = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - intersection
    return intersection / union if union > 0 else 0.0


def associate(previous: dict[str, float], candidates: list[dict[str, float]], width: int, height: int) -> tuple[dict[str, float] | None, float | None]:
    """Choose the locally continuous target; reject implausible identity jumps."""
    diagonal = math.hypot(width, height)
    scored: list[tuple[float, dict[str, float]]] = []
    for row in candidates:
        distance = math.hypot(row["cx"] - previous["cx"], row["cy"] - previous["cy"]) / diagonal
        overlap = iou(previous, row)
        # At 60 fps the same player should remain very close. Appearance alone
        # is insufficient because both front players may wear similar colours.
        score = distance - 0.055 * overlap - 0.010 * row["confidence"]
        scored.append((score, row))
    if not scored:
        return None, None
    score, chosen = min(scored, key=lambda pair: pair[0])
    return (chosen, score) if score <= 0.070 else (None, None)


def write_preview(project: Path, input_path: Path, event_id: str, detections: dict[int, list[dict[str, float]]], targets: dict[int, dict[str, float] | None], start: int, end: int, impact: int, fps: float, target_label: str) -> Path:
    preview_dir = project / "data" / "tracking" / "front_target_highres" / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    temporary = preview_dir / f"{event_id}.avi"
    output = preview_dir / f"{event_id}.mp4"
    temporary.unlink(missing_ok=True)
    capture = cv2.VideoCapture(str(input_path)); capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer: cv2.VideoWriter | None = None
    for frame_index in range(start, end + 1):
        ok, frame = capture.read()
        if not ok:
            break
        overlay = frame.copy()
        for item in detections.get(frame_index, []):
            x1, y1, x2, y2 = (round(item[key]) for key in ("x1", "y1", "x2", "y2"))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLORS["candidate"], 2)
        target = targets.get(frame_index)
        if target:
            x1, y1, x2, y2 = (round(target[key]) for key in ("x1", "y1", "x2", "y2"))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLORS["target"], 5)
            cv2.putText(overlay, target_label, (x1, max(42, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS["target"], 3, cv2.LINE_AA)
        relative = (frame_index - impact) / fps
        cv2.putText(overlay, f"{event_id} / F{frame_index} / approx {relative:+.3f}s", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
        if writer is None:
            preview_width = 1280; preview_height = round(overlay.shape[0] * preview_width / overlay.shape[1])
            writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"MJPG"), fps, (preview_width, preview_height))
        writer.write(cv2.resize(overlay, (preview_width, preview_height), interpolation=cv2.INTER_AREA))
    capture.release()
    if writer is not None:
        writer.release()
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(temporary), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], check=True)
    temporary.unlink(missing_ok=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--pre-sec", type=float, default=0.60)
    parser.add_argument("--post-sec", type=float, default=0.10)
    parser.add_argument("--event-id", action="append", default=None, help="Optional event id; repeat this option for a multi-event smoke test")
    parser.add_argument("--target-front-camera-end", choices=("near", "far"), default="near")
    parser.add_argument("--roi-y-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--seed-foot-y-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--seed-foot-x-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--seed-strategy", choices=("highest_confidence", "nearest_net"), default=None)
    parser.add_argument(
        "--direction-review", type=Path, default=None,
        help=("Optional front-view direction-review CSV.  When supplied, extract only "
              "human-confirmed front_left/front_right events from that file."),
    )
    parser.add_argument(
        "--output-prefix", default=None,
        help="Separate output prefix. Required when --direction-review is supplied.",
    )
    parser.add_argument(
        "--seed-overrides", type=Path, default=None,
        help=("Optional confirmed manual re-seed CSV from script 72. A selected "
              "candidate rank replaces the automatic impact-frame seed."),
    )
    parser.add_argument(
        "--seed-candidates", type=Path, default=None,
        help=("Optional candidate CSV produced by script 71.  Together with "
              "--seed-overrides, use the exact human-selected impact-frame box "
              "rather than reconstructing its rank from a new detector run."),
    )
    parser.add_argument(
        "--only-overridden-events", action="store_true",
        help="With --seed-overrides, extract only manually confirmed repair events.",
    )
    args = parser.parse_args()
    if not args.input.is_file() or not args.model.is_file():
        raise SystemExit("--input and --model must be existing files")
    project = Path(__file__).resolve().parents[1]
    os.environ.setdefault("YOLO_CONFIG_DIR", str(project / ".ultralytics"))
    from ultralytics import YOLO  # noqa: PLC0415

    if args.target_front_camera_end == "far":
        roi_y_range = tuple(args.roi_y_range or FAR_FRONT_ROI_Y_RANGE)
        seed_y_range = tuple(args.seed_foot_y_range or FAR_SEED_FOOT_Y_RANGE)
        seed_x_range = tuple(args.seed_foot_x_range or FAR_SEED_FOOT_X_RANGE)
        seed_strategy = args.seed_strategy or "nearest_net"
    else:
        roi_y_range = tuple(args.roi_y_range or NEAR_FRONT_ROI_Y_RANGE)
        seed_y_range = tuple(args.seed_foot_y_range or NEAR_SEED_FOOT_Y_RANGE)
        seed_x_range = tuple(args.seed_foot_x_range or NEAR_SEED_FOOT_X_RANGE)
        seed_strategy = args.seed_strategy or "highest_confidence"
    if not (0 <= roi_y_range[0] < roi_y_range[1] <= 1 and 0 <= seed_y_range[0] < seed_y_range[1] <= 1 and 0 <= seed_x_range[0] < seed_x_range[1] <= 1):
        raise SystemExit("ROI and seed ranges must be ordered fractions in [0, 1]")

    if args.direction_review:
        if not args.direction_review.is_file():
            raise SystemExit(f"Direction-review CSV not found: {args.direction_review}")
        if not args.output_prefix:
            raise SystemExit("--output-prefix is required with --direction-review to preserve pilot outputs")
        events = [
            {
                "event_id": row["candidate_id"],
                "video_id": row["video_id"],
                "impact_source_frame": row["impact_source_frame"],
                "target_front_player_id": f"{args.target_front_camera_end}_front",
            }
            for row in read(args.direction_review)
            if row.get("video_id") == args.video_id
            and row.get("review_status") == "confirmed"
            and (row.get("actual_direction_front_view") or row.get("actual_direction_target_front_view")) in ("front_left", "front_right")
        ]
    else:
        usable_ids = {
            row["event_id"] for row in read(project / "annotations" / "motion_prediction_track_review.csv")
            if row["tracking_status"] == "usable"
        }
        events = [
            row for row in read(project / "annotations" / "motion_prediction_events.csv")
            if row["video_id"] == args.video_id and row["inclusion_status"] == "included" and row["event_id"] in usable_ids and row["target_front_player_id"] == "near_front"
        ]
    if args.event_id:
        requested_ids = set(args.event_id)
        events = [row for row in events if row["event_id"] in requested_ids]
    seed_ranks: dict[str, int] = {}
    if args.seed_overrides:
        if not args.seed_overrides.is_file():
            raise SystemExit(f"Seed-override CSV not found: {args.seed_overrides}")
        for row in read(args.seed_overrides):
            if row.get("review_status") != "confirmed":
                continue
            try:
                rank = int(row["selected_candidate_rank"])
            except (KeyError, ValueError):
                raise SystemExit(f"Invalid selected_candidate_rank in {args.seed_overrides}: {row}")
            seed_ranks[row["event_id"]] = rank
    if args.seed_candidates and not args.seed_overrides:
        raise SystemExit("--seed-candidates requires --seed-overrides")
    selected_seed_boxes: dict[tuple[str, int], dict[str, float]] = {}
    if args.seed_candidates:
        if not args.seed_candidates.is_file():
            raise SystemExit(f"Seed-candidate CSV not found: {args.seed_candidates}")
        for row in read(args.seed_candidates):
            try:
                event_id = row["event_id"]
                rank = int(row["candidate_rank"])
                selected_seed_boxes[(event_id, rank)] = {
                    "x1": float(row["x1_px"]), "y1": float(row["y1_px"]),
                    "x2": float(row["x2_px"]), "y2": float(row["y2_px"]),
                    "cx": float(row["center_x_px"]), "cy": float(row["center_y_px"]),
                    "foot_x": float(row["center_x_px"]), "foot_y": float(row["y2_px"]),
                    "confidence": float(row["detector_confidence"]),
                }
            except (KeyError, ValueError) as exc:
                raise SystemExit(f"Invalid candidate row in {args.seed_candidates}: {row}") from exc
    if args.only_overridden_events:
        if not args.seed_overrides:
            raise SystemExit("--only-overridden-events requires --seed-overrides")
        events = [row for row in events if row["event_id"] in seed_ranks]
    if not events:
        raise SystemExit("No usable near_front events found")
    metadata = cv2.VideoCapture(str(args.input))
    fps = metadata.get(cv2.CAP_PROP_FPS); width = int(metadata.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(metadata.get(cv2.CAP_PROP_FRAME_HEIGHT)); frame_count = int(metadata.get(cv2.CAP_PROP_FRAME_COUNT)); metadata.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise SystemExit("Could not read input video metadata")
    model = YOLO(str(args.model))
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for event in events:
        event_id, impact = event["event_id"], int(event["impact_source_frame"])
        start, end = max(0, round(impact - args.pre_sec * fps)), min(frame_count - 1, round(impact + args.post_sec * fps))
        capture = cv2.VideoCapture(str(args.input)); capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        candidates_by_frame: dict[int, list[dict[str, float]]] = {}
        for frame_index in range(start, end + 1):
            ok, frame = capture.read()
            if not ok:
                break
            candidates_by_frame[frame_index] = person_detections(model, frame, width, height, args.confidence, args.imgsz, roi_y_range)
        capture.release()
        impact_candidates = candidates_by_frame.get(impact, [])
        manual_rank = seed_ranks.get(event_id)
        if manual_rank is None:
            seed = seed_target(impact_candidates, width, height, seed_y_range, seed_x_range, seed_strategy)
            seed_provenance = f"automatic_{seed_strategy}"
        elif args.seed_candidates:
            seed = selected_seed_boxes.get((event_id, manual_rank))
            if seed is None:
                raise SystemExit(
                    f"{event_id}: selected rank {manual_rank} from {args.seed_overrides} "
                    f"was not found in {args.seed_candidates}"
                )
            seed_provenance = "human_reseed_exact_box"
        else:
            eligible = [
                row for row in impact_candidates
                if seed_y_range[0] <= row["foot_y"] / height <= seed_y_range[1]
                and seed_x_range[0] <= row["foot_x"] / width <= seed_x_range[1]
            ]
            if not 1 <= manual_rank <= len(eligible):
                raise SystemExit(
                    f"{event_id}: manual seed rank {manual_rank} is outside "
                    f"the {len(eligible)} impact-frame candidates"
                )
            seed = eligible[manual_rank - 1]
            seed_provenance = "human_reseed"
        targets: dict[int, dict[str, float] | None] = {impact: seed}
        association: dict[int, tuple[str, float | None]] = {impact: ("seed" if seed else "seed_missing", 0.0 if seed else None)}
        for direction in (-1, 1):
            previous = seed
            frame_index = impact + direction
            while start <= frame_index <= end:
                if previous is None:
                    targets[frame_index] = None; association[frame_index] = ("missing_after_gap", None)
                else:
                    chosen, score = associate(previous, candidates_by_frame.get(frame_index, []), width, height)
                    targets[frame_index] = chosen; association[frame_index] = ("associated" if chosen else "missing", score)
                    previous = chosen
                frame_index += direction
        observed = 0
        for frame_index in range(start, end + 1):
            target = targets.get(frame_index)
            status, score = association.get(frame_index, ("missing", None))
            if target:
                observed += 1
            rows.append({
                "event_id": event_id, "video_id": args.video_id, "model": args.model.name,
                "source_frame": frame_index, "timestamp_sec": f"{frame_index / fps:.6f}",
                "relative_to_impact_sec": f"{(frame_index - impact) / fps:.6f}",
                "visible": int(target is not None), "association_status": status,
                "association_score": "" if score is None else f"{score:.6f}",
                "x1_px": "" if not target else f"{target['x1']:.3f}", "y1_px": "" if not target else f"{target['y1']:.3f}",
                "x2_px": "" if not target else f"{target['x2']:.3f}", "y2_px": "" if not target else f"{target['y2']:.3f}",
                "foot_x_px": "" if not target else f"{target['foot_x']:.3f}", "foot_y_px": "" if not target else f"{target['foot_y']:.3f}",
                "foot_x_norm": "" if not target else f"{target['foot_x'] / width:.8f}", "foot_y_norm": "" if not target else f"{target['foot_y'] / height:.8f}",
                "detector_confidence": "" if not target else f"{target['confidence']:.6f}",
            })
        preview = write_preview(project, args.input, event_id, candidates_by_frame, targets, start, end, impact, fps, f"{args.target_front_camera_end}_front")
        total = end - start + 1
        summaries.append({"event_id": event_id, "model": args.model.name, "target_front_camera_end": args.target_front_camera_end, "impact_source_frame": impact, "start_frame": start, "end_frame": end, "frames": total, "observed_frames": observed, "observation_rate": f"{observed / total:.4f}", "seed_status": association[impact][0], "seed_strategy": seed_provenance, "seed_candidate_rank": "" if manual_rank is None else manual_rank, "preview_path": str(preview)})
        print(f"{event_id}: {observed}/{total} ({observed / total:.1%})")
    output_dir = project / "data" / "tracking" / "front_target_highres"; output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_prefix or f"{args.video_id}_target_front"
    track_path = output_dir / f"{output_prefix}_tracks_{args.model.stem}.csv"
    summary_path = output_dir / f"{output_prefix}_summary_{args.model.stem}.csv"
    fields = list(rows[0])
    with track_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    print(track_path); print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
