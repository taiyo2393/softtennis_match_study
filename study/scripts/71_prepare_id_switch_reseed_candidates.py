"""Render human-selectable impact-frame seeds for ID-switch repairs.

This is intentionally restricted to events a reviewer marked ``id_switch``.
It does not repair a trajectory by itself: it records every plausible detector
box at the impact frame so a human can select the target front player before a
separate re-tracking and audit step.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2


NEAR_SEED_FOOT_Y_RANGE = (0.56, 0.73)
NEAR_SEED_FOOT_X_RANGE = (0.00, 1.00)
NEAR_FRONT_ROI_Y_RANGE = (0.47, 0.80)
FAR_SEED_FOOT_Y_RANGE = (0.46, 0.59)
FAR_SEED_FOOT_X_RANGE = (0.25, 0.75)
FAR_FRONT_ROI_Y_RANGE = (0.42, 0.60)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--track-review", type=Path, required=True)
    parser.add_argument("--track-summary", type=Path, required=True)
    parser.add_argument(
        "--event-review", type=Path, default=None,
        help=("Optional earlier reseed-review CSV. When supplied, prepare only "
              "rows whose review_status equals --event-review-status."),
    )
    parser.add_argument("--event-review-status", default="excluded")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--target-front-camera-end", choices=("near", "far"), default="near")
    parser.add_argument("--roi-y-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--seed-foot-y-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--seed-foot-x-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1536)
    args = parser.parse_args()
    required_paths = (args.input, args.model, args.track_review, args.track_summary)
    for path in required_paths:
        if not path.is_file():
            raise SystemExit(f"Required file not found: {path}")
    if args.event_review and not args.event_review.is_file():
        raise SystemExit(f"Event-review CSV not found: {args.event_review}")

    project = Path(__file__).resolve().parents[1]
    os.environ.setdefault("YOLO_CONFIG_DIR", str(project / ".ultralytics"))
    from ultralytics import YOLO  # noqa: PLC0415

    if args.target_front_camera_end == "far":
        roi_y_range = tuple(args.roi_y_range or FAR_FRONT_ROI_Y_RANGE)
        seed_y_range = tuple(args.seed_foot_y_range or FAR_SEED_FOOT_Y_RANGE)
        seed_x_range = tuple(args.seed_foot_x_range or FAR_SEED_FOOT_X_RANGE)
    else:
        roi_y_range = tuple(args.roi_y_range or NEAR_FRONT_ROI_Y_RANGE)
        seed_y_range = tuple(args.seed_foot_y_range or NEAR_SEED_FOOT_Y_RANGE)
        seed_x_range = tuple(args.seed_foot_x_range or NEAR_SEED_FOOT_X_RANGE)
    if not (0 <= roi_y_range[0] < roi_y_range[1] <= 1 and 0 <= seed_y_range[0] < seed_y_range[1] <= 1 and 0 <= seed_x_range[0] < seed_x_range[1] <= 1):
        raise SystemExit("ROI and seed ranges must be ordered fractions in [0, 1]")

    if args.event_review:
        switched = {
            row["event_id"] for row in read(args.event_review)
            if row.get("review_status") == args.event_review_status
        }
    else:
        switched = {
            row["event_id"] for row in read(args.track_review)
            if row.get("tracking_status") == "id_switch"
        }
    impacts = {
        row["event_id"]: int(row["impact_source_frame"])
        for row in read(args.track_summary)
        if row.get("event_id") in switched
    }
    if not impacts:
        raise SystemExit("No id_switch events found in the supplied review and summary files.")

    output_dir = project / "data" / "tracking" / "front_target_highres" / "reseed_candidates" / args.output_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = project / "data" / "tracking" / "front_target_highres" / f"{args.output_prefix}_seed_candidates.csv"
    model = YOLO(str(args.model))
    capture = cv2.VideoCapture(str(args.input))
    rows: list[dict[str, str]] = []
    for event_id, impact in impacts.items():
        capture.set(cv2.CAP_PROP_POS_FRAMES, impact)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not decode F{impact} for {event_id}")
        height, width = frame.shape[:2]
        y0, y1 = round(height * roi_y_range[0]), round(height * roi_y_range[1])
        result = model.predict(frame[y0:y1, :], classes=[0], conf=args.confidence, imgsz=args.imgsz, verbose=False)[0]
        candidates: list[dict[str, float]] = []
        for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            x1, top, x2, bottom = [float(value) for value in box]
            top += y0; bottom += y0
            foot_y = bottom / height
            foot_x = (x1 + x2) / 2 / width
            if seed_y_range[0] <= foot_y <= seed_y_range[1] and seed_x_range[0] <= foot_x <= seed_x_range[1]:
                candidates.append({
                    "x1": x1, "y1": top, "x2": x2, "y2": bottom,
                    "confidence": float(score),
                })
        if not candidates:
            raise RuntimeError(f"No seed candidates at F{impact} for {event_id}")

        overlay = frame.copy()
        for rank, candidate in enumerate(candidates, start=1):
            color = (0, 220, 255) if rank == 1 else (180, 180, 180)
            x1, y1, x2, y2 = (round(candidate[key]) for key in ("x1", "y1", "x2", "y2"))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 6)
            cv2.putText(overlay, f"#{rank}", (x1, max(70, y1 - 16)), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 5, cv2.LINE_AA)
            rows.append({
                "event_id": event_id,
                "impact_source_frame": str(impact),
                "candidate_rank": str(rank),
                "automatic_seed": str(rank == 1).lower(),
                "x1_px": f"{candidate['x1']:.3f}", "y1_px": f"{candidate['y1']:.3f}",
                "x2_px": f"{candidate['x2']:.3f}", "y2_px": f"{candidate['y2']:.3f}",
                "center_x_px": f"{(candidate['x1'] + candidate['x2']) / 2:.3f}",
                "center_y_px": f"{(candidate['y1'] + candidate['y2']) / 2:.3f}",
                "foot_y_norm": f"{candidate['y2'] / height:.6f}",
                "foot_x_norm": f"{(candidate['x1'] + candidate['x2']) / 2 / width:.6f}",
                "detector_confidence": f"{candidate['confidence']:.6f}",
                "image_path": str(output_dir / f"{event_id}.jpg"),
            })
        image_path = output_dir / f"{event_id}.jpg"
        preview_width = 1600
        preview_height = round(height * preview_width / width)
        cv2.imwrite(str(image_path), cv2.resize(overlay, (preview_width, preview_height), interpolation=cv2.INTER_AREA))

    capture.release()
    fields = list(rows[0])
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {output_csv} ({len(impacts)} events, {len(rows)} candidate boxes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
