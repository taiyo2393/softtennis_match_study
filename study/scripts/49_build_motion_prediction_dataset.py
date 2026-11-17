"""Build a leakage-free event-level table from human-approved role tracks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SAMPLE_OFFSETS = (-0.50, -0.25, -0.10, 0.00, 0.10, 0.20, 0.30)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def closest(rows: list[dict[str, str]], target_frame: int, max_frame_error: int = 1) -> dict[str, str] | None:
    if not rows:
        return None
    found = min(rows, key=lambda row: abs(int(row["source_frame"]) - target_frame))
    return found if abs(int(found["source_frame"]) - target_frame) <= max_frame_error else None


def key(offset: float) -> str:
    return f"m{abs(round(offset * 100)):03d}" if offset < 0 else f"p{round(offset * 100):03d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    events = [row for row in read(project / "annotations" / "motion_prediction_events.csv") if row["video_id"] == args.video_id and row["inclusion_status"] == "included"]
    review = {row["event_id"]: row for row in read(project / "annotations" / "motion_prediction_track_review.csv") if row["tracking_status"] == "usable"}
    tracks = read(project / "data" / "tracking" / "motion_prediction_events" / f"{args.video_id}_event_role_tracks.csv")
    by_event_role: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in tracks:
        by_event_role.setdefault((row["event_id"], row["role"]), []).append(row)
    for rows in by_event_role.values():
        rows.sort(key=lambda row: int(row["source_frame"]))

    feature_offsets = (-0.50, -0.25, -0.10, 0.00)
    output_fields = ["event_id", "video_id", "label_status", "label_note"]
    for prefix in ("front", "hitter"):
        for offset in feature_offsets:
            suffix = key(offset)
            output_fields.extend([f"{prefix}_x_{suffix}", f"{prefix}_y_{suffix}"])
    for horizon in (0.10, 0.20, 0.30):
        output_fields.extend([f"front_x_{key(horizon)}", f"dx_{key(horizon)}"])
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for event in events:
        event_id = event["event_id"]
        if event_id not in review:
            rejected.append({"event_id": event_id, "video_id": args.video_id, "label_status": "excluded", "label_note": "track review is not usable"})
            continue
        front = by_event_role.get((event_id, event["target_front_player_id"]), [])
        hitter = by_event_role.get((event_id, event["opponent_hitter_id"]), [])
        if not front or not hitter:
            rejected.append({"event_id": event_id, "video_id": args.video_id, "label_status": "excluded", "label_note": "target front or hitter track missing"})
            continue
        impact_frame = int(event["impact_source_frame"])
        samples: dict[tuple[str, float], dict[str, str] | None] = {}
        for role, rows in (("front", front), ("hitter", hitter)):
            for offset in SAMPLE_OFFSETS:
                samples[(role, offset)] = closest(rows, round(impact_frame + offset * (60000 / 1001)))
        required = [(role, offset) for role in ("front", "hitter") for offset in feature_offsets] + [("front", offset) for offset in (0.10, 0.20, 0.30)]
        missing = [(role, offset) for role, offset in required if samples[(role, offset)] is None]
        if missing:
            text = ", ".join(f"{role}@{offset:+.2f}" for role, offset in missing)
            rejected.append({"event_id": event_id, "video_id": args.video_id, "label_status": "excluded", "label_note": f"missing direct track samples: {text}"})
            continue
        row = {field: "" for field in output_fields}
        row.update({"event_id": event_id, "video_id": args.video_id, "label_status": "usable", "label_note": "direct observations only; no temporal interpolation"})
        for role in ("front", "hitter"):
            for offset in feature_offsets:
                sample = samples[(role, offset)]
                assert sample is not None
                suffix = key(offset); row[f"{role}_x_{suffix}"] = sample["foot_x_norm"]; row[f"{role}_y_{suffix}"] = sample["foot_y_norm"]
        front_at_zero = float(row["front_x_p000"])
        for horizon in (0.10, 0.20, 0.30):
            sample = samples[("front", horizon)]
            assert sample is not None
            suffix = key(horizon); row[f"front_x_{suffix}"] = sample["foot_x_norm"]; row[f"dx_{suffix}"] = f"{float(sample['foot_x_norm']) - front_at_zero:.8f}"
        accepted.append(row)
    output = project / "analysis" / "motion_prediction_dataset.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields); writer.writeheader(); writer.writerows(accepted)
    rejected_output = project / "analysis" / "motion_prediction_dataset_excluded.csv"
    with rejected_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["event_id", "video_id", "label_status", "label_note"]); writer.writeheader(); writer.writerows(rejected)
    print(f"Usable event rows: {len(accepted)}; excluded: {len(rejected)}")
    print(output); print(rejected_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
