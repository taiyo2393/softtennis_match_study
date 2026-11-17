"""Summarise pooled near/far directional motion after court-plane calibration.

Input x is normalized to court width.  The target front player's own left/right
is obtained from the camera end recorded during human identity review.  The
output remains descriptive: it has no classifier and does not claim that the
front player consciously predicted a shot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0, centre - radius), min(1, centre + radius)


def positive_direction(camera_end: str) -> str:
    # Court x grows from screen-left to screen-right.  A far player faces
    # toward the camera, whereas a near player faces away from it.
    if camera_end == "far":
        return "front_left"
    if camera_end == "near":
        return "front_right"
    raise ValueError(f"Unknown target_front_camera_end: {camera_end}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--movement-threshold-court-width", type=float, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Feature table not found: {args.input}")
    if not (0 < args.movement_threshold_court_width < 1):
        raise SystemExit("--movement-threshold-court-width must be in (0, 1)")
    usable = [
        row for row in read(args.input)
        if (
            row.get("feature_status") == "usable"
            and row.get("actual_direction_front_view") in ("front_left", "front_right")
            and row.get("lateral_displacement_court_width")
        )
    ]
    if not usable:
        raise SystemExit("No usable labelled court-plane displacement rows")
    event_rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in usable:
        displacement = float(row["lateral_displacement_court_width"])
        positive = positive_direction(row["target_front_camera_end"])
        negative = "front_right" if positive == "front_left" else "front_left"
        if displacement >= args.movement_threshold_court_width:
            movement = positive
        elif displacement <= -args.movement_threshold_court_width:
            movement = negative
        else:
            movement = "stay"
        actual = row["actual_direction_front_view"]
        matched = "" if movement == "stay" else int(movement == actual)
        toward_actual = displacement if actual == positive else -displacement
        result = {
            "event_id": row["event_id"],
            "video_id": row["video_id"],
            "target_front_player_id": row["target_front_player_id"],
            "target_front_camera_end": row["target_front_camera_end"],
            "impact_source_frame": row["impact_source_frame"],
            "timepoint_sec": row["timepoint_sec"],
            "actual_direction_target_front_view": actual,
            "lateral_displacement_court_width": f"{displacement:.6f}",
            "movement_direction_target_front_view": movement,
            "movement_is_directional": int(movement != "stay"),
            "direction_matches_actual": matched,
            "signed_displacement_toward_actual_court_width": f"{toward_actual:.6f}",
        }
        event_rows.append(result)
        grouped[row["timepoint_sec"]].append(result)
    summary_rows: list[dict[str, object]] = []
    for timepoint in sorted(grouped, key=float):
        group = grouped[timepoint]
        movement = Counter(str(row["movement_direction_target_front_view"]) for row in group)
        actual = Counter(str(row["actual_direction_target_front_view"]) for row in group)
        directional = movement["front_left"] + movement["front_right"]
        matches = sum(row["direction_matches_actual"] == 1 for row in group)
        lower, upper = wilson(matches, directional)
        signed = [float(row["signed_displacement_toward_actual_court_width"]) for row in group]
        summary_rows.append({
            "timepoint_sec": timepoint,
            "movement_threshold_court_width": f"{args.movement_threshold_court_width:.5f}",
            "events": len(group),
            "actual_front_left": actual["front_left"],
            "actual_front_right": actual["front_right"],
            "movement_front_left": movement["front_left"],
            "movement_front_right": movement["front_right"],
            "stay": movement["stay"],
            "directional_events": directional,
            "directional_rate": f"{directional / len(group):.3f}",
            "matching_direction_events": matches,
            "conditional_match_rate": "" if not directional else f"{matches / directional:.3f}",
            "conditional_match_wilson95_low": "" if lower is None else f"{lower:.3f}",
            "conditional_match_wilson95_high": "" if upper is None else f"{upper:.3f}",
            "all_event_correct_direction_rate": f"{matches / len(group):.3f}",
            "mean_signed_displacement_toward_actual_court_width": f"{sum(signed) / len(signed):.6f}",
        })
    prefix = args.output_prefix
    write(prefix.with_name(prefix.name + "_event_timepoints.csv"), list(event_rows[0]), event_rows)
    write(prefix.with_name(prefix.name + "_summary.csv"), list(summary_rows[0]), summary_rows)
    prefix.with_name(prefix.name + "_metadata.json").write_text(json.dumps({
        "input": str(args.input),
        "movement_threshold_court_width": args.movement_threshold_court_width,
        "definition": (
            "Movement direction is the sign of the preceding 100-ms lateral displacement "
            "in normalized court-width units, mapped into the target front player's own view."
        ),
        "reporting_guard": (
            "This pooled near/far summary requires a usable fixed-camera court calibration. "
            "It is a single-match exploratory association and does not establish anticipation, causality, "
            "or player-independent performance."
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(event_rows)} event-timepoint rows and {len(summary_rows)} summaries: {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
