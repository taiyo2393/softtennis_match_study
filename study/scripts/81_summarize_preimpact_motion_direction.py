"""Summarise whether pre-impact front-player movement points to shot direction.

The input is a fixed-camera 2-D feature table from script 69.  This script
does *not* use a classifier: it maps the sign of the preceding-window lateral
displacement into the target front player's left/right, then reports the
agreement with the human-confirmed eventual shot direction.  ``stay`` remains
a separate outcome and is never counted as a correct direction by default.
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


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--screen-positive-is", choices=("front_left", "front_right"), required=True)
    parser.add_argument("--movement-threshold-px", type=float, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input feature table not found: {args.input}")
    if args.movement_threshold_px <= 0:
        raise SystemExit("--movement-threshold-px must be positive")

    negative_direction = "front_right" if args.screen_positive_is == "front_left" else "front_left"
    rows = [
        row for row in read(args.input)
        if row.get("feature_status") == "usable"
        and row.get("actual_direction_front_view") in ("front_left", "front_right")
        and row.get("lateral_displacement_px")
    ]
    if not rows:
        raise SystemExit("No usable labelled displacement rows")

    by_time: dict[str, list[dict[str, object]]] = defaultdict(list)
    event_rows: list[dict[str, object]] = []
    for row in rows:
        displacement = float(row["lateral_displacement_px"])
        if displacement >= args.movement_threshold_px:
            movement = args.screen_positive_is
        elif displacement <= -args.movement_threshold_px:
            movement = negative_direction
        else:
            movement = "stay"
        actual = row["actual_direction_front_view"]
        match = "" if movement == "stay" else int(movement == actual)
        signed_toward_actual = displacement if actual == args.screen_positive_is else -displacement
        result = {
            "event_id": row["event_id"],
            "video_id": row["video_id"],
            "impact_source_frame": row["impact_source_frame"],
            "timepoint_sec": row["timepoint_sec"],
            "actual_direction_target_front_view": actual,
            "lateral_displacement_px": f"{displacement:.3f}",
            "movement_direction_target_front_view": movement,
            "movement_is_directional": int(movement != "stay"),
            "direction_matches_actual": match,
            "signed_displacement_toward_actual_px": f"{signed_toward_actual:.3f}",
        }
        event_rows.append(result)
        by_time[row["timepoint_sec"]].append(result)

    summary_rows: list[dict[str, object]] = []
    for timepoint in sorted(by_time, key=float):
        group = by_time[timepoint]
        movement_counts = Counter(str(row["movement_direction_target_front_view"]) for row in group)
        actual_counts = Counter(str(row["actual_direction_target_front_view"]) for row in group)
        directional = movement_counts["front_left"] + movement_counts["front_right"]
        matches = sum(row["direction_matches_actual"] == 1 for row in group)
        ci_low, ci_high = wilson_interval(matches, directional)
        signed = [float(row["signed_displacement_toward_actual_px"]) for row in group]
        summary_rows.append({
            "timepoint_sec": timepoint,
            "events": len(group),
            "actual_front_left": actual_counts["front_left"],
            "actual_front_right": actual_counts["front_right"],
            "movement_front_left": movement_counts["front_left"],
            "movement_front_right": movement_counts["front_right"],
            "stay": movement_counts["stay"],
            "directional_events": directional,
            "directional_rate": f"{directional / len(group):.3f}",
            "matching_direction_events": matches,
            "conditional_match_rate": "" if directional == 0 else f"{matches / directional:.3f}",
            "conditional_match_wilson95_low": "" if ci_low is None else f"{ci_low:.3f}",
            "conditional_match_wilson95_high": "" if ci_high is None else f"{ci_high:.3f}",
            "all_event_correct_direction_rate": f"{matches / len(group):.3f}",
            "mean_signed_displacement_toward_actual_px": f"{sum(signed) / len(signed):.3f}",
        })

    prefix = args.output_prefix
    write(
        prefix.with_name(prefix.name + "_event_timepoints.csv"),
        list(event_rows[0]), event_rows,
    )
    write(
        prefix.with_name(prefix.name + "_summary.csv"),
        list(summary_rows[0]), summary_rows,
    )
    prefix.with_name(prefix.name + "_metadata.json").write_text(
        json.dumps({
            "input": str(args.input),
            "movement_threshold_px": args.movement_threshold_px,
            "screen_positive_is_target_front_view": args.screen_positive_is,
            "screen_negative_is_target_front_view": negative_direction,
            "definition": (
                "At each pre-impact time point, movement direction is the sign of "
                "the prior 100-ms fixed-camera horizontal displacement if its "
                "absolute value reaches the fixed threshold; otherwise stay."
            ),
            "reporting_guard": (
                "Conditional agreement is reported only among directional movements; "
                "stay is separate and never counted as a correct direction. "
                "This is a single-match exploratory association, not evidence of "
                "anticipation, causality, or match-level generalisation."
            ),
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(
        f"Wrote {len(event_rows)} event-timepoint rows and {len(summary_rows)} "
        f"time summaries: {prefix}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
