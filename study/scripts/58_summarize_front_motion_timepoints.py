"""Create descriptive, threshold-transparent summaries of pre-impact motion."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if not total:
        return None, None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    rows = read(args.input)
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["timepoint_sec"], []).append(row)
    summary: list[dict[str, object]] = []
    for timepoint in sorted(groups, key=float):
        group = groups[timepoint]
        right = sum(row["movement_direction_front_view"] == "front_right" for row in group)
        left = sum(row["movement_direction_front_view"] == "front_left" for row in group)
        stay = sum(row["movement_direction_front_view"] == "stay" for row in group)
        missing = sum(row["movement_direction_front_view"] == "missing" for row in group)
        directional = right + left
        matches = sum(row["direction_matches_actual"] == "1" for row in group)
        low, high = wilson_interval(matches, directional)
        # Positive means the observed motion points toward the final direction.
        signed = [
            float(row["lateral_displacement_px"]) * (1 if row["actual_direction_front_view"] == "front_right" else -1)
            for row in group if row["lateral_displacement_px"]
        ]
        summary.append({
            "timepoint_sec": timepoint, "events": len(group), "front_left": left, "front_right": right,
            "stay": stay, "missing": missing, "directional_events": directional,
            "directional_rate": f"{directional / len(group):.3f}", "matching_direction_events": matches,
            "conditional_match_rate": "" if not directional else f"{matches / directional:.3f}",
            "wilson95_low": "" if low is None else f"{low:.3f}", "wilson95_high": "" if high is None else f"{high:.3f}",
            "mean_direction_signed_displacement_px": f"{sum(signed) / len(signed):.3f}",
        })
    output = args.input.with_name(args.input.stem.replace("_boxproxy", "") + "_summary.csv")
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    print(output)
    for row in summary:
        print(f"t={row['timepoint_sec']}s: directional={row['directional_events']}/{row['events']}, match={row['matching_direction_events']}/{row['directional_events'] or '-'}, stay={row['stay']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
