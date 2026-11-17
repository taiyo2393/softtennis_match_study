"""Attach stable target-player IDs and stratify motion-direction summaries.

Pixel thresholds are camera-perspective dependent.  This tool deliberately
keeps ``near`` and ``far`` camera ends separate and reports only descriptive
player-by-camera-end time courses.  It must not be used to pool the four
strata into one player-independent rate.
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
    if not total:
        return None, None
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-timepoints", type=Path, action="append", required=True)
    parser.add_argument("--identity-review", type=Path, required=True)
    parser.add_argument("--movement-threshold-px", type=float, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if args.movement_threshold_px <= 0:
        raise SystemExit("--movement-threshold-px must be positive")
    for path in [*args.event_timepoints, args.identity_review]:
        if not path.is_file():
            raise SystemExit(f"Required input not found: {path}")
    identities = {
        row["event_id"]: row
        for row in read(args.identity_review)
        if row.get("identity_status") == "confirmed" and row.get("target_front_player_id")
    }
    seen: set[tuple[str, str]] = set()
    enriched: list[dict[str, object]] = []
    for path in args.event_timepoints:
        for row in read(path):
            key = (row["event_id"], row["timepoint_sec"])
            if key in seen:
                raise SystemExit(f"Duplicate event/timepoint across input tables: {key}")
            seen.add(key)
            identity = identities.get(row["event_id"])
            if identity is None:
                continue
            enriched.append({
                **row,
                "target_front_player_id": identity["target_front_player_id"],
                "target_front_camera_end": identity["target_front_camera_end"],
            })
    if not enriched:
        raise SystemExit("No event-timepoint rows had a confirmed target-player ID")
    by_stratum: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in enriched:
        by_stratum[(
            str(row["target_front_player_id"]),
            str(row["target_front_camera_end"]),
            str(row["timepoint_sec"]),
        )].append(row)
    summaries: list[dict[str, object]] = []
    for (player, camera_end, timepoint), group in sorted(by_stratum.items(), key=lambda item: (item[0][0], item[0][1], float(item[0][2]))):
        movement = Counter(str(row["movement_direction_target_front_view"]) for row in group)
        actual = Counter(str(row["actual_direction_target_front_view"]) for row in group)
        directional = movement["front_left"] + movement["front_right"]
        matches = sum(row["direction_matches_actual"] == "1" or row["direction_matches_actual"] == 1 for row in group)
        lower, upper = wilson(matches, directional)
        summaries.append({
            "target_front_player_id": player,
            "target_front_camera_end": camera_end,
            "timepoint_sec": timepoint,
            "movement_threshold_px": f"{args.movement_threshold_px:.1f}",
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
        })
    prefix = args.output_prefix
    write(prefix.with_name(prefix.name + "_event_timepoints.csv"), list(enriched[0]), enriched)
    write(prefix.with_name(prefix.name + "_summary.csv"), list(summaries[0]), summaries)
    prefix.with_name(prefix.name + "_metadata.json").write_text(
        json.dumps({
            "identity_review": str(args.identity_review),
            "event_timepoint_inputs": [str(path) for path in args.event_timepoints],
            "movement_threshold_px": args.movement_threshold_px,
            "stratification": ["target_front_player_id", "target_front_camera_end"],
            "inference_guard": (
                "Camera-end strata remain separate because screen-pixel displacement "
                "is perspective dependent. Counts are descriptive within one match and "
                "are not pooled or used for player-independent inference."
            ),
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Wrote {len(enriched)} enriched rows and {len(summaries)} stratum-time summaries: {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
