"""Run the unchanged, GPU-only match2 player tracker once per point.

The inventory is passed as ball_path solely for the legacy protocol hash.
No ball detection, tracking, or rendering is invoked.
"""

import argparse
import csv
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
LEGACY = WORK / "outputs" / "ball_quality_20261111"
INVENTORY = WORK / "outputs" / "match2_point_inventory_20261117" / "point_inventory.csv"
DEFAULT_POINTS = (7, 8, 9, 10, 11)
TIMING_FIELDS = (
    "point", "first_frame", "last_frame", "frames", "seconds", "frames_per_second",
)
SUMMARY_FIELDS = (
    "point", "first_frame", "last_frame", "interval_frames",
    "camera_line_supported_frames", "four_player_frames", "four_player_ratio",
    "three_or_fewer_player_frames",
)
# Schema of the existing tracker; used only when it produces zero position rows.
PROXY_FIELDS = (
    "point", "segment", "source_frame", "timestamp_sec", "track_id",
    "x1", "y1", "x2", "y2", "bottom_center_x", "bottom_center_y",
    "confidence", "candidate_origin", "player_slot", "court_proxy_x", "court_proxy_y",
    "assignment_margin", "assignment_cost", "identity_source",
)


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fields, rows):
    """Replace one small aggregate CSV atomically, including its header if empty."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_scopes(inventory, points):
    rows = read_csv(inventory)
    required = {"point", "display_from_frame", "next_display_from_frame"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Inventory requires columns {sorted(required)}: {inventory}")
    indexed = {}
    for row in rows:
        point = int(row["point"])
        if point in indexed:
            raise ValueError(f"Duplicate point in inventory: {point}")
        indexed[point] = row
    scopes = {}
    for point in points:
        if point not in indexed:
            raise ValueError(f"Point {point} not found in inventory: {inventory}")
        row = indexed[point]
        first = int(row["display_from_frame"])
        last = int(row["next_display_from_frame"]) - 1
        if first < 0 or last < first:
            raise ValueError(f"Invalid frame interval for point {point}: {first}..{last}")
        scopes[point] = (first, first, last)
    return scopes


def summarize_point(point, first, last, destination):
    rows = read_csv(destination / "player_frame_availability.csv")
    expected_frames = set(range(first, last + 1))
    frames = [int(row["source_frame"]) for row in rows]
    if len(frames) != len(expected_frames) or set(frames) != expected_frames:
        raise ValueError(f"Point {point}: incomplete or duplicate availability frames; use --force")
    supported = []
    for row in rows:
        if int(row["point"]) != point or int(row["in_core"]) != 1:
            raise ValueError(f"Point {point}: availability contains a different point or preroll")
        line = int(row["camera_line_supported"])
        assigned = int(row["assigned_slots"])
        if line not in (0, 1) or not 0 <= assigned <= 4:
            raise ValueError(f"Point {point}: invalid line support or assigned_slots")
        if not line and assigned:
            raise ValueError(f"Point {point}: positions without camera-line support")
        if line:
            supported.append(assigned)
    if not (destination / "player_court_proxies.csv").exists() and any(supported):
        raise ValueError(f"Point {point}: assigned positions reported but proxy CSV missing")
    four = sum(assigned == 4 for assigned in supported)
    return dict(
        point=point, first_frame=first, last_frame=last, interval_frames=len(rows),
        camera_line_supported_frames=len(supported), four_player_frames=four,
        four_player_ratio=f"{four / len(supported):.6f}" if supported else "",
        three_or_fewer_player_frames=len(supported) - four,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, nargs="+", default=list(DEFAULT_POINTS))
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--tracking-dir", type=Path, default=HERE / "tracking")
    parser.add_argument("--force", action="store_true", help="Recompute selected existing points.")
    args = parser.parse_args()
    if any(point < 1 for point in args.points):
        parser.error("--points must be positive integers")
    if len(set(args.points)) != len(args.points):
        parser.error("--points must not contain duplicates")
    inventory = args.inventory.resolve()
    tracking = args.tracking_dir.resolve()
    scopes = load_scopes(inventory, sorted(args.points))
    tracking.mkdir(parents=True, exist_ok=True)
    timing_path = tracking / "timing.csv"
    timings = {int(row["point"]): row for row in read_csv(timing_path)} if timing_path.exists() else {}

    # Import only when at least one point actually needs GPU processing.
    track_players = None
    for point, scope in scopes.items():
        destination = tracking / f"point_{point:02d}"
        proxy_path = destination / "player_court_proxies.csv"
        if proxy_path.exists() and not args.force:
            print(f"point {point:02d}: skip (player_court_proxies.csv exists)", flush=True)
            continue
        if track_players is None:
            sys.path.insert(0, str(LEGACY))
            from build_match2_player_ball_review import track_players

        destination.mkdir(parents=True, exist_ok=True)
        # common.write() leaves an existing file untouched when there are no rows.
        # Remove just this output so a zero-assignment rerun cannot reuse old positions.
        if proxy_path.exists():
            proxy_path.unlink()
        if point in timings:
            del timings[point]
            write_csv(timing_path, TIMING_FIELDS, [timings[p] for p in sorted(timings)])
        first, _, last = scope
        print(f"point {point:02d}: F{first}..F{last}", flush=True)
        started = time.perf_counter()
        track_players(destination=destination, scopes={point: scope}, ball_path=inventory)
        seconds = time.perf_counter() - started

        # Validate the complete availability interval before creating an empty sentinel.
        summarize_point(point, first, last, destination)
        if not proxy_path.exists():
            write_csv(proxy_path, PROXY_FIELDS, [])
        frames = last - first + 1
        timings[point] = dict(
            point=point, first_frame=first, last_frame=last, frames=frames,
            seconds=f"{seconds:.6f}",
            frames_per_second=f"{frames / seconds:.6f}" if seconds else "",
        )
        # Persist every successful point so an interrupted later point loses no timing.
        write_csv(timing_path, TIMING_FIELDS, [timings[p] for p in sorted(timings)])
        print(f"point {point:02d}: {seconds:.3f} s", flush=True)

    if not timing_path.exists():
        # Skips do not fabricate zero-second measurements.
        write_csv(timing_path, TIMING_FIELDS, [])
    summary_path = tracking / "summary.csv"
    summary = {int(row["point"]): row for row in read_csv(summary_path)} if summary_path.exists() else {}
    for point, (first, _, last) in scopes.items():
        summary[point] = summarize_point(point, first, last, tracking / f"point_{point:02d}")
    write_csv(summary_path, SUMMARY_FIELDS, [summary[p] for p in sorted(summary)])
    print(f"Timing: {timing_path}\nAvailability: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
