"""Make contact-review sheets from player tracking, without ball inference.

Requires only OpenCV, Pillow and the Python standard library. Run after
track_points.py; source frame numbers are zero-based, as in its CSV outputs.
"""

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
DEFAULT_POINTS = [7, 8, 9, 10, 11]
MIN_RALLY_FRAMES = 120
FRAME_STRIDE = 6
COLUMNS, ROWS = 3, 4
FRAMES_PER_SHEET = COLUMNS * ROWS
TILE_WIDTH = 520
CROP_LEFT, CROP_RIGHT = 160, 160
CROP_TOP, CROP_BOTTOM = 250, 40
JPEG_QUALITY = 85
FONT = Path("C:/Windows/Fonts/meiryo.ttc")

# Exact names and RGB colors from build_044_spatiotemporal_pilot.py.
# Kept local so this CPU-only utility does not import its tracking dependencies.
SLOTS = ["near_front", "near_rear", "far_rear", "far_front"]
NAMES = dict(zip(SLOTS, ["N1", "N2", "F1", "F2"]))
RGB = dict(zip(SLOTS, [(65, 220, 120), (70, 170, 255),
                       (220, 90, 230), (250, 210, 65)]))

INDEX_FIELDS = ["rally", "sheet", "first_frame", "last_frame", "frames_in_sheet"]
SUMMARY_FIELDS = ["point", "rally_intervals", "rally_frames", "rally_seconds",
                  "rally_lengths_frames", "sheets", "sampled_frames"]
TIMING_FIELDS = ["point", "seconds", "rally_intervals", "sheets", "sampled_frames"]


def read_csv(path, required):
    """Read UTF-8 BOM CSV and reject incomplete input with a useful message."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing CSV columns: {sorted(missing)}")
        return list(reader)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def update_point_csv(path, fields, row):
    """Replace this point's result, retaining other points on subset reruns."""
    rows = read_csv(path, fields) if path.exists() else []
    rows = [old for old in rows if int(old["point"]) != row["point"]]
    rows.append(row)
    write_csv(path, fields, sorted(rows, key=lambda item: int(item["point"])))


def load_point(directory, point):
    with (directory / "protocol.json").open(encoding="utf-8-sig") as stream:
        protocol = json.load(stream)
    fps = float(Fraction(str(protocol["source_fps"])))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{directory}: invalid source_fps")
    source = Path(protocol["source_video"])

    rows = read_csv(directory / "player_frame_availability.csv",
                    ["point", "source_frame", "camera_line_supported", "assigned_slots"])
    availability = {}
    for row in rows:
        frame = int(row["source_frame"])
        supported, assigned = int(row["camera_line_supported"]), int(row["assigned_slots"])
        if (int(row["point"]) != point or frame < 0 or frame in availability
                or supported not in (0, 1) or not 0 <= assigned <= 4):
            raise ValueError(f"{directory}: invalid/duplicate availability row: {row}")
        availability[frame] = {"supported": supported, "assigned": assigned}
    if not availability:
        raise ValueError(f"{directory}: player_frame_availability.csv has no frames")

    people = defaultdict(list)
    proxies_path = directory / "player_court_proxies.csv"
    if not proxies_path.exists():
        raise FileNotFoundError(
            f"{proxies_path}: required A output missing; tracking may be incomplete "
            "or the tracker assigned no positions. Check A before making sheets."
        )
    rows = read_csv(proxies_path, ["point", "source_frame", "player_slot",
                                  "bottom_center_x", "bottom_center_y"])
    for row in rows:
        frame, slot = int(row["source_frame"]), row["player_slot"]
        if int(row["point"]) != point or frame not in availability or slot not in NAMES:
            raise ValueError(f"{proxies_path}: unexpected point/frame/slot: {row}")
        x, y = float(row["bottom_center_x"]), float(row["bottom_center_y"])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"{proxies_path}: non-finite marker at F{frame}")
        if any(person["slot"] == slot for person in people[frame]):
            raise ValueError(f"{proxies_path}: duplicate slot {slot} at F{frame}")
        people[frame].append({"slot": slot, "x": x, "y": y})
    for frame, row in availability.items():
        if len(people.get(frame, [])) != row["assigned"]:
            raise ValueError(f"{directory}: assigned_slots disagrees with proxies at F{frame}")
    return source, fps, availability, people


def rally_intervals(availability):
    """All >=120-frame contiguous supported runs, including possible replays."""
    intervals = []
    first = previous = None
    for frame in sorted(availability):
        supported = availability[frame]["supported"] == 1
        if first is not None and (not supported or frame != previous + 1):
            if previous - first + 1 >= MIN_RALLY_FRAMES:
                intervals.append((first, previous))
            first = previous = None
        if supported:
            if first is None:
                first = frame
            previous = frame
    if first is not None and previous - first + 1 >= MIN_RALLY_FRAMES:
        intervals.append((first, previous))
    return intervals


def make_tile(raw, frame, people, font):
    height, width = raw.shape[:2]
    right, bottom = width - CROP_RIGHT, height - CROP_BOTTOM
    crop_width, crop_height = right - CROP_LEFT, bottom - CROP_TOP
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"Crop constants remove the entire {width}x{height} frame")
    scale = TILE_WIDTH / crop_width
    tile_height = round(crop_height * scale)
    crop = raw[CROP_TOP:bottom, CROP_LEFT:right]
    tile = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tile = tile.resize((TILE_WIDTH, tile_height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(tile)
    for person in people:
        # These are bbox bottom centers, not independently verified ground feet.
        x = (person["x"] - CROP_LEFT) * scale
        y = (person["y"] - CROP_TOP) * scale
        if not (0 <= x < TILE_WIDTH and 0 <= y < tile_height):
            continue
        slot = person["slot"]
        color = RGB[slot]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color,
                     outline="black", width=1)
        name = NAMES[slot]
        box = draw.textbbox((0, 0), name, font=font, stroke_width=1)
        label_width, label_height = box[2] - box[0], box[3] - box[1]
        label_x = max(2, min(x + 5, TILE_WIDTH - label_width - 3))
        label_y = max(24, min(y - label_height - 4, tile_height - label_height - 3))
        draw.text((label_x, label_y), name, font=font, fill=color,
                  stroke_width=2, stroke_fill="black")
    draw.rectangle((0, 0, TILE_WIDTH - 1, 25), fill=(10, 18, 26))
    draw.text((6, 1), f"F{frame}", font=font, fill="white")
    count = f"{len(people)}人"
    count_box = draw.textbbox((0, 0), count, font=font)
    draw.text((TILE_WIDTH - (count_box[2] - count_box[0]) - 6, 1), count,
              font=font, fill="white")
    return tile


def make_point(point, tracking_dir, output_dir, font):
    started = time.perf_counter()
    source, fps, availability, people = load_point(tracking_dir / f"point_{point:02d}", point)
    intervals = rally_intervals(availability)
    destination = output_dir / f"point_{point:02d}"
    destination.mkdir(parents=True, exist_ok=True)

    # Only this selected point's generated sheets/index are replaced. Remove old
    # trailing sheets if a rerun has fewer qualifying runs or shorter intervals.
    for old in destination.glob("rally_*_sheet_*.jpg"):
        old.unlink()
    index_path = destination / "index.csv"
    if index_path.exists():
        index_path.unlink()

    index = []
    sample_count = 0
    capture = None
    try:
        if intervals:
            capture = cv2.VideoCapture(str(source))
            assert capture.isOpened(), f"Cannot open source_video: {source}"
            assert abs(capture.get(cv2.CAP_PROP_FPS) - fps) < 0.001, "Source FPS mismatch"
            # One initial seek only. All later frames/rallies advance by grab().
            first = intervals[0][0]
            assert capture.set(cv2.CAP_PROP_POS_FRAMES, first), f"Cannot seek to F{first}"
            assert round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == first, "Initial seek mismatch"
            next_frame = first
            for rally, (first, last) in enumerate(intervals, start=1):
                sheet, sheet_frames = None, []
                sheet_number = 1
                for frame in range(first, last + 1, FRAME_STRIDE):
                    while next_frame < frame:
                        assert capture.grab(), f"Cannot advance past F{next_frame}"
                        next_frame += 1
                        assert round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == next_frame, (
                            f"Frame mismatch while advancing to F{frame}"
                        )
                    assert next_frame == frame, f"Non-monotonic target F{frame}"
                    assert capture.grab(), f"Cannot grab F{frame}"
                    assert round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == frame + 1, (
                        f"Frame mismatch after grab F{frame}"
                    )
                    ok, raw = capture.retrieve()
                    assert ok and raw is not None, f"Cannot retrieve F{frame}"
                    assert round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == frame + 1, (
                        f"Frame mismatch after retrieve F{frame}"
                    )
                    next_frame = frame + 1
                    assert raw.shape[:2] == (1080, 1920), (
                        f"Tracking coordinates require 1920x1080 source, got {raw.shape[:2]}"
                    )
                    tile = make_tile(raw, frame, people.get(frame, []), font)
                    if sheet is None:
                        sheet = Image.new("RGB", (TILE_WIDTH * COLUMNS, tile.height * ROWS),
                                          (10, 18, 26))
                    cell = len(sheet_frames)
                    sheet.paste(tile, ((cell % COLUMNS) * TILE_WIDTH,
                                       (cell // COLUMNS) * tile.height))
                    sheet_frames.append(frame)
                    sample_count += 1
                    if len(sheet_frames) == FRAMES_PER_SHEET or frame + FRAME_STRIDE > last:
                        filename = f"rally_{rally}_sheet_{sheet_number:02d}.jpg"
                        sheet.save(destination / filename, format="JPEG", quality=JPEG_QUALITY)
                        index.append(dict(rally=rally, sheet=filename,
                                          first_frame=sheet_frames[0], last_frame=sheet_frames[-1],
                                          frames_in_sheet=len(sheet_frames)))
                        sheet, sheet_frames = None, []
                        sheet_number += 1
    finally:
        if capture is not None:
            capture.release()

    write_csv(index_path, INDEX_FIELDS, index)
    lengths = [last - first + 1 for first, last in intervals]
    summary = dict(point=point, rally_intervals=len(intervals), rally_frames=sum(lengths),
                   rally_seconds=f"{sum(lengths) / fps:.6f}",
                   rally_lengths_frames=";".join(map(str, lengths)), sheets=len(index),
                   sampled_frames=sample_count)
    update_point_csv(output_dir / "sheets_summary.csv", SUMMARY_FIELDS, summary)
    timing = dict(point=point, seconds=f"{time.perf_counter() - started:.6f}",
                  rally_intervals=len(intervals), sheets=len(index), sampled_frames=sample_count)
    update_point_csv(output_dir / "timing.csv", TIMING_FIELDS, timing)
    print(f"point {point:02d}: {len(intervals)} rally candidates, {len(index)} sheets, "
          f"{sample_count} sampled frames, {timing['seconds']} seconds", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", nargs="+", type=int, default=DEFAULT_POINTS)
    parser.add_argument("--tracking-dir", type=Path, default=HERE / "tracking")
    parser.add_argument("--output-dir", type=Path, default=HERE / "review")
    parser.add_argument("--font", type=Path, default=FONT,
                        help="Japanese-capable font; default C:/Windows/Fonts/meiryo.ttc")
    args = parser.parse_args()
    if any(point <= 0 for point in args.points):
        parser.error("--points must contain positive point numbers")
    if not args.font.is_file():
        parser.error(f"Font missing: {args.font}; specify a Japanese font with --font")
    font = ImageFont.truetype(str(args.font), 18)
    for point in sorted(set(args.points)):
        make_point(point, args.tracking_dir.resolve(), args.output_dir.resolve(), font)


if __name__ == "__main__":
    main()
