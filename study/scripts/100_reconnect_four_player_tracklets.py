"""Conservative offline reconnection of four-player tracklets in a pilot clip.

This is an exploratory postprocessor, not an identity-accuracy evaluation. It
never overwrites the raw detections or the four-player selection CSV.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def foot(row):
    return float(row["bottom_center_x"]), float(row["bottom_center_y"])


def height(row):
    return float(row["y2"]) - float(row["y1"])


def velocity(items, from_end):
    samples = items[-5:] if from_end else items[:5]
    if len(samples) < 2:
        return (0.0, 0.0)
    a, b = samples[0], samples[-1]
    delta = int(b["source_frame"]) - int(a["source_frame"])
    if delta <= 0:
        return (0.0, 0.0)
    ax, ay = foot(a)
    bx, by = foot(b)
    return ((bx - ax) / delta, (by - ay) / delta)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def proposal(old, new):
    before, after = old[-1], new[0]
    delta = int(after["source_frame"]) - int(before["source_frame"])
    if not 2 <= delta <= 31:
        return None
    h0, h1 = height(before), height(after)
    if min(h0, h1) <= 0 or not 0.6 <= h1 / h0 <= 1.6:
        return None
    p0, p1 = foot(before), foot(after)
    v0, v1 = velocity(old, True), velocity(new, False)
    forward = (p0[0] + v0[0] * delta, p0[1] + v0[1] * delta)
    backward = (p1[0] - v1[0] * delta, p1[1] - v1[1] * delta)
    scale = max(h0, h1)
    residual_forward = distance(forward, p1) / scale
    residual_backward = distance(backward, p0) / scale
    # The two-sided test rejects many plausible-looking but wrong crossings.
    if max(residual_forward, residual_backward) > 0.75:
        return None
    return {
        "gap_frames": delta - 1,
        "forward_error_height": round(residual_forward, 3),
        "backward_error_height": round(residual_backward, 3),
        "score": round(max(residual_forward, residual_backward), 3),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--mode", choices=("bytetrack", "botsort_reid"), default="bytetrack")
    parser.add_argument("--max-interpolate", type=int, default=12)
    args = parser.parse_args()
    if not 0 <= args.max_interpolate <= 30:
        parser.error("max-interpolate must be between 0 and 30")
    raw = read(args.pilot / f"{args.mode}.csv")
    selected = read(args.pilot / f"{args.mode}_selected_four.csv")
    if not raw or not selected:
        raise RuntimeError("Empty raw or selected tracking CSV")
    tracks = defaultdict(list)
    for row in raw:
        if int(row["track_id"]) >= 0:
            tracks[int(row["track_id"])].append(row)
    for rows in tracks.values():
        rows.sort(key=lambda row: int(row["source_frame"]))
    slots = defaultdict(list)
    for row in selected:
        slots[row["player_slot"]].append(row)
    for rows in slots.values():
        rows.sort(key=lambda row: int(row["source_frame"]))
    if len(slots) != 4:
        raise RuntimeError(f"Expected four seeded player slots; got {list(slots)}")
    reserved = {int(row["track_id"]) for row in selected}
    candidates = []
    # The selected output contains only the already trusted chain. Extend its
    # *tail*; never replace an observed middle segment or another player's ID.
    for slot, old in slots.items():
        last = int(old[-1]["source_frame"])
        for tid, new in tracks.items():
            if tid in reserved or len(new) < 5 or int(new[0]["source_frame"]) <= last:
                continue
            metrics = proposal(old, new)
            if metrics is None:
                continue
            candidates.append({"slot": slot, "new_track_id": tid,
                               "old_track_id": int(old[-1]["track_id"]),
                               "last_observed_frame": last,
                               "reappearance_frame": int(new[0]["source_frame"]),
                               **metrics})
    # One candidate for multiple slots, or near-tied alternatives, is
    # ambiguous. Do not assign it just to maximize coverage.
    by_slot = defaultdict(list)
    for item in candidates:
        by_slot[item["slot"]].append(item)
    accepted = []
    for slot, options in by_slot.items():
        options.sort(key=lambda item: item["score"])
        best = options[0]
        rival = options[1] if len(options) > 1 else None
        competing = [item for item in candidates if item["slot"] != slot
                     and item["new_track_id"] == best["new_track_id"]]
        if (rival and rival["score"] - best["score"] < 0.25) or competing:
            continue
        accepted.append(best)
    out = []
    fields = list(selected[0]) + ["position_source", "relink_confidence"]
    for row in selected:
        out.append({**row, "position_source": "observed", "relink_confidence": ""})
    for item in accepted:
        slot = item["slot"]
        old = slots[slot]
        new = tracks[item["new_track_id"]]
        for row in new:
            out.append({**row, "player_slot": slot,
                        "selection_status": "velocity_relinked_review",
                        "position_source": "observed_relinked",
                        "relink_confidence": "requires_identity_review"})
        if item["gap_frames"] <= args.max_interpolate:
            a, b = old[-1], new[0]
            fa, fb = int(a["source_frame"]), int(b["source_frame"])
            xa, ya = foot(a)
            xb, yb = foot(b)
            for frame in range(fa + 1, fb):
                fraction = (frame - fa) / (fb - fa)
                filler = {key: "" for key in fields}
                filler.update(source_frame=frame,
                              timestamp_sec=float(a["timestamp_sec"]) + fraction *
                              (float(b["timestamp_sec"]) - float(a["timestamp_sec"])),
                              track_id=item["new_track_id"],
                              bottom_center_x=xa + fraction * (xb - xa),
                              bottom_center_y=ya + fraction * (yb - ya),
                              player_slot=slot, selection_status="interpolated_gap",
                              position_source="interpolated",
                              relink_confidence="requires_identity_review")
                out.append(filler)
    out.sort(key=lambda row: (int(row["source_frame"]), row["player_slot"]))
    stem = f"{args.mode}_four_player_reconnection"
    write(args.pilot / f"{stem}.csv", out, fields)
    report = {"pilot": str(args.pilot), "mode": args.mode,
              "max_interpolate": args.max_interpolate,
              "candidate_count": len(candidates), "accepted_count": len(accepted),
              "candidates": candidates, "accepted": accepted,
              "warning": "Accepted links are provisional until independently checked."}
    (args.pilot / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("candidate_count", "accepted_count", "accepted")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
