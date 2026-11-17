"""Audit local audio transients around visually confirmed serve impacts.

This does not detect serves.  Given visual impact labels, it tests whether a
short-window acoustic transient has a stable audio-video offset.  A stable
offset is required before audio can be used even as a ranking feature.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--search-ms", type=float, default=200.0)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    directory = project / "data" / "segments"
    labels = [r for r in read(directory / f"{args.video_id}_serve_impact_review.csv") if r["impact_review_status"] == "impact_confirmed"]
    if not labels:
        raise SystemExit("No confirmed impact labels")
    sr, frame, hop = 48000, 240, 48  # 5 ms window, 1 ms resolution
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(args.input), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "pipe:1"], capture_output=True, check=True)
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    derivative = np.diff(audio, prepend=audio[0])
    cumulative = np.concatenate(([0.0], np.cumsum(derivative * derivative)))
    starts = np.arange(0, len(audio) - frame + 1, hop)
    transient = np.sqrt((cumulative[starts + frame] - cumulative[starts]) / frame + 1e-12)
    time = (starts + frame / 2) / sr
    rows = []
    radius = args.search_ms / 1000
    for label in labels:
        impact = float(label["human_valid_serve_impact_timestamp_sec"])
        nearby = np.flatnonzero((time >= impact - radius) & (time <= impact + radius))
        if not len(nearby):
            continue
        best = nearby[int(np.argmax(transient[nearby]))]
        background = transient[(time >= impact - 1.0) & (time <= impact + 1.0)]
        rows.append({
            "video_id": args.video_id, "window_id": label["window_id"],
            "visual_impact_timestamp_sec": f"{impact:.6f}", "local_audio_peak_timestamp_sec": f"{time[best]:.6f}",
            "audio_minus_visual_ms": f"{(time[best] - impact) * 1000:.3f}",
            "local_transient": f"{transient[best]:.8f}",
            "local_peak_over_median": f"{transient[best] / max(float(np.median(background)), 1e-9):.3f}",
            "interpretation": "local_peak_only_not_a_verified_hit",
        })
    path = directory / f"{args.video_id}_audio_video_offset_audit.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    offsets = np.array([float(row["audio_minus_visual_ms"]) for row in rows])
    print(f"Wrote {len(rows)} local audio-video comparisons to {path}")
    print(f"median_offset_ms={np.median(offsets):.3f}; median_abs_offset_ms={np.median(np.abs(offsets)):.3f}; within_25ms={np.mean(np.abs(offsets) <= 25) * 100:.1f}%")


if __name__ == "__main__":
    main()
