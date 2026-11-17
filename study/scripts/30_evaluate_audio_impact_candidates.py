"""Evaluate local audio-transient candidates against visually labelled impacts.

Audio-only results are a baseline, not an impact detector.  The later system
may use this score only after a video-derived play window has been selected.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np


OFFSET_SEC = 0.103917  # median local peak delay from script 29; match1 only


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def peaks_in_window(time: np.ndarray, score: np.ndarray, start: float, end: float, top_k: int) -> list[tuple[float, float]]:
    candidates = np.flatnonzero((time >= start) & (time <= end))
    if not len(candidates):
        return []
    # Local maxima over 25 ms. Ranking is local to a reviewed play window,
    # avoiding the failed global top-percentile threshold.
    radius = 5
    maxima = []
    for index in candidates:
        lo, hi = max(0, index - radius), min(len(score), index + radius + 1)
        if score[index] >= np.max(score[lo:hi]):
            maxima.append(index)
    maxima.sort(key=lambda index: score[index], reverse=True)
    selected = []
    for index in maxima:
        candidate_time = float(time[index] - OFFSET_SEC)
        if any(abs(candidate_time - chosen[0]) < 0.20 for chosen in selected):
            continue
        selected.append((candidate_time, float(score[index])))
        if len(selected) == top_k:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tolerance-ms", type=float, default=250.0)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    directory = project / "data" / "segments"
    labels = {row["window_id"]: row for row in read(directory / f"{args.video_id}_serve_impact_review.csv") if row["impact_review_status"] == "impact_confirmed"}
    windows = {row["window_id"]: row for row in read(directory / f"{args.video_id}_rally_window_candidates.csv")}
    sr, frame, hop = 48000, 240, 240  # 5 ms analysis and hop
    completed = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(args.input), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "pipe:1"], capture_output=True, check=True)
    audio = np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    difference = np.diff(audio, prepend=audio[0])
    csum = np.concatenate(([0.0], np.cumsum(difference * difference)))
    starts = np.arange(0, len(audio) - frame + 1, hop)
    score = np.sqrt((csum[starts + frame] - csum[starts]) / frame + 1e-12)
    time = (starts + frame / 2) / sr
    rows = []
    tolerance = args.tolerance_ms / 1000
    for ident, label in labels.items():
        window = windows[ident]
        candidates = peaks_in_window(time, score, float(window["source_start_timestamp_sec"]), float(window["source_end_timestamp_sec"]), args.top_k)
        truth = float(label["human_valid_serve_impact_timestamp_sec"])
        hit_rank = next((n for n, (candidate, _) in enumerate(candidates, start=1) if abs(candidate - truth) <= tolerance), "")
        rows.append({
            "video_id": args.video_id, "window_id": ident,
            "visual_impact_timestamp_sec": f"{truth:.6f}",
            "candidate_times_sec": ";".join(f"{candidate:.6f}" for candidate, _ in candidates),
            "top_k": args.top_k, "tolerance_ms": args.tolerance_ms,
            "hit_rank": hit_rank, "top_k_hit": "true" if hit_rank else "false",
            "method": "audio_local_transient_only_with_match1_offset",
        })
    path = directory / f"{args.video_id}_audio_candidate_baseline.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    hits = sum(row["top_k_hit"] == "true" for row in rows)
    print(f"Audio-only top-{args.top_k} recall within {args.tolerance_ms:.0f}ms: {hits}/{len(rows)} ({100 * hits / len(rows):.1f}%)")
    print(path)


if __name__ == "__main__":
    main()
