"""Evaluate simple motion, audio, and fused impact-candidate rankings.

This is a within-video feasibility baseline.  It avoids a learned action model:
visual score is court-region frame difference and audio score is local
high-frequency transient energy, corrected by the empirically audited match1
audio delay.  No score is a verified impact label.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np


COURT = np.array([[0.02, 0.90], [0.98, 0.90], [0.68, 0.51], [0.32, 0.51]], dtype=np.float32)
OFFSET_SEC = 0.103917


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return (values - median) / max(1.4826 * mad, 1e-6)


def choose(score: np.ndarray, times: np.ndarray, top_k: int) -> list[int]:
    selected = []
    for index in np.argsort(score)[::-1]:
        if any(abs(times[index] - times[previous]) < 0.30 for previous in selected):
            continue
        selected.append(int(index))
        if len(selected) == top_k:
            break
    return selected


def audio_series(input_path: Path) -> tuple[np.ndarray, np.ndarray]:
    sr, frame, hop = 48000, 240, 240
    task = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(input_path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "pipe:1"], capture_output=True, check=True)
    audio = np.frombuffer(task.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    diff = np.diff(audio, prepend=audio[0]); csum = np.concatenate(([0.0], np.cumsum(diff * diff)))
    starts = np.arange(0, len(audio) - frame + 1, hop)
    return (starts + frame / 2) / sr, np.sqrt((csum[starts + frame] - csum[starts]) / frame + 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tolerance-ms", type=float, default=250.0)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]; directory = project / "data" / "segments"
    labels = {r["window_id"]: r for r in read(directory / f"{args.video_id}_serve_impact_review.csv") if r["impact_review_status"] == "impact_confirmed"}
    windows = {r["window_id"]: r for r in read(directory / f"{args.video_id}_rally_window_candidates.csv")}
    atime, ascore = audio_series(args.input)
    cap = cv2.VideoCapture(str(args.input))
    fps = cap.get(cv2.CAP_PROP_FPS); stride = max(1, round(fps / args.sample_fps)); tolerance = args.tolerance_ms / 1000
    rows = []
    for ident, label in labels.items():
        row = windows[ident]; start, end = float(row["source_start_timestamp_sec"]), float(row["source_end_timestamp_sec"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(start * fps))
        previous = None; times = []; visual = []
        for source_frame in range(round(start * fps), round(end * fps) + 1):
            ok, frame = cap.read()
            if not ok: break
            if (source_frame - round(start * fps)) % stride: continue
            small = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if previous is not None:
                mask = np.zeros(gray.shape, np.uint8); poly = (COURT * np.array([gray.shape[1], gray.shape[0]])).astype(np.int32); cv2.fillPoly(mask, [poly], 255)
                visual.append(float(cv2.mean(cv2.absdiff(gray, previous), mask=mask)[0])); times.append(source_frame / fps)
            previous = gray
        if not times: continue
        times_np = np.array(times); visual_np = np.array(visual); audio_np = np.interp(times_np + OFFSET_SEC, atime, ascore)
        scores = {"visual": robust_z(visual_np), "audio": robust_z(audio_np)}
        scores["fusion_equal"] = scores["visual"] + scores["audio"]
        truth = float(label["human_valid_serve_impact_timestamp_sec"])
        record = {"video_id": args.video_id, "window_id": ident, "truth_timestamp_sec": f"{truth:.6f}", "sample_fps": args.sample_fps, "top_k": args.top_k, "tolerance_ms": args.tolerance_ms}
        for name, score in scores.items():
            picks = choose(score, times_np, args.top_k); candidate_times = [times_np[index] for index in picks]
            rank = next((number for number, candidate in enumerate(candidate_times, 1) if abs(candidate - truth) <= tolerance), "")
            record[f"{name}_candidate_times_sec"] = ";".join(f"{candidate:.6f}" for candidate in candidate_times)
            record[f"{name}_hit_rank"] = rank; record[f"{name}_top_k_hit"] = "true" if rank else "false"
        rows.append(record)
    cap.release()
    output = directory / f"{args.video_id}_visual_audio_candidate_baseline.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for name in ("visual", "audio", "fusion_equal"):
        hits = sum(row[f"{name}_top_k_hit"] == "true" for row in rows)
        print(f"{name} top-{args.top_k} recall within {args.tolerance_ms:.0f}ms: {hits}/{len(rows)} ({100*hits/len(rows):.1f}%)")
    print(output)


if __name__ == "__main__":
    main()
