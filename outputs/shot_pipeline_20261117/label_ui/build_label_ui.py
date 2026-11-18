"""Build offline human-annotation material. All frame intervals are [start, end)."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
FPS = Fraction(60000, 1001)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def probe(path, ffprobe):
    return json.loads(subprocess.check_output([
        str(ffprobe), "-v", "error", "-show_streams", "-of", "json", str(path)
    ], encoding="utf-8"))["streams"]


def thumbnail(frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (480, 270),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def read_frame(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot decode frame {index}")
    return frame


def compare_endpoint(source_cap, clip_cap, local_frame, source_frame):
    target = thumbnail(read_frame(clip_cap, local_frame))
    source_cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame - 3)
    errors = {}
    for offset in range(-3, 4):
        ok, original = source_cap.read()
        if not ok:
            raise RuntimeError(f"Cannot decode source frame {source_frame + offset}")
        errors[str(offset)] = round(float(np.abs(target - thumbnail(original)).mean()), 6)
    best = min(errors, key=errors.get)
    return {"clip_frame": local_frame, "source_frame": source_frame,
            "mae_0_to_255": errors["0"], "best_offset": int(best),
            "neighbor_mae": errors,
            "passed": int(best) == 0 and errors["0"] < 8.0}


def clip_intervals(pipeline):
    """Old contacts locate camera scenes only; they never become annotation data."""
    scenes = read_csv(pipeline / "scenes/scenes.csv")
    inventory = {int(r["point"]): r for r in read_csv(
        pipeline.parent / "match2_point_inventory_20261117/point_inventory.csv")}
    for point in range(7, 12):
        point_scenes = [s for s in scenes if int(s["point"]) == point]
        frames = [int(r["contact_frame"]) for r in read_csv(
            pipeline / f"labels/point_{point:02d}.csv")]
        selected = [s for s in point_scenes if s["is_main_camera"].lower() == "true"
                    and any(int(s["start_frame"]) <= f <= int(s["end_frame"])
                            for f in frames)]
        assert selected, f"No main-camera scene containing old labels: point {point}"
        assert all(any(int(s["start_frame"]) <= f <= int(s["end_frame"])
                       for s in selected) for f in frames), (point, "Label outside selected scenes")
        if point == 10:
            assert {s["scene_id"] for s in selected} == {"p10_s009"}
            preceding = next(s for s in point_scenes if s["scene_id"] == "p10_s008")
            assert preceding["is_main_camera"] == "uncertain"
            selected.append(preceding)
        selected.sort(key=lambda s: int(s["start_frame"]))
        start = min(int(s["start_frame"]) for s in selected)
        end = max(int(s["end_frame"]) for s in selected) + 1  # scenes.csv is inclusive
        point_start = int(inventory[point]["display_from_frame"])
        point_stop = int(inventory[point]["next_display_from_frame"])
        assert point_start <= start < end <= point_stop
        yield point, selected, start, end, point_start, point_stop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", type=Path, default=Path(r"C:\ffmpeg\bin\ffmpeg.exe"))
    parser.add_argument("--force", action="store_true", help="Re-encode existing clips")
    args = parser.parse_args()
    pipeline = HERE.parent
    protocol = json.loads((pipeline / "tracking/point_07/protocol.json").read_text(encoding="utf-8"))
    source = Path(protocol["source_video"])
    ffprobe = args.ffmpeg.with_name("ffprobe.exe")
    streams = probe(source, ffprobe)
    source_info = next(s for s in streams if s["codec_type"] == "video")
    assert Fraction(source_info["avg_frame_rate"]) == FPS, source_info
    assert abs(float(source_info.get("start_time", 0))) < 1e-6
    source_frames = int(source_info["nb_frames"])
    source_cap = cv2.VideoCapture(str(source))
    if not source_cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    (HERE / "clips").mkdir(exist_ok=True)
    candidates_v1 = read_csv(pipeline / "audio/candidates.csv")
    v2_path = pipeline / "audio/candidates_v2.csv"
    candidates_v2 = read_csv(v2_path) if v2_path.exists() else []
    records, checks = [], []
    for point, selected, start, end, point_start, point_stop in clip_intervals(pipeline):
        first, stop = max(point_start, start - 30), min(point_stop, end + 30)
        assert 0 <= first < stop <= source_frames
        count = stop - first
        clip = HERE / "clips" / f"point_{point:02d}.mp4"
        seek = f"{float(Fraction(first, 1) / FPS):.12f}"
        duration = f"{float(Fraction(count, 1) / FPS):.12f}"
        attempts = []
        for seek_mode in ("input", "output"):
            if args.force or not clip.exists() or seek_mode == "output":
                command = [str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-y"]
                if seek_mode == "input":
                    command += ["-ss", seek]
                command += ["-i", str(source)]
                if seek_mode == "output":
                    command += ["-ss", seek]
                command += ["-map", "0:v:0", "-map", "0:a:0", "-t", duration,
                            "-frames:v", str(count), "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "23", "-pix_fmt", "yuv420p", "-r", str(FPS),
                            "-fps_mode", "cfr", "-video_track_timescale", "60000",
                            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(clip)]
                print(f"Encoding {clip.name} ({seek_mode} -ss)", flush=True)
                subprocess.run(command, check=True)
            out_streams = probe(clip, ffprobe)
            video = next(s for s in out_streams if s["codec_type"] == "video")
            audio = next(s for s in out_streams if s["codec_type"] == "audio")
            assert video["codec_name"] == "h264" and video["pix_fmt"] == "yuv420p"
            assert audio["codec_name"] == "aac"
            assert Fraction(video["avg_frame_rate"]) == FPS
            assert int(video["nb_frames"]) == count, (clip, video["nb_frames"], count)
            assert abs(float(video.get("start_time", 0))) < 1e-6
            clip_cap = cv2.VideoCapture(str(clip))
            try:
                endpoints = [compare_endpoint(source_cap, clip_cap, 0, first),
                             compare_endpoint(source_cap, clip_cap, count - 1, stop - 1)]
            finally:
                clip_cap.release()
            attempts.append({"seek_mode": seek_mode, "endpoints": endpoints})
            print(f"{clip.name}: " + json.dumps(endpoints), flush=True)
            if all(e["passed"] for e in endpoints):
                break
        check = {"file": clip.name, "source_first_frame": first,
                 "source_end_frame_exclusive": stop, "frame_count": count, "attempts": attempts,
                 "passed": all(e["passed"] for e in endpoints)}
        checks.append(check)
        (HERE / "verification.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8", newline="\n")
        assert check["passed"], f"Frame mismatch: {clip}; see verification.json"

        def candidates(rows, kept_only=False):
            return [{"id": r["candidate_id"], "frame": int(r["candidate_frame"]),
                     "strength": float(r["strength"])} for r in rows
                    if int(r["point"]) == point
                    and (not kept_only or r["kept"].lower() == "true")
                    and first <= int(r["candidate_frame"]) < stop]

        records.append({"id": f"{point:02d}", "point": point, "rally": "all",
                        "file": "clips/" + clip.name, "source_first_frame": first,
                        "frame_count": count, "fps": float(FPS), "fps_num": FPS.numerator,
                        "fps_den": FPS.denominator, "interval_start_frame": start,
                        "interval_end_frame_exclusive": end,
                        "point_start_frame": point_start, "point_end_frame_exclusive": point_stop,
                        "scenes": [{"id": s["scene_id"], "start_frame": int(s["start_frame"]),
                                    "end_frame_exclusive": int(s["end_frame"]) + 1,
                                    "is_main_camera": s["is_main_camera"]} for s in selected],
                        "candidates": {"v1": candidates(candidates_v1),
                                       "v2": candidates(candidates_v2, True)}})
    source_cap.release()
    assert [r["id"] for r in records] == ["07", "08", "09", "10", "11"]
    payload = {"schema_version": 2, "generated_at": datetime.now(timezone.utc).isoformat(),
               "source_video": str(source), "interval_convention": "[start, end)",
               "v2_available": v2_path.exists(), "clips": records}
    (HERE / "data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    # Remove only the six known obsolete clips, after all new clips have passed.
    for name in ["point_07_rally_A.mp4", *[f"point_{p:02d}_rally_B.mp4" for p in range(7, 12)]]:
        (HERE / "clips" / name).unlink(missing_ok=True)
    print(f"PASS: {len(records)} clips, {2 * len(records)} endpoints; data.json written", flush=True)


if __name__ == "__main__":
    main()
