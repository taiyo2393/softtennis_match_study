"""Extract YOLO11 pose in tracked-player crops; writes only below this directory.

Frame numbers are zero based and intervals are half open.  This deliberately
does not track/relabel people or repair the known point-8 tracking failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent
CACHE = HERE / "pose_cache"
# Must be set BEFORE importing ultralytics (including indirect imports).
os.environ["YOLO_CONFIG_DIR"] = str(CACHE / "ultralytics_config")

import cv2
import numpy as np
import pandas as pd

SLOTS = ["near_front", "near_rear", "far_rear", "far_front"]
DEFAULT_MODEL = Path("C:/Users/ta239/Documents/Codex/2026-09-07/ai-2d-2d-1-2-0/softtennis_frontcourt_study/models/yolo11m-pose.pt")
KP_FIELDS = [f"kp_{i}_{field}" for i in range(17) for field in ("x", "y", "conf")]
FIELDS = ["point", "rally", "source_frame", "player_slot", "side", "track_id",
          "box_valid", "detection_valid", "status", "box_x1", "box_y1", "box_x2", "box_y2",
          "crop_x1", "crop_y1", "crop_x2", "crop_y2", "resized_width", "resized_height",
          "resize_scale_x", "resize_scale_y", "det_conf", "det_x1", "det_y1", "det_x2", "det_y2",
          "person_count", "selection_iou", "selection_center_distance"] + KP_FIELDS


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path, table):
    temp = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)


def base_row(point, rally, frame, slot, track):
    row = dict.fromkeys(FIELDS, np.nan)
    row.update(point=point, rally=rally, source_frame=frame, player_slot=slot,
               side="N" if slot.startswith("near") else "F", box_valid=False,
               detection_valid=False, status="no_tracking_box", person_count=0)
    if track is not None:
        row["track_id"] = track.track_id
        for key in ("x1", "y1", "x2", "y2"):
            row[f"box_{key}"] = float(getattr(track, key))
    return row


def make_crop(raw, row, pad_x, pad_y, size):
    """Padding is PER SIDE: .6*w left/right, .25*h above/below by default."""
    box = np.array([row[f"box_{k}"] for k in ("x1", "y1", "x2", "y2")])
    if not np.isfinite(box).all():
        row["status"] = "invalid_tracking_box"
        return None
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        row["status"] = "invalid_tracking_box"
        return None
    height, width = raw.shape[:2]
    left, right = max(0, math.floor(x1-pad_x*w)), min(width, math.ceil(x2+pad_x*w))
    top, bottom = max(0, math.floor(y1-pad_y*h)), min(height, math.ceil(y2+pad_y*h))
    if right <= left or bottom <= top:
        row["status"] = "empty_crop"
        return None
    cw, ch = right-left, bottom-top
    scale = size / max(cw, ch)
    rw, rh = max(1, round(cw*scale)), max(1, round(ch*scale))
    row.update(box_valid=True, status="no_pose_detection", crop_x1=left, crop_y1=top,
               crop_x2=right, crop_y2=bottom, resized_width=rw, resized_height=rh,
               resize_scale_x=rw/cw, resize_scale_y=rh/ch)
    return cv2.resize(raw[top:bottom, left:right], (rw, rh),
                      interpolation=cv2.INTER_LINEAR if scale >= 1 else cv2.INTER_AREA)


def assign_detection(result, row):
    if result.boxes is None or len(result.boxes) == 0:
        return
    assert result.keypoints is not None, "Model must output pose keypoints"
    dets = result.boxes.xyxy.cpu().numpy().astype(float)
    confs = result.boxes.conf.cpu().numpy().astype(float)
    sx, sy = row["resize_scale_x"], row["resize_scale_y"]
    dets[:, [0, 2]] = dets[:, [0, 2]]/sx + row["crop_x1"]
    dets[:, [1, 3]] = dets[:, [1, 3]]/sy + row["crop_y1"]
    box = np.array([row[f"box_{k}"] for k in ("x1", "y1", "x2", "y2")])
    wh = np.maximum(0, np.minimum(dets[:, 2:], box[2:])-np.maximum(dets[:, :2], box[:2]))
    overlap = wh.prod(axis=1)
    union = (dets[:, 2:]-dets[:, :2]).prod(axis=1) + (box[2:]-box[:2]).prod() - overlap
    iou = overlap/np.maximum(union, 1e-12)
    # Tie break by distance to the TRACKED box center, then confidence and index.
    centers = (dets[:, :2]+dets[:, 2:])/2
    distance = np.linalg.norm((centers-(box[:2]+box[2:])/2)/(box[2:]-box[:2]), axis=1)
    selected = min(range(len(dets)), key=lambda i: (-iou[i], distance[i], -confs[i], i))
    kp = result.keypoints.data[selected].cpu().numpy().astype(float)
    assert kp.shape == (17, 3), f"Expected COCO17 x/y/conf, got {kp.shape}"
    for i, (x, y, conf) in enumerate(kp):
        # Preserve all model confidence scores. Low-confidence coordinates are
        # masked by combine_audio_pose, not silently replaced with another person.
        row[f"kp_{i}_x"] = x/sx + row["crop_x1"]
        row[f"kp_{i}_y"] = y/sy + row["crop_y1"]
        row[f"kp_{i}_conf"] = conf
    for j, key in enumerate(("x1", "y1", "x2", "y2")):
        row[f"det_{key}"] = dets[selected, j]
    row.update(detection_valid=True, status="detected", det_conf=confs[selected],
               person_count=len(dets), selection_iou=iou[selected],
               selection_center_distance=distance[selected])


def summarize(table, expected_frames, complete):
    rows = []
    for slot, group in table.groupby("player_slot", sort=False):
        row = dict(point=int(group.point.iloc[0]), rally=str(group.rally.iloc[0]), player_slot=slot,
                   expected_frames=expected_frames, processed_frames=len(group), complete=complete,
                   n_box=int(group.box_valid.sum()), n_detected=int(group.detection_valid.sum()),
                   detection_rate_processed=float(group.detection_valid.mean()),
                   detection_rate_given_box=float(group.detection_valid.sum()/group.box_valid.sum()) if group.box_valid.sum() else np.nan,
                   known_tracking_error=bool(int(group.point.iloc[0]) == 8 and slot.startswith("near")))
        for name, i in (("left_wrist", 9), ("right_wrist", 10)):
            values = group.loc[group.detection_valid, f"kp_{i}_conf"].dropna()
            row[f"{name}_n_conf_ge_0_3"] = int((values >= .3).sum())
            row[f"{name}_valid_rate_processed"] = float((values >= .3).sum()/len(group))
            for q, label in ((.1, "p10"), (.25, "p25"), (.5, "p50"), (.75, "p75"), (.9, "p90")):
                row[f"{name}_conf_{label}"] = float(values.quantile(q)) if len(values) else np.nan
        rows.append(row)
    return rows


def extract_interval(model, args, interval, tracks, video, fps, torch):
    first, expected_end = int(interval.start_frame), int(interval.end_frame)
    end = min(expected_end, first+args.limit_frames) if args.limit_frames else expected_end
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened(), f"Cannot open {video}"
    assert abs(capture.get(cv2.CAP_PROP_FPS)-fps) < .001
    started = time.perf_counter()
    infer_sec, decode_sec, n_crops, batches = 0., 0., 0, 0
    rows, pending_images, pending_rows = [], [], []

    def flush():
        nonlocal infer_sec, n_crops, batches
        if not pending_images:
            return
        # Explicit synchronization includes GPU execution rather than launch time.
        if args.device != "cpu":
            torch.cuda.synchronize(device=int(args.device))
        tick = time.perf_counter()
        results = model.predict(source=pending_images, device=args.device, imgsz=args.crop_size,
                                conf=args.det_conf, iou=.7, max_det=10, rect=False,
                                batch=args.batch_size, verbose=False, save=False, stream=False,
                                project=str(CACHE/"predict_runs"), name="pose", exist_ok=True)
        if args.device != "cpu":
            torch.cuda.synchronize(device=int(args.device))
        infer_sec += time.perf_counter()-tick
        assert len(results) == len(pending_rows)
        for result, row in zip(results, pending_rows):
            assign_detection(result, row)
        n_crops += len(results)
        batches += 1
        pending_images.clear()
        pending_rows.clear()

    try:
        assert capture.set(cv2.CAP_PROP_POS_FRAMES, first), f"Cannot seek to F{first}"
        assert round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == first
        for frame in range(first, end):
            tick = time.perf_counter()
            assert round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == frame
            assert capture.grab(), f"Cannot grab F{frame}"
            assert round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == frame+1
            ok, raw = capture.retrieve()
            assert ok and raw is not None, f"Cannot retrieve F{frame}"
            assert round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == frame+1
            assert raw.shape[:2] == (1080, 1920), f"Tracking requires 1920x1080: {raw.shape}"
            decode_sec += time.perf_counter()-tick
            for slot in SLOTS:
                track = tracks.get((frame, slot))
                row = base_row(int(interval.point), str(interval.rally), frame, slot, track)
                rows.append(row)
                if track is not None:
                    crop = make_crop(raw, row, args.pad_x, args.pad_y, args.crop_size)
                    if crop is not None:
                        pending_images.append(crop)
                        pending_rows.append(row)
                        if len(pending_images) >= args.batch_size:
                            flush()
        flush()
    finally:
        capture.release()
    table = pd.DataFrame(rows, columns=FIELDS)
    assert len(table) == 4*(end-first)
    assert not table.duplicated(["source_frame", "player_slot"]).any()
    assert table.groupby("source_frame").size().eq(4).all()
    filename = f"point_{int(interval.point):02d}_rally_{interval.rally}.csv"
    write_csv(CACHE/filename, table)
    complete = end == expected_end
    elapsed = time.perf_counter()-started
    timing = dict(point=int(interval.point), rally=str(interval.rally), device=args.device,
                  start_frame=first, processed_end_frame=end, end_frame=expected_end,
                  expected_frames=expected_end-first, processed_frames=end-first, complete=complete,
                  crop_count=n_crops, batches=batches, seek_calls=1, elapsed_sec=elapsed,
                  decode_sec=decode_sec, inference_sec=infer_sec,
                  frames_per_sec=(end-first)/elapsed, crops_per_sec=n_crops/elapsed)
    manifest = dict(point=int(interval.point), rally=str(interval.rally), start_frame=first,
                    end_frame=expected_end, processed_end_frame=end, processed_frames=end-first,
                    expected_frames=expected_end-first, complete=complete,
                    cache_file=f"pose_cache/{filename}", cache_sha256=sha(CACHE/filename))
    print(f"point {interval.point:02d} rally {interval.rally}: {end-first}/{expected_end-first} frames, "
          f"{n_crops} crops, {int(table.detection_valid.sum())} detections, {elapsed:.2f}s", flush=True)
    return manifest, summarize(table, expected_end-first, complete), timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", help="CUDA index (default 0), or cpu")
    parser.add_argument("--limit-frames", type=int, default=0, metavar="N", help="First N frames of EACH interval; 0=full")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32, help="Crops per predict call, can span adjacent frames")
    parser.add_argument("--crop-size", type=int, default=384, help="Longest resized crop side; model input is padded square")
    parser.add_argument("--pad-x", type=float, default=.6, help="Fraction of box width ADDED ON EACH horizontal side")
    parser.add_argument("--pad-y", type=float, default=.25, help="Fraction of box height ADDED ON EACH vertical side")
    parser.add_argument("--det-conf", type=float, default=.25)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.limit_frames < 0 or args.batch_size < 1 or args.crop_size < 32 or args.crop_size % 32:
        parser.error("limit-frames>=0, batch-size>=1, crop-size must be a positive multiple of 32")
    if min(args.pad_x, args.pad_y) < 0 or not 0 < args.det_conf < 1 or args.cpu_threads < 1:
        parser.error("Invalid padding, detection threshold or CPU thread count")
    if args.device != "cpu" and not args.device.isdigit():
        parser.error("--device must be cpu or one CUDA index, e.g. 0")
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE/"ultralytics_config").mkdir(parents=True, exist_ok=True)
    # Invalidate old extraction manifest before writing any new cache files.
    # A failed extraction must never leave a valid-looking old manifest.
    manifest_path = HERE/"extraction_manifest.json"
    write_json(manifest_path, dict(status="running", full_run=False, intervals=[]))
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    import torch
    import ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(1)
    if args.device != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Retry with --device cpu --limit-frames 30")
        torch.empty(1, device=f"cuda:{args.device}").add_(1)
        torch.cuda.synchronize(device=int(args.device))
    tick = time.perf_counter()
    model = YOLO(str(args.model), task="pose")
    model_load_sec = time.perf_counter()-tick
    intervals = pd.read_csv(PIPE/"audio/intervals.csv")
    assert len(intervals) == 6 and not intervals.duplicated(["point", "rally"]).any()
    protocol_path = PIPE/"tracking/point_07/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    video, fps = Path(protocol["source_video"]), float(Fraction(protocol["source_fps"]))
    video_stat_before = (video.stat().st_size, video.stat().st_mtime_ns)
    source_paths = [PIPE/"audio/intervals.csv", protocol_path]
    tracks_by_point = {}
    for point in sorted(intervals.point.unique()):
        path = PIPE/f"tracking/point_{point:02d}/player_court_proxies.csv"
        source_paths.append(path)
        table = pd.read_csv(path)
        assert table.point.eq(point).all() and table.player_slot.isin(SLOTS).all()
        assert not table.duplicated(["source_frame", "player_slot"]).any()
        tracks_by_point[point] = {(int(r.source_frame), r.player_slot): r for r in table.itertuples()}
    source_hashes = {str(p.relative_to(PIPE).as_posix()): sha(p) for p in source_paths}
    model_hash = sha(args.model)
    all_started = time.perf_counter()
    summaries, timings, manifests = [], [], []
    for interval in intervals.itertuples():
        manifest, summary, timing = extract_interval(model, args, interval, tracks_by_point[interval.point], video, fps, torch)
        manifests.append(manifest)
        summaries.extend(summary)
        timings.append(timing)
    write_csv(HERE/"pose_summary.csv", pd.DataFrame(summaries))
    write_csv(HERE/"pose_timing.csv", pd.DataFrame(timings))
    elapsed = time.perf_counter()-all_started
    assert source_hashes == {str(p.relative_to(PIPE).as_posix()): sha(p) for p in source_paths}, "Input changed during extraction"
    assert model_hash == sha(args.model), "Read-only model changed"
    assert video_stat_before == (video.stat().st_size, video.stat().st_mtime_ns), "Source video changed"
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(manifest_path, dict(status="complete", full_run=all(m["complete"] for m in manifests),
                                  generated_utc=datetime.now(timezone.utc).isoformat(), config=config,
                                  intervals=manifests, input_sha256=source_hashes,
                                  source_video=str(video), source_fps=fps,
                                  video_bytes=video.stat().st_size, video_mtime_ns=video.stat().st_mtime_ns,
                                  model_sha256=model_hash, model_load_sec=model_load_sec,
                                  elapsed_sec=elapsed, script_sha256=sha(Path(__file__)),
                                  versions=dict(torch=torch.__version__, ultralytics=ultralytics.__version__,
                                                opencv=cv2.__version__, pandas=pd.__version__, numpy=np.__version__),
                                  crop_padding="Each horizontal side +0.6*width; each vertical side +0.25*height (CLI configurable)",
                                  selection="max IoU with tracked box; tie: normalized box-center distance then confidence"))
    print(f"extraction complete: full_run={all(m['complete'] for m in manifests)}, {elapsed:.2f}s; {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
