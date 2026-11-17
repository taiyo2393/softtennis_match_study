"""CPU-only audiovisual proposal filtering, with train/freeze/test separation.

Run with study/.venv/Scripts/python.exe -B -X utf8 <this file>.
Only files inside this directory are written. Source frames are zero-based.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, signal
from scipy.io import wavfile

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent
SLOTS = ["near_front", "near_rear", "far_rear", "far_front"]
COLORS = ["#16804a", "#238bce", "#ac4cb0", "#c79500"]
SIDES = {s: "N" if s.startswith("near") else "F" for s in SLOTS}
METHODS = ["audio_v1", "audio_v2", "audio_v1_A", "audio_v1_A_B"]
TRAIN, TEST = [7, 8], [9, 10, 11]
PEAK_RADIUS, BASE_RADIUS, MIN_GAP = 8, 30, 12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(name, table):
    table.to_csv(HERE / name, index=False, encoding="utf-8-sig")


def input_paths():
    return ([PIPE / "audio" / x for x in ["candidate_features_v2.csv", "candidates.csv", "candidates_v2.csv", "intervals.csv"]]
            + [PIPE / "labels" / f"point_{p:02d}.csv" for p in TRAIN + TEST]
            + [PIPE / "tracking" / f"point_{p:02d}" / name for p in TRAIN + TEST
               for name in ["protocol.json", "player_court_proxies.csv"]])


def training_provenance(manifest, video):
    """Bind frozen weights to train inputs; do not inspect held-out label values."""
    selected = {k: v for k, v in manifest.items() if k.startswith("audio/") or any(f"point_{p:02d}" in k for p in TRAIN)}
    return dict(input_sha256=selected, video_bytes=video.stat().st_size, video_mtime_ns=video.stat().st_mtime_ns)


def load_inputs():
    intervals = pd.read_csv(PIPE / "audio/intervals.csv")
    candidates = pd.read_csv(PIPE / "audio/candidate_features_v2.csv")
    required = {"candidate_id", "point", "rally", "candidate_frame", "strength", "rule_margin", "kept"}
    assert required <= set(candidates.columns), required - set(candidates.columns)
    assert candidates.candidate_id.is_unique
    assert candidates.kept.isin([True, False]).all()
    # Refuse an accidental replacement of the high-recall proposal population.
    keys = ["candidate_id", "point", "rally", "candidate_frame"]
    baseline = pd.read_csv(PIPE / "audio/candidates.csv")
    pd.testing.assert_frame_equal(candidates[keys].reset_index(drop=True), baseline[keys].reset_index(drop=True))
    v2 = pd.read_csv(PIPE / "audio/candidates_v2.csv")
    pd.testing.assert_frame_equal(candidates.loc[candidates.kept, keys].reset_index(drop=True), v2[keys].reset_index(drop=True))
    proto = json.loads((PIPE / "tracking/point_07/protocol.json").read_text(encoding="utf-8-sig"))
    fps = float(Fraction(proto["source_fps"]))
    for row in intervals.itertuples():
        assert math.isclose(row.duration_sec, (row.end_frame-row.start_frame)/fps, abs_tol=1e-9)
        c = candidates[(candidates.point == row.point) & (candidates.rally == row.rally)]
        assert c.candidate_frame.between(row.start_frame, row.end_frame-1).all()
    assert len(candidates) == sum(len(candidates[(candidates.point == r.point) & (candidates.rally == r.rally)]) for r in intervals.itertuples())
    return intervals, candidates, Path(proto["source_video"]), fps


def load_labels(points, intervals):
    tables = []
    for p in points:
        table = pd.read_csv(PIPE / f"labels/point_{p:02d}.csv")
        assert table.hitter_side.isin(["N", "F"]).all()
        assert table.confidence.isin(["clear", "inferred"]).all()
        table["label_id"] = [f"p{p:02d}_{r.rally}_{r.shot_index:03d}" for r in table.itertuples()]
        tables.append(table)
    labels = pd.concat(tables, ignore_index=True)
    assert labels.label_id.is_unique
    for r in intervals[intervals.point.isin(points)].itertuples():
        lab = labels[(labels.point == r.point) & (labels.rally == r.rally)]
        assert lab.contact_frame.between(r.start_frame, r.end_frame-1).all()
        assert len(lab) == r.n_labels_all
        assert (lab.confidence == "clear").sum() == r.n_labels_clear
    return labels


def extract_motion(points, intervals, video, fps):
    """One seek at each analysis interval start; no per-frame or per-candidate seeks."""
    rows, diagnostics = [], []
    cv2.setNumThreads(1)
    for p in points:
        tracking = pd.read_csv(PIPE / f"tracking/point_{p:02d}/player_court_proxies.csv")
        assert not tracking.duplicated(["source_frame", "player_slot"]).any()
        assert tracking.player_slot.isin(SLOTS).all()
        boxes = {(int(r.source_frame), r.player_slot): (r.x1, r.y1, r.x2, r.y2) for r in tracking.itertuples()}
        for interval in intervals[intervals.point == p].itertuples():
            started = time.perf_counter()
            first, end = int(interval.start_frame), int(interval.end_frame)
            capture = cv2.VideoCapture(str(video))
            assert capture.isOpened(), str(video)
            assert abs(capture.get(cv2.CAP_PROP_FPS)-fps) < .001
            assert capture.set(cv2.CAP_PROP_POS_FRAMES, first)
            assert round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == first
            previous = None
            interval_rows = []
            try:
                for frame in range(first, end):
                    assert round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == frame
                    assert capture.grab(), f"grab failed F{frame}"
                    assert round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == frame+1
                    ok, raw = capture.retrieve()
                    assert ok and raw is not None, f"retrieve failed F{frame}"
                    assert round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == frame+1
                    assert raw.shape[:2] == (1080, 1920), raw.shape
                    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                    difference = cv2.absdiff(gray, previous) if previous is not None else None
                    for slot in SLOTS:
                        value, area = np.nan, 0
                        box = boxes.get((frame, slot))
                        if box is not None and np.isfinite(box).all():
                            x1, y1, x2, y2 = box
                            w, h = x2-x1, y2-y1
                            assert w > 0 and h > 0
                            # Total width +60% (30% on each side), top +30% height.
                            left, right = max(0, math.floor(x1-.3*w)), min(1920, math.ceil(x2+.3*w))
                            top, bottom = max(0, math.floor(y1-.3*h)), min(1080, math.ceil(y2))
                            if right > left and bottom > top:
                                area = (right-left)*(bottom-top)
                                if difference is not None:
                                    value = float(difference[top:bottom, left:right].mean())
                        interval_rows.append(dict(point=p, rally=interval.rally, source_frame=frame,
                                                  player_slot=slot, motion_raw=value, roi_area_px=area,
                                                  bbox_present=box is not None))
                    previous = gray
            finally:
                capture.release()
            motion = pd.DataFrame(interval_rows)
            for slot in SLOTS:
                take = motion.player_slot == slot
                median = float(motion.loc[take, "motion_raw"].median())
                # A flat or wholly missing interval cannot provide a comparable ratio.
                motion.loc[take, "interval_motion_median"] = median
                motion.loc[take, "motion_normalized"] = motion.loc[take, "motion_raw"] / median if np.isfinite(median) and median > 0 else np.nan
                diagnostics.append(dict(point=p, rally=interval.rally, player_slot=slot,
                                        start_frame=first, end_frame_exclusive=end, n_frames=end-first,
                                        n_valid=int(motion.loc[take, "motion_normalized"].notna().sum()),
                                        n_bbox=int(motion.loc[take, "bbox_present"].sum()),
                                        interval_motion_median=median, seek_calls=1,
                                        first_difference_missing=True))
            rows.append(motion)
            print(f"motion point {p:02d} rally {interval.rally}: {end-first} sequential frames, {time.perf_counter()-started:.2f}s", flush=True)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(diagnostics)


def candidate_features(candidates, motion):
    table = candidates.copy()
    for slot in SLOTS:
        table[f"{slot}_peak_height"] = np.nan
        table[f"{slot}_peak_frame"] = np.nan
        table[f"{slot}_local_median"] = np.nan
        table[f"{slot}_peak_n_valid"] = 0
    for (point, rally), group in motion.groupby(["point", "rally"]):
        pivot = group.pivot(index="source_frame", columns="player_slot", values="motion_normalized")
        for idx, candidate in table[(table.point == point) & (table.rally == rally)].iterrows():
            f = int(candidate.candidate_frame)
            for slot in SLOTS:
                series = pivot[slot]
                local = series.loc[f-BASE_RADIUS:f+BASE_RADIUS].dropna()
                peak = series.loc[f-PEAK_RADIUS:f+PEAK_RADIUS].dropna()
                table.loc[idx, f"{slot}_peak_n_valid"] = len(peak)
                if len(peak) and len(local):
                    median = float(local.median())
                    table.loc[idx, f"{slot}_local_median"] = median
                    table.loc[idx, f"{slot}_peak_height"] = float(peak.max()-median)
                    table.loc[idx, f"{slot}_peak_frame"] = int(peak.idxmax())
    peaks = table[[f"{slot}_peak_height" for slot in SLOTS]].to_numpy(float)
    table["motion_peak_height"] = np.nan
    table["visual_hitter_slot"] = ""
    table["visual_side"] = ""
    table["side_reason"] = "all_missing_or_nonpositive"
    for i, values in enumerate(peaks):
        if np.isfinite(values).any():
            maximum = float(np.nanmax(values))
            table.loc[table.index[i], "motion_peak_height"] = maximum
            winners = np.flatnonzero(np.isclose(values, maximum, rtol=0, atol=1e-12))
            if maximum > 0 and len(winners) == 1:
                slot = SLOTS[int(winners[0])]
                table.loc[table.index[i], ["visual_hitter_slot", "visual_side", "side_reason"]] = [slot, SIDES[slot], "unique_positive_motion_peak"]
    return table


def match_labels(labels, candidates, tolerance):
    """Identical closest-first, all-label one-to-one convention to audio v1/v2."""
    pairs = []
    for lab in labels.itertuples():
        c = candidates[(candidates.point == lab.point) & (candidates.rally == lab.rally)]
        for cand in c.itertuples():
            delta = int(cand.candidate_frame-lab.contact_frame)
            if abs(delta) <= tolerance:
                pairs.append((abs(delta), int(lab.contact_frame), int(cand.candidate_frame), lab.label_id, cand.candidate_id, delta))
    used_l, used_c, matches = set(), set(), {}
    for _, _, _, lid, cid, delta in sorted(pairs):
        if lid not in used_l and cid not in used_c:
            matches[lid] = (cid, delta)
            used_l.add(lid)
            used_c.add(cid)
    return matches


def divide(a, b):
    return a/b if b else np.nan


def evaluate(intervals, labels, candidates, scope="", method=""):
    duration = float(intervals.duration_sec.sum())
    clear_ids = set(labels.loc[labels.confidence == "clear", "label_id"])
    result = dict(scope=scope, method=method, duration_sec=duration, n_candidates=len(candidates),
                  n_labels_all=len(labels), n_labels_clear=len(clear_ids), candidates_per_sec=len(candidates)/duration)
    matches = {}
    for tol in [6, 10]:
        matches[tol] = match_labels(labels, candidates, tol)
        nc = len(clear_ids & matches[tol].keys())
        result.update({f"n_matched_clear_f{tol}": nc, f"recall_clear_f{tol}": divide(nc, len(clear_ids)),
                       f"n_matched_all_f{tol}": len(matches[tol]), f"recall_all_f{tol}": divide(len(matches[tol]), len(labels)),
                       f"precision_one_to_one_f{tol}": divide(len(matches[tol]), len(candidates))})
    nearby = 0
    for cand in candidates.itertuples():
        lab = labels[(labels.point == cand.point) & (labels.rally == cand.rally)]
        nearby += bool((abs(lab.contact_frame-cand.candidate_frame) <= 10).any())
    result["n_candidates_near_label_f10"] = nearby
    result["precision_near_label_f10"] = divide(nearby, len(candidates))
    summed, union, total_frames = 0., 0., 0.
    for interval in intervals.itertuples():
        total_frames += interval.end_frame-interval.start_frame
        cs = candidates[(candidates.point == interval.point) & (candidates.rally == interval.rally)]
        last = interval.start_frame
        for f in sorted(cs.candidate_frame):
            lo, hi = max(interval.start_frame, f-10), min(interval.end_frame, f+10)
            summed += hi-lo
            union += max(0, hi-max(last, lo))
            last = max(last, hi)
    result.update(review_window_sum_ratio=summed/total_frames, review_union_ratio=union/total_frames,
                  review_window_sum_sec=summed/total_frames*duration, review_union_sec=union/total_frames*duration)
    by_c = candidates.set_index("candidate_id")
    by_l = labels.set_index("label_id")
    known, correct, details = 0, 0, []
    for lid, (cid, delta) in matches[10].items():
        visual = str(by_c.loc[cid, "visual_side"])
        known += visual in ["N", "F"]
        correct += visual == by_l.loc[lid, "hitter_side"]
    result.update(n_side_time_matched=len(matches[10]), n_side_known=known, n_side_correct=correct,
                  side_accuracy_f10=divide(correct, known), n_side_unknown_matched=len(matches[10])-known,
                  side_known_coverage_f10=divide(known, len(matches[10])))
    for lab in labels.itertuples():
        row = dict(scope=scope, method=method, label_id=lab.label_id, point=lab.point, rally=lab.rally,
                   contact_frame=lab.contact_frame, confidence=lab.confidence, hitter_side=lab.hitter_side)
        for tol in [6, 10]:
            found = matches[tol].get(lab.label_id)
            row[f"candidate_id_f{tol}"] = found[0] if found else ""
            row[f"delta_f{tol}"] = found[1] if found else np.nan
        cid = row["candidate_id_f10"]
        row["visual_side"] = by_c.loc[cid, "visual_side"] if cid else ""
        row["side_correct"] = (row["visual_side"] == lab.hitter_side) if row["visual_side"] in ["N", "F"] else np.nan
        details.append(row)
    return result, details


def fit_scales(table):
    transforms = {"audio": np.log1p(table.strength.clip(lower=0)),
                  "motion": np.log1p(table.motion_peak_height.clip(lower=0))}
    return {key: float(values.median()) if values.median() > 0 else 1. for key, values in transforms.items()}


def add_scores(table, scales, config):
    table = table.copy()
    table["audio_score"] = np.log1p(table.strength.clip(lower=0))/scales["audio"]
    table["motion_score"] = (np.log1p(table.motion_peak_height.clip(lower=0))/scales["motion"]).fillna(0.)
    # Rule margin is pre-existing, fixed on points 7/8. Bounded quality bonus.
    table["shape_score"] = 1/(1+np.exp(-table.rule_margin.clip(-20, 20)))
    table["combined_score"] = table.motion_score + config["audio_weight"]*table.audio_score + config["shape_weight"]*table.shape_score - config["threshold"]
    return table


def dp_select(table):
    """Maximum total node score, alternating sides, >=12F, optional empty path.

    Unknown visual sides have two states, but are still unknown for side scoring.
    Negative-scoring bridge nodes remain eligible: they can enable two positives.
    """
    selected = []
    for (_, _), group in table.groupby(["point", "rally"]):
        group = group.sort_values(["candidate_frame", "candidate_id"])
        nodes = []
        for row in group.itertuples():
            allowed = [row.visual_side] if row.visual_side in ["N", "F"] else ["N", "F"]
            for side in allowed:
                best, parent = float(row.combined_score), None
                for j, node in enumerate(nodes):
                    if node["side"] != side and row.candidate_frame-node["frame"] >= MIN_GAP:
                        value = node["value"]+float(row.combined_score)
                        if value > best+1e-12:
                            best, parent = value, j
                nodes.append(dict(cid=row.candidate_id, frame=row.candidate_frame, side=side, value=best, parent=parent))
        if nodes:
            j = max(range(len(nodes)), key=lambda k: nodes[k]["value"])
            if nodes[j]["value"] > 0:
                path = []
                while j is not None:
                    node = nodes[j]
                    path.append((node["cid"], node["side"]))
                    j = node["parent"]
                selected.extend(reversed(path))
    sides = dict(selected)
    result = table[table.candidate_id.isin(sides)].copy()
    result["dp_assigned_side"] = result.candidate_id.map(sides)
    return result


def select_method(features, scales, config, method):
    if method == "audio_v1":
        return features.copy()
    if method == "audio_v2":
        return features[features.kept].copy()
    scored = add_scores(features, scales, config)
    return scored[scored.combined_score > 0].copy() if method == "audio_v1_A" else dp_select(scored)


def train(features, intervals, labels):
    scales = fit_scales(features)
    base, _ = evaluate(intervals, labels, features)
    floors = {"n_matched_clear_f10": math.ceil(.9*base["n_matched_clear_f10"]),
              "n_matched_all_f10": math.ceil(.9*base["n_matched_all_f10"])}
    grid = dict(audio_weight=[.25, .5, 1.], shape_weight=[0., .5, 1.],
                threshold=[round(x, 2) for x in np.arange(.25, 4.01, .25)])
    searches, chosen = [], {}
    for method in METHODS[2:]:
        results = []
        for number, values in enumerate(itertools.product(*grid.values())):
            config = dict(zip(grid, values))
            selected = select_method(features, scales, config, method)
            metrics, _ = evaluate(intervals, labels, selected)
            deficit = sum(max(0, floor-metrics[key])/floor for key, floor in floors.items())
            row = dict(method=method, config_id=number, **config, **{k: v for k, v in metrics.items() if k not in ["method"]},
                       recall_floor_deficit=deficit, feasible=deficit == 0)
            results.append(row)
        # A failed floor is recorded, never relaxed silently after seeing test.
        best = min(results, key=lambda r: (r["recall_floor_deficit"], r["review_window_sum_ratio"],
                                          -r["n_matched_clear_f6"], -r["n_matched_all_f10"], r["config_id"]))
        chosen[method] = {key: best[key] for key in ["config_id", "audio_weight", "shape_weight", "threshold", "feasible", "recall_floor_deficit"]}
        searches.extend(results)
    config = dict(feature_definition=dict(width_multiplier=1.6, upward_fraction=.3, peak_radius_frames=PEAK_RADIUS,
                                         baseline_radius_frames=BASE_RADIUS, min_gap_frames=MIN_GAP,
                                         missing_motion_score=0., unknown_side="both DP states; excluded from visual-side accuracy"),
                  train_points=TRAIN, test_points=TEST, train_includes_point7_A=True, scales=scales, grid=grid,
                  recall_floor_fraction_of_v1=.9, recall_floors=floors,
                  selection_order=["min normalized recall-floor deficit (zero required if feasible)", "min summed review-window ratio", "max clear +/-6F matches", "max all +/-10F matches", "config ID"],
                  selected=chosen, test_result_used_for_tuning=False,
                  test_labels_loaded_before_freeze=False)
    write_csv("parameter_search_train.csv", pd.DataFrame(searches))
    write_json(HERE / "selected_config.json", config)
    print("FROZEN train-only config: " + json.dumps(chosen), flush=True)
    return config


def scope_masks(intervals):
    result = [("train_07_08_including_7A", intervals.point.isin(TRAIN)),
              ("train_07_08_excluding_7A", intervals.point.isin(TRAIN) & ~((intervals.point == 7) & (intervals.rally == "A"))),
              ("test_09_11", intervals.point.isin(TEST)),
              ("all_07_11_including_7A", intervals.point.isin(TRAIN+TEST)),
              ("all_07_11_excluding_7A", ~((intervals.point == 7) & (intervals.rally == "A")))]
    for p in TRAIN+TEST:
        result.append((f"point_{p:02d}_including_A" if p == 7 else f"point_{p:02d}", intervals.point == p))
    result.extend([("point_07_excluding_A", (intervals.point == 7) & (intervals.rally != "A")),
                   ("point_07_A_only", (intervals.point == 7) & (intervals.rally == "A"))])
    return result


def mask_keys(table, keys):
    return pd.Series([(p, r) in keys for p, r in zip(table.point, table.rally)], index=table.index, dtype=bool)


def audio_plot_series(intervals, video, fps, candidates):
    """Reconstruct the previous 4-12kHz score for figures only, never selection.

    Decode audio only. Temporary WAVs are removed and never touch input folders.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is needed to reconstruct the audio plot (as in the previous audio pipeline)")
    outputs = []
    with tempfile.TemporaryDirectory(prefix="audiovisual_plot_") as scratch:
        for seg in intervals.itertuples():
            first = max(0, math.floor((seg.start_frame/fps-1.5)*48000))
            last = math.ceil((seg.end_frame/fps+1.5)*48000)
            wav = Path(scratch) / f"p{seg.point:02d}_{seg.rally}.wav"
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                            "-ss", f"{first/48000:.9f}", "-i", str(video), "-t", f"{(last-first)/48000:.9f}",
                            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(wav)],
                           check=True, capture_output=True)
            rate, audio = wavfile.read(wav)
            assert rate == 48000 and audio.ndim == 1 and abs(len(audio)-(last-first)) <= 2
            audio = audio.astype(float)/32768.
            filtered = signal.sosfiltfilt(signal.butter(4, (4000, 12000), btype="bandpass", fs=48000, output="sos"), audio)
            starts = np.arange(0, len(audio)-96+1, 48)
            sums = np.r_[0., np.cumsum(filtered*filtered)]
            log_energy = np.log(np.maximum((sums[starts+96]-sums[starts])/96, 1e-12))
            score = log_energy-ndimage.median_filter(log_energy, size=501, mode="reflect")
            frames = ((starts+95/2)/48000+first/48000)*fps
            # Check against the recorded candidate strengths; no new proposals.
            cs = candidates[(candidates.point == seg.point) & (candidates.rally == seg.rally)]
            for c in cs.itertuples():
                k = int(np.argmin(abs(frames-c.candidate_frame_float)))
                assert abs(frames[k]-c.candidate_frame_float) < .001
                assert abs(score[k]-c.strength) < 1e-7, "Reconstructed sound score differs from v1"
            keep = (frames >= seg.start_frame) & (frames < seg.end_frame)
            outputs.append(pd.DataFrame(dict(point=seg.point, rally=seg.rally, source_frame_float=frames[keep], audio_strength=score[keep])))
    result = pd.concat(outputs, ignore_index=True)
    write_csv("audio_plot_timeseries.csv", result)
    return result


def plot_points(intervals, labels, features, motion, selections, audio_series):
    for point in TRAIN+TEST:
        segments = intervals[intervals.point == point]
        fig, axes = plt.subplots(3, len(segments), figsize=(19, 10), squeeze=False, sharex="col", gridspec_kw={"height_ratios": [1, 1.6, 1.5]})
        for col, seg in enumerate(segments.itertuples()):
            c = features[(features.point == point) & (features.rally == seg.rally)]
            lab = labels[(labels.point == point) & (labels.rally == seg.rally)]
            m = motion[(motion.point == point) & (motion.rally == seg.rally)]
            ax = axes[0, col]
            audio = audio_series[(audio_series.point == point) & (audio_series.rally == seg.rally)]
            ax.plot(audio.source_frame_float, audio.audio_strength, color=".5", lw=.6, label="4-12 kHz log energy - local median")
            ax.axhline(1., color=".7", ls="--", lw=.6, label="v1 threshold")
            ax.scatter(c.candidate_frame, c.strength, s=20, marker="x", color=".4", label="v1 candidates")
            v2 = c[c.kept]
            ax.scatter(v2.candidate_frame, v2.strength, s=20, facecolors="none", edgecolors="#005fa8", label="v2 retained")
            ax.set_title(f"match2 point {point:02d}, rally {seg.rally}")
            ax.set_ylabel("Audio strength")
            ax = axes[1, col]
            for slot, color in zip(SLOTS, COLORS):
                mm = m[m.player_slot == slot]
                ax.plot(mm.source_frame, mm.motion_normalized, color=color, lw=.75, label=slot)
            ax.set_ylabel("ROI gray diff / interval median")
            ax = axes[2, col]
            for i, method in enumerate(METHODS):
                selected = selections[method]
                selected = selected[(selected.point == point) & (selected.rally == seg.rally)]
                for side, color in [("N", "#1976b9"), ("F", "#b3438f"), ("", ".5")]:
                    ss = selected[selected.visual_side == side]
                    ax.scatter(ss.candidate_frame, np.full(len(ss), i), marker="v", s=30, color=color)
            for row in lab.itertuples():
                color = "#1976b9" if row.hitter_side == "N" else "#b3438f"
                ax.axvline(row.contact_frame, color=color, ls="-" if row.confidence == "clear" else ":", lw=.8, alpha=.8)
                ax.text(row.contact_frame, 3.65, row.hitter_side, fontsize=7, color=color, ha="center")
            ax.set_yticks(range(4), ["v1", "v2", "+ A", "+ A + B"])
            ax.set_ylim(-.5, 4.05)
            ax.set_xlabel("Zero-based source frame")
            ax.set_ylabel("Selected candidates / labels")
            for ax in axes[:, col]:
                ax.set_xlim(seg.start_frame, seg.end_frame)
                ax.grid(alpha=.18)
                ax.ticklabel_format(axis="x", useOffset=False, style="plain")
            axes[0, col].legend(fontsize=7, loc="upper right")
            axes[1, col].legend(fontsize=8, loc="upper right", ncol=2)
        fig.suptitle("Labels: clear solid / inferred dotted. Candidate colors: visual side N blue, F purple, unknown gray.", fontsize=11)
        fig.tight_layout()
        fig.savefig(HERE / f"point_{point:02d}_audiovisual.png", dpi=160)
        plt.close(fig)


def pct(value):
    return f"{100*value:.1f}%" if np.isfinite(value) else "—"


def table_markdown(evaluation, scopes):
    lines = ["| 集合 | 方法 | clear ±6F | clear ±10F | 全 ±10F | 適合率目安 ±10F | 候補数 / 秒 | 窓合計 / 区間 | 和集合 / 区間 | 側正解率（正解/既知; 不明） |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    names = {"train_07_08_including_7A": "第7・8点（7A含む）", "train_07_08_excluding_7A": "第7・8点（7A除く）", "test_09_11": "第9〜11点"}
    for scope in scopes:
        for r in evaluation[evaluation.scope == scope].itertuples():
            lines.append(f"| {names.get(scope, scope)} | {r.method} | {r.n_matched_clear_f6}/{r.n_labels_clear} ({pct(r.recall_clear_f6)}) | {r.n_matched_clear_f10}/{r.n_labels_clear} ({pct(r.recall_clear_f10)}) | {r.n_matched_all_f10}/{r.n_labels_all} ({pct(r.recall_all_f10)}) | {pct(r.precision_near_label_f10)} | {r.n_candidates} / {r.candidates_per_sec:.3f} | {pct(r.review_window_sum_ratio)} | {pct(r.review_union_ratio)} | {pct(r.side_accuracy_f10)} ({r.n_side_correct}/{r.n_side_known}; {r.n_side_unknown_matched}) |")
    return "\n".join(lines)


def make_readme(evaluation, config, diagnostics, elapsed):
    test = evaluation[evaluation.scope == "test_09_11"].set_index("method")
    baseline = test.loc["audio_v1"]
    observations = []
    for method in METHODS[2:]:
        r = test.loc[method]
        observations.append(f"- `{method}`：v1の{int(baseline.n_candidates)}候補から{int(r.n_candidates)}候補（{100*(1-r.n_candidates/baseline.n_candidates):.1f}%減）。窓合計{baseline.review_window_sum_sec:.3f}→{r.review_window_sum_sec:.3f}秒。clear ±10Fは{int(baseline.n_matched_clear_f10)}→{int(r.n_matched_clear_f10)}/{int(r.n_labels_clear)}件。")
    failed = [method for method in METHODS[2:] if not config['selected'][method]['feasible']]
    floor_warning = ("**学習側の再現率維持条件を満たす設定がなかった方式：" + ", ".join(failed) + "。この方式は不足量最小の参考設定であり、維持条件を達成した改善案ではない。**") if failed else "両方式とも学習側の再現率維持条件を満たした。"
    text = f"""# match2 第7〜11点：音と選手動きによる候補の絞り込み

既存の音・追跡・ラベルは読み取り専用とし、このディレクトリだけに生成した。CPUのみ、深層学習の追加使用なし。結果は既存のClaude映像ラベルへの一致であり、真の打球検出精度や人の確認時間の実証ではない。

## 実行

リポジトリのルートで実行する。

```powershell
& study\\.venv\\Scripts\\python.exe -B -X utf8 outputs\\shot_pipeline_20261117\\audiovisual\\combine_audio_video.py
# 特徴抽出と第7・8点だけでの探索・固定
& study\\.venv\\Scripts\\python.exe -B -X utf8 outputs\\shot_pipeline_20261117\\audiovisual\\combine_audio_video.py --phase train
# 保存済み設定を再探索せず評価（キャッシュの入力ハッシュを確認）
& study\\.venv\\Scripts\\python.exe -B -X utf8 outputs\\shot_pipeline_20261117\\audiovisual\\combine_audio_video.py --phase evaluate
```

既存キャッシュが入力と一致すれば使用する。全フレームを読み直す場合は `--refresh-motion`。全体実行時間は{elapsed:.2f}秒（キャッシュ利用状態はrun_metadata.json）。

NumPy/SciPy/pandas/matplotlib/OpenCVと、前回の音処理と同じ `ffmpeg` を使用する。ffmpegは図用の音声だけを復号し（`-vn`）、一時WAVは終了時に削除する。

## 方法と固定した条件

1. `candidate_features_v2.csv` の全190候補を母集団とする。ID・点・rally・整数フレームがv1候補と完全一致すること、`kept`の148件がv2と一致することをassertする。v2から落ちた候補も映像方式には残す。
2. 音と同じ `audio/intervals.csv` の半開区間 `[start_frame,end_frame)` を使用。第7点はA/Bを別区間とし、区間をまたぐピーク探索・DP遷移は行わない。フレームfは0始まり、fps=60000/1001。
3. 各区間の先頭で1回だけシーク。その後は全フレームをgrab/retrieveで順読みし、毎回の前後のフレーム番号と1920×1080をassertする。区間先頭は前フレームを読まないため差分を欠測とする。
4. A：現在フレームの箱の左右に各幅30%を追加（全幅1.6倍）、上に高さ30%を追加（下端は固定）。画面端で切り、現在と前フレームの同じ画素領域のグレースケール絶対差の平均を計算する。移動する箱同士の差分ではない。箱なし・無効箱は欠測で、補間しない。各選手・区間の有効値の中央値で割る。中央値ゼロは欠測とする。
5. 各候補・選手について、候補±8Fの最大値から候補±30Fの中央値を引いた値を「動きのピーク高さ」とする。窓は区間端で切り、欠測は無視する。どちらかに有効値がなければ候補特徴も欠測。最大の正のピークを持つ選手の側をAの推定側とする。同値最大・全欠測・最大値非正は側不明。
6. 得点：`motion_score + audio_weight * audio_score + shape_weight * shape_score - threshold`。`motion_score=log1p(max(peak_height,0))/scale_motion`、`audio_score=log1p(max(strength,0))/scale_audio`。尺度は第7・8点の候補特徴の中央値だけから固定。`shape_score=sigmoid(rule_margin)`は音v2の既存規則の余裕を使う。映像欠測の得点寄与は0、側は不明。映像重みは1。
7. 方法3は得点>0の候補を残す。方法4は全v1候補をDPに入れ、側が交互かつ間隔≥12Fの部分列で得点合計最大を選ぶ。初打の側は自由、空列も許す。側不明はN/Fの2状態。負得点でも後続をつなぐ候補はDP上で許し、総得点で判断する。DP内部の割当側は `dp_assigned_side` に保存し、Aの推定側を上書きしない。

### 学習・固定・評価の順

第7・8点（7Aを含む）だけのラベルを読み、特徴尺度・格子・選択方法を適用して `selected_config.json` を保存してから第9〜11点ラベルを読み込む。方式3・4は別々に選ぶが探索格子と目的関数は同じ。第9〜11点の結果を見た重み・閾値・特徴定義の変更は行っていない。既存音v2の特徴仮説には第9点の観察があるため、この3点は完全に未接触の独立テストではない。

- 格子：`{json.dumps(config['grid'])}`（各方式144設定）。
- 制約：v1の学習側±10F対応数の90%以上、clear≥{config['recall_floors']['n_matched_clear_f10']}、全≥{config['recall_floors']['n_matched_all_f10']}。満たす設定の窓合計割合最小→clear ±6F対応数最大→全±10F対応数最大→設定ID順。該当設定なしの場合だけ、正規化した不足量の合計最小を先に選び、未達を記録する。
- 方式3：`{json.dumps(config['selected']['audio_v1_A'])}`。
- 方式4：`{json.dumps(config['selected']['audio_v1_A_B'])}`。
- 尺度：`{json.dumps(config['scales'])}`。

{floor_warning}

## 共通の評価

再現率は、同じ点・rally内の全ラベルに対し絶対時間差が小さい順の1対1照合（前回と同じ）。同率はラベル時刻→候補時刻→ID順。±6F/±10Fを別に照合し、その対応のうちclearの件数を数える。側は照合に使わない。clear専用の対応を別に最適化していない。

適合率目安は指定の「±10F以内に何らかのラベルがある候補数/全候補数」。複数候補が同じラベル付近でも数える。前回の1対1対応数/候補数とは異なるため、CSVに両者を別列で保存する。いずれも未ラベル打球やラベル誤差があるため真の適合率ではない。

確認範囲は各候補の連続時間±10F窓（幅20F、21枚とは数えない）を区間端で切り、重複も数えた和÷区間長を主指標とする。和集合割合も併記。側正解率は時間で1対1照合でき、Aの側が既知の候補について正解数/既知数を計算する。不明は分母から除外し件数を併記（CSVに既知率も保存）。DPで補った側は採点しない。v1/v2にも同じA推定側を付けるので、音単独の側予測能力を意味しない。

{table_markdown(evaluation, ['train_07_08_including_7A', 'test_09_11'])}

### 第7点rally Aを除く固定設定の集計

この表のための再調整はしていない。点別・rally A単独・5点全体のA含む/除くは `evaluation.csv` に収録。

{table_markdown(evaluation, ['train_07_08_excluding_7A'])}

## 結果の解釈と限界

{chr(10).join(observations)}

削減率と再現率を同時に判断する必要がある。候補を減らしても打球を落とすなら、人が全区間を見直す必要が残り、確認時間の改善とは言えない。

**今回の判定：Aによる削減は限定的で、十分な確認量削減を実証できていない。Bの厳しい交互制約は、この側推定では採用できない。**

- ラベルはClaudeが6フレームおきのコマから付けたもので、音から6〜12Fずれる場合がある。±10Fは約0.167秒、±6Fは約0.100秒。時間を合わせてラベルを動かしていない。±10F内の一致には偶然一致も含まれ、検出の真の時間精度を示さない。
- 追跡は独立に正確さを保証されていない。第8点では手前選手の欠測や線審への誤追跡があり、箱領域の低い動きを「打球でない証拠」とは扱えない。箱の重なり・背景・カメラ揺れ・体の移動・相手の反応も差分に入る。
- 第8点のnear_frontの生差分中央値は{diagnostics.loc[(diagnostics.point == 8) & (diagnostics.player_slot == 'near_front'), 'interval_motion_median'].iloc[0]:.6f}で、他選手より小さい。追跡誤りと小さい中央値による増幅が側推定を偏らせる可能性がある。これは失敗原因の確定ではなく診断上の観察である。
- 側は動きの大きさからの推定。ラケット接触の確認ではない。奥の小さい箱は中央値正規化後もノイズの相対影響が大きくなり得る。ピーク時刻±8Fの許容は打者と相手の反応を混ぜ得る。
- 第8点にはラベル間隔が12F未満の箇所があり、指定の12F制約で両方を残せない。中間打球がv1段階で欠落すると、残った正しい候補が同側続きになり、交互制約が追加の正解を落とし得る。rally途中の誤ラベル・未ラベルにも影響される。
- 学習は同じ試合の2点、評価はclear18件のみ。重み探索144通り×2方式の学習側の数値は楽観的になり得る。第7Aを含む/除くで件数と分母が変わる。別の未使用試合で、原動画からの独立正解と確認修正時間を測る必要がある。
- 区間ごとの映像欠測は `motion_diagnostics.csv`。欠測は補間せず、区間先頭や追跡範囲外の候補は映像情報が少ない。生値と正規化値を保存し、中央値正規化の影響を後から確認できる。

## 出力

- `motion_timeseries.csv`：フレーム×4選手の生差分・正規化値・面積・箱有無。
- `audio_plot_timeseries.csv`：図用に復元した元の4–12 kHz音スコア。既存の全候補位置でstrengthとの一致をassertする。候補選択・再探索には使わない。
- `motion_diagnostics.csv`：区間×選手の中央値、有効数、箱数、シーク回数。
- `candidate_features.csv`：全v1候補に各選手の局所中央値・ピーク高さ/時刻/有効数・推定選手/側を追加。
- `selected_candidates.csv`：4方式の保持候補、Aの側とDPの割当側、方式3/4の得点。
- `evaluation.csv` / `label_matches.csv`：共通指標・分子分母、ラベルごとの対応とずれ・側。
- `parameter_search_train.csv` / `selected_config.json`：学習側探索と固定設定。`run_metadata.json`に入力/固定設定/実行スクリプトのハッシュを保存。
- `point_07_audiovisual.png`〜`point_11_audiovisual.png`：上段は1ms刻みの音スコアとv1/v2候補、中段は4人の正規化動き、下段は4方式の保持候補とラベル。全段でフレーム軸を一致させ、第7点はA/Bを分ける。ラベルはclear実線/inferred点線、側N青/F紫、不明灰色。

評価後の変更は図用音スコアの復元と説明・診断の追記のみ。重み・しきい値・映像特徴・候補選択・評価指標の変更や再探索は行っていない。
"""
    (HERE / "README.md").write_text(text, encoding="utf-8")


def get_motion(points, intervals, video, fps, manifest, refresh=False):
    csv_path, meta_path = HERE / "motion_timeseries.csv", HERE / "motion_cache.json"
    cached = pd.read_csv(csv_path) if csv_path.exists() and meta_path.exists() and not refresh else pd.DataFrame()
    diagnostic_path = HERE / "motion_diagnostics.csv"
    diagnostics = pd.read_csv(diagnostic_path) if len(cached) and diagnostic_path.exists() else pd.DataFrame()
    signature = {k: v for k, v in manifest.items() if "tracking" in k or k.endswith("intervals.csv")}
    signature["video_bytes"] = video.stat().st_size
    signature["video_mtime_ns"] = video.stat().st_mtime_ns
    signature["definition"] = "roi_width1.6_top0.3_first_missing_v1"
    if len(cached) and json.loads(meta_path.read_text(encoding="utf-8")) != signature:
        cached, diagnostics = pd.DataFrame(), pd.DataFrame()
    have = set(cached.point) if len(cached) else set()
    missing = [p for p in points if p not in have]
    if missing:
        new, diag = extract_motion(missing, intervals, video, fps)
        cached = pd.concat([cached, new], ignore_index=True)
        diagnostics = pd.concat([diagnostics, diag], ignore_index=True)
        write_csv("motion_timeseries.csv", cached)
        write_csv("motion_diagnostics.csv", diagnostics)
        write_json(meta_path, signature)
    return cached[cached.point.isin(points)].copy(), diagnostics[diagnostics.point.isin(points)].copy(), not missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all", "train", "evaluate"], default="all")
    parser.add_argument("--refresh-motion", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    assert not HERE.is_symlink()
    manifest = {str(p.relative_to(PIPE)): sha(p) for p in input_paths()}
    intervals, candidates, video, fps = load_inputs()
    if args.phase != "evaluate":
        motion, diag, train_cached = get_motion(TRAIN, intervals, video, fps, manifest, args.refresh_motion)
        train_features = candidate_features(candidates[candidates.point.isin(TRAIN)].reset_index(drop=True), motion)
        train_labels = load_labels(TRAIN, intervals)
        config = train(train_features, intervals[intervals.point.isin(TRAIN)], train_labels)
        config["training_provenance"] = training_provenance(manifest, video)
        config["frozen_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(HERE / "selected_config.json", config)
        if args.phase == "train":
            write_csv("train_candidate_features.csv", train_features)
            print("Training complete; test labels have not been loaded.", flush=True)
            return
    else:
        config = json.loads((HERE / "selected_config.json").read_text(encoding="utf-8"))
        assert config["training_provenance"] == training_provenance(manifest, video), "Training inputs changed; retrain before evaluation"
        train_cached = None
    frozen_sha = sha(HERE / "selected_config.json")
    # Only after the config is on disk may held-out labels enter this process.
    labels = load_labels(TRAIN+TEST, intervals)
    motion, diagnostics, all_cached = get_motion(TRAIN+TEST, intervals, video, fps, manifest, args.refresh_motion if args.phase == "evaluate" else False)
    features = candidate_features(candidates, motion)
    selections = {method: select_method(features, config["scales"], config["selected"].get(method, {}), method) for method in METHODS}
    write_csv("candidate_features.csv", features)
    selected = pd.concat([table.assign(method=method) for method, table in selections.items()], ignore_index=True)
    write_csv("selected_candidates.csv", selected)
    metrics, matches = [], []
    for scope, mask in scope_masks(intervals):
        ints = intervals[mask]
        keys = set(zip(ints.point, ints.rally))
        labs = labels[mask_keys(labels, keys)]
        for method, table in selections.items():
            row, detail = evaluate(ints, labs, table[mask_keys(table, keys)], scope, method)
            metrics.append(row)
            matches.extend(detail)
    evaluation = pd.DataFrame(metrics)
    write_csv("evaluation.csv", evaluation)
    write_csv("label_matches.csv", pd.DataFrame(matches))
    audio_series = audio_plot_series(intervals, video, fps, candidates)
    plot_points(intervals, labels, features, motion, selections, audio_series)
    assert frozen_sha == sha(HERE / "selected_config.json"), "Config changed after test loading"
    after = {str(p.relative_to(PIPE)): sha(p) for p in input_paths()}
    assert manifest == after, "An input was modified"
    elapsed = time.perf_counter()-started
    make_readme(evaluation, config, diagnostics, elapsed)
    write_json(HERE / "run_metadata.json", dict(input_sha256=manifest, inputs_unchanged=True, script_sha256=sha(__file__),
                                               selected_config_sha256_before_test=frozen_sha, selected_config_sha256_after_test=sha(HERE / "selected_config.json"),
                                               source_video=str(video), video_bytes=video.stat().st_size, source_fps=fps,
                                               phase=args.phase, train_motion_from_cache=train_cached, all_motion_from_cache=all_cached,
                                               elapsed_sec=elapsed, test_result_used_for_tuning=False,
                                               versions=dict(numpy=np.__version__, pandas=pd.__version__, opencv=cv2.__version__, matplotlib=matplotlib.__version__)))
    print(evaluation[evaluation.scope == "test_09_11"][["method", "n_candidates", "recall_clear_f10", "review_window_sum_ratio", "candidates_per_sec", "side_accuracy_f10"]].to_string(index=False), flush=True)
    print(f"Completed in {elapsed:.2f}s; inputs unchanged; configuration hash unchanged.", flush=True)


if __name__ == "__main__":
    main()
