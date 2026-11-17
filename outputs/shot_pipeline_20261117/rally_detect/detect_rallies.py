"""Audio/scene-only rally proposals. Run from repo root; no packages installed.

Only this directory is written. WAVs are decoded in TemporaryDirectory outside
the repository. Labels are passed to evaluation/selection, never to detection.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent
ROOT = PIPE.parents[1]
sys.path.insert(0, str(PIPE / "audio"))
import detect_hit_sounds as v1
import detect_hit_sounds_v2 as v2

POINTS = (7, 8, 9, 10, 11)
G_GRID = (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
K_GRID = (3, 4, 5, 6)
SHAPES = ("none", "sharp", "v2")
TRIMS = ("none", "rhythmic_jump1", "rhythmic_jump2")
FAMILIES = ("cluster_v1", "shape_only", "shape_and_serve")
TOLERANCE = 10
SCORE_DEFINITION = "0.5 * candidate_F1 + 0.5 * (2*sum(matched rally IoU)/(n_GT+n_pred))"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratio(a, b):
    return float(a / b) if b else 0.0


def configurations():
    return [dict(config_id=f"r{i:03d}", gap_sec=g, min_candidates=k,
                 shape=shape, serve_trim=trim)
            for i, (g, k, shape, trim) in enumerate(
                itertools.product(G_GRID, K_GRID, SHAPES, TRIMS))]


def load_inputs():
    inventory = pd.read_csv(ROOT / "outputs/match2_point_inventory_20261117/point_inventory.csv")
    inventory = inventory[inventory.point.isin(POINTS)].copy()
    scenes = pd.read_csv(PIPE / "scenes/scenes.csv", keep_default_na=False)
    scenes = scenes[scenes.point.isin(POINTS)].copy()
    assert set(scenes.is_main_camera) <= {"True", "False", "uncertain"}
    for p in POINTS:
        row = inventory[inventory.point == p].iloc[0]
        local = scenes[scenes.point == p].sort_values("start_frame")
        assert int(local.start_frame.iloc[0]) == int(row.display_from_frame)
        assert int(local.end_frame.iloc[-1]) + 1 == int(row.next_display_from_frame)
        assert np.all(local.start_frame.to_numpy()[1:] == local.end_frame.to_numpy()[:-1] + 1)
    labels = pd.concat([pd.read_csv(PIPE / f"labels/point_{p:02d}.csv",
                                    keep_default_na=False) for p in POINTS], ignore_index=True)
    labels["label_id"] = [f"p{r.point:02d}_{r.rally}_s{r.shot_index:02d}" for r in labels.itertuples()]
    assert labels.label_id.is_unique
    assert labels.confidence.isin(("clear", "inferred")).all()
    protocol = json.loads((PIPE / "tracking/point_07/protocol.json").read_text(encoding="utf-8"))
    assert protocol["source_fps"] == "60000/1001"
    payload = json.loads((PIPE / "audio/selected_config.json").read_text(encoding="utf-8"))
    config = v1.Config(**{f.name: payload[f.name] for f in fields(v1.Config)})
    assert (config.low_hz, config.high_hz, config.median_sec) == (4000, 12000, 0.5)
    return inventory, scenes, labels, Path(protocol["source_video"]), config


def extract(inventory, scenes, video, ffmpeg, config):
    """Decode full points plus padding; compute the frozen v1 feature only.

    Full-point filtering avoids scene-edge median artifacts. Peak finding is
    also performed on full points; scene membership is assigned afterwards.
    No label-derived window is used. False scenes are retained for plot audit,
    but cannot enter detection or its search in either camera policy.
    """
    tables, traces = [], []
    with tempfile.TemporaryDirectory(prefix="match2_rally_audio_") as tmp:
        assert not Path(tmp).resolve().is_relative_to(ROOT.resolve())
        for p in POINTS:
            inv = inventory[inventory.point == p].iloc[0]
            start, end = int(inv.display_from_frame), int(inv.next_display_from_frame)
            first = max(0, math.floor((start * v1.FRAME_SEC - v1.PADDING_SEC) * v1.SAMPLE_RATE))
            last = math.ceil((end * v1.FRAME_SEC + v1.PADDING_SEC) * v1.SAMPLE_RATE)
            wav = Path(tmp) / f"p{p:02d}.wav"
            subprocess.run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                            "-ss", f"{first / v1.SAMPLE_RATE:.9f}", "-i", str(video),
                            "-t", f"{(last-first) / v1.SAMPLE_RATE:.9f}", "-map", "0:a:0",
                            "-vn", "-ac", "1", "-ar", str(v1.SAMPLE_RATE),
                            "-c:a", "pcm_s16le", str(wav)], check=True, capture_output=True)
            rate, audio = wavfile.read(wav)
            assert rate == v1.SAMPLE_RATE and audio.ndim == 1
            assert abs(len(audio) - (last - first)) <= 2
            seg = v1.Segment(p, "unsegmented", start, end, pd.DataFrame())
            seg.audio = audio.astype(np.float64) / 32768.0
            seg.audio_start_sec = first / v1.SAMPLE_RATE
            starts = np.arange(0, len(audio) - v1.ENERGY_SAMPLES + 1, v1.HOP_SAMPLES)
            seg.feature_frames = ((starts + (v1.ENERGY_SAMPLES-1)/2) / v1.SAMPLE_RATE
                                  + seg.audio_start_sec) / v1.FRAME_SEC
            _, log, median = v2.band_envelope(seg, (config.low_hz, config.high_hz))
            score = log - median
            seg.features = {(config.low_hz, config.high_hz, config.median_sec): score}
            candidates = v2.add_features([seg], v1.detect([seg], config))
            candidates["candidate_id"] = [f"p{p:02d}_c{i:04d}" for i in range(len(candidates))]
            candidates = candidates.drop(columns="rally")  # unknown at inference
            for scene in scenes[scenes.point == p].itertuples():
                member = candidates.candidate_frame.between(scene.start_frame, scene.end_frame)
                candidates.loc[member, "scene_id"] = scene.scene_id
                candidates.loc[member, "camera_class"] = scene.is_main_camera
            assert candidates.scene_id.notna().all()
            tables.append(candidates)
            mask = (seg.feature_frames >= start) & (seg.feature_frames < end)
            # Peak-preserving 1F bins for the plot; inference stays at 1 ms.
            trace = pd.DataFrame({"frame": np.floor(seg.feature_frames[mask]).astype(int),
                                  "v1_score": score[mask]}).groupby("frame", as_index=False).max()
            trace.insert(0, "point", p)
            traces.append(trace)
            wav.unlink()
            print(f"Decoded full point {p}: {end-start}F, {len(candidates)} v1 candidates", flush=True)
    assert not Path(tmp).exists()
    return pd.concat(tables, ignore_index=True), pd.concat(traces, ignore_index=True)


def shape_mask(c, rule):
    keep = np.ones(len(c), dtype=bool)
    if rule != "none":
        keep &= (c.half_width_ms.to_numpy() <= 15) & (c.post50_drop_ln.to_numpy() >= 1)
    if rule == "v2":
        keep &= c.paired_rise_ln.to_numpy() >= 0.5
    return keep


def prefix_length(local, rule):
    """Trim only a short, weak, regularly spaced leading sequence.

    Require >=3 sounds before the jump, >=2 prefix gaps in [0.4,1.1]s,
    gap CV <=0.3, prefix <=4s and <=8 sounds. High-band rise must increase
    by >=1 or >=2 log-power units over prefix median, paired rise by >=0.5,
    and the proposed serve must exceed every prefix high-band rise.
    """
    if rule == "none":
        return 0
    threshold = 1. if rule == "rhythmic_jump1" else 2.
    frame = local.candidate_frame_float.to_numpy()
    high, paired = local.high_rise_ln.to_numpy(), local.paired_rise_ln.to_numpy()
    for j in range(3, min(9, len(local))):
        gaps = np.diff(frame[:j]) * v1.FRAME_SEC
        if (frame[j] - frame[0]) * v1.FRAME_SEC > 4:
            break
        if (np.all((gaps >= .4) & (gaps <= 1.1)) and np.std(gaps) / np.mean(gaps) <= .3
                and high[j] - np.median(high[:j]) >= threshold
                and high[j] > np.max(high[:j])
                and paired[j] - np.median(paired[:j]) >= .5):
            return j
    return 0


def detect(candidates, config):
    """Pure inference: no labels, point timing priors, pose or GT-rally IDs."""
    kept_ids, rallies = [], []
    c = candidates[shape_mask(candidates, config["shape"])]
    for scene_id, local in c.groupby("scene_id", sort=True):
        local = local.sort_values("candidate_frame_float")
        breaks = np.r_[True, np.diff(local.candidate_frame_float.to_numpy()) * v1.FRAME_SEC
                       > config["gap_sec"] + 1e-12]
        for _, group in local.groupby(np.cumsum(breaks), sort=True):
            if len(group) < config["min_candidates"]:
                continue
            trimmed = prefix_length(group, config["serve_trim"])
            retained = group.iloc[trimmed:]
            if len(retained) < config["min_candidates"]:
                continue
            point = int(retained.point.iloc[0])
            rally_id = f"p{point:02d}_d{len(rallies)+1:03d}"
            rallies.append(dict(point=point, detected_id=rally_id, scene_id=scene_id,
                                start_frame=int(retained.candidate_frame.iloc[0]),
                                end_frame=int(retained.candidate_frame.iloc[-1]),
                                n_candidates=len(retained), n_prefix_removed=trimmed,
                                raw_cluster_start_frame=int(group.candidate_frame.iloc[0]),
                                camera_class=str(retained.camera_class.iloc[0])))
            kept_ids.extend(retained.index.tolist())
    return candidates.loc[sorted(kept_ids)].copy(), rallies


def labels_for(labels, point, include_a):
    local = labels[labels.point == point]
    if not include_a:
        local = local[~((local.point == 7) & (local.rally == "A"))]
    return local.sort_values("contact_frame")


def match_shots(labels, candidates):
    """Maximum cardinality, then minimum absolute frame error; same point."""
    if not len(labels) or not len(candidates):
        return []
    delta = candidates.candidate_frame.to_numpy()[None, :] - labels.contact_frame.to_numpy()[:, None]
    cost = np.where(np.abs(delta) <= TOLERANCE, np.abs(delta), 2e6).astype(float)
    cost = np.c_[cost, np.full((len(labels), len(labels)), 1e6)]
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), int(delta[i, j])) for i, j in zip(rows, cols)
            if j < len(candidates) and abs(delta[i, j]) <= TOLERANCE]


def evaluate_point(labels, candidates, predictions):
    gt = [dict(rally=str(r), start_frame=int(g.contact_frame.min()),
               end_frame=int(g.contact_frame.max())) for r, g in labels.groupby("rally", sort=True)]
    pairs = match_shots(labels, candidates)
    clear = labels.confidence.eq("clear").to_numpy()
    n, m = len(gt), len(predictions)
    overlap, iou = np.zeros((n, m), dtype=int), np.zeros((n, m))
    for i, a in enumerate(gt):
        for j, b in enumerate(predictions):
            overlap[i, j] = max(0, min(a["end_frame"], b["end_frame"]) - max(a["start_frame"], b["start_frame"]) + 1)
            union = max(a["end_frame"], b["end_frame"]) - min(a["start_frame"], b["start_frame"]) + 1
            iou[i, j] = overlap[i, j] / union
    assigned = []
    if n and m:
        cost = np.c_[np.where(overlap > 0, -1000-iou, 1000), np.zeros((n, n))]
        rr, cc = linear_sum_assignment(cost)
        assigned = [(int(i), int(j)) for i, j in zip(rr, cc) if j < m and overlap[i, j] > 0]
    starts = [predictions[j]["start_frame"] - gt[i]["start_frame"] for i, j in assigned]
    ends = [predictions[j]["end_frame"] - gt[i]["end_frame"] for i, j in assigned]
    duration = sum((r["end_frame"] - r["start_frame"] + 1) * v1.FRAME_SEC for r in predictions)
    result = dict(n_points=1, n_labels_all=len(labels), n_labels_clear=int(clear.sum()),
                  n_candidates=len(candidates), n_matched_all=len(pairs),
                  n_matched_clear=sum(bool(clear[i]) for i, _, _ in pairs),
                  detected_duration_sec=duration, n_gt_rallies=n, n_detected_rallies=m,
                  n_rallies_matched=len(assigned), n_rallies_any_overlap=int((overlap > 0).any(axis=1).sum()),
                  n_extra_rallies=m-len(assigned), n_missed_rallies=n-len(assigned),
                  rally_iou_sum=sum(iou[i, j] for i, j in assigned),
                  n_rallies_iou50=sum(iou[i, j] >= .5 for i, j in assigned),
                  n_boundaries_within10=sum(abs(s) <= 10 and abs(e) <= 10 for s, e in zip(starts, ends)),
                  start_abs_sum=sum(abs(s) for s in starts), end_abs_sum=sum(abs(e) for e in ends),
                  start_signed_sum=sum(starts), end_signed_sum=sum(ends),
                  n_prefix_removed=sum(r.get("n_prefix_removed", 0) for r in predictions),
                  n_trimmed_rallies=sum(r.get("n_prefix_removed", 0) > 0 for r in predictions),
                  n_fragment_predictions=int(((overlap > 0).any(axis=0)).sum())-len(assigned),
                  n_merged_predictions=int(((overlap > 0).sum(axis=0) > 1).sum()))
    return result, (gt, assigned, overlap, iou, pairs)


SUM_KEYS = tuple(evaluate_point(pd.DataFrame(columns=["rally", "contact_frame", "confidence"]),
                                pd.DataFrame(columns=["candidate_frame"]), [])[0])


def finish(row):
    r = dict(row)
    r.update(recall_all=ratio(r["n_matched_all"], r["n_labels_all"]),
             recall_clear=ratio(r["n_matched_clear"], r["n_labels_clear"]),
             precision_proxy=ratio(r["n_matched_all"], r["n_candidates"]),
             f1=ratio(2*r["n_matched_all"], r["n_labels_all"] + r["n_candidates"]),
             candidates_per_shot=ratio(r["n_candidates"], r["n_labels_all"]),
             candidates_per_sec=ratio(r["n_candidates"], r["detected_duration_sec"]),
             candidates_per_point=ratio(r["n_candidates"], r["n_points"]),
             n_missing_shots=r["n_labels_all"]-r["n_matched_all"],
             missing_shots_per_point=ratio(r["n_labels_all"]-r["n_matched_all"], r["n_points"]),
             missing_clear_shots=r["n_labels_clear"]-r["n_matched_clear"],
             rally_recall=ratio(r["n_rallies_matched"], r["n_gt_rallies"]),
             rally_any_overlap_recall=ratio(r["n_rallies_any_overlap"], r["n_gt_rallies"]),
             rally_precision=ratio(r["n_rallies_matched"], r["n_detected_rallies"]),
             rally_f1=ratio(2*r["n_rallies_matched"], r["n_gt_rallies"]+r["n_detected_rallies"]),
             rally_soft_f1=ratio(2*r["rally_iou_sum"], r["n_gt_rallies"]+r["n_detected_rallies"]),
             start_mae_frames=ratio(r["start_abs_sum"], r["n_rallies_matched"]),
             end_mae_frames=ratio(r["end_abs_sum"], r["n_rallies_matched"]))
    r["selection_score"] = .5*(r["f1"]+r["rally_soft_f1"])
    return r


def pool(rows):
    return finish({k: sum(r[k] for r in rows) for k in SUM_KEYS})


def allowed(config, family):
    return ((config["shape"] == "none" and config["serve_trim"] == "none") if family == "cluster_v1"
            else config["serve_trim"] == "none" if family == "shape_only" else True)


def details(labels, candidates, predictions, context):
    stats, (gt, assigned, overlap, iou, pairs) = evaluate_point(labels, candidates, predictions)
    rally_rows, shot_rows, edge_rows, candidate_rows = [], [], [], []
    for i, a in enumerate(gt):
        matches = [j for ii, j in assigned if ii == i]
        j = matches[0] if matches else None
        b = predictions[j] if j is not None else {}
        rally_rows.append({**context, "status": "matched" if j is not None else "missed_gt",
                           "gt_rally": a["rally"], "gt_start_frame": a["start_frame"],
                           "gt_end_frame": a["end_frame"], **b,
                           "start_delta_frames": b["start_frame"]-a["start_frame"] if b else None,
                           "end_delta_frames": b["end_frame"]-a["end_frame"] if b else None,
                           "iou": float(iou[i, j]) if b else None,
                           "n_predictions_overlapping_gt": int((overlap[i] > 0).sum())})
    for j, b in enumerate(predictions):
        if not any(jj == j for _, jj in assigned):
            rally_rows.append({**context, **b, "status": "extra_fragment" if (overlap[:, j] > 0).any()
                               else "extra_no_labeled_rally", "gt_rally": ""})
        for i, a in enumerate(gt):
            if overlap[i, j] > 0:
                edge_rows.append({**context, "gt_rally": a["rally"], "detected_id": b["detected_id"],
                                  "overlap_frames": int(overlap[i, j]), "iou": float(iou[i, j]),
                                  "assigned_one_to_one": (i, j) in assigned})
    by_label = {i: (j, d) for i, j, d in pairs}
    by_cand = {j: (i, d) for i, j, d in pairs}
    for i, l in enumerate(labels.itertuples()):
        pair = by_label.get(i)
        shot_rows.append({**context, "label_id": l.label_id, "gt_rally": l.rally,
                          "contact_frame": l.contact_frame, "confidence": l.confidence,
                          "matched": pair is not None,
                          "candidate_id": candidates.iloc[pair[0]].candidate_id if pair else "",
                          "delta_frames": pair[1] if pair else None,
                          "inside_detected_interval": any(r["start_frame"] <= l.contact_frame <= r["end_frame"]
                                                           for r in predictions)})
    for j, c in enumerate(candidates.itertuples()):
        pair = by_cand.get(j)
        candidate_rows.append({**context, "candidate_id": c.candidate_id,
                               "candidate_frame": c.candidate_frame, "scene_id": c.scene_id,
                               "camera_class": c.camera_class,
                               "label_id": labels.iloc[pair[0]].label_id if pair else "",
                               "delta_frames": pair[1] if pair else None})
    return finish(stats), rally_rows, shot_rows, edge_rows, candidate_rows


def fit_and_evaluate(candidates, scenes, labels, configs):
    metric_rows, searches, frozen = [], [], []
    rally_rows, shot_rows, edge_rows, candidate_rows = [], [], [], []
    plot_data = {}
    folds = [("fixed", (7, 8), (9, 10, 11))] + [(f"loo_{p:02d}", tuple(q for q in POINTS if q != p), (p,)) for p in POINTS]
    for camera in ("true_only", "true_plus_uncertain"):
        eligible = candidates[candidates.camera_class.isin(("True",) if camera == "true_only" else ("True", "uncertain"))]
        by_point = {p: eligible[eligible.point == p] for p in POINTS}
        inference = {(cfg["config_id"], p): detect(by_point[p], cfg) for cfg in configs for p in POINTS}
        for include_a in (True, False):
            scenario = dict(camera_policy=camera, include_rally_a=include_a)
            labs = {p: labels_for(labels, p, include_a) for p in POINTS}
            cached = {(cfg["config_id"], p): evaluate_point(labs[p], *inference[cfg["config_id"], p])[0]
                      for cfg in configs for p in POINTS}
            for fold, train, test in folds:
                scores = {cfg["config_id"]: pool([cached[cfg["config_id"], p] for p in train]) for cfg in configs}
                for family in FAMILIES:
                    choices = [cfg for cfg in configs if allowed(cfg, family)]
                    def rank(cfg):
                        m = scores[cfg["config_id"]]
                        complexity = (cfg["shape"] != "none") + (cfg["serve_trim"] != "none")
                        return (-round(m["selection_score"], 12), -m["n_matched_clear"],
                                m["start_mae_frames"]+m["end_mae_frames"], m["n_candidates"],
                                complexity, cfg["config_id"])
                    chosen = min(choices, key=rank)
                    context = {**scenario, "protocol": fold, "family": family, "config_id": chosen["config_id"]}
                    frozen.append({**context, **chosen, "train_points": list(train), "test_points": list(test),
                                   "selection_score": scores[chosen["config_id"]]["selection_score"]})
                    for cfg in choices:
                        searches.append({**scenario, "protocol": fold, "family": family, **cfg,
                                         **scores[cfg["config_id"]], "selected": cfg == chosen})
                    for p in (POINTS if fold == "fixed" else test):
                        selected, pred = inference[chosen["config_id"], p]
                        ctx = {**context, "point": p, "partition": "train" if p in train else "test"}
                        m, rr, sr, er, cr = details(labs[p], selected, pred, ctx)
                        m.update(ctx)
                        m["n_raw_camera_candidates"] = len(by_point[p])
                        m["candidate_reduction_vs_raw"] = 1-ratio(len(selected), len(by_point[p]))
                        m["checks_plus_additions"] = len(selected)+m["n_missing_shots"]
                        metric_rows.append(m)
                        rally_rows.extend(rr); shot_rows.extend(sr); edge_rows.extend(er); candidate_rows.extend(cr)
                        if family == "shape_and_serve":
                            plot_data[camera, include_a, "fixed" if fold == "fixed" else "loo", p] = (selected, pred)
            print(f"Selected and evaluated {camera}, include_A={include_a}", flush=True)
    return metric_rows, searches, frozen, rally_rows, shot_rows, edge_rows, candidate_rows, plot_data


def summarize(per_point):
    rows = []
    for (camera, a, family), group in per_point.groupby(["camera_policy", "include_rally_a", "family"], sort=True):
        sets = [("fixed_train_07_08", group[(group.protocol == "fixed") & (group.partition == "train")]),
                ("fixed_test_09_11", group[(group.protocol == "fixed") & (group.partition == "test")]),
                ("fixed_all_train_mixed", group[group.protocol == "fixed"]),
                ("loo_all_07_11", group[group.protocol.str.startswith("loo_")]),
                ("loo_common_09_11", group[group.protocol.str.startswith("loo_") & group.point.isin((9,10,11))])]
        for scope, subset in sets:
            m = pool(subset.to_dict("records"))
            raw = int(subset.n_raw_camera_candidates.sum())
            rows.append(dict(camera_policy=camera, include_rally_a=bool(a), family=family, scope=scope,
                             **m, n_raw_camera_candidates=raw,
                             candidate_reduction_vs_raw=1-ratio(m["n_candidates"], raw),
                             checks_plus_additions=m["n_candidates"]+m["n_missing_shots"],
                             checks_plus_additions_per_point=ratio(m["n_candidates"]+m["n_missing_shots"], m["n_points"])))
    return pd.DataFrame(rows)


def baseline_tables(candidates, labels, inventory, scenes):
    old = pd.read_csv(PIPE / "audio/candidates.csv", keep_default_na=False)
    old_intervals = pd.read_csv(PIPE / "audio/intervals.csv")
    rows = []
    for a in (True, False):
        for scope, points in (("all_07_11", POINTS), ("train_07_08", (7,8)), ("test_09_11", (9,10,11))):
            for camera in ("historical", "true_only", "true_plus_uncertain"):
                c = old if camera == "historical" else candidates[candidates.camera_class.isin(
                    ("True",) if camera == "true_only" else ("True", "uncertain"))]
                for oracle_pad in (None, 0, 15):
                    local_rows, legacy_matches = [], 0
                    for p in points:
                        lab = labels_for(labels, p, a)
                        cp = c[c.point == p]
                        pad = oracle_pad or 0
                        intervals = [(int(g.contact_frame.min())-pad, int(g.contact_frame.max())+pad)
                                     for _, g in lab.groupby("rally")]
                        if oracle_pad is not None:
                            mask = np.zeros(len(cp), dtype=bool)
                            for lo, hi in intervals:
                                mask |= cp.candidate_frame.between(lo, hi).to_numpy()
                            cp = cp[mask]
                            duration = sum((hi-lo+1)*v1.FRAME_SEC for lo, hi in intervals)
                        elif camera == "historical":
                            # Historical intervals are half-open; no +1 here.
                            si = old_intervals[old_intervals.point == p]
                            duration = float(((si.end_frame-si.start_frame)*v1.FRAME_SEC).sum())
                        else:
                            sp = scenes[(scenes.point == p) & scenes.is_main_camera.isin(
                                ("True",) if camera == "true_only" else ("True", "uncertain"))]
                            duration = float((sp.n_frames*v1.FRAME_SEC).sum())
                        m = evaluate_point(lab, cp, [])[0]
                        if camera == "historical":
                            legacy_matches += len(v1.greedy_match(lab, cp, TOLERANCE))
                        m["detected_duration_sec"] = duration
                        local_rows.append(m)
                    pooled = pool(local_rows)
                    rows.append(dict(scope=scope, include_rally_a=a, camera_policy=camera,
                                     baseline=("all_input_candidates" if oracle_pad is None else
                                               "oracle_gt_rally" if oracle_pad == 0 else
                                               "oracle_gt_rally_plusminus15"), **pooled,
                                     legacy_greedy_matched_all=legacy_matches if camera == "historical" else None,
                                     legacy_greedy_f1=ratio(2*legacy_matches, pooled["n_candidates"]+pooled["n_labels_all"])
                                     if camera == "historical" else None))
    return pd.DataFrame(rows)


def figures(inventory, scenes, traces, candidates, labels, plot_data):
    colors = {"True": "#81c6a5", "uncertain": "#f4c46a", "False": "#ccd2dc"}
    for camera in ("true_only", "true_plus_uncertain"):
        for p in POINTS:
            fig, axes = plt.subplots(8, 1, figsize=(18, 12), sharex=True,
                                     gridspec_kw={"height_ratios": [0.6, 1.6, .6, .8, 1, 1, 1, 1]})
            for s in scenes[scenes.point == p].itertuples():
                axes[0].axvspan(s.start_frame, s.end_frame+1, color=colors[s.is_main_camera])
                if s.n_frames > 170:
                    axes[0].text((s.start_frame+s.end_frame)/2, .5, s.scene_id.split("_")[1],
                                 ha="center", va="center", fontsize=7)
            tr = traces[traces.point == p]
            axes[1].plot(tr.frame, tr.v1_score, color="#313e54", linewidth=.6)
            axes[1].axhline(1, color="#bf5466", linestyle="--", linewidth=.7)
            cp = candidates[(candidates.point == p) & candidates.camera_class.isin(
                ("True",) if camera == "true_only" else ("True", "uncertain"))]
            axes[2].scatter(cp.candidate_frame, np.full(len(cp), .5), marker="|", s=130, color="#626877")
            for r, g in labels[labels.point == p].groupby("rally"):
                lo, hi = g.contact_frame.min(), g.contact_frame.max()
                axes[3].axvspan(lo, hi+1, alpha=.25, color="#386ddb", hatch="//" if p == 7 and r == "A" else None)
                axes[3].text((lo+hi)/2, .8, f"GT {r}", ha="center", fontsize=8)
                for confidence, color in (("clear", "#2469cf"), ("inferred", "#d58e2f")):
                    subset = g[g.confidence == confidence]
                    axes[3].scatter(subset.contact_frame, np.full(len(subset), .3), marker="|", s=110, color=color)
            for ax, (a, protocol) in zip(axes[4:], ((True,"fixed"),(True,"loo"),(False,"fixed"),(False,"loo"))):
                selected, pred = plot_data[camera, a, protocol, p]
                for r in pred:
                    ax.axvspan(r["start_frame"], r["end_frame"]+1, color="#31a581", alpha=.25)
                    ax.text((r["start_frame"]+r["end_frame"])/2, .82,
                            f'{r["start_frame"]}-{r["end_frame"]}', ha="center", fontsize=7)
                ax.scatter(selected.candidate_frame, np.full(len(selected), .35), marker="|", s=130, color="#00826d")
                ax.set_ylabel(f'{protocol}\nA {"in" if a else "out"}', rotation=0, ha="right", va="center")
            for i, label in enumerate(("Scene", "v1 score\nmax / frame", "Full v1", "GT")):
                axes[i].set_ylabel(label)
            for i, ax in enumerate(axes):
                ax.grid(axis="x", alpha=.15)
                if i != 1:
                    ax.set_ylim(0, 1); ax.set_yticks([])
                ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            inv = inventory[inventory.point == p].iloc[0]
            axes[-1].set_xlim(inv.display_from_frame, inv.next_display_from_frame)
            axes[-1].set_xlabel("Source frame (0-based; 60000/1001 fps)")
            fig.suptitle(f"match2 point {p:02d} | {camera} | green=True, amber=uncertain, gray=False\n"
                         "GT: blue=clear, orange=inferred; A-out treats point-7 A as non-target; fixed p7/8 are training", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, .95))
            fig.savefig(HERE / f"timeline_{camera}_point_{p:02d}.png", dpi=140)
            plt.close(fig)


def markdown_table(frame, columns):
    lines = ["| " + " | ".join(label for key, label in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for key, _ in columns:
            v = row[key]
            values.append("—" if pd.isna(v) else f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_readme(summary, per_point, baselines, selected, rally_rows, elapsed):
    main = summary[summary.family.eq("shape_and_serve") & summary.scope.isin(
        ("fixed_test_09_11", "loo_all_07_11", "loo_common_09_11"))].copy()
    main["条件"] = main.camera_policy + ", A=" + main.include_rally_a.map({True:"含む",False:"除外"})
    cols = [("条件","条件"),("scope","評価範囲"),("n_rallies_matched","対応ラリー"),
            ("n_gt_rallies","正解ラリー"),("n_extra_rallies","余分"),
            ("start_mae_frames","開始MAE(F)"),("end_mae_frames","終了MAE(F)"),
            ("n_candidates","確認候補"),("recall_clear","clear再現率"),
            ("recall_all","全再現率"),("precision_proxy","適合率目安"),("f1","F1"),
            ("candidates_per_sec","候補/秒"),("candidates_per_shot","候補/打球"),
            ("candidates_per_point","候補/点"),("n_missing_shots","追加打球"),
            ("candidate_reduction_vs_raw","全場面候補から削減率")]
    primary = summary[summary.family.eq("shape_and_serve") & summary.camera_policy.eq("true_only")
                      & summary.include_rally_a]
    fixed = primary[primary.scope.eq("fixed_test_09_11")].iloc[0]
    loo = primary[primary.scope.eq("loo_all_07_11")].iloc[0]
    lines = ["# match2 第7〜11点：音と場面によるラリー区間推定", "",
             "**結論：今回の単純な音・場面規則では、サーブと最終打球の区間を十分正確に自動決定できなかった。** "
             f"Trueのみ・Aを含む条件では、固定テストのラリー重なりは {int(fixed.n_rallies_matched)}/{int(fixed.n_gt_rallies)} "
             f"だが開始MAE {fixed.start_mae_frames:.1f}F、終了MAE {fixed.end_mae_frames:.1f}F。"
             f"固定テストではIoU≥0.5も両境界±10Fも {int(fixed.n_rallies_iou50)} / {int(fixed.n_gt_rallies)}。"
             f"1点抜き全5点では候補{int(loo.n_candidates)}件（{loo.candidates_per_point:.1f}件/点）を確認し、"
             f"既存ラベルの不足{int(loo.n_missing_shots)}打球を追加する見積もり。"
             f"全True場面のv1 {int(loo.n_raw_camera_candidates)}件からは{100*loo.candidate_reduction_vs_raw:.1f}%減るが、"
             "ラベル前後だけで抽出していた前回190件より確認数が増える。区間自動化の成功や人の時間短縮は主張できない。", "",
             "本実験は、既存の暫定打球ラベルに対し、候補を確認する件数と取りこぼしを測る予備実験。"
             "人の確認時間・修正時間は実測していない。区間の終わりは最後の打球であり、ボールが落ちて得点が確定する時刻ではない。", "",
             "## 再実行", "", "リポジトリルートのPowerShellで実行：", "", "```powershell",
             r"& study\.venv\Scripts\python.exe -B -X utf8 -u outputs\shot_pipeline_20261117\rally_detect\detect_rallies.py",
             "```", "", f"今回の実行時間は {elapsed:.1f} 秒（図の生成・評価を含む）。追加パッケージなし。"
             "WAVはリポジトリ外の一時ディレクトリで生成・削除し、残さない。書込先はこのフォルダだけ。", "",
             "## 入力と推定の分離", "",
             "点区間は point_inventory.csv の [display_from_frame, next_display_from_frame)。"
             "場面は scenes/scenes.csv を直接読み、effective_scenes.csv・poseは使用しない。"
             "音声は各点全体に前後1.5秒を付けて48 kHz monoで抽出する。v1の固定設定"
             "（4–12 kHz、2 msエネルギー/1 ms刻み、0.5秒局所中央値との差、しきい値1、最小間隔0.15秒）"
             "で全区間からピークを出す。ラベルから抽出範囲を決めない。フィルタ・ピーク探索は点全体で行い、"
             "整数候補Fの場面所属でTrue/uncertainを分ける。場面の境界を越えて塊を結合しない。"
             "Falseの音も図の背景用に計測するが、ラリー推定には渡さない。", "",
             "特徴は既存v2の add_features をそのまま呼ぶ（半値幅、低/高域の上昇、両帯域上昇、"
             "低域ピーク遅延、50 ms後の残存/低下、20/100 ms集中度、スペクトル平坦度）。"
             "新しい全点候補を candidate_features.csv に保存。候補時刻は変更しない。", "",
             "図の音の値は1F内のv1スコアの最大値（鋭いピークを保持）。"
             "推定自体は元の1 ms刻みで行い、図用の集約値を使わない。", "",
             "ラリー推定関数の入力は候補と規則のみ。ラベルは規則の訓練選択と評価だけに渡す。"
             "ラベル内の各rallyの最初/最後のcontact_frameを正解区間とし、両端を含む。"
             "第7点Aを除外する条件ではAを検出対象外とし、そこだけに出た検出は余分と数える。"
             "Aの位置で推定を削る等の正解利用はしない。Aを含む/除く条件は別々に訓練選択する。", "",
             "## 事前に固定した探索と選択基準", "",
             "G=0.8,1.0,1.2,1.4,1.6,1.8,2.0秒、K=3,4,5,6。"
             "G秒以下で隣接候補を連結し、K候補以上を採る。形状規則は none / sharp / v2 の3種類。"
             "sharpは半値幅≤15 msかつ50 ms後の対数エネルギー低下≥1、v2はさらに両帯域上昇≥0.5。"
             "形状除去は連結の前に行い、拍手等でつながる塊を切る機会を与える。", "",
             "サーブ前除去は none / rhythmic_jump1 / rhythmic_jump2。先頭の3〜8候補が"
             "0.4〜1.1秒の間隔、間隔の変動係数≤0.3、先頭から新開始候補まで≤4秒を満たし、"
             "新開始候補の高域上昇が先頭部分の中央値より1または2以上、全先頭候補より大きく、"
             "両帯域上昇も先頭中央値より0.5以上大きい最初の候補までを除去する。"
             "除去後にもK件を要求する。これは『弱い等間隔の先頭音』という仮説であり、"
             "ボールつきの識別正解があるわけではない。塊の最後の採用候補を終了とする。", "",
             "cluster_v1（G,Kのみ、28通り）、shape_only（84通り）、shape_and_serve（252通り）を"
             "それぞれ訓練だけで選択し、比較する。主結果はshape_and_serve（不要な規則なしも選択可）。"
             "選択指標は候補F1とラリーIoUを用いる次式を等重みで最大化：", "",
             "`0.5 × 候補F1 + 0.5 × 2Σ(対応ラリーのIoU)/(正解ラリー数+検出ラリー数)`", "",
             "同点はclear一致数が多い、開始/終了MAE和が小さい、候補が少ない、規則が単純、ID順。"
             "この基準は本スクリプトに固定し、テストの出力を見て修正しない。高再現率の保証や時間最小化の基準ではない。", "",
             "今回、探索した先頭除去規則は全条件で除去を発動せず、主規則はすべてserve_trim=noneが選ばれた。"
             "shape_onlyとshape_and_serveの結果は同じ。したがって候補削減は音形状規則等の効果で、"
             "サーブ前ボールつきの除去ができた結果ではない。この保守的な等間隔規則が機能しなかったことを示し、"
             "強さの跳びだけを使う等の別規則まで否定する結果ではない。テスト観察後の規則追加・再選択は行っていない。", "",
             "固定分割は第7・8点で選択し、第9〜11点を評価。1点抜きは4点で選択して残り1点を評価し、"
             "5回のテスト予測を合算する。率は点平均ではなく件数合算。比較用に1点抜きの第9〜11点だけも併記。"
             "Trueのみ/True+uncertainの各条件で同じ探索を独立実施する。", "",
             "## 評価定義", "",
             "ラリーは同一点内で閉区間が1F以上重なれば対応候補。最大対応数、次にIoU和最大の1対1割当。"
             "単なる重なり再現率も別列で保存し、分裂・結合・IoU≥0.5・両境界±10F以内も記録する。"
             "開始/終了のずれは検出−正解（正=遅れ）。MAEは対応したラリーのみで、見逃しは分母に入らない。"
             "未対応検出は、正解と重なる断片(extra_fragment)と、正解ラリーとの重なりなし"
             "(extra_no_labeled_rally)に分け、場所と場面IDを保存する。後者がリプレイ・拍手かは自動断定しない。", "",
             "候補は検出区間を形成した、形状/先頭規則を通過した候補。打球は同一点・±10F以内で"
             "最大対応数、次に絶対ずれ和最小の1対1割当。clear再現率も全ラベル共通割当に基づく。"
             "適合率目安=一致数/候補数、F1=2一致数/(候補数+全ラベル数)。"
             "候補/秒の分母は検出ラリーの閉区間長の和、候補/打球は全対象ラベル数。"
             "raw比較とoracle比較ではそれぞれ場面全長/正解ラリー長が秒の分母になるため、率の直接比較に注意。", "",
             "作業量は1候補=1確認と仮定。追加打球数=全ラベル−1対1一致数。"
             "確認+追加は件数の和で、取りこぼしを探すための全動画確認時間や区間修正は含まない。"
             "したがって候補件数の削減率を人の時間削減率と呼ばない。", "",
             "## 主結果：固定分割と1点抜き", "", markdown_table(main, cols), "",
             "## 点別の確認数と追加数（主規則）", ""]
    pp = per_point[per_point.family.eq("shape_and_serve")].copy()
    lines.append(markdown_table(pp, [("camera_policy","場面"),("include_rally_a","Aを含む"),
        ("protocol","分割"),("partition","train/test"),("point","点"),("config_id","規則"),
        ("n_candidates","確認候補"),("n_missing_shots","追加打球"),("missing_clear_shots","追加clear"),
        ("f1","F1"),("n_extra_rallies","余分ラリー")]))
    lines += ["", "## 規則追加の比較（Trueのみ・Aを含む）", ""]
    abl = summary[summary.camera_policy.eq("true_only") & summary.include_rally_a & summary.scope.isin(
        ("fixed_test_09_11", "loo_all_07_11"))]
    lines.append(markdown_table(abl, [("family","規則群"),("scope","範囲"),("n_candidates","候補"),
        ("n_missing_shots","追加"),("recall_clear","clear再現"),("f1","F1"),
        ("n_extra_rallies","余分ラリー"),("start_mae_frames","開始MAE"),("end_mae_frames","終了MAE")]))
    lines += ["", "## 前回値とラリー正解を使う上限参照", "",
        "依頼時の参考値は旧ラベル前後±90F内のv1=190件、F1約0.59〜0.62、"
        "旧候補を『最初の打球−15F〜最後の打球+15F』に限ると136件・F1約0.73。"
        "この±15Fは audio_learned/README.md の前回追記に記載されていた。"
        "今回は正解区間を最初〜最後の打球そのものと指定されているので、"
        "oracle_gt_rallyは追加幅0F、oracle_gt_rally_plusminus15が前回を再現する条件。"
        "追加幅0Fだと旧候補は120件となり、136件と食い違う原因は区間定義である。"
        "前回の104件はラベル±10F内の候補の多対1集計であり、1対1一致数ではない。"
        "以下は既存190件と新規の全場面候補を、"
        "今回と同じ±10F・最大数1対1で再評価したもの。旧方式の最近傍greedy割当と一致数が異なる可能性がある。"
        "実際に136件の条件は旧greedyで83一致・F1=0.731、今回の最大対応数割当で84一致・F1=0.740となる。"
        "oracleはいずれも正解区間を利用する参考値で、自動ラリー検出の成績ではない。"
        "A除外oracleはAの候補も正解を使って除くが、自動方式のA除外はそこを負例として数える。", ""]
    b = baselines[baselines.scope.eq("all_07_11")]
    lines.append(markdown_table(b, [("camera_policy","入力"),("include_rally_a","Aを含む"),
        ("baseline","区間"),("n_candidates","候補"),("recall_clear","clear再現"),
        ("recall_all","全再現"),("precision_proxy","適合率目安"),("f1","F1"),("legacy_greedy_f1","旧greedy F1"),
        ("candidates_per_shot","候補/打球"),("candidates_per_point","候補/点"),("n_missing_shots","追加")]))
    lines += ["", "## 選択された主規則", ""]
    sf = pd.DataFrame(selected)
    lines.append(markdown_table(sf[sf.family.eq("shape_and_serve")], [("camera_policy","場面"),
        ("include_rally_a","Aを含む"),("protocol","分割"),("gap_sec","G秒"),
        ("min_candidates","K"),("shape","形状"),("serve_trim","先頭除去")]))
    lines += ["", "## 主規則のラリー対応と余分検出の場所", ""]
    rr = pd.DataFrame(rally_rows)
    rr = rr[rr.family.eq("shape_and_serve")]
    # Keep an explicit compact record in prose; all families are in CSV.
    lines.append(markdown_table(rr.astype(object).where(rr.notna(), ""), [("camera_policy","場面"),("include_rally_a","Aを含む"),
        ("protocol","分割"),("partition","train/test"),("point","点"),("status","対応"),
        ("gt_rally","正解"),("scene_id","検出場面"),("start_frame","検出開始"),("end_frame","検出終了"),
        ("start_delta_frames","開始ずれF"),("end_delta_frames","終了ずれF"),("iou","IoU")]))
    lines += ["", "## 研究上の結論と限界", "",
        "- ラベルはClaudeによる暫定ラベルで、人が独立検証した完全な正解ではない。clearも独立正解ではない。"
        "未ラベルの実打球が候補にあると適合率目安を過小評価し、未ラベル打球の見逃しは測れない。",
        "- v1の固定設定は第7・8点で選択済み。したがって1点抜きはラリー規則のみの条件付き交差検証で、"
        "第7・8点をテストにした回にも上流の設定選択が残る。全パイプラインの独立汎化評価ではない。",
        "- 場面分類の参照に第9・10点自身の打球フレームが使われている（scenes/README.md）。"
        "固定分割も場面分類を含めた完全独立テストではない。既存の第9点の音の観察に基づくv2仮説もあり、"
        "再利用された探索的評価である。第7点p07_s012と第11点p11_s001以外は俯瞰判定の目視確認が限定される。",
        "- Trueは俯瞰角度の支持であり、ライブである保証はない。uncertainの包含で候補/余分ラリーが増えても、"
        "それだけで拍手やリプレイを識別できたとはいえない。ラベル外ラリーの実在も未確認。",
        "- 単なるラリー重なりは開始/終了を正しく推定できたことを意味しない。IoU・境界ずれ・見逃しを併せて判断する。"
        "音だけではボールつき・バウンド・ラケット接触を安定識別できない可能性があり、規則追加の効果は"
        "同じテスト点のcluster_v1/shape_only比較で判断する。",
        "- 5点・同一試合の小標本であり、新規性や実用的な時間短縮はこの結果だけでは示せない。"
        "次は独立に確認した別の連続区間で、候補確認・不足探索・区間修正を含む人の所要時間を映像のみと比較する。", "",
        "## 出力", "",
        "- `summary.csv`：固定訓練/固定テスト/固定全点参考/1点抜き全点/共通第9〜11点、全条件・3規則群。",
        "- `per_point_metrics.csv`：点別の確認候補、追加打球、clear/all再現率、F1、境界/分裂/結合。",
        "- `rally_matches.csv`：正解対応、開始/終了の符号付きずれ、見逃し、余分検出の場面とフレーム。",
        "- `rally_overlap_edges.csv`：全重なりの辺。1対1対応から漏れた分裂・結合も監査可能。",
        "- `label_matches.csv` / `selected_candidates.csv`：打球ごとの不足と、実際に確認対象となる候補。",
        "- `candidate_features.csv`：全点を抽出したv1候補と既存v2と同じ特徴。Falseも監査用に保存。",
        "- `baseline_comparison.csv`：旧v1・新全場面v1・正解区間内oracleの統一指標。",
        "- `selected_configs.json` / `param_search.csv`：各foldで訓練だけを使った選択と全探索。",
        "- `timeline_{true_only|true_plus_uncertain}_point_XX.png`：場面、音の値、候補、GT、固定/1点抜きの区間。"
        "A含む/除外の予測は別段。青=clear、橙=inferred、GT Aは斜線。",
        "- `protocol.json` / `run_metadata.json`：探索定義・入力ハッシュ・入力不変確認・実行環境。",
        "- `verification.json`：保存CSVの件数・一意性・境界と、別実装DPによる全120点/モデル行の照合検証。",
        "- 256 KiB超の生成物はこのフォルダの `.gitignore` で除外。再実行で再生成する。", ""]
    (HERE / "README.md").write_text("\n".join(lines), encoding="utf-8")


def self_check():
    # Greedy can lose matches: l0=0,l1=9,c0=-9,c1=1 -> maximum cardinality is 2.
    lab = pd.DataFrame({"contact_frame":[0,9]})
    cand = pd.DataFrame({"candidate_frame":[-9,1]})
    assert len(match_shots(lab, cand)) == 2
    for rule in ("rhythmic_jump1", "rhythmic_jump2"):
        c = pd.DataFrame({"candidate_frame_float": np.array([0,.6,1.2,1.8])/v1.FRAME_SEC,
                          "high_rise_ln":[1,1.1,1,4], "paired_rise_ln":[.5,.6,.5,2]})
        assert prefix_length(c, rule) == 3
        c.loc[1, "candidate_frame_float"] = .1/v1.FRAME_SEC
        assert prefix_length(c, rule) == 0
    # The exact G boundary joins; a new scene never joins, even at a short gap.
    frames = np.array([0, .8, 1.6, 1.7, 2.5, 3.3])/v1.FRAME_SEC
    c = pd.DataFrame({"point": 7, "candidate_frame_float": frames,
                      "candidate_frame": np.floor(frames+.5).astype(int),
                      "scene_id": ["s1"]*3+["s2"]*3, "camera_class": "True"})
    cfg = dict(gap_sec=.8, min_candidates=3, shape="none", serve_trim="none")
    kept, pred = detect(c, cfg)
    assert len(kept) == 6 and len(pred) == 2
    assert all(r["n_candidates"] == 3 for r in pred)
    cfg["gap_sec"] = .799
    assert len(detect(c, cfg)[1]) == 0


def verify_saved_results(labels):
    """Audit exported CSVs with a separate ordered-matching DP implementation."""
    metrics = pd.read_csv(HERE / "per_point_metrics.csv")
    candidates = pd.read_csv(HERE / "selected_candidates.csv")
    shots = pd.read_csv(HERE / "label_matches.csv")
    rallies = pd.read_csv(HERE / "rally_matches.csv")
    keys = ["camera_policy", "include_rally_a", "protocol", "family", "point"]
    checked = 0
    for row in metrics.itertuples():
        def subset(table):
            mask = np.ones(len(table), dtype=bool)
            for key in keys:
                mask &= table[key].eq(getattr(row, key)).to_numpy()
            return table[mask]
        cp, sp, rp = subset(candidates), subset(shots), subset(rallies)
        assert len(cp) == row.n_candidates and cp.candidate_id.is_unique
        assert len(sp) == row.n_labels_all and sp.label_id.is_unique
        matched = sp[sp.matched]
        assert len(matched) == row.n_matched_all and matched.candidate_id.is_unique
        assert int((matched.confidence == "clear").sum()) == row.n_matched_clear
        assert len(rp[rp.status != "missed_gt"]) == row.n_detected_rallies
        labs = labels_for(labels, row.point, row.include_rally_a)
        x = sorted(labs.contact_frame.tolist())
        y = sorted(cp.candidate_frame.tolist())
        # Ordered matching maximizes count, independently of Hungarian costs.
        dp = np.zeros((len(x)+1, len(y)+1), dtype=int)
        for i, a in enumerate(x, 1):
            for j, b in enumerate(y, 1):
                dp[i, j] = max(dp[i-1, j], dp[i, j-1],
                               dp[i-1, j-1]+1 if abs(a-b) <= 10 else 0)
        assert int(dp[-1, -1]) == row.n_matched_all
        for r in rp[rp.status != "missed_gt"].itertuples():
            c = cp[cp.scene_id.eq(r.scene_id) & cp.candidate_frame.between(r.start_frame, r.end_frame)]
            assert len(c) == r.n_candidates
            assert int(c.candidate_frame.min()) == r.start_frame
            assert int(c.candidate_frame.max()) == r.end_frame
        checked += 1
    baseline = pd.read_csv(HERE / "baseline_comparison.csv")
    old15 = baseline[(baseline.camera_policy == "historical") & baseline.include_rally_a &
                     (baseline.scope == "all_07_11") & (baseline.baseline == "oracle_gt_rally_plusminus15")].iloc[0]
    assert old15.n_candidates == 136 and abs(old15.f1 - 168/227) < 1e-12
    assert old15.legacy_greedy_matched_all == 83 and abs(old15.legacy_greedy_f1-166/227) < 1e-12
    dump(HERE / "verification.json", dict(passed=True, verified_point_model_rows=checked,
         independent_matching="ordered dynamic programming maximum-cardinality +/-10F",
         checks=["CSV candidate and label uniqueness", "exported counts equal point metrics",
                 "clear/all matches and missing counts", "rally interval endpoints and candidate membership",
                 "old 136-candidate +/-15F baseline: legacy greedy F1=166/227, maximum matching F1=168/227"],
         historical_reproduced_candidates=int(old15.n_candidates), historical_reproduced_f1=float(old15.legacy_greedy_f1),
         historical_maximum_matching_f1=float(old15.f1)))


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", type=Path, default=Path(r"C:\ffmpeg\bin\ffmpeg.exe"))
    args = parser.parse_args()
    self_check()
    inventory, scenes, labels, video, config = load_inputs()
    inputs = [ROOT / "outputs/match2_point_inventory_20261117/point_inventory.csv",
              PIPE / "scenes/scenes.csv", PIPE / "scenes/README.md",
              PIPE / "audio_learned/README.md",
              PIPE / "tracking/point_07/protocol.json",
              *[PIPE / f"labels/point_{p:02d}.csv" for p in POINTS],
              *[PIPE / f"audio/{name}" for name in ("detect_hit_sounds.py", "detect_hit_sounds_v2.py",
                 "selected_config.json", "selected_config_v2.json", "candidates.csv", "intervals.csv",
                 "evaluation.csv", "candidate_features_v2.csv")]]
    before = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    configs = configurations()
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), points=list(POINTS),
                    frame_convention="0-based closed frame intervals; inventory upper bound exclusive",
                    fps="60000/1001", v1_config=config.__dict__,
                    search=dict(gap_sec=G_GRID, min_candidates=K_GRID, shape=SHAPES, serve_trim=TRIMS),
                    selection_score=SCORE_DEFINITION,
                    tie_breaks=["max matched clear", "min start/end MAE sum", "min candidates", "min complexity", "config ID"],
                    candidate_matching="max cardinality then min absolute error, same point, inclusive +/-10F",
                    rally_matching="max cardinality then max IoU, positive closed-interval overlap, same point",
                    include_a_false="A is non-target; A-only predictions are extras, not ignored",
                    label_input_to_inference=False, pose_used=False,
                    caveats=["v1 fixed upstream fitted on 7/8; LOO tests rally rule only",
                             "scene references include labels from 9/10; not independent pipeline holdout",
                             "prior v2 hypotheses include observation of point 9; exploratory reused test",
                             "Claude labels not independent complete human ground truth"])
    dump(HERE / "protocol.json", protocol)  # design recorded before scoring test folds
    candidates, traces = extract(inventory, scenes, video, args.ffmpeg, config)
    candidates.to_csv(HERE / "candidate_features.csv", index=False)
    traces.to_csv(HERE / "audio_trace.csv", index=False)
    output = fit_and_evaluate(candidates, scenes, labels, configs)
    metrics, searches, selected, rallies, shots, edges, selected_candidates, plot_data = output
    per_point = pd.DataFrame(metrics)
    summary = summarize(per_point)
    baselines = baseline_tables(candidates, labels, inventory, scenes)
    for filename, data in (("per_point_metrics.csv", per_point), ("summary.csv", summary),
                           ("baseline_comparison.csv", baselines), ("param_search.csv", searches),
                           ("rally_matches.csv", rallies), ("label_matches.csv", shots),
                           ("rally_overlap_edges.csv", edges), ("selected_candidates.csv", selected_candidates)):
        pd.DataFrame(data).to_csv(HERE / filename, index=False, encoding="utf-8")
    dump(HERE / "selected_configs.json", {"selection_definition": SCORE_DEFINITION, "folds": selected})
    verify_saved_results(labels)
    figures(inventory, scenes, traces, candidates, labels, plot_data)
    elapsed = time.perf_counter()-start
    write_readme(summary, per_point, baselines, selected, rallies, elapsed)
    after = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    assert before == after, "Existing input changed"
    assert not list(HERE.rglob("*.wav"))
    # Verify accounting of every produced point/fold and workload estimate.
    assert len(per_point) == 4*3*10
    assert (per_point.n_matched_all <= per_point.n_labels_all).all()
    assert (per_point.n_matched_all <= per_point.n_candidates).all()
    assert np.all(per_point.n_missing_shots == per_point.n_labels_all-per_point.n_matched_all)
    assert sum(r["n_prefix_removed"] for r in searches) == 0, "Update report: serve trimming now fires"
    generated = [p for p in HERE.iterdir() if p.is_file() and p.name not in (".gitignore", "detect_rallies.py")]
    large = sorted(p.name for p in generated if p.stat().st_size > 256*1024)
    (HERE / ".gitignore").write_text("# Regenerable artifacts >256 KiB\n" + "\n".join("/"+n for n in large)
                                     + "\n__pycache__/\n*.wav\n", encoding="utf-8")
    metadata = dict(completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_sec=elapsed,
                    python=sys.version, numpy=np.__version__, scipy=__import__("scipy").__version__,
                    pandas=pd.__version__, matplotlib=matplotlib.__version__, source_video=str(video),
                    source_video_bytes=video.stat().st_size, source_video_mtime_ns=video.stat().st_mtime_ns,
                    ffmpeg_version=subprocess.run([str(args.ffmpeg), "-version"], check=True,
                                                   capture_output=True, text=True).stdout.splitlines()[0],
                    script_sha256=sha(__file__), input_sha256=before, inputs_unchanged=before==after,
                    temporary_wav_directory_removed=True, n_full_point_candidates=len(candidates),
                    n_folds=len(selected), ignored_large_artifacts=large,
                    checks=["matching cardinality counterexample", "regular prefix positive/negative controls",
                            "exact G boundary and scene isolation",
                            "scene partition coverage", "sample count and timebase", "metric conservation",
                            "all expected point/fold outputs", "input hashes unchanged", "temporary WAV cleanup"])
    dump(HERE / "run_metadata.json", metadata)
    primary = summary[summary.camera_policy.eq("true_only") & summary.include_rally_a &
                      summary.family.eq("shape_and_serve") & summary.scope.isin(("fixed_test_09_11", "loo_all_07_11"))]
    print(primary[["scope", "n_candidates", "n_missing_shots", "recall_clear", "f1", "n_extra_rallies",
                   "start_mae_frames", "end_mae_frames"]].to_string(index=False), flush=True)
    print(f"Complete in {elapsed:.1f}s; existing inputs unchanged; no WAV retained.", flush=True)


if __name__ == "__main__":
    main()
