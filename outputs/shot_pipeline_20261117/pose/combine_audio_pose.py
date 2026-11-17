"""CPU audio/YOLO-pose fusion; train on points 7/8, freeze, then test on 9/10/11.

All input files are read-only. Only this new pose directory is written.
Partial inference caches are supported for a plumbing check, never a valid result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import itertools
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent
SLOTS = ["near_front", "near_rear", "far_rear", "far_front"]
SIDES = {s: "N" if s.startswith("near") else "F" for s in SLOTS}
COLORS = ["#16804a", "#238bce", "#ac4cb0", "#c79500"]
METHODS = ["audio_v1", "audio_v1_A_previous", "audio_v1_wrist", "audio_v1_wrist_alternating"]
TRAIN, TEST = [7, 8], [9, 10, 11]
PEAK_RADIUS, MIN_GAP = 8, 12
FEATURE_VERSION = "relative_wrist_torso_adjacent_v1"

# Reuse the exact previous matching and window accounting; no previous writer
# or main() is called. Loading this module only defines functions/constants.
spec = importlib.util.spec_from_file_location("previous_av_readonly", PIPE / "audiovisual/combine_audio_video.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_csv(name, table):
    table.to_csv(HERE / name, index=False, encoding="utf-8-sig")


def save_json(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_inputs():
    intervals, _, _, fps = previous.load_inputs()
    candidates = pd.read_csv(PIPE / "audiovisual/candidate_features.csv")
    baseline = pd.read_csv(PIPE / "audio/candidates.csv")
    keys = ["candidate_id", "point", "rally", "candidate_frame"]
    pd.testing.assert_frame_equal(candidates[keys], baseline[keys])
    assert candidates.candidate_id.is_unique
    # Strip prior pixel-motion features so no ambiguous legacy column can enter
    # wrist fitting. Original sound shape fields remain for provenance only.
    sound_columns = [c for c in candidates.columns if not c.startswith(tuple(SLOTS))
                     and c not in ["motion_peak_height", "visual_hitter_slot", "visual_side", "side_reason"]]
    candidates = candidates[sound_columns].copy()
    prior_selected = pd.read_csv(PIPE / "audiovisual/selected_candidates.csv")
    a_ids = set(prior_selected.loc[prior_selected.method == "audio_v1_A", "candidate_id"])
    assert a_ids <= set(candidates.candidate_id)
    return intervals, candidates, a_ids, fps


def load_pose(intervals):
    manifest_path = HERE / "extraction_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("Pose extraction did not finish; rerun extract_pose.py before evaluation")
    for relative, digest in manifest["input_sha256"].items():
        path = PIPE / relative
        if sha(path) != digest:
            raise ValueError(f"Stale pose extraction input: {path}")
    video_stat = Path(manifest["source_video"]).stat()
    if (video_stat.st_size != manifest["video_bytes"]
            or video_stat.st_mtime_ns != manifest["video_mtime_ns"]):
        raise ValueError("Source video changed since pose extraction")
    if sha(Path(manifest["config"]["model"])) != manifest["model_sha256"]:
        raise ValueError("Pose model changed since extraction")
    entries = {(int(r["point"]), str(r["rally"])): r for r in manifest["intervals"]}
    frames, coverage, hashes = [], [], {}
    for seg in intervals.itertuples():
        key = (int(seg.point), str(seg.rally))
        if key not in entries:
            raise ValueError(f"Extraction manifest has no interval {key}")
        entry = entries[key]
        path = HERE / "pose_cache" / f"point_{seg.point:02d}_rally_{seg.rally}.csv"
        table = pd.read_csv(path)
        required = {"point", "rally", "source_frame", "player_slot", "track_id", "detection_valid"}
        required |= {f"kp_{i}_{axis}" for i in range(17) for axis in ["x", "y", "conf"]}
        if not required <= set(table.columns):
            raise ValueError(f"{path}: missing columns {sorted(required-set(table.columns))}")
        assert not table.duplicated(["source_frame", "player_slot"]).any(), path
        assert table.player_slot.isin(SLOTS).all() and (table.point == seg.point).all()
        assert (table.rally == seg.rally).all()
        frame_ids = sorted(table.source_frame.unique())
        processed = int(entry["processed_frames"])
        assert frame_ids == list(range(int(seg.start_frame), int(seg.start_frame)+processed)), path
        assert len(table) == processed*4, (path, len(table), processed)
        assert table.groupby("source_frame").player_slot.nunique().eq(4).all()
        assert int(entry["start_frame"]) == seg.start_frame and int(entry["end_frame"]) == seg.end_frame
        assert int(entry["processed_end_frame"]) == seg.start_frame+processed
        complete = processed == seg.end_frame-seg.start_frame
        assert bool(entry["complete"]) == complete
        hashes[f"{seg.point:02d}_{seg.rally}"] = sha(path)
        if hashes[f"{seg.point:02d}_{seg.rally}"] != entry["cache_sha256"]:
            raise ValueError(f"Pose cache hash does not match extraction manifest: {path}")
        coverage.append(dict(point=seg.point, rally=seg.rally, expected_frames=seg.end_frame-seg.start_frame,
                             processed_frames=processed, frame_coverage=processed/(seg.end_frame-seg.start_frame),
                             complete=complete))
        frames.append(table)
    complete = all(r["complete"] for r in coverage)
    assert bool(manifest["full_run"]) == complete
    return pd.concat(frames, ignore_index=True), pd.DataFrame(coverage), hashes, manifest


def joint(table, number, threshold):
    xy = table[[f"kp_{number}_x", f"kp_{number}_y"]].to_numpy(dtype=float, copy=True)
    confidence = table[f"kp_{number}_conf"].to_numpy(float)
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(confidence) & (confidence >= threshold)
    xy[~valid] = np.nan
    return xy, valid


def read_only_input_hashes(include_test_labels=False):
    paths = [PIPE / "audiovisual" / name for name in ["combine_audio_video.py", "candidate_features.csv", "selected_candidates.csv", "evaluation.csv", "audio_plot_timeseries.csv"]]
    paths += [PIPE / "audio" / name for name in ["intervals.csv", "candidates.csv", "candidates_v2.csv", "candidate_features_v2.csv"]]
    paths += [PIPE / f"labels/point_{p:02d}.csv" for p in (TRAIN+TEST if include_test_labels else TRAIN)]
    paths += [PIPE / f"tracking/point_{p:02d}/{name}" for p in TRAIN+TEST for name in ["protocol.json", "player_court_proxies.csv"]]
    return {str(path.relative_to(PIPE)).replace("\\", "/"): sha(path) for path in paths}


def assert_input_hashes(expected):
    changed = [name for name, digest in expected.items() if sha(PIPE / name) != digest]
    if changed:
        raise RuntimeError(f"Read-only inputs changed during this run: {changed}")


def wrist_timeseries(pose, threshold, smooth):
    """No interpolation; adjacent frames and identical anchor/track are required."""
    output, diagnostics = [], []
    for (point, rally, slot), group in pose.groupby(["point", "rally", "player_slot"], sort=False):
        g = group.sort_values("source_frame").reset_index(drop=True)
        shoulders = [joint(g, i, threshold) for i in [5, 6]]
        hips = [joint(g, i, threshold) for i in [11, 12]]
        shoulder_valid = shoulders[0][1] & shoulders[1][1]
        hip_valid = hips[0][1] & hips[1][1]
        shoulder = (shoulders[0][0]+shoulders[1][0])/2
        hip = (hips[0][0]+hips[1][0])/2
        torso = np.linalg.norm(shoulder-hip, axis=1)
        torso[~(shoulder_valid & hip_valid) | (torso <= 1)] = np.nan
        median_torso = float(np.nanmedian(torso)) if np.isfinite(torso).any() else np.nan
        anchor_kind = np.where(hip_valid, "hip", np.where(shoulder_valid, "shoulder", "missing"))
        anchor = np.where(hip_valid[:, None], hip, shoulder)
        # With missing hips an instantaneous torso cannot be measured. The
        # label-free median torso of this player's interval supplies the scale.
        scale = np.where(np.isfinite(torso), torso, median_torso)
        scale_mode = np.where(np.isfinite(torso), "instantaneous", "interval_median")
        frames = g.source_frame.to_numpy(int)
        tracks = g.track_id.fillna(-1).to_numpy()
        adjacent = np.r_[False, np.diff(frames) == 1]
        same_track = np.r_[False, (tracks[1:] == tracks[:-1]) & (tracks[1:] != -1)]
        same_anchor = np.r_[False, anchor_kind[1:] == anchor_kind[:-1]]
        # Avoid a false velocity caused by a change of normalization convention.
        same_scale_mode = np.r_[False, scale_mode[1:] == scale_mode[:-1]]
        row = g[["point", "rally", "source_frame", "player_slot", "track_id", "detection_valid"]].copy()
        row["side"] = SIDES[slot]
        row["anchor_kind"] = anchor_kind
        row["torso_px"] = torso
        row["normalization_torso_px"] = scale
        row["scale_mode"] = scale_mode
        for name, number in [("left", 9), ("right", 10)]:
            xy, valid = joint(g, number, threshold)
            relative = (xy-anchor)/scale[:, None]
            valid &= np.isfinite(relative).all(axis=1) & (anchor_kind != "missing")
            relative[~valid] = np.nan
            velocity = np.linalg.norm(relative-np.roll(relative, 1, axis=0), axis=1)
            ok = adjacent & same_track & same_anchor & same_scale_mode & valid & np.r_[False, valid[:-1]]
            velocity[~ok] = np.nan
            row[f"{name}_relative_x"] = relative[:, 0]
            row[f"{name}_relative_y"] = relative[:, 1]
            row[f"{name}_wrist_speed_raw"] = velocity
            series = pd.Series(velocity)
            # Smoothing can use neighbours but never fill an invalid centre or
            # cross a track/anchor boundary. Each finite contiguous run is local.
            runs = (series.isna() | ~pd.Series(adjacent & same_track & same_anchor & same_scale_mode)).cumsum()
            smoothed = series.groupby(runs).transform(lambda x: x.rolling(smooth, center=True, min_periods=1).median())
            row[f"{name}_wrist_speed"] = smoothed.where(series.notna()).to_numpy()
        speeds = row[["left_wrist_speed", "right_wrist_speed"]]
        row["wrist_speed"] = speeds.max(axis=1, skipna=True)
        median = float(row.wrist_speed.median()) if row.wrist_speed.notna().any() else np.nan
        row["interval_wrist_median"] = median
        row["wrist_excess"] = row.wrist_speed-median
        diagnostics.append(dict(point=point, rally=rally, player_slot=slot, processed_frames=len(g),
                                valid_wrist_speed=int(row.wrist_speed.notna().sum()),
                                valid_wrist_speed_fraction=float(row.wrist_speed.notna().mean()),
                                valid_torso_frames=int(np.isfinite(torso).sum()), median_torso_px=median_torso,
                                shoulder_anchor_frames=int((anchor_kind == "shoulder").sum()),
                                interval_wrist_median=median, known_tracking_issue=point == 8 and slot.startswith("near")))
        output.append(row)
    return pd.concat(output, ignore_index=True), pd.DataFrame(diagnostics)


def add_wrist_features(candidates, timeseries, radius=PEAK_RADIUS):
    table = candidates.copy()
    for slot in SLOTS:
        for field in ["peak_height", "peak_frame", "interval_median"]:
            table[f"{slot}_{field}"] = np.nan
        table[f"{slot}_peak_n_valid"] = 0
    for (point, rally, slot), group in timeseries.groupby(["point", "rally", "player_slot"]):
        series = group.set_index("source_frame")
        for idx, row in table[(table.point == point) & (table.rally == rally)].iterrows():
            f = int(row.candidate_frame)
            peak = series.loc[f-radius:f+radius, "wrist_excess"].dropna()
            table.loc[idx, f"{slot}_peak_n_valid"] = len(peak)
            table.loc[idx, f"{slot}_interval_median"] = series.interval_wrist_median.iloc[0]
            if len(peak):
                table.loc[idx, f"{slot}_peak_height"] = float(peak.max())
                table.loc[idx, f"{slot}_peak_frame"] = int(peak.idxmax())
    table["wrist_peak_height"] = np.nan
    table["wrist_second_peak_height"] = np.nan
    table["wrist_peak_gap"] = np.nan
    table["wrist_side_gap"] = np.nan
    table["n_players_with_peak"] = 0
    table["n_sides_with_peak"] = 0
    table["visual_hitter_slot"] = ""
    table["visual_side"] = ""
    table["side_reason"] = "no_positive_peak_or_both_sides_unavailable"
    for idx, row in table.iterrows():
        peaks = np.array([row[f"{s}_peak_height"] for s in SLOTS], float)
        finite = np.flatnonzero(np.isfinite(peaks))
        sides = {SIDES[SLOTS[i]] for i in finite}
        table.loc[idx, ["n_players_with_peak", "n_sides_with_peak"]] = [len(finite), len(sides)]
        if not len(finite):
            continue
        order = finite[np.argsort(-peaks[finite], kind="stable")]
        winner = order[0]
        maximum = float(peaks[winner])
        table.loc[idx, "wrist_peak_height"] = maximum
        if len(order) > 1:
            table.loc[idx, "wrist_second_peak_height"] = float(peaks[order[1]])
            table.loc[idx, "wrist_peak_gap"] = float(maximum-peaks[order[1]])
        if len(sides) == 2:
            side_max = {side: max(peaks[i] for i in finite if SIDES[SLOTS[i]] == side) for side in ["N", "F"]}
            table.loc[idx, "wrist_side_gap"] = abs(side_max["N"]-side_max["F"])
        unique = sum(np.isclose(peaks[finite], maximum, rtol=0, atol=1e-12)) == 1
        if maximum > 0 and unique and len(sides) == 2:
            table.loc[idx, ["visual_hitter_slot", "visual_side", "side_reason"]] = [SLOTS[winner], SIDES[SLOTS[winner]], "unique_positive_wrist_peak_both_sides_observed"]
    table["known_point8_near_tracking_problem"] = table.point == 8
    return table


def evaluate(intervals, labels, selected, scope, method):
    result, details = previous.evaluate(intervals, labels, selected, scope=scope, method=method)
    result["side_source"] = "relative_wrist_peak; DP assignments excluded"
    result["side_correct_over_all_time_matches"] = previous.divide(result["n_side_correct"], result["n_side_time_matched"])
    matches = previous.match_labels(labels, selected, 10)
    cs = selected.set_index("candidate_id")
    ls = labels.set_index("label_id")
    accuracies = []
    for side in ["N", "F"]:
        matched = [(lid, cid) for lid, (cid, _) in matches.items() if ls.loc[lid, "hitter_side"] == side]
        known = sum(cs.loc[cid, "visual_side"] in ["N", "F"] for _, cid in matched)
        correct = sum(cs.loc[cid, "visual_side"] == side for _, cid in matched)
        result.update({f"n_side_{side}_matched": len(matched), f"n_side_{side}_known": known,
                       f"n_side_{side}_correct": correct, f"side_{side}_accuracy": previous.divide(correct, known),
                       f"side_{side}_coverage": previous.divide(known, len(matched)),
                       f"side_{side}_correct_over_matched": previous.divide(correct, len(matched))})
        accuracies.append(previous.divide(correct, known))
    result["side_balanced_accuracy"] = float(np.mean(accuracies)) if np.isfinite(accuracies).all() else np.nan
    result["n_candidates_wrist_any"] = int((selected.n_players_with_peak > 0).sum())
    result["n_candidates_wrist_both_sides"] = int((selected.n_sides_with_peak == 2).sum())
    result["candidate_wrist_both_sides_fraction"] = previous.divide(result["n_candidates_wrist_both_sides"], len(selected))
    result["label_majority_side_baseline"] = previous.divide(max((labels.hitter_side == "N").sum(), (labels.hitter_side == "F").sum()), len(labels))
    return result, details


def score(table, scales, config):
    table = table.copy()
    table["audio_score"] = np.log1p(table.strength.clip(lower=0))/scales["audio"]
    table["wrist_score"] = (np.log1p(table.wrist_peak_height.clip(lower=0))/scales["wrist"]).fillna(0.)
    table["combined_score"] = table.wrist_score+config["audio_weight"]*table.audio_score-config["threshold"]
    return table


def select(table, method, scales=None, config=None, a_ids=None):
    if method == METHODS[0]:
        return table.copy()
    if method == METHODS[1]:
        return table[table.candidate_id.isin(a_ids)].copy()
    scored = score(table, scales, config)
    if method == METHODS[2]:
        return scored[scored.combined_score > 0].copy()
    return previous.dp_select(scored)


def train(features, intervals, labels, provenance, feature_config):
    scales = {}
    for name, values in [("audio", features.strength), ("wrist", features.wrist_peak_height)]:
        values = np.log1p(values.clip(lower=0)).dropna()
        median = float(values.median()) if len(values) else np.nan
        scales[name] = median if np.isfinite(median) and median > 0 else 1.
    base, _ = evaluate(intervals, labels, features, "train", METHODS[0])
    floors = {key: math.ceil(.9*base[key]) for key in ["n_matched_clear_f10", "n_matched_all_f10"]}
    grid = {"audio_weight": [.25, .5, 1.], "threshold": [round(float(x), 2) for x in np.arange(.25, 4.01, .25)]}
    rows, chosen = [], {}
    for method in METHODS[2:]:
        trials = []
        for i, vals in enumerate(itertools.product(*grid.values())):
            candidate_config = dict(zip(grid, vals))
            picked = select(features, method, scales, candidate_config)
            metrics, _ = evaluate(intervals, labels, picked, "train", method)
            deficit = sum(max(0, floor-metrics[key])/max(1, floor) for key, floor in floors.items())
            trial = dict(**metrics, config_id=i, **candidate_config, recall_floor_deficit=deficit, feasible=deficit == 0)
            trials.append(trial)
        best = min(trials, key=lambda r: (r["recall_floor_deficit"], r["review_window_sum_ratio"],
                                          -r["n_matched_clear_f6"], -r["n_matched_all_f10"], r["config_id"]))
        chosen[method] = {k: best[k] for k in ["config_id", "audio_weight", "threshold", "feasible", "recall_floor_deficit"]}
        rows.extend(trials)
    config = dict(feature_version=FEATURE_VERSION, feature_config=feature_config, train_points=TRAIN, test_points=TEST,
                  train_includes_point8=True, scales=scales, grid=grid, selected=chosen,
                  recall_floor_fraction=.9, recall_floors=floors, training_provenance=provenance,
                  selection_order=["min normalized recall-floor deficit", "min summed window fraction", "max clear +/-6F matches", "max all +/-10F matches", "config ID"],
                  test_labels_loaded_before_freeze=False, test_results_used_for_tuning=False,
                  partial_cache_check=not provenance["full_run"])
    save_csv("parameter_search_train.csv", pd.DataFrame(rows))
    save_json("selected_config.json", config)
    print("FROZEN train-only configuration: " + json.dumps(chosen), flush=True)
    return config


def scopes(intervals):
    return [("train_07_08", intervals.point.isin(TRAIN)), ("test_09_11", intervals.point.isin(TEST)),
            ("train_excluding_08", intervals.point == 7), ("all_07_11", intervals.point.isin(TRAIN+TEST)),
            ("all_excluding_08", intervals.point != 8),
            *[(f"point_{p:02d}", intervals.point == p) for p in TRAIN+TEST]]


def subset(table, ints):
    keys = set(zip(ints.point, ints.rally))
    return table[previous.mask_keys(table, keys)].copy()


def label_frame_evaluation(labels, timeseries, intervals):
    results, all_details = [], []
    for name, radius in [("exact_label_frame", 0), ("label_centered_pm8", PEAK_RADIUS)]:
        queries = labels[["point", "rally", "contact_frame", "label_id", "hitter_side", "confidence"]].rename(columns={"contact_frame": "candidate_frame", "label_id": "candidate_id"})
        found = add_wrist_features(queries, timeseries, radius)
        found["variant"] = name
        found["side_known"] = found.visual_side.isin(["N", "F"])
        found["side_correct"] = (found.visual_side == found.hitter_side) & found.side_known
        all_details.append(found)
        for scope, mask in scopes(intervals):
            ss = subset(found, intervals[mask])
            for label_conf in ["all", "clear"]:
                g = ss if label_conf == "all" else ss[ss.confidence == "clear"]
                row = dict(scope=scope, variant=name, label_confidence=label_conf, n_labels=len(g),
                           n_side_known=int(g.side_known.sum()), n_side_correct=int(g.side_correct.sum()),
                           side_accuracy=previous.divide(g.side_correct.sum(), g.side_known.sum()),
                           side_known_coverage=previous.divide(g.side_known.sum(), len(g)),
                           correct_over_all_labels=previous.divide(g.side_correct.sum(), len(g)))
                acc = []
                for side in ["N", "F"]:
                    s = g[g.hitter_side == side]
                    row.update({f"n_{side}_labels": len(s), f"n_{side}_known": int(s.side_known.sum()),
                                f"n_{side}_correct": int(s.side_correct.sum()),
                                f"side_{side}_accuracy": previous.divide(s.side_correct.sum(), s.side_known.sum())})
                    acc.append(row[f"side_{side}_accuracy"])
                row["side_balanced_accuracy"] = float(np.mean(acc)) if np.isfinite(acc).all() else np.nan
                results.append(row)
    return pd.DataFrame(results), pd.concat(all_details, ignore_index=True)


def plot_points(intervals, labels, features, timeseries, selections, coverage):
    audio = pd.read_csv(PIPE / "audiovisual/audio_plot_timeseries.csv")
    for point in TRAIN+TEST:
        segs = intervals[intervals.point == point]
        fig, axes = plt.subplots(3, len(segs), figsize=(18, 10), squeeze=False, sharex="col", gridspec_kw={"height_ratios": [1, 1.6, 1.4]})
        for col, seg in enumerate(segs.itertuples()):
            key = (features.point == point) & (features.rally == seg.rally)
            c = features[key]
            lab = labels[(labels.point == point) & (labels.rally == seg.rally)]
            a = audio[(audio.point == point) & (audio.rally == seg.rally)]
            axes[0, col].plot(a.source_frame_float, a.audio_strength, color=".5", lw=.6, label="4-12kHz sound score (previous cache)")
            axes[0, col].scatter(c.candidate_frame, c.strength, s=20, marker="x", color="black", label="all v1 proposals")
            axes[0, col].set_ylabel("Audio score")
            for slot, color in zip(SLOTS, COLORS):
                mm = timeseries[(timeseries.point == point) & (timeseries.rally == seg.rally) & (timeseries.player_slot == slot)]
                axes[1, col].plot(mm.source_frame, mm.wrist_speed, color=color, lw=.8, label=slot)
            axes[1, col].set_ylabel("Relative wrist speed\n(torso lengths / frame)")
            for j, method in enumerate(METHODS):
                selected = selections[method]
                selected = selected[(selected.point == point) & (selected.rally == seg.rally)]
                for side, color in [("N", "#1976b9"), ("F", "#b3438f"), ("", ".6")]:
                    ss = selected[selected.visual_side == side]
                    axes[2, col].scatter(ss.candidate_frame, np.full(len(ss), j), marker="v", s=30, color=color)
            for row in lab.itertuples():
                color = "#1976b9" if row.hitter_side == "N" else "#b3438f"
                axes[2, col].axvline(row.contact_frame, color=color, ls="-" if row.confidence == "clear" else ":", lw=.8)
                axes[2, col].text(row.contact_frame, 3.6, row.hitter_side, color=color, fontsize=7, ha="center")
            axes[2, col].set_yticks(range(4), ["v1", "v1 + old A", "v1 + wrist", "v1 + wrist + DP"])
            axes[2, col].set_ylim(-.5, 4)
            axes[2, col].set_xlabel("Zero-based source frame")
            cov = coverage[(coverage.point == point) & (coverage.rally == seg.rally)].iloc[0]
            end = seg.start_frame+cov.processed_frames
            for ax in axes[:, col]:
                ax.set_xlim(seg.start_frame, seg.end_frame)
                ax.grid(alpha=.18)
                ax.ticklabel_format(axis="x", useOffset=False, style="plain")
                if not cov.complete:
                    ax.axvspan(end, seg.end_frame, color=".8", alpha=.3, label="pose not processed")
            axes[0, col].set_title(f"match2 point {point:02d} rally {seg.rally}" + ("; NEAR TRACKS UNRELIABLE" if point == 8 else ""))
            axes[0, col].legend(fontsize=7, loc="upper right")
            axes[1, col].legend(fontsize=8, loc="upper right", ncol=2)
        full = bool(coverage.complete.all())
        fig.suptitle(("FULL INFERENCE" if full else "PARTIAL CACHE CHECK - NOT A RESEARCH RESULT") + "; labels clear solid / inferred dotted; sides N blue / F purple / unknown gray", fontsize=10)
        fig.tight_layout()
        fig.savefig(HERE / f"point_{point:02d}_audio_pose.png", dpi=150)
        plt.close(fig)


def pct(value):
    return f"{100*value:.1f}%" if np.isfinite(value) else "—"


def metric_table(evaluation, selected_scopes):
    lines = ["| 集合 | 方法 | clear ±6F | clear ±10F | 全 ±10F | 適合率目安 | 候補数/秒 | 窓合計/区間 | 手首の側正解/既知（正解率; 既知率） |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    names = {"train_07_08": "第7・8点", "test_09_11": "第9〜11点", "train_excluding_08": "学習側・第8点除外", "all_excluding_08": "全体・第8点除外"}
    for scope in selected_scopes:
        for r in evaluation[evaluation.scope == scope].itertuples():
            lines.append(f"| {names.get(scope, scope)} | {r.method} | {r.n_matched_clear_f6}/{r.n_labels_clear} ({pct(r.recall_clear_f6)}) | {r.n_matched_clear_f10}/{r.n_labels_clear} ({pct(r.recall_clear_f10)}) | {r.n_matched_all_f10}/{r.n_labels_all} ({pct(r.recall_all_f10)}) | {pct(r.precision_near_label_f10)} | {r.n_candidates}/{r.candidates_per_sec:.3f} | {pct(r.review_window_sum_ratio)} | {r.n_side_correct}/{r.n_side_known} ({pct(r.side_accuracy_f10)}; {pct(r.side_known_coverage_f10)}) |")
    return "\n".join(lines)


def make_readme(evaluation, oracle, config, coverage, diagnostics, elapsed, extraction):
    full = bool(coverage.complete.all())
    mode = "全区間の推論を完了した結果" if full else "**各区間の先頭だけを処理した動作確認。以下の手首の数値は本評価として解釈しない。**"
    previous_evaluation = pd.read_csv(PIPE / "audiovisual/evaluation.csv")
    prior = previous_evaluation[(previous_evaluation.scope.isin(["train_07_08_including_7A", "test_09_11"])) & (previous_evaluation.method == "audio_v1_A")]
    prior_lines = [f"- {r.scope}: clear ±10F {r.n_matched_clear_f10}/{r.n_labels_clear}、全 ±10F {r.n_matched_all_f10}/{r.n_labels_all}、旧Aの側正解率 {r.n_side_correct}/{r.n_side_known} ({pct(r.side_accuracy_f10)})。" for r in prior.itertuples()]
    test = evaluation[evaluation.scope == "test_09_11"].set_index("method")
    v1, wrist, dp = (test.loc[m] for m in [METHODS[0], METHODS[2], METHODS[3]])
    old_a = prior[prior.scope == "test_09_11"].iloc[0]
    deficit_note = ("学習側でも再現率維持条件を満たさなかったため、採用できない。"
                    if not config["selected"][METHODS[3]]["feasible"] else "再現率と候補削減の両方で判断する必要がある。")
    interpretation = (
        f"評価側の側正解数は、旧A {int(old_a.n_side_correct)}/{int(old_a.n_side_known)} ({pct(old_a.side_accuracy_f10)})、"
        f"v1候補に付けた手首推定 {int(v1.n_side_correct)}/{int(v1.n_side_known)} ({pct(v1.side_accuracy_f10)})。"
        f"全対応数が同じ集計で正解数の差は{int(v1.n_side_correct-old_a.n_side_correct)}件で、側推定の有効な改善を示すには不十分。"
        f"手首で絞った方式も {int(wrist.n_side_correct)}/{int(wrist.n_side_known)} ({pct(wrist.side_accuracy_f10)})、"
        f"手前N {int(wrist.n_side_N_correct)}/{int(wrist.n_side_N_known)} ({pct(wrist.side_N_accuracy)})、"
        f"奥F {int(wrist.n_side_F_correct)}/{int(wrist.n_side_F_known)} ({pct(wrist.side_F_accuracy)}) と偏りがある。\n\n"
        f"手首方式は候補を {int(v1.n_candidates)}→{int(wrist.n_candidates)}件に減らした一方、"
        f"clear ±10F対応は {int(v1.n_matched_clear_f10)}→{int(wrist.n_matched_clear_f10)}/{int(wrist.n_labels_clear)}件になった。"
        f"交互DPはclear ±10F {int(dp.n_matched_clear_f10)}/{int(dp.n_labels_clear)}件に低下し、{deficit_note}"
        "候補削減だけを打球検出や確認作業の改善と解釈しない。"
    ) if full else "部分キャッシュのため、手首方式の比較結果から有効性を判断しない。"
    side_lines = ["| 集合 | 方法 | 手前N 正解/既知 | 奥F 正解/既知 | balanced accuracy | 正解/全時間対応 |", "|---|---|---:|---:|---:|---:|"]
    for r in evaluation[evaluation.scope.isin(["train_07_08", "test_09_11", "all_excluding_08"])].itertuples():
        side_lines.append(f"| {r.scope} | {r.method} | {r.n_side_N_correct}/{r.n_side_N_known} ({pct(r.side_N_accuracy)}) | {r.n_side_F_correct}/{r.n_side_F_known} ({pct(r.side_F_accuracy)}) | {pct(r.side_balanced_accuracy)} | {pct(r.side_correct_over_all_time_matches)} |")
    oracle_lines = ["| 集合 | フレーム条件 | 正解/既知 | 既知率 | 正解/全ラベル | N正解/既知 | F正解/既知 |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in oracle[(oracle.scope.isin(["train_07_08", "test_09_11", "all_excluding_08"])) & (oracle.label_confidence == "all")].itertuples():
        oracle_lines.append(f"| {r.scope} | {r.variant} | {r.n_side_correct}/{r.n_side_known} ({pct(r.side_accuracy)}) | {pct(r.side_known_coverage)} | {pct(r.correct_over_all_labels)} | {r.n_N_correct}/{r.n_N_known} | {r.n_F_correct}/{r.n_F_known} |")
    coverage_lines = ["| 点/rally | 処理フレーム/全区間 | 処理率 |", "|---|---:|---:|"]
    for r in coverage.itertuples():
        coverage_lines.append(f"| {r.point}/{r.rally} | {r.processed_frames}/{r.expected_frames} | {pct(r.frame_coverage)} |")
    diagnostic_lines = ["| 点/rally | 選手 | 速度有効/処理F | 有効率 |", "|---|---|---:|---:|"]
    for r in diagnostics.itertuples():
        diagnostic_lines.append(f"| {r.point}/{r.rally} | {r.player_slot} | {r.valid_wrist_speed}/{r.processed_frames} | {pct(r.valid_wrist_speed_fraction)} |")
    summary_lines = []
    summary_path = HERE / "pose_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary_lines.extend(["`pose_summary.csv` の実測値（信頼度の分布は検出された人の手首。検出欠測は有効率の分母に含む）：",
                              "| 点/rally | 選手 | 検出/処理F (率) | 箱あり時検出率 | 左/右手首 信頼度中央値 | 左/右手首≥0.3率 |",
                              "|---|---|---:|---:|---:|---:|"])
        for r in summary.itertuples():
            summary_lines.append(f"| {r.point}/{r.rally} | {r.player_slot} | {r.n_detected}/{r.processed_frames} ({pct(r.detection_rate_processed)}) | {pct(r.detection_rate_given_box)} | {r.left_wrist_conf_p50:.3f}/{r.right_wrist_conf_p50:.3f} | {pct(r.left_wrist_valid_rate_processed)}/{pct(r.right_wrist_valid_rate_processed)} |")
    timings = pd.read_csv(HERE / "pose_timing.csv") if (HERE / "pose_timing.csv").exists() else pd.DataFrame()
    run_path = HERE / "run_metadata.json"
    run_phase = read_json(run_path).get("phase", "all") if run_path.exists() else "all"
    if run_phase == "evaluate":
        combine_timing_text = f"今回の結合・評価・描画：{elapsed:.3f}秒（探索は含まない）。"
        train_meta_path = HERE / "train_run_metadata.json"
        if train_meta_path.exists():
            train_elapsed = float(read_json(train_meta_path)["elapsed_sec"])
            combine_timing_text += f" 別実行した学習・設定固定：{train_elapsed:.3f}秒、両段階の合計 {train_elapsed+elapsed:.3f}秒。"
    else:
        combine_timing_text = f"今回の結合・探索・評価・描画：{elapsed:.3f}秒。"
    timing_lines = ["| 点/rally | device | 処理F | 区間処理秒 | 順読秒 | 推論秒 |", "|---|---|---:|---:|---:|---:|"]
    for r in timings.itertuples():
        timing_lines.append(f"| {r.point}/{r.rally} | {r.device} | {r.processed_frames} | {r.elapsed_sec:.3f} | {r.decode_sec:.3f} | {r.inference_sec:.3f} |")
    smoke_path = HERE / "smoke_cpu30/extraction_manifest.json"
    smoke_text = ""
    if smoke_path.exists():
        smoke = read_json(smoke_path)
        processed = sum(r["processed_frames"] for r in smoke["intervals"])
        smoke_text = f"CPU先頭30F検証も完了：{processed}フレーム、{processed*4}行、抽出の記録時間{smoke['elapsed_sec']:.3f}秒（`smoke_cpu30/` に記録）。部分キャッシュで結合・探索・図生成が落ちずに完了した。"
    text = f'''# match2 第7〜11点：音候補とYOLO11 poseの手首速度

{mode}

{interpretation}

既存ファイルは読み取り専用。ラベルへの一致を評価しており、「本物の打球」とラケット接触を独立に確認した検出精度ではない。第7点A/Bは別区間とし、区間をまたぐ微分・ピーク探索・DP遷移をしない。

## 実行（リポジトリルート、PowerShell）

```powershell
& study\\.venv\\Scripts\\python.exe -B -X utf8 outputs\\shot_pipeline_20261117\\pose\\extract_pose.py --device cpu --limit-frames 30
& study\\.venv\\Scripts\\python.exe -B -X utf8 outputs\\shot_pipeline_20261117\\pose\\combine_audio_pose.py
# GPUで全体を推論。先頭だけのキャッシュを各区間の全体キャッシュに置き換える
& study\\.venv\\Scripts\\python.exe -B -X utf8 outputs\\shot_pipeline_20261117\\pose\\extract_pose.py --device 0
# 全体キャッシュで学習側だけから設定を固定して、その設定で評価
& study\\.venv\\Scripts\\python.exe -B -X utf8 outputs\\shot_pipeline_20261117\\pose\\combine_audio_pose.py --phase train
& study\\.venv\\Scripts\\python.exe -B -X utf8 outputs\\shot_pipeline_20261117\\pose\\combine_audio_pose.py --phase evaluate
```

既定の `--phase all` は train→設定保存→test の順。`evaluate` は固定設定と学習入力のハッシュを検証し、再探索しない。部分実行から全体推論に変えた場合は `train` または `all` で学習をやり直す。未処理部分を除いて分母を小さくすることはしない。

## 抽出と速度

抽出の実行設定は `extraction_manifest.json`。各区間冒頭で1回シークしてgrab/retrieveで順読みし、フレーム番号と画像サイズをassertする。追跡箱を各左右に幅の60%、各上下に高さの25%広げる（全幅2.2倍・全高1.5倍）。長辺384pxの切り抜きをバッチ推論し、追跡箱との対応で1人を選ぶ。YOLO設定先はimport前に `pose_cache/ultralytics_config` に固定する。キャッシュは元画像座標の17関節と信頼度を持ち、4選手×全処理フレームを欠測も含めて保存する。

手首9/10、両腰11/12、両肩5/6に信頼度≥{config['feature_config']['kp_conf']}を適用。腰中点を原点、肩中点との距離を胴長とし、`(手首−原点)/胴長` の連続フレーム間ユークリッド差を速度（胴長/フレーム）にする。腰が欠ければ肩中点を原点とし、その区間・選手の有効な胴長の中央値を尺度に使う。胴長が一度も得られなければ欠測。連続フレーム・同一track ID・同一原点/尺度モードでのみ差を計算する。欠測を補間せず、有効な連続区間内で{config['feature_config']['smooth_window']}F中央値平滑化後、左右手首の大きい方を選手の速度とする。

この相対座標は画面の平行移動を理論上相殺し、胴長による正規化は一様拡大縮小を抑える。ただし回転、透視変換、姿勢の変化、推定点の揺れ、誤人物対応は相殺しない。検出失敗は非打球の証拠にしない。

## 候補と設定の固定

`audiovisual/candidate_features.csv` の全v1候補を使用し、元の `audio/candidates.csv` とID・点・rally・フレームの完全一致を検査する。各候補±8Fの各選手の速度最大値から、その選手・区間内速度中央値を引いた値をピーク高さとする。最大選手、2番目との差、N/Fそれぞれの最大値の差、有効点数を保存する。最大が正かつ一意で、両側に少なくとも1選手の速度が得られた場合にだけ側を出す。未知側を正解率の分母から除くが、既知率と正解/全対応も必ず併記する。

比較方式は音v1、音v1+前回A、音v1+手首、音v1+手首+交互DP。前回Aの候補は既存 `selected_candidates.csv` からそのまま読む。主表の側は全方式に同一の手首推定を付けた比較で、音v1自体に側の予測能力があるという意味ではない。旧A自身の側正解率は別途再掲する。

手首方式の得点は `log1p(max(peak,0))/scale_wrist + audio_weight*log1p(strength)/scale_audio - threshold`。尺度は第7・8点の候補の変換値中央値、欠測寄与0。前回音v2の形状特徴は新方式に使わない。DPは前回と同じく、側交互・間隔≥12Fの最大得点部分列（空列可、負得点の橋渡し可）。未知側はN/F両状態を許し、`dp_assigned_side`を保存するが手首の側正解率には使わない。

- 特徴設定：`{json.dumps(config['feature_config'])}`
- 格子：`{json.dumps(config['grid'])}`（方式ごと48設定）
- 尺度：`{json.dumps(config['scales'])}`
- 設定：`{json.dumps(config['selected'])}`
- 再現率維持条件：学習側v1の±10F対応数90%以上、clear≥{config['recall_floors']['n_matched_clear_f10']}、全≥{config['recall_floors']['n_matched_all_f10']}。不足量最小→確認窓和最小→clear±6F対応最大→全±10F対応最大→ID順。`feasible=false` は維持条件未達の参考設定。

第7・8点のラベルだけで尺度・設定を保存してから第9〜11点のラベルを読む。第8点を除いた集計は固定設定の感度分析で、再調整しない。第8点は手前のピンク選手が追跡されずN1が線審なので、手前の検出率も側推定も選手を測定している保証がなく信頼できない。誤追跡を姿勢推定だけで修正しない。

前回Aの音形状仮説には第9点の観察が含まれ、この3点は研究全体として完全未接触の独立テストではない。今回の手首方式はその形状特徴を得点に使わず、第9〜11点の結果を見た再調整をしていない。

## 比較結果

{metric_table(evaluation, ['train_07_08', 'test_09_11'])}

第8点を除く固定設定の集計：

{metric_table(evaluation, ['train_excluding_08', 'all_excluding_08'])}

前回Aの元の数値（ファイルから再掲。手首の側推定とは異なる）：

{chr(10).join(prior_lines)}

{chr(10).join(side_lines)}

## 音候補を使わない側評価

`exact_label_frame` はラベルの打球フレームそのものの平滑化速度。`label_centered_pm8` はその±8Fのピークで、時刻を与えたときの診断。いずれも未知率を併記し、理論上の上限とは断定しない。clear限定はCSVに収録する。

{chr(10).join(oracle_lines)}

## 処理範囲・検出率・時間

{chr(10).join(coverage_lines)}

{chr(10).join(diagnostic_lines)}

{chr(10).join(summary_lines)}

抽出の区間処理と要約保存 {extraction['elapsed_sec']:.3f}秒（`extraction_manifest.json`）。モデル読込 {extraction['model_load_sec']:.3f}秒は別計時。この記録時間はコマンド全体の実時間ではなく、入力読込・ハッシュ検証の前後処理を含まない。下表の区間処理の合計 {float(timings.elapsed_sec.sum()):.3f}秒にはモデルのコンストラクタ時間を含まず、初回predictのウォームアップは推論時間に含む。

{chr(10).join(timing_lines)}

{smoke_text}

{combine_timing_text} 入力ハッシュと完全/部分実行状態は `run_metadata.json`。

## 指標と限界

前回の関数を読み取り専用で利用し、同じ点/rally内の全ラベルと候補を時間差の小さい順で1対1照合（同率はラベル時刻→候補時刻→ID）。±6F/±10Fは別照合し、その中のclear対応を数える。側は照合条件に使わない。適合率目安は±10F内に何らかのラベルがある候補/全候補（多対1可）で、1対1適合率もCSVに保存する。確認範囲は連続時間の±10F窓（幅20F）の区間内の和/区間長、重複も足すため100%超があり得る。和集合比率も保存する。

ラベルは6F間隔の映像からの既存Claudeラベルで、独立な接触正解ではない。±10F一致は偶然一致と時刻誤差を含む。左右側の多数派を出すだけでも評価側は{pct(float(evaluation[(evaluation.scope == 'test_09_11') & (evaluation.method == 'audio_v1')].label_majority_side_baseline.iloc[0]))}となるため、全体正解率だけで有効性を主張しない。身体運動は打球・準備・反応・走行を混ぜる。±8F窓は相手の反応を拾い、姿勢誤差の微分は偽の高速ピークを作る。奥の小さい選手を拡大しても原画像の情報は増えない。

交互制約は途中の候補欠落、誤側、短間隔ラベルで正しい打球をさらに落とし得る。候補削減、clear再現率、側の既知率とN/F別精度を併せて判断する。学習2点・評価3点の同一試合であり、確認時間短縮や他試合への一般化は実証していない。独立な人手ラベルと未使用試合による検証が次に必要。

## 出力

- `wrist_timeseries.csv` / `wrist_diagnostics.csv`: 相対座標、左右速度、最大速度、原点・尺度、欠測率。
- `candidate_features.csv` / `selected_candidates.csv`: 全v1候補の手首特徴と各方式の保持結果。前回Aの保持候補はそのまま。
- `evaluation.csv` / `label_matches.csv`: 同一指標・側N/F別分子分母・未知率・第8点除外集計。
- `label_frame_side_evaluation.csv` / `label_frame_side_details.csv`: 音を使わない正解時刻での側診断。
- `previous_baseline_evaluation.csv`: 前回v1/Aの元評価を読み込んで再掲（旧動きの側）。
- `parameter_search_train.csv` / `selected_config.json`: 学習側の探索と凍結設定。
- `inference_coverage.csv` / `run_metadata.json`: 部分実行状態と入力同一性。
- `point_07_audio_pose.png`〜`point_11_audio_pose.png`: 音・4人の速度・候補とラベルを共通フレーム軸で描画。未処理部分は灰色。
'''
    (HERE / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all", "train", "evaluate"], default="all")
    parser.add_argument("--kp-conf", type=float, default=.3)
    parser.add_argument("--smooth-window", type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.kp_conf <= 1 or args.smooth_window < 1 or args.smooth_window % 2 != 1:
        parser.error("kp-conf must be [0,1], smooth-window a positive odd integer")
    started = time.perf_counter()
    HERE.mkdir(parents=True, exist_ok=True)
    input_hashes = read_only_input_hashes(include_test_labels=False)
    intervals, candidates, a_ids, fps = load_inputs()
    pose, coverage, hashes, extraction = load_pose(intervals)
    full = bool(coverage.complete.all())
    if not full:
        print("WARNING: partial pose cache; this is a plumbing check, not a research result.", flush=True)
    feature_config = dict(kp_conf=args.kp_conf, smooth_window=args.smooth_window, peak_radius=PEAK_RADIUS,
                          baseline="per-player interval median", min_gap_frames=MIN_GAP,
                          shoulder_fallback_scale="interval median torso", unknown_side="both-side evidence required")
    timeseries, diagnostics = wrist_timeseries(pose, args.kp_conf, args.smooth_window)
    features = add_wrist_features(candidates, timeseries)
    train_ints = intervals[intervals.point.isin(TRAIN)]
    train_features = features[features.point.isin(TRAIN)]
    provenance = dict(full_run=full, feature_version=FEATURE_VERSION,
                      feature_functions_sha256=hashlib.sha256("\n".join(inspect.getsource(f) for f in [joint, wrist_timeseries, add_wrist_features, score, select]).encode("utf-8")).hexdigest(),
                      previous_module_sha256=sha(PIPE / "audiovisual/combine_audio_video.py"),
                      train_pose_cache_sha256={k: v for k, v in hashes.items() if int(k[:2]) in TRAIN},
                      train_label_sha256={str(p): sha(PIPE / f"labels/point_{p:02d}.csv") for p in TRAIN},
                      audio_candidates_sha256=sha(PIPE / "audiovisual/candidate_features.csv"),
                      intervals_sha256=sha(PIPE / "audio/intervals.csv"),
                      inference_config=extraction.get("config", {}))
    if args.phase == "evaluate":
        config = read_json(HERE / "selected_config.json")
        if config["training_provenance"] != provenance or config["feature_config"] != feature_config:
            raise ValueError("Frozen configuration inputs differ. Run --phase train on this cache first; no silent refit.")
        print("Using validated frozen train-only configuration.", flush=True)
    else:
        train_labels = previous.load_labels(TRAIN, intervals)
        config = train(train_features, train_ints, train_labels, provenance, feature_config)
    save_csv("wrist_timeseries.csv", timeseries)
    save_csv("wrist_diagnostics.csv", diagnostics)
    save_csv("candidate_features.csv", features)
    save_csv("inference_coverage.csv", coverage)
    if args.phase == "train":
        assert_input_hashes(input_hashes)
        save_json("train_run_metadata.json", dict(full_inference=full, partial_cache_check=not full,
                                                  elapsed_sec=time.perf_counter()-started, input_sha256=input_hashes,
                                                  inputs_unchanged=True, test_labels_loaded=False,
                                                  selected_config_sha256=sha(HERE / "selected_config.json")))
        print("Training complete; held-out labels were not loaded.", flush=True)
        return
    # This line must remain below config serialization/validation.
    input_hashes.update({f"labels/point_{p:02d}.csv": sha(PIPE / f"labels/point_{p:02d}.csv") for p in TEST})
    labels = previous.load_labels(TRAIN+TEST, intervals)
    selections = {m: select(features, m, config["scales"], config["selected"].get(m), a_ids) for m in METHODS}
    save_csv("selected_candidates.csv", pd.concat([table.assign(method=m) for m, table in selections.items()], ignore_index=True))
    rows, details = [], []
    for scope, mask in scopes(intervals):
        ints = intervals[mask]
        lab = subset(labels, ints)
        for method, selected in selections.items():
            result, matched = evaluate(ints, lab, subset(selected, ints), scope, method)
            cov = subset(coverage, ints)
            result.update(full_inference=full, partial_cache_check=not full,
                          processed_frame_fraction=float(cov.processed_frames.sum()/cov.expected_frames.sum()),
                          known_point8_tracking_problem=bool((ints.point == 8).any()))
            rows.append(result)
            details.extend(matched)
    evaluation = pd.DataFrame(rows)
    save_csv("evaluation.csv", evaluation)
    save_csv("label_matches.csv", pd.DataFrame(details))
    oracle, oracle_details = label_frame_evaluation(labels, timeseries, intervals)
    oracle["partial_cache_check"] = not full
    save_csv("label_frame_side_evaluation.csv", oracle)
    save_csv("label_frame_side_details.csv", oracle_details)
    old = pd.read_csv(PIPE / "audiovisual/evaluation.csv")
    detection_metrics = ["n_candidates", "n_matched_clear_f6", "n_matched_clear_f10", "n_matched_all_f10",
                         "precision_near_label_f10", "review_window_sum_ratio", "review_union_ratio"]
    for scope, old_scope in [("train_07_08", "train_07_08_including_7A"), ("test_09_11", "test_09_11")]:
        for method, old_method in [(METHODS[0], "audio_v1"), (METHODS[1], "audio_v1_A")]:
            new_row = evaluation[(evaluation.scope == scope) & (evaluation.method == method)].iloc[0]
            old_row = old[(old.scope == old_scope) & (old.method == old_method)].iloc[0]
            np.testing.assert_allclose(new_row[detection_metrics].to_numpy(float), old_row[detection_metrics].to_numpy(float), rtol=0, atol=1e-10)
    save_csv("previous_baseline_evaluation.csv", old[old.method.isin(["audio_v1", "audio_v1_A"])].assign(metrics_source="previous audiovisual/evaluation.csv; pixel-motion sides"))
    plot_points(intervals, labels, features, timeseries, selections, coverage)
    elapsed = time.perf_counter()-started
    assert_input_hashes(input_hashes)
    save_json("run_metadata.json", dict(created_utc=datetime.now(timezone.utc).isoformat(), elapsed_sec=elapsed,
                                       phase=args.phase, full_inference=full, partial_cache_check=not full,
                                       cache_sha256=hashes, extraction_manifest_sha256=sha(HERE / "extraction_manifest.json"),
                                       selected_config_sha256=sha(HERE / "selected_config.json"), script_sha256=sha(__file__),
                                       previous_module_sha256=sha(PIPE / "audiovisual/combine_audio_video.py"), fps=fps,
                                       input_sha256=input_hashes, inputs_unchanged=True,
                                       test_labels_loaded_after_config_freeze=True))
    make_readme(evaluation, oracle, config, coverage, diagnostics, elapsed, extraction)
    print(evaluation[evaluation.scope.isin(["train_07_08", "test_09_11"])][["scope", "method", "n_candidates", "recall_clear_f10", "side_accuracy_f10", "side_known_coverage_f10"]].to_string(index=False))
    print(f"Completed in {elapsed:.3f}s; full_inference={full}", flush=True)


if __name__ == "__main__":
    main()
