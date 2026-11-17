"""Filter the frozen v1 audio proposals, using points 7/8 only for selection.

Run detect_hit_sounds.py --v2 (default: train, freeze, then evaluate), or use
--phase train / --phase evaluate to separate the two stages. No WAV is retained.
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
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import ndimage, signal

import detect_hit_sounds as base


FEATURE_VERSION = "impact_shape_v2_1"
# A conjunction of at most three finite thresholds. None disables a gate.
GRID = {
    "max_width_ms": [None, 6., 10., 15., 25., 40., 80.],
    "min_paired_rise_ln": [None, 0., 0.5, 1., 1.5, 2., 3.],
    "min_post50_drop_ln": [None, 0., 0.5, 1., 2., 3.],
    "min_concentration": [None, 0.3, 0.5, 0.7],
}
GATES = {
    "max_width_ms": ("half_width_ms", -1., 10.),
    "min_paired_rise_ln": ("paired_rise_ln", 1., 1.),
    "min_post50_drop_ln": ("post50_drop_ln", 1., 1.),
    "min_concentration": ("energy_concentration_20_100", 1., 0.2),
}
BASELINE_ARTIFACTS = (
    "candidates.csv", "evaluation.csv", "intervals.csv", "label_matches.csv",
    "param_search.csv", "selected_config.json", "run_metadata.json",
    *[f"spectrogram_point_{p:02d}.png" for p in base.TRAIN_POINTS + base.TEST_POINTS],
)
README_START = "<!-- BEGIN AUDIO V2 RESULTS -->"
README_END = "<!-- END AUDIO V2 RESULTS -->"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def baseline_config(output):
    payload = json.loads((output / "selected_config.json").read_text(encoding="utf-8"))
    return base.Config(**{f.name: payload[f.name] for f in fields(base.Config)})


def baseline_manifest(output):
    return {name: sha256(output / name) for name in BASELINE_ARTIFACTS}


def window_burden(segments, candidates):
    """Continuous +/-10F windows clipped separately, with overlaps counted."""
    total_frames = 0.
    for seg in segments:
        centers = candidates.loc[(candidates.point == seg.point) &
                                 (candidates.rally == seg.rally), "candidate_frame"].to_numpy()
        total_frames += np.maximum(0., np.minimum(seg.end_frame, centers + 10) -
                                   np.maximum(seg.start_frame, centers - 10)).sum()
    duration = sum(s.duration_sec for s in segments)
    seconds = float(total_frames * base.FRAME_SEC)
    return {"review_window_sum_sec": seconds,
            "review_time_ratio_plusminus10_frames": seconds / duration}


def metrics(segments, candidates, scope, split):
    row, matches = base.evaluate(segments, candidates, scope, split)
    row.update(window_burden(segments, candidates))
    row["review_union_sec"] = row["review_coverage_plusminus10_frames"] * row["duration_sec"]
    return row, matches


def band_envelope(seg, band):
    starts = np.arange(0, len(seg.audio) - base.ENERGY_SAMPLES + 1, base.HOP_SAMPLES)
    sos = signal.butter(4, band, btype="bandpass", fs=base.SAMPLE_RATE, output="sos")
    audio = signal.sosfiltfilt(sos, seg.audio)
    sums = np.r_[0., np.cumsum(audio * audio)]
    power = (sums[starts + base.ENERGY_SAMPLES] - sums[starts]) / base.ENERGY_SAMPLES
    log_power = np.log(np.maximum(power, 1e-12))
    median = ndimage.median_filter(log_power, size=501, mode="reflect")
    return power, log_power, median


def width_at_half_excess(power, background, index):
    """Linear-power FWHM above a 501 ms median background; 1 ms sampling.

    Crossings are linearly interpolated. A side without a crossing within
    100 ms is capped at that boundary (width_censored records that case).
    """
    level = background[index] + 0.5 * max(0., power[index] - background[index])
    left_limit, right_limit = max(0, index - 100), min(len(power) - 1, index + 100)
    left = right = index
    while left > left_limit and power[left] > level:
        left -= 1
    while right < right_limit and power[right] > level:
        right += 1
    left_cross = float(left)
    right_cross = float(right)
    if left < index and power[left] <= level:
        left_cross += (level - power[left]) / max(power[left + 1] - power[left], 1e-30)
    if right > index and power[right] <= level:
        right_cross -= (level - power[right]) / max(power[right - 1] - power[right], 1e-30)
    return max(0., right_cross - left_cross), bool(power[left] > level or power[right] > level)


def add_features(segments, candidates):
    """No label input is used in feature extraction; candidate times never move."""
    records = []
    for seg in segments:
        high, high_log, high_median = band_envelope(seg, (4000, 12000))
        low, low_log, low_median = band_envelope(seg, (500, 2000))
        low_rise, high_rise = low_log - low_median, high_log - high_median
        excess = np.maximum(high - np.exp(high_median), 0.)
        # Mean of short-time spectra over +/-100 ms; harmonic persistence
        # lowers flatness. This optional feature is diagnostic, not searched.
        freqs, times, spectrum = signal.spectrogram(
            seg.audio, fs=base.SAMPLE_RATE, nperseg=512, noverlap=32,
            window="hann", scaling="density", mode="psd")
        band = (freqs >= 500) & (freqs <= 12000)
        local = candidates[(candidates.point == seg.point) & (candidates.rally == seg.rally)]
        for cand in local.itertuples():
            index = int(np.argmin(np.abs(seg.feature_frames - cand.candidate_frame_float)))
            width, censored = width_at_half_excess(high, np.exp(high_median), index)
            near4 = slice(max(0, index - 4), min(len(high), index + 5))
            near20 = slice(max(0, index - 20), min(len(high), index + 21))
            near100 = slice(max(0, index - 100), min(len(high), index + 101))
            low_index = max(0, index - 4) + int(np.argmax(low_rise[near4]))
            lo20_index = max(0, index - 20) + int(np.argmax(low_rise[near20]))
            low_at = float(low_rise[low_index])
            high_at = float(high_rise[index])
            after = float(np.mean(high[index + 45:index + 56]))
            post_ratio = after / max(float(high[index]), 1e-12)
            total = float(excess[near100].sum())
            local_time = cand.candidate_time_sec - seg.audio_start_sec
            use = np.abs(times - local_time) <= 0.100
            mean_spectrum = np.maximum(spectrum[np.ix_(band, use)].mean(axis=1), 1e-20)
            flatness = float(np.exp(np.log(mean_spectrum).mean()) / mean_spectrum.mean())
            records.append({
                "candidate_id": cand.candidate_id, "half_width_ms": width,
                "width_censored": censored, "low_rise_ln": low_at, "high_rise_ln": high_at,
                "paired_rise_ln": min(low_at, high_at),
                "low_peak_lag_within4_ms": int(low_index - index),
                "low_peak_lag_within20_ms": int(lo20_index - index),
                "post50_energy_ratio": post_ratio,
                "post50_drop_ln": float(-np.log(max(post_ratio, 1e-12))),
                "energy_concentration_20_100": float(excess[near20].sum() / total) if total else 0.,
                "spectral_flatness_100ms": flatness,
            })
    features = pd.DataFrame(records)
    result = candidates.merge(features, on="candidate_id", how="left", validate="one_to_one")
    numeric = [v[0] for v in GATES.values()] + ["spectral_flatness_100ms"]
    if not np.isfinite(result[numeric].to_numpy()).all():
        raise ValueError("Non-finite candidate features")
    return result


def candidate_targets(segments, candidates):
    """Many-to-one TRAIN target, distinct from 1-to-1 recall evaluation."""
    labels = pd.concat([s.labels for s in segments], ignore_index=True)
    positive, nearest = [], []
    for cand in candidates.itertuples():
        local = labels[(labels.point == cand.point) & (labels.rally == cand.rally)]
        delta = int(np.abs(local.contact_frame.to_numpy() - cand.candidate_frame).min())
        positive.append(delta <= 10)
        nearest.append(delta)
    return np.asarray(positive, dtype=bool), nearest


def gate_margin(candidates, rule):
    margins = []
    for key, (feature, sign, scale) in GATES.items():
        if rule[key] is not None:
            margins.append(sign * (candidates[feature].to_numpy() - rule[key]) / scale)
    return np.min(margins, axis=0) if margins else np.full(len(candidates), np.inf)


def train_rules(segments, candidates, output, config, manifest, pipeline):
    target, nearest = candidate_targets(segments, candidates)
    candidates = candidates.copy()
    candidates["train_target_within10"] = target
    candidates["nearest_train_label_abs_frames"] = nearest
    baseline, _ = metrics(segments, candidates, "train_07_08", "train")
    floors = {f"n_matched_{kind}_f10": math.ceil(.9 * baseline[f"n_matched_{kind}_f10"])
              for kind in ("clear", "all")}
    # All grid definitions and the selection criterion exist before test loading.
    records, rules, metric_cache = [], {}, {}
    for index, values in enumerate(itertools.product(*GRID.values())):
        rule = dict(zip(GRID, values))
        active = sum(v is not None for v in values)
        if active > 3:
            continue
        rule_id = f"rule_{index:04d}"
        rules[rule_id] = rule
        keep = gate_margin(candidates, rule) >= 0
        tp, fp, fn = int((keep & target).sum()), int((keep & ~target).sum()), int((~keep & target).sum())
        mask_key = keep.tobytes()
        if mask_key not in metric_cache:
            metric_cache[mask_key], _ = metrics(segments, candidates[keep], "train_07_08", "train")
        row = metric_cache[mask_key]
        records.append({"rule_id": rule_id, **rule, "n_active_gates": active,
                        "candidate_target_tp": tp, "candidate_target_fp": fp, "candidate_target_fn": fn,
                        "candidate_target_f1": 2 * tp / (2 * tp + fp + fn),
                        "candidate_target_recall": tp / int(target.sum()),
                        "candidate_target_precision": tp / (tp + fp) if tp + fp else 0.,
                        "recall_floor_pass": all(row[k] >= v for k, v in floors.items()), **row})
    table = pd.DataFrame(records)
    eligible = table[table.recall_floor_pass].sort_values(
        ["review_time_ratio_plusminus10_frames", "candidate_target_f1", "n_active_gates", "rule_id"],
        ascending=[True, False, True, True])
    if eligible.empty:
        raise RuntimeError("No rule retains 90% of baseline train matches")
    chosen_id = str(eligible.iloc[0].rule_id)
    table["selected"] = table.rule_id.eq(chosen_id)
    table.to_csv(output / "param_search_v2.csv", index=False)
    frozen = {
        "feature_version": FEATURE_VERSION, "rule_id": chosen_id, "rule": rules[chosen_id],
        "baseline_config": {f.name: getattr(config, f.name) for f in fields(base.Config)},
        "selected_on": list(base.TRAIN_POINTS), "grid": GRID, "max_active_gates": 3,
        "n_rules": len(table), "train_target": "any clear or inferred label within inclusive +/-10 integer frames in same point/rally",
        "recall_floor": floors, "recall_floor_fraction_of_baseline": .9,
        "selection_order": ["min summed clipped review-window ratio", "max candidate-target F1",
                            "min active gates", "rule ID ascending"],
        "train_only_objective_revision": "Initial candidate-F1-first search kept 110/114 train proposals; before any test loading, changed to summed-burden-first with the same recall floors. Initial freeze retained in initial_f1_selection_v2.json.",
        "margin_definition": {k: {"feature": f, "sign": sign, "scale": scale}
                              for k, (f, sign, scale) in GATES.items()},
        "margin_threshold": 0.,
        "curve_thresholds": ["-inf", *np.linspace(-4., 4., 81).tolist(), "+inf"],
        "feature_script_sha256": sha256(__file__),
        "baseline_script_sha256": sha256(Path(base.__file__)),
        "baseline_artifact_sha256": manifest,
        "train_label_sha256": {str(p): sha256(pipeline / "labels" / f"point_{p:02d}.csv")
                               for p in base.TRAIN_POINTS},
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "held_out_loaded_before_freeze": False,
        "design_caveat": "Feature hypotheses supplied by user include point-9 spectrogram observations; test is reused, not fully untouched.",
    }
    dump_json(output / "selected_config_v2.json", frozen)
    candidates.to_csv(output / "train_features_v2.csv", index=False)
    print("FROZEN train-only rule: " + json.dumps(frozen["rule"]), flush=True)
    print(eligible.head(5)[["rule_id", "candidate_target_f1", "n_candidates",
                            "recall_clear_f10", "recall_all_f10",
                            "review_time_ratio_plusminus10_frames"]].to_string(index=False), flush=True)
    return frozen


def plot_point(segments, candidates, config, output):
    fig, axes = plt.subplots(2, len(segments), squeeze=False, sharex="col",
                             figsize=(24 if len(segments) > 1 else 22, 8),
                             gridspec_kw={"width_ratios": [s.duration_sec for s in segments]})
    for column, seg in enumerate(segments):
        top, bottom = axes[:, column]
        frequencies, times, spectrum = signal.spectrogram(
            seg.audio, fs=base.SAMPLE_RATE, window="hann", nperseg=512,
            noverlap=384, scaling="density", mode="psd")
        frame_times = (times + seg.audio_start_sec) / base.FRAME_SEC
        select_t = (frame_times >= seg.start_frame) & (frame_times < seg.end_frame)
        select_f = frequencies <= 12000
        db = 10 * np.log10(np.maximum(spectrum[np.ix_(select_f, select_t)], 1e-14))
        top.pcolormesh(frame_times[select_t], frequencies[select_f] / 1000, db,
                       shading="auto", cmap="magma", vmin=-110, vmax=-40, rasterized=True)
        top.set_ylim(0, 12)
        top.set_ylabel("Frequency (kHz)")
        top.set_title(f"Point {seg.point:02d}, rally {seg.rally} | frozen v2 filtering")
        core = (seg.feature_frames >= seg.start_frame) & (seg.feature_frames < seg.end_frame)
        score = seg.features[(config.low_hz, config.high_hz, config.median_sec)]
        bottom.plot(seg.feature_frames[core], score[core], color="#202020", linewidth=.7)
        bottom.axhline(config.threshold, color="#777777", linestyle="--", linewidth=1)
        for label in seg.labels.itertuples():
            for ax in (top, bottom):
                ax.axvline(label.contact_frame,
                           color="#2187db" if label.confidence == "clear" else "#f6a832",
                           linestyle="-" if label.confidence == "clear" else ":", linewidth=1., alpha=.75)
        local = candidates[(candidates.point == seg.point) & (candidates.rally == seg.rally)]
        for keep, color, marker, height in ((False, "#a5a5a5", "x", 11.2),
                                           (True, "#34e399", "v", 11.7)):
            part = local[local.kept == keep]
            top.scatter(part.candidate_frame_float, np.full(len(part), height), marker=marker,
                        s=32, color=color, linewidths=.8, zorder=5)
            bottom.scatter(part.candidate_frame_float, part.strength, marker=marker,
                           s=28, color=color, linewidths=.8, zorder=5)
        bottom.set_xlim(seg.start_frame, seg.end_frame)
        bottom.set_xlabel("Source video frame (60000/1001 fps)")
        bottom.set_ylabel("High-band ln energy - median")
        bottom.grid(alpha=.15)
        bottom.ticklabel_format(axis="x", useOffset=False, style="plain")
    handles = [Line2D([0], [0], color="#2187db", label="clear label"),
               Line2D([0], [0], color="#f6a832", linestyle=":", label="inferred label"),
               Line2D([0], [0], color="#34e399", marker="v", linestyle="None", label="kept"),
               Line2D([0], [0], color="#a5a5a5", marker="x", linestyle="None", label="rejected")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output / f"spectrogram_point_{segments[0].point:02d}_v2.png", dpi=180)
    plt.close(fig)


def tradeoff(segments, candidates, frozen, output):
    rows = []
    thresholds = [float(t.replace("+", "")) if isinstance(t, str) else t
                  for t in frozen["curve_thresholds"]]
    for threshold in thresholds:
        # Explicit endpoints also work if the selected rule has no active gate.
        if threshold == float("inf"):
            kept = candidates.iloc[:0]
        elif threshold == -float("inf"):
            kept = candidates
        else:
            kept = candidates[candidates.rule_margin >= threshold]
        for subset, scope in ((segments, "test_09_11"),
                              *[([s for s in segments if s.point == p], f"point_{p:02d}")
                                for p in base.TEST_POINTS]):
            local = kept[kept.point.isin([s.point for s in subset])]
            row, _ = metrics(subset, local, scope, "test")
            rows.append({"margin_threshold": threshold, **row})
    table = pd.DataFrame(rows)
    table.to_csv(output / "tradeoff_v2.csv", index=False)
    pooled = table[table.scope == "test_09_11"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, x, title in zip(axes,
                            ("review_coverage_plusminus10_frames", "review_time_ratio_plusminus10_frames"),
                            ("Union of review windows", "Sum of review windows (primary)")):
        for col, label, color in (("recall_clear_f6", "clear +/-6F", "#166ab0"),
                                  ("recall_clear_f10", "clear +/-10F", "#e07816"),
                                  ("recall_all_f10", "all +/-10F", "#368a4b")):
            ax.plot(pooled[x] * 100, pooled[col] * 100, ".-", color=color, label=label, markersize=3)
            chosen = pooled[pooled.margin_threshold == 0]
            ax.scatter(chosen[x] * 100, chosen[col] * 100, marker="*", s=180,
                       color=color, edgecolors="black", linewidths=.6, zorder=6)
        ax.set_xlabel(title + " / interval duration (%)")
        ax.set_ylim(-3, 103)
        ax.set_title(title)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Existing-label recall (%)")
    axes[0].legend(loc="lower right")
    fig.suptitle("Points 9-11 | frozen rule; star = train-selected threshold (no test selection)")
    fig.tight_layout()
    fig.savefig(output / "tradeoff_v2.png", dpi=170)
    # Explicit filename requested by the user; no v1 file used this name.
    (output / "tradeoff.png").write_bytes((output / "tradeoff_v2.png").read_bytes())
    plt.close(fig)


def verify_baseline(output, candidates, rows):
    old = pd.read_csv(output / "candidates.csv", keep_default_na=False)
    identity = ["candidate_id", "point", "rally", "candidate_frame"]
    if not candidates[identity].reset_index(drop=True).equals(old[identity]):
        raise AssertionError("v1 candidate identities did not reproduce")
    for col in ("candidate_frame_float", "candidate_time_sec", "strength"):
        np.testing.assert_allclose(candidates[col], old[col], rtol=1e-10, atol=1e-10)
    old_metrics = pd.read_csv(output / "evaluation.csv").set_index("scope")
    for row in rows:
        if row["variant"] != "v1":
            continue
        for key in ("n_candidates", "recall_clear_f6", "recall_clear_f10", "recall_all_f6",
                    "recall_all_f10", "review_coverage_plusminus10_frames"):
            np.testing.assert_allclose(row[key], old_metrics.loc[row["scope"], key], rtol=1e-12)


def write_results(output, frozen, table, elapsed):
    pct = base.percent
    rule = frozen["rule"]
    text = [README_START, "## v2：候補の音響特徴による絞り込み", "",
            "既存の4–12 kHz・閾値1.0・中央値0.5秒・最小間隔0.15秒の候補をそのまま作り、"
            "時刻を動かさずに選別する。元のCSV・JSON・図は上書きしない。上の前回結果は保存した。", "",
            "```powershell",
            "& study\\.venv\\Scripts\\python.exe -B -X utf8 outputs\\shot_pipeline_20261117\\audio\\detect_hit_sounds.py --v2",
            "# 探索だけ：上のコマンドに --phase train。固定設定の評価だけ：--phase evaluate",
            "```", "",
            "`--output-dir` は前回の候補・設定・評価ファイルと親のlabelsがあるディレクトリを指定する。"
            "WAVは一時ディレクトリだけに保存し、正常終了・例外終了時に削除する。commitは行わない。", "",
            "### 特徴量と固定した規則", "",
            "すべて48 kHz mono、2 ms平均二乗・1 ms刻み。時刻補正なし。帯域は4次Butterworthのゼロ位相処理。",
            "- `half_width_ms`：高域の線形エネルギーから501 ms局所中央値を引いたピークの半値幅。"
            "交点を補間し、左右各100 msで打ち切る。打切りは`width_censored`に記録。",
            "- `low_rise_ln` / `high_rise_ln`：0.5–2 / 4–12 kHzの対数エネルギーと501 ms局所中央値との差。"
            "低域は候補の±4 ms内の最大、高域は候補時刻。両方の小さい値を`paired_rise_ln`にする。"
            "低域ピークのずれも±4/±20 ms内で保存する。",
            "- `post50_drop_ln`：高域ピークエネルギー÷45–55 ms後の平均エネルギーの自然対数。大きいほど急減。",
            "- `energy_concentration_20_100`：局所中央値より大きい高域エネルギーの±20 ms積算÷±100 ms積算。",
            "- `spectral_flatness_100ms`：±100 msの短時間平均PSDについて0.5–12 kHzで幾何平均÷算術平均。"
            "調波音の参考特徴として保存するが、探索には使わない。", "",
            f"選択規則 **{frozen['rule_id']}**（以下の有効条件をすべて満たす）：", ""]
    descriptions = {
        "max_width_ms": lambda v: f"半値幅 ≤ **{v:g} ms**",
        "min_paired_rise_ln": lambda v: f"両帯域の立ち上がり（対数局所差の小さい方） ≥ **{v:g}**",
        "min_post50_drop_ln": lambda v: f"50 ms後の低下（自然対数） ≥ **{v:g}**（後/ピーク ≤ {math.exp(-v):.3f}）",
        "min_concentration": lambda v: f"±20/±100 msエネルギー比 ≥ **{v:g}**",
    }
    text += ["- " + descriptions[k](v) for k, v in rule.items() if v is not None]
    text += ["", "第7・8点だけで探索。候補の教師値は、同じ点/rallyにclearまたはinferredのラベルが"
             "整数フレーム±10F以内に1件でもあれば正例（候補間のラベル共有を許す）。"
             "評価用の1対1対応とは分ける。", "",
             "探索格子（`None`は条件なし）：`" + json.dumps(GRID) + "`。"
             f"有効条件は最大3つ、計{frozen['n_rules']}規則。"
             f"前回の探索側±10F対応数の90%以上（clear ≥ {frozen['recall_floor']['n_matched_clear_f10']}件、"
             f"全 ≥ {frozen['recall_floor']['n_matched_all_f10']}件）を保つ規則から、"
             "主指標の窓合計割合最小を選ぶ。同率は候補教師値に対するF1最大→条件数最小→規則ID順。"
             "この90%制約は探索時の設計値で、評価側の保証ではない。"
             "教師値は同じラベル近傍の複数候補も正例にするため、F1を主目的にすると重複候補を残しやすい。", "",
             "**評価前の探索側だけでの目的関数変更**：当初の候補F1最大→窓合計最小では114件中110件を保持した。"
             "第9〜11点を一度も読み込む前に、同じ90%再現率維持制約の下で窓合計最小→F1最大へ変更した。"
             "当初の選択は`initial_f1_selection_v2.json`に保存。評価側を理由にした再探索・規則変更はない。", "",
             "`selected_config_v2.json`を保存してから第9〜11点のラベル・音声を読み込む。"
             "特徴定義・格子・選択順・曲線の閾値列もこの時点で固定し、評価結果による再調整は行っていない。"
             "ただし特徴仮説の入力には第9点スペクトログラムの観察が含まれるため、"
             "第9〜11点は既存評価区間への固定規則の再評価であり、完全未接触の独立テストではない。", "",
             "### 前回との比較", "",
             "**主指標は各候補±10F窓の秒数の単純合計÷区間秒数**。各窓を区間端で切り、"
             "重複部分も候補ごとに数えるため100%を超える。連続時間の20F幅で、21枚とは数えない。"
             "別に前回と同じ和集合割合（重複を除く確認範囲）も示す。"
             "どちらも理論的な再生量であり、実測の人の確認・操作・修正時間ではない。", "",
             "| 集合 | 方法 | clear±6F | clear±10F | 全±6F | 全±10F | 適合率目安±6F/±10F | 候補数 | 候補/秒 | 窓合計/区間（主） | 確認範囲（和集合） |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for scope in ("train_07_08", "test_09_11"):
        for variant in ("v1", "v2"):
            r = table[(table.scope == scope) & (table.variant == variant)].iloc[0]
            text.append(f"| {'第7・8点（探索）' if scope.startswith('train') else '第9〜11点（評価）'} | {variant} | "
                        f"{int(r.n_matched_clear_f6)}/{int(r.n_labels_clear)} ({pct(r.recall_clear_f6)}) | "
                        f"{int(r.n_matched_clear_f10)}/{int(r.n_labels_clear)} ({pct(r.recall_clear_f10)}) | "
                        f"{pct(r.recall_all_f6)} | {pct(r.recall_all_f10)} | "
                        f"{pct(r.precision_proxy_f6)} / {pct(r.precision_proxy_f10)} | "
                        f"{int(r.n_candidates)} | {r.candidates_per_sec:.3f} | "
                        f"**{pct(r.review_time_ratio_plusminus10_frames)}** | {pct(r.review_coverage_plusminus10_frames)} |")
    old, new = [table[(table.scope == "test_09_11") & (table.variant == v)].iloc[0] for v in ("v1", "v2")]
    text += ["", f"評価区間は合計{new.duration_sec:.6f}秒。窓合計は{old.review_window_sum_sec:.3f}→"
             f"{new.review_window_sum_sec:.3f}秒（{pct(1-new.review_window_sum_sec/old.review_window_sum_sec)}減）、"
             f"候補は{int(old.n_candidates)}→{int(new.n_candidates)}件。"
             f"clearの±10F対応は{int(old.n_matched_clear_f10)}→{int(new.n_matched_clear_f10)}/{int(new.n_labels_clear)}件。"
             "再現率と削減量を併記し、削減だけを改善の根拠にしない。", "",
             "### 点別の固定規則の結果", "",
             "| 点 | clear±6F | clear±10F | 全±10F | 候補数 | 候補/秒 | 窓合計/区間 | 確認範囲（和集合） |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in table[(table.variant == "v2") & table.scope.str.startswith("point_")].itertuples():
        text.append(f"| {r.scope} | {int(r.n_matched_clear_f6)}/{int(r.n_labels_clear)} ({pct(r.recall_clear_f6)}) | "
                    f"{pct(r.recall_clear_f10)} | {pct(r.recall_all_f10)} | {r.n_candidates} | {r.candidates_per_sec:.3f} | "
                    f"{pct(r.review_time_ratio_plusminus10_frames)} | {pct(r.review_coverage_plusminus10_frames)} |")
    text += ["", "### 曲線と出力", "",
             "`tradeoff_v2.png`（要求名`tradeoff.png`も同内容）に第9〜11点の合算を描く。"
             "横軸は和集合割合と主指標の窓合計割合の2パネル。縦軸はclear±6/±10Fと全±10F。"
             "星は探索で決めた閾値0。選択された各条件の余裕を固定尺度で正規化し、その最小値をスコアにする。"
             "幅は10 ms、立ち上がり・低下は対数1、集中比は0.2が1単位。"
             "閾値−4〜4を0.1刻み、および全候補・候補なしを描く。"
             "評価側で良い点を選び直さず、曲線は感度分析だけに使う。"
             "貪欲対応の競合により曲線は単調とは限らず、単調化しない。", "",
             "- `candidate_features_v2.csv`：前回の全候補・特徴量・固定規則の余裕・保持/除外・条件別除外理由。",
             "- `candidates_v2.csv`：残った候補のみ（元のcandidate_idと時刻を維持）。",
             "- `train_features_v2.csv`：第7・8点の候補教師値・最近ラベル差付き特徴量。",
             "- `evaluation_v2.csv`：v1/v2の点別・探索/評価合算。従来指標と窓合計秒・割合を保存。",
             "- `label_matches_v2.csv`：v1/v2の各ラベルの1対1対応。",
             "- `param_search_v2.csv` / `selected_config_v2.json`：探索全規則と固定選択。",
             "- `tradeoff_v2.csv`：閾値別の評価合算・点別指標。",
             "- `spectrogram_point_XX_v2.png`：保持は緑の三角、除外は灰の×。ラベルは従来の青/橙。",
             "- `intervals_v2.csv` / `run_metadata_v2.json`：区間・入力ハッシュ・規則固定時刻・一時WAV削除・前回一致検証。",
             "- PNGはgitignore対象。前回スクリプトは`--v2`なしで再実行できる（その場合は前回出力を更新する）。", "",
             "### 研究上の解釈", "",
             "短い広帯域音という仮説に基づく選別であり、打球・バウンド・足音の音源分類を独立検証していない。"
             "低域/高域の局所差の同時増加は広帯域性の代理で、真の音源同期や音響伝搬の証明ではない。"
             "ラベル誤差が±6Fより大きい可能性を踏まえて±10Fを探索に使うが、ラベルを音に合わせて動かしていない。"
             "±10Fの一致にも偶然一致が含まれ得る。clearは独立正解ではなく、未ラベルの打球もあり得る。", "",
             "同一試合2点からの探索、小さい評価母数（clear18件）、第9点由来の特徴仮説により一般化の主張は限定される。"
             "人の確認量削減を実証するには、別の未使用区間で元動画から独立に正解を作り、"
             "全区間確認と候補窓確認の最終欠落率・確認修正の実時間を比較する必要がある。", "",
             f"今回のv2実行時間：{elapsed:.2f}秒。", README_END]
    path = output / "README.md"
    old_bytes = path.read_bytes()
    if README_START.encode() in old_bytes:
        before, rest = old_bytes.split(README_START.encode(), 1)
        _, after = rest.split(README_END.encode(), 1)
        path.write_bytes(before + "\n".join(text).encode("utf-8") + after)
    else:
        # Append bytes to preserve all original README bytes, including CRLF.
        with path.open("ab") as stream:
            stream.write(("\n\n" + "\n".join(text) + "\n").encode("utf-8"))


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--ffmpeg", type=Path, default=Path(r"C:\ffmpeg\bin\ffmpeg.exe"))
    parser.add_argument("--phase", choices=("all", "train", "evaluate"), default="all")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    pipeline = args.pipeline_dir.resolve()
    manifest = baseline_manifest(output)
    config = baseline_config(output)
    protocol = pipeline / "tracking" / "point_07" / "protocol.json"
    video = Path(json.loads(protocol.read_text(encoding="utf-8"))["source_video"])
    if not video.is_file() or not args.ffmpeg.is_file():
        raise FileNotFoundError("Source video or ffmpeg missing")
    train_segments = base.load_segments(pipeline, base.TRAIN_POINTS)
    with tempfile.TemporaryDirectory(prefix="match2_hit_audio_v2_") as scratch:
        scratch_path = Path(scratch)
        base.extract_and_featurize(train_segments, video, args.ffmpeg, scratch_path)
        train_candidates = add_features(train_segments, base.detect(train_segments, config))
        frozen_path = output / "selected_config_v2.json"
        if args.phase in ("all", "train"):
            frozen = train_rules(train_segments, train_candidates, output, config, manifest, pipeline)
        else:
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            if frozen["feature_script_sha256"] != sha256(__file__) or frozen["baseline_script_sha256"] != sha256(base.__file__):
                raise ValueError("Code changed since freeze: train again before evaluating")
            if frozen["baseline_artifact_sha256"] != manifest:
                raise ValueError("Baseline artifacts changed since freeze")
            for p in base.TRAIN_POINTS:
                if frozen["train_label_sha256"][str(p)] != sha256(pipeline / "labels" / f"point_{p:02d}.csv"):
                    raise ValueError("Training labels changed since freeze")
        frozen_sha = sha256(frozen_path)
        if args.phase == "train":
            print("Training complete; held-out labels/audio not loaded.", flush=True)
            return
        # First held-out read: rule, feature code, selection and sweep are fixed.
        test_segments = base.load_segments(pipeline, base.TEST_POINTS)
        base.extract_and_featurize(test_segments, video, args.ffmpeg, scratch_path)
        segments = train_segments + test_segments
        all_candidates = base.detect(segments, config)
        candidates = add_features(segments, all_candidates)
        candidates["split"] = np.where(candidates.point.isin(base.TRAIN_POINTS), "train", "test")
        candidates["rule_margin"] = gate_margin(candidates, frozen["rule"])
        candidates["kept"] = candidates.rule_margin >= 0
        candidates["reject_reasons"] = [";".join(k for k, (feature, sign, _) in GATES.items()
                                                   if frozen["rule"][k] is not None and
                                                   sign * (getattr(c, feature) - frozen["rule"][k]) < 0)
                                          for c in candidates.itertuples()]
        candidates.to_csv(output / "candidate_features_v2.csv", index=False)
        kept = candidates[candidates.kept].copy()
        kept.to_csv(output / "candidates_v2.csv", index=False)
        rows, details = [], []
        scopes = [([s for s in segments if s.point == p], f"point_{p:02d}",
                   "train" if p in base.TRAIN_POINTS else "test")
                  for p in base.TRAIN_POINTS + base.TEST_POINTS]
        scopes += [(train_segments, "train_07_08", "train"), (test_segments, "test_09_11", "test")]
        for subset, scope, split in scopes:
            points = [s.point for s in subset]
            for variant, source in (("v1", candidates), ("v2", kept)):
                row, matches = metrics(subset, source[source.point.isin(points)], scope, split)
                rows.append({"variant": variant, **row})
                if scope.startswith("point_"):
                    details.extend({"variant": variant, **m} for m in matches)
            if scope.startswith("point_"):
                plot_point(subset, candidates[candidates.point.isin(points)], config, output)
        verify_baseline(output, candidates, rows)
        evaluation = pd.DataFrame(rows)
        evaluation.to_csv(output / "evaluation_v2.csv", index=False)
        pd.DataFrame(details).to_csv(output / "label_matches_v2.csv", index=False)
        tradeoff(test_segments, candidates[candidates.split == "test"], frozen, output)
        intervals = [{"point": s.point, "rally": s.rally, "start_frame": s.start_frame,
                      "end_frame": s.end_frame, "duration_sec": s.duration_sec} for s in segments]
        pd.DataFrame(intervals).to_csv(output / "intervals_v2.csv", index=False)
    if frozen_sha != sha256(frozen_path):
        raise AssertionError("Frozen rule changed during evaluation")
    if manifest != baseline_manifest(output):
        raise AssertionError("v1 artifact overwritten")
    elapsed = time.perf_counter() - started
    metadata = {"source_video": str(video), "source_video_bytes": video.stat().st_size,
                "source_video_mtime_ns": video.stat().st_mtime_ns,
                "input_sha256": {str(p): sha256(p) for p in
                                 [protocol, *[pipeline / "labels" / f"point_{p:02d}.csv"
                                              for p in base.TRAIN_POINTS + base.TEST_POINTS]]},
                "script_sha256": sha256(__file__), "baseline_script_sha256": sha256(base.__file__),
                "frozen_config_sha256_before_evaluation": frozen_sha,
                "frozen_config_sha256_after_evaluation": sha256(frozen_path),
                "baseline_artifact_sha256": manifest, "baseline_artifacts_unchanged": True,
                "baseline_candidates_and_metrics_reproduced": True,
                "held_out_retuning": False, "feature_design_used_point9_observations": True,
                "scratch_deleted": not scratch_path.exists(), "elapsed_sec": elapsed,
                "ffmpeg_version": subprocess.run([str(args.ffmpeg), "-version"], check=True,
                                                 capture_output=True, text=True).stdout.splitlines()[0],
                "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                             "scipy": __import__("scipy").__version__,
                             "matplotlib": __import__("matplotlib").__version__},
                "main_review_metric": "sum of separately clipped continuous +/-10F windows / duration; overlaps retained",
                "matching": "unchanged v1 closest-first greedy 1-to-1 on all labels, separately at +/-6 and +/-10F"}
    dump_json(output / "run_metadata_v2.json", metadata)
    write_results(output, frozen, evaluation, elapsed)
    print(evaluation[evaluation.scope.isin(["train_07_08", "test_09_11"])][
        ["scope", "variant", "n_candidates", "recall_clear_f6", "recall_clear_f10",
         "candidates_per_sec", "review_time_ratio_plusminus10_frames",
         "review_coverage_plusminus10_frames"]].to_string(index=False), flush=True)
    print(f"Finished in {elapsed:.2f}s; temporary WAV removed: {not scratch_path.exists()}", flush=True)


if __name__ == "__main__":
    main()
