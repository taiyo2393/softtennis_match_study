"""Audio-only impact proposals; select on points 7/8, evaluate once on 9/10/11.

Run from the repository root with study/.venv/Scripts/python.exe -B -X utf8.
Only files inside --output-dir are written. Decoded WAVs live in tempfile.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
from scipy.io import wavfile


SAMPLE_RATE = 48000
FRAME_SEC = 1001 / 60000
ENERGY_SAMPLES = 96  # 2 ms, with a 1 ms hop
HOP_SAMPLES = 48
PADDING_SEC = 1.5
TRAIN_POINTS = (7, 8)
TEST_POINTS = (9, 10, 11)
BANDS = ((1000, 4000), (2000, 6000), (3000, 10000),
         (1000, 10000), (4000, 12000))
MEDIAN_WINDOWS = (0.3, 0.5, 1.0)
THRESHOLDS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
MIN_INTERVALS = (0.06, 0.10, 0.15, 0.20)


@dataclass(frozen=True)
class Config:
    config_id: str
    low_hz: int
    high_hz: int
    median_sec: float
    threshold: float
    min_interval_sec: float
    method: str = "bandpass_log_energy_minus_local_median"
    energy_ms: float = 2.0
    hop_ms: float = 1.0
    filter_order: int = 4
    channel: str = "mono_ffmpeg_downmix"
    time_offset_sec: float = 0.0


@dataclass
class Segment:
    point: int
    rally: str
    start_frame: int
    end_frame: int
    labels: pd.DataFrame
    audio: np.ndarray | None = None
    audio_start_sec: float = 0.0
    feature_frames: np.ndarray | None = None
    features: dict | None = None

    @property
    def key(self):
        return f"{self.point:02d}{self.rally}"

    @property
    def duration_sec(self):
        return (self.end_frame - self.start_frame) * FRAME_SEC


def configurations():
    for i, ((low, high), median, height, distance) in enumerate(itertools.product(
            BANDS, MEDIAN_WINDOWS, THRESHOLDS, MIN_INTERVALS), 1):
        yield Config(f"cfg_{i:03d}", low, high, median, height, distance)


def load_segments(pipeline: Path, points: tuple[int, ...]) -> list[Segment]:
    segments = []
    for point in points:
        labels = pd.read_csv(pipeline / "labels" / f"point_{point:02d}.csv",
                             keep_default_na=False)
        required = {"point", "rally", "shot_index", "contact_frame",
                    "hitter_side", "confidence"}
        if not required.issubset(labels.columns) or labels.empty:
            raise ValueError(f"Invalid or empty labels for point {point}")
        if not labels.confidence.isin(["clear", "inferred"]).all():
            raise ValueError("Unknown confidence")
        if not labels.hitter_side.isin(["N", "F"]).all():
            raise ValueError("Unknown hitter_side")
        if not labels.point.eq(point).all():
            raise ValueError("Point mismatch")
        if not np.equal(labels.contact_frame, np.floor(labels.contact_frame)).all():
            raise ValueError("Non-integer contact frame")
        labels["label_id"] = [f"p{point:02d}_{r.rally}_s{int(r.shot_index):02d}"
                              for r in labels.itertuples()]
        if labels.label_id.duplicated().any():
            raise ValueError("Duplicate label_id")
        for rally, group in labels.groupby("rally", sort=True):
            group = group.sort_values(["contact_frame", "shot_index"]).reset_index(drop=True)
            segments.append(Segment(point, str(rally), int(group.contact_frame.min()) - 90,
                                    int(group.contact_frame.max()) + 90, group))
    for left, right in zip(segments, segments[1:]):
        if left.end_frame > right.start_frame:
            raise ValueError("Overlapping intervals would double-count audio")
    return segments


def extract_and_featurize(segments: list[Segment], video: Path, ffmpeg: Path,
                         scratch: Path) -> None:
    """Read audio only. Pad intervals to avoid filter/median boundary effects."""
    for seg in segments:
        first_sample = max(0, math.floor((seg.start_frame * FRAME_SEC - PADDING_SEC)
                                        * SAMPLE_RATE))
        last_sample = math.ceil((seg.end_frame * FRAME_SEC + PADDING_SEC) * SAMPLE_RATE)
        seg.audio_start_sec = first_sample / SAMPLE_RATE
        wav = scratch / f"point_{seg.key}.wav"
        cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-ss", f"{seg.audio_start_sec:.9f}", "-i", str(video),
               "-t", f"{(last_sample - first_sample) / SAMPLE_RATE:.9f}",
               "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
               "-c:a", "pcm_s16le", str(wav)]
        subprocess.run(cmd, check=True, capture_output=True)
        rate, audio = wavfile.read(wav)
        if rate != SAMPLE_RATE or audio.ndim != 1:
            raise ValueError("Expected 48 kHz mono WAV")
        if abs(len(audio) - (last_sample - first_sample)) > 2:
            raise ValueError(f"Unexpected decoded sample count: {seg.key}")
        seg.audio = audio.astype(np.float64) / 32768.0
        starts = np.arange(0, len(audio) - ENERGY_SAMPLES + 1, HOP_SAMPLES)
        seg.feature_frames = ((starts + (ENERGY_SAMPLES - 1) / 2) / SAMPLE_RATE
                              + seg.audio_start_sec) / FRAME_SEC
        seg.features = {}
        for band in BANDS:
            sos = signal.butter(4, band, btype="bandpass", fs=SAMPLE_RATE, output="sos")
            filtered = signal.sosfiltfilt(sos, seg.audio)
            squared_sum = np.r_[0.0, np.cumsum(filtered * filtered)]
            power = (squared_sum[starts + ENERGY_SAMPLES] - squared_sum[starts]) / ENERGY_SAMPLES
            log_energy = np.log(np.maximum(power, 1e-12))
            for window in MEDIAN_WINDOWS:
                median_size = int(round(window * SAMPLE_RATE / HOP_SAMPLES))
                median_size += 1 - median_size % 2
                baseline = ndimage.median_filter(log_energy, size=median_size, mode="reflect")
                seg.features[(*band, window)] = log_energy - baseline
        print(f"Features ready: point {seg.key}, {seg.duration_sec:.3f}s", flush=True)


def detect(segments: list[Segment], config: Config) -> pd.DataFrame:
    records = []
    for seg in segments:
        score = seg.features[(config.low_hz, config.high_hz, config.median_sec)]
        distance = math.ceil(config.min_interval_sec * SAMPLE_RATE / HOP_SAMPLES)
        peaks, _ = signal.find_peaks(score, height=config.threshold, distance=distance)
        for index in peaks:
            frame_float = float(seg.feature_frames[index])
            frame = int(math.floor(frame_float + 0.5))
            # Evaluation intervals are half-open continuous frame-time intervals.
            if not (seg.start_frame <= frame_float < seg.end_frame
                    and seg.start_frame <= frame < seg.end_frame):
                continue
            records.append({"point": seg.point, "rally": seg.rally,
                            "candidate_frame": frame, "candidate_frame_float": frame_float,
                            "candidate_time_sec": frame_float * FRAME_SEC,
                            "strength": float(score[index])})
    columns = ["point", "rally", "candidate_frame", "candidate_frame_float",
               "candidate_time_sec", "strength"]
    result = pd.DataFrame(records, columns=columns)
    result.insert(0, "candidate_id", [f"c{i:04d}" for i in range(1, len(result) + 1)])
    return result


def greedy_match(labels: pd.DataFrame, candidates: pd.DataFrame,
                 tolerance: int) -> dict[str, tuple[str, int]]:
    """Closest-first global greedy matching within each point/rally.

    The same all-label assignment underlies clear/all recall and side strata.
    Tie-break: label frame, candidate frame, label ID, candidate ID.
    """
    pairs = []
    for label in labels.itertuples():
        for candidate in candidates.itertuples():
            if label.point != candidate.point or label.rally != candidate.rally:
                continue
            delta = int(candidate.candidate_frame - label.contact_frame)
            if abs(delta) <= tolerance:
                pairs.append((abs(delta), int(label.contact_frame),
                              int(candidate.candidate_frame), label.label_id,
                              candidate.candidate_id, delta))
    used_labels, used_candidates, matches = set(), set(), {}
    for _, _, _, label_id, candidate_id, delta in sorted(pairs):
        if label_id not in used_labels and candidate_id not in used_candidates:
            matches[label_id] = (candidate_id, delta)
            used_labels.add(label_id)
            used_candidates.add(candidate_id)
    return matches


def fraction(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def review_coverage(segments, candidates, radius_frames=10):
    """Union of theoretical +/-radius review windows, clipped to analysis intervals."""
    length = 0.0
    for seg in segments:
        subset = candidates[(candidates.point == seg.point) & (candidates.rally == seg.rally)]
        end = float(seg.start_frame)
        for center in sorted(subset.candidate_frame):
            lo, hi = max(seg.start_frame, center - radius_frames), min(seg.end_frame, center + radius_frames)
            length += max(0, hi - max(end, lo))
            end = max(end, hi)
    return fraction(length * FRAME_SEC, sum(seg.duration_sec for seg in segments))


def evaluate(segments: list[Segment], candidates: pd.DataFrame,
             scope: str, split: str) -> tuple[dict, list[dict]]:
    labels = pd.concat([seg.labels for seg in segments], ignore_index=True)
    duration = sum(seg.duration_sec for seg in segments)
    clear = labels[labels.confidence == "clear"]
    row = {"scope": scope, "split": split, "points": ";".join(map(str, sorted(labels.point.unique()))),
           "duration_sec": duration, "n_labels_all": len(labels), "n_labels_clear": len(clear),
           "n_candidates": len(candidates), "candidates_per_sec": len(candidates) / duration,
           "labels_per_sec": len(labels) / duration, "clear_labels_per_sec": len(clear) / duration,
           "candidate_to_label_ratio": fraction(len(candidates), len(labels)),
           "review_coverage_plusminus10_frames": review_coverage(segments, candidates)}
    details = []
    for tolerance in (6, 10):
        matches = greedy_match(labels, candidates, tolerance)
        hit_clear = sum(label_id in matches for label_id in clear.label_id)
        row[f"n_matched_all_f{tolerance}"] = len(matches)
        row[f"n_matched_clear_f{tolerance}"] = hit_clear
        row[f"recall_all_f{tolerance}"] = fraction(len(matches), len(labels))
        row[f"recall_clear_f{tolerance}"] = fraction(hit_clear, len(clear))
        row[f"precision_proxy_f{tolerance}"] = fraction(len(matches), len(candidates))
        # Sensitivity only; never used to select settings. Inferred cannot compete here.
        clear_only = greedy_match(clear, candidates, tolerance)
        row[f"recall_clear_separate_matching_f{tolerance}"] = fraction(len(clear_only), len(clear))
        for side in ("N", "F"):
            for confidence, subset in (("all", labels), ("clear", clear)):
                side_labels = subset[subset.hitter_side == side]
                count = sum(label_id in matches for label_id in side_labels.label_id)
                row[f"n_labels_{confidence}_{side}"] = len(side_labels)
                row[f"n_matched_{confidence}_{side}_f{tolerance}"] = count
                row[f"recall_{confidence}_{side}_f{tolerance}"] = fraction(count, len(side_labels))
        for name, ids in (("all", set(labels.label_id)), ("clear", set(clear.label_id))):
            deltas = [delta for label_id, (_, delta) in matches.items() if label_id in ids]
            row[f"delta_{name}_median_f{tolerance}"] = float(np.median(deltas)) if deltas else np.nan
            row[f"delta_{name}_min_f{tolerance}"] = min(deltas) if deltas else np.nan
            row[f"delta_{name}_max_f{tolerance}"] = max(deltas) if deltas else np.nan
            row[f"abs_delta_{name}_median_f{tolerance}"] = float(np.median(np.abs(deltas))) if deltas else np.nan
        for label in labels.itertuples():
            candidate_id, delta = matches.get(label.label_id, ("", np.nan))
            details.append({"point": label.point, "rally": label.rally, "label_id": label.label_id,
                            "contact_frame": label.contact_frame, "confidence": label.confidence,
                            "hitter_side": label.hitter_side, "tolerance_frames": tolerance,
                            "candidate_id": candidate_id, "delta_frames": delta,
                            "matched": bool(candidate_id)})
    return row, details


def search(train_segments, output):
    rows, configs = [], {}
    for config in configurations():
        candidates = detect(train_segments, config)
        metrics, _ = evaluate(train_segments, candidates, "train_07_08", "train")
        feasible = metrics["n_candidates"] <= 2 * metrics["n_labels_all"]
        rows.append({**asdict(config), **metrics, "density_constraint_pass": feasible})
        configs[config.config_id] = config
    table = pd.DataFrame(rows)
    eligible = table[table.density_constraint_pass].sort_values(
        ["n_matched_clear_f6", "n_candidates", "config_id"], ascending=[False, True, True])
    if eligible.empty:
        table.to_csv(output / "param_search.csv", index=False, encoding="utf-8")
        raise RuntimeError("No setting meets the candidate density constraint; no held-out evaluation.")
    best_id = str(eligible.iloc[0].config_id)
    table["selected"] = table.config_id.eq(best_id)
    table.to_csv(output / "param_search.csv", index=False, encoding="utf-8")
    chosen = configs[best_id]
    # Persist the train-only choice before loading held-out labels or audio.
    (output / "selected_config.json").write_text(
        json.dumps({**asdict(chosen), "selected_on": list(TRAIN_POINTS),
                    "density_limit": "n_candidates <= 2 * n_labels_all (pooled train)",
                    "order": ["max n_matched_clear_f6", "min n_candidates", "config_id ascending"]},
                   indent=2) + "\n", encoding="utf-8")
    print(f"Frozen train-only configuration: {chosen}", flush=True)
    return chosen


def plot_point(segments, candidates, config, output):
    fig, axes = plt.subplots(2, len(segments), squeeze=False, sharex="col",
                             figsize=(24 if len(segments) > 1 else 22, 8),
                             gridspec_kw={"width_ratios": [s.duration_sec for s in segments]})
    for column, seg in enumerate(segments):
        top, bottom = axes[:, column]
        frequencies, times, spectrum = signal.spectrogram(
            seg.audio, fs=SAMPLE_RATE, window="hann", nperseg=512, noverlap=384,
            scaling="density", mode="psd")
        frame_times = (times + seg.audio_start_sec) / FRAME_SEC
        select_t = (frame_times >= seg.start_frame) & (frame_times < seg.end_frame)
        select_f = frequencies <= 12000
        db = 10 * np.log10(np.maximum(spectrum[np.ix_(select_f, select_t)], 1e-14))
        top.pcolormesh(frame_times[select_t], frequencies[select_f] / 1000, db,
                       shading="auto", cmap="magma", vmin=-110, vmax=-40, rasterized=True)
        top.set_ylim(0, 12)
        top.set_ylabel("Frequency (kHz)")
        top.set_title(f"Point {seg.point:02d}, rally {seg.rally} | spectrogram (dB/Hz, -110 to -40)")
        score = seg.features[(config.low_hz, config.high_hz, config.median_sec)]
        core = (seg.feature_frames >= seg.start_frame) & (seg.feature_frames < seg.end_frame)
        bottom.plot(seg.feature_frames[core], score[core], color="#202020", linewidth=0.7)
        bottom.axhline(config.threshold, color="#777777", linestyle="--", linewidth=1)
        for label in seg.labels.itertuples():
            for ax in (top, bottom):
                ax.axvline(label.contact_frame, color="#2187db" if label.confidence == "clear" else "#f6a832",
                           linestyle="-" if label.confidence == "clear" else ":", linewidth=1.15, alpha=0.85)
        local = candidates[(candidates.point == seg.point) & (candidates.rally == seg.rally)]
        top.scatter(local.candidate_frame, np.full(len(local), 11.7), marker="v", s=34,
                    color="#f1396a", edgecolors="white", linewidth=0.3, zorder=5)
        bottom.scatter(local.candidate_frame, local.strength, marker="^", s=28,
                       color="#d32350", zorder=5)
        bottom.set_xlabel("Source video frame (60000/1001 fps)")
        bottom.set_ylabel("ln(2 ms energy) - local median")
        bottom.set_title(f"{config.low_hz}-{config.high_hz} Hz, median {config.median_sec:g}s, "
                         f"threshold {config.threshold:g}, min gap {config.min_interval_sec:g}s")
        bottom.grid(alpha=0.15)
        bottom.set_xlim(seg.start_frame, seg.end_frame)
        bottom.ticklabel_format(axis="x", useOffset=False, style="plain")
    handles = [Line2D([0], [0], color="#2187db", label="clear label", linestyle="-"),
               Line2D([0], [0], color="#f6a832", label="inferred label", linestyle=":"),
               Line2D([0], [0], color="#d32350", marker="^", linestyle="None", label="audio candidate")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output / f"spectrogram_point_{segments[0].point:02d}.png", dpi=180)
    plt.close(fig)


def percent(value):
    return f"{100 * value:.1f}%" if np.isfinite(value) else "算出不可"


def write_readme(output, config, evaluation, intervals, elapsed):
    lines = ["# match2 音声による打球瞬間の候補抽出", "",
             "## 目的と実施範囲", "",
             "中継音声だけから確認候補を出し、Claudeが映像から付けた既存ラベルとの一致を測る予備実験。"
             "候補前後の映像を人が確認する方式の成立性を調べる。動画の画像解析・新しい目視ラベル付けは行わない。"
             "clearも人が独立に検証した正解ではないため、以下は既存ラベルとの一致率である。", "",
             "## 再実行", "", "リポジトリルートのPowerShellで実行：", "", "```powershell",
             "& study\\.venv\\Scripts\\python.exe -B -X utf8 outputs\\shot_pipeline_20261117\\audio\\detect_hit_sounds.py",
             "```", "", "既定のffmpegは `C:\\ffmpeg\\bin\\ffmpeg.exe`。`--ffmpeg`、`--pipeline-dir`、"
             "`--output-dir` で変更可能。生成物だけを出力先に上書きする。WAVは `tempfile.TemporaryDirectory` "
             "内に抽出し、終了時に削除する。元動画・元ラベル・既存成果物は変更しない。git commitは行わない。", "",
             "## 区間と時刻", "",
             "各rallyの最初のcontact_frame−90から最後+90まで。第7点はA/Bを別区間とし、Aも今回の評価に含める。"
             "区間は連続時刻の半開区間 `[start_frame, end_frame)`、長さは `(end-start)×1001/60000` 秒。"
             "全体の待ち時間やラリー区間の自動発見は評価対象外で、既存ラベルによる区間が与えられている。", "",
             "| 点/rally | 開始F | 終了F（含まない） | 秒 | clear | 全ラベル |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in intervals:
        lines.append(f"| {row['point']:02d}{row['rally']} | {row['start_frame']} | {row['end_frame']} | "
                     f"{row['duration_sec']:.6f} | {row['n_labels_clear']} | {row['n_labels_all']} |")
    lines += ["", "フレームfの時刻は `f×1001/60000` 秒。音声は開始時刻0の48 kHz Opusステレオという入力条件に従い、"
              "ffmpegで `-vn -ac 1 -ar 48000` のmono PCM16 WAVに変換する。各区間の前後1.5秒を追加抽出し、"
              "フィルタと中央値の端効果を避けてから区間内だけを採用する。音速・中継の遅延補正は0秒固定で、"
              "ラベルに合わせた時刻移動をしない。候補時刻は2 ms窓のサンプル中心。小数フレームも保存し、"
              "照合は `floor(frame_float+0.5)` で最近整数に丸めたcandidate_frameを使う。", "",
              "## 検出法と探索", "",
              "4次Butterworth帯域通過をsosfiltfiltでゼロ位相適用 → 2 ms平均二乗エネルギー（1 ms刻み）"
              "→ 自然対数（下限1e-12）→ 中央の局所中央値を減算 → scipy.signal.find_peaksの高さと最小間隔で検出。"
              "中央値窓は1 ms刻みで最近奇数へ切り上げるため、実際は301/501/1001サンプル。強さはこの対数差で、"
              "確率ではない。ステレオ差分とspectral fluxは今回の探索には含めない。", "",
              f"- 帯域Hz：{', '.join(f'{lo}–{hi}' for lo, hi in BANDS)}",
              f"- 局所中央値秒：{MEDIAN_WINDOWS}", f"- 高さ閾値：{THRESHOLDS}",
              f"- 最小間隔秒：{MIN_INTERVALS}", "- 合計480設定。全設定の探索用成績をparam_search.csvに保存。", "",
              "**設定選択基準**：第7・8点を合算し、候補/秒が全ラベルの打球数/秒の2倍以下 "
              "（候補118件以下）の設定からclearの±6フレーム再現率が最大のものを選ぶ。"
              "同率なら候補数が少ないもの、さらに同率ならconfig_id昇順。±10・適合率・側別・評価用成績は選択に使わない。"
              "第7・8点だけを読み込んで探索し、selected_config.jsonを保存した後に第9〜11点を読み込む。"
              "評価結果を見た再探索・再調整は行わない。", "",
              f"選択：**{config.config_id}、{config.low_hz}–{config.high_hz} Hz、中央値{config.median_sec:g}秒、"
              f"閾値{config.threshold:g}、最小間隔{config.min_interval_sec:g}秒**。", "",
              "## 1対1対応と指標", "",
              "各許容幅（±6、±10）で独立に、同じ点/rallyの候補と全ラベルの許容ペアを絶対フレーム差の近い順に並べる。"
              "両端が未使用のペアだけを採用する貪欲法。距離同率ではラベルF、候補F、label_id、candidate_id順。"
              "同じ対応結果をclear/allおよび手前N/奥Fで集計する。側ごとに対応をやり直さない。"
              "inferredも対応に参加するので、近接するclearとの競合は起こり得る。"
              "参考列 `recall_clear_separate_matching_f6/f10` はclearのみで別に1対1対応した感度分析であり、設定選択には使わない。", "",
              "- recall_clear/all_f6/f10：対応した該当ラベル数÷該当ラベル総数。",
              "- precision_proxy_f6/f10：全ラベルに対応した候補数÷候補総数。打球の全網羅が未確認なので**適合率の目安**。",
              "- candidates_per_sec / labels_per_sec：合計件数÷合計区間秒。合計行は点ごとの率の単純平均ではない。",
              "- delta_*：候補F−ラベルFの符号付きずれ。中央値・最小・最大、絶対値中央値を保存。空欄は対応/分母なし。",
              "- review_coverage_plusminus10_frames：候補±10Fの窓の和集合が区間秒に占める割合。重複を除き端で切る。"
              "確認映像の理論的範囲であり、人の操作・修正時間や打球捕捉の保証ではない。", "",
              "## 結果", "", "| 集合 | clear±6 | clear±10 | 全±6 | 全±10 | 適合率目安±6 | 候補/秒 | ラベル/秒 | 候補数 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in evaluation:
        lines.append(f"| {row['scope']} | {row['n_matched_clear_f6']}/{row['n_labels_clear']} "
                     f"({percent(row['recall_clear_f6'])}) | {percent(row['recall_clear_f10'])} | "
                     f"{percent(row['recall_all_f6'])} | {percent(row['recall_all_f10'])} | "
                     f"{percent(row['precision_proxy_f6'])} | {row['candidates_per_sec']:.3f} | "
                     f"{row['labels_per_sec']:.3f} | {row['n_candidates']} |")
    train, test = evaluation[-2:]
    lines += ["", "第7・8点は設定探索に使った値で、性能推定は第9〜11点を優先する。", "",
              "| 集合 | 手前N clear±6 | 奥F clear±6 | 手前N 全±6 | 奥F 全±6 | ずれ中央値 [範囲] F（全±6） |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in evaluation:
        cells = []
        for confidence, side in (("clear", "N"), ("clear", "F"), ("all", "N"), ("all", "F")):
            cells.append(f"{row[f'n_matched_{confidence}_{side}_f6']}/{row[f'n_labels_{confidence}_{side}']} "
                         f"({percent(row[f'recall_{confidence}_{side}_f6'])})")
        lines.append(f"| {row['scope']} | " + " | ".join(cells) +
                     f" | {row['delta_all_median_f6']:g} [{row['delta_all_min_f6']:g}, {row['delta_all_max_f6']:g}] |")
    lines += ["", f"評価用clearで±6Fに候補が対応しなかったのは **{test['n_labels_clear'] - test['n_matched_clear_f6']}"
              f"/{test['n_labels_clear']}件**。評価用の候補±10Fの和集合は解析区間の"
              f"**{percent(test['review_coverage_plusminus10_frames'])}**。"
              "この範囲だけの確認で十分かは、見逃しと未ラベル打球の独立確認が必要。", "",
              f"探索側は候補{train['n_candidates']}件/全ラベル{train['n_labels_all']}件="
              f"{train['candidate_to_label_ratio']:.3f}倍で2倍制約を満たしたが、評価側は候補"
              f"{test['n_candidates']}件/全ラベル{test['n_labels_all']}件="
              f"**{test['candidate_to_label_ratio']:.3f}倍**で2倍を超えた。"
              "評価側の上限超過を理由に設定を変更していない。", "",
              "**現状の判定：候補生成は動作するが、この設定から大幅な確認範囲の削減を主張する根拠は得られなかった。** "
              "±10Fでの高い再現率は、密な候補の確認窓がほぼ全区間を覆う条件で得られた値である。"
              "打球音を特異的に検出した証拠や、人の確認時間が短縮した証拠とは区別する。", "",
              "## 出力", "",
              "- candidates.csv：点/rally、候補ID、整数/小数F、秒、強さ、選択設定の全項目。",
              "- evaluation.csv：点別と探索用合計・評価用合計。全指標と側別の分母/分子。",
              "- param_search.csv：480設定すべての第7・8点での成績、密度制約、選択フラグ。",
              "- label_matches.csv：各ラベルの±6/±10での対応IDとずれ。不一致も残す。",
              "- intervals.csv / selected_config.json / run_metadata.json：区間、固定設定、入力ハッシュ・実行情報。",
              "- spectrogram_point_07.png〜11.png：上段0〜12 kHzスペクトログラム、下段検出スコア。"
              "clear実線（青）、inferred点線（橙）、候補三角（赤）。第7点はA/B別パネル。"
              "pngは既存gitignoreの対象。", "",
              "## 限界と研究上の判定", "",
              "1. Claudeの1回の映像ラベルであり、clearにも±6F程度の不確かさがある。この目安自体も独立検証していない。"
              "±6Fは約0.100秒、±10Fは約0.167秒。真の接触時刻に対する検出誤差とは区別する。",
              "2. inferredは前後関係からの推定を含み、未ラベルの中間打球も既存noteにある。"
              "ラベルのない候補を誤検出とは断定できず、対応した候補もバウンド・足音・声などとの偶然一致があり得る。"
              "本実験は音イベントの種類を独立に判別していない。",
              "3. 同一試合の連続した5点だけで、評価用clearは18件（手前5・奥13）。"
              "側別差の原因を音量と断定できない。第11点はエンド交代もある。"
              "480設定からの選択は探索側を楽観的にする。評価側は固定しているが、別試合・別収録への一般化は未検証。",
              "4. 解析区間はラベルから与えられた短い区間である。試合全編での候補密度や自動ラリー切り出しは評価していない。"
              "探索用の候補密度上限は評価用で満たされる保証がない。mono化でステレオの空間情報を捨てている。",
              "5. 音の伝搬・放送音声の遅延を補正しない。sosfiltfiltと中央中央値は未来の音声も使うためオフライン処理。"
              "貪欲対応は最大対応数を保証せず、密集した候補の偶然一致を差し引く指標でもない。",
              "6. 卒論の中心的な有効性を主張するには、独立に確認した打球ラベルを用い、"
              "全区間確認と候補前後確認の比較で、最終データの欠落/誤りと人の確認・修正時間を測る必要がある。"
              "今回の一致率と理論窓割合だけから作業時間短縮や戦術データの正しさは主張できない。", "",
              f"実行所要時間（この実行環境、抽出・探索・評価・図作成を含む）：{elapsed:.2f}秒。", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--ffmpeg", type=Path, default=Path(r"C:\ffmpeg\bin\ffmpeg.exe"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = args.pipeline_dir / "tracking" / "point_07" / "protocol.json"
    video = Path(json.loads(protocol.read_text(encoding="utf-8"))["source_video"])
    if not video.is_file() or not args.ffmpeg.is_file():
        raise FileNotFoundError("Source video or ffmpeg missing")
    train_segments = load_segments(args.pipeline_dir, TRAIN_POINTS)
    with tempfile.TemporaryDirectory(prefix="match2_hit_audio_") as scratch:
        extract_and_featurize(train_segments, video, args.ffmpeg, Path(scratch))
        chosen = search(train_segments, output)
        test_segments = load_segments(args.pipeline_dir, TEST_POINTS)
        extract_and_featurize(test_segments, video, args.ffmpeg, Path(scratch))
        segments = train_segments + test_segments
        candidates = detect(segments, chosen)
        for key, value in asdict(chosen).items():
            candidates[key] = value
        candidates["split"] = np.where(candidates.point.isin(TRAIN_POINTS), "train", "test")
        candidates.to_csv(output / "candidates.csv", index=False, encoding="utf-8")
        rows, details = [], []
        for point in TRAIN_POINTS + TEST_POINTS:
            subset = [s for s in segments if s.point == point]
            local_candidates = candidates[candidates.point == point]
            row, matches = evaluate(subset, local_candidates, f"point_{point:02d}",
                                    "train" if point in TRAIN_POINTS else "test")
            rows.append(row)
            details.extend(matches)
            plot_point(subset, local_candidates, chosen, output)
        for subset, points, scope, split in ((train_segments, TRAIN_POINTS, "train_07_08", "train"),
                                              (test_segments, TEST_POINTS, "test_09_11", "test")):
            row, _ = evaluate(subset, candidates[candidates.point.isin(points)], scope, split)
            rows.append(row)
        pd.DataFrame(rows).to_csv(output / "evaluation.csv", index=False, encoding="utf-8")
        pd.DataFrame(details).to_csv(output / "label_matches.csv", index=False, encoding="utf-8")
        intervals = [{"point": s.point, "rally": s.rally, "start_frame": s.start_frame,
                      "end_frame": s.end_frame, "duration_sec": s.duration_sec,
                      "n_labels_all": len(s.labels), "n_labels_clear": int(s.labels.confidence.eq("clear").sum())}
                     for s in segments]
        pd.DataFrame(intervals).to_csv(output / "intervals.csv", index=False, encoding="utf-8")
        scratch_path = Path(scratch)
    elapsed = time.perf_counter() - started
    inputs = [protocol] + [args.pipeline_dir / "labels" / f"point_{p:02d}.csv" for p in TRAIN_POINTS + TEST_POINTS]
    metadata = {"source_video": str(video), "source_video_bytes": video.stat().st_size,
                "source_video_mtime_ns": video.stat().st_mtime_ns,
                "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "ffmpeg_version": subprocess.run([str(args.ffmpeg), "-version"], check=True,
                                                 capture_output=True, text=True).stdout.splitlines()[0],
                "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                             "scipy": __import__("scipy").__version__, "matplotlib": matplotlib.__version__},
                "frame_sec": FRAME_SEC, "sample_rate": SAMPLE_RATE,
                "scratch_deleted": not scratch_path.exists(), "elapsed_sec": elapsed,
                "matching": "global closest-first greedy on all labels within point/rally; rounded integer frames",
                "train_points": TRAIN_POINTS, "test_points": TEST_POINTS, "n_configs": len(list(configurations()))}
    (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(output, chosen, rows, intervals, elapsed)
    print(pd.DataFrame(rows)[["scope", "n_labels_clear", "n_matched_clear_f6", "recall_clear_f6",
                              "recall_clear_f10", "precision_proxy_f6", "candidates_per_sec"]].to_string(index=False))
    print(f"Finished in {elapsed:.2f}s. Temporary WAV directory removed: {not scratch_path.exists()}")


if __name__ == "__main__":
    import sys

    if "--v2" in sys.argv:
        # Keep the original no-flag experiment and all its functions unchanged.
        sys.argv.remove("--v2")
        from detect_hit_sounds_v2 import main as main_v2

        main_v2()
    else:
        main()
