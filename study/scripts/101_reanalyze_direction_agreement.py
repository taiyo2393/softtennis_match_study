"""Reanalyze match1 direction agreement; read existing inputs without modification.

Run with the system Python 3.12 specified for this study. No video, model fitting,
or third-party packages beyond numpy/scipy/pandas/matplotlib are used.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import scipy
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "study/analysis/reanalysis_20261117"
INPUTS = {
    "main80": ROOT / "study/analysis/rear_hitter_main80/main80_event_timepoints.csv",
    "all98": ROOT / "study/analysis/match1_court_plane_directional_motion_t0p015_event_timepoints.csv",
}
TIMES = (-0.5, -0.3, -0.2, -0.1)
THRESHOLDS = (0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030)
PRIMARY_THRESHOLD = 0.015
SEED = 20261117
BOOTSTRAP_REPS = 10000
CLUSTER_GAP = 600
GAPS = (300, 600, 1200)
EXPECTED = {
    "main80": ((13, 6), (20, 14), (11, 9), (18, 13)),
    "all98": ((14, 6), (25, 16), (15, 11), (22, 15)),
}
EXPECTED_MAIN80_ASSOCIATION = (
    (((3, 3), (4, 3)), 0.497, 1.0000),
    (((9, 1), (5, 5)), 0.500, 0.1409),
    (((6, 0), (2, 3)), 0.521, 0.0606),
    (((8, 2), (3, 5)), 0.512, 0.1448),
)
D_COL = "signed_displacement_toward_actual_court_width"
DIRECTION_COL = "actual_direction_target_front_view"
EVENT_COLS = [
    "event_id", "video_id", "target_front_player_id", "target_front_camera_end",
    "impact_source_frame", DIRECTION_COL,
]
CSV_NAMES = (
    "primary_thr015", "three_way_thr015", "threshold_sensitivity",
    "threshold_free", "rally_clusters", "pair_stratified",
)
METHOD_SOURCES = {
    "binomtest": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html",
    "clopper_pearson": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html",
    "fisher_exact": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html",
    "mannwhitneyu": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directional_mask(d: np.ndarray, threshold: float) -> np.ndarray:
    """Equality at a positive threshold is directional; zero is always stay."""
    return d != 0 if threshold == 0 else np.abs(d) >= threshold


def load_and_validate(cohort: str, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = EVENT_COLS + ["timepoint_sec", "lateral_displacement_court_width", D_COL]
    if cohort == "main80":
        required += ["hitter_id"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{cohort}: missing columns: {sorted(missing)}")
    if frame[required].isna().any().any():
        raise ValueError(f"{cohort}: missing required values (no silent exclusions)")
    expected_n = 80 if cohort == "main80" else 98
    assert len(frame) == 4 * expected_n, f"{cohort}: unexpected row count"
    assert frame.event_id.nunique() == expected_n, f"{cohort}: unexpected event count"
    assert set(frame.timepoint_sec) == set(TIMES), f"{cohort}: unexpected timepoints"
    assert not frame.duplicated(["event_id", "timepoint_sec"]).any(), "Duplicate event/time"
    assert frame.groupby("event_id").size().eq(4).all(), "Incomplete event timepoints"
    for column in ["impact_source_frame", "timepoint_sec", "lateral_displacement_court_width", D_COL]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy()).all():
            raise ValueError(f"{cohort}: nonfinite {column}")
    assert (frame.impact_source_frame >= 0).all()
    assert frame.impact_source_frame.eq(np.floor(frame.impact_source_frame)).all()
    frame["impact_source_frame"] = frame.impact_source_frame.astype(int)
    assert set(frame[DIRECTION_COL]) <= {"front_left", "front_right"}
    assert set(frame.target_front_camera_end) <= {"near", "far"}
    invariants = EVENT_COLS[1:] + (["hitter_id"] if cohort == "main80" else [])
    assert frame.groupby("event_id")[invariants].nunique().eq(1).all().all(), "Event metadata changes across times"
    d = frame[D_COL].to_numpy(dtype=float)
    lateral = frame.lateral_displacement_court_width.to_numpy(dtype=float)
    # An independent orientation check; the analysis itself uses recorded d.
    v_from_court = lateral * np.where(frame.target_front_camera_end.eq("near"), 1, -1)
    v = d * np.where(frame[DIRECTION_COL].eq("front_right"), 1, -1)
    assert np.allclose(v, v_from_court, atol=1e-12, rtol=0), f"{cohort}: displacement orientation mismatch"
    frame["v_front_right_court_width"] = v
    frame["cohort"] = cohort
    for time, expected in zip(TIMES, EXPECTED[cohort]):
        values = frame.loc[frame.timepoint_sec.eq(time), D_COL].to_numpy()
        mask = directional_mask(values, PRIMARY_THRESHOLD)
        observed = (int(mask.sum()), int(((values > 0) & mask).sum()))
        assert observed == expected, (
            f"REPRODUCTION FAILED: {cohort}, t={time}: (n,k)={observed}; expected {expected}. "
            "Definitions have not been changed. No analysis output has been written."
        )
    return frame.sort_values(["impact_source_frame", "event_id", "timepoint_sec"]).reset_index(drop=True)


def add_clusters(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    events = frame[EVENT_COLS + ["cohort"]].drop_duplicates("event_id").copy()
    events = events.sort_values(["impact_source_frame", "event_id"]).reset_index(drop=True)
    gaps = events.impact_source_frame.diff()
    summary = {}
    for gap in GAPS:
        labels = (gaps.gt(gap) | gaps.isna()).cumsum().astype(int)
        events[f"cluster_id_gap{gap}"] = labels
        sizes = labels.value_counts().to_numpy()
        summary[str(gap)] = {
            "cluster_count": int(len(sizes)),
            "event_count": int(sizes.sum()),
            "size_distribution": {str(k): int(v) for k, v in sorted(Counter(sizes.tolist()).items())},
            "size_min": int(sizes.min()), "size_median": float(np.median(sizes)),
            "size_mean": float(sizes.mean()), "size_max": int(sizes.max()),
        }
    events["cluster_id"] = events[f"cluster_id_gap{CLUSTER_GAP}"]
    events["previous_impact_gap_frames"] = gaps
    events["cluster_gap_frames"] = CLUSTER_GAP
    events["cluster_is_estimated"] = True
    events["events_in_cluster"] = events.groupby("cluster_id").event_id.transform("size")
    result = frame.merge(events[["event_id", "cluster_id"]], on="event_id", validate="many_to_one")
    return result, events, summary


def exact_binomial(k: int, n: int) -> dict:
    if n == 0:
        return dict(agreement_rate=np.nan, agreement_ci95_low=np.nan,
                    agreement_ci95_high=np.nan, binom_p_two_sided=np.nan)
    test = stats.binomtest(k, n, p=0.5, alternative="two-sided")
    ci = test.proportion_ci(confidence_level=0.95, method="exact")
    return dict(agreement_rate=k / n, agreement_ci95_low=float(ci.low),
                agreement_ci95_high=float(ci.high), binom_p_two_sided=float(test.pvalue))


def majority(direction: pd.Series, prefix: str) -> dict:
    counts = direction.value_counts()
    left, right = int(counts.get("front_left", 0)), int(counts.get("front_right", 0))
    total = left + right
    winner = "tie" if left == right else ("front_left" if left > right else "front_right")
    return {f"{prefix}_actual_front_left": left, f"{prefix}_actual_front_right": right,
            f"{prefix}_majority_direction": winner,
            f"{prefix}_majority_rate": max(left, right) / total if total else np.nan}


def agreement_row(group: pd.DataFrame, threshold: float) -> dict:
    d = group[D_COL].to_numpy(dtype=float)
    mask = directional_mask(d, threshold)
    n, k = int(mask.sum()), int(((d > 0) & mask).sum())
    v = group.v_front_right_court_width.to_numpy(dtype=float)[mask]
    actual_right = group.loc[mask, DIRECTION_COL].eq("front_right").to_numpy()
    moved_right, moved_left = v > 0, v < 0
    # Rows: movement right/left; columns: actual direction right/left.
    contingency = np.array([
        [int((moved_right & actual_right).sum()), int((moved_right & ~actual_right).sum())],
        [int((moved_left & actual_right).sum()), int((moved_left & ~actual_right).sum())],
    ], dtype=np.int64)
    assert int(contingency.sum()) == n, "Decided movements must be nonzero"
    assert int(np.trace(contingency)) == k, "Direction table and signed agreement disagree"
    moved_right_count, moved_left_count = (int(value) for value in contingency.sum(axis=1))
    chance, fisher_p = np.nan, np.nan
    if n:
        p_move_right = moved_right_count / n
        p_actual_right = int(actual_right.sum()) / n
        chance = p_move_right * p_actual_right + (1 - p_move_right) * (1 - p_actual_right)
        if (contingency.sum(axis=1) > 0).all() and (contingency.sum(axis=0) > 0).all():
            fisher_p = float(stats.fisher_exact(contingency, alternative="two-sided").pvalue)
    row = dict(cohort=group.cohort.iloc[0], timepoint_sec=float(group.timepoint_sec.iloc[0]),
               threshold_court_width=threshold, total_events=len(group), n=n, k=k,
               disagreement_count=n - k, stay_count=len(group) - n, decision_rate=n / len(group),
               moved_right_actual_right=int(contingency[0, 0]),
               moved_right_actual_left=int(contingency[0, 1]),
               moved_left_actual_right=int(contingency[1, 0]),
               moved_left_actual_left=int(contingency[1, 1]),
               moved_right_count=moved_right_count, moved_left_count=moved_left_count,
               chance_agreement_independence=chance, fisher_p_two_sided=fisher_p)
    row.update(exact_binomial(k, n))
    row.update(majority(group.loc[mask, DIRECTION_COL], "decided"))
    row.update(majority(group[DIRECTION_COL], "all"))
    row["agreement_minus_decided_majority"] = row["agreement_rate"] - row["decided_majority_rate"]
    row["agreement_minus_chance"] = row["agreement_rate"] - chance
    return row


def holm_adjust(pvalues: list[float]) -> np.ndarray:
    """Holm step-down adjustment, including monotonicity, for a four-time family."""
    p = np.asarray(pvalues, dtype=float)
    # Undefined pair-level tests still occupy a place in the four-time family.
    order = np.argsort(np.where(np.isfinite(p), p, 1.0), kind="stable")
    adjusted = np.full(len(p), np.nan)
    ordered = np.minimum(1, np.maximum.accumulate(np.where(np.isfinite(p[order]), p[order], 1) * np.arange(len(p), 0, -1)))
    adjusted[order] = ordered
    adjusted[~np.isfinite(p)] = np.nan
    return adjusted


def add_holm(rows: list[dict]) -> None:
    for test in ("binom", "fisher"):
        adjusted = holm_adjust([row[f"{test}_p_two_sided"] for row in rows])
        for row, pvalue in zip(rows, adjusted):
            row[f"{test}_p_holm_4timepoints"] = float(pvalue)


def assert_main80_association(frame: pd.DataFrame) -> None:
    """Reproduce the reviewer's tables and rounded independence/Fisher values."""
    for time, (expected_table, expected_chance, expected_fisher) in zip(TIMES, EXPECTED_MAIN80_ASSOCIATION):
        row = agreement_row(frame.loc[frame.timepoint_sec.eq(time)], PRIMARY_THRESHOLD)
        observed_table = (
            (row["moved_right_actual_right"], row["moved_right_actual_left"]),
            (row["moved_left_actual_right"], row["moved_left_actual_left"]),
        )
        context = f"REPRODUCTION FAILED: main80, t={time}, thr={PRIMARY_THRESHOLD}"
        assert observed_table == expected_table, f"{context}: table={observed_table}; expected {expected_table}"
        assert round(row["chance_agreement_independence"], 3) == expected_chance, (
            f"{context}: chance={row['chance_agreement_independence']}; expected {expected_chance:.3f}"
        )
        assert round(row["fisher_p_two_sided"], 4) == expected_fisher, (
            f"{context}: Fisher p={row['fisher_p_two_sided']}; expected {expected_fisher:.4f}"
        )


def auc_probability(v: np.ndarray, is_right: np.ndarray) -> float:
    if not is_right.any() or is_right.all():
        return float("nan")
    difference = v[is_right, None] - v[~is_right][None, :]
    return float(np.mean((difference > 0) + 0.5 * (difference == 0)))


def bootstrap_intervals(group: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Resample G whole clusters G times; retain all events and duplicated draws.

    Cluster multiplicities weight both events and right/left pairs. The AUC kernel
    is reduced to a G x G matrix, avoiding 10000 repeated rank computations.
    Pooled statistics remain event-weighted, not equal-weighted cluster averages.
    """
    _, code = np.unique(group.cluster_id.to_numpy(), return_inverse=True)
    g = int(code.max() + 1)
    d = group[D_COL].to_numpy(dtype=float)
    v = group.v_front_right_court_width.to_numpy(dtype=float)
    right = group[DIRECTION_COL].eq("front_right").to_numpy()
    size = np.bincount(code, minlength=g)
    sum_d = np.bincount(code, weights=d, minlength=g)
    n_right = np.bincount(code[right], minlength=g)
    n_left = np.bincount(code[~right], minlength=g)
    kernel = np.zeros((g, g))
    if right.any() and (~right).any():
        diff = v[right, None] - v[~right][None, :]
        pair_scores = (diff > 0).astype(float) + 0.5 * (diff == 0)
        np.add.at(kernel, (code[right, None], code[~right][None, :]), pair_scores)
    # Explicit uniform cluster draws with replacement, not observation resampling.
    draws = rng.integers(0, g, size=(BOOTSTRAP_REPS, g))
    weights = np.zeros((BOOTSTRAP_REPS, g), dtype=np.int64)
    np.add.at(weights, (np.arange(BOOTSTRAP_REPS)[:, None], draws), 1)
    mean_samples = (weights @ sum_d) / (weights @ size)
    denominator = (weights @ n_right) * (weights @ n_left)
    valid = denominator > 0
    auc_samples = np.einsum("bi,ij,bj->b", weights[valid], kernel, weights[valid], optimize=True) / denominator[valid]
    mean_ci = np.quantile(mean_samples, [0.025, 0.975], method="linear")
    auc_ci = np.quantile(auc_samples, [0.025, 0.975], method="linear") if valid.any() else [np.nan, np.nan]
    return dict(auc_ci95_low=float(auc_ci[0]), auc_ci95_high=float(auc_ci[1]),
                d_mean_ci95_low=float(mean_ci[0]), d_mean_ci95_high=float(mean_ci[1]),
                bootstrap_reps=BOOTSTRAP_REPS, bootstrap_auc_valid_reps=int(valid.sum()),
                bootstrap_auc_invalid_reps=int((~valid).sum()),
                bootstrap_mean_valid_reps=BOOTSTRAP_REPS, estimated_clusters=g,
                cluster_gap_frames=CLUSTER_GAP, cluster_is_estimated=True)


def threshold_free_row(group: pd.DataFrame, rng: np.random.Generator) -> dict:
    d = group[D_COL].to_numpy(dtype=float)
    v = group.v_front_right_court_width.to_numpy(dtype=float)
    right = group[DIRECTION_COL].eq("front_right").to_numpy()
    auc = auc_probability(v, right)
    if right.any() and (~right).any():
        mw = stats.mannwhitneyu(v[right], v[~right], alternative="two-sided",
                                method="asymptotic", use_continuity=True)
        assert np.isclose(auc, mw.statistic / (int(right.sum()) * int((~right).sum())))
        u, p = float(mw.statistic), float(mw.pvalue)
    else:
        u, p = np.nan, np.nan
    row = dict(cohort=group.cohort.iloc[0], timepoint_sec=float(group.timepoint_sec.iloc[0]),
               total_events=len(group), actual_front_right=int(right.sum()),
               actual_front_left=int((~right).sum()), auc=auc, mannwhitney_u=u,
               mannwhitney_p_two_sided=p, mannwhitney_method="asymptotic_tie_corrected_continuity",
               d_median_court_width=float(np.median(d)), d_mean_court_width=float(np.mean(d)))
    row.update(bootstrap_intervals(group, rng))
    return row


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(row) + " |" for row in rows])


def pct(value: float) -> str:
    return "算出不可" if not np.isfinite(value) else f"{100 * value:.1f}%"


def num(value: float, digits: int = 4) -> str:
    return "算出不可" if not np.isfinite(value) else f"{value:.{digits}f}"


def ci(row: dict, low: str, high: str, percent: bool = False) -> str:
    f = pct if percent else num
    return f"[{f(row[low])}, {f(row[high])}]"


def report_range(rows: list[dict], column: str, percent: bool = False) -> str:
    values = [row[column] for row in rows if np.isfinite(row[column])]
    if not values:
        return "算出不可"
    if percent:
        return f"{100 * min(values):.1f}–{100 * max(values):.1f}%"
    return f"{min(values):.4f}–{max(values):.4f}"


def report_holm_summary(rows: list[dict], column: str, test_name: str) -> str:
    defined = [row for row in rows if np.isfinite(row[column])]
    significant = [row for row in defined if row[column] < 0.05]
    label = f"{len(rows)}時点のHolm補正後の{test_name}"
    if significant:
        times = "、".join(
            f"{row['timepoint_sec']:.1f}秒（Holm p={num(row[column])}）"
            for row in significant
        )
        result = f"{label}は{times}で有意だった（有意水準5%）。"
    elif defined:
        result = f"{label}は算出可能な全時点で有意でなかった（有意水準5%）。"
    else:
        result = f"{label}は全時点で算出できなかった。"
    if len(defined) < len(rows):
        result += f"{len(rows) - len(defined)}時点は検定値が未定義だった。"
    return result


def make_report(tables: dict[str, pd.DataFrame], cluster_summaries: dict) -> str:
    primary = tables["primary_thr015"].query("cohort == 'main80'").sort_values("timepoint_sec").to_dict("records")
    three = tables["three_way_thr015"].query("cohort == 'main80'").sort_values("timepoint_sec").to_dict("records")
    sensitivity = tables["threshold_sensitivity"].query("cohort == 'main80'").sort_values(["timepoint_sec", "threshold_court_width"]).to_dict("records")
    free = tables["threshold_free"].query("cohort == 'main80'").sort_values("timepoint_sec").to_dict("records")
    all_primary = tables["primary_thr015"].query("cohort == 'all98'").sort_values("timepoint_sec").to_dict("records")
    all_free = tables["threshold_free"].query("cohort == 'all98'").sort_values("timepoint_sec").to_dict("records")
    main_n, all_n = primary[0]["total_events"], all_primary[0]["total_events"]
    threshold = primary[0]["threshold_court_width"]
    times = "/".join(f"{row['timepoint_sec']:.1f}" for row in primary)
    agreement_range = report_range(primary, "agreement_rate", percent=True)
    decision_range = report_range(primary, "decision_rate", percent=True)
    binom_summary = report_holm_summary(primary, "binom_p_holm_4timepoints", "二項検定")
    fisher_summary = report_holm_summary(primary, "fisher_p_holm_4timepoints", "Fisherの正確検定")
    mw_summary = f"AUCのMann–Whitney U検定の未補正p値は{report_range(free, 'mannwhitney_p_two_sided')}だった。"
    # Describe the largest observed agreement without hard-coding its time or counts.
    example = max(primary, key=lambda row: row["agreement_rate"])
    example_three = next(row for row in three if row["timepoint_sec"] == example["timepoint_sec"])
    comparison = next(row for row in primary if row["timepoint_sec"] == -0.3)
    sensitivity_hits = [
        row for row in sensitivity
        if np.isfinite(row["binom_p_two_sided"]) and row["binom_p_two_sided"] < 0.05
    ]
    sensitivity_list = "、".join(
        f"（{row['timepoint_sec']:.1f}秒, thr={row['threshold_court_width']:.3f}, {row['k']}/{row['n']}）"
        for row in sensitivity_hits
    ) or "該当なし"
    all_majority = all_primary[0]
    majority_count = max(all_majority["all_actual_front_left"], all_majority["all_actual_front_right"])
    majority_label = (
        "左右同数" if all_majority["all_majority_direction"] == "tie"
        else all_majority["all_majority_direction"]
    )
    sections = [
        "# match1 前衛の打球前の横移動と相手後衛の打球方向：統計的再解析",
        f"実行日と環境・入力SHA-256は `metadata.json` に記録。主解析は main80（{main_n}場面、{sum(row['total_events'] for row in primary)}行）、感度解析は all98（{all_n}場面、{sum(row['total_events'] for row in all_primary)}行）。各場面の {times} 秒を使用し、t = 0 秒は含めない。",
        "## 結果の解釈",
        f"main80 の一致率は{agreement_range}だった。{binom_summary}{fisher_summary}{mw_summary}有意な結果でも、この1試合の探索的解析から一般的な関連が確認されたとは扱わない。有意でないことも、関連が存在しないことの証明ではない。",
        f"一致率は**判定ありの場面が分母の条件付きの値**であり、main80の判定率は{decision_range}だった。表Aで一致率が最も高い{example['timepoint_sec']:.1f}秒の{pct(example['agreement_rate'])}は{example['k']}/{example['n']}で、全{main_n}場面のうち一致は{example_three['agreement_count']}/{example_three['total_events']}＝{100 * example_three['agreement_fraction_all']:.2f}%。一致率の95%CIは{ci(example, 'agreement_ci95_low', 'agreement_ci95_high', True)}である。",
        "## 方法と定義",
        f"`d = signed_displacement_toward_actual_court_width`。正の閾値では `abs(d) >= thr` を判定あり、そのうち `d > 0` を一致、`d < 0` を不一致とした。`abs(d) < thr` は判定なし（stay）。thr = 0のみ `d != 0` を判定ありとする。既存の方向ラベルを使って数え直すのではなく、記録されたdから再計算した。thr = {threshold:.3f} の指定された{len(primary) + len(all_primary)}組の（判定数, 一致数）はすべてassertで再現確認した。変位の単位はコート幅に対する割合。",
        "前衛から見た右向き変位vは、実際がfront_rightならv=d、front_leftならv=−d。AUCは実際の方向が右の場面と左の場面を全組合せで比較し、P(v_right > v_left) + 0.5 P(同値) とした。AUC=0.5が方向による順位差のない基準。これは学習・交差検証を行った予測モデルの性能値ではない。",
        f"一致率の95%CIはClopper–Pearson法。二項検定はp=0.5を帰無仮説とする両側検定で、{len(primary)}時点の多重比較は**Holmで補正**した（main80とall98はcohortごとに別の検定族）。多数派基準は、同じ判定ありの場面で**いつも多い方を答える場合の参照値**である。全場面での多数派も別に示す。多数派は同じデータから得た記述的な参照値であり、その基準に対する有意差検定や汎化性能の評価はしていない。",
        f"前衛の動きと打球方向の関連を調べる偶然の水準は、**独立のときの期待一致率**であり、多数派基準とは区別する。判定ありの場面における右移動の割合をp_move_right、実際の右打球の割合をp_actual_rightとして、`p_move_right * p_actual_right + (1 - p_move_right) * (1 - p_actual_right)`を計算した。右移動は `v_front_right_court_width > 0`。移動方向（右・左）×実際の打球方向（右・左）の2×2表に両側Fisherの正確検定を適用し、二項検定とは別にcohortごとの{len(primary)}時点内でHolm補正した。行または列の和が0のときFisherのp値は未定義とする。",
        f"AUCとdの平均の95%CIは、後述の推定ラリーまとまりを等確率で復元抽出するクラスタ・ブートストラップ（{BOOTSTRAP_REPS}回、seed={SEED}）の2.5/97.5百分位点。各反復で元と同数のまとまりを抽出し、その全場面と重複を保持して、場面を単位としたAUC・平均を再計算した。両方向がそろわないAUC反復は除外し、その数をCSVに記録。AUCのp値は、同値補正・連続性補正を行う両側Mann–Whitney U検定（漸近法）。多重比較の補正はしていない。",
        "**二項検定・Fisherの正確検定・Clopper–Pearson CI・Mann–Whitneyのp値は場面間の独立性を仮定する。Holm補正は同じラリー内の依存を補正しない。** クラスタ・ブートストラップで依存を考慮したのはDのAUCと平均のCIであり、Aのp値・CIをラリー単位で補正したものではない。DのCIも時点ごとの95%CIで、同時信頼区間ではない。",
        f"## 表A：主検定（main80、thr={threshold:.3f}）",
        md_table(["時点(s)", "一致k / 判定n", "一致率 [95%CI]", "偶然の水準", "二項p", "Holm p", "Fisher p / Holm p", "判定ありの多数派", f"全{main_n}場面の多数派"], [
            [f"{r['timepoint_sec']:.1f}", f"{r['k']}/{r['n']}", pct(r['agreement_rate']) + " " + ci(r, 'agreement_ci95_low', 'agreement_ci95_high', True),
             pct(r['chance_agreement_independence']),
             num(r['binom_p_two_sided']), num(r['binom_p_holm_4timepoints']),
             num(r['fisher_p_two_sided']) + ' / ' + num(r['fisher_p_holm_4timepoints']),
             f"{pct(r['decided_majority_rate'])} ({r['decided_majority_direction']}; 左{r['decided_actual_front_left']}/右{r['decided_actual_front_right']})",
             f"{pct(r['all_majority_rate'])} (左{r['all_actual_front_left']}/右{r['all_actual_front_right']})"] for r in primary]),
        f"{comparison['timepoint_sec']:.1f}秒では右移動{comparison['moved_right_count']}場面・左移動{comparison['moved_left_count']}場面で、一致率{pct(comparison['agreement_rate'])}に対し独立のときの期待一致率は{pct(comparison['chance_agreement_independence'])}、差は{100 * comparison['agreement_minus_chance']:+.1f}ポイントだった。Fisher p={num(comparison['fisher_p_two_sided'])}、Holm p={num(comparison['fisher_p_holm_4timepoints'])}。閾値と時点を選んで高い一致率だけを強調しない。",
        f"## 表B：3値集計（main80、thr={threshold:.3f}、分母はすべて{main_n}）",
        md_table(["時点(s)", "一致 件数(割合)", "不一致 件数(割合)", "判定なし 件数(割合)", "判定率"], [
            [f"{r['timepoint_sec']:.1f}", f"{r['agreement_count']} ({pct(r['agreement_fraction_all'])})",
             f"{r['disagreement_count']} ({pct(r['disagreement_fraction_all'])})", f"{r['stay_count']} ({pct(r['stay_fraction_all'])})", pct(r['decision_rate'])] for r in three]),
        "## 表C：閾値の感度（main80）",
        "各p値は補正なしの記述用。閾値探索を含む多重比較を補正した検定結果ではなく、事後的に最良の閾値を選ぶ根拠にはしない。nは判定数、kは一致数。",
        md_table(["時点(s)", "thr", "n", "k", "一致率", "95%CI", "二項p（未補正）"], [
            [f"{r['timepoint_sec']:.1f}", f"{r['threshold_court_width']:.3f}", str(r['n']), str(r['k']), pct(r['agreement_rate']),
             ci(r, 'agreement_ci95_low', 'agreement_ci95_high', True), num(r['binom_p_two_sided'])] for r in sensitivity]),
        f"表Cで補正なしの二項p < 0.05になった（時点, 閾値, k/n）の一覧：{sensitivity_list}。閾値を後から選んだ結果で、確認された関連ではない。別の試合で確かめる場合は、時点と閾値を先に固定する。",
        f"## 表D：閾値を使わない指標（main80、全{main_n}場面）",
        md_table(["時点(s)", "AUC [クラスタ95%CI]", "MW p（未補正）", "d中央値", "d平均 [クラスタ95%CI]"], [
            [f"{r['timepoint_sec']:.1f}", num(r['auc']) + " " + ci(r, 'auc_ci95_low', 'auc_ci95_high'), num(r['mannwhitney_p_two_sided']),
             num(r['d_median_court_width'], 6), num(r['d_mean_court_width'], 6) + " " + ci(r, 'd_mean_ci95_low', 'd_mean_ci95_high')] for r in free]),
        "dは正ほど実際の打球方向へ動いたことを意味する。表Dの平均・中央値は判定なしの場面も含む。",
        "## 推定ラリーのまとまり（E）",
        f"**ラリーのまとまりは推定であり、人手のラリーIDではない。** 各cohort内で場面をimpact_source_frame昇順（同値はevent_id順）に並べ、直前との差が{CLUSTER_GAP}フレーム（約{CLUSTER_GAP / 60:g}秒）を**超えたら**新しいまとまりとした。{CLUSTER_GAP}フレームちょうどは分割しない。除外場面のあるmain80とall98では、残された場面からcohortごとに別々に推定している。短いラリー間隔の結合、長い打球間隔の分割、抽出対象外の打球の影響がありうる。",
        md_table(["cohort"] + [f"{gap}フレーム" + ("（使用）" if gap == CLUSTER_GAP else "") for gap in GAPS], [
            [cohort] + [str(cluster_summaries[cohort][str(g)]['cluster_count']) for g in GAPS] for cohort in INPUTS]),
    ]
    for cohort in INPUTS:
        summary = cluster_summaries[cohort][str(CLUSTER_GAP)]
        distribution = "、".join(f"{size}場面×{count}まとまり" for size, count in summary['size_distribution'].items())
        sections.append(f"{cohort} の{CLUSTER_GAP}フレーム基準：{summary['cluster_count']}まとまり。場面数分布は{distribution}。最小{summary['size_min']}、中央値{summary['size_median']:.1f}、平均{summary['size_mean']:.2f}、最大{summary['size_max']}場面。")
    sections += [
        f"## {all_n}場面の感度解析",
        "all98はmain80を含む同一試合の拡張集合であり、独立した追試ではない。A〜Eを同じ定義で計算し、全値は各CSVのcohort=all98に保存した。以下にAとDを要約する。",
        md_table(["時点(s)", "k/n（一致率）", "一致率95%CI", "判定率", "多数派（判定あり）", "偶然の水準", "二項p / Holm p", "Fisher p / Holm p", "AUC [クラスタ95%CI]", "MW p（未補正）"], [
            [f"{a['timepoint_sec']:.1f}", f"{a['k']}/{a['n']} ({pct(a['agreement_rate'])})", ci(a, 'agreement_ci95_low', 'agreement_ci95_high', True),
             pct(a['decision_rate']), pct(a['decided_majority_rate']), pct(a['chance_agreement_independence']),
             num(a['binom_p_two_sided']) + ' / ' + num(a['binom_p_holm_4timepoints']),
             num(a['fisher_p_two_sided']) + ' / ' + num(a['fisher_p_holm_4timepoints']),
             num(d['auc']) + ' ' + ci(d, 'auc_ci95_low', 'auc_ci95_high'), num(d['mannwhitney_p_two_sided'])] for a, d in zip(all_primary, all_free)]),
        f"全{all_n}場面の多数派は{majority_label}（{majority_count}/{all_n}={pct(all_majority['all_majority_rate'])}、左{all_majority['all_actual_front_left']}/右{all_majority['all_actual_front_right']}）。"
        + report_holm_summary(all_primary, 'binom_p_holm_4timepoints', '二項検定')
        + report_holm_summary(all_primary, 'fisher_p_holm_4timepoints', 'Fisherの正確検定')
        + f"Mann–Whitney U検定の未補正p値は{report_range(all_free, 'mannwhitney_p_two_sided')}。{main_n}場面の結果を強める独立の証拠とは扱わない。",
        "## 組合せ別の記述（F、main80のみ）",
        f"`pair_stratified.csv` に(target_front_player_id, hitter_id)別・時点別のAとDの値を保存した。Aの二項検定とFisherの正確検定のHolm列は、検定ごと・組ごとの{len(primary)}時点内で補正しているが、組別解析全体に対する多重比較補正ではない。組の間の差の検定は行っていない。少数場面の記述用であり、組や選手の優劣を推論しない。組内ブートストラップでもcohort全体で作ったcluster_idを保持し、その組の場面があるまとまりを復元抽出する。",
        "## 研究上の限界と発表で述べられる範囲",
        "これは**1試合の観察**であり、前衛が打球方向を「予測した」ことや因果関係は示さない。時点、閾値、解析対象の選択を含む探索的な再解析で、事前登録された確認的研究ではない。左右方向の偏り、対戦組合せ、抽出条件、計測誤差などの影響を分離していない。閾値で判定ありを選ぶため、条件付き一致率を全場面の性能へ一般化できない。推定ラリーを用いたCIでも、1試合・少数選手から他試合や他選手への一般化はできない。",
        f"中間発表の補足としては、「{main_n}場面中、移動量の閾値{threshold:.3f}以上の場面は各時点{min(row['n'] for row in primary)}–{max(row['n'] for row in primary)}場面で、その一致率は{agreement_range}だった。独立のときの期待一致率は{report_range(primary, 'chance_agreement_independence', True)}である。"
        + binom_summary + fisher_summary + mw_summary
        + "これは1試合の探索的な結果であり、予測性能や一般的な関連の確認を意味しない」と述べる。",
        "## 再実行と出力",
        "```powershell\n& 'C:\\Users\\ta239\\AppData\\Local\\Programs\\Python\\Python312\\python.exe' study/scripts/101_reanalyze_direction_agreement.py\n```",
        f"{len(CSV_NAMES)}個のCSV（UTF-8 BOM付き）、metadata.json、figure_agreement_ci.png（Yu Gothic、140dpi）、本報告を `{OUT.relative_to(ROOT).as_posix()}/` へ生成する。比率はCSVでは0–1、図・本文の一致率は百分率。未定義の数値はCSVで空欄。入力ファイルは読み取り専用で扱い、終了前にSHA-256の不変も確認する。",
        "実装の関数仕様：[SciPy binomtest](" + METHOD_SOURCES['binomtest'] + ")、[Clopper–Pearson CI](" + METHOD_SOURCES['clopper_pearson'] + ")、[Fisherの正確検定](" + METHOD_SOURCES['fisher_exact'] + ")、[Mann–Whitney U](" + METHOD_SOURCES['mannwhitneyu'] + ")。",
    ]
    return "\n\n".join(sections) + "\n"


def make_figure(primary: pd.DataFrame, free: pd.DataFrame) -> str:
    font_path = font_manager.findfont(font_manager.FontProperties(family="Yu Gothic"), fallback_to_default=False)
    plt.rcParams.update({"font.family": "Yu Gothic", "font.size": 10, "axes.unicode_minus": False})
    a = primary.query("cohort == 'main80'").sort_values("timepoint_sec")
    d = free.query("cohort == 'main80'").sort_values("timepoint_sec")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    y = np.arange(len(TIMES))
    for ax, table, value, lo, hi, scale, title in [
        (axes[0], a, "agreement_rate", "agreement_ci95_low", "agreement_ci95_high", 100, "一致率と95%信頼区間\nClopper–Pearson、thr = 0.015"),
        (axes[1], d, "auc", "auc_ci95_low", "auc_ci95_high", 1, "AUCと95%信頼区間\n推定ラリー単位のブートストラップ"),
    ]:
        center = table[value].to_numpy() * scale
        low, high = table[lo].to_numpy() * scale, table[hi].to_numpy() * scale
        # Draw interval segments directly; percentile CI need not contain the point estimate.
        ax.hlines(y, low, high, color="#245b78", linewidth=2)
        ax.scatter(center, y, color="#245b78", zorder=3, s=35)
        ax.vlines(low, y - .06, y + .06, color="#245b78")
        ax.vlines(high, y - .06, y + .06, color="#245b78")
        ax.axvline(0.5 * scale, color="#777777", linestyle="--", linewidth=1)
        ax.set(xlim=(0, scale), title=title, yticks=y, yticklabels=[f"{t:.1f}秒" for t in TIMES])
        ax.grid(axis="x", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_xlabel("一致率（%）、破線は50%")
    axes[1].set_xlabel("AUC、破線は0.5")
    axes[0].set_ylabel("相手後衛の打球前の時点")
    axes[0].set_ylim(len(TIMES) - .5, -.5)  # inverted; leaves room for the k/n labels
    for i, row in enumerate(a.to_dict("records")):
        axes[0].annotate(f"{row['k']}/{row['n']}（判定率{100 * row['decision_rate']:.1f}%）", (.02, i + .29),
                         xycoords=("axes fraction", "data"), fontsize=9)
    fig.suptitle("match1 主解析80場面：前衛の横移動と相手後衛の打球方向", fontsize=13)
    fig.text(.5, .025, "一致率は判定ありの場面が分母。AUCは全80場面。区間は時点ごと。1試合の探索的解析。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94), w_pad=2)
    fig.savefig(OUT / "figure_agreement_ci.png", dpi=140, facecolor="white")
    plt.close(fig)
    return font_path


def main() -> int:
    if sys.flags.optimize:
        raise RuntimeError("Run without -O: reproduction assertions must remain enabled")
    input_hashes = {cohort: sha256(path) for cohort, path in INPUTS.items()}
    frames = {cohort: load_and_validate(cohort, path) for cohort, path in INPUTS.items()}
    # Confirm the sensitivity cohort really contains identical main80 measurements.
    shared_cols = EVENT_COLS + ["timepoint_sec", "lateral_displacement_court_width", D_COL]
    main_shared = frames['main80'][shared_cols].sort_values(["event_id", "timepoint_sec"]).reset_index(drop=True)
    all_shared = frames['all98'].loc[frames['all98'].event_id.isin(frames['main80'].event_id), shared_cols].sort_values(["event_id", "timepoint_sec"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(main_shared, all_shared, check_dtype=False)
    assert_main80_association(frames['main80'])
    # All reproduction assertions run before any output directory/file creation.
    print("PASS: all eight required (n,k) assertions and four main80 direction tables/chance/Fisher assertions; inputs validated.", flush=True)
    rows = {name: [] for name in CSV_NAMES}
    cluster_summaries = {}
    pair_groups = list(frames['main80'].groupby(["target_front_player_id", "hitter_id"], sort=True))
    streams = iter(np.random.SeedSequence(SEED).spawn((len(INPUTS) + len(pair_groups)) * len(TIMES)))
    bootstrap_jobs = []
    for cohort, raw in frames.items():
        frame, events, cluster_summary = add_clusters(raw)
        frames[cohort] = frame
        rows['rally_clusters'].extend(events.to_dict('records'))
        cluster_summaries[cohort] = cluster_summary
        primary_rows = []
        for time in TIMES:
            group = frame.loc[frame.timepoint_sec.eq(time)]
            agreement = agreement_row(group, PRIMARY_THRESHOLD)
            primary_rows.append(agreement)
            rows['three_way_thr015'].append(dict(
                cohort=cohort, timepoint_sec=time, threshold_court_width=PRIMARY_THRESHOLD,
                total_events=len(group), agreement_count=agreement['k'],
                disagreement_count=agreement['n'] - agreement['k'], stay_count=agreement['stay_count'],
                agreement_fraction_all=agreement['k'] / len(group),
                disagreement_fraction_all=(agreement['n'] - agreement['k']) / len(group),
                stay_fraction_all=agreement['stay_count'] / len(group), decision_rate=agreement['decision_rate']))
            for threshold in THRESHOLDS:
                rows['threshold_sensitivity'].append(agreement_row(group, threshold))
            seed = next(streams)
            rows['threshold_free'].append(threshold_free_row(group, np.random.default_rng(seed)))
            bootstrap_jobs.append(dict(cohort=cohort, timepoint_sec=time, pair=None, spawn_key=list(seed.spawn_key)))
        add_holm(primary_rows)
        rows['primary_thr015'].extend(primary_rows)
        print(f"Calculated {cohort}: {len(events)} events, {cluster_summary[str(CLUSTER_GAP)]['cluster_count']} estimated clusters.", flush=True)
    for (front, hitter), group in frames['main80'].groupby(["target_front_player_id", "hitter_id"], sort=True):
        pair_rows = []
        for time in TIMES:
            at_time = group.loc[group.timepoint_sec.eq(time)]
            row = agreement_row(at_time, PRIMARY_THRESHOLD)
            seed = next(streams)
            row.update(threshold_free_row(at_time, np.random.default_rng(seed)))
            row.update(target_front_player_id=front, hitter_id=hitter, analysis_role="descriptive_only")
            pair_rows.append(row)
            bootstrap_jobs.append(dict(cohort='main80', timepoint_sec=time, pair=[front, hitter], spawn_key=list(seed.spawn_key)))
        add_holm(pair_rows)
        rows['pair_stratified'].extend(pair_rows)
    tables = {name: pd.DataFrame(value) for name, value in rows.items()}
    assert all('cohort' in table for table in tables.values())
    assert len(tables['primary_thr015']) == len(tables['three_way_thr015']) == len(tables['threshold_free']) == 8
    assert len(tables['threshold_sensitivity']) == 56
    assert len(tables['rally_clusters']) == 178
    # Resolve the explicitly requested font before writing any results.
    font_manager.findfont(font_manager.FontProperties(family="Yu Gothic"), fallback_to_default=False)
    assert input_hashes == {cohort: sha256(path) for cohort, path in INPUTS.items()}, "Input changed during analysis"
    OUT.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    font_path = make_figure(tables['primary_thr015'], tables['threshold_free'])
    (OUT / '再解析_結果.md').write_text(make_report(tables, cluster_summaries), encoding='utf-8')
    metadata = {
        "analysis_id": "match1_direction_agreement_reanalysis_20261117",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve().relative_to(ROOT)), "script_sha256": sha256(Path(__file__)),
        "python": {"executable": sys.executable, "version": platform.python_version(), "platform": platform.platform()},
        "library_versions": {module.__name__: module.__version__ for module in (np, scipy, pd, matplotlib)},
        "inputs": {cohort: {"path": str(path.relative_to(ROOT)), "sha256": input_hashes[cohort],
                            "rows": len(frames[cohort]), "events": int(frames[cohort].event_id.nunique()),
                            "encoding": "utf-8-sig"} for cohort, path in INPUTS.items()},
        "input_hashes_unchanged_after_analysis": True,
        "definitions": {
            "d": D_COL + "; positive means toward the actual shot direction; unit=court width fraction",
            "decision": "abs(d) >= threshold; at threshold=0 only, d != 0",
            "agreement": "decision and d > 0", "disagreement": "decision and d < 0",
            "stay": "abs(d) < threshold; at threshold=0 only, d == 0",
            "v": "d for actual front_right; -d for actual front_left",
            "auc": "P(v_right > v_left) + 0.5*P(v_right == v_left); all events, no movement threshold",
            "majority_baseline": "max(actual front_left, actual front_right)/events; separately within decided and all events; descriptive, selected in-sample",
            "chance_agreement_independence": "Expected agreement under independence of movement and actual direction, using decided-event marginals: p_move_right*p_actual_right + (1-p_move_right)*(1-p_actual_right); movement right iff v > 0; undefined if no decided events",
            "agreement_ci": "two-sided 95% Clopper-Pearson exact; assumes independent Bernoulli events",
            "binomial_test": "two-sided exact binomial, null p=0.5; assumes independent Bernoulli events",
            "fisher_exact": "scipy.stats.fisher_exact, alternative=two-sided, on decided-event 2x2 counts; rows=movement right(v>0)/left(v<0), columns=actual front_right/front_left; tests independence conditional on marginals and assumes independent events; undefined if any row or column sum is zero",
            "holm": "applied separately to binomial and Fisher tests; four timepoints per cohort; pair rows separately within each pair; undefined p retains its family slot",
            "mannwhitney": "two-sided asymptotic; ties corrected; continuity correction; unadjusted; event independence assumed",
            "interpretation": "Single-match exploratory observation; does not establish anticipation, prediction ability, causality, or generalization",
        },
        "timepoints_sec": TIMES, "thresholds_court_width": THRESHOLDS,
        "primary_threshold_court_width": PRIMARY_THRESHOLD,
        "reproduction_assertions": {cohort: [{"timepoint_sec": t, "n": counts[0], "k": counts[1], "passed": True}
                                             for t, counts in zip(TIMES, EXPECTED[cohort])] for cohort in INPUTS},
        "main80_association_reproduction_assertions": [
            {"timepoint_sec": t, "threshold_court_width": PRIMARY_THRESHOLD, "table": table,
             "chance_rounded_3dp": chance, "fisher_p_rounded_4dp": fisher, "passed": True}
            for t, (table, chance, fisher) in zip(TIMES, EXPECTED_MAIN80_ASSOCIATION)
        ],
        "bootstrap": {
            "seed": SEED, "replicates": BOOTSTRAP_REPS, "rng": "numpy.default_rng / PCG64",
            "streams": "SeedSequence(seed).spawn; cohort insertion order, then lexicographically sorted pairs; times chronological",
            "jobs": bootstrap_jobs, "confidence_level": .95, "interval": "percentile, numpy.quantile(method='linear')",
            "unit": "estimated rally cluster; G draws from G clusters, equal probability, replacement; keep all events and duplicates",
            "estimand": "pooled event-weighted AUC and d mean; not equal-weighted cluster statistics",
            "undefined_auc": "omit one-direction replicates without redrawing; valid/invalid counts in CSV; mean uses all replicates",
            "pair_policy": "retain cohort-level cluster labels; resample clusters represented within the pair",
        },
        "clusters": {"estimated_not_manual_rally_ids": True, "gap_frames": CLUSTER_GAP,
                     "reference_gaps_frames": GAPS, "approximate_fps_for_time_interpretation": 60,
                     "definition": "within each cohort independently; unique events ordered by impact frame then event_id; new cluster iff gap > cutoff",
                     "summaries": cluster_summaries},
        "figure": {"font_family": "Yu Gothic", "font_file": font_path, "dpi": 140},
        "csv_encoding": "utf-8-sig", "csv_missing_numeric": "empty field; statistic undefined, not zero",
        "method_documentation": METHOD_SOURCES,
        "outputs": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(OUT.iterdir())
                    if p.is_file() and p.name in [*(f'{n}.csv' for n in CSV_NAMES), 'figure_agreement_ci.png', '再解析_結果.md']},
    }
    (OUT / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    assert input_hashes == {cohort: sha256(path) for cohort, path in INPUTS.items()}, "Input changed during output"
    print("\nMain80 primary results (CI=Clopper-Pearson; Holm=4 timepoints):")
    for a, d in zip(rows['primary_thr015'][:4], rows['threshold_free'][:4]):
        print(f"t={a['timepoint_sec']:.1f}: k/n={a['k']}/{a['n']}, agreement={pct(a['agreement_rate'])} "
              f"CI={ci(a, 'agreement_ci95_low', 'agreement_ci95_high', True)}, "
              f"binom p={a['binom_p_two_sided']:.6f}, Holm p={a['binom_p_holm_4timepoints']:.6f}; "
              f"AUC={d['auc']:.6f}, cluster CI={ci(d, 'auc_ci95_low', 'auc_ci95_high')}, "
              f"MW p={d['mannwhitney_p_two_sided']:.6f}")
    print(f"Wrote 9 outputs to {OUT}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
