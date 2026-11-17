"""Link match1 main80 front-player motions to observed point outcomes.

Run: study\\.venv\\Scripts\\python.exe -B study/scripts/102_link_front_motion_to_point_outcome.py
Inputs are read-only. No video is opened, and no model is fitted.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "study/analysis/front_motion_outcome_20261117"
INPUTS = {
    "main80": ROOT / "study/analysis/rear_hitter_main80/main80_event_timepoints.csv",
    "points": ROOT / "outputs/match1_point_inventory_20261117/point_inventory.csv",
    "identity": ROOT / "study/annotations/match1_hitter_identity_review.csv",
    "previous_direction_auc": ROOT / "study/analysis/reanalysis_20261117/threshold_free.csv",
    "reference_script": ROOT / "study/scripts/101_reanalyze_direction_agreement.py",
}
TIMES = (-0.5, -0.3, -0.2, -0.1)
THRESHOLD = 0.015
SEED = 20261117
BOOTSTRAP_REPS = 10000
ALPHA = 0.05
D_COL = "signed_displacement_toward_actual_court_width"
DIRECTION_COL = "actual_direction_target_front_view"
CLASSES = ("toward", "away", "stay", "all")

# Evidence: study/annotations/match1_hitter_identity_review.csv identifies
# H1 as 橋場 (pair H: 橋場・菊山), H2 as 上岡 (pair K: 上岡・丸山).
# The target front player belongs to the hitter's opponent pair:
# P1 faces H2, hence P1=H; P2 faces H1, hence P2=K.
HITTER_PAIR = {"H1": "H", "H2": "K"}
FRONT_PAIR = {"P1": "H", "P2": "K"}
HITTER_NAME = {"H1": "橋場", "H2": "上岡"}
EVENT_META = [
    "event_id", "video_id", "target_front_player_id", "hitter_id",
    "target_front_camera_end", "impact_source_frame", DIRECTION_COL,
]
POINT_META = [
    "point", "game", "games_before", "score_before", "point_winner",
    "display_from_frame", "next_display_from_frame",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    assert set(columns) <= set(frame.columns), f"{label}: missing {set(columns) - set(frame.columns)}"


def classify(d: np.ndarray) -> np.ndarray:
    return np.where(d >= THRESHOLD, "toward", np.where(d <= -THRESHOLD, "away", "stay"))


def time_suffix(time: float) -> str:
    return f"minus{abs(time):.1f}".replace(".", "p")


def load_and_link() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(INPUTS["main80"], encoding="utf-8-sig")
    required = EVENT_META + ["timepoint_sec", D_COL, "lateral_displacement_court_width"]
    require_columns(frame, required, "main80")
    assert len(frame) == 320 and frame.event_id.nunique() == 80
    assert not frame[required].isna().any().any(), "Missing main80 data; no silent exclusions"
    assert set(frame.timepoint_sec) == set(TIMES)
    assert not frame.duplicated(["event_id", "timepoint_sec"]).any()
    assert frame.groupby("event_id").size().eq(len(TIMES)).all()
    assert frame.groupby("event_id")[EVENT_META[1:]].nunique().eq(1).all().all()
    for column in ["impact_source_frame", "timepoint_sec", D_COL, "lateral_displacement_court_width"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        assert np.isfinite(frame[column].to_numpy()).all(), f"Nonfinite {column}"
    assert frame.impact_source_frame.ge(0).all()
    assert frame.impact_source_frame.eq(np.floor(frame.impact_source_frame)).all()
    frame["impact_source_frame"] = frame.impact_source_frame.astype(int)
    assert set(frame.video_id) == {"match1"}
    assert set(frame.hitter_id) == set(HITTER_PAIR)
    assert set(frame.target_front_player_id) == set(FRONT_PAIR)
    assert set(frame[DIRECTION_COL]) <= {"front_left", "front_right"}
    assert set(frame.target_front_camera_end) <= {"near", "far"}
    events = frame[EVENT_META].drop_duplicates("event_id").sort_values(
        ["impact_source_frame", "event_id"]
    ).reset_index(drop=True)
    assert not events.impact_source_frame.duplicated().any(), "Ambiguous ordering of labeled impacts"

    identity = pd.read_csv(INPUTS["identity"], encoding="utf-8-sig")
    identity_cols = ["event_id", "target_front_player_id", "hitter_id", "impact_source_frame"]
    require_columns(identity, identity_cols + ["hitter_description", "review_status"], "identity")
    assert not identity.event_id.duplicated().any()
    reviewed = events.merge(identity, on="event_id", suffixes=("", "_review"),
                            how="left", validate="one_to_one", indicator=True)
    assert reviewed._merge.eq("both").all(), "Main80 event absent from identity review"
    for column in identity_cols[1:]:
        assert reviewed[column].eq(reviewed[f"{column}_review"]).all(), f"Identity mismatch: {column}"
    assert reviewed.review_status.eq("confirmed").all()
    for hitter, name in HITTER_NAME.items():
        assert set(reviewed.loc[reviewed.hitter_id.eq(hitter), "hitter_description"]) == {name}
    events["hitter_pair"] = events.hitter_id.map(HITTER_PAIR)
    events["front_pair"] = events.target_front_player_id.map(FRONT_PAIR)
    assert events.front_pair.ne(events.hitter_pair).all(), "Front and hitter must be opponent pairs (all 80)"

    points = pd.read_csv(INPUTS["points"], encoding="utf-8-sig")
    require_columns(points, POINT_META, "points")
    points = points[POINT_META].copy()
    assert len(points) == 63 and not points.point.duplicated().any()
    assert not points[POINT_META[:-3] + ["display_from_frame"]].isna().any().any()
    for column in ["point", "game", "display_from_frame", "next_display_from_frame"]:
        points[column] = pd.to_numeric(points[column], errors="raise")
        values = points[column].dropna().to_numpy()
        assert np.isfinite(values).all() and np.equal(values, np.floor(values)).all()
    points = points.sort_values("display_from_frame").reset_index(drop=True)
    assert points.point.tolist() == list(range(1, len(points) + 1))
    assert points.display_from_frame.ge(0).all() and points.display_from_frame.diff().iloc[1:].gt(0).all()
    assert points.next_display_from_frame.iloc[:-1].notna().all()
    assert pd.isna(points.next_display_from_frame.iloc[-1])
    assert set(points.point_winner.dropna()) <= {"H", "K"}
    assert points.point_winner.iloc[:-1].notna().all() and pd.isna(points.point_winner.iloc[-1])
    starts = points.display_from_frame.to_numpy(dtype=float)
    ends = points.next_display_from_frame.fillna(np.inf).to_numpy(dtype=float)
    assert np.all(ends > starts)
    # Gaps between inventory intervals are allowed; overlap is not.
    assert np.all(ends[:-1] <= starts[1:]), "Overlapping point intervals"
    impacts = events.impact_source_frame.to_numpy()
    matches = (impacts[:, None] >= starts) & (impacts[:, None] < ends)
    assert np.all(matches.sum(axis=1) == 1), "Every one of the 80 events must match exactly one point"
    assigned = points.iloc[matches.argmax(axis=1)].reset_index(drop=True)
    events = pd.concat([events, assigned], axis=1)
    events["front_pair_won_point"] = pd.array(
        [pd.NA if pd.isna(winner) else pair == winner
         for pair, winner in zip(events.front_pair, events.point_winner)], dtype="boolean"
    )
    events["impact_order_in_point"] = events.groupby("point").cumcount() + 1
    events["labeled_impacts_in_point"] = events.groupby("point").event_id.transform("size")
    events["is_last_labeled_impact_in_point"] = events.impact_order_in_point.eq(events.labeled_impacts_in_point)
    assert events.groupby("point").is_last_labeled_impact_in_point.sum().eq(1).all()
    assert len(events) == 80 and events.point.notna().all()

    linkage_cols = ["event_id", "point", "front_pair", "hitter_pair", "point_winner",
                    "front_pair_won_point", "impact_order_in_point", "is_last_labeled_impact_in_point"]
    long = frame.merge(events[linkage_cols], on="event_id", validate="many_to_one")
    # Match 101's front-player rightward displacement convention and independently
    # check it against the court-plane lateral displacement / camera orientation.
    long["v_front_right_court_width"] = long[D_COL] * np.where(
        long[DIRECTION_COL].eq("front_right"), 1, -1
    )
    v_from_lateral = long.lateral_displacement_court_width * np.where(
        long.target_front_camera_end.eq("near"), 1, -1
    )
    assert np.allclose(long.v_front_right_court_width, v_from_lateral, atol=1e-12, rtol=0)
    long["movement_class"] = classify(long[D_COL].to_numpy())
    long = long.sort_values(["impact_source_frame", "event_id", "timepoint_sec"]).reset_index(drop=True)
    for time in TIMES:
        selected = long.loc[long.timepoint_sec.eq(time)].set_index("event_id")
        suffix = time_suffix(time)
        events[f"d_{suffix}_court_width"] = events.event_id.map(selected[D_COL])
        events[f"movement_{suffix}"] = events.event_id.map(selected.movement_class)
    assert events.filter(regex="^d_").notna().all().all()
    return long, events, points


def holm_adjust(pvalues: list[float]) -> np.ndarray:
    """Same four-timepoint Holm step-down definition as 101, including NaNs."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(np.where(np.isfinite(p), p, 1.0), kind="stable")
    adjusted = np.full(len(p), np.nan)
    ordered = np.minimum(1, np.maximum.accumulate(
        np.where(np.isfinite(p[order]), p[order], 1) * np.arange(len(p), 0, -1)
    ))
    adjusted[order] = ordered
    adjusted[~np.isfinite(p)] = np.nan
    return adjusted


def exact_rate(k: int, n: int) -> dict:
    if not n:
        return dict(front_pair_win_rate=np.nan, win_ci95_low=np.nan, win_ci95_high=np.nan)
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=1 - ALPHA, method="exact")
    return dict(front_pair_win_rate=k / n, win_ci95_low=float(ci.low), win_ci95_high=float(ci.high))


def movement_table(long: pd.DataFrame, subset: str) -> pd.DataFrame:
    rows = []
    for time in TIMES:
        group = long.loc[long.timepoint_sec.eq(time)]
        known = group.loc[group.front_pair_won_point.notna()]
        contingency = np.array([
            [int((known.movement_class.eq(label) & known.front_pair_won_point.eq(won)).sum())
             for won in (True, False)] for label in ("toward", "away")
        ], dtype=int)
        odds, p = np.nan, np.nan
        # Follow 101: absent row or column margins => test is not estimable.
        if (contingency.sum(axis=1) > 0).all() and (contingency.sum(axis=0) > 0).all():
            fisher = stats.fisher_exact(contingency, alternative="two-sided")
            odds, p = float(fisher.statistic), float(fisher.pvalue)
        for label in CLASSES:
            selected = group if label == "all" else group.loc[group.movement_class.eq(label)]
            eligible = selected.loc[selected.front_pair_won_point.notna()]
            n, k = len(eligible), int(eligible.front_pair_won_point.sum())
            row = dict(subset=subset, timepoint_sec=time, threshold_court_width=THRESHOLD,
                       movement_class=label, total_events=len(selected), known_outcome_events=n,
                       unknown_outcome_events=len(selected) - n, front_pair_won_count=k,
                       front_pair_lost_count=n - k, point_clusters=int(eligible.point.nunique()),
                       toward_won=int(contingency[0, 0]), toward_lost=int(contingency[0, 1]),
                       away_won=int(contingency[1, 0]), away_lost=int(contingency[1, 1]),
                       fisher_odds_ratio=odds, fisher_p_two_sided=p,
                       inference_assumes_independent_events=True)
            row.update(exact_rate(k, n))
            rows.append(row)
    tests = [next(row["fisher_p_two_sided"] for row in rows if row["timepoint_sec"] == t) for t in TIMES]
    adjusted = dict(zip(TIMES, holm_adjust(tests)))
    for row in rows:
        row["fisher_p_holm_4timepoints"] = float(adjusted[row["timepoint_sec"]])
    table = pd.DataFrame(rows)
    for time in TIMES:
        part = table.loc[table.timepoint_sec.eq(time)].set_index("movement_class")
        for column in ["total_events", "known_outcome_events", "unknown_outcome_events",
                       "front_pair_won_count", "front_pair_lost_count"]:
            assert part.loc[list(CLASSES[:-1]), column].sum() == part.loc["all", column]
    return table


def auc_probability(values: np.ndarray, positive: np.ndarray) -> float:
    if not positive.any() or positive.all():
        return float("nan")
    difference = values[positive, None] - values[~positive][None, :]
    return float(np.mean((difference > 0) + 0.5 * (difference == 0)))


def auc_and_mannwhitney(values: np.ndarray, positive: np.ndarray) -> dict:
    auc = auc_probability(values, positive)
    u, p = np.nan, np.nan
    if positive.any() and (~positive).any():
        test = stats.mannwhitneyu(values[positive], values[~positive], alternative="two-sided",
                                  method="asymptotic", use_continuity=True)
        u, p = float(test.statistic), float(test.pvalue)
        assert np.isclose(auc, u / (int(positive.sum()) * int((~positive).sum())), atol=1e-14)
    return dict(auc=auc, mannwhitney_u=u, mannwhitney_p_two_sided=p,
                mannwhitney_method="asymptotic_tie_corrected_continuity",
                mannwhitney_assumes_independent_events=True)


def point_cluster_auc_ci(values: np.ndarray, positive: np.ndarray, points: np.ndarray,
                         rng: np.random.Generator) -> dict:
    """101's G x G AUC kernel, now with actual point IDs as whole clusters.

    Draw G points with replacement from G represented eligible points, keeping
    every event of each drawn point (including multiplicity). The estimator is
    pooled/event-weighted, not an equally weighted mean of point-level AUCs.
    """
    unique, code = np.unique(points, return_inverse=True)
    g = len(unique)
    assert g > 0
    n_positive = np.bincount(code[positive], minlength=g)
    n_negative = np.bincount(code[~positive], minlength=g)
    kernel = np.zeros((g, g))
    difference = values[positive, None] - values[~positive][None, :]
    scores = (difference > 0).astype(float) + 0.5 * (difference == 0)
    np.add.at(kernel, (code[positive, None], code[~positive][None, :]), scores)
    draws = rng.integers(0, g, size=(BOOTSTRAP_REPS, g))
    weights = np.zeros((BOOTSTRAP_REPS, g), dtype=np.int64)
    np.add.at(weights, (np.arange(BOOTSTRAP_REPS)[:, None], draws), 1)
    denominator = (weights @ n_positive) * (weights @ n_negative)
    valid = denominator > 0
    numerator = np.einsum("bi,ij,bj->b", weights, kernel, weights, optimize=True)
    samples = np.full(BOOTSTRAP_REPS, np.nan)
    samples[valid] = numerator[valid] / denominator[valid]
    # Independently verify duplicated-point weights against explicit resampling.
    for i in range(min(5, BOOTSTRAP_REPS)):
        indices = np.concatenate([np.flatnonzero(code == cluster) for cluster in draws[i]])
        explicit = auc_probability(values[indices], positive[indices])
        assert np.isclose(samples[i], explicit, atol=1e-14, equal_nan=True)
    ci = np.quantile(samples[valid], [ALPHA / 2, 1 - ALPHA / 2], method="linear") if valid.any() else [np.nan, np.nan]
    return dict(auc_ci95_low=float(ci[0]), auc_ci95_high=float(ci[1]),
                bootstrap_reps=BOOTSTRAP_REPS, bootstrap_auc_valid_reps=int(valid.sum()),
                bootstrap_auc_invalid_reps=int((~valid).sum()), point_clusters=g,
                cluster_is_estimated=False, bootstrap_ci_method="percentile_linear")


def displacement_table(long: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for time in TIMES:
        group = long.loc[long.timepoint_sec.eq(time)]
        known = group.loc[group.front_pair_won_point.notna()]
        d = known[D_COL].to_numpy(dtype=float)
        won = known.front_pair_won_point.to_numpy(dtype=bool)
        row = dict(timepoint_sec=time, total_events=len(group), known_outcome_events=len(known),
                   unknown_outcome_events=len(group) - len(known), won_n=int(won.sum()), lost_n=int((~won).sum()))
        for label, mask in [("won", won), ("lost", ~won)]:
            values = d[mask]
            row[f"{label}_point_clusters"] = int(known.loc[mask, "point"].nunique())
            row[f"{label}_d_mean_court_width"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{label}_d_median_court_width"] = float(np.median(values)) if len(values) else np.nan
            quantiles = np.quantile(values, [0.25, 0.75], method="linear") if len(values) else [np.nan, np.nan]
            row[f"{label}_d_q25_court_width"], row[f"{label}_d_q75_court_width"] = map(float, quantiles)
        row.update(auc_and_mannwhitney(d, won))
        row.update(point_cluster_auc_ci(d, won, known.point.to_numpy(), rng))
        rows.append(row)
    return pd.DataFrame(rows)


def direction_table(long: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    previous = pd.read_csv(INPUTS["previous_direction_auc"], encoding="utf-8-sig")
    require_columns(previous, ["cohort", "timepoint_sec", "total_events", "actual_front_right",
                               "actual_front_left", "auc", "auc_ci95_low", "auc_ci95_high",
                               "estimated_clusters", "cluster_gap_frames", "bootstrap_reps",
                               "bootstrap_auc_valid_reps", "bootstrap_auc_invalid_reps"], "previous AUC")
    previous = previous.loc[previous.cohort.eq("main80")]
    assert len(previous) == len(TIMES) and set(previous.timepoint_sec) == set(TIMES)
    rows = []
    for time in TIMES:
        group = long.loc[long.timepoint_sec.eq(time)]
        old = previous.loc[previous.timepoint_sec.eq(time)].iloc[0]
        values = group.v_front_right_court_width.to_numpy(dtype=float)
        right = group[DIRECTION_COL].eq("front_right").to_numpy()
        row = dict(cohort="main80", timepoint_sec=time, total_events=len(group),
                   actual_front_right=int(right.sum()), actual_front_left=int((~right).sum()))
        row.update(auc_and_mannwhitney(values, right))
        row.update(point_cluster_auc_ci(values, right, group.point.to_numpy(), rng))
        # Compare memberships, not just the number of estimated/actual clusters.
        ordered = group[["event_id", "point", "impact_source_frame"]].sort_values(
            ["impact_source_frame", "event_id"]
        ).copy()
        ordered["estimated_cluster"] = (
            ordered.impact_source_frame.diff().gt(int(old.cluster_gap_frames)).cumsum() + 1
        )
        split_points = int(ordered.groupby("point").estimated_cluster.nunique().gt(1).sum())
        merged_clusters = int(ordered.groupby("estimated_cluster").point.nunique().gt(1).sum())
        assert ordered.estimated_cluster.nunique() == int(old.estimated_clusters)
        row["previous_partition_matches_actual_points"] = split_points == 0 and merged_clusters == 0
        row["actual_points_split_by_previous_partition"] = split_points
        row["previous_clusters_merging_actual_points"] = merged_clusters
        assert len(group) == old.total_events
        assert int(right.sum()) == old.actual_front_right and int((~right).sum()) == old.actual_front_left
        assert np.isclose(row["auc"], old.auc, atol=1e-10, rtol=0), "101 point estimate changed"
        for column in ["auc", "auc_ci95_low", "auc_ci95_high", "estimated_clusters", "cluster_gap_frames",
                       "bootstrap_reps", "bootstrap_auc_valid_reps", "bootstrap_auc_invalid_reps"]:
            row[f"previous_{column}"] = old[column]
        row["previous_cluster_is_estimated"] = True
        rows.append(row)
    return pd.DataFrame(rows)


def point_counts(events: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    result = points.copy()
    grouped = events.groupby("point")
    count_columns = {
        "labeled_event_count": grouped.size(),
        "known_outcome_event_count": grouped.front_pair_won_point.count(),
        "front_pair_won_event_count": grouped.front_pair_won_point.sum(),
        "front_H_event_count": events.loc[events.front_pair.eq("H")].groupby("point").size(),
        "front_K_event_count": events.loc[events.front_pair.eq("K")].groupby("point").size(),
    }
    for column, counts in count_columns.items():
        result[column] = result.point.map(counts).fillna(0).astype(int)
    for label, function in [("first", "min"), ("last", "max")]:
        result[f"{label}_labeled_impact_source_frame"] = result.point.map(
            grouped.impact_source_frame.agg(function)
        ).astype("Int64")
    assert result.labeled_event_count.sum() == len(events)
    assert result.front_H_event_count.add(result.front_K_event_count).eq(result.labeled_event_count).all()
    return result


def number(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if np.isfinite(value) else "算出不可"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%" if np.isfinite(value) else "算出不可"


def interval(row: dict, low: str, high: str, percent: bool = False) -> str:
    formatter = pct if percent else number
    return f"[{formatter(row[low])}, {formatter(row[high])}]"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(row) + " |" for row in rows])


def significance(p: float) -> str:
    if not np.isfinite(p):
        return "検定値を算出できなかった"
    return "有意だった" if p < ALPHA else "有意でなかった"


def test_summary(rows: list[dict], column: str, label: str) -> str:
    defined = [row for row in rows if np.isfinite(row[column])]
    significant = [row for row in defined if row[column] < ALPHA]
    if significant:
        times = "、".join(f"{row['timepoint_sec']:.1f}秒（p={number(row[column])}）" for row in significant)
        text = f"{label}は{times}で有意だった。"
    elif defined:
        text = f"{label}は算出可能な全時点で有意でなかった。"
    else:
        text = f"{label}は全時点で算出できなかった。"
    if len(defined) < len(rows):
        text += f"算出不可は{len(rows) - len(defined)}時点だった。"
    return text


def movement_report(table: pd.DataFrame) -> str:
    rows = table.to_dict("records")
    counts = md_table(["時点（秒）", "分類", "対象場面", "得点/結果既知", "前衛側得点率", "CP 95%CI"], [
        [f"{r['timepoint_sec']:.1f}", r["movement_class"], str(r["total_events"]),
         f"{r['front_pair_won_count']}/{r['known_outcome_events']}", pct(r["front_pair_win_rate"]),
         interval(r, "win_ci95_low", "win_ci95_high", True)] for r in rows
    ])
    tests = [r for r in rows if r["movement_class"] == "all"]
    test_table = md_table(["時点（秒）", "toward [得点,失点]", "away [得点,失点]", "Fisher両側p", "Holm p"], [
        [f"{r['timepoint_sec']:.1f}", f"[{r['toward_won']},{r['toward_lost']}]",
         f"[{r['away_won']},{r['away_lost']}]", number(r["fisher_p_two_sided"]),
         number(r["fisher_p_holm_4timepoints"])] for r in tests
    ])
    return counts + "\n\n" + test_table + "\n\n" + test_summary(tests, "fisher_p_holm_4timepoints", "4時点のHolm補正後のFisher検定")


def make_report(tables: dict[str, pd.DataFrame], events: pd.DataFrame) -> str:
    a = tables["movement_by_outcome"]
    b = tables["displacement_by_outcome"].to_dict("records")
    c = tables["direction_auc_point_clusters"].to_dict("records")
    last = tables["last_impact_sensitivity"]
    points = tables["point_event_counts"]
    represented = points.loc[points.labeled_event_count.gt(0)]
    baseline = a.loc[a.timepoint_sec.eq(TIMES[0]) & a.movement_class.eq("all")].iloc[0]
    unknown = int(events.front_pair_won_point.isna().sum())
    both_pairs = int(events.groupby("point").front_pair.nunique().eq(2).sum())
    t02 = a.loc[a.timepoint_sec.eq(-0.2)].set_index("movement_class")
    class_text = "、".join(
        f"{label}は{int(t02.loc[label, 'front_pair_won_count'])}/{int(t02.loc[label, 'known_outcome_events'])}"
        f"（{pct(t02.loc[label, 'front_pair_win_rate'])}、CP 95%CI "
        f"{interval(t02.loc[label].to_dict(), 'win_ci95_low', 'win_ci95_high', True)}）"
        for label in CLASSES[:-1]
    )
    bt = "\n".join(
        f"- {r['timepoint_sec']:.1f}秒：AUC={number(r['auc'])}、実際の点をクラスタとする95%CI "
        f"{interval(r, 'auc_ci95_low', 'auc_ci95_high')}、Mann–Whitney両側p={number(r['mannwhitney_p_two_sided'])}で"
        f"{significance(r['mannwhitney_p_two_sided'])}。" for r in b
    )
    b_table = md_table(["時点（秒）", "得点/失点 n", "d中央値 得点/失点", "AUC", "点クラスタ95%CI", "MW両側p"], [
        [f"{r['timepoint_sec']:.1f}", f"{r['won_n']}/{r['lost_n']}",
         f"{number(r['won_d_median_court_width'], 6)}/{number(r['lost_d_median_court_width'], 6)}",
         number(r["auc"]), interval(r, "auc_ci95_low", "auc_ci95_high"), number(r["mannwhitney_p_two_sided"])] for r in b
    ])
    c_table = md_table(["時点（秒）", "方向一致AUC", "実際の点95%CI", "101 推定まとまり95%CI", "クラスタ数 実点/推定"], [
        [f"{r['timepoint_sec']:.1f}", number(r["auc"]), interval(r, "auc_ci95_low", "auc_ci95_high"),
         interval(r, "previous_auc_ci95_low", "previous_auc_ci95_high"),
         f"{r['point_clusters']}/{int(r['previous_estimated_clusters'])}"] for r in c
    ])
    point_table = md_table(["点", "ゲーム", "点前スコア", "勝者", "場面数", "前衛H/K"], [
        [str(int(r.point)), str(int(r.game)), str(r.score_before), str(r.point_winner) if pd.notna(r.point_winner) else "不明",
         str(int(r.labeled_event_count)), f"{int(r.front_H_event_count)}/{int(r.front_K_event_count)}"]
        for r in represented.itertuples()
    ])
    zero_points = "、".join(str(int(p)) for p in points.loc[points.labeled_event_count.eq(0), "point"])
    invalid_b = sum(r["bootstrap_auc_invalid_reps"] for r in b)
    invalid_c = sum(r["bootstrap_auc_invalid_reps"] for r in c)
    last_baseline = last.loc[last.timepoint_sec.eq(TIMES[0]) & last.movement_class.eq("all")].iloc[0]
    same_partition = all(r["previous_partition_matches_actual_points"] for r in c)
    cluster_note = (
        "今回の対象場面について両者の所属を照合したところ、場面の分割も完全に一致した。"
        "したがって今回の旧新CIの微差は、異なる乱数列を用いた有限回の再標本化による変動である。"
        if same_partition else
        "両者の所属を照合すると場面の分割には相違があった。旧新CIの違いには、"
        "クラスタ分割の相違と有限回の再標本化による変動が含まれる。"
    )
    c_contains = [r["auc_ci95_low"] <= 0.5 <= r["auc_ci95_high"] for r in c]
    c_ci_text = ("実際の点による95%CIは全時点で0.5を含んだ。" if all(c_contains)
                 else "実際の点による95%CIが0.5を含まない時点：" + "、".join(
                     f"{r['timepoint_sec']:.1f}秒" for r, includes in zip(c, c_contains) if not includes) + "。")
    return f"""# match1 前衛の打球前横移動と点の結果（探索的集計）

## 対象と対応付け

1試合（match1）の{len(events)}場面を、点一覧の{len(points)}点のうち{len(represented)}点に対応付けた。各場面は `display_from_frame <= impact_source_frame < next_display_from_frame`、最後の点は下限のみで割り当てた。全場面の一意な割り当てと、前衛ペアと打者ペアが異なることのassertは通過した。前衛P1はH（橋場・菊山）、P2はK（上岡・丸山）。打者H1=橋場、H2=上岡という既存の確認済み注釈とも照合した。

結果既知は{int(baseline.known_outcome_events)}場面、勝者不明による除外は{unknown}場面。全場面での前衛側得点率（場面数で重み付けした基準）は{int(baseline.front_pair_won_count)}/{int(baseline.known_outcome_events)}={pct(baseline.front_pair_win_rate)}、CP 95%CI {interval(baseline.to_dict(), 'win_ci95_low', 'win_ci95_high', True)}だった。これは点単位の勝率ではない。

同じ点の場面は点の勝者という結果を共有するため、独立ではない。両ペアの前衛を含む点は{both_pairs}点あり、その点では前衛ペアに応じて得点/失点が反転する。B・CのCIは実際の点を単位に再標本化する。一方、指定されたA・DのClopper–Pearson CIとFisher検定、Bの通常のMann–Whitney検定は場面の独立性を仮定する参考値であり、点内相関を調整した推測ではない。Holm補正もこの非独立性を解消しない。有意でない結果から関係がないとは断定しない。

## A 動きの分類と前衛側の得点

dは実際の打球方向を正とする横変位（コート幅比）。閾値は{THRESHOLD}で、towardはd >= 閾値、awayはd <= −閾値、stayは|d| < 閾値。allは全場面の基準。割合の分母は各分類の結果既知場面数で、CIはClopper–Pearsonの両側95%CI。toward/away × 得点/失点のFisher正確検定は両側、4時点内でHolm補正した。有意水準は{pct(ALPHA)}。

{movement_report(a)}

−0.2秒では{class_text}。toward対awayのFisher両側p={number(t02.loc['all', 'fisher_p_two_sided'])}、Holm p={number(t02.loc['all', 'fisher_p_holm_4timepoints'])}で{significance(t02.loc['all', 'fisher_p_holm_4timepoints'])}。

## B 閾値を使わないdと得失点の比較

AUC = P(d_得点 > d_失点) + 0.5 P(d_得点 = d_失点)。0.5より大きいと得点場面のdが大きい傾向を表す。得点・失点の異なる点のペアだけに限定する定義ではなく、プールした全場面を用いる。Mann–Whitneyは101番に合わせて同順位補正・連続性補正ありの漸近法、両側。ここでのp値は未補正の探索的参考値で、Bに対する追加の多重比較補正は行っていない。

{b_table}

{bt}

CIは結果既知の場面を持つ実際の{b[0]['point_clusters']}点から同数の点を復元抽出し、点内の全場面を複製回数も含めて保持するクラスタ・ブートストラップ{BOOTSTRAP_REPS}回の百分位法（2.5%, 97.5%、線形補間）。AUCは場面数による重みを保ち、点ごとのAUCの単純平均ではない。乱数シードは{SEED}。片群が消えた反復はCIから除外して記録し、B全時点の無効反復は合計{invalid_b}回だった。

## C 方向一致AUCの実際の点によるCI

前衛から見て右向きの変位v = d ×（実際の打球が右なら+1、左なら−1）を用い、AUC = P(v_実際右 > v_実際左) + 0.5 P(同値)を再計算した。閾値は用いない。vはカメラ側を考慮した横変位とも一致することを確認した。101番の `threshold_free.csv` のmain80と点推定値・左右件数が一致することをassertし、CIだけを実際の点をクラスタにして計算し直した。

{c_table}

101番のCIはフレーム間隔{int(c[0]['previous_cluster_gap_frames'])}超による推定まとまりで、既存CSVから読み取った。今回のCは{c[0]['point_clusters']}点をクラスタとする。{cluster_note}{c_ci_text}C全時点の無効反復は合計{invalid_c}回だった。

## D 各点で最後のラベル場面に限定した感度分析

対象は{int(last_baseline.total_events)}場面（結果既知{int(last_baseline.known_outcome_events)}、勝者不明{int(last_baseline.unknown_outcome_events)}）。全場面より少なく、toward/awayの各セルはさらに小さいため精密な比較には足りない。各点内の80場面のうち時間的に最後のものを採ったので、実際のラリー最終打球であるとは限らず、点終了までの距離も測っていない。Aと同じ分類・CI・両側検定を用い、Dの4時点内で別にHolm補正した。

{movement_report(last)}

## 指導教員の指摘との関係と限界

方向とタイミングだけではフェイントや元の位置に戻る動きを区別できないという指摘に対し、今回は動きと最終的な点の得失点との関係を調べた。ただし、dの符号は依然として実際の打球方向に依存する。点の結果を加えてもフェイントや復位を直接識別できず、動きの意図や軌跡も分からない。これらを区別するには、元の位置・途中の方向反転・相手の反応などを別に観察・注釈する必要がある。

点の結果は、この打球のあとの多くの打球で決まる。後続プレー、選手やペア、点の状況などを調整していないため、この動きが得点を生んだという因果は言えない。単一試合の選択された場面についての探索的記述であり、一般的な戦術の優劣や、未知の試合に対する予測性能は検証していない。

## 点ごとの場面数

全点（0件を含む）は `point_event_counts.csv` に保存した。対象場面を持つ点は以下のとおり。

{point_table}

対象場面が0件の点：{zero_points}。

## 出力と再現方法

`event_point_outcome.csv` は1場面1行。`d_minus0p5_court_width` などが各時点のd、`movement_minus0p5` などが分類。`impact_order_in_point` は80場面内の点ごとの時系列順、`is_last_labeled_impact_in_point` はその最終場面。勝者不明なら `point_winner` と `front_pair_won_point` は空欄のまま保持し、A・B・Dの分母から除外する。Cは勝者を使わないので全場面を保持する。A・DのFisher列は同時点の各分類行に同じ値を併記しており、検定は各時点1回である。

`movement_by_outcome.csv`、`displacement_by_outcome.csv`、`direction_auc_point_clusters.csv`、`last_impact_sensitivity.csv` がA〜Dの集計。`metadata.json` に入力のSHA-256、定義、種、ライブラリ・実行環境、検証結果を記録した。本文の数値は集計表と対応付け表から自動生成した。

実行：`study\\.venv\\Scripts\\python.exe -B study/scripts/102_link_front_motion_to_point_outcome.py`
"""


def main() -> int:
    hashes = {key: sha256(path) for key, path in INPUTS.items()}
    long, events, points = load_and_link()
    # A fixed, documented order of draws: all B timepoints, then all C timepoints.
    rng = np.random.default_rng(SEED)
    tables = {
        "event_point_outcome": events,
        "point_event_counts": point_counts(events, points),
        "movement_by_outcome": movement_table(long, "all_labeled_impacts"),
        "displacement_by_outcome": displacement_table(long, rng),
        "direction_auc_point_clusters": direction_table(long, rng),
        "last_impact_sensitivity": movement_table(
            long.loc[long.is_last_labeled_impact_in_point], "last_labeled_impact_per_point"
        ),
    }
    report = make_report(tables, events)
    assert hashes == {key: sha256(path) for key, path in INPUTS.items()}, "Input changed during analysis"
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)), "script_sha256": sha256(Path(__file__)),
        "inputs": {key: {"path": str(path.relative_to(ROOT)), "sha256": hashes[key]} for key, path in INPUTS.items()},
        "seed": SEED, "bootstrap_reps": BOOTSTRAP_REPS, "timepoints_sec": list(TIMES),
        "threshold_court_width": THRESHOLD, "alpha": ALPHA,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "python_executable": sys.executable, "platform": platform.platform(),
        "pair_mapping": {"front": FRONT_PAIR, "hitter": HITTER_PAIR, "hitter_names": HITTER_NAME,
                         "H": "橋場・菊山", "K": "上岡・丸山", "evidence": str(INPUTS['identity'].relative_to(ROOT))},
        "definitions": {
            "point_assignment": "display_from_frame <= impact_source_frame < next_display_from_frame; last interval has no upper bound",
            "point_interval_gaps": "Allowed, but every one of the 80 events must match exactly one interval; overlapping intervals forbidden",
            "front_pair_won_point": "front_pair == point_winner; nullable boolean; unknown remains blank, excluded from A/B/D only",
            "order_and_last": "Chronological impact order among the 80 labeled events in each point; last is not necessarily the actual final stroke",
            "d": D_COL + "; positive toward actual ball direction; unit=court width",
            "movement": "toward: d >= 0.015; away: d <= -0.015; stay: abs(d) < 0.015",
            "baseline": "All outcome-known events pooled, event-weighted; not a point-weighted win rate",
            "binomial_ci": "Two-sided Clopper-Pearson 95%; assumes independent events (not cluster-adjusted)",
            "fisher": "Two-sided, rows=toward/away, columns=front win/loss; undefined if any margin is zero; repeated within each timepoint's category rows",
            "holm": "Four-timepoint families separately for A and D, including undefined tests in family size; no additional B/C adjustment",
            "mannwhitney": "Two-sided asymptotic, tie-corrected, continuity correction; event-level reference p, not cluster-adjusted; same as 101",
            "outcome_auc": "P(d_won > d_lost) + 0.5*P(equal), pooled across eligible events including within-point event pairs",
            "v": "d if actual_direction_target_front_view == front_right else -d; checked against lateral displacement and near/far orientation",
            "direction_auc": "P(v_actual_right > v_actual_left) + 0.5*P(equal), same estimator and all 80 events as 101",
            "bootstrap": "G actual represented eligible points drawn G times uniformly with replacement; keep all events and duplicated draws; pooled event-weighted AUC; percentile CI with numpy linear quantile; omit and count one-class replicates",
            "bootstrap_rng": "numpy.default_rng(20261117), PCG64; B times [-0.5,-0.3,-0.2,-0.1] then C same order, one sequential stream",
            "previous_direction_ci": "Read main80 rows from 101 threshold_free.csv, estimated gap600 clusters; not recomputed or overwritten",
            "partition_comparison": "Reconstruct 101's frame-gap clusters on the same 80 events; compare point-to-estimated and estimated-to-point memberships, not just counts",
        },
        "validation": {
            "all_events_assigned_once": True, "front_pair_differs_from_hitter_pair_all_events": True,
            "identity_review_matches_all_events": True, "v_orientation_matches_lateral": True,
            "direction_auc_matches_101": True, "auc_matches_mannwhitney_U": True,
            "bootstrap_first_five_each_call_match_explicit_resampling": True,
            "input_hashes_unchanged": True,
        },
        "counts": {"matches": int(events.video_id.nunique()), "events": len(events), "input_points": len(points),
                   "represented_points": int(events.point.nunique()),
                   "unknown_outcome_events": int(events.front_pair_won_point.isna().sum()),
                   "front_won_events": int(events.front_pair_won_point.sum()),
                   "front_lost_events": int(events.front_pair_won_point.eq(False).sum()),
                   "last_labeled_events": int(events.is_last_labeled_impact_in_point.sum()),
                   "previous_estimated_partition_matches_actual_points": bool(
                       tables["direction_auc_point_clusters"].previous_partition_matches_actual_points.all())},
        "csv_encoding": "utf-8-sig", "missing_csv_values": "empty field",
        "output_rows": {f"{key}.csv": len(table) for key, table in tables.items()},
        "limitations": ["One match, selected events", "Events within a point are not independent",
                        "Point winner may be determined by many later strokes; no causal inference",
                        "Adding outcome does not identify feints or return-to-origin motions; d is retrospective"],
    }
    # All analysis assertions have passed before any result file is written.
    OUT.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", encoding="utf-8-sig", index=False, na_rep="", float_format="%.12g")
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "結果.md").write_text(report, encoding="utf-8")
    assert hashes == {key: sha256(path) for key, path in INPUTS.items()}, "Input changed during output"
    print(f"PASS: {len(events)} events assigned once across {events.point.nunique()} actual points; opponent-pair and identity assertions passed.")
    print(f"Known={events.front_pair_won_point.count()}, front won={events.front_pair_won_point.sum()}, lost={events.front_pair_won_point.eq(False).sum()}")
    print("A, t=-0.2s:")
    print(tables["movement_by_outcome"].loc[lambda x: x.timepoint_sec.eq(-0.2),
          ["movement_class", "known_outcome_events", "front_pair_won_count", "front_pair_win_rate",
           "win_ci95_low", "win_ci95_high", "fisher_p_two_sided", "fisher_p_holm_4timepoints"]].to_string(index=False))
    print("B, threshold-free outcome comparison:")
    print(tables["displacement_by_outcome"][["timepoint_sec", "auc", "auc_ci95_low", "auc_ci95_high", "mannwhitney_p_two_sided"]].to_string(index=False))
    print(f"Wrote {len(tables) + 2} outputs to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
