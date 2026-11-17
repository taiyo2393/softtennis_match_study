"""Exploratorily evaluate pre-impact front-motion / shot-direction association.

For each pre-impact time point, this fits a fixed two-feature ridge logistic
model using only information available at that time point:

* target-front fixed-camera x position;
* target-front lateral velocity over the preceding 100 ms.

The impact-frame position and any post-impact frame are deliberately excluded.
Evaluation is leave-one-event-out *within one match*.  It is therefore a
feasibility analysis, not a test of generalisation, anticipation, or causality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


AVAILABLE_FEATURES = (
    "position_median_x_screen_fraction",
    "lateral_velocity_screen_fraction_per_sec",
)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-clipped))


def ridge_logistic_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """IRLS fit with an unpenalised intercept and fixed L2 penalty."""
    design = np.column_stack((np.ones(len(x)), x))
    beta = np.zeros(design.shape[1], dtype=float)
    penalty = np.diag([0.0] + [ridge] * x.shape[1])
    for _ in range(100):
        probability = sigmoid(design @ beta)
        weights = np.clip(probability * (1.0 - probability), 1e-7, None)
        gradient = design.T @ (y - probability) - penalty @ beta
        hessian = design.T @ (weights[:, None] * design) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta_next = beta + step
        if np.max(np.abs(beta_next - beta)) < 1e-8:
            return beta_next
        beta = beta_next
    return beta


def loo_probabilities(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    probabilities = np.empty(len(y), dtype=float)
    for held_out in range(len(y)):
        train = np.arange(len(y)) != held_out
        mean = x[train].mean(axis=0)
        scale = x[train].std(axis=0, ddof=0)
        scale[scale < 1e-12] = 1.0
        beta = ridge_logistic_fit((x[train] - mean) / scale, y[train], ridge)
        probabilities[held_out] = sigmoid(np.array([1.0, *((x[held_out] - mean) / scale)]) @ beta)
    return probabilities


def balanced_accuracy(y: np.ndarray, predicted: np.ndarray) -> float:
    positive = y == 1
    negative = y == 0
    if not positive.any() or not negative.any():
        return float("nan")
    sensitivity = float((predicted[positive] == 1).mean())
    specificity = float((predicted[negative] == 0).mean())
    return (sensitivity + specificity) / 2.0


def bootstrap_ci(y: np.ndarray, predicted: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    values: list[float] = []
    for _ in range(draws):
        indices = rng.integers(0, len(y), size=len(y))
        score = balanced_accuracy(y[indices], predicted[indices])
        if not math.isnan(score):
            values.append(score)
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def cohen_d(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or len(right) < 2:
        return None
    pooled = math.sqrt(((len(left) - 1) * left.var(ddof=1) + (len(right) - 1) * right.var(ddof=1)) / (len(left) + len(right) - 2))
    return None if pooled < 1e-12 else float((left.mean() - right.mean()) / pooled)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--features", default=",".join(AVAILABLE_FEATURES),
        help="Comma-separated pre-impact features; defaults to position and velocity.",
    )
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input feature table not found: {args.input}")
    if args.ridge <= 0 or args.permutations < 100 or args.bootstrap_draws < 100:
        raise SystemExit("Ridge must be positive; permutations and bootstrap draws must be >= 100")
    features = tuple(item.strip() for item in args.features.split(",") if item.strip())
    if not features or any(item not in AVAILABLE_FEATURES for item in features):
        raise SystemExit(f"--features must be one or more of {AVAILABLE_FEATURES}")

    rows = [row for row in read(args.input) if row.get("feature_status") == "usable"]
    by_time: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("actual_direction_front_view") not in ("front_left", "front_right"):
            continue
        row["lateral_velocity_screen_fraction_per_sec"] = str(
            float(row["lateral_velocity_px_per_sec"]) / 3840.0
        )
        by_time[row["timepoint_sec"]].append(row)
    if not by_time:
        raise SystemExit("No usable labelled feature rows")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, object]] = []
    descriptive_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "input": str(args.input),
        "model": {
            "type": "two_feature_ridge_logistic_regression",
            "features": list(features),
            "excluded_features": ["impact_median_x_px", "relative_to_impact_x_px", "post_impact_frames"],
            "ridge": args.ridge,
            "standardisation": "training-fold only",
            "positive_class": "front_right",
        },
        "evaluation_scope": "leave-one-event-out within one match only; no match-level generalisation or causal claim",
        "timepoints": {},
    }
    for timepoint in sorted(by_time, key=float):
        group = sorted(by_time[timepoint], key=lambda row: row["event_id"])
        event_ids = [row["event_id"] for row in group]
        if len(event_ids) != len(set(event_ids)):
            raise SystemExit(f"Duplicate event id at time point {timepoint}")
        x = np.asarray([[float(row[feature]) for feature in features] for row in group], dtype=float)
        y = np.asarray([int(row["actual_direction_front_view"] == "front_right") for row in group], dtype=int)
        if len(np.unique(y)) != 2:
            raise SystemExit(f"Both direction classes are required at {timepoint}")
        probability = loo_probabilities(x, y, args.ridge)
        predicted = (probability >= .5).astype(int)
        bacc = balanced_accuracy(y, predicted)
        accuracy = float((predicted == y).mean())
        rng = np.random.default_rng(args.seed + round(float(timepoint) * 1000))
        ci_low, ci_high = bootstrap_ci(y, predicted, rng, args.bootstrap_draws)
        permutation_scores = []
        for _ in range(args.permutations):
            shuffled = rng.permutation(y)
            shuffled_prediction = (loo_probabilities(x, shuffled, args.ridge) >= .5).astype(int)
            permutation_scores.append(balanced_accuracy(shuffled, shuffled_prediction))
        permutation_p = (1 + sum(score >= bacc for score in permutation_scores)) / (1 + len(permutation_scores))
        summary["timepoints"][timepoint] = {
            "events": len(y), "front_left": int((y == 0).sum()), "front_right": int((y == 1).sum()),
            "accuracy": accuracy, "balanced_accuracy": bacc,
            "balanced_accuracy_bootstrap_95ci": [ci_low, ci_high],
            "permutation_p_unadjusted": permutation_p,
        }
        for row, truth, probability_value, prediction in zip(group, y, probability, predicted):
            prediction_rows.append({
                "event_id": row["event_id"], "timepoint_sec": timepoint,
                "actual_direction_front_view": row["actual_direction_front_view"],
                "predicted_direction_front_view": "front_right" if prediction else "front_left",
                "front_right_probability": f"{probability_value:.6f}",
                "position_median_x_screen_fraction": row["position_median_x_screen_fraction"],
                "lateral_velocity_screen_fraction_per_sec": row["lateral_velocity_screen_fraction_per_sec"],
            })
        for feature in features:
            left = x[y == 0, features.index(feature)]
            right = x[y == 1, features.index(feature)]
            descriptive_rows.append({
                "timepoint_sec": timepoint, "feature": feature,
                "front_left_n": len(left), "front_left_mean": f"{left.mean():.8f}", "front_left_median": f"{np.median(left):.8f}",
                "front_right_n": len(right), "front_right_mean": f"{right.mean():.8f}", "front_right_median": f"{np.median(right):.8f}",
                "left_minus_right_mean": f"{left.mean() - right.mean():.8f}",
                "cohen_d_left_minus_right": "" if cohen_d(left, right) is None else f"{cohen_d(left, right):.6f}",
            })

    with args.output_prefix.with_name(args.output_prefix.name + "_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
    with args.output_prefix.with_name(args.output_prefix.name + "_feature_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(descriptive_rows[0])); writer.writeheader(); writer.writerows(descriptive_rows)
    args.output_prefix.with_name(args.output_prefix.name + "_evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote exploratory LOOCV results for {len(by_time)} time points: {args.output_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
