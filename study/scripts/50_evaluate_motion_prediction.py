"""Compare short-horizon movement predictors with event-wise leave-one-out CV.

No frame from an evaluated event is included in the corresponding training set.
The result is a single-match pilot evaluation, not a generalization claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float) -> float:
    mean = train_x.mean(axis=0); scale = train_x.std(axis=0); scale[scale == 0] = 1.0
    x = (train_x - mean) / scale; test = (test_x - mean) / scale
    x = np.column_stack([np.ones(len(x)), x]); test = np.r_[1.0, test]
    penalty = np.eye(x.shape[1]); penalty[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + alpha * penalty, x.T @ train_y)
    return float(test @ coef)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    nonzero = np.abs(actual) >= 0.002  # 0.2% image-width dead zone, fixed before evaluation.
    sign = np.sign(actual[nonzero]) == np.sign(predicted[nonzero])
    return {"mae_norm": float(np.mean(np.abs(error))), "rmse_norm": float(np.sqrt(np.mean(error ** 2))), "sign_accuracy_nonzero": float(np.mean(sign)) if len(sign) else float("nan"), "n_nonzero": int(nonzero.sum())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    dataset = args.dataset or project / "analysis" / "motion_prediction_dataset.csv"
    rows = read(dataset)
    if len(rows) < 5:
        raise SystemExit(f"Need at least 5 usable events for leave-one-event-out evaluation; got {len(rows)}")
    horizons = ("p010", "p020", "p030")
    models = {
        "ridge_front_history": ["front_x_m050", "front_y_m050", "front_x_m025", "front_y_m025", "front_x_m010", "front_y_m010", "front_x_p000", "front_y_p000"],
        "ridge_front_plus_hitter": ["front_x_m050", "front_y_m050", "front_x_m025", "front_y_m025", "front_x_m010", "front_y_m010", "front_x_p000", "front_y_p000", "hitter_x_m050", "hitter_y_m050", "hitter_x_m025", "hitter_y_m025", "hitter_x_m010", "hitter_y_m010", "hitter_x_p000", "hitter_y_p000"],
    }
    output_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for horizon in horizons:
        y = np.array([float(row[f"dx_{horizon}"]) for row in rows])
        front_x0 = np.array([float(row["front_x_p000"]) for row in rows])
        front_xm10 = np.array([float(row["front_x_m010"]) for row in rows])
        seconds = int(horizon.removeprefix("p")) / 100
        predictions: dict[str, np.ndarray] = {
            "static": np.zeros(len(rows)),
            "constant_velocity": (front_x0 - front_xm10) / 0.10 * seconds,
        }
        for name, fields in models.items():
            x = np.array([[float(row[field]) for field in fields] for row in rows])
            current = np.zeros(len(rows))
            for test_index in range(len(rows)):
                train = np.arange(len(rows)) != test_index
                current[test_index] = ridge_predict(x[train], y[train], x[test_index], args.alpha)
            predictions[name] = current
        for name, predicted in predictions.items():
            item = metrics(y, predicted)
            summaries.append({"horizon_sec": seconds, "model": name, "n_events": len(rows), "alpha": args.alpha, **item})
            for row, actual, value in zip(rows, y, predicted):
                output_rows.append({"event_id": row["event_id"], "horizon_sec": seconds, "model": name, "actual_dx_norm": f"{actual:.8f}", "predicted_dx_norm": f"{value:.8f}", "absolute_error_norm": f"{abs(value - actual):.8f}"})
    output = project / "analysis" / "motion_prediction_event_predictions.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0])); writer.writeheader(); writer.writerows(output_rows)
    summary = project / "analysis" / "motion_prediction_evaluation_summary.json"
    summary.write_text(json.dumps({"dataset": str(dataset), "evaluation": "leave-one-event-out; all events originate from one match", "alpha": args.alpha, "metric_unit": "image-width normalized horizontal displacement", "results": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in summaries:
        print(f"h={item['horizon_sec']:.1f} {item['model']}: MAE={item['mae_norm']:.5f}, RMSE={item['rmse_norm']:.5f}, n={item['n_events']}")
    print(output); print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
