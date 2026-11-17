"""Summarize validation-only threshold results and select a preset operating point.

The script intentionally reads only ``validation`` summaries.  It applies the
predeclared semi-automatic rule: maximize candidate presence recall while
requiring candidate presence precision of at least ``--min-precision``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.40, 0.50, 0.60, 0.70])
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    result_dir = project / "analysis" / "balltrack_evaluation"
    rows: list[dict[str, object]] = []
    for threshold in args.thresholds:
        run_name = f"threshold{int(threshold * 100):03d}_median120"
        path = result_dir / f"{args.video_id}_validation_{run_name}_summary.json"
        if not path.is_file():
            raise SystemExit(f"Missing validation summary: {path}")
        with path.open(encoding="utf-8") as file:
            summary = json.load(file)
        metrics = summary["metrics"]
        rows.append({
            "threshold": threshold,
            "run_name": run_name,
            "presence_precision": metrics["presence_precision"],
            "presence_recall": metrics["presence_recall"],
            "localized_recall": metrics["localized_recall"],
            "mean_distance_px_when_both_visible": metrics["mean_distance_px_when_both_visible"],
        })

    eligible = [row for row in rows if float(row["presence_precision"]) >= args.min_precision]
    if not eligible:
        raise SystemExit(f"No threshold meets precision >= {args.min_precision:.3f}")
    selected = max(eligible, key=lambda row: (float(row["presence_recall"]), float(row["presence_precision"])))
    for row in rows:
        row["selected_by_validation_rule"] = row is selected

    csv_path = result_dir / f"{args.video_id}_validation_threshold_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    note_path = result_dir / f"{args.video_id}_validation_threshold_selection.md"
    lines = [
        "# Validation-only threshold selection",
        "",
        f"Rule: maximize presence recall subject to presence precision >= {args.min_precision:.2f}.",
        f"Selected threshold: **{float(selected['threshold']):.2f}** ({selected['run_name']}).",
        "",
        "| threshold | presence precision | presence recall | localized recall | mean error (px) | selected |",
        "| ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in rows:
        lines.append(
            "| {threshold:.2f} | {presence_precision:.1%} | {presence_recall:.1%} | {localized_recall:.1%} | {mean_distance_px_when_both_visible:.1f} | {selected} |".format(
                **row, selected="yes" if row["selected_by_validation_rule"] else ""
            )
        )
    lines.extend([
        "",
        "The test split is not read by this script and must only be evaluated after this selection is frozen.",
        "",
    ])
    note_path.write_text("\n".join(lines), encoding="utf-8")
    print(csv_path)
    print(note_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
