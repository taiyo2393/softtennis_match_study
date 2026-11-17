"""Export reviewed video windows for later screening, without calling them trials.

Rows with confirmed starts and ends are eligible for the next *screening* step.
They are not rally_id or trial_id records: a human must still establish the
relevant shot event before a row enters a movement analysis dataset.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[1] / "data" / "segments"
    rows = read(directory / f"{args.video_id}_rally_window_candidates.csv")
    approved = [r for r in rows if r["start_confirmation"] == "confirmed" and r["end_confirmation"] == "confirmed"]

    fields = [
        "screening_segment_id", "video_id", "source_start_timestamp_sec",
        "source_end_timestamp_sec", "duration_sec", "start_candidate_id",
        "end_boundary_id", "source_window_id", "segment_status", "restriction",
    ]
    output = []
    for i, row in enumerate(approved, start=1):
        start = float(row["human_start_timestamp_sec"] or row["source_start_timestamp_sec"])
        end = float(row["source_end_timestamp_sec"])
        output.append({
            "screening_segment_id": f"{args.video_id}_screen_{i:03d}",
            "video_id": args.video_id,
            "source_start_timestamp_sec": f"{start:.6f}",
            "source_end_timestamp_sec": f"{end:.6f}",
            "duration_sec": f"{end-start:.6f}",
            "start_candidate_id": row["start_candidate_id"],
            "end_boundary_id": row["end_boundary_id"],
            "source_window_id": row["window_id"],
            "segment_status": "eligible_for_event_screening",
            "restriction": "Not a rally_id or trial_id. Confirm hitter, impact, target front player, direction, and visibility before analysis.",
        })
    write(directory / f"{args.video_id}_reviewed_screening_segments.csv", fields, output)

    status = Counter(r["window_status"] for r in rows)
    report = directory / f"{args.video_id}_boundary_review_summary.md"
    report.write_text(
        f"# 境界レビュー集計: {args.video_id}\n\n"
        f"- 全ラリー窓候補: {len(rows)}\n"
        f"- 開始・終了とも確認済み（次のイベント選別に進める区間）: {len(approved)}\n"
        f"- 除外または未解決: {len(rows)-len(approved)}\n"
        f"- 状態内訳: " + ", ".join(f"{key}={value}" for key, value in sorted(status.items())) + "\n\n"
        "この集計は境界候補への人手確認結果であり、ラリー数・試技数・打球数ではない。\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(output)} reviewed screening segments and {report.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
