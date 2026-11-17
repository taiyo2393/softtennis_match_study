"""Summarise the audited semi-automatic labelling pilot without inflated claims."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def wilson(successes: int, total: int) -> tuple[str, str]:
    if not total:
        return "", ""
    z, p = 1.96, successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return f"{max(0.0, center - radius):.3f}", f"{min(1.0, center + radius):.3f}"


def row(metric: str, successes: int, total: int, interpretation: str) -> dict[str, object]:
    low, high = wilson(successes, total)
    return {"metric": metric, "numerator": successes, "denominator": total, "estimate": "" if not total else f"{successes / total:.3f}", "wilson95_low": low, "wilson95_high": high, "interpretation": interpretation}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    direction = read(project / "annotations" / f"shot_direction_review_{args.video_id}.csv")
    target_review = read(project / "annotations" / f"highres_front_track_review_{args.video_id}_yolo11m.csv")
    pose_review = read(project / "annotations" / f"target_front_pose_review_{args.video_id}_yolo11m.csv")
    pose = read(project / "data" / "pose" / "target_front" / f"{args.video_id}_target_front_pose_yolo11m.csv")
    confirmed = [item for item in direction if item["review_status"] == "confirmed"]
    ai_proposed = [item for item in confirmed if item["ai_front_view_proposal"]]
    ai_correct = [item for item in ai_proposed if item["actual_direction_front_view"] == item["ai_front_view_proposal"]]
    results = [
        row("human_confirmed_shot_direction_coverage", len(confirmed), len(direction), "Events with a confirmed post-impact direction label."),
        row("high_confidence_ai_direction_coverage", len(ai_proposed), len(confirmed), "Share of confirmed events for which AI offered a high-confidence direction candidate."),
        row("ai_direction_agreement_given_proposal", len(ai_correct), len(ai_proposed), "Agreement with the same human review, conditional on an AI proposal; not independent accuracy and not generalisable."),
        row("audited_target_front_track_usable_rate", sum(item["tracking_status"] == "usable" for item in target_review), len(target_review), "Human audit of the target-front track in this selected pilot only."),
        row("audited_target_front_pose_usable_rate", sum(item["pose_status"] == "usable" for item in pose_review), len(pose_review), "Human audit that the ankle pose follows the selected target front in this pilot."),
        row("bilateral_ankle_frame_coverage", sum(item["ankle_pair_visible"] == "1" for item in pose), len(pose), "Model output coverage; this is not landmark-location accuracy."),
    ]
    output_dir = project / "analysis"; output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.video_id}_semiautomated_pilot_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    print(output)
    for item in results:
        print(f"{item['metric']}: {item['numerator']}/{item['denominator']} = {item['estimate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
