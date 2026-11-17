"""Create a small, stratified human-audit set for fixed-camera classification."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--per-class", type=int, default=10)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def evenly_spaced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if len(rows) <= count:
        return rows
    positions = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    return [rows[position] for position in positions]


def extract_image(video: Path, timestamp: float, output: Path) -> None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{timestamp:.6f}",
        "-i", str(video), "-frames:v", "1", "-vf", "scale=960:-2", "-q:v", "3", "-y", str(output),
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))


def main() -> int:
    args = parse_args()
    project = Path(__file__).resolve().parents[1]
    segment_dir = project / "data" / "segments"
    scores = read_csv(segment_dir / f"{args.video_id}_fixed_view_scores.csv")
    by_view = {view: [row for row in scores if row["proposed_view"] == view] for view in ("main_camera", "other_camera")}
    selected = [("main_camera", row) for row in evenly_spaced(by_view["main_camera"], args.per_class)]
    selected += [("other_camera", row) for row in evenly_spaced(by_view["other_camera"], args.per_class)]
    output_dir = segment_dir / "audit" / args.video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, object]] = []
    for index, (expected, row) in enumerate(selected, start=1):
        timestamp = float(row["timestamp_sec"])
        image_name = f"{index:02d}_{expected}_{timestamp:.3f}s.jpg"
        extract_image(args.input, timestamp, output_dir / image_name)
        audit_rows.append({
            "audit_id": f"{args.video_id}_audit_{index:02d}", "video_id": args.video_id,
            "timestamp_sec": f"{timestamp:.6f}", "auto_view": expected,
            "background_difference": row["background_difference"], "image_filename": image_name,
            "human_view": "unreviewed", "review_note": "", "reviewed_at": "",
        })
    output = segment_dir / f"{args.video_id}_fixed_view_audit.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    print(f"Wrote {len(audit_rows)} audit images and {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
