"""Build before/cut/after contact sheets and an HTML index for cut review."""

from __future__ import annotations

import argparse
import csv
import html
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--window-sec", type=float, default=0.9)
    parser.add_argument("--tile-width", type=int, default=480)
    return parser.parse_args()


def make_sheet(video: Path, timestamp: float, output: Path, window: float, width: int) -> None:
    start = max(0.0, timestamp - window / 2)
    # Three frames show the scene before, at, and after the candidate without
    # materialising a full-frame image sequence on disk.
    vf = f"fps=3,scale={width}:-2,tile=3x1:padding=4:margin=4"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{start:.6f}", "-i", str(video), "-t", f"{window:.6f}",
        "-an", "-vf", vf, "-frames:v", "1", "-y", str(output),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"Could not make {output.name}: {completed.stderr.strip()}")


def main() -> int:
    args = parse_args()
    if not args.input.is_file() or not args.candidates.is_file():
        raise FileNotFoundError("input video or candidates CSV was not found")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH")

    with args.candidates.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cards: list[str] = []
    for index, row in enumerate(rows, start=1):
        candidate_id = row["candidate_id"]
        timestamp = float(row["timestamp_sec"])
        filename = f"{index:03d}_{candidate_id}_{timestamp:.3f}s.jpg"
        make_sheet(args.input, timestamp, args.output_dir / filename, args.window_sec, args.tile_width)
        cards.append(
            "<article>"
            f"<h2>{html.escape(candidate_id)}</h2>"
            f"<p>候補時刻: {timestamp:.3f} s / scene score: {html.escape(row['scene_score'])}</p>"
            f"<img src='{html.escape(filename)}' alt='{html.escape(candidate_id)}'>"
            "<p>左から候補前・候補付近・候補後。判定はCSVの review_status に記録。</p>"
            "</article>"
        )

    page = """<!doctype html><html lang='ja'><meta charset='utf-8'>
<title>カット候補レビュー</title><style>
body{font-family:system-ui,sans-serif;margin:20px;background:#f5f5f5}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px}
article{background:#fff;padding:12px;border-radius:8px}h1{font-size:22px}h2{font-size:16px;margin:0}p{margin:6px 0;color:#333}img{width:100%;height:auto;border:1px solid #ddd}
</style><h1>カット候補レビュー</h1><p>主カメラ・リプレイ・選手アップ・不明を候補CSVに記録する。</p><main>""" + "\n".join(cards) + "</main></html>"
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {len(rows)} contact sheets and {args.output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
