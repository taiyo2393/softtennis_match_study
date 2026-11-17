"""Serve a local review UI for the candidate person-tracking pilot.

The reviewer records only a short interval-level decision and, when visible,
the target front player's candidate track ID.  This is deliberately not an
attempt to repair every detection box by hand.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import re
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


OPTIONS = ("unreviewed", "usable", "id_switch", "target_missing", "not_target_front", "unusable")
FIELDS = ["video_id", "segment_id", "chunk_id", "start_timestamp_sec", "end_timestamp_sec", "suggested_status",
          "human_status", "target_front_track_id", "review_note", "reviewed_at"]
APP_JS = """function jump(t) {
  const video = document.querySelector('#video');
  video.currentTime = t;
  video.play().catch(() => {});
  window.scrollTo({top: 0, behavior: 'smooth'});
}
async function save() {
  const decisions = [...document.querySelectorAll('article')].map(card => ({
    chunk_id: card.dataset.id,
    human_status: card.querySelector('.status').value,
    target_front_track_id: card.querySelector('.track').value,
    review_note: card.querySelector('.note').value
  }));
  const response = await fetch('/api/save', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({decisions})});
  const data = await response.json();
  document.querySelector('#status').textContent = data.message || data.error;
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--segment-id", required=True)
    parser.add_argument("--chunk-sec", type=float, default=2.0)
    parser.add_argument("--port", type=int, default=8768)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def prepare_rows(args: argparse.Namespace, project: Path) -> tuple[Path, Path]:
    tracking = project / "data" / "tracking" / f"{args.video_id}_{args.segment_id}_person_tracking_pilot.csv"
    review_path = project / "annotations" / f"{args.video_id}_{args.segment_id}_tracking_review.csv"
    if review_path.exists():
        return review_path, tracking
    detections = read_csv(tracking)
    timestamps = [float(row["source_timestamp_sec"]) for row in detections]
    if not timestamps:
        raise SystemExit("Tracking CSV contains no candidate boxes")
    start, end = min(timestamps), max(timestamps)
    rows = []
    current, number = start, 1
    while current <= end + 1e-6:
        chunk_end = min(end, current + args.chunk_sec)
        chunk = [row for row in detections if current <= float(row["source_timestamp_sec"]) <= chunk_end]
        counts: dict[str, int] = {}
        for row in chunk: counts[row["track_id"]] = counts.get(row["track_id"], 0) + 1
        # A stable four-player scene is normally represented by four or fewer persistent IDs.
        # This flag is only a review priority, never a final exclusion decision.
        persistent = sum(value >= max(2, round(len(chunk) / 8)) for value in counts.values())
        suggested = "review_priority" if persistent > 4 or persistent < 3 else "normal_review"
        rows.append({"video_id": args.video_id, "segment_id": args.segment_id, "chunk_id": f"chunk_{number:03d}",
                     "start_timestamp_sec": f"{current:.6f}", "end_timestamp_sec": f"{chunk_end:.6f}",
                     "suggested_status": suggested, "human_status": "unreviewed", "target_front_track_id": "",
                     "review_note": "", "reviewed_at": ""})
        current += args.chunk_sec; number += 1
    review_path.parent.mkdir(parents=True, exist_ok=True); write_csv(review_path, rows)
    return review_path, tracking


def render(rows: list[dict[str, str]], video_name: str, segment_start: float) -> str:
    cards = []
    for row in rows:
        options = "".join(f"<option value='{option}' {'selected' if option == row['human_status'] else ''}>{option}</option>" for option in OPTIONS)
        cards.append(f"""<article>
<h2>{html.escape(row['chunk_id'])} <small>{html.escape(row['suggested_status'])}</small></h2>
<p>{float(row['start_timestamp_sec']):.3f}–{float(row['end_timestamp_sec']):.3f} 秒 <a class='jump' href='/clips/{html.escape(row['chunk_id'], quote=True)}.mp4'>この区間を再生</a></p>
<label>判定 <select name='status__{html.escape(row['chunk_id'], quote=True)}'>{options}</select></label>
<label>対象前衛のID <input name='track__{html.escape(row['chunk_id'], quote=True)}' inputmode='numeric' value='{html.escape(row.get('target_front_track_id',''), quote=True)}' placeholder='例: 2'></label>
<label>メモ <input name='note__{html.escape(row['chunk_id'], quote=True)}' value='{html.escape(row.get('review_note',''), quote=True)}'></label></article>""")
    return f"""<!doctype html><html lang='ja'><meta charset='utf-8'><title>人物追跡の監査</title><style>
body{{font-family:system-ui,sans-serif;margin:20px;background:#f3f4f6}}header{{position:sticky;top:0;z-index:1;background:#fff;padding:12px;border-bottom:1px solid #ddd}}video{{display:block;width:min(960px,100%);background:#111;margin-top:8px}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px;margin-top:16px}}article{{background:#fff;border-radius:8px;padding:12px}}h1,h2,p{{margin-top:0}}h2{{font-size:16px}}small{{font-weight:normal;color:#9b1c1c}}label{{display:block;margin-top:8px}}input{{width:70%}}button,select,.jump{{font-size:15px;padding:4px}}.jump{{background:#e5e7eb;color:#111;text-decoration:none;border-radius:4px}}
</style><form method='post' action='/save'><header><h1>人物追跡の区間監査</h1>
<p>青枠のIDを見て、対象前衛が連続して追えている区間だけ <code>usable</code> にします。IDが途中で変われば <code>id_switch</code>、対象が消えれば <code>target_missing</code>。観客等の余計な枠は、対象前衛が安定していれば無視して構いません。</p>
<video id='video' controls preload='metadata'><source src='/video/{html.escape(video_name, quote=True)}' type='video/mp4'></video>
<button type='submit'>監査結果を保存</button></header><main>{''.join(cards)}</main></form></html>"""


def main() -> int:
    args = parse_args(); project = Path(__file__).resolve().parents[1]
    review_path, tracking_path = prepare_rows(args, project)
    params = json.loads((tracking_path.with_suffix(".json")).read_text(encoding="utf-8"))
    original_video_path = tracking_path.with_suffix(".mp4")
    browser_video_path = original_video_path.with_name(f"{original_video_path.stem}_browser.mp4")
    video_path = browser_video_path if browser_video_path.exists() else original_video_path
    clip_dir = project / "data" / "tracking" / "review_clips" / f"{args.video_id}_{args.segment_id}"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *values: object) -> None: return
        def send_text(self, status: HTTPStatus, body: str, content_type: str) -> None:
            data = body.encode("utf-8"); self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def send_video(self, path: Path) -> None:
            size = path.stat().st_size; start, end = 0, size - 1
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if match:
                if match.group(1): start = int(match.group(1))
                if match.group(2): end = min(int(match.group(2)), end)
                if start > end or start >= size: self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE); return
                status = HTTPStatus.PARTIAL_CONTENT
            else:
                status = HTTPStatus.OK
            self.send_response(status); self.send_header("Content-Type", "video/mp4"); self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if status == HTTPStatus.PARTIAL_CONTENT: self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as file:
                file.seek(start); remaining = end - start + 1
                while remaining:
                    block = file.read(min(64 * 1024, remaining))
                    if not block: break
                    try:
                        self.wfile.write(block)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(block)
        def do_GET(self) -> None:
            if urlparse(self.path).path == "/":
                self.send_text(HTTPStatus.OK, render(read_csv(review_path), video_path.name, float(params["start_timestamp_sec"])), "text/html; charset=utf-8"); return
            if urlparse(self.path).path == "/app.js":
                self.send_text(HTTPStatus.OK, APP_JS, "text/javascript; charset=utf-8"); return
            if urlparse(self.path).path == f"/video/{video_path.name}":
                self.send_video(video_path); return
            if urlparse(self.path).path.startswith("/clips/"):
                clip = clip_dir / Path(urlparse(self.path).path.removeprefix("/clips/")).name
                if clip.is_file(): self.send_video(clip); return
            self.send_error(HTTPStatus.NOT_FOUND)
        def do_POST(self) -> None:
            if urlparse(self.path).path == "/save":
                try:
                    size = int(self.headers["Content-Length"]); form = parse_qs(self.rfile.read(size).decode("utf-8"), keep_blank_values=True); rows = read_csv(review_path)
                    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
                    for row in rows:
                        key = row["chunk_id"]; status = form.get(f"status__{key}", ["unreviewed"])[0]
                        if status not in OPTIONS: raise ValueError("invalid status")
                        track = form.get(f"track__{key}", [""])[0].strip()
                        if track and not track.isdigit(): raise ValueError("target front track ID must be an integer")
                        row["human_status"], row["target_front_track_id"], row["review_note"], row["reviewed_at"] = status, track, form.get(f"note__{key}", [""])[0], now
                    write_csv(review_path, rows); self.send_response(HTTPStatus.SEE_OTHER); self.send_header("Location", "/"); self.end_headers(); return
                except (ValueError, KeyError, UnicodeDecodeError) as error: self.send_text(HTTPStatus.BAD_REQUEST, str(error), "text/plain; charset=utf-8"); return
            if urlparse(self.path).path != "/api/save": self.send_error(HTTPStatus.NOT_FOUND); return
            try:
                size = int(self.headers["Content-Length"]); decisions = {item["chunk_id"]: item for item in json.loads(self.rfile.read(size))["decisions"]}; rows = read_csv(review_path)
                if set(decisions) != {row["chunk_id"] for row in rows}: raise ValueError("chunk IDs do not match")
                now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
                for row in rows:
                    item = decisions[row["chunk_id"]]; status = str(item.get("human_status", "unreviewed"))
                    if status not in OPTIONS: raise ValueError("invalid status")
                    track = str(item.get("target_front_track_id", "")).strip()
                    if track and not track.isdigit(): raise ValueError("target front track ID must be an integer")
                    row["human_status"], row["target_front_track_id"], row["review_note"], row["reviewed_at"] = status, track, str(item.get("review_note", "")), now
                write_csv(review_path, rows); self.send_text(HTTPStatus.OK, json.dumps({"message": f"{len(rows)}区間を保存しました"}), "application/json; charset=utf-8")
            except (ValueError, KeyError, json.JSONDecodeError) as error: self.send_text(HTTPStatus.BAD_REQUEST, json.dumps({"error": str(error)}), "application/json; charset=utf-8")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Open http://127.0.0.1:{args.port}/"); server.serve_forever(); return 0


if __name__ == "__main__": raise SystemExit(main())
