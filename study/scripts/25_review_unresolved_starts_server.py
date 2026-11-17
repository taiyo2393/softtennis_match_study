"""Serve bounded unresolved windows for human confirmation of the actual start."""

from __future__ import annotations

import argparse
import csv
import html
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


STATUSES = (
    "unreviewed",
    "complete_valid_rally",
    "complete_rally_after_first_serve_fault",
    "serve_fault_without_completed_rally",
    "rally_open_at_window_end",
    "rally_already_underway_at_window_start",
    "not_play",
    "needs_raw_video",
)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--port", type=int, default=8772)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    directory = project / "data" / "segments"
    csv_path = directory / f"{args.video_id}_unresolved_start_review.csv"
    clip_dir = directory / "unresolved_start_review" / args.video_id

    def page() -> str:
        cards = []
        for row in read(csv_path):
            start = float(row["source_start_timestamp_sec"])
            options = "".join(f"<option value='{value}' {'selected' if value == row['review_status'] else ''}>{value}</option>" for value in STATUSES)
            cards.append(f"""<article><h2>{html.escape(row['window_id'])}</h2>
<p>原動画: {start:.3f}–{float(row['source_end_timestamp_sec']):.3f}秒 ／ 長さ {float(row['duration_sec']):.1f}秒</p>
<video controls preload='metadata' src='/clip/{html.escape(row['window_id'])}.mp4'></video>
<label>判定 <select name='status__{row['window_id']}'>{options}</select></label>
<label>有効サーブ・インパクトの動画内時刻（秒） <input name='offset__{row['window_id']}' value=''></label>
<label>メモ <input name='note__{row['window_id']}' value='{html.escape(row['review_note'])}'></label></article>""")
        return """<!doctype html><html lang='ja'><meta charset='utf-8'><title>未解決開始点の確認</title>
<style>body{font-family:system-ui,sans-serif;background:#f3f4f6;margin:20px}header{position:sticky;top:0;background:white;padding:12px;z-index:1}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:12px;margin-top:16px}article{background:white;padding:12px;border-radius:8px}h2{font-size:16px;margin:0}video{display:block;width:100%;margin-top:8px;background:#111}label{display:block;margin-top:7px}input{width:70%;margin-left:6px}</style>
<form method='post' action='/save'><header><h1>未解決開始点の確認</h1><p>動画は候補区間だけで停止します。音声候補はこの映像では不適合のため表示しません。<code>complete_valid_rally</code> は<strong>有効サーブのインパクト</strong>からラリー終了までがこの動画内にある場合です。ファーストサーブがフォルト後にセカンドサーブから完結した場合は <code>complete_rally_after_first_serve_fault</code> を選び、<strong>セカンドの有効サーブ・インパクト</strong>を入力します。</p><button>確認結果を保存</button></header><main>""" + "".join(cards) + "</main></form></html>"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                data = page().encode()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if parsed.path.startswith("/clip/"):
                path = clip_dir / Path(parsed.path.removeprefix("/clip/")).name
                if path.is_file():
                    data = path.read_bytes(); self.send_response(200); self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            self.send_error(404)
        def do_POST(self) -> None:
            if urlparse(self.path).path != "/save": self.send_error(404); return
            form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode(), keep_blank_values=True)
            rows = read(csv_path); fields = list(rows[0])
            for row in rows:
                ident = row["window_id"]; status = form.get(f"status__{ident}", [row["review_status"]])[0]
                if status not in STATUSES: self.send_error(400, "invalid status"); return
                offset = form.get(f"offset__{ident}", [""])[0].strip()
                if status in ("complete_valid_rally", "complete_rally_after_first_serve_fault"):
                    try: row["human_rally_start_timestamp_sec"] = f"{float(row['source_start_timestamp_sec']) + float(offset):.6f}"
                    except ValueError: self.send_error(400, f"{ident}: enter a numeric clip time"); return
                else: row["human_rally_start_timestamp_sec"] = ""
                row["review_status"] = status; row["review_note"] = form.get(f"note__{ident}", [""])[0]
            write(csv_path, fields, rows); self.send_response(303); self.send_header("Location", "/"); self.end_headers()
    print(f"Open http://127.0.0.1:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
