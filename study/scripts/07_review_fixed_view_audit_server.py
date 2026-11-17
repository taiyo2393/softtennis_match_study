"""Serve a blinded local UI for auditing fixed-camera classifications."""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


OPTIONS = ("unreviewed", "main_camera", "other_camera", "unknown")
FIELDS = ["audit_id", "video_id", "timestamp_sec", "auto_view", "background_difference", "image_filename", "human_view", "review_note", "reviewed_at"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--port", type=int, default=8767)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def render(rows: list[dict[str, str]]) -> str:
    cards = []
    for row in rows:
        selected = row.get("human_view", "unreviewed")
        options = "".join(f"<option value='{item}' {'selected' if item == selected else ''}>{item}</option>" for item in OPTIONS)
        cards.append(f"""<article data-id='{html.escape(row['audit_id'], quote=True)}'>
<h2>{html.escape(row['audit_id'])}</h2><img src='/images/{html.escape(row['image_filename'], quote=True)}' alt='監査画像'>
<label>この画像は固定・俯瞰の主カメラか？<select class='view'>{options}</select></label>
<label>メモ<input class='note' value='{html.escape(row.get('review_note', ''), quote=True)}'></label></article>""")
    return """<!doctype html><html lang='ja'><meta charset='utf-8'><title>固定画角の監査</title><style>
body{font-family:system-ui,sans-serif;margin:20px;background:#f3f4f6}header{position:sticky;top:0;z-index:1;background:#fff;padding:12px;border-bottom:1px solid #ddd}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:14px;margin-top:16px}article{background:#fff;border-radius:8px;padding:12px}h1{margin:0;font-size:22px}h2{font-size:16px;margin:0 0 8px}img{width:100%;border:1px solid #ddd}label{display:block;margin-top:8px}select{margin-left:8px}.note{margin-left:8px;width:70%}button{font-size:16px;padding:8px 16px}#status{margin-left:12px}
</style><header><h1>固定主カメラ判定の監査（20画像）</h1><p>固定・俯瞰のコート画面なら <code>main_camera</code>、それ以外なら <code>other_camera</code>。自動判定は表示していません。</p><button onclick='save()'>監査結果を保存</button><span id='status'></span></header><main>""" + "\n".join(cards) + """</main><script>
async function save(){const decisions=[...document.querySelectorAll('article')].map(card=>({audit_id:card.dataset.id,human_view:card.querySelector('.view').value,review_note:card.querySelector('.note').value}));const response=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({decisions})});const data=await response.json();document.querySelector('#status').textContent=data.message||data.error;}
</script></html>"""


def main() -> int:
    args = parse_args()
    project = Path(__file__).resolve().parents[1]
    segment_dir = project / "data" / "segments"
    csv_path = segment_dir / f"{args.video_id}_fixed_view_audit.csv"
    image_dir = segment_dir / "audit" / args.video_id

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *values: object) -> None:
            return
        def text(self, status: HTTPStatus, body: str, content_type: str) -> None:
            data = body.encode("utf-8"); self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.text(HTTPStatus.OK, render(read_rows(csv_path)), "text/html; charset=utf-8"); return
            if parsed.path.startswith("/images/"):
                asset = image_dir / Path(unquote(parsed.path.removeprefix("/images/"))).name
                if asset.is_file():
                    data = asset.read_bytes(); self.send_response(HTTPStatus.OK); self.send_header("Content-Type", mimetypes.guess_type(asset.name)[0] or "application/octet-stream"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            self.send_error(HTTPStatus.NOT_FOUND)
        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/save": self.send_error(HTTPStatus.NOT_FOUND); return
            try:
                size = int(self.headers["Content-Length"]); decisions = {item["audit_id"]: item for item in json.loads(self.rfile.read(size))["decisions"]}; rows = read_rows(csv_path)
                if set(decisions) != {row["audit_id"] for row in rows}: raise ValueError("audit IDs do not match")
                now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
                for row in rows:
                    item = decisions[row["audit_id"]]; view = item.get("human_view", "unreviewed")
                    if view not in OPTIONS: raise ValueError("invalid view")
                    row["human_view"], row["review_note"], row["reviewed_at"] = view, str(item.get("review_note", "")), now
                write_rows(csv_path, rows); self.text(HTTPStatus.OK, json.dumps({"message": f"{len(rows)}件を保存しました"}), "application/json; charset=utf-8")
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self.text(HTTPStatus.BAD_REQUEST, json.dumps({"error": str(error)}), "application/json; charset=utf-8")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Open http://127.0.0.1:{args.port}/")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
