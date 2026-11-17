"""Serve a local browser UI for classifying the intervals before and after each cut."""

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


STATUS_OPTIONS = ("unreviewed", "main_camera", "other_camera", "unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["video_id", "candidate_id", "timestamp_sec", "scene_score", "before_view", "after_view", "review_note", "reviewed_at"]
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def page(rows: list[dict[str, str]], decisions: dict[str, dict[str, str]], image_dir: Path) -> str:
    cards: list[str] = []
    for index, row in enumerate(rows, start=1):
        candidate_id = row["candidate_id"]
        decision = decisions.get(candidate_id, {})
        timestamp = float(row["timestamp_sec"])
        image_name = f"{index:03d}_{candidate_id}_{timestamp:.3f}s.jpg"
        before = decision.get("before_view", "unreviewed")
        after = decision.get("after_view", "unreviewed")
        before_options = "".join(
            f"<option value='{state}' {'selected' if state == before else ''}>{state}</option>"
            for state in STATUS_OPTIONS
        )
        after_options = "".join(
            f"<option value='{state}' {'selected' if state == after else ''}>{state}</option>"
            for state in STATUS_OPTIONS
        )
        note = html.escape(decision.get("review_note", ""), quote=True)
        cards.append(f"""
        <article data-id='{html.escape(candidate_id, quote=True)}'>
          <h2>{html.escape(candidate_id)}</h2>
          <p>{timestamp:.3f} s / scene score {html.escape(row['scene_score'])}</p>
          <div class='legend'><span>左：切替前</span><span>中央：切替中（判定不要）</span><span>右：切替後</span></div>
          <img src='/images/{html.escape(image_name, quote=True)}' alt='{html.escape(candidate_id)}'>
          <div class='decision-grid'><label>左の安定区間<br><select class='before'>{before_options}</select></label>
          <label>右の安定区間<br><select class='after'>{after_options}</select></label></div>
          <label>メモ <input class='note' value='{note}' placeholder='例: 選手アップから主カメラ'></label>
        </article>""")
    return """<!doctype html><html lang='ja'><meta charset='utf-8'>
<title>主カメラ区間レビュー</title><style>
body{font-family:system-ui,sans-serif;margin:20px;background:#f3f4f6}header{position:sticky;top:0;background:#fff;padding:12px;border-bottom:1px solid #ddd;z-index:1}button{font-size:16px;padding:8px 16px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:14px;margin-top:16px}article{background:#fff;padding:12px;border-radius:8px}h1{font-size:22px;margin:0}h2{font-size:16px;margin:0}p{margin:6px 0}img{width:100%;border:1px solid #ddd}.legend{display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;font-size:13px;margin:8px 0}.legend span{padding:4px;text-align:center;background:#e5e7eb}.legend span:nth-child(2){background:#fde68a}.decision-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.decision-grid label{background:#eff6ff;padding:8px}.decision-grid select{margin-top:4px;width:100%}label{display:block;margin-top:8px}.note{width:88%;margin-left:8px}#result{margin-left:12px}
</style><header><h1>カット候補レビュー</h1><p><code>main_camera</code> は固定・俯瞰のコート画面。中央の残像・フェードは判定せず、解析では切替点の前後1秒を自動除外します。</p><button onclick='save()'>判定を保存</button><span id='result'></span></header><main>""" + "\n".join(cards) + """</main><script>
async function save(){const decisions=[...document.querySelectorAll('article')].map(card=>({candidate_id:card.dataset.id,before_view:card.querySelector('.before').value,after_view:card.querySelector('.after').value,review_note:card.querySelector('.note').value}));const response=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({decisions})});const data=await response.json();document.querySelector('#result').textContent=data.message||data.error;}
</script></html>"""


def main() -> int:
    args = parse_args()
    project = Path(__file__).resolve().parents[1]
    segment_dir = project / "data" / "segments"
    candidates_path = segment_dir / f"{args.video_id}_cut_candidates.csv"
    reviews_path = segment_dir / f"{args.video_id}_transition_review.csv"
    image_dir = segment_dir / "review" / args.video_id
    candidates = read_csv(candidates_path)
    if not candidates or not image_dir.is_dir():
        raise FileNotFoundError("Candidates CSV or review images were not found. Run scripts 01 and 02 first.")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *values: object) -> None:
            return

        def send_text(self, status: HTTPStatus, content: str, content_type: str) -> None:
            data = content.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                decisions = {row["candidate_id"]: row for row in read_csv(reviews_path)}
                self.send_text(HTTPStatus.OK, page(candidates, decisions, image_dir), "text/html; charset=utf-8")
                return
            if parsed.path.startswith("/images/"):
                name = Path(unquote(parsed.path.removeprefix("/images/"))).name
                asset = image_dir / name
                if asset.is_file():
                    data = asset.read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", mimetypes.guess_type(asset.name)[0] or "application/octet-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/save":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers["Content-Length"])
                payload = json.loads(self.rfile.read(length))
                requested = {item["candidate_id"]: item for item in payload["decisions"]}
                valid_ids = {row["candidate_id"] for row in candidates}
                if set(requested) != valid_ids:
                    raise ValueError("candidate IDs do not match the review set")
                now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
                result = []
                for row in candidates:
                    item = requested[row["candidate_id"]]
                    before = item.get("before_view", "unreviewed")
                    after = item.get("after_view", "unreviewed")
                    if before not in STATUS_OPTIONS or after not in STATUS_OPTIONS:
                        raise ValueError("invalid view status")
                    result.append({"video_id": args.video_id, "candidate_id": row["candidate_id"], "timestamp_sec": row["timestamp_sec"], "scene_score": row["scene_score"], "before_view": before, "after_view": after, "review_note": str(item.get("review_note", "")), "reviewed_at": now})
                write_csv(reviews_path, result)
                self.send_text(HTTPStatus.OK, json.dumps({"message": f"{len(result)}件を保存しました"}), "application/json; charset=utf-8")
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                self.send_text(HTTPStatus.BAD_REQUEST, json.dumps({"error": str(error)}), "application/json; charset=utf-8")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Open http://127.0.0.1:{args.port}/ in a browser. Ctrl+C stops the server.")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
