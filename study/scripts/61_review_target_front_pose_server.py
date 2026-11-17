"""Serve visual audit of target-front ankle-pose candidates."""

from __future__ import annotations

import argparse
import csv
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


OPTIONS = ("unreviewed", "usable", "wrong_person", "ankles_wrong", "ankles_missing", "unusable")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--pose-stem", default="match1_target_front_pose_yolo11m")
    parser.add_argument(
        "--review-prefix", default=None,
        help="Separate annotation prefix for this pose-review batch.",
    )
    parser.add_argument("--port", type=int, default=8783)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    pose_path = project / "data" / "pose" / "target_front" / f"{args.pose_stem}.csv"
    preview_dir = project / "data" / "pose" / "target_front" / "previews"
    review_path = (
        project / "annotations" / f"{args.review_prefix}_pose_review.csv"
        if args.review_prefix
        else project / "annotations" / f"target_front_pose_review_{args.video_id}_yolo11m.csv"
    )
    if not pose_path.is_file():
        raise SystemExit("Run script 59 before starting this review server.")
    event_rows: dict[str, list[dict[str, str]]] = {}
    for row in read(pose_path):
        event_rows.setdefault(row["event_id"], []).append(row)
    fields = ["event_id", "video_id", "pose_model", "pose_status", "review_note"]
    existing = {row["event_id"]: row for row in read(review_path)} if review_path.is_file() else {}
    for event_id in event_rows:
        existing.setdefault(event_id, {"event_id": event_id, "video_id": args.video_id, "pose_model": "yolo11m-pose.pt", "pose_status": "unreviewed", "review_note": ""})
    write(review_path, fields, list(existing.values()))

    def page(number: int, message: str = "") -> str:
        event_ids = list(event_rows); number = max(1, min(number, len(event_ids))); event_id = event_ids[number - 1]
        review = {row["event_id"]: row for row in read(review_path)}[event_id]
        rows = event_rows[event_id]; paired = sum(row["ankle_pair_visible"] == "1" for row in rows)
        choices = "".join(f"<option value='{item}' {'selected' if review['pose_status'] == item else ''}>{item}</option>" for item in OPTIONS)
        previous = f"<a href='/?page={number - 1}'>← 前</a>" if number > 1 else ""; following = f"<a href='/?page={number + 1}'>次 →</a>" if number < len(event_ids) else ""
        return f"""<!doctype html><html lang='ja'><meta charset='utf-8'><title>対象前衛Pose監査</title><style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:auto;padding:18px;background:#f3f4f6}}section{{background:#fff;border-radius:10px;padding:16px;margin-bottom:14px}}video{{width:100%;background:#111}}label{{display:block;margin:12px 0}}input,select,button{{margin-left:8px;font-size:16px;padding:4px}}.note{{background:#fff5df;padding:12px;border-radius:6px;line-height:1.6}}.keys{{background:#e0f2fe;padding:10px;border-radius:6px}}</style><section><h1>対象前衛の足首Pose監査（{number}/{len(event_ids)}）</h1><p>{previous}　{following}</p><p>{message}</p></section><section><h2>{event_id}</h2><p>足首ペア取得：{paired}/{len(rows)} フレーム</p><video controls preload='metadata' src='/preview/{event_id}.mp4'></video><p class='keys'><strong>Y</strong>：usable　<strong>W</strong>：wrong_person　<strong>A</strong>：ankles_wrong　<strong>M</strong>：ankles_missing　<strong>X</strong>：unusable　<strong>Enter</strong>：保存して次へ</p><p class='note'>黄枠が既に監査済みの対象前衛、紫丸が左右の足首、緑線と緑点が足首中点です。インパクト前の全区間で緑点が対象前衛の足元に追従していれば <code>usable</code>。別人の骨格なら <code>wrong_person</code>、足元ではない位置なら <code>ankles_wrong</code>、足首の欠落が解析窓を妨げるなら <code>ankles_missing</code> を選びます。</p><form id='review-form' method='post' action='/save?page={number}'><label>Poseの判定 <select id='status' name='status'>{choices}</select></label><label>メモ <input id='note' name='note' value='{review['review_note']}' size='48'></label><button>この判定を保存して次へ</button></form></section><script>const status=document.getElementById('status');const form=document.getElementById('review-form');document.addEventListener('keydown',(event)=>{{const tag=document.activeElement&&document.activeElement.tagName;if(['INPUT','SELECT','BUTTON','VIDEO'].includes(tag))return;const key=event.key.toLowerCase();const keys={{y:'usable',w:'wrong_person',a:'ankles_wrong',m:'ankles_missing',x:'unusable'}};if(keys[key]){{status.value=keys[key];event.preventDefault();}}if(event.key==='Enter'){{event.preventDefault();form.requestSubmit();}}}});</script></html>"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return

        def send_video(self, path: Path) -> None:
            size = path.stat().st_size; start, end = 0, size - 1; raw = self.headers.get("Range", "")
            if raw.startswith("bytes="):
                pieces = raw.removeprefix("bytes=").split("-"); start = int(pieces[0]) if pieces[0] else 0; end = min(int(pieces[1]), end) if len(pieces) > 1 and pieces[1] else end; status = HTTPStatus.PARTIAL_CONTENT
            else: status = HTTPStatus.OK
            self.send_response(status); self.send_header("Content-Type", "video/mp4"); self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(end - start + 1))
            if status == HTTPStatus.PARTIAL_CONTENT: self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as file:
                file.seek(start); remaining = end - start + 1
                while remaining:
                    block = file.read(min(64 * 1024, remaining))
                    if not block: break
                    try: self.wfile.write(block)
                    except (BrokenPipeError, ConnectionResetError): return
                    remaining -= len(block)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                try: number = int(parse_qs(parsed.query).get("page", ["1"])[0])
                except ValueError: number = 1
                data = page(number).encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if parsed.path.startswith("/preview/"):
                path = preview_dir / Path(parsed.path.removeprefix("/preview/")).name
                if path.is_file(): self.send_video(path); return
            self.send_error(404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/save": self.send_error(404); return
            try: number = int(parse_qs(urlparse(self.path).query).get("page", ["1"])[0])
            except ValueError: number = 1
            event_ids = list(event_rows); number = max(1, min(number, len(event_ids))); event_id = event_ids[number - 1]
            form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode(), keep_blank_values=True); status = form.get("status", ["unreviewed"])[0]
            if status not in OPTIONS: self.send_error(400); return
            rows = {row["event_id"]: row for row in read(review_path)}; rows[event_id] = {"event_id": event_id, "video_id": args.video_id, "pose_model": "yolo11m-pose.pt", "pose_status": status, "review_note": form.get("note", [""])[0].strip()}; write(review_path, fields, list(rows.values()))
            next_number = min(number + 1, len(event_ids))
            self.send_response(303); self.send_header("Location", f"/?page={next_number}"); self.end_headers()

    print(f"Open http://127.0.0.1:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
