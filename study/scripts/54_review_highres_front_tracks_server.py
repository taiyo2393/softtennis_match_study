"""Serve high-resolution, one-target front-track overlays for human audit."""

from __future__ import annotations

import argparse
import csv
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


OPTIONS = ("unreviewed", "usable", "id_switch", "target_missing", "tracking_error", "unusable")


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
    parser.add_argument("--model-stem", default="yolo11m")
    parser.add_argument(
        "--output-prefix", default=None,
        help="Read a separately named tracking batch made by script 53.",
    )
    parser.add_argument("--target-front-camera-end", choices=("near", "far"), default="near")
    parser.add_argument("--port", type=int, default=8781)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    track_dir = project / "data" / "tracking" / "front_target_highres"
    if args.output_prefix:
        summary_path = track_dir / f"{args.output_prefix}_summary_{args.model_stem}.csv"
        review_path = project / "annotations" / f"{args.output_prefix}_track_review_{args.model_stem}.csv"
    else:
        summary_path = track_dir / f"{args.video_id}_target_front_summary_{args.model_stem}.csv"
        review_path = project / "annotations" / f"highres_front_track_review_{args.video_id}_{args.model_stem}.csv"
    if not summary_path.is_file():
        raise SystemExit("Run script 53 before starting this review server.")
    fields = ["event_id", "video_id", "model", "tracking_status", "review_note"]
    summaries = read(summary_path)
    existing = {row["event_id"]: row for row in read(review_path)} if review_path.is_file() else {}
    for summary in summaries:
        existing.setdefault(summary["event_id"], {"event_id": summary["event_id"], "video_id": args.video_id, "model": summary["model"], "tracking_status": "unreviewed", "review_note": ""})
    write(review_path, fields, list(existing.values()))

    def document(number: int, message: str = "") -> str:
        summaries_now = read(summary_path); number = max(1, min(number, len(summaries_now)))
        summary = summaries_now[number - 1]
        old = {row["event_id"]: row for row in read(review_path)}.get(summary["event_id"], {})
        choices = "".join(f"<option value='{item}' {'selected' if old.get('tracking_status') == item else ''}>{item}</option>" for item in OPTIONS)
        previous = f"<a href='/?page={number - 1}'>← 前</a>" if number > 1 else ""
        following = f"<a href='/?page={number + 1}'>次 →</a>" if number < len(summaries_now) else ""
        target_text = "奥側コートの対象前衛" if args.target_front_camera_end == "far" else "手前側の対象前衛"
        return f"""<!doctype html><html lang='ja'><meta charset='utf-8'><title>高解像度・対象前衛追跡の監査</title><style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:auto;padding:18px;background:#f3f4f6}}section{{background:#fff;border-radius:10px;padding:16px;margin-bottom:14px}}video{{width:100%;background:#111}}label{{display:block;margin:12px 0}}input,select,button{{margin-left:8px;font-size:16px;padding:4px}}.warn{{background:#fff5df;padding:10px;border-radius:6px}}.keys{{background:#e0f2fe;padding:10px;border-radius:6px}}</style><section><h1>高解像度・対象前衛追跡の監査（{number}/{len(summaries_now)}）</h1><p>{previous}　{following}</p><p>{message}</p></section><section><h2>{summary['event_id']}</h2><p>モデル：{summary['model']}<br>追跡窓：原動画 F{summary['start_frame']}〜F{summary['end_frame']}（インパクト F{summary['impact_source_frame']}）<br>AI観測率：{float(summary['observation_rate']):.1%}</p><video controls preload='metadata' src='/preview/{summary['event_id']}.mp4'></video><p class='keys'><strong>Y</strong>：usable　<strong>I</strong>：id_switch　<strong>M</strong>：target_missing　<strong>E</strong>：tracking_error　<strong>X</strong>：unusable　<strong>Enter</strong>：保存して次へ</p><form id='review-form' method='post' action='/save?page={number}'><label>追跡の判定 <select id='status' name='status'>{choices}</select></label><label>メモ <input id='note' name='note' value='{old.get('review_note','')}'></label><p class='warn'>黄枠 <code>far_front</code> が、インパクト前から<strong>{target_text}</strong>だけを連続して追えているときだけ <code>usable</code>。灰枠は他の人物候補です。観測率100%でも、別の人物へ枠が移れば <code>id_switch</code> または <code>tracking_error</code> を選んでください。</p><button>この判定を保存して次へ</button></form></section><script>const status=document.getElementById('status');const form=document.getElementById('review-form');document.addEventListener('keydown',(event)=>{{const tag=document.activeElement&&document.activeElement.tagName;if(['INPUT','SELECT','BUTTON','VIDEO'].includes(tag))return;const key=event.key.toLowerCase();const keys={{y:'usable',i:'id_switch',m:'target_missing',e:'tracking_error',x:'unusable'}};if(keys[key]){{status.value=keys[key];event.preventDefault();}}if(event.key==='Enter'){{event.preventDefault();form.requestSubmit();}}}});</script></html>"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return

        def send_video(self, path: Path) -> None:
            size = path.stat().st_size; start, end = 0, size - 1
            raw = self.headers.get("Range", "")
            if raw.startswith("bytes="):
                parts = raw.removeprefix("bytes=").split("-"); start = int(parts[0]) if parts[0] else 0; end = min(int(parts[1]), end) if len(parts) > 1 and parts[1] else end; status = HTTPStatus.PARTIAL_CONTENT
            else:
                status = HTTPStatus.OK
            self.send_response(status); self.send_header("Content-Type", "video/mp4"); self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(end - start + 1))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
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
                data = document(number).encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if parsed.path.startswith("/preview/"):
                path = track_dir / "previews" / Path(parsed.path.removeprefix("/preview/")).name
                if path.is_file(): self.send_video(path); return
            self.send_error(404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/save": self.send_error(404); return
            try: number = int(parse_qs(urlparse(self.path).query).get("page", ["1"])[0])
            except ValueError: number = 1
            summaries_now = read(summary_path); number = max(1, min(number, len(summaries_now))); summary = summaries_now[number - 1]
            form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode(), keep_blank_values=True); status = form.get("status", ["unreviewed"])[0]
            if status not in OPTIONS: self.send_error(400); return
            rows = {row["event_id"]: row for row in read(review_path)}
            rows[summary["event_id"]] = {"event_id": summary["event_id"], "video_id": args.video_id, "model": summary["model"], "tracking_status": status, "review_note": form.get("note", [""])[0].strip()}
            write(review_path, fields, list(rows.values()))
            next_number = min(number + 1, len(summaries_now))
            self.send_response(303); self.send_header("Location", f"/?page={next_number}"); self.end_headers()

    print(f"Open http://127.0.0.1:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
