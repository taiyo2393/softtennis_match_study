"""Serve short role-track overlays for event-level human quality review."""

from __future__ import annotations

import argparse
import csv
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


OPTIONS = ("unreviewed", "usable", "id_switch", "target_missing", "role_assignment_error", "unusable")


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
    parser.add_argument("--port", type=int, default=8779)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    track_dir = project / "data" / "tracking" / "motion_prediction_events"
    summary_path = track_dir / f"{args.video_id}_event_role_track_summary.csv"
    review_path = project / "annotations" / "motion_prediction_track_review.csv"
    if not summary_path.is_file():
        raise SystemExit("Run script 47 before starting the track review server.")
    fields = ["event_id", "video_id", "tracking_status", "review_note"]
    summaries_at_start = read(summary_path)
    if not review_path.exists():
        write(review_path, fields, [{"event_id": row["event_id"], "video_id": args.video_id, "tracking_status": "unreviewed", "review_note": ""} for row in summaries_at_start])
    else:
        # New included events can be added after an earlier review pass.  Keep
        # existing human decisions and append only the missing review rows.
        existing = {row["event_id"]: row for row in read(review_path)}
        changed = False
        for summary in summaries_at_start:
            if summary["event_id"] not in existing:
                existing[summary["event_id"]] = {"event_id": summary["event_id"], "video_id": args.video_id, "tracking_status": "unreviewed", "review_note": ""}
                changed = True
        if changed:
            write(review_path, fields, list(existing.values()))

    def page(number: int, message: str = "") -> str:
        summaries = read(summary_path); number = max(1, min(number, len(summaries)))
        summary = summaries[number - 1]
        old = {row["event_id"]: row for row in read(review_path)}.get(summary["event_id"], {})
        options = "".join(f"<option value='{item}' {'selected' if old.get('tracking_status') == item else ''}>{item}</option>" for item in OPTIONS)
        previous = f"<a href='/?page={number - 1}'>← 前</a>" if number > 1 else ""
        next_page = f"<a href='/?page={number + 1}'>次 →</a>" if number < len(summaries) else ""
        return f"""<!doctype html><html lang='ja'><meta charset='utf-8'><title>前衛軌跡の監査</title><style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:auto;padding:18px;background:#f3f4f6}}section{{background:#fff;border-radius:10px;padding:16px;margin-bottom:14px}}video{{width:100%;background:#111}}label{{display:block;margin:12px 0}}input,select{{margin-left:8px;font-size:16px;padding:4px}}.warn{{background:#fff5df;padding:10px;border-radius:6px}}</style><section><h1>前衛軌跡の監査（{number}/{len(summaries)}）</h1><p>{previous}　{next_page}</p><p>{message}</p></section><section><h2>{summary['event_id']}</h2><p>インパクト時の役割割当：{summary['role_assignment_status']}<br>対象前衛：{summary['target_front_role']} / track ID {summary['target_front_track_id']}<br>対象の観測率：{float(summary['target_front_observation_rate']):.1%}</p><video controls preload='metadata' src='/preview/{summary['event_id']}.mp4'></video><form method='post' action='/save?page={number}'><label>追跡の判定 <select name='status'>{options}</select></label><label>メモ <input name='note' value='{old.get('review_note','')}'></label><p class='warn'>青緑の <code>near_front</code> が手前前衛を全窓で追えている場合だけ usable。ID交替、対象消失、役割の取り違えはそれぞれ該当する理由を選ぶ。</p><button>この判定を保存</button></form></section></html>"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None: return
        def send_video(self, path: Path) -> None:
            size = path.stat().st_size; start, end = 0, size - 1
            range_header = self.headers.get("Range", "")
            if range_header.startswith("bytes="):
                text = range_header.removeprefix("bytes=").split("-")
                start = int(text[0]) if text[0] else 0; end = min(int(text[1]), end) if len(text) > 1 and text[1] else end; status = HTTPStatus.PARTIAL_CONTENT
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
                preview = track_dir / "previews" / Path(parsed.path.removeprefix("/preview/")).name
                if preview.is_file(): self.send_video(preview); return
            self.send_error(404)
        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/save": self.send_error(404); return
            try: number = int(parse_qs(parsed.query).get("page", ["1"])[0])
            except ValueError: number = 1
            summaries = read(summary_path); number = max(1, min(number, len(summaries))); event_id = summaries[number - 1]["event_id"]
            form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode(), keep_blank_values=True); status = form.get("status", ["unreviewed"])[0]
            if status not in OPTIONS: self.send_error(400); return
            rows = {row["event_id"]: row for row in read(review_path)}; rows[event_id] = {"event_id": event_id, "video_id": args.video_id, "tracking_status": status, "review_note": form.get("note", [""])[0].strip()}; write(review_path, fields, list(rows.values()))
            self.send_response(303); self.send_header("Location", f"/?page={number}"); self.end_headers()

    print(f"Open http://127.0.0.1:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
