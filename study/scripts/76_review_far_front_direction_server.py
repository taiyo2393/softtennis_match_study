"""Serve blinded direction labels for the far-court front-player cohort."""

from __future__ import annotations

import argparse
import csv
import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse


DIRECTIONS = ("unreviewed", "front_left", "front_right", "uncertain", "excluded")
FIELDS = [
    "candidate_id", "source_candidate_id", "video_id", "impact_id", "impact_source_frame",
    "impact_timestamp_sec", "target_front_camera_end", "actual_direction_target_front_view",
    "review_status", "direction_source", "routing_source", "routing_note", "review_note",
    "annotator_id", "label_definition",
]


def read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8794)
    args = parser.parse_args()
    if not args.queue.is_file():
        raise SystemExit(f"Far-front queue not found: {args.queue}")
    candidates = read(args.queue)
    if not candidates:
        raise SystemExit("Far-front queue is empty")
    existing = {row["candidate_id"]: row for row in read(args.review)}
    for candidate in candidates:
        existing.setdefault(candidate["candidate_id"], {
            **{field: candidate.get(field, "") for field in FIELDS},
            "actual_direction_target_front_view": "unreviewed", "review_status": "unreviewed",
            "direction_source": "", "review_note": "", "annotator_id": "human_blind_video_review",
            "label_definition": "far_front_at_impact_pass_side_v1",
        })
    write(args.review, [existing[row["candidate_id"]] for row in candidates])

    def document(page: int, message: str = "") -> str:
        page = max(1, min(page, len(candidates)))
        candidate = candidates[page - 1]
        record = {row["candidate_id"]: row for row in read(args.review)}[candidate["candidate_id"]]
        selected = record.get("actual_direction_target_front_view", "unreviewed")
        options = "".join(f"<option value='{value}' {'selected' if value == selected else ''}>{value}</option>" for value in DIRECTIONS)
        previous = f"<a href='/?page={page - 1}'>← 前</a>" if page > 1 else ""
        following = f"<a href='/?page={page + 1}'>次 →</a>" if page < len(candidates) else ""
        reviewed = sum(row.get("review_status") != "unreviewed" for row in read(args.review))
        return f"""<!doctype html><html lang='ja'><meta charset='utf-8'><title>奥側前衛：実打球方向の確認</title><style>body{{font-family:system-ui,sans-serif;max-width:1280px;margin:auto;padding:18px;background:#f3f4f6}}section{{background:#fff;border-radius:10px;padding:16px;margin-bottom:14px}}video{{display:block;width:100%;background:#111}}select,input,button{{font:inherit;padding:7px;margin:4px}}.note{{background:#fff5df;padding:12px;border-radius:6px;line-height:1.65}}.keys{{background:#e0f2fe;padding:10px;border-radius:6px}}</style><section><h1>実打球方向：奥側前衛視点の人手確認（{page}/{len(candidates)}、完了 {reviewed}/{len(candidates)}）</h1><p>{previous}　{following}</p><p>{esc(message)}</p></section><section><h2>{esc(candidate['candidate_id'])}</h2><p>原動画インパクト F{esc(candidate['impact_source_frame'])} ／ クリップ内 F{esc(candidate['impact_local_frame'])}</p><video id='review-video' controls preload='metadata' src='/clip/{esc(candidate['candidate_id'])}.mp4'></video><p class='note'>対象は<strong>奥側コートの前衛</strong>です。手前後衛が打った球について、インパクト時点の対象前衛から見てボールがどちら側を通るかを記録します。画面上の左右とは一致しません。対象前衛の左側なら <code>front_left</code>、右側なら <code>front_right</code>。ロブ、陣形不成立、対象前衛を定められない場面は <code>excluded</code>、左右が映像から定められない場面は <code>uncertain</code> です。</p><p class='keys'><strong>A</strong>：front_left　<strong>D</strong>：front_right　<strong>U</strong>：uncertain　<strong>X</strong>：excluded　<strong>Enter</strong>：保存</p><form id='review-form' method='post' action='/save?page={page}'><label>実打球方向（奥側対象前衛視点）<select id='direction' name='direction'>{options}</select></label><label>メモ <input id='note' name='note' value='{esc(record.get('review_note',''))}' size='46'></label><button>この判定を保存して次へ</button></form></section><script>const direction=document.getElementById('direction');const form=document.getElementById('review-form');document.addEventListener('keydown',(event)=>{{const tag=document.activeElement&&document.activeElement.tagName;if(['INPUT','SELECT','BUTTON','VIDEO'].includes(tag))return;const key=event.key.toLowerCase();if(key==='a'){{direction.value='front_left';event.preventDefault();}}if(key==='d'){{direction.value='front_right';event.preventDefault();}}if(key==='u'){{direction.value='uncertain';event.preventDefault();}}if(key==='x'){{direction.value='excluded';event.preventDefault();}}if(event.key==='Enter'){{event.preventDefault();form.requestSubmit();}}}});</script></html>"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return

        def send_video(self, path: Path) -> None:
            size = path.stat().st_size; start, end = 0, size - 1
            raw = self.headers.get("Range", "")
            if raw.startswith("bytes="):
                begin, _, finish = raw.removeprefix("bytes=").partition("-")
                start = int(begin) if begin else 0; end = min(int(finish), end) if finish else end
                status = HTTPStatus.PARTIAL_CONTENT
            else:
                status = HTTPStatus.OK
            self.send_response(status); self.send_header("Content-Type", "video/mp4"); self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(end - start + 1))
            if status == HTTPStatus.PARTIAL_CONTENT: self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start); remaining = end - start + 1
                while remaining:
                    block = handle.read(min(64 * 1024, remaining))
                    if not block: return
                    try: self.wfile.write(block)
                    except (BrokenPipeError, ConnectionResetError): return
                    remaining -= len(block)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                try: page = int(parse_qs(parsed.query).get("page", ["1"])[0])
                except ValueError: page = 1
                data = document(page, parse_qs(parsed.query).get("message", [""])[0]).encode("utf-8")
                self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if parsed.path.startswith("/clip/"):
                name = Path(parsed.path.removeprefix("/clip/")).name
                candidate = next((item for item in candidates if item["candidate_id"] == name.removesuffix(".mp4")), None)
                path = Path(candidate["source_clip_path"]) if candidate else None
                if path and path.is_file(): self.send_video(path); return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/save": self.send_error(HTTPStatus.NOT_FOUND); return
            try: page = int(parse_qs(urlparse(self.path).query).get("page", ["1"])[0])
            except ValueError: page = 1
            page = max(1, min(page, len(candidates))); candidate = candidates[page - 1]
            form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"), keep_blank_values=True)
            direction = form.get("direction", ["unreviewed"])[0]
            if direction not in DIRECTIONS: self.send_error(HTTPStatus.BAD_REQUEST, "Invalid direction"); return
            status = "confirmed" if direction in ("front_left", "front_right") else direction
            rows = {row["candidate_id"]: row for row in read(args.review)}
            record = rows[candidate["candidate_id"]]
            record.update({"actual_direction_target_front_view": direction, "review_status": status, "direction_source": "blind_human_video_review" if status == "confirmed" else "human_video_review", "review_note": form.get("note", [""])[0].strip()})
            rows[candidate["candidate_id"]] = record; write(args.review, [rows[item["candidate_id"]] for item in candidates])
            location = "/?" + urlencode({"page": min(page + 1, len(candidates)), "message": "保存しました。"})
            self.send_response(HTTPStatus.SEE_OTHER); self.send_header("Location", location); self.end_headers()

    print(f"Open http://127.0.0.1:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
