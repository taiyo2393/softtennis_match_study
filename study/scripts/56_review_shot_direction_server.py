"""Serve AI-assisted human confirmation of actual post-impact shot direction."""

from __future__ import annotations

import argparse
import csv
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


DIRECTIONS = ("unreviewed", "front_left", "front_right", "uncertain", "excluded")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--port", type=int, default=8782)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    candidates_path = project / "data" / "balltrack" / "direction_proposals" / f"{args.video_id}_shot_direction_candidates.csv"
    review_path = project / "annotations" / f"shot_direction_review_{args.video_id}.csv"
    if not candidates_path.is_file():
        raise SystemExit("Run script 55 before starting this review server.")
    event_by_id = {row["event_id"]: row for row in read(project / "annotations" / "motion_prediction_events.csv")}
    fields = [
        "event_id", "video_id", "impact_source_frame", "target_front_player_id",
        "actual_direction_front_view", "review_status", "direction_source",
        "ai_screen_direction", "ai_front_view_proposal", "ai_proposal_status",
        "review_note", "annotator_id",
    ]
    candidates = read(candidates_path)
    existing = {row["event_id"]: row for row in read(review_path)} if review_path.is_file() else {}
    for candidate in candidates:
        event = event_by_id[candidate["event_id"]]
        record = existing.setdefault(candidate["event_id"], {
            "event_id": candidate["event_id"], "video_id": args.video_id,
            "impact_source_frame": event["impact_source_frame"], "target_front_player_id": event["target_front_player_id"],
            "actual_direction_front_view": "unreviewed", "review_status": "unreviewed", "direction_source": "",
            "ai_screen_direction": candidate["ai_screen_direction"], "ai_front_view_proposal": candidate["ai_front_view_proposal"],
            "ai_proposal_status": candidate["proposal_status"], "review_note": "", "annotator_id": "human_video_review",
        })
        # Refresh the AI-side fields when direction proposals are regenerated.
        # Human labels themselves are preserved.
        record.update({
            "ai_screen_direction": candidate["ai_screen_direction"],
            "ai_front_view_proposal": candidate["ai_front_view_proposal"],
            "ai_proposal_status": candidate["proposal_status"],
        })
        if record.get("actual_direction_front_view") in ("front_left", "front_right"):
            record["direction_source"] = (
                "ai_confirmed" if record["actual_direction_front_view"] == candidate["ai_front_view_proposal"]
                else ("manual_corrected" if candidate["ai_front_view_proposal"] else "manual_no_proposal")
            )
    write(review_path, fields, list(existing.values()))

    def document(page: int, message: str = "") -> str:
        candidates_now = read(candidates_path); page = max(1, min(page, len(candidates_now)))
        candidate = candidates_now[page - 1]
        old = {row["event_id"]: row for row in read(review_path)}.get(candidate["event_id"], {})
        options = "".join(f"<option value='{value}' {'selected' if old.get('actual_direction_front_view') == value else ''}>{value}</option>" for value in DIRECTIONS)
        previous = f"<a href='/?page={page - 1}'>← 前</a>" if page > 1 else ""
        following = f"<a href='/?page={page + 1}'>次 →</a>" if page < len(candidates_now) else ""
        proposal = candidate["ai_front_view_proposal"] or "候補なし"
        screen = candidate["ai_screen_direction"] or "候補なし"
        return f"""<!doctype html><html lang='ja'><meta charset='utf-8'><title>打球方向の確認</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:auto;padding:18px;background:#f3f4f6}}section{{background:#fff;border-radius:10px;padding:16px;margin-bottom:14px}}video{{width:100%;background:#111}}label{{display:block;margin:12px 0}}input,select{{margin-left:8px;font-size:16px;padding:4px}}.note{{background:#fff5df;padding:12px;border-radius:6px;line-height:1.6}}.meta{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}}</style>
<section><h1>実打球方向のAI補助確認（{page}/{len(candidates_now)}）</h1><p>{previous}　{following}</p><p>{escape(message)}</p></section>
<section><h2>{escape(candidate['event_id'])}</h2><div class='meta'><div>人手確定インパクト：原動画 F{candidate['impact_source_frame']}</div><div>AI候補の状態：{escape(candidate['proposal_status'])}</div><div>AIの画面上の移動：{escape(screen)}</div><div>固定主カメラ換算の提案：<b>{escape(proposal)}</b></div></div>
<video controls preload='metadata' src='/preview/{escape(candidate['event_id'])}.mp4'></video>
<p class='note'>緑線はインパクト後だけから選んだ連続BallTrack候補、灰色点は採用しなかった候補です。この試合の固定主カメラでは、監査済みの対応として画面左/右を対象前衛から見た左/右へそのまま記録します。他試合・別画角ではこの対応を再監査してください。ここで記録するのは<b>打球後に確定した実打球方向</b>であり、打球前の前衛動作を作る入力には使いません。</p>
<form method='post' action='/save?page={page}'><label>対象前衛から見た実打球方向 <select name='direction'>{options}</select></label><label>メモ <input name='note' value='{escape(old.get('review_note',''))}' size='48'></label><button>この判定を保存</button></form></section></html>"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return

        def send_video(self, path: Path) -> None:
            size = path.stat().st_size; start, end = 0, size - 1
            raw = self.headers.get("Range", "")
            if raw.startswith("bytes="):
                pieces = raw.removeprefix("bytes=").split("-")
                start = int(pieces[0]) if pieces[0] else 0
                end = min(int(pieces[1]), end) if len(pieces) > 1 and pieces[1] else end
                status = HTTPStatus.PARTIAL_CONTENT
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
                    if not block:
                        break
                    try:
                        self.wfile.write(block)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(block)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                try:
                    page = int(parse_qs(parsed.query).get("page", ["1"])[0])
                except ValueError:
                    page = 1
                data = document(page).encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if parsed.path.startswith("/preview/"):
                name = Path(parsed.path.removeprefix("/preview/")).name
                path = project / "data" / "balltrack" / "direction_proposals" / args.video_id / "previews" / name
                if path.is_file():
                    self.send_video(path); return
            self.send_error(404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/save":
                self.send_error(404); return
            try:
                page = int(parse_qs(urlparse(self.path).query).get("page", ["1"])[0])
            except ValueError:
                page = 1
            candidates_now = read(candidates_path); page = max(1, min(page, len(candidates_now))); candidate = candidates_now[page - 1]
            form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode(), keep_blank_values=True)
            direction = form.get("direction", ["unreviewed"])[0]
            if direction not in DIRECTIONS:
                self.send_error(400); return
            if direction in ("front_left", "front_right"):
                status = "confirmed"
                source = "ai_confirmed" if direction == candidate["ai_front_view_proposal"] else ("manual_corrected" if candidate["ai_front_view_proposal"] else "manual_no_proposal")
            elif direction == "uncertain":
                status, source = "uncertain", "human_video_review"
            elif direction == "excluded":
                status, source = "excluded", "human_video_review"
            else:
                status, source = "unreviewed", ""
            rows = {row["event_id"]: row for row in read(review_path)}
            record = rows[candidate["event_id"]]
            record.update({"actual_direction_front_view": direction, "review_status": status, "direction_source": source, "review_note": form.get("note", [""])[0].strip()})
            rows[candidate["event_id"]] = record; write(review_path, fields, list(rows.values()))
            self.send_response(303); self.send_header("Location", f"/?page={page}"); self.end_headers()

    print(f"Open http://127.0.0.1:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
