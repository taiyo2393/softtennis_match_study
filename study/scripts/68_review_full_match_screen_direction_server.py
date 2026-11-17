"""Serve blinded human target-front-view direction labels for the pilot.

The AI candidate is deliberately hidden during this 30-event calibration.
Showing it while constructing the reference label would inflate apparent
agreement.  The reviewer labels the side on which the ball passes the target
front player at impact, using the plain post-impact clip.  Camera-coordinate
BallTrack trajectories are kept only as input diagnostics: they are not the
player-relative outcome label.
"""

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
    "candidate_id", "pilot_id", "impact_id", "video_id", "impact_source_frame",
    "impact_timestamp_sec", "actual_direction_front_view", "review_status",
    "direction_source", "ai_screen_direction", "ai_proposal_status", "review_note",
    "annotator_id", "label_definition",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def escaped(value: str) -> str:
    return html.escape(value, quote=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--pilot-id", default="direction_pilot_01")
    parser.add_argument("--port", type=int, default=8786)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    candidate_path = project / "data" / "balltrack" / "direction_proposals" / f"full_match_{args.pilot_id}" / args.video_id / "screen_direction_candidates.csv"
    queue_path = project / "data" / "segments" / f"{args.video_id}_full_match_{args.pilot_id}_queue.csv"
    clip_dir = project / "data" / "segments" / f"full_match_{args.pilot_id}" / args.video_id
    # Keep the earlier screen-coordinate trial CSV untouched.  It was a
    # different definition and must not be silently reinterpreted.
    review_path = project / "annotations" / f"{args.video_id}_full_match_{args.pilot_id}_front_direction_blind_review.csv"
    queue_by_id = {row["candidate_id"]: row for row in read_csv(queue_path)}
    if not queue_by_id:
        raise SystemExit("Direction-review queue is empty or missing.")
    candidates = read_csv(candidate_path)
    # A human direction label is still valid when BallTrack has not run.  This
    # allows a complete-match expansion to be reviewed without fabricating an
    # AI proposal; BallTrack can be added later for an explicitly separate
    # candidate-quality evaluation.
    if not candidates:
        candidates = [
            {
                "candidate_id": row["candidate_id"], "pilot_id": args.pilot_id,
                "video_id": row["video_id"], "impact_id": row["impact_id"],
                "impact_source_frame": row["impact_source_frame"],
                "impact_timestamp_sec": row["impact_timestamp_sec"],
                "ready_segment_id": row["ready_segment_id"], "play_window_id": row["play_window_id"],
                "ai_screen_direction": "", "proposal_status": "not_run",
            }
            for row in queue_by_id.values()
        ]
        candidates.sort(key=lambda row: int(row["impact_source_frame"]))
    if set(row["candidate_id"] for row in candidates) - set(queue_by_id):
        raise SystemExit("Candidate/queue IDs do not match.")

    existing = {row["candidate_id"]: row for row in read_csv(review_path)}
    for candidate in candidates:
        record = existing.setdefault(candidate["candidate_id"], {
            "candidate_id": candidate["candidate_id"], "pilot_id": args.pilot_id,
            "impact_id": candidate["impact_id"], "video_id": args.video_id,
            "impact_source_frame": candidate["impact_source_frame"],
            "impact_timestamp_sec": candidate["impact_timestamp_sec"],
            "actual_direction_front_view": "unreviewed", "review_status": "unreviewed",
            "direction_source": "", "ai_screen_direction": candidate["ai_screen_direction"],
            "ai_proposal_status": candidate["proposal_status"], "review_note": "",
            "annotator_id": "human_blind_video_review",
            "label_definition": "target_front_at_impact_pass_side_v1",
        })
        # Candidate fields are frozen by script 67; do not change any human label.
        record["ai_screen_direction"] = candidate["ai_screen_direction"]
        record["ai_proposal_status"] = candidate["proposal_status"]
    write_csv(review_path, [existing[row["candidate_id"]] for row in candidates])

    def document(page: int, message: str = "") -> str:
        page = max(1, min(page, len(candidates)))
        candidate = candidates[page - 1]
        record = {row["candidate_id"]: row for row in read_csv(review_path)}[candidate["candidate_id"]]
        queue = queue_by_id[candidate["candidate_id"]]
        selected = record.get("actual_direction_front_view", "unreviewed")
        options = "".join(
            f"<option value='{item}' {'selected' if item == selected else ''}>{item}</option>"
            for item in DIRECTIONS
        )
        previous = f"<a href='/?page={page - 1}'>← 前</a>" if page > 1 else ""
        following = f"<a href='/?page={page + 1}'>次 →</a>" if page < len(candidates) else ""
        reviewed = sum(row.get("review_status") != "unreviewed" for row in read_csv(review_path))
        return f"""<!doctype html><html lang='ja'><meta charset='utf-8'><title>打球方向：前衛視点ブラインド確認</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1280px;margin:auto;padding:18px;background:#f3f4f6;color:#111827}}section{{background:#fff;border-radius:10px;padding:16px;margin-bottom:14px}}video{{display:block;width:100%;background:#111}}select,input,button{{font:inherit;padding:7px;margin:4px}}.note{{background:#fff5df;padding:12px;border-radius:6px;line-height:1.65}}.meta{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}}.keys{{background:#e0f2fe;padding:10px;border-radius:6px}}.save{{background:#bbf7d0;border:2px solid #15803d;font-weight:700}}
</style>
<section><h1>実打球方向：前衛視点のブラインド人手確認（{page}/{len(candidates)}、完了 {reviewed}/{len(candidates)}）</h1><p>{previous}　{following}</p><p>{escaped(message)}</p></section>
<section><h2>{escaped(candidate['candidate_id'])}</h2><div class='meta'><div>人手確定インパクト：原動画 F{escaped(candidate['impact_source_frame'])}</div><div>クリップ内インパクト：F{escaped(queue['impact_local_frame'])}</div><div>固定主カメラ区間：{escaped(candidate['ready_segment_id'])}</div><div>ラリー区間：{escaped(candidate['play_window_id'])}</div></div>
<video id='review-video' controls preload='metadata' src='/clip/{escaped(candidate['candidate_id'])}.mp4'></video>
<p class='note'>ここでは<strong>インパクト時点の対象前衛から見て、ボールがどちら側を通るか</strong>を記録します。画面内での横移動方向ではありません。打球がほぼ直線的でも、対象前衛の左側へ来るなら <code>front_left</code> です。ロブ、主解析の想定外の打球、対象前衛を定められない場面は <code>excluded</code>、映像から左右を判定できない場面は <code>uncertain</code> を選びます。AI候補・軌跡の重ね表示は、独立した正解ラベルを作るため表示しません。</p>
<p class='keys'><strong>A</strong>：front_left　<strong>D</strong>：front_right　<strong>U</strong>：uncertain　<strong>X</strong>：excluded　<strong>Enter</strong>：保存</p>
<form id='review-form' method='post' action='/save?page={page}'><label>実打球方向（対象前衛視点）<select id='direction' name='direction'>{options}</select></label><label>メモ（任意）<input id='note' name='note' value='{escaped(record.get('review_note',''))}' size='46'></label><button class='save'>この判定を保存して次へ</button></form>
</section>
<script>
const direction = document.getElementById('direction');
const form = document.getElementById('review-form');
document.addEventListener('keydown', (event) => {{ const tag = document.activeElement && document.activeElement.tagName; if (['INPUT','SELECT','BUTTON','VIDEO'].includes(tag)) return; const key = event.key.toLowerCase(); if (key === 'a') {{ direction.value = 'front_left'; event.preventDefault(); }} if (key === 'd') {{ direction.value = 'front_right'; event.preventDefault(); }} if (key === 'u') {{ direction.value = 'uncertain'; event.preventDefault(); }} if (key === 'x') {{ direction.value = 'excluded'; event.preventDefault(); }} if (event.key === 'Enter') {{ event.preventDefault(); form.requestSubmit(); }} }});
</script></html>"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return

        def send_video(self, path: Path) -> None:
            size = path.stat().st_size
            start, end = 0, size - 1
            raw = self.headers.get("Range", "")
            if raw.startswith("bytes="):
                begin, _, finish = raw.removeprefix("bytes=").partition("-")
                start = int(begin) if begin else 0
                end = min(int(finish), end) if finish else end
                status = HTTPStatus.PARTIAL_CONTENT
            else:
                status = HTTPStatus.OK
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as file:
                file.seek(start)
                remaining = end - start + 1
                while remaining:
                    block = file.read(min(64 * 1024, remaining))
                    if not block:
                        return
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
                message = parse_qs(parsed.query).get("message", [""])[0]
                data = document(page, message).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path.startswith("/clip/"):
                name = Path(parsed.path.removeprefix("/clip/")).name
                path = clip_dir / name
                if path.is_file():
                    self.send_video(path)
                    return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/save":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                page = int(parse_qs(parsed.query).get("page", ["1"])[0])
            except ValueError:
                page = 1
            page = max(1, min(page, len(candidates)))
            candidate = candidates[page - 1]
            length = int(self.headers.get("Content-Length", "0"))
            form = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
            direction = form.get("direction", ["unreviewed"])[0]
            if direction not in DIRECTIONS:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid direction")
                return
            if direction in ("front_left", "front_right"):
                status = "confirmed"
            elif direction in ("uncertain", "excluded"):
                status = direction
            else:
                status = "unreviewed"
            rows = {row["candidate_id"]: row for row in read_csv(review_path)}
            record = rows[candidate["candidate_id"]]
            record.update({
                "actual_direction_front_view": direction,
                "review_status": status,
                "direction_source": "blind_human_video_review" if status == "confirmed" else "human_video_review",
                "review_note": form.get("note", [""])[0].strip(),
            })
            rows[candidate["candidate_id"]] = record
            write_csv(review_path, [rows[row["candidate_id"]] for row in candidates])
            next_page = min(page + 1, len(candidates))
            location = "/?" + urlencode({"page": next_page, "message": "保存しました。"})
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", location)
            self.end_headers()

    print(f"Open http://127.0.0.1:{args.port}/")
    print(f"Blind review output: {review_path}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
