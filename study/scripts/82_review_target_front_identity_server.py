"""Assign a stable physical-player ID to each audited target-front event.

Camera end (near/far) is not a player identity because teams change ends.
This review presents only events that already have a confirmed direction and
usable audited ankle pose.  The reviewer assigns P1--P4 consistently across
the whole match; names are optional and can be mapped to P1--P4 later.
"""

from __future__ import annotations

import argparse
import csv
import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PLAYER_IDS = ("P1", "P2", "P3", "P4")
STATUSES = ("unreviewed", "confirmed", "identity_uncertain")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def direction_events(path: Path, video_id: str, camera_end: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for row in read(path):
        direction = row.get("actual_direction_target_front_view") or row.get("actual_direction_front_view")
        if (
            row.get("video_id") == video_id
            and row.get("review_status") == "confirmed"
            and direction in ("front_left", "front_right")
        ):
            events.append({
                "event_id": row["candidate_id"],
                "video_id": video_id,
                "impact_source_frame": row["impact_source_frame"],
                "target_front_camera_end": camera_end,
            })
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--near-direction-review", type=Path, required=True)
    parser.add_argument("--far-direction-review", type=Path, required=True)
    parser.add_argument("--near-pose-review", type=Path, required=True)
    parser.add_argument("--far-pose-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8803)
    args = parser.parse_args()
    for path in (
        args.near_direction_review, args.far_direction_review,
        args.near_pose_review, args.far_pose_review,
    ):
        if not path.is_file():
            raise SystemExit(f"Required input not found: {path}")
    project = Path(__file__).resolve().parents[1]
    output = args.output or project / "annotations" / f"{args.video_id}_target_front_identity_review.csv"
    preview_dir = project / "data" / "pose" / "target_front" / "previews"
    near_pose_usable = {
        row["event_id"] for row in read(args.near_pose_review)
        if row.get("pose_status") == "usable"
    }
    far_pose_usable = {
        row["event_id"] for row in read(args.far_pose_review)
        if row.get("pose_status") == "usable"
    }
    events = [
        event for event in direction_events(args.near_direction_review, args.video_id, "near")
        if event["event_id"] in near_pose_usable
    ] + [
        event for event in direction_events(args.far_direction_review, args.video_id, "far")
        if event["event_id"] in far_pose_usable
    ]
    events.sort(key=lambda row: (int(row["impact_source_frame"]), row["event_id"]))
    if len({row["event_id"] for row in events}) != len(events):
        raise SystemExit("Duplicate event_id across near/far direction inputs")
    fields = [
        "event_id", "video_id", "impact_source_frame", "target_front_camera_end",
        "target_front_player_id", "identity_status", "review_note",
    ]
    old = {row["event_id"]: row for row in read(output)} if output.is_file() else {}
    rows: list[dict[str, str]] = []
    for event in events:
        saved = old.get(event["event_id"], {})
        rows.append({
            **event,
            "target_front_player_id": saved.get("target_front_player_id", ""),
            "identity_status": saved.get("identity_status", "unreviewed"),
            "review_note": saved.get("review_note", ""),
        })
    write(output, fields, rows)

    def page(number: int, message: str = "") -> str:
        current = read(output)
        number = max(1, min(number, len(current)))
        item = current[number - 1]
        complete = sum(row["identity_status"] != "unreviewed" for row in current)
        player_options = "<option value=''>選択してください</option>" + "".join(
            f"<option value='{player}' {'selected' if item['target_front_player_id'] == player else ''}>{player}</option>"
            for player in PLAYER_IDS
        )
        previous = f"<a href='/?page={number - 1}'>← 前</a>" if number > 1 else ""
        following = f"<a href='/?page={number + 1}'>次 →</a>" if number < len(current) else ""
        end_jp = "手前側コート" if item["target_front_camera_end"] == "near" else "奥側コート"
        return f"""<!doctype html><html lang='ja'><meta charset='utf-8'><title>対象前衛の物理選手ID確認</title><style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:auto;padding:18px;background:#f3f4f6}}section{{background:#fff;border-radius:10px;padding:16px;margin-bottom:14px}}video{{width:100%;background:#111}}label{{display:block;margin:12px 0}}input,select,button{{margin-left:8px;font-size:16px;padding:4px}}.note{{background:#fff5df;padding:12px;border-radius:6px;line-height:1.6}}.keys{{background:#e0f2fe;padding:10px;border-radius:6px}}</style><section><h1>対象前衛の物理選手ID確認（{number}/{len(current)}、完了 {complete}）</h1><p>{previous}　{following}</p><p>{html.escape(message)}</p></section><section><h2>{html.escape(item['event_id'])}</h2><p>インパクト F{html.escape(item['impact_source_frame'])} ／ {end_jp}の対象前衛</p><video controls preload='metadata' src='/preview/{html.escape(item['event_id'])}.mp4'></video><p class='keys'><strong>1</strong>〜<strong>4</strong>：P1〜P4を選択　<strong>X</strong>：人物IDが確定不能　<strong>Enter</strong>：保存して次へ</p><p class='note'>黄色枠・緑点の選手が対象です。<strong>P1〜P4は試合を通して同じ実選手に固定</strong>してください。チェンジコート後も、画面上の手前／奥ではなく同じ人物なら同じP番号を選びます。名前を使う必要はありません。</p><form id='review-form' method='post' action='/save?page={number}'><label>対象前衛の選手ID <select id='player' name='player'>{player_options}</select></label><label><input id='uncertain' type='checkbox' name='uncertain' {'checked' if item['identity_status'] == 'identity_uncertain' else ''}> 人物IDを確定できない</label><label>メモ <input id='note' name='note' value='{html.escape(item['review_note'])}' size='48'></label><button>この判定を保存して次へ</button></form></section><script>const player=document.getElementById('player');const uncertain=document.getElementById('uncertain');const form=document.getElementById('review-form');document.addEventListener('keydown',(event)=>{{const tag=document.activeElement&&document.activeElement.tagName;if(['INPUT','SELECT','BUTTON','VIDEO'].includes(tag))return;if(['1','2','3','4'].includes(event.key)){{player.value='P'+event.key;uncertain.checked=false;event.preventDefault();}}if(event.key.toLowerCase()==='x'){{player.value='';uncertain.checked=true;event.preventDefault();}}if(event.key==='Enter'){{event.preventDefault();form.requestSubmit();}}}});</script></html>"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return

        def send_video(self, path: Path) -> None:
            size = path.stat().st_size; start, end = 0, size - 1
            raw = self.headers.get("Range", "")
            if raw.startswith("bytes="):
                parts = raw.removeprefix("bytes=").split("-")
                start = int(parts[0]) if parts[0] else 0
                end = min(int(parts[1]), end) if len(parts) > 1 and parts[1] else end
                status = HTTPStatus.PARTIAL_CONTENT
            else:
                status = HTTPStatus.OK
            self.send_response(status); self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(end - start + 1))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start); remaining = end - start + 1
                while remaining:
                    block = handle.read(min(64 * 1024, remaining))
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
                    number = int(parse_qs(parsed.query).get("page", ["1"])[0])
                except ValueError:
                    number = 1
                document = page(number).encode()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(document))); self.end_headers(); self.wfile.write(document)
                return
            if parsed.path.startswith("/preview/"):
                video = preview_dir / Path(parsed.path.removeprefix("/preview/")).name
                if video.is_file():
                    self.send_video(video)
                    return
            self.send_error(404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/save":
                self.send_error(404)
                return
            try:
                number = int(parse_qs(urlparse(self.path).query).get("page", ["1"])[0])
            except ValueError:
                number = 1
            current = read(output); number = max(1, min(number, len(current))); item = current[number - 1]
            form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode(), keep_blank_values=True)
            player = form.get("player", [""])[0]
            uncertain = "uncertain" in form
            if player not in ("", *PLAYER_IDS):
                self.send_error(400)
                return
            item["target_front_player_id"] = "" if uncertain else player
            item["identity_status"] = "identity_uncertain" if uncertain else ("confirmed" if player else "unreviewed")
            item["review_note"] = form.get("note", [""])[0].strip()
            current[number - 1] = item
            write(output, fields, current)
            next_number = min(number + 1, len(current))
            self.send_response(303); self.send_header("Location", f"/?page={next_number}"); self.end_headers()

    print(f"Open http://127.0.0.1:{args.port}/ ({len(rows)} events)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
