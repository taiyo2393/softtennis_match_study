"""Serve a small, auditable manual seed-selection review for ID switches."""

from __future__ import annotations

import argparse
import csv
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--target-front-camera-end", choices=("near", "far"), default="near")
    parser.add_argument("--port", type=int, default=8791)
    args = parser.parse_args()
    if not args.candidates.is_file():
        raise SystemExit(f"Candidate CSV not found: {args.candidates}")
    candidates = read(args.candidates)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in candidates:
        grouped.setdefault(row["event_id"], []).append(row)
    events = list(grouped)
    if not events:
        raise SystemExit("No seed candidates found")
    fields = ["event_id", "impact_source_frame", "selected_candidate_rank", "review_status", "review_note"]
    existing = {row["event_id"]: row for row in read(args.review)} if args.review.is_file() else {}
    for event_id in events:
        existing.setdefault(event_id, {
            "event_id": event_id,
            "impact_source_frame": grouped[event_id][0]["impact_source_frame"],
            "selected_candidate_rank": "",
            "review_status": "unreviewed",
            "review_note": "",
        })
    write(args.review, fields, list(existing.values()))

    def document(number: int, message: str = "") -> str:
        number = max(1, min(number, len(events)))
        event_id = events[number - 1]
        choices = grouped[event_id]
        prior = {row["event_id"]: row for row in read(args.review)}[event_id]
        selected = prior.get("selected_candidate_rank", "")
        option_html = "<option value=''>選択してください</option>" + "".join(
            f"<option value='{row['candidate_rank']}' {'selected' if selected == row['candidate_rank'] else ''}>"
            f"#{row['candidate_rank']}（検出信頼度 {float(row['detector_confidence']):.2f}）</option>"
            for row in choices
        )
        previous = f"<a href='/?page={number - 1}'>← 前</a>" if number > 1 else ""
        following = f"<a href='/?page={number + 1}'>次 →</a>" if number < len(events) else ""
        image_name = Path(choices[0]["image_path"]).name
        target_text = "奥側コートの対象前衛" if args.target_front_camera_end == "far" else "手前コートの対象前衛"
        return f"""<!doctype html><html lang='ja'><meta charset='utf-8'><title>ID切替：対象前衛の再シード</title><style>body{{font-family:system-ui,sans-serif;max-width:1280px;margin:auto;padding:18px;background:#f3f4f6}}section{{background:#fff;border-radius:10px;padding:16px;margin-bottom:14px}}img{{width:100%;background:#111}}label{{display:block;margin:12px 0}}input,select,button{{margin-left:8px;font-size:16px;padding:5px}}.help{{background:#fff5df;padding:10px;border-radius:6px}}</style><section><h1>ID切替：対象前衛の再シード（{number}/{len(events)}）</h1><p>{previous}　{following}</p><p>{message}</p></section><section><h2>{event_id}（インパクト F{choices[0]['impact_source_frame']}）</h2><p class='help'>灰枠も含め、番号付きの人物候補です。<strong>{target_text}</strong>を一人だけ選びます。黄色の #1 は単なる表示順であり、正解を示しません。対象を確定できなければ「除外」を選びます。</p><img src='/image/{image_name}'><form id='review-form' method='post' action='/save?page={number}'><label>対象前衛の枠 <select id='candidate' name='candidate'>{option_html}</select></label><label><input id='exclude' type='checkbox' name='exclude' {'checked' if prior.get('review_status') == 'excluded' else ''}> 対象前衛を確定できないため除外</label><label>メモ <input id='note' name='note' value='{prior.get('review_note','')}'></label><p><button>保存して次へ</button></p></form></section><script>const candidate=document.getElementById('candidate');const exclude=document.getElementById('exclude');const form=document.getElementById('review-form');document.addEventListener('keydown',(event)=>{{const tag=document.activeElement&&document.activeElement.tagName;if(['INPUT','SELECT','BUTTON'].includes(tag))return;if(/^[1-9]$/.test(event.key)){{candidate.value=event.key;event.preventDefault();}}if(event.key.toLowerCase()==='x'){{exclude.checked=!exclude.checked;event.preventDefault();}}if(event.key==='Enter'){{event.preventDefault();form.requestSubmit();}}}});</script></html>"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                try: number = int(parse_qs(parsed.query).get("page", ["1"])[0])
                except ValueError: number = 1
                data = document(number).encode("utf-8")
                self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if parsed.path.startswith("/image/"):
                path = args.candidates.parent / "reseed_candidates" / args.candidates.stem.removesuffix("_seed_candidates") / Path(parsed.path.removeprefix("/image/")).name
                if path.is_file():
                    data = path.read_bytes(); self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/save": self.send_error(HTTPStatus.NOT_FOUND); return
            try: number = int(parse_qs(parsed.query).get("page", ["1"])[0])
            except ValueError: number = 1
            number = max(1, min(number, len(events)))
            event_id = events[number - 1]
            form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode(), keep_blank_values=True)
            excluded = "exclude" in form
            selected = form.get("candidate", [""])[0]
            permitted = {row["candidate_rank"] for row in grouped[event_id]}
            if not excluded and selected not in permitted:
                self.send_error(HTTPStatus.BAD_REQUEST, "Select a candidate or exclude the event."); return
            rows = {row["event_id"]: row for row in read(args.review)}
            rows[event_id] = {
                "event_id": event_id,
                "impact_source_frame": grouped[event_id][0]["impact_source_frame"],
                "selected_candidate_rank": "" if excluded else selected,
                "review_status": "excluded" if excluded else "confirmed",
                "review_note": form.get("note", [""])[0].strip(),
            }
            write(args.review, fields, list(rows.values()))
            next_number = min(number + 1, len(events))
            self.send_response(HTTPStatus.SEE_OTHER); self.send_header("Location", f"/?page={next_number}"); self.end_headers()

    print(f"Open http://127.0.0.1:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
