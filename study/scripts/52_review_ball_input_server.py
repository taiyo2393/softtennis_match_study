"""Serve AI-assisted human confirmation of ball positions used as model inputs.

This review is intentionally not blinded: its purpose is to turn an AI ball
candidate into a human-confirmed feature, while recording whether the candidate
was accepted, corrected, or missing.  A separate blinded set evaluates the
BallTrack detector itself.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2


VISIBILITY = ("unreviewed", "visible", "not_visible", "uncertain")
METHODS = ("", "ai_candidate_confirmed", "manual_corrected", "manual_no_candidate")
CONFIDENCE = ("", "high", "medium", "low")


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def options(values: tuple[str, ...], selected: str) -> str:
    return "".join(
        f"<option value='{value}' {'selected' if value == selected else ''}>{value or '—'}</option>"
        for value in values
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--per-page", type=int, default=6)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    labels_path = project / "annotations" / f"front_motion_ball_inputs_{args.video_id}.csv"
    if not labels_path.is_file():
        raise SystemExit("Run script 51 before starting this server.")

    def frame_path(candidate_id: str, local_frame: int) -> Path:
        return project / "data" / "balltrack" / "racketvision" / args.video_id / candidate_id / "frames" / f"{local_frame:06d}.jpg"

    def candidate_image(row: dict[str, str]) -> bytes | None:
        image = cv2.imread(str(frame_path(row["candidate_id"], int(row["local_frame"]))))
        if image is None:
            return None
        if row["ai_visible"] == "1":
            x, y = round(float(row["ai_x_px"])), round(float(row["ai_y_px"]))
            cv2.drawMarker(image, (x, y), (0, 210, 255), cv2.MARKER_CROSS, 28, 3)
            cv2.circle(image, (x, y), 12, (0, 210, 255), 2)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return encoded.tobytes() if ok else None

    def page(page_index: int, message: str = "") -> str:
        fields, rows = read(labels_path)
        total_pages = max(1, math.ceil(len(rows) / args.per_page))
        page_index = max(0, min(total_pages - 1, page_index))
        start, end = page_index * args.per_page, (page_index + 1) * args.per_page
        reviewed = sum(row["human_visibility"] != "unreviewed" for row in rows)
        cards: list[str] = []
        for row in rows[start:end]:
            sample_id = html.escape(row["sample_id"])
            candidate_id = html.escape(row["candidate_id"])
            local_frame = int(row["local_frame"])
            ai_visible = row["ai_visible"] == "1"
            initial_x = row["human_x_px"] or (row["ai_x_px"] if ai_visible else "")
            initial_y = row["human_y_px"] or (row["ai_y_px"] if ai_visible else "")
            ai_note = (
                f"候補あり：({html.escape(row['ai_x_px'])}, {html.escape(row['ai_y_px'])})、信頼度 {html.escape(row['ai_confidence'])}"
                if ai_visible else "候補なし"
            )
            cards.append(f"""<article data-sample='{sample_id}'>
<h2>{sample_id}</h2>
<p>入力時点 <strong>{html.escape(row['input_offset_sec'])}秒</strong> ／ 原動画 F{html.escape(row['source_frame'])} ／ クリップ F{local_frame}</p>
<img class='frame' src='/candidate/{candidate_id}/{local_frame}.jpg' alt='AI候補を重ねた入力フレーム' onclick='markPoint(this,event)'>
<p class='ai'>AI球候補（黄の十字・円）：{ai_note}</p>
<input type='hidden' name='sample_id' value='{sample_id}'>
<label>人の判定 <select class='visibility' name='visibility__{sample_id}'>{options(VISIBILITY, row['human_visibility'])}</select></label>
<label>ボール中心 x <input class='x' name='x__{sample_id}' value='{html.escape(initial_x)}' inputmode='decimal'></label>
<label>ボール中心 y <input class='y' name='y__{sample_id}' value='{html.escape(initial_y)}' inputmode='decimal'></label>
<p class='help'>AI候補が正しければ、判定を <code>visible</code>、方法を <code>ai_candidate_confirmed</code> にして保存してください。候補を修正する、または候補なしでボールを見つけた場合は、画像上をクリックするか座標を入力し、該当する手動方法を選択してください。</p>
<label>確定方法 <select name='method__{sample_id}'>{options(METHODS, row.get('review_method', ''))}</select></label>
<label>確信度 <select name='confidence__{sample_id}'>{options(CONFIDENCE, row['label_confidence'])}</select></label>
<label>メモ <input name='notes__{sample_id}' value='{html.escape(row['notes'])}'></label>
</article>""")
        navigation = " ".join(
            f"<a href='/?page={index}'>{index + 1}</a>" if index != page_index else f"<strong>{index + 1}</strong>"
            for index in range(total_pages)
        )
        return f"""<!doctype html><html lang='ja'><meta charset='utf-8'><title>前衛動作予測：球位置入力の確認</title><style>
body{{font-family:system-ui,sans-serif;background:#f3f4f6;margin:18px}} header{{position:sticky;top:0;z-index:2;background:#fff;padding:12px}} main{{max-width:1320px;margin:12px auto}} article{{background:#fff;padding:14px;border-radius:8px;margin:14px 0}} h2{{font-size:16px;margin:0}} .frame{{display:block;width:100%;max-width:1280px;margin:auto;cursor:crosshair;background:#111}} label{{display:block;margin-top:7px}} input{{width:220px}} .ai{{font-size:14px}} .help{{font-size:14px;background:#f8fafc;padding:8px}}</style>
<script>
function markPoint(image,event){{const c=image.closest('article'),r=image.getBoundingClientRect(),x=Math.round((event.clientX-r.left)*image.naturalWidth/r.width),y=Math.round((event.clientY-r.top)*image.naturalHeight/r.height);c.querySelector('.x').value=x;c.querySelector('.y').value=y;c.querySelector('.visibility').value='visible';}}
</script>
<form method='post' action='/save'><header><h1>前衛動作予測：ボール入力位置の人手確認</h1><p>この画面の座標だけが予測器の入力になります。AI候補は確認を速くする補助であり、候補をそのまま正解として使いません。黄マーカーが違う場合は、実際のボール中心をクリックまたは入力してください。</p><p>確認済み {reviewed}/{len(rows)}。ページ {page_index + 1}/{total_pages}。{message}</p><button>このページを保存</button>　{navigation}</header><main>{''.join(cards)}</main><input type='hidden' name='page' value='{page_index}'></form></html>"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                page_index = int(parse_qs(parsed.query).get("page", ["0"])[0])
                data = page(page_index).encode()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if parsed.path.startswith("/candidate/"):
                parts = parsed.path.split("/")
                if len(parts) == 4 and parts[3].endswith(".jpg") and Path(parts[3]).stem.isdigit():
                    _, rows = read(labels_path)
                    candidate_id = Path(parts[2]).name; local_frame = int(Path(parts[3]).stem)
                    row = next((item for item in rows if item["candidate_id"] == candidate_id and int(item["local_frame"]) == local_frame), None)
                    if row:
                        data = candidate_image(row)
                        if data:
                            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            self.send_error(404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/save":
                self.send_error(404); return
            form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode(), keep_blank_values=True)
            page_index = int(form.get("page", ["0"])[0])
            fields, rows = read(labels_path); by_id = {row["sample_id"]: row for row in rows}
            try:
                for sample_id in form.get("sample_id", []):
                    row = by_id[sample_id]
                    visibility = form.get(f"visibility__{sample_id}", ["unreviewed"])[0]
                    x = form.get(f"x__{sample_id}", [""])[0].strip(); y = form.get(f"y__{sample_id}", [""])[0].strip()
                    method = form.get(f"method__{sample_id}", [""])[0]
                    if visibility not in VISIBILITY or method not in METHODS:
                        raise ValueError(f"{sample_id}: 不正な選択値です")
                    if visibility == "visible":
                        if not x or not y or not method:
                            raise ValueError(f"{sample_id}: visible には座標と確定方法が必要です")
                        float(x); float(y)
                    else:
                        x, y, method = "", "", ""
                    row.update({"human_visibility": visibility, "human_x_px": x, "human_y_px": y, "review_method": method, "label_confidence": form.get(f"confidence__{sample_id}", [""])[0], "notes": form.get(f"notes__{sample_id}", [""])[0].strip()})
                write(labels_path, fields, rows)
            except (KeyError, ValueError) as error:
                data = page(page_index, f"<b>保存しませんでした：{html.escape(str(error))}</b>").encode()
                self.send_response(400); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            self.send_response(303); self.send_header("Location", f"/?page={page_index}"); self.end_headers()

    print(f"Open http://127.0.0.1:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
