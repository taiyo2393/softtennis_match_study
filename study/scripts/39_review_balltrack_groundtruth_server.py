"""Serve a blinded frame-level ground-truth review UI for BallTrack.

The reviewer sees only original frames and neighbouring raw frames.  Model
coordinates/confidence stay in the CSV but are deliberately never rendered.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


TRUTH_STATES = ("unreviewed", "visible", "not_visible", "uncertain")
CONFIDENCE = ("", "high", "medium", "low")


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def options(values: tuple[str, ...], selected: str) -> str:
    return "".join(
        f"<option value='{value}' {'selected' if value == selected else ''}>{value or '—'}</option>"
        for value in values
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--per-page", type=int, default=6)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    labels_path = project / "annotations" / f"balltrack_groundtruth_{args.video_id}_pilot.csv"
    if not labels_path.is_file():
        raise SystemExit(f"Create the samples first: {labels_path}")

    def page(page_index: int, message: str = "") -> str:
        fields, rows = read(labels_path)
        total_pages = max(1, math.ceil(len(rows) / args.per_page))
        page_index = max(0, min(total_pages - 1, page_index))
        start, end = page_index * args.per_page, (page_index + 1) * args.per_page
        reviewed = sum(row["truth_visibility"] != "unreviewed" for row in rows)
        cards: list[str] = []
        for row in rows[start:end]:
            candidate = html.escape(row["candidate_id"])
            frame = int(row["local_frame"])
            sample = html.escape(row["sample_id"])
            x = html.escape(row["truth_x_px"])
            y = html.escape(row["truth_y_px"])
            notes = html.escape(row["notes"])
            cards.append(f"""<article data-candidate='{candidate}' data-frame='{frame}' data-current='{frame}'>
<h2>{sample}</h2>
<p>原動画 {float(row['source_timestamp_sec']):.3f} 秒 ／ フレーム {frame}</p>
<img class='raw-frame' src='/frame/{candidate}/{frame}.jpg' alt='原動画フレーム' onclick='markPoint(this,event)'>
<p><button type='button' onclick='contextFrame(this,-2)'>−2</button> <button type='button' onclick='contextFrame(this,-1)'>−1</button> <button type='button' onclick='contextFrame(this,0)'>評価フレームへ戻る</button> <button type='button' onclick='contextFrame(this,1)'>+1</button> <button type='button' onclick='contextFrame(this,2)'>+2</button> <button type='button' onclick='fullScreen(this)'>画像を全画面表示</button> <output>評価フレーム</output></p>
<input type='hidden' name='sample_id' value='{sample}'>
<input type='hidden' class='x' name='x__{sample}' value='{x}'><input type='hidden' class='y' name='y__{sample}' value='{y}'>
<label>正解 <select class='truth' name='truth__{sample}'>{options(TRUTH_STATES, row['truth_visibility'])}</select></label>
<p class='point'>座標: {x or '未入力'}, {y or '未入力'}。可視なら<strong>評価フレーム上</strong>のボール中心をクリック。</p>
<label>確信度 <select name='confidence__{sample}'>{options(CONFIDENCE, row['label_confidence'])}</select></label>
<label>メモ <input name='notes__{sample}' value='{notes}'></label>
</article>""")
        navigation = " ".join(
            f"<a href='/?page={index}'>{index + 1}</a>" if index != page_index else f"<strong>{index + 1}</strong>"
            for index in range(total_pages)
        )
        return f"""<!doctype html><html lang='ja'><meta charset='utf-8'><title>BallTrack正解付与</title><style>
body{{font-family:system-ui,sans-serif;background:#f3f4f6;margin:18px}} header{{position:sticky;top:0;z-index:2;background:#fff;padding:12px}} main{{display:block;max-width:1320px;margin:12px auto}} article{{background:#fff;padding:14px;border-radius:8px;margin:14px 0}} h2{{font-size:16px;margin:0}} .raw-frame{{display:block;width:100%;max-width:1280px;margin:auto;cursor:crosshair;background:#111}} .raw-frame:fullscreen{{object-fit:contain;background:#111}} label{{display:block;margin-top:7px}} input{{width:65%}} .point{{font-size:14px}} output{{margin-left:8px}}</style>
<script>
function card(element){{return element.closest('article')}}
function imagePath(c,frame){{return '/frame/'+c.dataset.candidate+'/'+frame+'.jpg'}}
function contextFrame(button,delta){{const c=card(button),base=Number(c.dataset.frame);const next=delta===0?base:Math.max(0,base+delta);c.dataset.current=next;c.querySelector('img').src=imagePath(c,next);c.querySelector('output').textContent=next===base?'評価フレーム':'文脈フレーム '+next+'（ここではクリックしない）';}}
function fullScreen(button){{const image=card(button).querySelector('img');if(image.requestFullscreen){{image.requestFullscreen();}}}}
function markPoint(image,event){{const c=card(image),base=Number(c.dataset.frame),current=Number(c.dataset.current);if(current!==base){{alert('正解座標は「評価フレームへ戻る」を押してから入力してください。');return;}}const r=image.getBoundingClientRect(),x=Math.round((event.clientX-r.left)*image.naturalWidth/r.width),y=Math.round((event.clientY-r.top)*image.naturalHeight/r.height);c.querySelector('.x').value=x;c.querySelector('.y').value=y;c.querySelector('.truth').value='visible';c.querySelector('.point').innerHTML='座標: <strong>'+x+', '+y+'</strong>（クリックで更新可）';}}
</script>
<form method='post' action='/save'><header><h1>BallTrack：ブラインド正解位置付与</h1><p>モデル候補・信頼度は表示しません。原画像だけを見て、評価フレームでボールが見えるなら中心をクリック、見えなければ <code>not_visible</code>、判定不能なら <code>uncertain</code> を選びます。前後フレームは確認用であり、座標入力は評価フレームに限ります。</p><p>レビュー済み {reviewed}/{len(rows)}。ページ {page_index + 1}/{total_pages}。{message}</p><button>このページを保存</button>　{navigation}</header><main>{''.join(cards)}</main><input type='hidden' name='page' value='{page_index}'></form></html>"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                page_index = int(parse_qs(parsed.query).get("page", ["0"])[0])
                data = page(page_index).encode()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if parsed.path.startswith("/frame/"):
                parts = parsed.path.split("/")
                if len(parts) == 4 and parts[3].endswith(".jpg"):
                    candidate = Path(parts[2]).name
                    frame = Path(parts[3]).stem
                    if frame.isdigit():
                        image = project / "data" / "balltrack" / "racketvision" / args.video_id / candidate / "frames" / f"{int(frame):06d}.jpg"
                        if image.is_file():
                            data = image.read_bytes(); self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            self.send_error(404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/save":
                self.send_error(404); return
            form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode(), keep_blank_values=True)
            page_index = int(form.get("page", ["0"])[0])
            fields, rows = read(labels_path)
            by_id = {row["sample_id"]: row for row in rows}
            try:
                for sample_id in form.get("sample_id", []):
                    row = by_id[sample_id]
                    truth = form.get(f"truth__{sample_id}", ["unreviewed"])[0]
                    x = form.get(f"x__{sample_id}", [""])[0].strip()
                    y = form.get(f"y__{sample_id}", [""])[0].strip()
                    if truth not in TRUTH_STATES:
                        raise ValueError(f"{sample_id}: 不正な正解状態です")
                    if truth == "visible":
                        if not x or not y:
                            raise ValueError(f"{sample_id}: visible にはボール中心のクリックが必要です")
                        float(x); float(y)
                    else:
                        x, y = "", ""
                    row.update({"truth_visibility": truth, "truth_x_px": x, "truth_y_px": y,
                                "label_confidence": form.get(f"confidence__{sample_id}", [""])[0],
                                "notes": form.get(f"notes__{sample_id}", [""])[0].strip()})
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
