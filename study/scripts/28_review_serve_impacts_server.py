"""Serve 60-fps clips for frame-level valid-serve-impact confirmation."""

from __future__ import annotations

import argparse
import csv
import html
import cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


STATUSES = ("unreviewed", "impact_confirmed", "impact_not_visible", "needs_raw_video")
FRAME_SEC = 1001 / 60000


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
    parser.add_argument("--port", type=int, default=8773)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    directory = project / "data" / "segments"
    csv_path = directory / f"{args.video_id}_serve_impact_review.csv"
    clip_dir = directory / "serve_impact_review" / args.video_id

    def page() -> str:
        cards = []
        for row in read(csv_path):
            ident = row["window_id"]
            options = "".join(f"<option value='{v}' {'selected' if v == row['impact_review_status'] else ''}>{v}</option>" for v in STATUSES)
            max_frame = max(0, round((float(row["clip_end_timestamp_sec"]) - float(row["clip_start_timestamp_sec"])) / FRAME_SEC) - 1)
            cards.append(f"""<article data-id='{ident}' data-max-frame='{max_frame}'><h2>{html.escape(ident)}</h2>
<p>以前の動作開始目安: 原動画 {float(row['legacy_motion_start_timestamp_sec']):.3f}秒。下の動画開始は原動画 {float(row['clip_start_timestamp_sec']):.3f}秒。</p>
<video controls preload='metadata' src='/clip/{ident}.mp4'></video>
<button type='button' onclick='step(this,-1)'>1フレーム戻る</button> <button type='button' onclick='step(this,1)'>1フレーム進む</button> <output>動画内 0.000秒</output>
<div class='framebox'><img class='frameview' alt='フレーム送りで表示される静止画'><small>フレーム送りを押すと、この静止画が更新されます。</small></div>
<label>判定 <select name='status__{ident}'>{options}</select></label>
<label>インパクトの動画内時刻（秒） <input class='offset' name='offset__{ident}' value=''></label>
<button type='button' onclick='copyTime(this)'>現在時刻を入力欄へ</button>
<label>メモ <input name='note__{ident}' value='{html.escape(row['impact_review_note'])}'></label></article>""")
        return """<!doctype html><html lang='ja'><meta charset='utf-8'><title>有効サーブ・インパクト確認</title><style>body{font-family:system-ui,sans-serif;background:#f3f4f6;margin:20px}header{position:sticky;top:0;background:#fff;padding:12px;z-index:2}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:12px;margin-top:16px}article{background:#fff;padding:12px;border-radius:8px}h2{font-size:16px;margin:0}video,.frameview{width:100%;background:#111;margin:8px 0}.framebox{display:none}.framebox.active{display:block}.framebox small{display:block}label{display:block;margin-top:7px}input{width:70%;margin-left:6px}output{margin-left:8px}</style><script>
const frame=""" + repr(FRAME_SEC) + """;
function video(button){return button.closest('article').querySelector('video')}
function show(card){const v=card.querySelector('video');card.querySelector('output').textContent='動画内 '+v.currentTime.toFixed(3)+'秒';}
function renderFrame(card,index){const img=card.querySelector('.frameview');card.querySelector('.framebox').classList.add('active');img.src='/frame/'+card.dataset.id+'/'+index+'.jpg?cache='+Date.now();card.querySelector('output').textContent='動画内 '+(index*frame).toFixed(3)+'秒（フレーム '+index+'）';}
function step(button,dir){const card=button.closest('article');const v=video(button);const current=card.dataset.frame===undefined?Math.round(v.currentTime/frame):Number(card.dataset.frame);const next=Math.max(0,Math.min(Number(card.dataset.maxFrame),current+dir));card.dataset.frame=next;v.pause();v.currentTime=next*frame;renderFrame(card,next);}
function copyTime(button){const card=button.closest('article');const time=card.dataset.frame===undefined?card.querySelector('video').currentTime:Number(card.dataset.frame)*frame;card.querySelector('.offset').value=time.toFixed(6);}
addEventListener('timeupdate',e=>{if(e.target.tagName==='VIDEO')show(e.target.closest('article'))},true);</script>
<form method='post' action='/save'><header><h1>有効サーブ・インパクト確認（60fps）</h1><p>ボールがラケットに当たったフレームを選びます。再生・停止後に1フレーム送りで合わせ、<strong>現在時刻を入力欄へ</strong>を押してください。音声は補助的に聞いてよいですが、映像で最終確認します。</p><button>確認結果を保存</button></header><main>""" + "".join(cards) + "</main></form></html>"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None: return
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                data = page().encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if parsed.path.startswith("/clip/"):
                path = clip_dir / Path(parsed.path.removeprefix("/clip/")).name
                if path.is_file():
                    data = path.read_bytes(); self.send_response(200); self.send_header("Content-Type", "video/mp4"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if parsed.path.startswith("/frame/"):
                parts = parsed.path.split("/")
                if len(parts) == 4 and parts[3].endswith(".jpg"):
                    ident = Path(parts[2]).name
                    try: index = int(Path(parts[3]).stem)
                    except ValueError: self.send_error(400); return
                    path = clip_dir / f"{ident}.mp4"
                    if path.is_file() and index >= 0:
                        capture = cv2.VideoCapture(str(path)); capture.set(cv2.CAP_PROP_POS_FRAMES, index); ok, frame = capture.read(); capture.release()
                        if ok:
                            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                            if ok:
                                data = encoded.tobytes(); self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            self.send_error(404)
        def do_POST(self) -> None:
            if urlparse(self.path).path != "/save": self.send_error(404); return
            form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode(), keep_blank_values=True)
            rows = read(csv_path); fields = list(rows[0])
            for row in rows:
                ident = row["window_id"]; status = form.get(f"status__{ident}", [row["impact_review_status"]])[0]
                if status not in STATUSES: self.send_error(400, "invalid status"); return
                offset = form.get(f"offset__{ident}", [""])[0].strip()
                if status == "impact_confirmed":
                    try: row["human_valid_serve_impact_timestamp_sec"] = f"{float(row['clip_start_timestamp_sec']) + float(offset):.6f}"
                    except ValueError: self.send_error(400, f"{ident}: numeric clip time is required"); return
                else: row["human_valid_serve_impact_timestamp_sec"] = ""
                row["impact_review_status"] = status; row["impact_review_note"] = form.get(f"note__{ident}", [""])[0]
            write(csv_path, fields, rows); self.send_response(303); self.send_header("Location", "/"); self.end_headers()
    print(f"Open http://127.0.0.1:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
