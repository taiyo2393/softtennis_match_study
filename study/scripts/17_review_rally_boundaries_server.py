"""Serve a no-JavaScript review form for proposed rally-end boundaries."""

from __future__ import annotations

import argparse, csv
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


STATUSES = ("unreviewed", "confirmed", "not_rally_boundary", "needs_raw_video")
# A boundary can be real without ending a rally: a first-serve fault ends a
# serve attempt but must not be counted as a completed rally or point.
TYPES = ("", "rally_end", "serve_fault_end", "between_points", "camera_only", "unknown")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file: return list(csv.DictReader(file))


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--video-id", required=True); parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]; directory = project / "data" / "segments"
    csv_path = directory / f"{args.video_id}_rally_boundary_candidates.csv"; clip_dir = directory / "boundary_review" / args.video_id

    def page() -> str:
        rows = [row for row in read(csv_path) if row["candidate_type"] == "main_camera_departure_candidate"]
        feature_path = directory / f"{args.video_id}_boundary_audio_features.csv"
        audio = {row["boundary_id"]: row for row in read(feature_path)} if feature_path.exists() else {}
        cards = []
        for row in rows:
            ident = row["boundary_id"]; status = "".join(f"<option value='{v}' {'selected' if v==row['human_status'] else ''}>{v}</option>" for v in STATUSES)
            kinds = "".join(f"<option value='{v}' {'selected' if v==row['human_boundary_type'] else ''}>{v or '---'}</option>" for v in TYPES)
            feature = audio.get(ident, {}); sound = f"音響候補: 前3秒 {feature.get('onset_count_3s_before','-')}件 / 後3秒 {feature.get('onset_count_3s_after','-')}件"
            cards.append(f"""<article><h2>{ident} <small>{row['review_priority']}</small></h2><p>候補時刻: {float(row['timestamp_sec']):.3f} 秒（動画の3秒地点）</p><p>{sound}</p><a href='/clip/{ident}.mp4'>音声付き・前後3秒の動画を再生</a><label>判定 <select name='status__{ident}'>{status}</select></label><label>種別 <select name='type__{ident}'>{kinds}</select></label><label>確定時刻（必要時のみ）<input name='time__{ident}' value='{row['human_timestamp_sec']}' inputmode='decimal'></label><label>メモ <input name='note__{ident}' value='{row['review_note']}'></label></article>""")
        return """<!doctype html><html lang='ja'><meta charset='utf-8'><title>ラリー境界候補の確認</title><style>body{font-family:system-ui,sans-serif;background:#f3f4f6;margin:20px}header{position:sticky;top:0;background:white;padding:12px;z-index:1}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-top:16px}article{background:white;padding:12px;border-radius:8px}h2{font-size:16px;margin:0}small{color:#9b1c1c}label{display:block;margin-top:8px}input{width:75%;margin-left:5px}a{display:inline-block;margin-top:8px;padding:5px;background:#e5e7eb;color:#111;text-decoration:none;border-radius:4px}</style><form method='post' action='/save'><header><h1>プレー境界候補の確認</h1><p>動画は候補時刻の前後3秒・15 fpsです。<strong>再生3秒地点の赤枠が候補時刻</strong>です。音響候補数は打球音の確定値ではなく、確認優先度の補助情報です。ラリー終了なら <code>confirmed / rally_end</code>、ファースト／セカンドサーブのフォルトなら <code>confirmed / serve_fault_end</code>、画角変化だけなら <code>not_rally_boundary / camera_only</code> にします。</p><button type='submit'>確認結果を保存</button></header><main>""" + "".join(cards) + "</main></form></html>"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None: return
        def send_file(self, path: Path) -> None:
            data = path.read_bytes(); self.send_response(200); self.send_header("Content-Type", "video/mp4"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                data=page().encode(); self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data); return
            if parsed.path.startswith("/clip/"):
                path=clip_dir / Path(parsed.path.removeprefix("/clip/")).name
                if path.is_file(): self.send_file(path); return
            self.send_error(404)
        def do_POST(self) -> None:
            if urlparse(self.path).path != "/save": self.send_error(404); return
            form=parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode(),keep_blank_values=True); rows=read(csv_path); fields=list(rows[0])
            for row in rows:
                if row["candidate_type"] != "main_camera_departure_candidate": continue
                ident=row["boundary_id"]; status=form.get(f"status__{ident}",["unreviewed"])[0]; kind=form.get(f"type__{ident}",[""])[0]
                if status not in STATUSES or kind not in TYPES: self.send_error(400,"invalid form value"); return
                row["human_status"],row["human_boundary_type"],row["human_timestamp_sec"],row["review_note"] = status,kind,form.get(f"time__{ident}",[""])[0],form.get(f"note__{ident}",[""])[0]
            write(csv_path,fields,rows); self.send_response(303); self.send_header("Location","/"); self.end_headers()
    print(f"Open http://127.0.0.1:{args.port}/"); ThreadingHTTPServer(("127.0.0.1",args.port),Handler).serve_forever(); return 0


if __name__ == "__main__": raise SystemExit(main())
