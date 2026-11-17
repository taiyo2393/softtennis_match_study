"""Serve a no-JavaScript review UI for candidate rally starts."""

from __future__ import annotations

import argparse,csv
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs,urlparse


STATUSES=("unreviewed","confirmed","not_rally_start","needs_raw_video")


def read(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def write(path:Path,fields:list[str],rows:list[dict[str,str]])->None:
    temp=path.with_suffix(".tmp")
    with temp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    temp.replace(path)


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--video-id",required=True);parser.add_argument("--port",type=int,default=8771);args=parser.parse_args()
    project=Path(__file__).resolve().parents[1];directory=project/"data"/"segments";csv_path=directory/f"{args.video_id}_rally_window_candidates.csv";clip_dir=directory/"rally_start_review"/args.video_id
    def page()->str:
        cards=[]
        for row in read(csv_path):
            ident=row["window_id"];opts="".join(f"<option value='{x}' {'selected' if x==row['start_confirmation'] else ''}>{x}</option>" for x in STATUSES)
            cards.append(f"""<article><h2>{ident}</h2><p>候補開始: {float(row['source_start_timestamp_sec']):.3f} 秒（動画の3秒地点・緑枠）</p><a href='/clip/{ident}.mp4'>前後3秒の動画を再生</a><label>開始判定 <select name='status__{ident}'>{opts}</select></label><label>開始時刻（必要時のみ）<input name='time__{ident}' value='{row.get('human_start_timestamp_sec','')}' inputmode='decimal'></label><label>メモ <input name='note__{ident}' value='{row.get('start_review_note','')}'></label></article>""")
        return """<!doctype html><html lang='ja'><meta charset='utf-8'><title>ラリー開始候補の確認</title><style>body{font-family:system-ui,sans-serif;background:#f3f4f6;margin:20px}header{position:sticky;top:0;background:white;padding:12px;z-index:1}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-top:16px}article{background:white;padding:12px;border-radius:8px}h2{font-size:16px;margin:0}label{display:block;margin-top:8px}input{width:75%;margin-left:5px}a{display:inline-block;margin-top:8px;padding:5px;background:#e5e7eb;color:#111;text-decoration:none;border-radius:4px}</style><form method='post' action='/save'><header><h1>ラリー開始候補の確認</h1><p><strong>再生3秒地点の緑枠が開始候補時刻</strong>です。サーブ前の定点画角再入なら <code>confirmed</code>、ラリー途中から映った／候補が誤りなら <code>not_rally_start</code> にします。開始時刻は、ラリー開始（サーブ動作開始）を粗く補正する場合だけ記入します。</p><button type='submit'>確認結果を保存</button></header><main>"""+"".join(cards)+"</main></form></html>"
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_:object)->None:return
        def do_GET(self)->None:
            parsed=urlparse(self.path)
            if parsed.path=="/":
                data=page().encode();self.send_response(200);self.send_header("Content-Type","text/html; charset=utf-8");self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data);return
            if parsed.path.startswith("/clip/"):
                path=clip_dir/Path(parsed.path.removeprefix("/clip/")).name
                if path.is_file():
                    data=path.read_bytes();self.send_response(200);self.send_header("Content-Type","video/mp4");self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data);return
            self.send_error(404)
        def do_POST(self)->None:
            if urlparse(self.path).path!="/save":self.send_error(404);return
            form=parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode(),keep_blank_values=True);rows=read(csv_path);fields=list(rows[0])
            for row in rows:
                ident=row["window_id"];status=form.get(f"status__{ident}",[row["start_confirmation"]])[0]
                if status not in STATUSES:self.send_error(400,"invalid status");return
                row["start_confirmation"]=status;row["human_start_timestamp_sec"]=form.get(f"time__{ident}",[""])[0];row["start_review_note"]=form.get(f"note__{ident}",[""])[0]
                row["window_status"]="complete_boundary" if status=="confirmed" and row["end_confirmation"]=="confirmed" else "excluded_or_unresolved"
            write(csv_path,fields,rows);self.send_response(303);self.send_header("Location","/");self.end_headers()
    print(f"Open http://127.0.0.1:{args.port}/");ThreadingHTTPServer(("127.0.0.1",args.port),Handler).serve_forever();return 0


if __name__=="__main__":raise SystemExit(main())
