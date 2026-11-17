"""Serve a no-JavaScript form for manual or assisted trial annotation."""

from __future__ import annotations

import argparse, csv, mimetypes, re
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


TRIAL_FIELDS = ["annotation_id", "session_id", "trial_id", "video_id", "segment_id", "rally_id", "source_start_timestamp_sec", "source_end_timestamp_sec", "hitter_id", "target_front_player_id", "hitter_handedness", "front_player_handedness", "impact_timestamp_sec", "impact_source_frame", "actual_ball_direction_court", "actual_ball_direction_relative_to_front", "shot_type", "starting_court_side", "occlusion_flag", "exclusion_reason", "annotation_confidence", "completion_status"]


def append_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not exists: writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True); parser.add_argument("--segment-id", required=True)
    parser.add_argument("--method", choices=("manual", "assisted"), required=True)
    parser.add_argument("--session-id", required=True); parser.add_argument("--annotator-id", required=True)
    parser.add_argument("--port", type=int, default=8769); args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]; ann = project / "annotations"; tracking = project / "data" / "tracking"
    if args.method == "manual": video = project / "data" / "proxy" / "annotation" / f"{args.video_id}_{args.segment_id}_clean_1080p60.mp4"
    else: video = tracking / f"{args.video_id}_{args.segment_id}_person_tracking_pilot_browser.mp4"
    if not video.exists(): raise SystemExit(f"Required video missing: {video}")
    with (project / "data" / "segments" / "analysis_ready_main_camera_segments.csv").open(newline="", encoding="utf-8") as file:
        seg = next(row for row in csv.DictReader(file) if row["segment_id"] == args.segment_id)
    source_start = float(seg["start_timestamp_sec"]); start_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    append_csv(ann / "annotation_sessions.csv", ["session_id", "annotator_id", "method", "video_id", "trial_set_id", "started_at", "ended_at", "duration_sec", "instructions_version", "notes"],
               {"session_id": args.session_id, "annotator_id": args.annotator_id, "method": args.method, "video_id": args.video_id, "trial_set_id": args.segment_id, "started_at": start_at, "ended_at": "", "duration_sec": "", "instructions_version": "v1", "notes": ""})

    def page(message: str = "") -> str:
        return f"""<!doctype html><html lang='ja'><meta charset='utf-8'><title>{args.method} アノテーション</title><style>body{{font-family:system-ui,sans-serif;margin:20px;max-width:1000px}}video{{width:100%;background:#111}}label{{display:block;margin:9px 0}}input,select{{margin-left:8px;padding:4px}}fieldset{{margin-top:14px}}.note{{background:#fff7d6;padding:10px}}button{{padding:8px 16px;font-size:16px}}</style><h1>{args.method}：試技アノテーション</h1><p class='note'>動画のローカル時刻に {source_start:.6f} 秒を加えた値が原動画時刻です。インパクトフレームは原動画59.94 fpsで保存します。最終確定前に元4K動画で確認してください。</p><p>{message}</p><video controls preload='metadata'><source src='/video' type='video/mp4'></video><form method='post' action='/save'><fieldset><legend>1試技を登録</legend><label>ラリーID <input name='rally_id' required placeholder='例: rally_001'></label><label>試技開始（原動画秒）<input name='start' required inputmode='decimal'></label><label>試技終了（原動画秒）<input name='end' required inputmode='decimal'></label><label>打者ID <input name='hitter' required></label><label>対象前衛ID <input name='front' required></label><label>打者利き手 <select name='hhand'><option>right</option><option>left</option><option>unknown</option></select></label><label>前衛利き手 <select name='fhand'><option>right</option><option>left</option><option>unknown</option></select></label><label>インパクト（原動画秒）<input name='impact' required inputmode='decimal'></label><label>方向（前衛から見た左右）<select name='direction'><option value='left'>left</option><option value='right'>right</option><option value='unknown'>unknown</option></select></label><label>ショット種 <input name='shot_type' value='groundstroke'></label><label>遮蔽 <select name='occlusion'><option value='none'>none</option><option value='partial'>partial</option><option value='severe'>severe</option></select></label><label>確信度 <select name='confidence'><option value='high'>high</option><option value='medium'>medium</option><option value='low'>low</option></select></label><label>状態 <select name='status'><option value='complete'>complete</option><option value='excluded'>excluded</option></select></label><label>除外理由 <input name='exclusion'></label></fieldset><button type='submit'>この試技を保存</button></form></html>"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None: return
        def video(self) -> None:
            size=video.stat().st_size; start,end=0,size-1; match=re.fullmatch(r"bytes=(\d*)-(\d*)",self.headers.get("Range",""))
            if match:
                start=int(match.group(1) or 0); end=min(int(match.group(2) or end),end); status=HTTPStatus.PARTIAL_CONTENT
            else: status=HTTPStatus.OK
            self.send_response(status); self.send_header("Content-Type","video/mp4"); self.send_header("Accept-Ranges","bytes"); self.send_header("Content-Length",str(end-start+1))
            if status==HTTPStatus.PARTIAL_CONTENT:self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with video.open("rb") as f:
                    f.seek(start); remain=end-start+1
                    while remain:
                        data=f.read(min(65536,remain))
                        if not data: break
                        self.wfile.write(data); remain-=len(data)
            except (BrokenPipeError,ConnectionResetError): return
        def do_GET(self) -> None:
            if urlparse(self.path).path=="/":
                data=page().encode(); self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data); return
            if urlparse(self.path).path=="/video": self.video(); return
            self.send_error(404)
        def do_POST(self) -> None:
            if urlparse(self.path).path!="/save": self.send_error(404); return
            form=parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode(),keep_blank_values=True); get=lambda key:form.get(key,[""])[0].strip()
            try:
                impact=float(get("impact")); start=float(get("start")); end=float(get("end"))
                if not (start<=impact<=end): raise ValueError("インパクト時刻が試技区間外です")
                existing=sum(1 for _ in csv.DictReader((ann/"trial_annotations.csv").open(encoding="utf-8")))
                row={"annotation_id":f"{args.session_id}_{existing+1:03d}","session_id":args.session_id,"trial_id":f"{args.video_id}_{args.segment_id}_{get('rally_id')}","video_id":args.video_id,"segment_id":args.segment_id,"rally_id":get("rally_id"),"source_start_timestamp_sec":get("start"),"source_end_timestamp_sec":get("end"),"hitter_id":get("hitter"),"target_front_player_id":get("front"),"hitter_handedness":get("hhand"),"front_player_handedness":get("fhand"),"impact_timestamp_sec":get("impact"),"impact_source_frame":str(round(impact*60000/1001)),"actual_ball_direction_court":"","actual_ball_direction_relative_to_front":get("direction"),"shot_type":get("shot_type"),"starting_court_side":"","occlusion_flag":get("occlusion"),"exclusion_reason":get("exclusion"),"annotation_confidence":get("confidence"),"completion_status":get("status")}
                append_csv(ann/"trial_annotations.csv",TRIAL_FIELDS,row); self.send_response(303); self.send_header("Location","/"); self.end_headers()
            except ValueError as e:
                data=page(f"<b>保存しませんでした：{e}</b>").encode(); self.send_response(400); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
    print(f"Open http://127.0.0.1:{args.port}/"); ThreadingHTTPServer(("127.0.0.1",args.port),Handler).serve_forever(); return 0


if __name__ == "__main__": raise SystemExit(main())
