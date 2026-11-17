"""Serve the motion-prediction pilot queue and save one screened event per clip."""

from __future__ import annotations

import argparse
import csv
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


FPS = 60000 / 1001
STATUSES = ("unreviewed", "included", "excluded", "needs_review")
OCCLUSION = ("none", "partial", "severe")
YES_NO_UNKNOWN = ("yes", "no", "unknown")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def option_values(values: tuple[str, ...], selected: str) -> str:
    return "".join(
        f"<option value='{value}' {'selected' if value == selected else ''}>{value}</option>"
        for value in values
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--port", type=int, default=8775)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    segment_dir = project / "data" / "segments"
    queue_path = segment_dir / f"{args.video_id}_motion_prediction_pilot_queue.csv"
    events_path = project / "annotations" / "motion_prediction_events.csv"
    clip_dir = segment_dir / "motion_prediction_pilot" / args.video_id
    if not queue_path.is_file():
        raise SystemExit(f"Pilot queue not found: {queue_path}")

    def page(message: str = "") -> str:
        event_rows = {row["play_window_id"]: row for row in read(events_path)}
        cards: list[str] = []
        for row in read(queue_path):
            candidate_id = row["candidate_id"]
            old = event_rows.get(row["play_window_id"], {})
            status = old.get("inclusion_status", row["screening_status"])
            balltrack_csv = project / "data" / "balltrack" / "racketvision" / args.video_id / candidate_id / "balltrack_predictions.csv"
            balltrack_link = ""
            if balltrack_csv.is_file():
                predictions = read(balltrack_csv)
                visible = sum(int(prediction["visible"]) for prediction in predictions)
                balltrack_link = f"<p><a href='/balltrack/{candidate_id}.mp4' target='_blank'>BallTrack候補の重ね描画を開く</a>（候補 {visible}/{len(predictions)} フレーム）</p>"
            cards.append(f"""<article data-id='{candidate_id}'>
<h2>{html.escape(candidate_id)}</h2>
<p>候補窓: {html.escape(row['play_window_id'])} ／ 原動画 {float(row['source_start_timestamp_sec']):.3f}–{float(row['source_end_timestamp_sec']):.3f} 秒</p>
<video controls preload='metadata' src='/clip/{candidate_id}.mp4'></video>
{balltrack_link}
<button type='button' onclick='step(this,-1)'>1フレーム戻る</button>
<button type='button' onclick='step(this,1)'>1フレーム進む</button>
<output>動画内 0.000秒</output>
<label>判定 <select name='status__{candidate_id}'>{option_values(STATUSES, status)}</select></label>
<label>打者ID <input name='hitter__{candidate_id}' value='{html.escape(old.get('opponent_hitter_id', ''))}' placeholder='例: far_rear_A'></label>
<label>対象前衛ID <input name='front__{candidate_id}' value='{html.escape(old.get('target_front_player_id', ''))}' placeholder='例: near_front_B'></label>
<label>インパクトの動画内時刻（秒） <input class='offset' name='impact__{candidate_id}' value=''></label>
<button type='button' onclick='copyTime(this)'>現在時刻を入力欄へ</button>
<label>打者Pose取得可 <select name='hpose__{candidate_id}'>{option_values(YES_NO_UNKNOWN, old.get('pose_hitter_available', 'unknown'))}</select></label>
<label>前衛Pose取得可 <select name='fpose__{candidate_id}'>{option_values(YES_NO_UNKNOWN, old.get('pose_front_available', 'unknown'))}</select></label>
<label>遮蔽 <select name='occlusion__{candidate_id}'>{option_values(OCCLUSION, old.get('occlusion_flag', 'none'))}</select></label>
<label>除外理由／メモ <input name='note__{candidate_id}' value='{html.escape(old.get('notes', ''))}'></label>
</article>""")
        return """<!doctype html><html lang='ja'><meta charset='utf-8'><title>動作予測パイロット</title><style>
body{font-family:system-ui,sans-serif;background:#f3f4f6;margin:20px}header{position:sticky;top:0;background:#fff;padding:12px;z-index:2}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:12px;margin-top:16px}article{background:#fff;padding:12px;border-radius:8px}h2{font-size:16px;margin:0}video{width:100%;background:#111;margin:8px 0}label{display:block;margin-top:7px}input{width:68%;margin-left:6px}output{margin-left:8px}</style><script>
const frame=""" + repr(1 / FPS) + """;
function video(button){return button.closest('article').querySelector('video')}
function step(button,dir){const card=button.closest('article'),v=video(button);const current=card.dataset.frame===undefined?Math.round(v.currentTime/frame):Number(card.dataset.frame);const max=Math.max(0,Math.round(v.duration/frame)-1);const next=Math.max(0,Math.min(max,current+dir));card.dataset.frame=next;v.pause();v.currentTime=next*frame;card.querySelector('output').textContent='動画内 '+(next*frame).toFixed(3)+'秒（フレーム '+next+'）';}
function copyTime(button){const card=button.closest('article'),v=video(button);const time=card.dataset.frame===undefined?v.currentTime:Number(card.dataset.frame)*frame;card.querySelector('.offset').value=time.toFixed(6);}
addEventListener('timeupdate',e=>{if(e.target.tagName==='VIDEO'){const card=e.target.closest('article');card.querySelector('output').textContent='動画内 '+e.target.currentTime.toFixed(3)+'秒';}},true);</script>
<form method='post' action='/save'><header><h1>動作予測：候補イベント確認（60fps）</h1><p>各動画から、相手後衛のグラウンドストロークを1件だけ選びます。対象前衛は打者の相手チームのネット側選手です。インパクトは映像で確定し、音声は補助に留めます。該当なし・不明なら除外してください。全窓を採用する必要はありません。</p><p><strong>入力に使うのは打球前0.5秒まで、予測対象は打球後0.1/0.2/0.3秒です。</strong>その全区間で定点画角・人物可視性が必要です。</p><p>""" + message + """</p><button>確認結果を保存</button></header><main>""" + "".join(cards) + "</main></form></html>"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                data = page().encode()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if parsed.path.startswith("/clip/"):
                path = clip_dir / Path(parsed.path.removeprefix("/clip/")).name
                if path.is_file():
                    data = path.read_bytes(); self.send_response(200); self.send_header("Content-Type", "video/mp4"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if parsed.path.startswith("/balltrack/"):
                candidate_id = Path(parsed.path.removeprefix("/balltrack/")).stem
                path = project / "data" / "balltrack" / "racketvision" / args.video_id / candidate_id / "balltrack_overlay.mp4"
                if path.is_file():
                    data = path.read_bytes(); self.send_response(200); self.send_header("Content-Type", "video/mp4"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            self.send_error(404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/save":
                self.send_error(404); return
            form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode(), keep_blank_values=True)
            queue = read(queue_path)
            existing = {row["event_id"]: row for row in read(events_path)}
            fields = list(next(csv.reader(events_path.open(encoding="utf-8"))))
            try:
                for row in queue:
                    candidate_id = row["candidate_id"]
                    status = form.get(f"status__{candidate_id}", ["unreviewed"])[0]
                    if status not in STATUSES:
                        raise ValueError(f"{candidate_id}: invalid status")
                    event_id = f"{candidate_id}_event_001"
                    old = existing.get(event_id, {field: "" for field in fields})
                    impact_offset = form.get(f"impact__{candidate_id}", [""])[0].strip()
                    if status == "included":
                        if not impact_offset:
                            raise ValueError(f"{candidate_id}: 採用にはインパクト時刻が必要です")
                        offset = float(impact_offset)
                        if not 0 <= offset <= float(row["duration_sec"]):
                            raise ValueError(f"{candidate_id}: インパクト時刻が動画外です")
                        impact = float(row["source_start_timestamp_sec"]) + offset
                        old.update({
                            "event_id": event_id, "video_id": args.video_id, "play_window_id": row["play_window_id"],
                            "source_fps": "60000/1001", "opponent_hitter_id": form.get(f"hitter__{candidate_id}", [""])[0].strip(),
                            "opponent_hitter_role": "rear_groundstroke", "target_front_player_id": form.get(f"front__{candidate_id}", [""])[0].strip(),
                            "target_front_role": "net_side_at_impact", "impact_source_frame": str(round(impact * FPS)),
                            "impact_timestamp_sec": f"{impact:.6f}", "pre_window_start_sec": f"{impact - 0.5:.6f}",
                            "pre_window_end_sec": f"{impact:.6f}", "horizon_100_frame": str(round((impact + 0.1) * FPS)),
                            "horizon_200_frame": str(round((impact + 0.2) * FPS)), "horizon_300_frame": str(round((impact + 0.3) * FPS)),
                            "pose_hitter_available": form.get(f"hpose__{candidate_id}", ["unknown"])[0],
                            "pose_front_available": form.get(f"fpose__{candidate_id}", ["unknown"])[0],
                            "occlusion_flag": form.get(f"occlusion__{candidate_id}", ["none"])[0], "camera_stability_flag": "fixed_main_camera_reviewed",
                            "inclusion_status": "included", "exclusion_reason": "", "annotation_confidence": "pilot", "notes": form.get(f"note__{candidate_id}", [""])[0].strip(),
                        })
                    else:
                        old.update({"event_id": event_id, "video_id": args.video_id, "play_window_id": row["play_window_id"], "inclusion_status": status, "exclusion_reason": form.get(f"note__{candidate_id}", [""])[0].strip()})
                    existing[event_id] = old
                write(events_path, fields, list(existing.values()))
            except ValueError as error:
                data = page(f"<b>保存しませんでした：{html.escape(str(error))}</b>").encode()
                self.send_response(400); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            self.send_response(303); self.send_header("Location", "/"); self.end_headers()

    print(f"Open http://127.0.0.1:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
