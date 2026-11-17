"""Serve one AI-assisted, human-confirmed motion-prediction event at a time.

The BallTrack timestamps are only navigation aids.  A reviewer must verify the
rear-court groundstroke and final impact frame from the video before saving an
included event.
"""

from __future__ import annotations

import argparse
import csv
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2


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
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def options(values: tuple[str, ...], selected: str) -> str:
    return "".join(f"<option value='{value}' {'selected' if value == selected else ''}>{value}</option>" for value in values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--port", type=int, default=8778)
    parser.add_argument(
        "--review-mode", choices=("candidate", "all_frames"), default="candidate",
        help="candidate opens AI suggestions; all_frames is a completely manual one-output-frame-at-a-time review.",
    )
    parser.add_argument(
        "--queue-path", type=Path, default=None,
        help="CSV queue to review. Defaults to the original pilot queue.",
    )
    parser.add_argument(
        "--proposal-path", type=Path, default=None,
        help="Unverified BallTrack-contact proposal CSV. Defaults to the original pilot proposals.",
    )
    parser.add_argument(
        "--clip-dir", type=Path, default=None,
        help="Directory containing candidate clips. Defaults to the original pilot clip directory.",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    queue_path = args.queue_path or (project / "data" / "segments" / f"{args.video_id}_motion_prediction_pilot_queue.csv")
    events_path = project / "annotations" / "motion_prediction_events.csv"
    proposal_path = args.proposal_path or (project / "analysis" / f"{args.video_id}_motion_prediction_impact_candidates.csv")
    clip_dir = args.clip_dir or (project / "data" / "segments" / "motion_prediction_pilot" / args.video_id)
    queue_path = queue_path.resolve()
    proposal_path = proposal_path.resolve()
    clip_dir = clip_dir.resolve()
    if not queue_path.is_file():
        raise SystemExit(f"Queue not found: {queue_path}")
    if args.review_mode == "candidate" and not proposal_path.is_file():
        raise SystemExit("Impact-proposal CSV not found. Run script 45, or use --review-mode all_frames.")
    if not clip_dir.is_dir():
        raise SystemExit(f"Clip directory not found: {clip_dir}")

    def frame_path(candidate_id: str, frame: int) -> Path:
        return project / "data" / "balltrack" / "racketvision" / args.video_id / candidate_id / "frames" / f"{frame:06d}.jpg"

    def difference_from_previous(candidate_id: str, frame: int) -> float | None:
        """Mean absolute pixel difference from the immediately previous output frame.

        The broadcast is nominally 59.94 fps, but some intervals contain
        repeated source pictures.  The score is deliberately shown to the
        reviewer instead of silently treating every container frame as an
        independent observation.
        """
        if frame <= 0:
            return None
        before = cv2.imread(str(frame_path(candidate_id, frame - 1)))
        current = cv2.imread(str(frame_path(candidate_id, frame)))
        if before is None or current is None or before.shape != current.shape:
            return None
        return float(cv2.mean(cv2.absdiff(before, current))[0])

    def nearest_visually_changed_frame(candidate_id: str, center: int, direction: int, max_frame: int) -> int | None:
        """Find the nearest later/earlier decoded frame that differs from center."""
        reference = cv2.imread(str(frame_path(candidate_id, center)))
        if reference is None:
            return None
        for frame in range(center + direction, (max_frame + 1 if direction > 0 else -1), direction):
            image = cv2.imread(str(frame_path(candidate_id, frame)))
            if image is not None and image.shape == reference.shape:
                # 0.20 clearly separates the repeated-picture pairs in this clip
                # from a new broadcast picture (about 0.05 vs. 0.4 in the pilot).
                if float(cv2.mean(cv2.absdiff(reference, image))[0]) >= 0.20:
                    return frame
        return None

    def document(page_number: int, message: str = "", inspect: int | None = None,
                 viewed_frame: int | None = None, selected_frame: int | None = None,
                 manual_sec: float | None = None) -> str:
        queue = read(queue_path)
        page_number = max(1, min(page_number, len(queue)))
        row = queue[page_number - 1]
        candidate_id = row["candidate_id"]
        event_id = f"{candidate_id}_event_001"
        old = {item["event_id"]: item for item in read(events_path)}.get(event_id, {})
        status = old.get("inclusion_status", "unreviewed")
        existing_offset = ""
        if selected_frame is not None:
            existing_offset = f"{selected_frame / FPS:.6f}"
        elif old.get("impact_timestamp_sec"):
            existing_offset = f"{float(old['impact_timestamp_sec']) - float(row['source_start_timestamp_sec']):.6f}"
        proposals = ([item for item in read(proposal_path) if item["candidate_id"] == candidate_id]
                     if args.review_mode == "candidate" else [])
        proposal_buttons = "".join(
            f"<a class='proposal' href='/?page={page_number}&inspect={item['local_frame']}&view={item['local_frame']}'>"
            f"候補 {float(item['video_offset_sec']):.3f}秒 / F{item['local_frame']} を開く</a>"
            for item in proposals
        ) or "<span>候補なし（動画を通常再生して確認）</span>"
        inspected = next((item for item in proposals if int(item["local_frame"]) == inspect), None)
        frame_strip = ""
        review_navigation = ""
        if args.review_mode == "all_frames":
            capture = cv2.VideoCapture(str(clip_dir / f"{candidate_id}.mp4"))
            max_frame = max(0, round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
            capture.release()
            saved_frame = None
            if old.get("impact_timestamp_sec"):
                saved_frame = round((float(old["impact_timestamp_sec"]) - float(row["source_start_timestamp_sec"])) * FPS)
            center = max(0, min(viewed_frame if viewed_frame is not None else (saved_frame or 0), max_frame))
            selected_query = f"&impactframe={selected_frame}" if selected_frame is not None else ""
            def all_frame_link(frame: int) -> str:
                return f"/?page={page_number}&view={frame}{selected_query}"
            difference = difference_from_previous(candidate_id, center)
            repeat_note = "（直前とほぼ同一画像）" if difference is not None and difference < 0.20 else ""
            set_current = f"/?page={page_number}&view={center}&impactframe={center}"
            frame_controls = (
                f"<a id='prev-frame' class='step' href='{all_frame_link(center - 1)}'>← 1出力フレーム（F{center - 1}）</a>" if center > 0 else ""
            ) + (
                f"<a id='next-frame' class='step' href='{all_frame_link(center + 1)}'>1出力フレーム（F{center + 1}）→</a>" if center < max_frame else ""
            )
            selected_note = "<strong>このフレームをインパクトとして選択中です。</strong>" if center == selected_frame else ""
            frame_strip = f"""<section class='strip all-frames'><h3>全フレーム手動確認：F{center}/{max_frame}（{center / FPS:.3f}秒）</h3><p>AI候補は使いません。左右の操作または<strong>←／→キー</strong>で、すべての出力フレームを順番に確認してください。<strong>Iキー</strong>は現在フレームをインパクトに設定します。{repeat_note}</p><div class='frame-controls'>{frame_controls}<a id='set-impact-frame' class='set-impact' href='{set_current}'>このフレームをインパクトに設定</a></div>{selected_note}<img class='single-frame' src='/frame/{candidate_id}/{center}.jpg' alt='frame {center}'></section><script>document.addEventListener('keydown', function(event) {{const tag = document.activeElement && document.activeElement.tagName; if (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(tag)) return; const id = event.key === 'ArrowLeft' ? 'prev-frame' : (event.key === 'ArrowRight' ? 'next-frame' : (event.key.toLowerCase() === 'i' ? 'set-impact-frame' : '')); const target = id && document.getElementById(id); if (target) {{event.preventDefault(); window.location.assign(target.href);}}}});</script>"""
            review_navigation = "<h3>全フレーム手動確認</h3><p>動画を再生して対象打球の近くまで進み、下の大きな静止画を1出力フレームずつ送ります。AI候補は表示・使用しません。</p>"
        # A BallTrack suggestion is not required.  When the reviewer has a
        # rough video time, manual_sec opens the same frame browser without
        # implying that the ball tracker saw that contact.
        if args.review_mode == "candidate" and (inspected or manual_sec is not None):
            manual_mode = inspected is None
            anchor = int(inspected["local_frame"]) if inspected else round(manual_sec * FPS)
            capture = cv2.VideoCapture(str(clip_dir / f"{candidate_id}.mp4"))
            max_frame = max(0, round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
            capture.release()
            center = max(0, min(viewed_frame if viewed_frame is not None else anchor, max_frame))
            selected_query = f"&impactframe={selected_frame}" if selected_frame is not None else ""
            def view_link(frame: int) -> str:
                if manual_mode:
                    return f"/?page={page_number}&manual_sec={manual_sec:.6f}&view={frame}{selected_query}"
                return f"/?page={page_number}&inspect={anchor}&view={frame}{selected_query}"
            previous_changed = nearest_visually_changed_frame(candidate_id, center, -1, max_frame)
            next_changed = nearest_visually_changed_frame(candidate_id, center, 1, max_frame)
            frame_controls = (
                f"<a class='step' href='{view_link(center - 1)}'>← 1出力フレーム（F{center - 1}）</a>" if center > 0 else ""
            ) + (
                f"<a class='step' href='{view_link(center + 1)}'>1出力フレーム（F{center + 1}）→</a>" if center < max_frame else ""
            ) + (
                f"<a class='step changed' href='{view_link(previous_changed)}'>← 前の異なる画像（F{previous_changed}）</a>" if previous_changed is not None else ""
            ) + (
                f"<a class='step changed' href='{view_link(next_changed)}'>次の異なる画像（F{next_changed}）→</a>" if next_changed is not None else ""
            )
            cards = []
            for frame in range(max(0, center - 4), min(max_frame, center + 4) + 1):
                picked = " selected" if frame == selected_frame else ""
                difference = difference_from_previous(candidate_id, frame)
                repeat_note = "" if difference is None or difference >= 0.20 else "<br><small>直前とほぼ同一</small>"
                image_url = f"/frame/{candidate_id}/{frame}.jpg" if manual_mode else f"/crop/{candidate_id}/{anchor}/{frame}.jpg"
                card_class = " manual-frame" if manual_mode else ""
                card_url = (
                    f"/?page={page_number}&manual_sec={manual_sec:.6f}&view={center}&impactframe={frame}"
                    if manual_mode else f"/?page={page_number}&inspect={anchor}&view={center}&impactframe={frame}"
                )
                cards.append(
                    f"<a class='frame{card_class}{picked}' href='{card_url}'>"
                    f"<img src='{image_url}' alt='frame {frame}'><br>F{frame}<br>{frame / FPS:.3f}秒{repeat_note}</a>"
                )
            origin = "手動起点（AI候補なし）の全画面" if manual_mode else "AI候補周辺の追跡位置"
            frame_strip = f"<section class='strip'><h3>F{center} の前後4フレーム（{origin}）</h3><p>接触に最も近い画像を押すと、そのフレーム時刻がインパクト欄に入ります。<strong>この放送映像には、隣接する59.94 fps出力フレームがほぼ同一の箇所があります。</strong>「直前とほぼ同一」は別の時間情報ではないため、緑の「異なる画像」操作を使ってください。</p><div class='frame-controls'>{frame_controls}</div><div class='frames'>{''.join(cards)}</div></section>"
        queue_event_ids = {f"{item['candidate_id']}_event_001" for item in queue}
        batch_events = [item for item in read(events_path) if item.get("event_id") in queue_event_ids]
        summary = {
            "included": sum(item.get("inclusion_status") == "included" for item in batch_events),
            "excluded": sum(item.get("inclusion_status") == "excluded" for item in batch_events),
            "needs_review": sum(item.get("inclusion_status") == "needs_review" for item in batch_events),
        }
        review_summary_note = (
            "AI候補は<strong>打球・バウンド・誤追跡を区別しない</strong>ので、必ず映像で確認してください。"
            if args.review_mode == "candidate" else
            "この画面ではAI候補を使用せず、全出力フレームを人手で確認します。"
        )
        return f"""<!doctype html><html lang='ja'><meta charset='utf-8'><title>動作予測イベント確認</title>
<style>body{{font-family:system-ui,sans-serif;background:#f3f4f6;margin:0}}main{{max-width:1200px;margin:auto;padding:18px}}header,article,.strip{{background:#fff;border-radius:10px;padding:16px;margin-bottom:14px}}video{{width:min(100%,1000px);background:#111;display:block}}label{{display:block;margin-top:10px;font-weight:600}}input,select{{margin-left:8px;font-size:16px;padding:4px;min-width:220px}}.proposal,.step,.set-impact{{display:inline-block;margin:4px;padding:7px;background:#eaf2ff;border:1px solid #6b8fc5;border-radius:5px;color:#111;text-decoration:none}}.set-impact{{background:#fff2cc;border-color:#9d7f00;font-weight:700}}.step.changed{{background:#e4f8e8;border-color:#4c9a5c}}.manual{{margin:10px 0;padding:10px;background:#eef8ff;border-radius:6px}}.warn{{background:#fff5df;padding:10px;border-radius:6px}}nav a{{margin-right:14px}}.frames{{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px}}.frame{{width:min(100%,500px);border:2px solid #ddd;color:#111;text-decoration:none;padding:5px;background:#fff;text-align:center;font-size:18px}}.frame.manual-frame{{width:min(100%,1000px)}}.frame.selected{{border-color:#d22;background:#fff0f0}}.frame img{{width:100%;display:block}}.single-frame{{width:100%;max-width:1100px;display:block;margin-top:12px;border:2px solid #333}}small{{color:#9b1c1c}}</style>
<main><header><h1>動作予測：イベント確認（{page_number}/{len(queue)}）</h1><p><strong>この20区間内の進捗：</strong>採用済み {summary['included']}、除外 {summary['excluded']}、要確認 {summary['needs_review']}。{review_summary_note}</p><nav>{'' if page_number == 1 else f"<a href='/?page={page_number - 1}'>← 前</a>"}<a href='/?page={page_number + 1 if page_number < len(queue) else len(queue)}'>次 →</a></nav><p>{message}</p></header>
<article><h2>{html.escape(candidate_id)}</h2><p>原動画 {float(row['source_start_timestamp_sec']):.3f}–{float(row['source_end_timestamp_sec']):.3f}秒。対象は、<strong>奥側後衛が打つグラウンドストローク</strong>と、その直後の<strong>手前側前衛</strong>の横移動です。サーブ・ボレー・ロブ・遮蔽は除外します。</p>
<video controls preload='metadata' src='/clip/{candidate_id}.mp4'></video><p>動画は全区間の確認用です。フレーム単位のインパクト候補は、下の拡大静止画で選びます。</p>
{review_navigation if args.review_mode == 'all_frames' else f"<h3>AIの未確認候補（リンクを押すと前後フレームを開く）</h3><div>{proposal_buttons}</div><form class='manual' method='get'><input type='hidden' name='page' value='{page_number}'><strong>AI候補がない場面：</strong><label>動画内のおおよその時刻（秒）<input name='manual_sec' type='number' min='0' max='{float(row['duration_sec']):.3f}' step='0.001' value='{'' if manual_sec is None else f'{manual_sec:.3f}'}' placeholder='例: 4.700'><button>この時刻の前後フレームを開く</button></label><small>動画の再生バーに表示される時刻でよい。ここではAI候補を作らず、指定時刻から人がフレームをたどる。</small></form>"}
{frame_strip}
<form method='post' action='/save?page={page_number}'><label>判定 <select name='status'>{options(STATUSES, status)}</select></label><label>インパクトの動画内時刻（秒） <input id='impact' name='impact' value='{existing_offset}' placeholder='採用時は必須'></label><label>打者ID <input name='hitter' value='{html.escape(old.get('opponent_hitter_id', 'far_rear'))}'></label><label>対象前衛ID <input name='front' value='{html.escape(old.get('target_front_player_id', 'near_front'))}'></label><label>打者Pose取得可 <select name='hpose'>{options(YES_NO_UNKNOWN, old.get('pose_hitter_available', 'unknown'))}</select></label><label>前衛Pose取得可 <select name='fpose'>{options(YES_NO_UNKNOWN, old.get('pose_front_available', 'unknown'))}</select></label><label>遮蔽 <select name='occlusion'>{options(OCCLUSION, old.get('occlusion_flag', 'none'))}</select></label><label>メモ／除外理由 <input name='note' value='{html.escape(old.get('notes', old.get('exclusion_reason', '')))}'></label><p class='warn'>採用するのは、インパクト前0.5秒から後0.3秒まで定点画角・打者・手前前衛が確認できる場面のみです。</p><button>このイベントを保存</button></form></article></main></html>"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                try:
                    page_number = int(parse_qs(parsed.query).get("page", ["1"])[0])
                except ValueError:
                    page_number = 1
                try:
                    inspect = int(parse_qs(parsed.query).get("inspect", [""])[0])
                except ValueError:
                    inspect = None
                try:
                    viewed_frame = int(parse_qs(parsed.query).get("view", [""])[0])
                except ValueError:
                    viewed_frame = None
                try:
                    selected_frame = int(parse_qs(parsed.query).get("impactframe", [""])[0])
                except ValueError:
                    selected_frame = None
                try:
                    manual_sec = float(parse_qs(parsed.query).get("manual_sec", [""])[0])
                except ValueError:
                    manual_sec = None
                if manual_sec is not None and manual_sec < 0:
                    manual_sec = None
                if manual_sec is not None:
                    # The manual path must not inherit an AI proposal merely
                    # because the user opened it from a proposal page.
                    inspect = None
                data = document(page_number, inspect=inspect, viewed_frame=viewed_frame, selected_frame=selected_frame, manual_sec=manual_sec).encode()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if parsed.path.startswith("/clip/"):
                path = clip_dir / Path(parsed.path.removeprefix("/clip/")).name
                if path.is_file():
                    data = path.read_bytes(); self.send_response(200); self.send_header("Content-Type", "video/mp4"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if parsed.path.startswith("/crop/"):
                parts = parsed.path.removeprefix("/crop/").split("/")
                if len(parts) == 3 and parts[2].endswith(".jpg"):
                    candidate_id, inspected_frame, target_frame = parts[0], int(parts[1]), int(parts[2].removesuffix(".jpg"))
                    proposal = next((item for item in read(proposal_path) if item["candidate_id"] == candidate_id and int(item["local_frame"]) == inspected_frame), None)
                    image_path = project / "data" / "balltrack" / "racketvision" / args.video_id / candidate_id / "frames" / f"{target_frame:06d}.jpg"
                    if proposal and image_path.is_file():
                        image = cv2.imread(str(image_path))
                        if image is not None:
                            center_x, center_y = round(float(proposal["x_px"])), round(float(proposal["y_px"]))
                            radius = 240; left, top = max(0, center_x - radius), max(0, center_y - radius)
                            right, bottom = min(image.shape[1], center_x + radius), min(image.shape[0], center_y + radius)
                            crop = image[top:bottom, left:right]
                            cv2.putText(crop, f"F{target_frame}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                            ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                            if ok:
                                data = encoded.tobytes(); self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if parsed.path.startswith("/frame/"):
                parts = parsed.path.removeprefix("/frame/").split("/")
                if len(parts) == 2 and parts[1].endswith(".jpg"):
                    candidate_id, frame_name = parts[0], parts[1].removesuffix(".jpg")
                    try:
                        image_path = frame_path(candidate_id, int(frame_name))
                    except ValueError:
                        image_path = Path()
                    if image_path.is_file():
                        data = image_path.read_bytes(); self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            self.send_error(404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/save":
                self.send_error(404); return
            form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode(), keep_blank_values=True)
            try:
                page_number = int(parse_qs(parsed.query).get("page", ["1"])[0])
            except ValueError:
                page_number = 1
            queue = read(queue_path)
            page_number = max(1, min(page_number, len(queue)))
            row = queue[page_number - 1]
            candidate_id = row["candidate_id"]
            status = form.get("status", ["unreviewed"])[0]
            if status not in STATUSES:
                self.send_error(400, "Invalid status"); return
            fields = next(csv.reader(events_path.open(encoding="utf-8")))
            existing = {item["event_id"]: item for item in read(events_path)}
            event_id = f"{candidate_id}_event_001"
            output = existing.get(event_id, {field: "" for field in fields})
            note = form.get("note", [""])[0].strip()
            if status == "included":
                try:
                    offset = float(form.get("impact", [""])[0])
                except ValueError:
                    offset = -1
                if not 0 <= offset <= float(row["duration_sec"]):
                    data = document(page_number, "<strong>保存しませんでした：採用には動画内インパクト時刻が必要です。</strong>").encode()
                    self.send_response(400); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
                impact = float(row["source_start_timestamp_sec"]) + offset
                output.update({"event_id": event_id, "video_id": args.video_id, "play_window_id": row["play_window_id"], "rally_id": row["play_window_id"], "source_fps": "60000/1001", "opponent_hitter_id": form.get("hitter", [""])[0].strip(), "opponent_hitter_role": "rear_groundstroke", "target_front_player_id": form.get("front", [""])[0].strip(), "target_front_role": "near_net_at_impact", "impact_source_frame": str(round(impact * FPS)), "impact_timestamp_sec": f"{impact:.6f}", "pre_window_start_sec": f"{impact - 0.5:.6f}", "pre_window_end_sec": f"{impact:.6f}", "horizon_100_frame": str(round((impact + 0.1) * FPS)), "horizon_200_frame": str(round((impact + 0.2) * FPS)), "horizon_300_frame": str(round((impact + 0.3) * FPS)), "court_geometry_id": "match1_fixed_main_camera", "pose_hitter_available": form.get("hpose", ["unknown"])[0], "pose_front_available": form.get("fpose", ["unknown"])[0], "front_track_status": "unprocessed", "occlusion_flag": form.get("occlusion", ["none"])[0], "camera_stability_flag": "fixed_main_camera_reviewed", "inclusion_status": "included", "exclusion_reason": "", "annotation_confidence": "human_video_review", "annotator_id": "", "notes": note})
            else:
                output.update({"event_id": event_id, "video_id": args.video_id, "play_window_id": row["play_window_id"], "rally_id": row["play_window_id"], "inclusion_status": status, "exclusion_reason": note if status == "excluded" else "", "notes": note})
            existing[event_id] = output
            write(events_path, fields, list(existing.values()))
            self.send_response(303); self.send_header("Location", f"/?page={page_number}"); self.end_headers()

    print(f"Open http://127.0.0.1:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
