"""Serve the complete source match with playback and frame-accurate review aids.

This viewer never creates a full frame-image dump. The 4K source is streamed
with HTTP range requests, and a still image is decoded only for the requested
output frame. It can save a human-confirmed rear-impact frame, but it is not
an AI event detector and it does not decide whether that event is usable for
the later movement analysis.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from datetime import datetime, timezone

import cv2


CHUNK_BYTES = 1024 * 1024
IMPACT_FIELDS = [
    "impact_id", "video_id", "impact_source_frame", "impact_timestamp_sec", "source_fps",
    "hitter_role", "hitter_camera_end", "target_front_camera_end", "hitter_player_id",
    "target_front_player_id", "shot_context", "target_front_depth_state",
    "primary_cohort_status", "annotation_status", "inclusion_status", "annotation_mode",
    "annotator_id", "note", "created_at_utc",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write human labels atomically so a browser interruption cannot truncate them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=IMPACT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class SourceFrames:
    """Decode an output frame on demand, retaining sequential stepping state."""

    def __init__(self, video: Path) -> None:
        self.video = video
        self.capture = cv2.VideoCapture(str(video))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = round(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = round(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = round(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.lock = threading.Lock()
        self.next_frame: int | None = None
        if self.fps <= 0 or self.frame_count <= 0 or self.width <= 0 or self.height <= 0:
            self.capture.release()
            raise RuntimeError(f"Cannot read video metadata: {video}")

    def jpeg(self, frame: int) -> bytes | None:
        """Return one output frame as JPEG without writing it to disk.

        For the common next-frame request, OpenCV continues decoding from the
        prior frame. For any jump it seeks to the requested output-frame index.
        """
        if not 0 <= frame < self.frame_count:
            return None
        with self.lock:
            if frame != self.next_frame:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = self.capture.read()
            if not ok or image is None:
                return None
            self.next_frame = frame + 1
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return encoded.tobytes() if ok else None


def parse_range(value: str | None, total: int) -> tuple[int, int] | None:
    """Parse a single HTTP byte range; multiple ranges are intentionally unsupported."""
    if not value:
        return None
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if not match:
        return None
    start_text, end_text = match.groups()
    if not start_text and not end_text:
        return None
    if not start_text:
        length = int(end_text)
        if length <= 0:
            return None
        return max(0, total - length), total - 1
    start = int(start_text)
    end = int(end_text) if end_text else total - 1
    if start >= total or end < start:
        return None
    return start, min(end, total - 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original complete match video")
    parser.add_argument("--video-id", default=None, help="Stable identifier saved with each impact; defaults to input filename stem")
    parser.add_argument("--annotation-output", type=Path, default=None, help="CSV for human-confirmed rear-impact frames")
    parser.add_argument("--port", type=int, default=8785)
    args = parser.parse_args()
    video = args.input.resolve()
    if not video.is_file():
        raise SystemExit(f"Video not found: {video}")
    video_id = args.video_id or video.stem
    project = Path(__file__).resolve().parents[1]
    impact_output = (args.annotation_output or (project / "annotations" / f"{video_id}_full_match_rear_impact_review.csv")).resolve()
    source = SourceFrames(video)
    duration = source.frame_count / source.fps

    def clamp_frame(frame: int) -> int:
        return max(0, min(frame, source.frame_count - 1))

    def page(frame: int, message: str = "") -> str:
        frame = clamp_frame(frame)
        timestamp = frame / source.fps
        impacts = read_csv(impact_output)
        impact_id = f"{video_id}_rear_impact_{frame:06d}"
        existing_impact = next((row for row in impacts if row["impact_id"] == impact_id), None)
        saved_summary = f"この動画で確定済み {len(impacts)}件"
        safe_message = f"<p>{html.escape(message)}</p>" if message else ""
        saved_frames = sorted(
            int(row["impact_source_frame"])
            for row in impacts
            if row.get("impact_source_frame", "").isdigit()
        )
        def frame_link(offset: int, label: str) -> str:
            target = clamp_frame(frame + offset)
            return f"<a class='jump' data-offset='{offset}' href='/?frame={target}'>{label}</a>"
        time_value = f"{timestamp:.3f}"
        return f"""<!doctype html><html lang='ja'><meta charset='utf-8'>
<title>試合全体：フレーム確認</title>
<style>
body{{margin:0;background:#14161a;color:#f5f5f5;font-family:system-ui,sans-serif}}main{{max-width:1900px;margin:auto;padding:16px}}section{{background:#22262d;border-radius:10px;padding:14px;margin-bottom:14px}}h1,h2,p{{margin-top:0}}video{{display:block;width:100%;max-height:82vh;background:#000}}button,input{{font:inherit;padding:7px;margin:3px}}.controls,.frame-controls{{display:flex;flex-wrap:wrap;gap:5px;align-items:center}}.jump{{display:inline-block;color:#111;background:#dbeafe;padding:8px 10px;border-radius:5px;text-decoration:none}}.jump.disabled{{pointer-events:none;opacity:.4}}.frame-info{{font-weight:700;margin-bottom:8px}}.frame-stage{{position:relative;background:#000;min-height:200px}}#frame-image{{width:100%;display:block;background:#000;max-height:90vh;object-fit:contain}}.frame-stage:fullscreen{{width:100vw;height:100vh;display:grid;place-items:center}}.frame-stage:fullscreen #frame-image{{width:100vw;height:100vh;max-height:none;object-fit:contain}}.frame-overlay{{display:none}}.frame-stage:fullscreen .frame-overlay{{display:block;position:fixed;z-index:10;left:0;right:0;bottom:0;padding:14px 20px;background:linear-gradient(transparent,rgba(0,0,0,.85));font-size:18px;font-weight:700;text-align:center}}.hint{{color:#cbd5e1}}.status{{font-variant-numeric:tabular-nums;font-weight:700}}.save{{background:#bbf7d0;border:2px solid #15803d;font-weight:700;font-size:1.08rem}}.danger{{background:#fecaca;border:2px solid #b91c1c}}.saved{{background:#14532d;padding:9px;border-radius:6px}}
</style>
<main><section><h1>試合全体：再生・コマ送り</h1>
<p class='hint'>原動画: {html.escape(video.name)} ／ {source.width}×{source.height} ／ {source.fps:.5f} fps ／ {duration:.3f}秒。全{source.frame_count:,}フレームを保存せず、要求した静止画だけを元4Kから復号します。</p>
<video id='player' controls preload='metadata' src='/video'></video>
<div class='controls'><button onclick="document.getElementById('player').requestFullscreen()">動画を全画面表示</button><button onclick="enterFrameFullscreen()">静止画を全画面表示</button><button onclick="openCurrentVideoTime()">動画の現在時刻を静止画へ</button><button onclick="shiftVideo(-30)">動画 -30秒</button><button onclick="shiftVideo(-10)">動画 -10秒</button><button onclick="shiftVideo(-5)">動画 -5秒</button><button onclick="shiftVideo(5)">動画 +5秒</button><button onclick="shiftVideo(10)">動画 +10秒</button><button onclick="shiftVideo(30)">動画 +30秒</button><label>原動画時刻（秒）<input id='time-input' type='number' min='0' max='{duration:.3f}' step='0.001' value='{time_value}'></label><button onclick="openTypedTime()">この時刻へ移動</button></div>
</section>
<section><h2>静止画コマ送り</h2><p class='frame-info'>出力フレーム <span id='frame-number'>F{frame}</span> / {source.frame_count - 1}　原動画 <span id='frame-time'>{timestamp:.6f}</span>秒</p>
<div class='frame-controls'>{frame_link(-1800, '−30秒')}{frame_link(-600, '−10秒')}{frame_link(-300, '−5秒')}{frame_link(-60, '−1秒')}{frame_link(-15, '−15F')}{frame_link(-1, 'A: −1F')}{frame_link(1, 'D: +1F')}{frame_link(15, '+15F')}{frame_link(60, '+1秒')}{frame_link(300, '+5秒')}{frame_link(600, '+10秒')}{frame_link(1800, '+30秒')}</div>
<p class='hint'>静止画を全画面表示した後も <strong>A</strong>＝前の1F、<strong>D</strong>＝次の1F、<strong>Enter</strong>＝このフレームを確定・保存です。Shift+A/D は±15F、Ctrl+A/D は±60F、PageUp/PageDown は±5秒です。Escで全画面を終了します。</p>
<div id='frame-stage' class='frame-stage'><img id='frame-image' src='/frame/{frame}.jpg' alt='原動画フレーム {frame}'><div id='frame-overlay' class='frame-overlay'></div></div>
</section>
<section><h2>相手後衛インパクトの確定</h2><p id='saved-summary'><strong>{saved_summary}</strong>。ラケットとボールが最も近接・接触している<strong>原動画出力フレーム</strong>だけを保存します。打球方向、選手ID、ロブ・レシーブ、前衛位置などの細かい設定は後から追加します。</p><p class='hint'>全画面で <strong>Enter</strong> を押すと保存します。誤記録を直す場合だけ、下の削除ボタンを使ってください。</p><form id='impact-form' method='post' action='/save-impact'><input id='impact-frame' type='hidden' name='frame' value='{frame}'><button id='save-button' class='save' type='submit'>Enter：このフレームをインパクトとして確定・保存</button><button id='delete-button' class='danger' type='button' {'style="display:none"' if not existing_impact else ''} onclick='deleteCurrentImpact()'>このフレームの記録を削除</button></form><p id='save-status' class='saved' {'style="display:none"' if not existing_impact else ''}>{'このフレームは保存済みです。' if existing_impact else ''}</p>{safe_message}</section>
</main>
<script>
const fps = {source.fps:.12f};
let currentFrame = {frame};
const lastFrame = {source.frame_count - 1};
const savedFrames = new Set({json.dumps(saved_frames)});
let isSaving = false;
const player = document.getElementById('player');
player.addEventListener('loadedmetadata', () => {{ player.currentTime = {timestamp:.9f}; }});
function shiftVideo(seconds) {{ player.pause(); player.currentTime = Math.max(0, Math.min(player.duration || {duration:.9f}, player.currentTime + seconds)); }}
function goToFrame(frame) {{ currentFrame = Math.max(0, Math.min(lastFrame, Math.round(frame))); const image = document.getElementById('frame-image'); image.src = '/frame/' + currentFrame + '.jpg?cache=' + Date.now(); image.alt = '原動画フレーム ' + currentFrame; document.getElementById('frame-number').textContent = 'F' + currentFrame; document.getElementById('frame-time').textContent = (currentFrame / fps).toFixed(6); document.getElementById('impact-frame').value = currentFrame; history.replaceState(null, '', '/?frame=' + currentFrame); updateSaveState(); }}
function openCurrentVideoTime() {{ goToFrame(player.currentTime * fps); }}
function openTypedTime() {{ const t = Number(document.getElementById('time-input').value); if (Number.isFinite(t)) goToFrame(Math.max(0, Math.min({duration:.9f}, t)) * fps); }}
function enterFrameFullscreen() {{ document.getElementById('frame-stage').requestFullscreen(); }}
function updateSaveState(message = '') {{ const saved = savedFrames.has(currentFrame); const saveButton = document.getElementById('save-button'); const deleteButton = document.getElementById('delete-button'); const status = document.getElementById('save-status'); saveButton.disabled = saved; saveButton.textContent = saved ? 'このフレームは保存済みです' : 'Enter：このフレームをインパクトとして確定・保存'; deleteButton.style.display = saved ? '' : 'none'; status.style.display = saved || message ? '' : 'none'; status.textContent = message || (saved ? 'このフレームは保存済みです。' : ''); document.getElementById('frame-overlay').textContent = 'F' + currentFrame + '　A: −1F　D: +1F　Enter: 確定' + (saved ? '　✓ 保存済み' : ''); }}
async function saveCurrentImpact() {{ if (savedFrames.has(currentFrame) || isSaving) return; isSaving = true; const body = new URLSearchParams({{frame: String(currentFrame), ajax: '1'}}); try {{ const response = await fetch('/save-impact', {{method: 'POST', headers: {{'Content-Type': 'application/x-www-form-urlencoded'}}, body}}); const data = await response.json(); if (!response.ok || !data.ok) throw new Error(data.message || '保存できませんでした。'); savedFrames.add(currentFrame); updateSaveState(data.message); }} catch (error) {{ updateSaveState('保存エラー：' + error.message); }} finally {{ isSaving = false; }} }}
async function deleteCurrentImpact() {{ if (!savedFrames.has(currentFrame)) return; const body = new URLSearchParams({{frame: String(currentFrame), impact_id: '{video_id}_rear_impact_' + String(currentFrame).padStart(6, '0'), ajax: '1'}}); try {{ const response = await fetch('/delete-impact', {{method: 'POST', headers: {{'Content-Type': 'application/x-www-form-urlencoded'}}, body}}); const data = await response.json(); if (!response.ok || !data.ok) throw new Error(data.message || '削除できませんでした。'); savedFrames.delete(currentFrame); updateSaveState(data.message); }} catch (error) {{ updateSaveState('削除エラー：' + error.message); }} }}
document.getElementById('impact-form').addEventListener('submit', (event) => {{ event.preventDefault(); saveCurrentImpact(); }});
document.querySelectorAll('.jump').forEach((link) => {{ link.addEventListener('click', (event) => {{ event.preventDefault(); goToFrame(currentFrame + Number(link.dataset.offset)); }}); }});
document.addEventListener('keydown', (event) => {{ const tag = document.activeElement && document.activeElement.tagName; if (['INPUT','TEXTAREA','BUTTON','VIDEO'].includes(tag)) return; if (event.key === 'Enter') {{ event.preventDefault(); saveCurrentImpact(); return; }} let delta = 0; const key = event.key.toLowerCase(); if (key === 'a') delta = event.ctrlKey ? -60 : (event.shiftKey ? -15 : -1); if (key === 'd') delta = event.ctrlKey ? 60 : (event.shiftKey ? 15 : 1); if (event.key === 'PageUp') delta = -300; if (event.key === 'PageDown') delta = 300; if (delta) {{ event.preventDefault(); goToFrame(currentFrame + delta); }} }});
updateSaveState();
</script></html>"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return

        def send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def send_video(self) -> None:
            size = video.stat().st_size
            selected = parse_range(self.headers.get("Range"), size)
            if self.headers.get("Range") and selected is None:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start, end = selected if selected is not None else (0, size - 1)
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT if selected is not None else HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if selected is not None:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with video.open("rb") as file:
                file.seek(start)
                remaining = length
                while remaining:
                    block = file.read(min(CHUNK_BYTES, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                try:
                    requested = int(parse_qs(parsed.query).get("frame", ["0"])[0])
                except ValueError:
                    requested = 0
                message = parse_qs(parsed.query).get("message", [""])[0]
                data = page(requested, message).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers(); self.wfile.write(data)
                return
            if parsed.path == "/video":
                self.send_video()
                return
            match = re.fullmatch(r"/frame/(\d+)\.jpg", parsed.path)
            if match:
                image = source.jpeg(int(match.group(1)))
                if image is not None:
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(image)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers(); self.wfile.write(image)
                    return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path not in {"/save-impact", "/delete-impact"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            form = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
            ajax = form.get("ajax", [""])[0] == "1"
            try:
                frame = clamp_frame(int(form.get("frame", [""])[0]))
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid output frame")
                return
            impact_id = f"{video_id}_rear_impact_{frame:06d}"
            impacts = read_csv(impact_output)
            if parsed.path == "/save-impact":
                if not any(row["impact_id"] == impact_id for row in impacts):
                    impacts.append({
                        "impact_id": impact_id,
                        "video_id": video_id,
                        "impact_source_frame": str(frame),
                        "impact_timestamp_sec": f"{frame / source.fps:.6f}",
                        "source_fps": f"{source.fps:.12f}",
                        "hitter_role": "rear",
                        "hitter_camera_end": "",
                        "target_front_camera_end": "",
                        "hitter_player_id": "",
                        "target_front_player_id": "",
                        "shot_context": "",
                        "target_front_depth_state": "",
                        "primary_cohort_status": "unclassified",
                        "annotation_status": "human_confirmed_rear_impact",
                        "inclusion_status": "unclassified",
                        "annotation_mode": "full_match_frame_viewer",
                        "annotator_id": "",
                        "note": "",
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    })
                    impacts.sort(key=lambda row: int(row["impact_source_frame"]))
                    write_csv(impact_output, impacts)
                    message = "保存しました。"
                else:
                    message = "保存済みです。同一フレームは重複保存しません。"
            else:
                requested_id = form.get("impact_id", [""])[0]
                if requested_id != impact_id:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Impact id does not match frame")
                    return
                retained = [row for row in impacts if row["impact_id"] != impact_id]
                if len(retained) != len(impacts):
                    write_csv(impact_output, retained)
                    message = "このフレームの記録を削除しました。"
                else:
                    message = "削除対象の記録はありません。"
            if ajax:
                self.send_json(HTTPStatus.OK, {"ok": True, "message": message, "frame": frame})
                return
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", f"/?{urlencode({'frame': frame, 'message': message})}")
            self.end_headers()

    print(f"Open http://127.0.0.1:{args.port}/")
    print(f"Source: {video} ({source.frame_count} output frames at {source.fps:.5f} fps)")
    print(f"Impact labels: {impact_output}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
