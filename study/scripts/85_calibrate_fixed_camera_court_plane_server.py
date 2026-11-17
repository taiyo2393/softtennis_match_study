"""One-time manual court-plane calibration for a fixed broadcast camera.

Four court-corner clicks define an image-to-court homography.  Two independent
net/sideline clicks are held out and used only to check the calibration.  The
output coordinate system is normalized: x=0..1 from screen-left sideline to
screen-right sideline, y=0..1 from near baseline to far baseline.  This avoids
assuming official court dimensions while making lateral displacement comparable
between near and far court ends.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


POINT_LABELS = (
    "1. 奥側・画面左のベースラインとダブルスサイドラインの交点",
    "2. 奥側・画面右のベースラインとダブルスサイドラインの交点",
    "3. 手前側・画面右のベースラインとダブルスサイドラインの交点",
    "4. 手前側・画面左のベースラインとダブルスサイドラインの交点",
    "5. ネットと画面左のダブルスサイドラインの交点（検証用）",
    "6. ネットと画面右のダブルスサイドラインの交点（検証用）",
)
WORLD_CORNERS = np.float32(((0, 1), (1, 1), (1, 0), (0, 0)))
WORLD_NET = np.float32(((0, 0.5), (1, 0.5)))


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def make_preview(video: Path, frame_number: int, max_width: int) -> tuple[bytes, int, int, int, int]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise SystemExit(f"Could not read source frame {frame_number}: {video}")
    raw_height, raw_width = frame.shape[:2]
    preview_width = min(raw_width, max_width)
    preview_height = round(raw_height * preview_width / raw_width)
    if preview_width != raw_width:
        frame = cv2.resize(frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, (cv2.IMWRITE_JPEG_QUALITY, 93))
    if not ok:
        raise SystemExit("Could not encode calibration preview")
    return encoded.tobytes(), raw_width, raw_height, preview_width, preview_height


def as_points(raw: str, preview_width: int, preview_height: int) -> list[tuple[float, float]]:
    try:
        candidate = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("点のデータを読み取れませんでした") from error
    if not isinstance(candidate, list) or len(candidate) != 6:
        raise ValueError("6点を順番どおりに指定してください")
    points: list[tuple[float, float]] = []
    for point in candidate:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("座標の形式が不正です")
        x, y = float(point[0]), float(point[1])
        if not (0 <= x <= preview_width and 0 <= y <= preview_height):
            raise ValueError("画像の外側は指定できません")
        points.append((x, y))
    return points


def calibration_payload(
    video: Path,
    source_frame: int,
    raw_width: int,
    raw_height: int,
    preview_width: int,
    preview_height: int,
    preview_points: list[tuple[float, float]],
) -> dict[str, object]:
    scale_x = raw_width / preview_width
    scale_y = raw_height / preview_height
    raw_points = [(x * scale_x, y * scale_y) for x, y in preview_points]
    transform = cv2.getPerspectiveTransform(np.float32(raw_points[:4]), WORLD_CORNERS)
    net_estimate = cv2.perspectiveTransform(np.float32([raw_points[4:]]), transform)[0]
    validation_errors = [math.dist(estimate, expected) for estimate, expected in zip(net_estimate, WORLD_NET)]
    maximum_error = max(validation_errors)
    return {
        "video_path": str(video),
        "source_frame": source_frame,
        "source_frame_indexing": "zero_based_frame_index_used_by_OpenCV",
        "image_size_px": {"width": raw_width, "height": raw_height},
        "preview_size_px": {"width": preview_width, "height": preview_height},
        "point_order": list(POINT_LABELS),
        "preview_points_px": [[round(x, 3), round(y, 3)] for x, y in preview_points],
        "image_points_px": [[round(x, 3), round(y, 3)] for x, y in raw_points],
        "court_coordinates": {
            "definition": "normalized court plane: x 0=screen-left sideline, 1=screen-right sideline; y 0=near baseline, 1=far baseline",
            "corner_points": WORLD_CORNERS.tolist(),
            "held_out_net_points": WORLD_NET.tolist(),
        },
        "image_to_court_homography": transform.tolist(),
        "validation": {
            "method": "net/sideline points 5-6 are not used to fit the homography",
            "mapped_held_out_net_points": np.round(net_estimate, 6).tolist(),
            "errors_in_normalized_court_units": [round(error, 6) for error in validation_errors],
            "maximum_error_in_normalized_court_units": round(maximum_error, 6),
            "acceptance_rule": "maximum error <= 0.03 normalized court units",
            "status": "usable" if maximum_error <= 0.03 else "needs_recheck",
        },
        "analysis_guard": (
            "This fixed-camera calibration permits near/far pooling only after ankle positions "
            "are transformed to the court plane. It does not validate player identity, "
            "impact labels, or causal/anticipation interpretations."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--source-frame", type=int, default=492)
    parser.add_argument("--max-preview-width", type=int, default=1440)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8804)
    args = parser.parse_args()
    if not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")
    if args.source_frame < 0 or args.max_preview_width < 640:
        raise SystemExit("--source-frame must be non-negative and --max-preview-width must be at least 640")
    project = Path(__file__).resolve().parents[1]
    output = args.output or project / "config" / f"{args.video.stem}_court_plane_calibration.json"
    image, raw_width, raw_height, preview_width, preview_height = make_preview(
        args.video, args.source_frame, args.max_preview_width
    )

    def result_html(saved: dict[str, object] | None) -> str:
        if saved is None:
            return "<p class='waiting'>まだ保存されていません。6点をクリックして保存してください。</p>"
        validation = saved["validation"]
        mapped = validation["mapped_held_out_net_points"]
        errors = validation["errors_in_normalized_court_units"]
        status = validation["status"]
        cls = "ok" if status == "usable" else "bad"
        label = "使用可能（ただし固定カメラに限定）" if status == "usable" else "見直しが必要"
        return (
            f"<div class='{cls}'><strong>検証結果：{label}</strong><br>"
            f"ネット左：推定 ({mapped[0][0]:.3f}, {mapped[0][1]:.3f})、誤差 {errors[0]:.3f}<br>"
            f"ネット右：推定 ({mapped[1][0]:.3f}, {mapped[1][1]:.3f})、誤差 {errors[1]:.3f}<br>"
            f"最大誤差：{validation['maximum_error_in_normalized_court_units']:.3f}（基準 0.030 以下）</div>"
        )

    def page(message: str = "") -> str:
        saved = load_json(output)
        loaded_points = saved.get("preview_points_px", []) if saved else []
        return f"""<!doctype html><html lang='ja'><meta charset='utf-8'><title>固定主カメラ：コート平面較正</title>
<style>
body{{font-family:system-ui,'Yu Gothic',sans-serif;max-width:1500px;margin:auto;padding:18px;background:#f3f4f6;line-height:1.6}}
section{{background:#fff;border-radius:10px;padding:16px;margin-bottom:14px}}canvas{{max-width:100%;height:auto;cursor:crosshair;border:2px solid #1f2937;background:#111}}button{{font-size:16px;padding:7px 13px;margin:4px}}.note{{background:#fff5df;padding:12px;border-radius:6px}}.keys{{background:#e0f2fe;padding:10px;border-radius:6px}}.ok{{background:#dcfce7;padding:12px;border-radius:6px}}.bad{{background:#fee2e2;padding:12px;border-radius:6px}}.waiting{{background:#f3f4f6;padding:12px;border-radius:6px}}ol{{margin:6px 0}}code{{background:#eee;padding:2px 4px}}
</style><section><h1>固定主カメラのコート平面較正</h1><p>原動画 F{args.source_frame}（{raw_width} × {raw_height} px）を使用します。<strong>これは試合ごと・カメラ画角ごとに一度だけ行う手入力</strong>です。</p>{html.escape(message)}</section>
<section><h2>クリック順序</h2><ol>{''.join(f'<li>{html.escape(label)}</li>' for label in POINT_LABELS)}</ol><p class='note'><strong>1〜4は白線同士が交わる点を選んでください。</strong> 5〜6は検証専用で、変換の計算には使いません。奥側は画面上側、手前側は画面下側です。リプレイや別カメラの画面には使えません。</p><p class='keys'><strong>Z</strong>：1点戻す　<strong>R</strong>：全消去　<strong>Enter</strong>：6点入力済みなら保存・検証</p></section>
<section><h2>コートをクリック</h2><canvas id='court' width='{preview_width}' height='{preview_height}' aria-label='コート較正画像'></canvas><p id='progress'></p><button id='undo' type='button'>1点戻す</button><button id='reset' type='button'>全てやり直す</button><form id='save-form' method='post' action='/save'><input id='points' type='hidden' name='points'><button id='save' type='submit'>この6点で保存・検証</button></form></section>
<section><h2>保存済み結果</h2>{result_html(saved)}<p>保存先：<code>{html.escape(str(output))}</code></p></section>
<script>
const labels={json.dumps(POINT_LABELS, ensure_ascii=False)};
const canvas=document.getElementById('court'), ctx=canvas.getContext('2d'), progress=document.getElementById('progress'), form=document.getElementById('save-form'), hidden=document.getElementById('points');
const image=new Image(); let points={json.dumps(loaded_points, ensure_ascii=False)}; image.src='/image';
function draw(){{ctx.clearRect(0,0,canvas.width,canvas.height); if(image.complete)ctx.drawImage(image,0,0,canvas.width,canvas.height); points.forEach((p,i)=>{{ctx.beginPath();ctx.arc(p[0],p[1],10,0,Math.PI*2);ctx.fillStyle='#ef4444';ctx.fill();ctx.lineWidth=3;ctx.strokeStyle='white';ctx.stroke();ctx.fillStyle='white';ctx.font='bold 15px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(i+1),p[0],p[1]);}}); const next=points.length<6?labels[points.length]:'6点入力済み。保存・検証してください。'; progress.textContent=`入力 ${'{'}points.length{'}'}/6：${'{'}next{'}'}`; document.getElementById('save').disabled=points.length!==6;}}
image.onload=draw;
canvas.addEventListener('click',(event)=>{{if(points.length>=6)return;const rect=canvas.getBoundingClientRect();points.push([(event.clientX-rect.left)*canvas.width/rect.width,(event.clientY-rect.top)*canvas.height/rect.height]);draw();}});
document.getElementById('undo').onclick=()=>{{points.pop();draw();}}; document.getElementById('reset').onclick=()=>{{points=[];draw();}};
form.addEventListener('submit',(event)=>{{if(points.length!==6){{event.preventDefault();alert('6点を指定してください。');return;}}hidden.value=JSON.stringify(points);}});
document.addEventListener('keydown',(event)=>{{if(['INPUT','TEXTAREA','SELECT','BUTTON'].includes(document.activeElement.tagName))return;if(event.key.toLowerCase()==='z'){{points.pop();draw();}}if(event.key.toLowerCase()==='r'){{points=[];draw();}}if(event.key==='Enter'&&points.length===6){{event.preventDefault();form.requestSubmit();}}}}); draw();
</script></html>"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                document = page(parse_qs(parsed.query).get("message", [""])[0]).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(document)))
                self.end_headers(); self.wfile.write(document)
                return
            if parsed.path == "/image":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(image)))
                self.end_headers(); self.wfile.write(image)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/save":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                form = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
                points = as_points(form.get("points", [""])[0], preview_width, preview_height)
                payload = calibration_payload(
                    args.video, args.source_frame, raw_width, raw_height, preview_width, preview_height, points
                )
                write_json_atomic(output, payload)
            except (ValueError, UnicodeDecodeError) as error:
                self.send_response(HTTPStatus.SEE_OTHER)
                self.send_header("Location", f"/?message={html.escape(str(error))}")
                self.end_headers()
                return
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", "/?message=保存して検証しました。下の検証結果を確認してください。")
            self.end_headers()

    print(f"Open http://127.0.0.1:{args.port}/")
    print(f"Output: {output}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
