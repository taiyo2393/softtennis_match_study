"""Local review of the 12 preselected raw frames from match2 point one."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import sys
from threading import Lock
from urllib.parse import urlsplit

import build_044_spatiotemporal_pilot as common

sys.path.insert(0, str(common.OLD))
from serve_ball_phase_challenge import atomic_write, validate_center

SOURCE = common.HERE / 'match2_ball_first_point'
OUT = common.HERE / 'match2_first_point_ball_review'
PAGE = common.HERE / 'match2_first_point_reference.html'


def prepare(source, out):
    out.mkdir(exist_ok=True)
    labels = out / 'manual_ball_references.csv'
    frames = common.load(source / 'protocol.json')['reference_frames_selected_before_inference']
    images = {str(frame): source / 'raw_reference_frames' / f'F{frame}.png' for frame in frames}
    assert len(images) == 12 and all(path.is_file() for path in images.values())
    for name, digest in common.load(source / 'publication_hashes.json').items():
        assert common.sha(source / name) == digest, name
    reference_hashes = common.load(source / 'verification.json').get('reference_images', {})
    if reference_hashes:
        assert all(common.sha(path) == reference_hashes[path.name] for path in images.values())
    if not labels.exists():
        atomic_write(labels, [dict(
            source_frame=frame, truth_visibility='', truth_x_px='', truth_y_px='',
            reviewer_id='', annotated_at_utc='') for frame in images])
    assert [row['source_frame'] for row in common.read(labels)] == list(images)
    receipt = out / 'rally_boundary_review.json'
    if source == SOURCE and not receipt.exists():
        common.dump(receipt, dict(
            match='match2', point=1,
            question_item_id='["request_user_input_async","call_1aa9f0cf55b94315ad6e2377a23d0c16",0]',
            question='match2第1点の動画で、手前ベースライン付近に球が落ちるF12574前後がラリー終端で合っていますか？ 終わり方が分かれば、アウトなど一言添えてください。',
            answer='アウト', ending_type='out',
            event_scope='User classified the indicated near-baseline ending as out.',
            AI_estimated_end_frame_range=[12570, 12578], exact_contact_frame_human_validated=False,
            video_path=str(SOURCE / 'match2_first_point_raw_review.mp4'),
            video_sha256=common.sha(SOURCE / 'match2_first_point_raw_review.mp4'),
            score_winner_human_validated=False))
    return images


def serve(port, source=SOURCE, out=OUT, title='match2 第1点の球位置'):
    images = prepare(source, out)
    labels = out / 'manual_ball_references.csv'
    markup = PAGE.read_text(encoding='utf-8').replace('match2 第1点の球位置', title).encode('utf-8')
    prefix = '/' + secrets.token_urlsafe(18) + '/'
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_data(self, data, kind, status=200):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, data, status=200):
            self.send_data(json.dumps(data, ensure_ascii=False).encode('utf-8'),
                           'application/json; charset=utf-8', status)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == prefix:
                self.send_data(markup, 'text/html; charset=utf-8')
            elif path == prefix + 'api/cases':
                with lock:
                    self.send_json({'cases': common.read(labels)})
            elif path.startswith(prefix):
                match = re.fullmatch(r'image/(\d+)\.png', path[len(prefix):])
                if match and match[1] in images:
                    self.send_data(images[match[1]].read_bytes(), 'image/png')
                else:
                    self.send_error(404)
            else:
                self.send_error(404)

        def do_POST(self):
            if urlsplit(self.path).path != prefix + 'api/save':
                self.send_error(404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('入力サイズが不正です。')
                if self.headers.get('Origin') not in (None, f'http://127.0.0.1:{self.server.server_port}'):
                    raise ValueError('この確認画面から保存してください。')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('入力形式が不正です。')
                answer = validate_center(payload, set(images))
                with lock:
                    rows = common.read(labels)
                    row = next(r for r in rows if r['source_frame'] == payload['case_id'])
                    row.update({key: answer[key] for key in row if key != 'source_frame'})
                    atomic_write(labels, rows)
                self.send_json({'cases': rows})
            except (ValueError, TypeError, KeyError) as exc:
                self.send_json({'error': str(exc)}, 400)

    with ThreadingHTTPServer(('127.0.0.1', port), Handler) as server:
        state = dict(url=f'http://127.0.0.1:{server.server_port}{prefix}', cases=len(images))
        common.dump(out / 'server_state.json', state)
        print(state['url'], flush=True)
        server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--points-2-3', action='store_true')
    args = parser.parse_args()
    if args.points_2_3:
        serve(args.port, common.HERE / 'match2_points_2_3',
              common.HERE / 'match2_points_2_3_ball_review', 'match2 第2点・第3点の球位置')
    else:
        serve(args.port)
