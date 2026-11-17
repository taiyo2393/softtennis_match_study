"""Training-only124case review; four existing human centers reused, no predictions."""
import sys
sys.path.insert(0, 'C:\\Users\\ta239\\Research\\softtennis_match_study\\outputs\\ball_quality_20261103')
import argparse
import json
import os
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from integrate_044_reviewed_tracks import ROOT, read, load, sha
from serve_ball_phase_challenge import validate, atomic_write

OUT = Path(__file__).resolve().parent
LOCK = threading.Lock()


def verify_history(folder):
    state = {r['case_id']: r for r in read(folder / 'initial_labels_snapshot.csv')}
    events = folder / 'annotation_events.jsonl'
    if events.exists():
        for line in events.read_text(encoding='utf-8').splitlines():
            event = json.loads(line)
            assert state[event['case_id']] == event['before']
            state[event['case_id']] = event['after']
    assert [state[r['case_id']] for r in read(folder / 'initial_labels_snapshot.csv')] == read(folder / 'manual_sequence_labels.csv')


def prepare_answer(payload, ids):
    allowed = {'case_id', 'revision', 'truth_visibility', 'phase_status', 'x', 'y', 'notes'}
    if not isinstance(payload, dict) or not set(payload) <= allowed:
        raise ValueError('保存する項目が不正です。')
    answer = validate(payload, ids)
    return {k: answer[k] for k in ['phase_status', 'truth_visibility', 'truth_x_px', 'truth_y_px', 'reviewer_id', 'annotated_at_utc', 'notes']}


def main(port):
    manifest = load(OUT / 'manifest.json')
    assert all(sha(Path(p)) == h for p, h in manifest['protected_input_hashes'].items())
    assert all(sha(OUT / n) == h for n, h in manifest['frozen_artifact_hashes'].items())
    cases = load(OUT / 'annotation_cases.json')
    by_id = {c['case_id']: c for c in cases}
    assert len(by_id) == 124
    for c in cases:
        assert sha(Path(c['source_image'])) == c['source_image_sha256']
        assert sha(Path(c['context_video'])) == c['context_video_sha256']
    verify_history(OUT)
    path = OUT / 'manual_sequence_labels.csv'
    prefix = '/' + secrets.token_urlsafe(18) + '/'

    def public():
        labels = {r['case_id']: r for r in read(path)}
        return dict(cases=[{**{k: c[k] for k in ['case_id', 'window_id', 'candidate_id', 'local_frame', 'source_frame', 'center_reused', 'context_target_time_sec']},
            **{k: labels[c['case_id']][k] for k in ['truth_visibility', 'truth_x_px', 'truth_y_px', 'phase_status', 'review_status', 'notes']}} for c in cases], revision=sha(path))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def blob(self, data, mime, status=200, headers=None):
            self.send_response(status)
            for name, value in {'Content-Type': mime, 'Content-Length': str(len(data)), 'Cache-Control': 'no-store', **(headers or {})}.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def json(self, data, status=200):
            self.blob(json.dumps(data, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8', status)

        def do_GET(self):
            route = urlparse(self.path).path
            if not route.startswith(prefix):
                self.send_error(404)
                return
            route = route[len(prefix):]
            if route == '':
                self.blob((OUT / 'annotation.html').read_bytes(), 'text/html; charset=utf-8')
                return
            if route == 'api/cases':
                with LOCK:
                    self.json(public())
                return
            match = re.fullmatch(r'(image|video)/([ABCD]\d{2})\.(jpg|mp4)', route)
            if not match or match[2] not in by_id:
                self.send_error(404)
                return
            kind, cid, ext = match.groups()
            if (kind, ext) not in [('image', 'jpg'), ('video', 'mp4')]:
                self.send_error(404)
                return
            data = Path(by_id[cid]['source_image' if kind == 'image' else 'context_video']).read_bytes()
            mime = 'image/jpeg' if kind == 'image' else 'video/mp4'
            rh = self.headers.get('Range')
            if rh and kind == 'video':
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', rh)
                if not match or not any(match.groups()):
                    self.send_error(416)
                    return
                start = max(0, len(data) - int(match[2])) if not match[1] else int(match[1])
                end = len(data) - 1 if not match[1] or not match[2] else min(int(match[2]), len(data) - 1)
                if start > end or start >= len(data):
                    self.send_error(416)
                    return
                self.blob(data[start:end + 1], mime, 206, {'Content-Range': f'bytes {start}-{end}/{len(data)}', 'Accept-Ranges': 'bytes'})
            else:
                self.blob(data, mime, headers={'Accept-Ranges': 'bytes'} if kind == 'video' else None)

        def do_POST(self):
            if urlparse(self.path).path != prefix + 'api/save':
                self.send_error(404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16384:
                    raise ValueError('入力サイズが不正です。')
                if self.headers.get('Origin') not in (None, f'http://127.0.0.1:{self.server.server_port}'):
                    raise ValueError('保存元の画面が不正です。')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('入力形式が不正です。')
                with LOCK:
                    if payload.get('revision') != sha(path):
                        self.json(dict(error='別の画面で更新されています。最新状態を読み直してください。'), 409)
                        return
                    case = by_id.get(payload.get('case_id'))
                    if case and case['center_reused']:
                        payload = dict(payload, truth_visibility='visible', x=case['reused_x'], y=case['reused_y'])
                    answer = prepare_answer(payload, set(by_id))
                    rows = read(path)
                    row = next(r for r in rows if r['case_id'] == payload['case_id'])
                    before = dict(row)
                    row.update(answer, review_status='human_phase_only_existing_center' if case['center_reused'] else 'human_frame_review', center_source='existing_human_click' if case['center_reused'] else 'new_human_frame_review')
                    event = dict(case_id=row['case_id'], before=before, after=dict(row), model_predictions_hidden=True)
                    # A durable event is written first; interrupted CSV updates are detected on restart.
                    with (OUT / 'annotation_events.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(event, ensure_ascii=False) + '\n')
                        f.flush()
                        os.fsync(f.fileno())
                    atomic_write(path, rows)
                    self.json(public())
            except (ValueError, KeyError, TypeError, StopIteration) as error:
                self.json(dict(error=str(error)), 400)

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    state = dict(url=f'http://127.0.0.1:{server.server_port}{prefix}', pid=os.getpid(), annotation_cases=124,
        bind='127.0.0.1', server_script_sha256=sha(Path(__file__)))
    (OUT / 'server_state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    print(json.dumps(state), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=0)
    main(parser.parse_args().port)
