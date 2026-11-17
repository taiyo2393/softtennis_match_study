"""Read-only packet verification and synthetic save checks; never labels real frames."""
import argparse
import importlib.util
import json
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from integrate_044_reviewed_tracks import ROOT, SOURCE, read, load, sha, cv2, np
from serve_ball_phase_challenge import atomic_write, validate

OUT = ROOT.parent / 'ball_quality_20261109/dense_training_sequences'


def main(http=False):
    manifest = load(OUT / 'manifest.json')
    assert all(sha(Path(p)) == h for p, h in manifest['protected_input_hashes'].items())
    assert all(sha(OUT / p) == h for p, h in manifest['frozen_artifact_hashes'].items())
    images = load(OUT / 'source_image_hashes.json')
    assert all(sha(Path(p)) == h for p, h in images.items())
    labels = read(OUT / 'manual_sequence_labels.csv')
    before = sha(OUT / 'manual_sequence_labels.csv')
    assert labels == read(OUT / 'initial_labels_snapshot.csv')
    cases = load(OUT / 'annotation_cases.json')
    queries = {r['sample_id']: r for r in read(OUT / 'query_model_inputs.csv')}
    old = read(ROOT / 'balltrack_expanded_centers_pilot/expanded_train_labels.csv')
    phases = {r['sample_id']: r['phase_status'] for r in read(OUT.parent / 'training_phase_scope_review/phase_labels.csv')}
    source_labels = {r['sample_id']: r for r in old}
    inventory = {r['ball_pilot_id']: r for r in read(ROOT / 'play_start_audit/ball_pilot_start_inventory.csv')}
    clips = sorted({r['candidate_id'] for r in old}, key=lambda c: (int(inventory[c]['clip_first_source_frame']), c))
    selected = [clips[int((j + .5) * len(clips) / 4)] for j in range(4)]
    windows = load(OUT / 'windows.json')
    assert [w['candidate_id'] for w in windows] == selected
    assert len(cases) == len(labels) == len(queries) == 124
    assert len({r['source_frame'] for r in labels}) == 124
    assert sum(r['review_status'] == 'unreviewed' for r in labels) == 120
    target_jpegs_from_source = 0
    decoded_context = 0
    protected_clips = {}
    for w in windows:
        cid = w['candidate_id']
        anchor_rows = sorted([r for r in old if r['candidate_id'] == cid and phases[r['sample_id']] == 'play' and r['truth_visibility'] == 'visible'], key=lambda r: int(r['local_frame']))
        assert w['anchor_local_frame'] == int(anchor_rows[len(anchor_rows)//2]['local_frame'])
        wc = [c for c in cases if c['window_id'] == w['window_id']]
        assert [c['local_frame'] for c in wc] == list(range(w['anchor_local_frame']-30, w['anchor_local_frame']+31, 2))
        video = cv2.VideoCapture(w['context_video'])
        assert abs(video.get(cv2.CAP_PROP_FPS) - w['fps']) < .001
        frames = 0
        while True:
            ok, im = video.read()
            if not ok: break
            assert im.shape == (800, 1280, 3)
            frames += 1
        video.release()
        assert frames == w['context_frames'] == 181
        decoded_context += frames
        # Recreate source JPG using the extraction encoder, rather than trusting an index alone.
        meta = load(Path(inventory[cid]['run_metadata_path']) if 'run_metadata_path' in inventory[cid] else SOURCE / 'data/balltrack/racketvision/match1' / cid / 'runs/threshold030_median120/run_metadata.json')
        clip = Path(meta['clip'])
        protected_clips[str(clip)] = sha(clip)
        cap = cv2.VideoCapture(str(clip))
        targets = {c['local_frame']: c for c in wc}
        for t in range(max(targets) + 1):
            ok, frame = cap.read(); assert ok
            if t in targets:
                encoded, jpg = cv2.imencode('.jpg', frame)
                assert encoded and jpg.tobytes() == Path(targets[t]['source_image']).read_bytes(), (cid, t)
                target_jpegs_from_source += 1
        cap.release()
    for c, row in zip(cases, labels, strict=True):
        assert c['sample_id'] == row['sample_id'] and int(row['source_frame']) == int(inventory[c['candidate_id']]['clip_first_source_frame']) + c['local_frame']
        assert c['source_image_sha256'] == sha(Path(c['source_image']))
        assert c['context_video_sha256'] == sha(Path(c['context_video']))
        assert int(c['context_target_time_sec'] * c['context_fps']) == c['local_frame'] - c['context_first_local_frame']
        q = queries[c['sample_id']]
        paths, hashes = json.loads(q['input_frames_json']), json.loads(q['input_frame_sha256_json'])
        assert [int(Path(p).stem) for p in paths] == list(range(c['local_frame']-3, c['local_frame']+1))
        assert all(sha(Path(p)) == h for p, h in zip(paths, hashes, strict=True))
        assert paths[-1] == c['source_image'] and q['finetune_role'] == row['finetune_role'] == 'train'
        if c['center_reused']:
            src = source_labels[c['sample_id']]
            assert row['truth_x_px'] == src['truth_x_px'] and row['truth_y_px'] == src['truth_y_px'] and phases[c['sample_id']] == 'play'
        else:
            assert row['truth_visibility'] == 'unreviewed' and row['phase_status'] == 'unknown'
            assert row['truth_x_px'] == row['truth_y_px'] == ''
    spec = importlib.util.spec_from_file_location('dense_review', OUT / 'serve_review.py')
    server = importlib.util.module_from_spec(spec); spec.loader.exec_module(server)
    server.verify_history(OUT)
    with tempfile.TemporaryDirectory(prefix='dense_synthetic_') as name:
        folder = Path(name)
        initial = dict(case_id='synthetic', phase_status='unknown', truth_visibility='unreviewed', truth_x_px='', truth_y_px='', reviewer_id='', annotated_at_utc='', notes='')
        atomic_write(folder / 'initial_labels_snapshot.csv', [initial])
        answer = server.prepare_answer(dict(case_id='synthetic', phase_status='play', truth_visibility='visible', x=960, y=540), {'synthetic'})
        updated = dict(initial, **answer)
        (folder / 'annotation_events.jsonl').write_text(json.dumps(dict(case_id='synthetic', before=initial, after=updated)) + '\n', encoding='utf-8')
        atomic_write(folder / 'manual_sequence_labels.csv', [updated]); server.verify_history(folder)
        for patch in [dict(x=float('nan')), dict(x=1920), dict(y=-1), dict(phase_status='unknown'), dict(prediction_x_px=123)]:
            payload = dict(case_id='synthetic', phase_status='play', truth_visibility='visible', x=960, y=540); payload.update(patch)
            try: server.prepare_answer(payload, {'synthetic'})
            except ValueError: pass
            else: raise AssertionError('Invalid synthetic label accepted')
    result = dict(status='passed', metadata_selection_recomputed=True, original_clip_jpegs_exact=target_jpegs_from_source,
        source_images_hashed=len(images), target_times_verified=124, reused_centers_verified=4, pending_new_human_targets=120,
        decoded_video_frames=decoded_context, context_videos=4, input_sequences_verified=124,
        target_seek_times_verified=124, synthetic_save_history_passed=True, labels_unchanged=True,
        phase_not_propagated=True, no_inference_or_training=True, source_clip_hashes=protected_clips, script_sha256=sha(Path(__file__)))
    if http:
        url = load(OUT / 'server_state.json')['url']
        def get(route, headers=None):
            with urllib.request.urlopen(urllib.request.Request(url+route, headers=headers or {}), timeout=15) as response:
                return response.status, response.read()
        assert get('')[1] == (OUT / 'annotation.html').read_bytes()
        public = json.loads(get('api/cases')[1])
        assert len(public['cases']) == 124 and sum(c['review_status'] == 'unreviewed' for c in public['cases']) == 120
        assert not any(any('prediction' in k for k in c) for c in public['cases'])
        for c in cases:
            assert get('image/' + c['case_id'] + '.jpg')[1] == Path(c['source_image']).read_bytes()
        for wid in 'ABCD':
            c = next(c for c in cases if c['window_id'] == wid)
            data = Path(c['context_video']).read_bytes()
            for header, expected in [('bytes=0-99', data[:100]), ('bytes=-100', data[-100:])]:
                status, content = get('video/' + c['case_id'] + '.mp4', {'Range':header})
                assert status == 206 and content == expected
        for payload, status in [(dict(case_id='A01', revision='stale'), 409),
                (dict(case_id='A01', revision=public['revision'], prediction_x_px=2), 400),
                (dict(case_id='A01', revision=public['revision'], phase_status='play', truth_visibility='visible'), 400),
                (dict(case_id='other', revision=public['revision'], phase_status='play', truth_visibility='visible', x=0, y=0), 400)]:
            req = urllib.request.Request(url+'api/save', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}, method='POST')
            try: urllib.request.urlopen(req, timeout=15)
            except urllib.error.HTTPError as e: assert e.code == status
            else: raise AssertionError('Invalid HTTP write accepted')
        result.update(http_images_exact=124, http_video_ranges=8, invalid_posts_rejected=4, public_predictions_hidden=True)
    assert sha(OUT / 'manual_sequence_labels.csv') == before and not (OUT / 'annotation_events.jsonl').exists()
    assert all(sha(Path(p)) == h for p, h in manifest['protected_input_hashes'].items())
    (OUT / ('http_verification.json' if http else 'verification.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--http', action='store_true')
    main(parser.parse_args().http)
