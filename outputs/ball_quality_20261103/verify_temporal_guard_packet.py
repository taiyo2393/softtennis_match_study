"""Verify frozen26case geometry/provenance and the blind local interface."""
import argparse
import json
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from integrate_044_reviewed_tracks import ROOT, read, load, sha, cv2
from compare_ball_candidate_decoders import components
from compare_training_tile_detail import TILES
from compare_temporal_candidate_paths import select
from compare_grouped_tile_gate import F
from ball_temporal_consistency_rule import decide
from calibrate_point_supervised_detector import expand
from verify_frozen_tile_gate_confirmation import independently_extract
from serve_temporal_guard_packet import prepare_answer, verify_history
from serve_ball_phase_challenge import atomic_write

BASE = ROOT.parent / 'ball_quality_20261109'
OUT = BASE / 'temporal_guard_next_sequences'


def main(http):
    manifest = load(OUT / 'manifest.json')
    assert all(sha(Path(p)) == h for p, h in manifest['protected_input_hashes'].items())
    assert all(sha(OUT / n) == h for n, h in manifest['frozen_artifact_hashes'].items())
    assert all(sha(Path(p)) == h for p, h in manifest['helper_hashes'].items())
    before = sha(OUT / 'manual_sequence_labels.csv')
    assert before == manifest['initial_labels_sha256']
    assert read(OUT / 'manual_sequence_labels.csv') == read(OUT / 'initial_labels_snapshot.csv')
    assert all(r['review_status'] == 'unreviewed' for r in read(OUT / 'manual_sequence_labels.csv'))
    verify_history(OUT)
    assert not (OUT / 'annotation_events.jsonl').exists()
    source_hashes = load(OUT / 'source_image_hashes.json')
    assert all(sha(Path(p)) == h for p, h in source_hashes.items())
    contexts = load(OUT / 'global_contexts.json')
    pools = load(OUT / 'candidate_pools.json')
    maps = read(OUT / 'map_records.csv')
    packed = np.load(OUT / 'component_rectangles.npz')
    rebuilt = {arm: {sid: [] for sid in pools[arm]} for arm in pools}
    assert len(maps) == 582 and len(packed['offsets']) == 583
    for i, r in enumerate(maps):
        lo, hi = packed['offsets'][i:i + 2]
        cs = components(expand(packed['indices'][lo:hi], packed['values'][lo:hi]))
        if r['arm'] == 'global':
            assert cs == contexts[r['candidate_id']][r['local_frame']]
            continue
        ti = int(r['tile_index'])
        ox, oy, w, h = TILES[ti]
        for c in cs:
            c.update(tile_index=ti, tile_x=ox, tile_y=oy, x_px=int((c['x'] + c['w'] / 2) * w / 512 + ox), y_px=int((c['y'] + c['h'] / 2) * h / 288 + oy))
            rebuilt[r['arm']][r['sample_id']].append(c)
    assert rebuilt == pools
    head = dict(np.load(BASE / 'frozen_tile_gate_confirmation/frozen_training_head.npz'))
    feature_rows = {r['sample_id']: r for r in read(OUT / 'candidate_features.csv')}
    preds = {(r['method'], r['sample_id']): r for r in read(OUT / 'frozen_predictions.csv')}
    assert len(preds) == 78
    camera = load(OUT / 'camera_features.json')
    cases = load(OUT / 'annotation_cases.json')
    windows = load(OUT / 'reserved_windows.json')
    features_verified = 0
    for w in windows:
        cid = w['candidate_id']
        context = {int(t): cs for t, cs in contexts[cid].items()}
        mask = np.zeros(int(w['frame_count']), dtype=bool)
        for t, v in camera[cid]['mask_pairs']:
            mask[t] = v
        anchors = {}
        for t in range(w['center_local_frame'] - 20, w['center_local_frame'] + 21):
            c, _ = select(context, t, mask, dict(order='displacement', strength=.1))
            if c:
                anchors[t] = c
        for c in [c for c in cases if c['candidate_id'] == cid]:
            sid, t = c['sample_id'], int(c['local_frame'])
            original, _ = select(context, t, mask, dict(order='displacement', strength=.1))
            native, degraded = pools['native'][sid], pools['degraded'][sid]
            proposal = max(native, key=lambda p: (p['component_peak'], p['bbox_area'], -p['tile_index'], -p['component_index'])) if native else None
            score = 0.
            if proposal:
                x = independently_extract(proposal, native, degraded, TILES)
                assert np.array_equal(x, np.array([float(feature_rows[sid][f]) for f in F]))
                score = float(1 / (1 + np.exp(-(((x - head['mean']) / head['scale']) @ head['weights'] + head['bias']))))
                assert abs(score - float(feature_rows[sid]['proposal_gate_score'])) < 1e-12
                features_verified += 1
            supplement = original if original else proposal if proposal and score >= .5 else None
            d = decide(t, supplement, original is not None, anchors, mask)
            for method, point in [('original_temporal', original), ('static_supplement', supplement), ('two_anchor_guard_all_outputs', supplement if d['keep'] else None)]:
                p = preds[method, sid]
                assert int(p['visible']) == (point is not None)
                if point:
                    assert (float(p['x_px']), float(p['y_px']), float(p['confidence'])) == (point['x_px'], point['y_px'], point['bbox_mean'])
    for c in cases:
        image = cv2.imread(c['source_image'])
        assert image.shape == (1080, 1920, 3)
        cap = cv2.VideoCapture(c['context_video'])
        frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert frame.shape == (800, 1280, 3)
            frames += 1
        cap.release()
        assert frames == c['decoded_video_frames'] == 128
    # Synthetic validation/save/history only. No invented answer to a real sample.
    payload = dict(case_id='synthetic', phase_status='play', truth_visibility='visible', x=960, y=540, revision='fixture')
    answer = prepare_answer(payload, {'synthetic'})
    assert answer['truth_x_px'] == '960'
    with tempfile.TemporaryDirectory(prefix='guard_packet_synthetic_') as name:
        folder = Path(name)
        initial = dict(case_id='synthetic', phase_status='unknown', truth_visibility='unreviewed', truth_x_px='', truth_y_px='', reviewer_id='', annotated_at_utc='', notes='')
        atomic_write(folder / 'initial_labels_snapshot.csv', [initial])
        updated = dict(initial, **answer)
        (folder / 'annotation_events.jsonl').write_text(json.dumps(dict(case_id='synthetic', before=initial, after=updated)) + '\n', encoding='utf-8')
        atomic_write(folder / 'manual_sequence_labels.csv', [updated])
        verify_history(folder)
    for patch in [dict(x=1920), dict(x=float('nan')), dict(phase_status='unknown'), dict(prediction_x_px=123)]:
        try:
            prepare_answer(dict(payload, **patch), {'synthetic'})
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid synthetic input accepted')
    result = dict(status='passed', pending_human_targets=26, frozen_predictions_verified=78, component_maps_reconstructed=582,
        independently_verified_feature_rows=features_verified, decoded_annotation_videos=26, display_frames=26 * 128,
        original_images_1920x1080=True, context_source_aspect_16by9=True, synthetic_save_and_history_passed=True,
        labels_unchanged=True, accuracy_computed=False, script_sha256=sha(Path(__file__)))
    if http:
        state = load(OUT / 'server_state.json')
        def get(route, headers=None):
            with urllib.request.urlopen(urllib.request.Request(state['url'] + route, headers=headers or {}), timeout=15) as response:
                return response.status, response.read()
        status, html = get('')
        assert status == 200 and html == (OUT / 'annotation.html').read_bytes()
        _, data = get('api/cases')
        public = json.loads(data)
        allowed = {'case_id', 'window_id', 'candidate_id', 'local_frame', 'source_frame', 'center_reused', 'truth_visibility', 'truth_x_px', 'truth_y_px', 'phase_status', 'review_status', 'notes'}
        assert len(public['cases']) == 26 and all(set(c) == allowed for c in public['cases'])
        for c in cases:
            _, image = get('image/' + c['case_id'] + '.jpg')
            assert image == Path(c['source_image']).read_bytes()
            for header, expected in [('bytes=0-99', Path(c['context_video']).read_bytes()[:100]), ('bytes=-100', Path(c['context_video']).read_bytes()[-100:])]:
                status, data = get('video/' + c['case_id'] + '.mp4', {'Range': header})
                assert status == 206 and data == expected
        invalid = [dict(case_id='E01', revision='stale'), dict(case_id='E01', revision=public['revision'], prediction_x_px=123),
            dict(case_id='E01', revision=public['revision'], phase_status='play', truth_visibility='visible')]
        for payload in invalid:
            request = urllib.request.Request(state['url'] + 'api/save', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}, method='POST')
            try:
                urllib.request.urlopen(request, timeout=15)
            except urllib.error.HTTPError as e:
                assert e.code in [400, 409]
            else:
                raise AssertionError('Invalid research request accepted')
        result.update(http_exact_images=26, http_video_range_checks=52, invalid_requests_rejected=3, predictions_hidden_in_public_api=True)
    assert sha(OUT / 'manual_sequence_labels.csv') == before
    assert not (OUT / 'annotation_events.jsonl').exists()
    assert all(sha(Path(p)) == h for p, h in manifest['protected_input_hashes'].items())
    (OUT / ('http_verification.json' if http else 'verification.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--http', action='store_true')
    main(parser.parse_args().http)
