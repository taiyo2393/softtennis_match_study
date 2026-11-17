"""Reconcile model fit, exact cached tile components, output provenance and scores."""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from integrate_044_reviewed_tracks import ROOT, read, load, sha
from compare_grouped_tile_gate import F
from compare_ball_candidate_decoders import components
from calibrate_point_supervised_detector import expand

BASE = ROOT.parent / 'ball_quality_20261109'
OUT = BASE / 'frozen_tile_gate_confirmation'
CHECK = ROOT.parent / 'ball_quality_20261104'


def independently_extract(c, native, degraded, tiles):
    x, y = c['x_px'], c['y_px']
    dist = lambda q: math.hypot(q['x_px'] - x, q['y_px'] - y)
    covering = [i for i, (a, b, w, h) in enumerate(tiles) if a <= x < a + w and b <= y < b + h]
    support = {q['tile_index'] for q in native if q['tile_index'] in covering and dist(q) <= 16}
    close = [q['component_peak'] for q in degraded if dist(q) <= 16]
    return np.array([c['component_peak'], c['bbox_mean'], math.log1p(c['component_pixels']),
        c['component_pixels'] / (c['w'] * c['h']), math.log(c['w'] / c['h']), len(support) / len(covering),
        max(close, default=0.), min(min((dist(q) for q in degraded), default=64.) / 64, 1.),
        min(c['x'], c['y'], 512 - c['x'] - c['w'], 288 - c['y'] - c['h']) / 144,
        c['component_peak'] - max((q['component_peak'] for q in native if dist(q) > 32), default=0.)])


def main():
    protocol = load(OUT / 'protocol.json')
    manifest = load(OUT / 'inference_manifest.json')
    report = load(OUT / 'evaluation_report.json')
    assert all(sha(Path(p)) == h for p, h in protocol['protected_input_hashes'].items())
    assert all(sha(Path(p)) == h for p, h in protocol['helper_hashes'].items())
    assert sha(OUT / 'executed_source.py') == protocol['script_sha256']
    assert all(sha(OUT / n) == h for n, h in manifest['artifact_hashes'].items())
    assert sha(OUT / 'protocol.json') == report['protocol_sha256']
    assert sha(OUT / 'inference_manifest.json') == report['inference_manifest_sha256']
    model = dict(np.load(OUT / 'frozen_training_head.npz'))
    examples = read(BASE / 'training_tile_gate_grouped/candidate_features_and_training_labels.csv')
    es = [e for e in examples if e['fit_label'] != '']
    x = np.array([[float(e[f]) for f in F] for e in es], dtype=np.float64)
    y = np.array([float(e['fit_label']) for e in es])
    assert len(es) == 81 and sum(y) == 76
    assert np.array_equal(x.mean(0), model['mean'])
    scale = x.std(0)
    scale[scale < 1e-8] = 1.
    assert np.array_equal(scale, model['scale'])
    normalized = (x - model['mean']) / model['scale']
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    probability = sigmoid(normalized @ model['weights'] + model['bias'])
    balance = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (1 - y).sum()))
    residual = (probability - y) * balance
    gradient = np.r_[normalized.T @ residual / len(y) + model['weights'], residual.mean()]
    assert np.max(np.abs(gradient)) < 1e-6
    all_x = np.array([[float(e[f]) for f in F] for e in examples])
    scores = sigmoid(((all_x - model['mean']) / model['scale']) @ model['weights'] + model['bias'])
    saved_scores = {r['sample_id']: r for r in read(OUT / 'training_candidate_scores.csv')}
    assert all(abs(s - float(saved_scores[e['sample_id']]['proposal_gate_score'])) < 1e-12 for e, s in zip(examples, scores, strict=True))
    fit_manifest = load(OUT / 'fit_manifest.json')
    assert fit_manifest['fit_sample_ids'] == [e['sample_id'] for e in es]
    queries = read(CHECK / 'query_model_inputs.csv')
    assert not set(fit_manifest['fit_sample_ids']) & {q['sample_id'] for q in queries}
    input_hashes = load(OUT / 'input_image_hashes.json')
    assert all(sha(Path(p)) == h for p, h in input_hashes.items())
    old_context = load(CHECK / 'frozen_component_cache.json')
    replay = read(OUT / 'global_replay.csv')
    assert len(replay) == 84 and all(r['exact_component_match'] == 'True' for r in replay)
    tiles = load(BASE / 'training_tile_detail_comparison/protocol.json')['tiles_xywh']
    pools = load(OUT / 'candidate_pools.json')
    rebuilt = {arm: {q['sample_id']: [] for q in queries} for arm in pools}
    maps = read(OUT / 'map_records.csv')
    packed = np.load(OUT / 'component_rectangles.npz')
    assert len(maps) == 936 and len(packed['offsets']) == 937
    map_keys = set()
    for i, r in enumerate(maps):
        lo, hi = packed['offsets'][i:i + 2]
        hm = expand(packed['indices'][lo:hi], packed['values'][lo:hi])
        ti = int(r['tile_index'])
        key = r['sample_id'], r['arm'], ti
        assert key not in map_keys
        map_keys.add(key)
        ox, oy, width, height = tiles[ti]
        for c in components(hm):
            c.update(tile_index=ti, tile_x=ox, tile_y=oy,
                x_px=int((c['x'] + c['w'] / 2) * width / 512 + ox),
                y_px=int((c['y'] + c['h'] / 2) * height / 288 + oy))
            rebuilt[r['arm']][r['sample_id']].append(c)
    assert rebuilt == pools
    features = {r['sample_id']: r for r in read(OUT / 'confirmation_candidate_features.csv')}
    candidate = {r['sample_id']: r for r in read(OUT / 'frozen_confirmation_predictions.csv')}
    baseline = {r['sample_id']: r for r in read(CHECK / 'frozen_predictions.csv') if r['method'] == 'displacement_0.1'}
    verified = 0
    for q in queries:
        sid = q['sample_id']
        native, degraded = pools['native'][sid], pools['degraded'][sid]
        selected = max(native, key=lambda c: (c['component_peak'], c['bbox_area'], -c['tile_index'], -c['component_index'])) if native else None
        if selected:
            raw = independently_extract(selected, native, degraded, tiles)
            assert np.array_equal(raw, np.array([float(features[sid][f]) for f in F]))
            score = float(sigmoid(((raw - model['mean']) / model['scale']) @ model['weights'] + model['bias']))
            assert abs(score - float(features[sid]['proposal_gate_score'])) < 1e-12
            assert abs(score - float(candidate[sid]['proposal_gate_score'])) < 1e-12
            verified += 1
        else:
            score = 0.
            assert sid not in features and float(candidate[sid]['proposal_gate_score']) == 0
        a, b = baseline[sid], candidate[sid]
        if int(a['visible']):
            assert all(a[k] == b[k] for k in ['visible', 'x_px', 'y_px', 'confidence'])
            assert b['output_source'] == 'unchanged_original_temporal'
        elif selected and score >= .5:
            assert int(b['visible']) == 1 and b['output_source'] == 'accepted_actual_native_tile'
            assert (float(b['x_px']), float(b['y_px'])) == (selected['x_px'], selected['y_px'])
            assert float(b['confidence']) == selected['bbox_mean']
        else:
            assert int(b['visible']) == 0 and b['output_source'] == 'missing'
    labels = read(CHECK / 'completed_review/manual_sequence_labels.csv')
    point_report = {(r['method'], r['sample_id']): r for r in read(OUT / 'confirmation_per_frame.csv')}
    pair_changes = {r['sample_id']: r for r in read(OUT / 'confirmation_paired_changes.csv')}
    summary = {(r['method'], r['window_id']): r for r in read(OUT / 'confirmation_per_window.csv')}
    totals, outcomes, unresolved = {}, {}, []
    for method, predictions in [('current_temporal', baseline), ('logistic10_frozen', candidate)]:
        counts = Counter()
        per_window = {w: Counter() for w in ['A', 'B', 'C', 'D']}
        outcomes[method] = {}
        for g in labels:
            sid = g['sample_id']
            p = predictions[sid]
            assert g['phase_status'] == 'play'
            if g['truth_visibility'] == 'uncertain':
                outcome = 'excluded_ball_uncertain'
            elif g['truth_visibility'] == 'not_visible':
                outcome = 'output_when_not_observable' if int(p['visible']) else 'no_output_when_not_observable'
            elif not int(p['visible']):
                outcome = 'missing_visible_ball'
            else:
                error = math.dist((float(g['truth_x_px']), float(g['truth_y_px'])), (float(p['x_px']), float(p['y_px'])))
                outcome = 'correct16px' if error <= 16 else 'wrong_position'
                assert abs(float(point_report[method, sid]['error_px']) - error) < 1e-10
            counts[outcome] += 1
            per_window[g['window_id']][outcome] += 1
            outcomes[method][sid] = outcome
            assert point_report[method, sid]['outcome'] == outcome
            if method == 'logistic10_frozen' and outcome == 'missing_visible_ball':
                unresolved.append(dict(case_id=g['case_id'], sample_id=sid, window_id=g['window_id'], local_frame=g['local_frame'], source_frame=g['source_frame']))
        for key, value in counts.items():
            assert report['totals'][method].get(key, 0) == value
        for window, wc in per_window.items():
            assert all(int(summary[method, window][k]) == n for k, n in wc.items())
        totals[method] = dict(counts)
    gains = losses = 0
    for g in labels:
        sid = g['sample_id']
        a, b = outcomes['current_temporal'][sid], outcomes['logistic10_frozen'][sid]
        gain, loss = int(b == 'correct16px' and a != 'correct16px'), int(a == 'correct16px' and b != 'correct16px')
        assert int(pair_changes[sid]['gained_correct']) == gain and int(pair_changes[sid]['lost_correct']) == loss
        gains += gain
        losses += loss
    assert (gains, losses) == (report['gained_correct'], report['lost_correct'])
    for r in read(OUT / 'confirmation_adjacent_pairs.csv'):
        by_frame = {int(g['local_frame']): g for g in labels if g['window_id'] == r['window_id']}
        a, b = by_frame[int(r['first_local_frame'])], by_frame[int(r['last_local_frame'])]
        predictions = baseline if r['method'] == 'current_temporal' else candidate
        pa, pb = predictions[a['sample_id']], predictions[b['sample_id']]
        if r['displacement_error_px'] != '':
            actual = [float(pb[k]) - float(pa[k]) for k in ['x_px', 'y_px']]
            truth = [float(b[k]) - float(a[k]) for k in ['truth_x_px', 'truth_y_px']]
            assert abs(float(r['displacement_error_px']) - math.dist(actual, truth)) < 1e-10
    assert all(sha(Path(p)) == h for p, h in protocol['protected_input_hashes'].items())
    assert all(sha(OUT / n) == h for n, h in manifest['artifact_hashes'].items())
    verification = dict(status='passed', training_fit_examples=81, training_standardization_exact=True,
        objective_stationarity_max_abs_gradient=float(np.max(np.abs(gradient))),
        training_scores_verified=len(examples), exact_tile_maps_reconstructed=936,
        candidate_pools_exact=True, independently_recomputed_candidate_features_and_scores=verified,
        final_predictions_provenance_verified=52, per_method_evaluation_rows_verified=104,
        observed_gains=gains, observed_losses=losses, remaining_missing_visible=unresolved,
        labels_model_and_operational_references_unchanged=True,
        automatic_adoption=False, independent_external_validation=False,
        script_sha256=sha(Path(__file__)), artifacts={p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'verification.json'})
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in verification.items() if k != 'artifacts'}, indent=2))


if __name__ == '__main__':
    main()
