"""Verify both fit-stage exclusions, fixed weights, head scores and OOF metrics."""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT, read, load, sha
from compare_grouped_tile_gate import F
from compare_training_tile_detail import TILES
from verify_frozen_tile_gate_confirmation import independently_extract

BASE = ROOT.parent / 'ball_quality_20261109'
OUT = BASE / 'crossfit_native_candidate_ranker'


def main():
    protocol = load(OUT / 'protocol.json')
    result = load(OUT / 'result.json')
    assert all(sha(Path(p)) == h for p, h in protocol['protected_input_hashes'].items())
    assert all(sha(Path(p)) == h for p, h in protocol['helper_hashes'].items())
    assert sha(OUT / 'executed_source.py') == protocol['script_sha256']
    labels = {g['sample_id']: g for g in read(ROOT / 'balltrack_expanded_centers_pilot/expanded_train_labels.csv')}
    phases = {g['sample_id']: g['phase_status'] for g in read(BASE / 'training_phase_scope_review/phase_labels.csv')}
    fit_set = {sid for sid, g in labels.items() if phases[sid] == 'play' and g['truth_visibility'] == 'visible'}
    assert len(fit_set) == 68
    original_file = next(Path(p) for p in protocol['protected_input_hashes'] if p.endswith('checkpoints\\balltrack_best.pth'))
    original = torch.load(original_file, map_location='cpu', weights_only=False)
    original = original.get('state_dict', original)
    original = {(n[7:] if n.startswith('module.') else n): p for n, p in original.items()}
    trainable = {n for n in original if n.endswith(('weight', 'bias')) and (n.startswith('up_block_3.') or n.startswith('predictor.'))}
    folds = load(OUT / 'folds.json')
    assert len(folds) == 8
    tested = set()
    model_checks, score_count, example_count, changed_tensors = [], 0, 0, []
    oof = {(r['held_clip'], r['method'], r['sample_id']): r for r in read(OUT / 'oof_predictions.csv')}
    assert len(oof) == 246
    baseline = {r['sample_id']: r for r in read(ROOT / 'ball_temporal_path_comparison/train_displacement_0.1.csv')}
    degraded_reference = load(BASE / 'training_tile_detail_comparison/candidate_pools.json')['degraded']
    for fold in folds:
        held = fold['held_clip']
        folder = OUT / ('fold_' + held[-3:])
        expected_fit = {sid for sid in fit_set if labels[sid]['candidate_id'] != held}
        expected_held = {sid for sid, g in labels.items() if g['candidate_id'] == held}
        assert expected_fit == set(fold['detector_fit_sample_ids'])
        assert expected_held == set(fold['held_sample_ids']) and not tested & expected_held
        tested.update(expected_held)
        assert set(fold['classifier_fit_frames']) <= expected_fit
        tr_hashes = {h for sid in expected_fit for h in json.loads(labels[sid]['input_frame_sha256_json'])}
        te_hashes = {h for sid in expected_held for h in json.loads(labels[sid]['input_frame_sha256_json'])}
        assert not tr_hashes & te_hashes
        checkpoint = folder / 'detector_final03.pth'
        assert sha(checkpoint) == fold['detector_sha256']
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert set(loaded['meta']['detector_fit_sample_ids']) == expected_fit and loaded['meta']['held_clip'] == held
        state = loaded['state_dict']
        assert set(state) == set(original)
        assert all(torch.equal(state[n], original[n]) for n in set(original) - trainable)
        changed_tensors.append(sum(not torch.equal(state[n], original[n]) for n in trainable))
        history = load(folder / 'training_history.json')
        assert [e['epoch'] for e in history] == [1, 2, 3]
        expected_updates = math.ceil(len(expected_fit) * 9 / 2)
        assert all(e['updates'] == expected_updates for e in history)
        assert fold['detector_updates'] == expected_updates * 3
        pools = load(folder / 'candidate_pools.json')
        examples = read(folder / 'candidate_features.csv')
        assert set(pools) == expected_fit | expected_held
        frame_examples = {}
        for e in examples:
            sid = e['sample_id']
            frame_examples.setdefault(sid, []).append(e)
            assert (e['fit'] == '1') == (sid in expected_fit)
            cs = pools[sid]
            same_center = [c for c in cs if (c['x_px'], c['y_px']) == (int(e['x_px']), int(e['y_px']))]
            assert same_center
            chosen = max(same_center, key=lambda c: (c['component_peak'], c['bbox_area'], -c['tile_index'], -c['component_index']))
            assert (chosen['tile_index'], chosen['component_index']) == (int(e['tile_index']), int(e['component_index']))
            ox, oy, w, h = TILES[chosen['tile_index']]
            assert chosen['x_px'] == int((chosen['x'] + chosen['w'] / 2) * w / 512 + ox)
            assert chosen['y_px'] == int((chosen['y'] + chosen['h'] / 2) * h / 288 + oy)
            assert chosen['component_peak'] > .3
            expected_features = independently_extract(chosen, pools[sid], degraded_reference[sid], TILES)
            assert np.array_equal(expected_features, np.array([float(e[f]) for f in F]))
            if sid in expected_fit:
                g = labels[sid]
                actual = int(math.dist((float(g['truth_x_px']), float(g['truth_y_px'])), (int(e['x_px']), int(e['y_px']))) <= 16)
                assert actual == int(e['fit_label'])
            else:
                assert e['fit_label'] == e['loss_weight'] == e['normalization_weight'] == ''
        fit_examples = [e for e in examples if e['fit'] == '1']
        X = np.array([[float(e[f]) for f in F] for e in fit_examples], dtype=np.float64)
        Y = np.array([float(e['fit_label']) for e in fit_examples])
        W, N = np.zeros(len(Y)), np.zeros(len(Y))
        groups = sorted({e['sample_id'] for e in fit_examples})
        for sid in groups:
            mask = np.array([e['sample_id'] == sid for e in fit_examples])
            N[mask] = 1 / (len(groups) * mask.sum())
            classes = set(Y[mask])
            for cls in classes:
                selected = mask & (Y == cls)
                W[selected] = 1 / (len(groups) * len(classes) * selected.sum())
        if fold['classifier_status'] == 'fitted':
            head_path = folder / 'head.npz'
            assert sha(head_path) == fold['head_sha256']
            model = dict(np.load(head_path))
            assert np.allclose(N, [float(e['normalization_weight']) for e in fit_examples], atol=1e-15)
            assert np.allclose(W, [float(e['loss_weight']) for e in fit_examples], atol=1e-15)
            mean = (X * N[:, None]).sum(0)
            scale = np.sqrt(((X - mean) ** 2 * N[:, None]).sum(0))
            scale[scale < 1e-8] = 1.
            assert np.array_equal(mean, model['mean']) and np.array_equal(scale, model['scale'])
            for e in examples:
                x = np.array([float(e[f]) for f in F])
                value = float(1 / (1 + np.exp(-(((x - model['mean']) / model['scale']) @ model['weights'] + model['bias']))))
                assert abs(value - float(e['score'])) < 1e-12
                score_count += 1
        else:
            assert all(float(e['score']) == 0 for e in examples)
        for sid in expected_held:
            for method in ['fold_native_peak', 'fold_native_ranked']:
                saved = oof[held, method, sid]
                b = baseline[sid]
                assert saved['detector_sha256'] == fold['detector_sha256']
                if int(b['visible']):
                    assert all(saved[k] == b[k] for k in ['visible', 'x_px', 'y_px', 'confidence'])
                    continue
                if method == 'fold_native_peak':
                    c = max(pools[sid], key=lambda c: (c['component_peak'], c['bbox_area'], -c['tile_index'], -c['component_index'])) if pools[sid] else None
                else:
                    es = frame_examples.get(sid, [])
                    winner = max(es, key=lambda e: (float(e['score']), next(c['component_peak'] for c in pools[sid] if (c['tile_index'], c['component_index']) == (int(e['tile_index']), int(e['component_index']))), next(c['bbox_area'] for c in pools[sid] if (c['tile_index'], c['component_index']) == (int(e['tile_index']), int(e['component_index']))), -int(e['tile_index']), -int(e['component_index']))) if es else None
                    c = next(c for c in pools[sid] if (c['tile_index'], c['component_index']) == (int(winner['tile_index']), int(winner['component_index']))) if winner and float(winner['score']) >= .5 else None
                assert int(saved['visible']) == (c is not None)
                if c:
                    assert tuple(float(saved[k]) for k in ['x_px', 'y_px', 'confidence']) == (c['x_px'], c['y_px'], c['bbox_mean'])
        model_checks.append(dict(held=held, detector_fit_frames=len(expected_fit), classifier_frames=len(groups), input_overlap=0))
        example_count += len(examples)
    assert tested == set(labels)
    def kind(g, p):
        if g['truth_visibility'] != 'visible':
            return 'output_when_not_observable' if int(p['visible']) else 'no_output_when_not_observable'
        if not int(p['visible']):
            return 'missing_visible_ball'
        return 'correct16px' if math.dist((float(p['x_px']), float(p['y_px'])), (float(g['truth_x_px']), float(g['truth_y_px']))) <= 16 else 'wrong_position'
    per_clip = []
    raw_lost = []
    for method in ['original_temporal', 'fold_native_peak', 'fold_native_ranked']:
        for phase in ['play', 'nonplay']:
            counts = Counter()
            for sid, g in labels.items():
                if phases[sid] != phase:
                    continue
                p = baseline[sid] if method == 'original_temporal' else oof[g['candidate_id'], method, sid]
                counts[kind(g, p)] += 1
            reported = next(r for r in result['summary'] if r['method'] == method and r['phase_status'] == phase)
            assert all(counts[k] == reported[k] for k in ['correct16px', 'wrong_position', 'missing_visible_ball', 'output_when_not_observable', 'no_output_when_not_observable'])
        for held in sorted({g['candidate_id'] for g in labels.values()}):
            counts = Counter()
            for sid, g in labels.items():
                if g['candidate_id'] != held or phases[sid] != 'play':
                    continue
                p = baseline[sid] if method == 'original_temporal' else oof[held, method, sid]
                counts[kind(g, p)] += 1
            per_clip.append(dict(method=method, clip_id=held, **{k: counts[k] for k in ['correct16px', 'wrong_position', 'missing_visible_ball', 'output_when_not_observable', 'no_output_when_not_observable']}))
    for sid, g in labels.items():
        if phases[sid] != 'play':
            continue
        raw = oof[g['candidate_id'], 'fold_native_peak', sid]
        ranked = oof[g['candidate_id'], 'fold_native_ranked', sid]
        if kind(g, raw) == 'correct16px' and kind(g, ranked) != 'correct16px':
            raw_lost.append(sid)
    assert sorted(raw_lost) == sorted(result['raw_correct_play_targets_lost'])
    from integrate_044_reviewed_tracks import write
    write(OUT / 'verified_play_per_clip.csv', per_clip)
    assert all(sha(Path(p)) == h for p, h in load(OUT / 'input_image_hashes.json').items())
    assert all(sha(Path(p)) == h for p, h in protocol['protected_input_hashes'].items())
    verification = dict(status='passed', detector_models_checked=8, classifier_models_checked=8,
        held_input_overlap=0, detector_or_classifier_fit_includes_held=False,
        frozen_parameters_and_BN_exact=True, changed_trainable_tensor_counts=changed_tensors,
        candidate_geometry_rows_checked=example_count, candidate_head_scores_recomputed=score_count,
        held_output_provenance_verified=246, phase_metrics_recomputed=True,
        fold_checks=model_checks, protected_inputs_unchanged=True, operational_adoption=False,
        warning='Current fit stages exclude held clips;historically selected reference/features/hyperparameters are not fully nested or independent external validation.',
        script_sha256=sha(Path(__file__)))
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
