"""Reproduce source crops, frozen embeddings, head scores and held outcomes."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights
from integrate_044_reviewed_tracks import ROOT, read, load, write, sha, cv2
from compare_grouped_tile_gate import F

BASE = ROOT.parent / 'ball_quality_20261109'
OUT = BASE / 'crossfit_candidate_appearance'
PREV = BASE / 'crossfit_native_candidate_ranker'


def main():
    protocol = load(OUT / 'protocol.json')
    result = load(OUT / 'result.json')
    assert all(sha(Path(p)) == h for p, h in protocol['protected_input_hashes'].items())
    assert sha(OUT / 'executed_source.py') == protocol['script_sha256']
    patches = read(OUT / 'patch_manifest.csv')
    pack = np.load(OUT / 'frozen_image_features.npz')
    hashes, matrix = list(pack['patch_hashes']), pack['features']
    assert matrix.shape == (539, 512) and len(patches) == 1197
    embed = dict(zip(hashes, matrix, strict=True))
    pixels = {}
    for p in patches:
        assert sha(Path(p['source_image'])) == p['source_sha256']
        image = cv2.imread(p['source_image'])
        assert image.shape == (1080, 1920, 3)
        x, y = int(p['x_px']), int(p['y_px'])
        padded = cv2.copyMakeBorder(image, 16, 16, 16, 16, cv2.BORDER_REFLECT_101)
        crop = padded[y:y + 32, x:x + 32]
        assert crop.shape == (32, 32, 3)
        assert hashlib.sha256(crop.tobytes()).hexdigest() == p['patch_bgr_sha256']
        pixels[p['patch_bgr_sha256']] = crop
    assert set(pixels) == set(hashes)
    weight = next(Path(p) for p in protocol['protected_input_hashes'] if p.endswith('resnet18-f37072fd.pth'))
    network = resnet18(weights=None)
    network.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True), strict=True)
    network.fc = torch.nn.Identity()
    network.eval().requires_grad_(False).to('cuda')
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    actual = []
    with torch.inference_mode():
        for start in range(0, len(hashes), 32):
            images = [transform(torch.from_numpy(pixels[h][:, :, ::-1].copy()).permute(2, 0, 1)) for h in hashes[start:start + 32]]
            actual.append(network(torch.stack(images).to('cuda')).cpu().numpy())
    actual = np.concatenate(actual).astype(np.float64)
    max_difference = float(np.max(np.abs(actual - matrix)))
    assert np.allclose(actual, matrix, rtol=1e-5, atol=1e-5)
    patch_by = {(p['held_clip'], p['sample_id'], p['candidate_index']): p['patch_bgr_sha256'] for p in patches}
    scores = {(r['held_clip'], r['sample_id'], r['candidate_index']): r for r in read(OUT / 'candidate_scores.csv')}
    parent_folds = {f['held_clip']: f for f in load(PREV / 'folds.json')}
    fold_records = load(OUT / 'folds.json')
    labels = {r['sample_id']: r for r in read(ROOT / 'balltrack_expanded_centers_pilot/expanded_train_labels.csv')}
    phases = {r['sample_id']: r['phase_status'] for r in read(BASE / 'training_phase_scope_review/phase_labels.csv')}
    predictions = {r['sample_id']: r for r in read(OUT / 'oof_predictions.csv')}
    original = {r['sample_id']: r for r in read(ROOT / 'ball_temporal_path_comparison/train_displacement_0.1.csv')}
    checked = set()
    gradients = []
    for fold in fold_records:
        held = fold['held_clip']
        parent = parent_folds[held]
        assert fold['detector_sha256'] == parent['detector_sha256']
        assert fold['fit_frame_ids'] == parent['classifier_fit_frames']
        assert fold['held_frame_ids'] == parent['held_sample_ids']
        assert all(labels[sid]['candidate_id'] != held and phases[sid] == 'play' and labels[sid]['truth_visibility'] == 'visible' for sid in fold['fit_frame_ids'])
        es = read(PREV / ('fold_' + held[-3:]) / 'candidate_features.csv')
        x = np.stack([np.r_[np.array([float(e[f]) for f in F]), embed[patch_by[held, e['sample_id'], e['candidate_index']]]] for e in es])
        fit = np.array([e['fit'] == '1' for e in es])
        X = x[fit]
        Y = np.array([float(e['fit_label']) for e in es if e['fit'] == '1'])
        ids = [e['sample_id'] for e in es if e['fit'] == '1']
        groups = sorted(set(ids))
        normal, weights = np.zeros(len(ids)), np.zeros(len(ids))
        for sid in groups:
            selected = np.array([v == sid for v in ids])
            normal[selected] = 1 / (len(groups) * selected.sum())
            classes = set(Y[selected])
            for cls in classes:
                mask = selected & (Y == cls)
                weights[mask] = 1 / (len(groups) * len(classes) * mask.sum())
        path = OUT / ('fold_' + held[-3:]) / 'head.npz'
        assert sha(path) == fold['head_sha256']
        model = dict(np.load(path))
        mean = (X * normal[:, None]).sum(0)
        scale = np.sqrt(((X - mean) ** 2 * normal[:, None]).sum(0))
        scale[scale < 1e-8] = 1.
        assert np.array_equal(mean, model['mean']) and np.array_equal(scale, model['scale'])
        values = 1 / (1 + np.exp(-(((x - mean) / scale) @ model['weights'] + model['bias'])))
        for e, value in zip(es, values, strict=True):
            assert abs(float(scores[held, e['sample_id'], e['candidate_index']]['score']) - value) < 1e-12
        z = (X - mean) / scale
        p = 1 / (1 + np.exp(-(z @ model['weights'] + model['bias'])))
        residual = (p - Y) * weights
        gradient = np.r_[z.T @ residual + model['weights'], residual.sum()]
        gradients.append(float(np.max(np.abs(gradient))))
        assert gradients[-1] < 1e-5
        pools = load(PREV / ('fold_' + held[-3:]) / 'candidate_pools.json')
        for sid in fold['held_frame_ids']:
            assert sid not in checked
            checked.add(sid)
            saved, b = predictions[sid], original[sid]
            assert saved['detector_sha256'] == parent['detector_sha256']
            if int(b['visible']):
                assert all(saved[k] == b[k] for k in ['visible', 'x_px', 'y_px', 'confidence'])
                continue
            candidates = []
            for e, value in zip(es, values, strict=True):
                if e['sample_id'] != sid:
                    continue
                c = next(c for c in pools[sid] if (c['tile_index'], c['component_index']) == (int(e['tile_index']), int(e['component_index'])))
                candidates.append((float(value), c))
            selected = max(candidates, key=lambda v: (v[0], v[1]['component_peak'], v[1]['bbox_area'], -v[1]['tile_index'], -v[1]['component_index'])) if candidates else None
            c = selected[1] if selected and selected[0] >= .5 else None
            assert int(saved['visible']) == (c is not None)
            if c:
                assert tuple(float(saved[k]) for k in ['x_px', 'y_px', 'confidence']) == (c['x_px'], c['y_px'], c['bbox_mean'])
    assert checked == set(labels)
    def kind(g, p):
        if g['truth_visibility'] != 'visible':
            return 'output_when_not_observable' if int(p['visible']) else 'no_output_when_not_observable'
        if not int(p['visible']):
            return 'missing_visible_ball'
        return 'correct16px' if math.dist((float(p['x_px']), float(p['y_px'])), (float(g['truth_x_px']), float(g['truth_y_px']))) <= 16 else 'wrong_position'
    for phase in ['play', 'nonplay']:
        counts = Counter(kind(labels[sid], p) for sid, p in predictions.items() if phases[sid] == phase)
        row = next(r for r in result['summary'] if r['method'] == 'geometry10_plus_image512' and r['phase_status'] == phase)
        assert all(row[k] == counts[k] for k in ['correct16px', 'wrong_position', 'missing_visible_ball', 'output_when_not_observable', 'no_output_when_not_observable'])
    old = {r['sample_id']: r for r in read(PREV / 'oof_predictions.csv') if r['method'] == 'fold_native_ranked'}
    changes = []
    for sid, p in predictions.items():
        if phases[sid] != 'play':
            continue
        before, after = kind(labels[sid], old[sid]), kind(labels[sid], p)
        if before != after:
            changes.append(dict(sample_id=sid, local_frame=p['local_frame'], source_frame=p['source_frame'], geometry_outcome=before, appearance_outcome=after))
    if changes:
        write(OUT / 'verified_play_changes.csv', changes)
    assert all(sha(Path(p)) == h for p, h in protocol['protected_input_hashes'].items())
    verification = dict(status='passed', source_crops_verified=1197, unique_image_features_recomputed=539,
        maximum_embedding_abs_difference=max_difference, heads_checked=8, head_scores_verified=1197,
        optimization_max_abs_gradients=gradients, held_predictions_verified=123,
        detector_fit_and_head_fit_exclusion_preserved=True, geometry_and_candidate_sources_unchanged=True,
        verified_play_outcome_changes=changes, protected_inputs_unchanged=True,
        operational_adoption=False, script_sha256=sha(Path(__file__)))
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
