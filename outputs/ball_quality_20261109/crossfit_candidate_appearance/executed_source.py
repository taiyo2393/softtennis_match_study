"""Add frozen pixel features to identical clip-excluded candidates;one fixed head."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from integrate_044_reviewed_tracks import ROOT, SOURCE, read, load, write, sha, cv2
from compare_grouped_tile_gate import F
from crossfit_native_candidate_ranker import fit_head, predict_head
from evaluate_training_phase_scope import classify

BASE = ROOT.parent / 'ball_quality_20261109'
PREV = BASE / 'crossfit_native_candidate_ranker'
OUT = BASE / 'crossfit_candidate_appearance'
WEIGHTS = Path(r'C:\Users\ta239\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def crop32(image, x, y):
    assert image.shape == (1080, 1920, 3) and 0 <= x < 1920 and 0 <= y < 1080
    padded = cv2.copyMakeBorder(image, 16, 16, 16, 16, cv2.BORDER_REFLECT_101)
    result = padded[y:y + 32, x:x + 32]
    assert result.shape == (32, 32, 3)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Preserve one fixed feature addition;no post-result retries.'
    assert load(PREV / 'verification.json')['status'] == 'passed'
    folds = load(PREV / 'folds.json')
    labels = {g['sample_id']: g for g in read(ROOT / 'balltrack_expanded_centers_pilot/expanded_train_labels.csv')}
    phases = {g['sample_id']: g['phase_status'] for g in read(BASE / 'training_phase_scope_review/phase_labels.csv')}
    original = {r['sample_id']: r for r in read(ROOT / 'ball_temporal_path_comparison/train_displacement_0.1.csv')}
    protected_paths = [WEIGHTS, PREV / 'protocol.json', PREV / 'result.json', PREV / 'verification.json',
        PREV / 'folds.json', PREV / 'oof_predictions.csv',
        ROOT / 'crossfit_native_candidate_ranker.py', ROOT / 'balltrack_expanded_centers_pilot/expanded_train_labels.csv',
        BASE / 'training_phase_scope_review/phase_labels.csv',
        ROOT / 'ball_temporal_path_comparison/train_displacement_0.1.csv',
        BASE / 'temporal_guard_next_sequences/completed_review/manual_sequence_labels.csv',
        BASE / 'temporal_guard_next_sequences/evaluation_report.json',
        ROOT.parent / 'ball_quality_20261104/completed_review/manual_sequence_labels.csv',
        ROOT / 'reviewed_ball_overrides.json', ROOT / 'reviewed_person_overrides.json', ROOT / 'reviewed_tracking_bundles.json']
    for fold in folds:
        folder = PREV / ('fold_' + fold['held_clip'][-3:])
        protected_paths += [folder / n for n in ['fold.json', 'head.npz', 'detector_final03.pth', 'candidate_features.csv', 'candidate_pools.json']]
    protected = {str(p): sha(p) for p in protected_paths}
    protocol = dict(status='fixed_before_new_pixel_features_or_head_fits', date_jst='2026-11-09',
        candidate_source='Exactly1197existing fold-specific candidate rows from8detectors trained without held clip. No new detections,thresholds,view choices,GTcrops or detector training.',
        proposal='One geometry10+frozenResNet18_512 concatenation,522features. Geometry-only10 saved heads are the matched baseline. No alternate image-only head or feature-size search.',
        pixels='Fixed32x32nativeBGRsourcecrop centered at integer candidate coordinate;reflect101padding only at image edges;BGR->RGB. Candidate centers choose crops,never manual centers.',
        backbone='Local cached torchvisionResNet18 IMAGENET1K_V1,fc Identity,eval,allweights andBNfrozen. No downloads or fine tuning.',
        transform=str(ResNet18_Weights.IMAGENET1K_V1.transforms()), torchvision_version=torchvision.__version__, torch_version=torch.__version__,
        context='Upsampling32nativepixels to classifier input adds no source detail. This method comparison changes feature content and dimensionality;not a causal isolation of pretraining.',
        fitting='Reuse SAME fit flags and target-position labels as each parent fold,frame-equal normalization,within-frame equal-class weighted BCE+0.5L2,CPUfloat64,zero-init,LBFGS300,threshold0.5. Held clip never fits mean,scale,head or detector.',
        scores='Uncalibrated sigmoid,not probability. Frozen feature caching across folds shares no learned label information.',
        output='Rank all exact-center-deduplicated candidates byscore,thenpeak,area,lower tile/index. Only fill missing original temporal outputs with actual observed centers. No interpolation or coordinate modification.',
        primary_safety='Retain ALL64raw-fold correct PLAYtargets;wrong_position0,output_when_not_observable<=1,correct>52original. If fails,stop before full-data head or fresh confirmation. Do not switch thresholds or choose another variant.',
        limitations='Post-failure design on8known same-match training clips. Current detector/head fit exclusion preserved,methodological history not fully nested,not external performance. Old36candidate labels and EF26/ABCD52not used in fits.',
        protected_input_hashes=protected, script_sha256=sha(Path(__file__)), automatic_adoption=False)
    dump(OUT / 'protocol.json', protocol)
    import shutil
    shutil.copy2(Path(__file__), OUT / 'executed_source.py')
    backbone = resnet18(weights=None)
    backbone.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True), strict=True)
    backbone.fc = torch.nn.Identity()
    backbone.eval().requires_grad_(False)
    backbone.to('cuda')
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    fold_examples, patch_manifest, patch_arrays = {}, [], {}
    source_hashes = {}
    for fold in folds:
        held = fold['held_clip']
        examples = read(PREV / ('fold_' + held[-3:]) / 'candidate_features.csv')
        fold_examples[held] = examples
        cached_images = {}
        for e in examples:
            sid = e['sample_id']
            g = labels[sid]
            path = json.loads(g['input_frames_json'])[-1]
            digest = json.loads(g['input_frame_sha256_json'])[-1]
            assert sha(Path(path)) == digest
            source_hashes[path] = digest
            if sid not in cached_images:
                cached_images[sid] = cv2.imread(path)
            x, y = int(e['x_px']), int(e['y_px'])
            bgr = crop32(cached_images[sid], x, y)
            patch_hash = hashlib.sha256(bgr.tobytes()).hexdigest()
            patch_arrays.setdefault(patch_hash, bgr)
            patch_manifest.append(dict(held_clip=held, sample_id=sid, candidate_index=e['candidate_index'],
                source_image=path, source_sha256=digest, x_px=x, y_px=y, patch_bgr_sha256=patch_hash))
    assert len(patch_manifest) == 1197
    hashes = sorted(patch_arrays)
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(hashes), 32):
            batch = [transform(torch.from_numpy(patch_arrays[h][:, :, ::-1].copy()).permute(2, 0, 1)) for h in hashes[start:start + 32]]
            result = backbone(torch.stack(batch).to('cuda')).cpu().numpy()
            embeddings.append(result)
    matrix = np.concatenate(embeddings).astype(np.float64)
    assert matrix.shape == (len(hashes), 512) and np.isfinite(matrix).all()
    embed = dict(zip(hashes, matrix, strict=True))
    np.savez_compressed(OUT / 'frozen_image_features.npz', patch_hashes=np.array(hashes), features=matrix)
    write(OUT / 'patch_manifest.csv', patch_manifest)
    dump(OUT / 'source_image_hashes.json', source_hashes)
    patch_by = {(r['held_clip'], r['sample_id'], r['candidate_index']): r['patch_bgr_sha256'] for r in patch_manifest}
    scores, predictions, model_records = [], [], []
    for fold in folds:
        held = fold['held_clip']
        es = fold_examples[held]
        folder = OUT / ('fold_' + held[-3:])
        folder.mkdir()
        X = np.stack([np.r_[np.array([float(e[f]) for f in F]), embed[patch_by[held, e['sample_id'], e['candidate_index']]]] for e in es])
        train = np.array([e['fit'] == '1' for e in es])
        y = np.array([float(e['fit_label']) for e in es if e['fit'] == '1'], dtype=np.float64)
        fit_ids = [e['sample_id'] for e in es if e['fit'] == '1']
        assert all(labels[sid]['candidate_id'] != held and phases[sid] == 'play' and labels[sid]['truth_visibility'] == 'visible' for sid in fit_ids)
        assert set(fit_ids) == set(fold['classifier_fit_frames'])
        model, sw, nw = fit_head(X[train], y, fit_ids)
        np.savez_compressed(folder / 'head.npz', **model)
        values = predict_head(model, X)
        by_sid = {}
        for e, s in zip(es, values, strict=True):
            by_sid.setdefault(e['sample_id'], []).append((e, float(s)))
            scores.append(dict(held_clip=held, sample_id=e['sample_id'], candidate_index=e['candidate_index'],
                fit=e['fit'], score=float(s), head_sha256=sha(folder / 'head.npz')))
        pools = load(PREV / ('fold_' + held[-3:]) / 'candidate_pools.json')
        for sid in fold['held_sample_ids']:
            candidates = by_sid.get(sid, [])
            def key(item):
                e, score = item
                c = next(c for c in pools[sid] if (c['tile_index'], c['component_index']) == (int(e['tile_index']), int(e['component_index'])))
                return score, c['component_peak'], c['bbox_area'], -c['tile_index'], -c['component_index']
            winner = max(candidates, key=key) if candidates else None
            c = next(c for c in pools[sid] if (c['tile_index'], c['component_index']) == (int(winner[0]['tile_index']), int(winner[0]['component_index']))) if winner and winner[1] >= .5 else None
            g, b = labels[sid], original[sid]
            if int(b['visible']):
                point = {k: b[k] for k in ['visible', 'x_px', 'y_px', 'confidence']}
                origin = 'unchanged_original_temporal'
            else:
                point = dict(visible=int(c is not None), x_px=c['x_px'] if c else 0, y_px=c['y_px'] if c else 0, confidence=c['bbox_mean'] if c else 0.)
                origin = 'actual_candidate' if c else 'missing'
            predictions.append(dict(**{k: g[k] for k in ['sample_id', 'candidate_id', 'local_frame', 'source_frame', 'truth_visibility', 'truth_x_px', 'truth_y_px']},
                method='geometry10_plus_image512', **point, output_source=origin, held_clip=held,
                detector_sha256=fold['detector_sha256'], head_sha256=sha(folder / 'head.npz')))
        record = dict(held_clip=held, detector_sha256=fold['detector_sha256'], fit_frame_ids=sorted(set(fit_ids)),
            held_frame_ids=fold['held_sample_ids'], fit_candidates=int(train.sum()), feature_dimension=522,
            head_sha256=sha(folder / 'head.npz'), detector_not_refit=True)
        dump(folder / 'fold.json', record)
        model_records.append(record)
        print('Frozen appearance fold', held[-3:], 'complete', flush=True)
    write(OUT / 'candidate_scores.csv', scores)
    write(OUT / 'oof_predictions.csv', predictions)
    dump(OUT / 'folds.json', model_records)
    parent = read(PREV / 'oof_predictions.csv')
    methods = [('original_temporal', list(original.values())),
        ('fold_native_peak', [r for r in parent if r['method'] == 'fold_native_peak']),
        ('geometry10', [r for r in parent if r['method'] == 'fold_native_ranked']),
        ('geometry10_plus_image512', predictions)]
    summaries, per_clip, outcomes = [], [], {}
    for name, rows in methods:
        outcomes[name] = {r['sample_id']: classify(r, r)[0] for r in rows}
        for phase in ['play', 'nonplay']:
            counts = Counter(outcomes[name][sid] for sid in original if phases[sid] == phase)
            summaries.append(dict(method=name, phase_status=phase, **{k: counts[k] for k in ['correct16px', 'wrong_position', 'missing_visible_ball', 'output_when_not_observable', 'no_output_when_not_observable']}))
        for held in sorted(fold_examples):
            counts = Counter(outcomes[name][sid] for sid, g in labels.items() if g['candidate_id'] == held and phases[sid] == 'play')
            per_clip.append(dict(method=name, held_clip=held, **{k: counts[k] for k in ['correct16px', 'wrong_position', 'missing_visible_ball', 'output_when_not_observable', 'no_output_when_not_observable']}))
    paired = [dict(sample_id=sid, phase_status=phases[sid], raw_outcome=outcomes['fold_native_peak'][sid], geometry_outcome=outcomes['geometry10'][sid],
        appearance_outcome=outcomes['geometry10_plus_image512'][sid],
        raw_correct_lost=int(phases[sid] == 'play' and outcomes['fold_native_peak'][sid] == 'correct16px' and outcomes['geometry10_plus_image512'][sid] != 'correct16px')) for sid in original]
    lost = [r['sample_id'] for r in paired if r['raw_correct_lost']]
    final = next(r for r in summaries if r['method'] == 'geometry10_plus_image512' and r['phase_status'] == 'play')
    safe = not lost and final['correct16px'] > 52 and final['wrong_position'] == 0 and final['output_when_not_observable'] <= 1
    write(OUT / 'summary.csv', summaries)
    write(OUT / 'per_clip_summary.csv', per_clip)
    write(OUT / 'paired_changes.csv', paired)
    assert all(sha(Path(p)) == h for p, h in protected.items())
    assert all(sha(Path(p)) == h for p, h in source_hashes.items())
    result = dict(status='fixed_pixel_feature_addition_complete', unique_patches=len(hashes), candidate_feature_rows=1197,
        feature_dimension=522, folds=8, held_predictions=123, summary=summaries,
        raw_correct_play_targets_lost=lost, passes_internal_gate=safe,
        frozen_backbone_not_finetuned=True, detector_models_and_candidate_rows_unchanged=True,
        new_confirmation_inference=False, full_fit_performed=False, operational_adoption=False, canonical_tracking_changed=False,
        interpretation='Same candidate/fold method comparison;post-failure design on known same-match data,not independent external validation.',
        decision='Requires separate full-fit check before new confirmation.' if safe else 'Stop:fixed pixel-feature head fails safety. No alternate size,threshold or head chosen.')
    dump(OUT / 'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
