"""Clip-excluded detector adaptation AND candidate-head fitting,one fixed recipe."""
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT, SOURCE, read, load, write, sha, cv2
from compare_training_tile_detail import TILES, transform, mapped, peak_choice
from compare_ball_candidate_decoders import components
from compare_grouped_tile_gate import F, features
from run_manual_ball_finetune import tensors
from evaluate_training_phase_scope import classify

BASE = ROOT.parent / 'ball_quality_20261109'
PREV = BASE / 'native_tile_point_pilot'
OLD = BASE / 'training_tile_detail_comparison'
OUT = BASE / 'crossfit_native_candidate_ranker'
SEED = 20261109


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def deduplicate(pool):
    """Exact source-center duplicates only;geometry/peak decides,not labels."""
    grouped = {}
    for c in pool:
        grouped.setdefault((c['x_px'], c['y_px']), []).append(c)
    return [max(v, key=lambda c: (c['component_peak'], c['bbox_area'], -c['tile_index'], -c['component_index'])) for v in grouped.values()]


def fit_head(x, y, ids):
    groups = sorted(set(ids))
    sw = np.zeros(len(y), dtype=np.float64)
    normal = np.zeros(len(y), dtype=np.float64)
    for sid in groups:
        selected = np.array([v == sid for v in ids])
        normal[selected] = 1 / (len(groups) * selected.sum())
        classes = sorted(set(y[selected]))
        for cls in classes:
            mask = selected & (y == cls)
            sw[mask] = 1 / (len(groups) * len(classes) * mask.sum())
    assert np.isclose(normal.sum(), 1) and np.isclose(sw.sum(), 1)
    mean = (x * normal[:, None]).sum(0)
    scale = np.sqrt(((x - mean) ** 2 * normal[:, None]).sum(0))
    scale[scale < 1e-8] = 1.
    X, Y, W = torch.from_numpy((x - mean) / scale), torch.from_numpy(y), torch.from_numpy(sw)
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, bias], lr=1., max_iter=300, line_search_fn='strong_wolfe', tolerance_grad=1e-9, tolerance_change=1e-12)
    def closure():
        optimizer.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(X @ w + bias, Y, reduction='none') * W).sum() + .5 * w.square().sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    return dict(mean=mean, scale=scale, weights=w.detach().numpy(), bias=np.array(float(bias.detach()))), sw, normal


def predict_head(model, x):
    z = ((x - model['mean']) / model['scale']) @ model['weights'] + model['bias']
    return 1 / (1 + np.exp(-z))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'No retries or extra setting choices after results.'
    labels = read(ROOT / 'balltrack_expanded_centers_pilot/expanded_train_labels.csv')
    phases = {r['sample_id']: r['phase_status'] for r in read(BASE / 'training_phase_scope_review/phase_labels.csv')}
    visible = [g for g in labels if phases[g['sample_id']] == 'play' and g['truth_visibility'] == 'visible']
    groups = sorted({g['candidate_id'] for g in labels})
    assert len(labels) == 123 and len(visible) == 68 and len(groups) == 8
    by_gt = {g['sample_id']: g for g in labels}
    original = {r['sample_id']: r for r in read(ROOT / 'ball_temporal_path_comparison/train_displacement_0.1.csv')}
    old_protocol = load(OLD / 'protocol.json')
    plans = old_protocol['plans']
    modelroot = SOURCE / 'third_party/RacketVision/source/BallTrack'
    weight = modelroot / 'checkpoints/balltrack_best.pth'
    paths = [ROOT / 'balltrack_expanded_centers_pilot/expanded_train_labels.csv',
        BASE / 'training_phase_scope_review/phase_labels.csv', ROOT / 'ball_temporal_path_comparison/train_displacement_0.1.csv',
        weight, OLD / 'protocol.json', OLD / 'candidate_pools.json', OLD / 'input_image_hashes.json',
        PREV / 'protocol.json', PREV / 'result.json', PREV / 'final_epoch03.pth', PREV / 'verification.json',
        BASE / 'temporal_guard_next_sequences/completed_review/manual_sequence_labels.csv',
        BASE / 'temporal_guard_next_sequences/evaluation_report.json',
        ROOT.parent / 'ball_quality_20261104/completed_review/manual_sequence_labels.csv',
        ROOT / 'reviewed_ball_overrides.json', ROOT / 'reviewed_person_overrides.json', ROOT / 'reviewed_tracking_bundles.json']
    paths += [Path(p['median_path']) for p in plans]
    protected = {str(p): sha(p) for p in paths}
    helpers = [ROOT / n for n in ['compare_training_tile_detail.py', 'compare_ball_candidate_decoders.py', 'compare_grouped_tile_gate.py', 'run_manual_ball_finetune.py', 'evaluate_training_phase_scope.py']]
    protocol = dict(status='fixed_before_crossfit_detector_or_head_training', date_jst='2026-11-09',
        scope='Training8clips only. Detector and classifier are BOTH refit without each held clip. No conditional-head-only result called out-of-fold pipeline performance.',
        detector='Same previous fixed native2x recipe:original initialization,last up_block_3+predictor,BN buffers fixed,source WBCELoss,Gaussian sigma3.5,all9crops of visible-play fit frames,3epochs,Adam1e-4,batch2,clip1,seed20261109,only final epoch3.',
        generation='All target-frame native9tilecomponents>0.3;deduplicate only EXACT source center,highestpeak,area,lower tile/index ties. No teacher-dependent proposal choice.',
        features=F, reference_features='The unchanged ORIGINAL-model degraded9tile pool supplies degraded support. Current fold model supplies native geometry/response/support. No coordinate/time/clip IDs or manual center in head inputs.',
        head_labels='All exact-center-deduplicated native candidates at human-visible PLAYfit frames. Positive iff within16px of human center,otherwise target-localization negative,not generic non-ball. Unobservable/nonplay/held centers never fit detector or head.',
        head='One logistic10head:frame-equal normalization(each candidate1/n_frame);within-frame equal-class weighting;CPUfloat64,zero-init,LBFGS300,strong_wolfe,weighted BCE+0.5L2weights,bias unpenalized,threshold0.5. No parameter search.',
        ranking='Highest head score among ALL same-frame candidates,thenpeak,area,lower tile/component. Accept if>=0.5. Score uncalibrated. Preserve every existing original temporal output;only supplement missing positions. Raw peak-selected fold detector is the matched control.',
        separation='Fit source4frame hashes disjoint from held labeled inputs. Store exact detector fit frame IDs,head frame IDs,fold hashes and held outputs. Full-clip median is unsupervised offline preprocessing,not a label.',
        safety='Across held PLAYtargets retain every correct raw-fold-detector missing-only output;require correct>52original baseline,wrong_position<=0,output_when_not_observable<=1. Failure stops before all full head fits,new26/52/other confirmation inference. Baseline outputs preserved by construction.',
        full_fit='Only if crossfit passes,fit same head recipe on existing final-all68 detector pools,then require ALL64correct retained and0wrong/<=1unobservable output. Full fit is a training check,not held performance.',
        limitations='Methods motivated by known training failures;8same-match clips,sparse adjacent labels,possible adaptive development bias. Crossfit reduces direct fit leakage but is not independent external testing.',
        new_detector_hyperparameters=False, new_human_labels=0, automatic_adoption=False,
        protected_input_hashes=protected, helper_hashes={str(p): sha(p) for p in helpers}, script_sha256=sha(Path(__file__)))
    dump(OUT / 'protocol.json', protocol)
    import shutil
    shutil.copy2(Path(__file__), OUT / 'executed_source.py')
    sys.path.insert(0, str(modelroot))
    from inference import BallInferencer
    from model.loss_utils import WBCELoss
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    engine = BallInferencer(str(modelroot / 'configs/tracknetv3_base.py'), str(weight), device='cuda', thre=.3, batchsize=4)
    net = engine.model
    initial_state = {n: p.detach().cpu().clone() for n, p in net.state_dict().items()}
    params, names = [], []
    for name, p in net.named_parameters():
        trainable = name.startswith('up_block_3.') or name.startswith('predictor.')
        p.requires_grad_(trainable)
        if trainable:
            params.append(p)
            names.append(name)
    frozen = set(initial_state) - set(names)
    # Pixel inputs may be cached for all clips; only fit IDs select targets and gradients.
    medians = {}
    for plan in plans:
        m = np.load(plan['median_path'])['median']
        for tile, box in enumerate(TILES):
            medians[plan['candidate_id'] + '_' + str(tile)] = np.moveaxis(transform(m, box, 'native').astype(np.float32) / 255., -1, 0)
    expected_images = load(OLD / 'input_image_hashes.json')
    image_hashes = {}
    fit_input_cache = {}
    for g in visible:
        images = []
        for path, digest in zip(json.loads(g['input_frames_json']), json.loads(g['input_frame_sha256_json']), strict=True):
            assert sha(Path(path)) == digest
            image_hashes[path] = digest
            images.append(cv2.imread(path))
        fit_input_cache[g['sample_id']] = [dict(frames=np.stack([transform(im, box, 'native') for im in images]), median_key=g['candidate_id'] + '_' + str(i), row={}) for i, box in enumerate(TILES)]
    old_degraded = load(OLD / 'candidate_pools.json')['degraded']
    all_features, folds, oof_rows, history, all_pools = [], [], [], [], {}
    yy, xx = np.ogrid[:288, :512]
    for held in groups:
        fold_dir = OUT / ('fold_' + held[-3:])
        fold_dir.mkdir()
        fit_rows = [g for g in visible if g['candidate_id'] != held]
        fit_ids = {g['sample_id'] for g in fit_rows}
        held_rows = [g for g in labels if g['candidate_id'] == held]
        held_ids = {g['sample_id'] for g in held_rows}
        assert not fit_ids & held_ids
        train_hashes = {h for g in fit_rows for h in json.loads(g['input_frame_sha256_json'])}
        held_hashes = {h for g in held_rows for h in json.loads(g['input_frame_sha256_json'])}
        assert not train_hashes & held_hashes
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        net.load_state_dict(initial_state, strict=True)
        net.eval()
        samples = []
        for g in fit_rows:
            for tile, box in enumerate(TILES):
                ox, oy, w, h = box
                contains = ox <= float(g['truth_x_px']) < ox + w and oy <= float(g['truth_y_px']) < oy + h
                if contains:
                    gx, gy = int((float(g['truth_x_px']) - ox) * 512 / w), int((float(g['truth_y_px']) - oy) * 288 / h)
                    target = np.exp(-.5 * ((xx - gx) ** 2 + (yy - gy) ** 2) / 3.5 ** 2).astype(np.float32)[None]
                else:
                    target = np.zeros((1, 288, 512), np.float32)
                samples.append(dict(fit_input_cache[g['sample_id']][tile], target=target))
        optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=0.)
        scaler = torch.amp.GradScaler('cuda')
        generator = np.random.default_rng(SEED)
        epoch_history = []
        for epoch in range(1, 4):
            losses = []
            order = generator.permutation(len(samples))
            for start in range(0, len(order), 2):
                batch = [samples[i] for i in order[start:start + 2]]
                x = tensors(batch, medians, engine.device)
                y = torch.from_numpy(np.stack([s['target'] for s in batch])).to(engine.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    output, _, _ = net.forward(frames=x)
                loss = WBCELoss(output.float(), y)
                assert torch.isfinite(loss)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(params, 1.)
                assert torch.isfinite(norm)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
            epoch_history.append(dict(epoch=epoch, updates=len(losses), mean_loss=float(np.mean(losses))))
            print('Fold', held[-3:], 'detector epoch', epoch, 'complete', flush=True)
        assert all(torch.equal(net.state_dict()[n].detach().cpu(), initial_state[n]) for n in frozen)
        checkpoint = fold_dir / 'detector_final03.pth'
        torch.save(dict(state_dict={n: t.detach().cpu() for n, t in net.state_dict().items()},
            meta=dict(held_clip=held, detector_fit_sample_ids=sorted(fit_ids), protocol_sha256=sha(OUT / 'protocol.json'))), checkpoint)
        required = fit_ids | held_ids
        pools = {sid: [] for sid in required}
        for plan in plans:
            cid = plan['candidate_id']
            wanted = {int(g['local_frame']): g['sample_id'] for g in labels if g['candidate_id'] == cid and g['sample_id'] in required}
            if not wanted:
                continue
            for ts in plan['batches']:
                if not any(t in wanted for t in ts):
                    continue
                images = {}
                for j in sorted({j for t in ts for j in range(t - 3, t + 1)}):
                    path = Path(plan['frame_dir']) / f'{j:06d}.jpg'
                    digest = sha(path)
                    assert digest == expected_images[str(path)]
                    image_hashes[str(path)] = digest
                    images[j] = cv2.imread(str(path))
                for tile, box in enumerate(TILES):
                    batch = [dict(frames=np.stack([transform(images[j], box, 'native') for j in range(t - 3, t + 1)]), median_key=cid + '_' + str(tile), row={}) for t in ts]
                    with torch.inference_mode(), torch.amp.autocast('cuda'):
                        output, _, _ = net.forward(frames=tensors(batch, medians, engine.device))
                    for t, hm in zip(ts, output[:, 0].float().cpu().numpy(), strict=True):
                        if t in wanted:
                            pools[wanted[t]].extend(mapped(components(hm), tile))
        dump(fold_dir / 'candidate_pools.json', pools)
        examples = []
        selected_candidates = {}
        for sid, pool in sorted(pools.items()):
            unique = deduplicate(pool)
            selected_candidates[sid] = unique
            for index, c in enumerate(unique):
                x = features(c, pool, old_degraded[sid], TILES)
                g = by_gt[sid]
                is_fit = sid in fit_ids
                error = math.hypot(c['x_px'] - float(g['truth_x_px']), c['y_px'] - float(g['truth_y_px'])) if is_fit else None
                examples.append(dict(sample_id=sid, candidate_index=index, x_px=c['x_px'], y_px=c['y_px'],
                    tile_index=c['tile_index'], component_index=c['component_index'], fit=int(is_fit),
                    fit_label=int(error <= 16) if is_fit else '', **dict(zip(F, x.tolist()))))
        train_examples = [e for e in examples if e['fit']]
        X = np.array([[e[f] for f in F] for e in train_examples], dtype=np.float64)
        Y = np.array([e['fit_label'] for e in train_examples], dtype=np.float64)
        if len(Y) and set(Y) == {0., 1.}:
            head, sw, nw = fit_head(X, Y, [e['sample_id'] for e in train_examples])
            np.savez_compressed(fold_dir / 'head.npz', **head)
            status = 'fitted'
        else:
            head, status = None, 'reject_proposals_missing_fit_class'
            sw = nw = np.zeros(len(Y))
        for e, s, n in zip(train_examples, sw, nw, strict=True):
            e.update(loss_weight=float(s), normalization_weight=float(n))
        # Complete rows share fields,including held candidates which never supply fit labels.
        for e in examples:
            e.setdefault('loss_weight', '')
            e.setdefault('normalization_weight', '')
            c = selected_candidates[e['sample_id']][e['candidate_index']]
            e['score'] = float(predict_head(head, np.array([[e[f] for f in F]], dtype=np.float64))[0]) if head else 0.
        write(fold_dir / 'candidate_features.csv', examples)
        by_examples = {(e['sample_id'], e['candidate_index']): e for e in examples}
        for g in held_rows:
            sid = g['sample_id']
            unique = selected_candidates[sid]
            raw = peak_choice(pools[sid])
            ranked = sorted(range(len(unique)), key=lambda i: (by_examples[sid, i]['score'], unique[i]['component_peak'], unique[i]['bbox_area'], -unique[i]['tile_index'], -unique[i]['component_index']), reverse=True)
            chosen = unique[ranked[0]] if ranked and by_examples[sid, ranked[0]]['score'] >= .5 else None
            for method, c in [('fold_native_peak', raw), ('fold_native_ranked', chosen)]:
                baseline = original[sid]
                if int(baseline['visible']):
                    output = {k: baseline[k] for k in ['visible', 'x_px', 'y_px', 'confidence']}
                    source = 'unchanged_original_temporal'
                else:
                    output = dict(visible=int(c is not None), x_px=c['x_px'] if c else 0, y_px=c['y_px'] if c else 0, confidence=c['bbox_mean'] if c else 0.)
                    source = 'actual_native_candidate' if c else 'missing'
                oof_rows.append(dict(**{k: g[k] for k in ['sample_id', 'candidate_id', 'local_frame', 'source_frame', 'truth_visibility', 'truth_x_px', 'truth_y_px']},
                    method=method, **output, output_source=source, held_clip=held, detector_sha256=sha(checkpoint)))
        fold_record = dict(held_clip=held, detector_fit_frames=len(fit_rows), detector_fit_sample_ids=sorted(fit_ids),
            detector_crop_samples=len(samples), detector_updates=sum(r['updates'] for r in epoch_history),
            classifier_fit_frames=sorted({e['sample_id'] for e in train_examples}), classifier_examples=len(train_examples),
            classifier_positive=int((Y == 1).sum()), classifier_negative=int((Y == 0).sum()), classifier_status=status,
            held_sample_ids=sorted(held_ids), input_hash_overlap=0, detector_sha256=sha(checkpoint),
            head_sha256=sha(fold_dir / 'head.npz') if head else '', frozen_weights_and_BN_unchanged=True)
        dump(fold_dir / 'fold.json', fold_record)
        dump(fold_dir / 'training_history.json', epoch_history)
        folds.append(fold_record)
        write(OUT / 'oof_predictions.csv', oof_rows)
        dump(OUT / 'folds.json', folds)
        print('Fold', held[-3:], 'complete: head positives', fold_record['classifier_positive'], 'negatives', fold_record['classifier_negative'], flush=True)
    assert len(oof_rows) == 246 and len(folds) == 8
    counts_by, summary, paired = {}, [], []
    for method in ['original_temporal', 'fold_native_peak', 'fold_native_ranked']:
        rows = list(original.values()) if method == 'original_temporal' else [r for r in oof_rows if r['method'] == method]
        counts_by[method] = {r['sample_id']: classify(r, r)[0] for r in rows}
        for phase in ['play', 'nonplay']:
            counts = Counter(counts_by[method][sid] for sid in original if phases[sid] == phase)
            summary.append(dict(method=method, phase_status=phase, **{k: counts[k] for k in ['correct16px', 'wrong_position', 'missing_visible_ball', 'output_when_not_observable', 'no_output_when_not_observable']}))
    for sid in original:
        a, b = counts_by['fold_native_peak'][sid], counts_by['fold_native_ranked'][sid]
        paired.append(dict(sample_id=sid, phase_status=phases[sid], raw_outcome=a, ranked_outcome=b,
            correct_lost=int(phases[sid] == 'play' and a == 'correct16px' and b != 'correct16px')))
    write(OUT / 'summary.csv', summary)
    write(OUT / 'paired_changes.csv', paired)
    candidate = next(r for r in summary if r['method'] == 'fold_native_ranked' and r['phase_status'] == 'play')
    lost = [r['sample_id'] for r in paired if r['correct_lost']]
    safe = not lost and candidate['correct16px'] > 52 and candidate['wrong_position'] == 0 and candidate['output_when_not_observable'] <= 1
    # This script stops at crossfit. A passing proposal requires a separately recorded full fit.
    assert all(sha(Path(p)) == h for p, h in protected.items())
    assert all(sha(Path(p)) == h for p, h in image_hashes.items())
    dump(OUT / 'input_image_hashes.json', image_hashes)
    result = dict(status='detector_and_head_clip_crossfit_complete', folds=8,
        held_predictions=246, original_train_frames=68, total_detector_updates=sum(f['detector_updates'] for f in folds),
        summary=summary, raw_correct_play_targets_lost=lost, passes_internal_gate=safe,
        no_held_clip_in_detector_or_head_fit=True, conditional_head_only_leakage_avoided=True,
        full_fit_performed=False, new_confirmation_inference=False, operational_adoption=False,
        canonical_tracking_changed=False,
        decision='Proceed only to separate full-fit training check.' if safe else 'Stop: fixed candidate fails crossfit gate. No full head fit,threshold retry or confirmation inference.',
        limitations='Internal development on8known same-match training clips,not independent external validation;old52andEF26labels only hashed for protection.')
    dump(OUT / 'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
