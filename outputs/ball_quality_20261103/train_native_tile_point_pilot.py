"""One training-only native2x point-heatmap adaptation; final epoch fixed upfront."""
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT, SOURCE, load, read, write, sha, cv2
from compare_training_tile_detail import TILES, transform, mapped, peak_choice
from compare_ball_candidate_decoders import components
from run_manual_ball_finetune import tensors
from evaluate_training_phase_scope import classify
from calibrate_point_supervised_detector import compact, expand

BASE = ROOT.parent / 'ball_quality_20261109'
OLD = BASE / 'training_tile_detail_comparison'
OUT = BASE / 'native_tile_point_pilot'
SEED = 20261109
EPOCHS = 3


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'One fixed pilot; preserve the first settings and result.'
    labels_path = ROOT / 'balltrack_expanded_centers_pilot/expanded_train_labels.csv'
    labels = read(labels_path)
    phases = {r['sample_id']: r['phase_status'] for r in read(BASE / 'training_phase_scope_review/phase_labels.csv')}
    fit_rows = [r for r in labels if phases[r['sample_id']] == 'play' and r['truth_visibility'] == 'visible']
    assert len(fit_rows) == 68 and len({r['candidate_id'] for r in fit_rows}) == 8
    modelroot = SOURCE / 'third_party/RacketVision/source/BallTrack'
    weight = modelroot / 'checkpoints/balltrack_best.pth'
    old_protocol = load(OLD / 'protocol.json')
    paths = [labels_path, weight, modelroot / 'model/loss_utils.py', OLD / 'protocol.json', OLD / 'candidate_pools.json',
        OLD / 'input_image_hashes.json', BASE / 'training_phase_scope_review/phase_labels.csv',
        BASE / 'frozen_tile_gate_confirmation/training_predictions.csv', BASE / 'frozen_tile_gate_confirmation/frozen_training_head.npz',
        ROOT / 'ball_temporal_path_comparison/train_displacement_0.1.csv',
        BASE / 'temporal_guard_next_sequences/completed_review/manual_sequence_labels.csv',
        BASE / 'temporal_guard_next_sequences/evaluation_report.json',
        ROOT.parent / 'ball_quality_20261104/completed_review/manual_sequence_labels.csv',
        ROOT / 'reviewed_ball_overrides.json', ROOT / 'reviewed_person_overrides.json', ROOT / 'reviewed_tracking_bundles.json']
    for plan in old_protocol['plans']:
        paths.append(Path(plan['median_path']))
    protected = {str(p): sha(p) for p in paths}
    helpers = [ROOT / n for n in ['compare_training_tile_detail.py', 'compare_ball_candidate_decoders.py',
        'run_manual_ball_finetune.py', 'evaluate_training_phase_scope.py', 'calibrate_point_supervised_detector.py']]
    protocol = dict(status='fixed_before_any_new_training_or_model_outputs', date_jst='2026-11-09',
        role='Training-only feasibility pilot for candidate generation;no validation inference or operational selection.',
        motivation='Frozen candidate oracle shows no16pxcenter candidate for8of9training deficits and5of5olderconfirmation deficits. Temporal reranking cannot output a nonexistent same-frame center.',
        fit_frames=68, fit_clips=8, fit_samples=68 * 9,
        labels='Only previously human-confirmed play+visible training centers. Nine target-ball heatmaps per frame. Crops not containing the annotated active-ball center have zero TARGET-center heatmap,not a claim of physical/generic ball absence. Human-not-observable and nonplay frames excluded from fit.',
        crop='Same nine960x540regions ->512x288 as prior native arm;all9for every fit frame,never GT-selected crop. Same median+BGRt-3..t transform. No flips,random crops or new views.',
        target='Gaussian sigma3.5 networkpixels at int((human_x-tile_x)*512/960),int((human_y-tile_y)*288/540)when center lies in crop. Outside-center crop is zero target map. No predicted pseudo-centers.',
        initialization='Unchanged original TrackNet weights', trainable=['up_block_3', 'predictor'],
        frozen='All other weights and ALL BatchNorm running buffers;net.eval during gradient updates.',
        seed=SEED, epochs=EPOCHS, batch_size=2, learning_rate=1e-4, optimizer='Adam', weight_decay=0.,
        gradient_clip_norm=1., loss='Unchanged source WBCELoss,FP32reduction,AMPforward/GradScaler. Equal weight for all612crops;fixed seeded per-epoch permutation.',
        checkpoint='Only final epoch3 is the candidate. No epoch or threshold selection,hard-case weights,early stopping,or post-result retry.',
        control='Before training,replay ALL123targets/nine native tiles under exact saved batch4 groups;component dictionaries must exactly match immutable original native pools.',
        evaluation='After training,run final epoch on all123training targets using SAME saved native batches and threshold0.3. Report raw peak-selected and missing-only additions preserving original global temporal output. No image-head reuse/refit;head10was trained on other response distribution.',
        candidate_generation='Report GT-assisted union-center coverage as oracle upper bound separately from actual selected correctness;may not equate them.',
        development_gate='Missing-only final model must retain ALL59previous static-supplement correct PLAYtargets;correct count must exceed59,wrong_position<=0,output_when_not_observable<=1. Otherwise stop before any new52/26/validation/audit inference. Passing still requires separate frozen evaluation and permits no adoption.',
        limitations='Training scores only,one seed,68same-match points,unequal foreground-crop counts;cannot attribute to crop-vs-full-view training without a matched full-view training arm. This tests adaptation within fixed native inference,not a new general technique.',
        protected_input_hashes=protected, helper_hashes={str(p): sha(p) for p in helpers},
        script_sha256=sha(Path(__file__)), automatic_adoption=False)
    dump(OUT / 'protocol.json', protocol)
    import shutil
    shutil.copy2(Path(__file__), OUT / 'executed_source.py')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(modelroot))
    from inference import BallInferencer
    from model.loss_utils import WBCELoss
    engine = BallInferencer(str(modelroot / 'configs/tracknetv3_base.py'), str(weight), device='cuda', thre=.3, batchsize=4)
    net = engine.model
    net.eval()
    initial_state = {n: p.detach().cpu().clone() for n, p in net.state_dict().items()}
    medians, image_hashes = {}, {}
    for plan in old_protocol['plans']:
        median = np.load(plan['median_path'])['median']
        for tile, box in enumerate(TILES):
            medians[plan['candidate_id'] + '_' + str(tile)] = np.moveaxis(transform(median, box, 'native').astype(np.float32) / 255., -1, 0)
    original_pools = load(OLD / 'candidate_pools.json')['native']
    expected_images = load(OLD / 'input_image_hashes.json')

    def batches(plan):
        fd = Path(plan['frame_dir'])
        for ts in plan['batches']:
            images = {}
            for j in sorted({j for t in ts for j in range(t - 3, t + 1)}):
                p = fd / f'{j:06d}.jpg'
                digest = sha(p)
                assert digest == expected_images[str(p)]
                image_hashes[str(p)] = digest
                images[j] = cv2.imread(str(p))
                assert images[j].shape == (1080, 1920, 3)
            yield ts, images

    replay_count = 0
    for plan in old_protocol['plans']:
        cid = plan['candidate_id']
        target_by = {int(g['local_frame']): g for g in labels if g['candidate_id'] == cid}
        for ts, images in batches(plan):
            for tile, box in enumerate(TILES):
                batch = [dict(frames=np.stack([transform(images[j], box, 'native') for j in range(t - 3, t + 1)]), median_key=cid + '_' + str(tile), row={}) for t in ts]
                with torch.inference_mode(), torch.amp.autocast('cuda'):
                    output, _, _ = net.forward(frames=tensors(batch, medians, engine.device))
                for t, hm in zip(ts, output[:, 0].float().cpu().numpy(), strict=True):
                    if t not in target_by:
                        continue
                    sid = target_by[t]['sample_id']
                    expected = [c for c in original_pools[sid] if c['tile_index'] == tile]
                    assert mapped(components(hm), tile) == expected, (sid, tile, 'original_control_mismatch')
                    replay_count += 1
        print('Original native control matched:', cid, flush=True)
    assert replay_count == 123 * 9
    # Cache fixed crop images and targets from genuine centers only.
    samples, sample_manifest = [], []
    yy, xx = np.ogrid[:288, :512]
    for g in fit_rows:
        cid, t = g['candidate_id'], int(g['local_frame'])
        fd = Path(json.loads(g['input_frames_json'])[0]).parent
        images = [cv2.imread(str(fd / f'{j:06d}.jpg')) for j in range(t - 3, t + 1)]
        for tile, box in enumerate(TILES):
            ox, oy, w, h = box
            visible = ox <= float(g['truth_x_px']) < ox + w and oy <= float(g['truth_y_px']) < oy + h
            gx = int((float(g['truth_x_px']) - ox) * 512 / w) if visible else None
            gy = int((float(g['truth_y_px']) - oy) * 288 / h) if visible else None
            target = np.exp(-.5 * ((xx - gx) ** 2 + (yy - gy) ** 2) / 3.5 ** 2).astype(np.float32)[None] if visible else np.zeros((1, 288, 512), np.float32)
            samples.append(dict(frames=np.stack([transform(im, box, 'native') for im in images]), median_key=cid + '_' + str(tile), row={}, target=target))
            sample_manifest.append(dict(sample_index=len(samples) - 1, sample_id=g['sample_id'], candidate_id=cid,
                local_frame=t, tile_index=tile, target_center_in_crop=visible, gaussian_center_x=gx if visible else '', gaussian_center_y=gy if visible else '',
                source_center_x=g['truth_x_px'], source_center_y=g['truth_y_px'], phase='human_confirmed_play',
                target_sha256=__import__('hashlib').sha256(target.tobytes()).hexdigest()))
    assert len(samples) == 612
    write(OUT / 'training_crop_manifest.csv', sample_manifest)
    params, names = [], []
    for name, p in net.named_parameters():
        enabled = name.startswith('up_block_3.') or name.startswith('predictor.')
        p.requires_grad_(enabled)
        if enabled:
            params.append(p)
            names.append(name)
    dump(OUT / 'trainable_parameters.json', dict(names=names, count=sum(p.numel() for p in params)))
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=0.)
    scaler = torch.amp.GradScaler('cuda')
    generator = np.random.default_rng(SEED)
    history = []
    for epoch in range(1, EPOCHS + 1):
        net.eval()
        order = generator.permutation(len(samples))
        losses = []
        for start in range(0, len(order), 2):
            batch = [samples[i] for i in order[start:start + 2]]
            x = tensors(batch, medians, engine.device)
            target = torch.from_numpy(np.stack([s['target'] for s in batch])).to(engine.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                output, _, _ = net.forward(frames=x)
            loss = WBCELoss(output.float(), target)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            assert torch.isfinite(norm)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            if start % 100 == 0:
                print('Native point epoch', epoch, 'sample', start, '/', len(samples), 'loss', losses[-1], flush=True)
        history.append(dict(epoch=epoch, updates=len(losses), mean_loss=float(np.mean(losses))))
        dump(OUT / 'training_history.json', history)
    frozen = [n for n in initial_state if n not in names]
    assert all(torch.equal(net.state_dict()[n].detach().cpu(), initial_state[n]) for n in frozen)
    checkpoint = OUT / 'final_epoch03.pth'
    torch.save(dict(state_dict={n: t.detach().cpu() for n, t in net.state_dict().items()},
        meta=dict(epoch=3, protocol_sha256=sha(OUT / 'protocol.json'), source_weights_sha256=sha(weight))), checkpoint)
    # Only final epoch3,all123training targets,no validation labels used for choice.
    candidate_pools = {g['sample_id']: [] for g in labels}
    indices, vals, offsets, map_records = [], [], [0], []
    for plan in old_protocol['plans']:
        cid = plan['candidate_id']
        target_by = {int(g['local_frame']): g for g in labels if g['candidate_id'] == cid}
        for ts, images in batches(plan):
            for tile, box in enumerate(TILES):
                batch = [dict(frames=np.stack([transform(images[j], box, 'native') for j in range(t - 3, t + 1)]), median_key=cid + '_' + str(tile), row={}) for t in ts]
                with torch.inference_mode(), torch.amp.autocast('cuda'):
                    output, _, _ = net.forward(frames=tensors(batch, medians, engine.device))
                for t, hm in zip(ts, output[:, 0].float().cpu().numpy(), strict=True):
                    if t not in target_by:
                        continue
                    sid = target_by[t]['sample_id']
                    cs = components(hm)
                    candidate_pools[sid].extend(mapped(cs, tile))
                    ix, v = compact(hm)
                    assert components(expand(ix, v)) == cs
                    indices.append(ix)
                    vals.append(v)
                    offsets.append(offsets[-1] + len(ix))
                    map_records.append(dict(sample_id=sid, candidate_id=cid, local_frame=t, tile_index=tile, batch_frames_json=json.dumps(ts)))
        print('Final native model decoded:', cid, flush=True)
    dump(OUT / 'candidate_pools.json', candidate_pools)
    np.savez_compressed(OUT / 'component_rectangles.npz', indices=np.concatenate(indices), values=np.concatenate(vals), offsets=np.array(offsets, dtype=np.int64))
    write(OUT / 'map_records.csv', map_records)
    dump(OUT / 'input_image_hashes.json', image_hashes)
    original = {r['sample_id']: r for r in read(ROOT / 'ball_temporal_path_comparison/train_displacement_0.1.csv')}
    previous = {r['sample_id']: r for r in read(BASE / 'frozen_tile_gate_confirmation/training_predictions.csv')}
    raw, fallback, metrics_rows = [], [], []
    for g in labels:
        sid = g['sample_id']
        c = peak_choice(candidate_pools[sid])
        r = {k: g[k] for k in ['sample_id', 'candidate_id', 'local_frame', 'source_frame', 'truth_visibility', 'truth_x_px', 'truth_y_px']}
        r.update(visible=int(c is not None), x_px=c['x_px'] if c else 0, y_px=c['y_px'] if c else 0, confidence=c['bbox_mean'] if c else 0.)
        raw.append(dict(r))
        f = dict(r)
        if int(original[sid]['visible']):
            f.update({k: original[sid][k] for k in ['visible', 'x_px', 'y_px', 'confidence']})
        fallback.append(f)
    write(OUT / 'raw_peak_predictions.csv', raw)
    write(OUT / 'missing_only_predictions.csv', fallback)
    for name, predictions in [('original_temporal', list(original.values())), ('previous_static_head', list(previous.values())), ('final_native_raw_peak', raw), ('final_native_missing_only', fallback)]:
        for phase in ['play', 'nonplay']:
            counts = Counter(classify(r, r)[0] for r in predictions if phases[r['sample_id']] == phase)
            metrics_rows.append(dict(method=name, phase_status=phase, **{k: counts[k] for k in ['correct16px', 'wrong_position', 'missing_visible_ball', 'output_when_not_observable', 'no_output_when_not_observable']}))
    write(OUT / 'training_summary.csv', metrics_rows)
    coverage = []
    for pool_name, pool in [('original_native', original_pools), ('final_native', candidate_pools)]:
        hits = sum(any(math.hypot(c['x_px'] - float(g['truth_x_px']), c['y_px'] - float(g['truth_y_px'])) <= 16 for c in pool[g['sample_id']]) for g in fit_rows)
        coverage.append(dict(pool=pool_name, visible_play_targets=68, candidate_center_coverage16=hits, oracle_not_accuracy=True))
    previous_correct = {r['sample_id'] for r in previous.values() if phases[r['sample_id']] == 'play' and classify(r, r)[0] == 'correct16px'}
    final_correct = {r['sample_id'] for r in fallback if phases[r['sample_id']] == 'play' and classify(r, r)[0] == 'correct16px'}
    lost = sorted(previous_correct - final_correct)
    final = next(r for r in metrics_rows if r['method'] == 'final_native_missing_only' and r['phase_status'] == 'play')
    safe = not lost and final['correct16px'] > 59 and final['wrong_position'] <= 0 and final['output_when_not_observable'] <= 1
    assert all(sha(Path(p)) == h for p, h in protected.items())
    assert all(sha(Path(p)) == h for p, h in image_hashes.items())
    result = dict(status='fixed_native_point_training_pilot_complete', training_seed=SEED, updates=sum(h['updates'] for h in history),
        control_native_maps_exact=1107, final_native_maps=1107, training_source_frames=68, training_crop_samples=612,
        target_center_crops=sum(r['target_center_in_crop'] for r in sample_manifest),
        candidate_coverage=coverage, training_results=metrics_rows, lost_previous_correct_play_targets=lost,
        passes_training_gate=safe, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        frozen_weights_and_BN_unchanged=True, new_validation_inference=False, EF26_or_ABCD52_used_in_fit=False,
        operational_adoption=False, canonical_tracking_changed=False,
        decision='Candidate for separately frozen confirmation only.' if safe else 'Stop before validation;one fixed final model fails training gate. No threshold or epoch retry.',
        limitations='In-sample training diagnosis after one fixed final epoch,not crossvalidated or independent accuracy.')
    dump(OUT / 'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
