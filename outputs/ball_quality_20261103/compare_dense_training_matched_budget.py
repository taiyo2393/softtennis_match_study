"""One fixed clip-excluded dense-label trial with matched update/batch budgets."""
import json
import math
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT, SOURCE, read, load, write, sha, cv2
from compare_training_tile_detail import TILES, transform, mapped, peak_choice, prediction
from compare_ball_candidate_decoders import components
from calibrate_point_supervised_detector import compact, expand
from run_manual_ball_finetune import tensors
from evaluate_training_phase_scope import classify

BASE = ROOT.parent / 'ball_quality_20261109'
PREV = BASE / 'crossfit_native_candidate_ranker'
DATA = BASE / 'dense_training_sequences/completed_review'
OLD = BASE / 'training_tile_detail_comparison'
OUT = BASE / 'dense_training_matched_budget'
SEED = 20261109
OUTCOMES = ['correct16px', 'wrong_position', 'missing_visible_ball', 'output_when_not_observable', 'no_output_when_not_observable']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def batch_plan(sparse_crops):
    return [min(2, sparse_crops - j) for _ in range(3) for j in range(0, sparse_crops, 2)]


def dense_trace(samples, sizes):
    rng = np.random.default_rng(SEED)
    order, cursor, pass_index = rng.permutation(len(samples)), 0, 0
    trace = []
    for update, size in enumerate(sizes, 1):
        picked = []
        for _ in range(size):
            if cursor == len(order):
                order, cursor = rng.permutation(len(samples)), 0
                pass_index += 1
            picked.append(int(order[cursor])); cursor += 1
        trace.append(dict(update=update, batch_size=size, sample_indexes_json=json.dumps(picked), data_pass_index=pass_index))
    return trace


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Preserve this one fixed proposal; no post-result retries.'
    assert load(PREV / 'verification.json')['status'] == load(DATA / 'verification.json')['status'] == 'passed'
    assert load(DATA / 'export_consistency_check.json')['status'] == 'passed'
    rows = read(DATA / 'merged_training_observation_states.csv')
    visible = read(DATA / 'training_visible_centers.csv')
    sparse = read(ROOT / 'balltrack_expanded_centers_pilot/expanded_train_labels.csv')
    legacy_ids = {r['sample_id'] for r in sparse}
    dense_labels = read(DATA / 'manual_sequence_labels.csv')
    dense_ids = {r['sample_id'] for r in dense_labels}
    by_id = {r['sample_id']: r for r in rows}
    original = {r['sample_id']: r for r in read(ROOT / 'ball_temporal_path_comparison/train_displacement_0.1.csv')}
    parent_folds = load(PREV / 'folds.json')
    old_plans = load(OLD / 'protocol.json')['plans']
    expected_images = dict(load(OLD / 'input_image_hashes.json'))
    expected_images.update(load(DATA / 'source_image_hashes.json'))
    assert len(rows) == 243 and len(visible) == 181 and len(legacy_ids) == 123 and len(dense_ids) == 124
    assert len(legacy_ids & dense_ids) == 4
    groups = sorted({r['candidate_id'] for r in rows})
    assert len(groups) == 8 and all(r['finetune_role'] == 'train' for r in rows)
    modelroot = SOURCE / 'third_party/RacketVision/source/BallTrack'
    weight = modelroot / 'checkpoints/balltrack_best.pth'
    paths = [weight, DATA / 'merged_training_observation_states.csv', DATA / 'training_visible_centers.csv',
        DATA / 'manual_sequence_labels.csv', DATA / 'annotation_events.jsonl', DATA / 'result.json', DATA / 'verification.json', DATA / 'export_consistency_check.json',
        PREV / 'protocol.json', PREV / 'folds.json', PREV / 'verification.json', PREV / 'oof_predictions.csv',
        OLD / 'protocol.json', OLD / 'input_image_hashes.json', DATA / 'source_image_hashes.json',
        ROOT / 'ball_temporal_path_comparison/train_displacement_0.1.csv', ROOT / 'balltrack_expanded_centers_pilot/expanded_train_labels.csv',
        BASE / 'temporal_guard_next_sequences/completed_review/manual_sequence_labels.csv',
        ROOT.parent / 'ball_quality_20261104/completed_review/manual_sequence_labels.csv',
        ROOT / 'reviewed_ball_overrides.json', ROOT / 'reviewed_person_overrides.json', ROOT / 'reviewed_tracking_bundles.json']
    for fold in parent_folds:
        folder = PREV / ('fold_' + fold['held_clip'][-3:])
        paths += [folder / n for n in ['fold.json', 'training_history.json', 'detector_final03.pth', 'candidate_pools.json']]
    paths += [Path(r['median_path']) for r in visible]
    protected = {str(p): sha(p) for p in paths}
    helper_paths = [ROOT / n for n in ['compare_training_tile_detail.py', 'compare_ball_candidate_decoders.py', 'calibrate_point_supervised_detector.py', 'run_manual_ball_finetune.py', 'evaluate_training_phase_scope.py']]
    budgets = []
    plans = []
    for parent, old_plan in zip(parent_folds, old_plans, strict=True):
        cid = parent['held_clip']; assert cid == old_plan['candidate_id']
        sizes = batch_plan(parent['detector_crop_samples'])
        assert len(sizes) == parent['detector_updates']
        assert sum(sizes) == 3 * parent['detector_crop_samples']
        budgets.append(dict(held_clip=cid, updates=len(sizes), training_crop_presentations=sum(sizes), batch_sizes=sizes,
            sparse_fit_frames=parent['detector_fit_frames'], dense_fit_frames=sum(r['candidate_id'] != cid for r in visible)))
        old_targets = {int(r['local_frame']): r['sample_id'] for r in rows if r['candidate_id'] == cid and r['sample_id'] in legacy_ids}
        new_targets = {int(r['local_frame']): r['sample_id'] for r in rows if r['candidate_id'] == cid and r['sample_id'] not in legacy_ids}
        new_times = sorted(new_targets)
        plans.append(dict(candidate_id=cid, median_path=old_plan['median_path'], frame_dir=old_plan['frame_dir'],
            legacy_batches=old_plan['batches'], legacy_targets=old_targets,
            additional_batches=[new_times[i:i+4] for i in range(0, len(new_times), 4)], additional_targets=new_targets))
    protocol = dict(status='fixed_before_new_training_or_outputs', date_jst='2026-11-09',
        question='At the SAME per-fold optimizer-update and batch-size budget, does adding113human-play-visible centers improve held-clip ball localization?',
        design='8clip folds. Sparse68center pretrained-detector checkpoints are immutable paired controls. Dense181center detector starts at original weights for EVERY fold; excludes ALLheld-clip centers,including new dense ones.',
        compute_control='Exact update count and per-update batch sizes copied from corresponding sparse3epoch run. Dense corpus is traversed in seeded shuffled passes,stopping at the same presentation budget. Dense effective epochs therefore differ;no extra updates,early stopping or checkpoint selection.',
        initialization='Original TrackNet weights,train only up_block_3+predictor;net.eval freezesALLBNbuffers and all other parameters.',
        optimizer='Same Adam1e-4,weight_decay0,FP32sourceWBCELoss,CUDA AMP/GradScaler,batch schedule from sparse arm,clip_grad_norm1,seed20261109. Only final matched-budget checkpoint.',
        supervision='181human visible-play centers only,all9native960x540->512x288crops per frame,sigma3.5networkpx. Outside annotated target-center crops zero TARGETmap,not physical ball absence. Unobservable and nonplay never provide targets.',
        no_new_temporal_loss='The detector already inputs median+BGRt-3..t. Adding densely sampled labels does not itself fit a trajectory model or smoothness loss.',
        selection='Allcomponentsstrictly>0.3,highestpeak thenarea thenlower tile/component. No classifier,pixel-head,threshold change,interpolation,GT-rankedproposal or new selection rule.',
        primary='Legacy123targets in original batch groups;primary play77(68visible9unobservable). Supplement only original temporal missing positions;retain every original existing output. Compare saved sparse-fold raw missing-only control vs dense-fold raw missing-only.',
        secondary='New124densepacket targets(117visible7unobservable) in held clips:raw native-peak sparse vsdense,without global fallback.4anchors retain legacy batch groups;120newtargets fixed sorted batches4. Per-window counts and observable adjacent-pair outcomes;no frame-independence inference.',
        replay='Sparse checkpoint must exactly reproduce ALLlegacy123held candidate pools and missing-only predictions before pairing;both arms use identical per-target batches.',
        safety='LegacyPLAYdense must retain ALL64correct sparse-control targets,wrong_position0,output_on_unobservable<=1. Densepacket raw arm must lose no sparse-correct target and not increase wrong positions/unobservable outputs. Failure stops before full-data fits,new confirmation inference,head/threshold changes or adoption.',
        scope='Internal development in8known same-match training clips. New centers cluster in4one-second windows and change sampling distribution. Not181independent rallies,not a pure causal test of temporal supervision,not untouched external validation.',
        preprocessing='Full-clip median is fixed unsupervised offline input,not causal real-time preprocessing. Legacy edited clock only.',
        no_confirmation_labels_fitted=True, no_automatic_adoption=True, budgets=budgets, inference_plans=plans,
        protected_input_hashes=protected, helper_hashes={str(p): sha(p) for p in helper_paths}, script_sha256=sha(Path(__file__)))
    dump(OUT / 'protocol.json', protocol)
    shutil.copy2(Path(__file__), OUT / 'executed_source.py')
    sys.path.insert(0, str(modelroot))
    from inference import BallInferencer
    from model.loss_utils import WBCELoss
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    engine = BallInferencer(str(modelroot / 'configs/tracknetv3_base.py'), str(weight), device='cuda', thre=.3, batchsize=4)
    net = engine.model
    initial = {n: p.detach().cpu().clone() for n, p in net.state_dict().items()}
    params, train_names = [], []
    for name, p in net.named_parameters():
        trainable = name.startswith('up_block_3.') or name.startswith('predictor.')
        p.requires_grad_(trainable)
        if trainable: params.append(p); train_names.append(name)
    frozen = set(initial) - set(train_names)
    medians, cache, image_hashes = {}, {}, {}
    yy, xx = np.ogrid[:288, :512]
    for plan in plans:
        median = np.load(plan['median_path'])['median']
        for tile, box in enumerate(TILES):
            medians[plan['candidate_id'] + '_' + str(tile)] = np.moveaxis(transform(median, box, 'native').astype(np.float32)/255., -1, 0)
    for g in visible:
        images = []
        for p, h in zip(json.loads(g['input_frames_json']), json.loads(g['input_frame_sha256_json']), strict=True):
            assert sha(Path(p)) == h; image_hashes[p] = h
            images.append(cv2.imread(p))
        samples = []
        for tile, box in enumerate(TILES):
            ox, oy, w, h = box
            contains = ox <= float(g['truth_x_px']) < ox+w and oy <= float(g['truth_y_px']) < oy+h
            if contains:
                gx, gy = int((float(g['truth_x_px'])-ox)*512/w), int((float(g['truth_y_px'])-oy)*288/h)
                target = np.exp(-.5*((xx-gx)**2+(yy-gy)**2)/3.5**2).astype(np.float32)[None]
            else: target = np.zeros((1, 288, 512), np.float32)
            samples.append(dict(frames=np.stack([transform(im, box, 'native') for im in images]),
                median_key=g['candidate_id']+'_'+str(tile), target=target, sample_id=g['sample_id'], tile_index=tile, row={}))
        cache[g['sample_id']] = samples
    records, indices, values, offsets = [], [], [], [0]
    all_pools = {'sparse': {}, 'dense': {}}
    folds, all_predictions = [], []
    previous_predictions = {r['sample_id']: r for r in read(PREV / 'oof_predictions.csv') if r['method'] == 'fold_native_peak'}
    def infer_held(plan, arm):
        pools = {}
        for batch_kind in ['legacy', 'additional']:
            wanted = plan[batch_kind+'_targets']
            for times in plan[batch_kind+'_batches']:
                images = {}
                for t in sorted({j for t in times for j in range(t-3, t+1)}):
                    path = str(Path(plan['frame_dir']) / f'{t:06d}.jpg')
                    assert sha(Path(path)) == expected_images[path]
                    image_hashes[path] = expected_images[path]
                    images[t] = cv2.imread(path)
                for tile, box in enumerate(TILES):
                    batch = [dict(frames=np.stack([transform(images[j], box, 'native') for j in range(t-3, t+1)]), median_key=plan['candidate_id']+'_'+str(tile), row={}) for t in times]
                    with torch.inference_mode(), torch.amp.autocast('cuda'):
                        output, _, _ = net.forward(frames=tensors(batch, medians, engine.device))
                    for t, hm in zip(times, output[:, 0].float().cpu().numpy(), strict=True):
                        if t not in wanted: continue
                        sid = wanted[t]
                        cs = components(hm); pools.setdefault(sid, []).extend(mapped(cs, tile))
                        idx, val = compact(hm); assert components(expand(idx, val)) == cs
                        indices.append(idx); values.append(val); offsets.append(offsets[-1]+len(idx))
                        records.append(dict(map_index=len(records), sample_id=sid, held_clip=plan['candidate_id'], arm=arm,
                            tile_index=tile, batch_kind=batch_kind, batch_frames_json=json.dumps(times), component_count=len(cs)))
        return pools
    started = time.monotonic()
    for parent, plan, budget in zip(parent_folds, plans, budgets, strict=True):
        cid = parent['held_clip']; folder = OUT / ('fold_'+cid[-3:]); folder.mkdir()
        held_rows = [r for r in rows if r['candidate_id'] == cid]
        fit_rows = [r for r in visible if r['candidate_id'] != cid]
        fit_ids, held_ids = {r['sample_id'] for r in fit_rows}, {r['sample_id'] for r in held_rows}
        fit_hashes = {h for g in fit_rows for h in json.loads(g['input_frame_sha256_json'])}
        held_hashes = {h for g in held_rows for h in json.loads(g['input_frame_sha256_json'])}
        assert not fit_ids & held_ids and not fit_hashes & held_hashes
        control_folder = PREV / ('fold_'+cid[-3:])
        control = torch.load(control_folder / 'detector_final03.pth', map_location='cpu', weights_only=False)
        assert control['meta']['held_clip'] == cid and set(control['meta']['detector_fit_sample_ids']) == set(parent['detector_fit_sample_ids'])
        assert all(not sid.startswith(cid+'_frame_') for sid in control['meta']['detector_fit_sample_ids'])
        net.load_state_dict(control['state_dict'], strict=True); net.eval()
        sparse_pools = infer_held(plan, 'sparse')
        expected = load(control_folder / 'candidate_pools.json')
        assert all(sparse_pools[sid] == expected[sid] for sid in held_ids & legacy_ids), (cid, 'Control replay mismatch')
        all_pools['sparse'].update(sparse_pools)
        print('Fold', cid[-3:], 'sparse replay complete; updates', budget['updates'], 'dense fit frames', len(fit_rows), flush=True)
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        net.load_state_dict(initial, strict=True); net.eval()
        samples = [s for r in fit_rows for s in cache[r['sample_id']]]
        sample_index = [dict(sample_index=i, sample_id=s['sample_id'], tile_index=s['tile_index']) for i, s in enumerate(samples)]
        trace = dense_trace(samples, budget['batch_sizes'])
        optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=0.)
        scaler = torch.amp.GradScaler('cuda')
        loss_rows = []
        for entry in trace:
            batch = [samples[i] for i in json.loads(entry['sample_indexes_json'])]
            x = tensors(batch, medians, engine.device)
            y = torch.from_numpy(np.stack([s['target'] for s in batch])).to(engine.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                output, _, _ = net.forward(frames=x)
            loss = WBCELoss(output.float(), y); assert torch.isfinite(loss)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.); assert torch.isfinite(norm)
            scaler.step(optimizer); scaler.update()
            loss_rows.append(dict(update=entry['update'], loss=float(loss.detach()), gradient_norm=float(norm), grad_scale=scaler.get_scale()))
            if entry['update'] % 100 == 0:
                print('Fold', cid[-3:], 'dense update', entry['update'], '/', len(trace), 'elapsed', round(time.monotonic()-started,1), flush=True)
        assert all(int(optimizer.state[p]['step']) == budget['updates'] for p in params)
        assert all(torch.equal(net.state_dict()[n].detach().cpu(), initial[n]) for n in frozen)
        checkpoint = folder / 'detector_final_matched.pth'
        torch.save(dict(state_dict={n: p.detach().cpu() for n, p in net.state_dict().items()},
            meta=dict(held_clip=cid, detector_fit_sample_ids=sorted(fit_ids), update_count=len(trace),
                batch_sizes=budget['batch_sizes'], protocol_sha256=sha(OUT / 'protocol.json'))), checkpoint)
        dense_pools = infer_held(plan, 'dense'); all_pools['dense'].update(dense_pools)
        write(folder / 'training_sample_index.csv', sample_index); write(folder / 'training_trace.csv', trace); write(folder / 'training_losses.csv', loss_rows)
        for arm, pools in [('sparse', sparse_pools), ('dense', dense_pools)]:
            for g in held_rows:
                sid = g['sample_id']; chosen = peak_choice(pools[sid])
                p = prediction(g, chosen, arm+'_raw_native_peak')
                p.update(phase_status=g['phase_status'], held_clip=cid, evaluation_set='legacy_and_dense_anchor' if sid in legacy_ids & dense_ids else 'legacy' if sid in legacy_ids else 'new_dense', output_source='actual_native_candidate' if chosen else 'missing')
                all_predictions.append(p)
                if sid in legacy_ids:
                    q = dict(p, method=arm+'_legacy_missing_only')
                    if int(original[sid]['visible']):
                        q.update({k: original[sid][k] for k in ['visible', 'x_px', 'y_px', 'confidence']})
                        q['output_source'] = 'unchanged_original_temporal'
                    if arm == 'sparse':
                        assert all(float(q[k]) == float(previous_predictions[sid][k]) for k in ['visible', 'x_px', 'y_px', 'confidence'])
                    all_predictions.append(q)
        presentations = Counter(samples[i]['sample_id'] for e in trace for i in json.loads(e['sample_indexes_json']))
        fold = dict(held_clip=cid, dense_fit_sample_ids=sorted(fit_ids), sparse_fit_sample_ids=parent['detector_fit_sample_ids'],
            held_sample_ids=sorted(held_ids), dense_fit_frames=len(fit_rows), dense_crop_corpus=len(samples),
            update_count=len(trace), batch_sizes=budget['batch_sizes'], presentations=sum(presentations.values()),
            dense_effective_corpus_passes=sum(presentations.values())/len(samples), distinct_presented_center_frames=len(presentations),
            sparse_control_sha256=sha(control_folder / 'detector_final03.pth'), dense_checkpoint_sha256=sha(checkpoint),
            input_hash_overlap=0, frozen_weights_and_bn_unchanged=True, sparse_legacy_targets_exactly_replayed=len(held_ids & legacy_ids))
        dump(folder / 'fold.json', fold); folds.append(fold)
        dump(OUT / 'folds.json', folds); write(OUT / 'held_predictions.csv', all_predictions)
        dump(OUT / 'progress.json', dict(completed_folds=len(folds), elapsed_sec=time.monotonic()-started))
        print('Fold', cid[-3:], 'complete', flush=True)
    assert len(records) == 243*2*9 and len(all_predictions) == 243*2+123*2
    dump(OUT / 'candidate_pools.json', all_pools)
    write(OUT / 'map_records.csv', records)
    np.savez_compressed(OUT / 'component_rectangles.npz', indices=np.concatenate(indices), values=np.concatenate(values), offsets=np.array(offsets,dtype=np.int64))
    dump(OUT / 'input_image_hashes.json', image_hashes)
    summaries, outcomes, per_clip, changes = [], {}, [], []
    configs = [('legacy_play','sparse_legacy_missing_only', legacy_ids, 'play'), ('legacy_play','dense_legacy_missing_only', legacy_ids, 'play'),
        ('dense_packet_play','sparse_raw_native_peak', dense_ids, 'play'), ('dense_packet_play','dense_raw_native_peak', dense_ids, 'play'),
        ('legacy_nonplay','sparse_legacy_missing_only', legacy_ids, 'nonplay'), ('legacy_nonplay','dense_legacy_missing_only', legacy_ids, 'nonplay')]
    for scope, method, ids, phase in configs:
        predictions = {r['sample_id']: r for r in all_predictions if r['method'] == method and r['sample_id'] in ids and r['phase_status'] == phase}
        counts = Counter()
        for sid, p in predictions.items():
            outcome, error = classify(by_id[sid], p); counts[outcome] += 1; outcomes[scope, method, sid] = outcome
        summaries.append(dict(scope=scope, method=method, targets=len(predictions), **{k: counts[k] for k in OUTCOMES}))
        for cid in groups:
            cc = Counter(outcomes[scope, method, sid] for sid in predictions if by_id[sid]['candidate_id'] == cid)
            if cc: per_clip.append(dict(scope=scope, method=method, candidate_id=cid, **{k:cc[k] for k in OUTCOMES}))
    for scope, methods, ids in [('legacy_play',['sparse_legacy_missing_only','dense_legacy_missing_only'],legacy_ids),('dense_packet_play',['sparse_raw_native_peak','dense_raw_native_peak'],dense_ids)]:
        for sid in sorted(ids):
            if by_id[sid]['phase_status'] != 'play': continue
            a,b = [outcomes[scope,m,sid] for m in methods]
            changes.append(dict(scope=scope,sample_id=sid,candidate_id=by_id[sid]['candidate_id'],sparse_outcome=a,dense_outcome=b,correct_lost=int(a=='correct16px' and b!='correct16px'),correct_gained=int(a!='correct16px' and b=='correct16px')))
    write(OUT/'summary.csv',summaries); write(OUT/'per_clip_summary.csv',per_clip); write(OUT/'paired_outcomes.csv',changes)
    def summary(scope,method): return next(r for r in summaries if r['scope']==scope and r['method']==method)
    primary = summary('legacy_play','dense_legacy_missing_only')
    secondary_a, secondary_b = [summary('dense_packet_play',m) for m in ['sparse_raw_native_peak','dense_raw_native_peak']]
    legacy_lost = [r['sample_id'] for r in changes if r['scope']=='legacy_play' and r['correct_lost']]
    dense_lost = [r['sample_id'] for r in changes if r['scope']=='dense_packet_play' and r['correct_lost']]
    safe = not legacy_lost and primary['correct16px']>=64 and primary['wrong_position']==0 and primary['output_when_not_observable']<=1 and not dense_lost and secondary_b['wrong_position']<=secondary_a['wrong_position'] and secondary_b['output_when_not_observable']<=secondary_a['output_when_not_observable']
    assert all(sha(Path(p)) == h for p,h in protected.items())
    assert all(sha(Path(p)) == h for p,h in image_hashes.items())
    result = dict(status='matched_budget_dense_clip_excluded_comparison_complete', folds=8, training_updates=sum(f['update_count'] for f in folds),
        sparse_training_centers=68,dense_training_centers=181,held_predictions=len(all_predictions),stored_maps=len(records),
        summary=summaries,legacy_correct_lost=legacy_lost,dense_packet_correct_lost=dense_lost,passes_internal_gate=bool(safe),
        exact_sparse_legacy_replay_targets=123,compute_budget_matched=True,held_clip_fit_exclusion=True,all_bn_and_other_weights_fixed=True,
        new_head_fit=False,threshold_retry=False,full_data_training=False,new_confirmation_inference=False,operational_adoption=False,canonical_tracking_changed=False,
        elapsed_sec=time.monotonic()-started,decision='Only a separate full-data training check is next.' if safe else 'Reject this fixed dense-data proposal at the current budget;no parameter retry,head/full fit or confirmation inference.',
        limitation='Known same-match internal comparison;temporal labels densely sampled in4shortwindows,not independent trial frames or external validation.')
    dump(OUT/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__': main()
