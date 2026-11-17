"""Freeze one training-only head, then perform a fixed same-match confirmation.

Separate stages prevent using confirmation predictions to alter the head.
Original reports, labels, weights and operational tracking remain read-only.
"""
import argparse
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT, SOURCE, read, load, write, sha, cv2
from compare_grouped_tile_gate import F, features, apply_gate
from compare_frozen_ball_backbones import fit, predict
from compare_training_tile_detail import TILES, transform, mapped, peak_choice
from compare_ball_candidate_decoders import components
from run_manual_ball_finetune import tensors
from evaluate_training_phase_scope import classify as training_classify
from evaluate_reserved_ball_sequences import classify
from calibrate_point_supervised_detector import compact, expand

BASE = ROOT.parent / 'ball_quality_20261109'
OUT = BASE / 'frozen_tile_gate_confirmation'
TRAIN = BASE / 'training_tile_gate_grouped'
TILE = BASE / 'training_tile_detail_comparison'
PHASE = BASE / 'training_phase_scope_review'
CHECK = ROOT.parent / 'ball_quality_20261104'


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def protected_ok():
    protocol = load(OUT / 'protocol.json')
    assert all(sha(Path(p)) == h for p, h in protocol['protected_input_hashes'].items())
    assert sha(Path(__file__)) == protocol['script_sha256']
    assert all(sha(Path(p)) == h for p, h in protocol['helper_hashes'].items())
    return protocol


def phase_counts(rows, phases):
    return Counter(training_classify(r, r)[0] for r in rows if phases[r['sample_id']]['phase_status'] == 'play')


def train():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Preserve completed settings and model; no retries with different conditions.'
    old_report = load(PHASE / 'evaluation_report.json')
    assert old_report['phase_counts'] == {'play': 77, 'nonplay': 46}
    source_files = [TRAIN / 'candidate_features_and_training_labels.csv', TRAIN / 'protocol.json',
        TRAIN / 'folds.json', TRAIN / 'result.json', TILE / 'candidate_pools.json', TILE / 'native_peak_predictions.csv',
        ROOT / 'balltrack_expanded_centers_pilot/expanded_train_labels.csv',
        ROOT / 'ball_temporal_path_comparison/train_displacement_0.1.csv',
        PHASE / 'phase_labels.csv', PHASE / 'annotation_events.jsonl', PHASE / 'evaluation_report.json',
        PHASE / 'completed_verification.json',
        CHECK / 'completed_review/manual_sequence_labels.csv', CHECK / 'evaluation_report.json',
        CHECK / 'frozen_predictions.csv', CHECK / 'frozen_component_cache.json', CHECK / 'inference_batches.csv',
        CHECK / 'query_model_inputs.csv', CHECK / 'reserved_windows.json', CHECK / 'source_image_hashes.json',
        ROOT / 'reviewed_ball_overrides.json', ROOT / 'reviewed_person_overrides.json', ROOT / 'reviewed_tracking_bundles.json',
        SOURCE / 'annotations/balltrack_groundtruth_match1_pilot.csv',
        SOURCE / 'third_party/RacketVision/source/BallTrack/checkpoints/balltrack_best.pth']
    helpers = [ROOT / n for n in ['compare_grouped_tile_gate.py', 'compare_frozen_ball_backbones.py',
        'compare_training_tile_detail.py', 'compare_ball_candidate_decoders.py', 'run_manual_ball_finetune.py',
        'evaluate_training_phase_scope.py', 'evaluate_reserved_ball_sequences.py', 'calibrate_point_supervised_detector.py']]
    for w in load(CHECK / 'reserved_windows.json'):
        source_files.append(Path(w['median_path']))
    protocol = dict(status='fixed_before_full_fit_and_new_confirmation_inference', date_jst='2026-11-09',
        candidate='One logistic10 head with unchanged original TrackNet and native/degraded9tile features',
        training='Exactly the original81 manually localized selected candidates:76within16 positives,5outside16 negatives,from8trainingclips. Retain mixed-phase training examples; phase labels are used only for safety reporting. No52case label in fit.',
        head='Same training-only normalization,class-balanced BCE +0.5L2(weights),unpenalized bias,CPUfloat64,LBFGS300,strong_wolfe,zero init,threshold0.5. No search.',
        feature_names=F, training_scope_reason='Preserves the data and algorithm used in the preceding grouped comparison; no phase-filtered refit or new feature selection.',
        full_fit_safety='On human play targets,preserve every original output; strictly improve correct count; wrong_position and output_when_not_observable must EACH be no greater than original. Failure stops before confirmation; no alternate model.',
        confirmation='All52 preselected times in4previously labelled same-match expanded clips. Previously inspected baseline failures informed diagnosis and scale hypothesis; this is development confirmation,NOT untouched or external validation.',
        inference='Original median+BGRt-3..t,512x288,/255,CUDA AMP,cudnn.benchmark=False. Reproduce all84original globalcontext component dictionaries with exact saved batch4; tile batches use those same saved time groups. All9tiles,strict heatmap>0.3,existing deterministic peak selection.',
        application='Only supplement missing original temporal outputs with actual selected native centers accepted at0.5. Preserve original output,coordinates,confidence on all52 targets. No interpolation.',
        batch_note='Inference batch groups are part of this reproducible experiment,not an assertion of unrestricted runtime equivalence.',
        expected_reference='52 times:45visible,3uncertain centers,4notobservable; frozen temporal33correct,12missing,0wrong,0output-on-notobservable. Uncertain centers excluded identically; predictions still retained for all52.',
        confirmation_safety='Preserve all original correct times; require >=1additional correct16px; wrong_position<=baseline AND output_when_not_observable<=baseline,each separately. Report all4windows and sampled-pair displacement/errors and sampled correct runs. No threshold changes,alternate selection or fit after results.',
        decision='Pass means eligible for further independent-match/continuous-track evaluation,NOT operational adoption. Failure rejects this frozen candidate for use; no repair tuned to these52.',
        phase_scope='Human main-rally scope; no automatic start/end performance claim.',
        offline='Temporal baseline uses future8frames and full-clip median. Not causal or real-time.',
        automatic_adoption=False, detector_training=False, new_annotations=False,
        protected_input_hashes={str(p): sha(p) for p in source_files},
        helper_hashes={str(p): sha(p) for p in helpers}, script_sha256=sha(Path(__file__)))
    dump(OUT / 'protocol.json', protocol)
    shutil.copy2(Path(__file__), OUT / 'executed_source.py')
    protected_ok()
    es = read(TRAIN / 'candidate_features_and_training_labels.csv')
    data = [e for e in es if e['fit_label'] != '']
    assert len(data) == 81 and Counter(e['fit_label'] for e in data) == Counter({'1': 76, '0': 5})
    x = np.array([[float(e[f]) for f in F] for e in data], dtype=np.float64)
    y = np.array([float(e['fit_label']) for e in data], dtype=np.float64)
    labels = read(ROOT / 'balltrack_expanded_centers_pilot/expanded_train_labels.csv')
    train_ids = {e['sample_id'] for e in data}
    query = read(CHECK / 'query_model_inputs.csv')  # Has no human ball centers.
    assert not train_ids & {q['sample_id'] for q in query}
    train_hashes = {h for g in labels for h in json.loads(g['input_frame_sha256_json'])}
    check_hashes = {h for q in query for h in json.loads(q['input_frame_sha256_json'])}
    assert not train_hashes & check_hashes
    # Verify source-time non-overlap of full clips,including the full-clip medians.
    windows = load(CHECK / 'reserved_windows.json')
    train_bounds = []
    for cid in sorted({r['candidate_id'] for r in labels}):
        group = [r for r in labels if r['candidate_id'] == cid]
        offset = int(group[0]['source_frame']) - int(group[0]['local_frame'])
        frame_dir = Path(json.loads(group[0]['input_frames_json'])[0]).parent
        frames = [int(p.stem) for p in frame_dir.glob('*.jpg')]
        lo, hi = offset + min(frames), offset + max(frames)
        for w in windows:
            assert hi + 120 < w['first_source_frame'] or w['last_source_frame'] + 120 < lo
        train_bounds.append(dict(candidate_id=cid, first_source_frame=lo, last_source_frame=hi))
    dump(OUT / 'fit_manifest.json', dict(feature_names=F, fit_sample_ids=[e['sample_id'] for e in data],
        training_positive=76, training_negative=5, input_hash_overlap=0, full_clip_120frame_guard_passed=True,
        train_bounds=train_bounds, confirmation_clip_ids=[w['candidate_id'] for w in windows]))
    torch.set_num_threads(4)
    model = fit(x, y)
    np.savez_compressed(OUT / 'frozen_training_head.npz', **model)
    all_x = np.array([[float(e[f]) for f in F] for e in es], dtype=np.float64)
    scores = dict(zip([e['sample_id'] for e in es], predict(model, all_x).tolist(), strict=True))
    write(OUT / 'training_candidate_scores.csv', [dict(sample_id=sid, proposal_gate_score=s, accepted=int(s >= .5)) for sid, s in scores.items()])
    old = {r['sample_id']: r for r in read(ROOT / 'ball_temporal_path_comparison/train_displacement_0.1.csv')}
    native = {r['sample_id']: r for r in read(TILE / 'native_peak_predictions.csv')}
    result = apply_gate(labels, old, native, scores, 'logistic10_frozen_full_fit')
    write(OUT / 'training_predictions.csv', result)
    phases = {r['sample_id']: r for r in read(PHASE / 'phase_labels.csv')}
    baseline, candidate = phase_counts(list(old.values()), phases), phase_counts(result, phases)
    preserved = all(all(str(r[k]) == str(old[r['sample_id']][k]) for k in ['visible', 'x_px', 'y_px', 'confidence']) for r in result if int(old[r['sample_id']]['visible']))
    safe = preserved and candidate['correct16px'] > baseline['correct16px'] and all(candidate[k] <= baseline[k] for k in ['wrong_position', 'output_when_not_observable'])
    protected_ok()
    record = dict(status='frozen_full_fit_passed_training_scope' if safe else 'stopped_full_fit_failed_training_scope',
        passed=safe, baseline=dict(baseline), candidate=dict(candidate), original_outputs_preserved=preserved,
        model_sha256=sha(OUT / 'frozen_training_head.npz'), no_confirmation_labels_used_in_fit=True,
        threshold=.5, automatic_adoption=False)
    dump(OUT / 'full_fit_decision.json', record)
    print(json.dumps(record, indent=2))


def infer():
    protected_ok()
    decision = load(OUT / 'full_fit_decision.json')
    assert decision['passed'], 'Training safety failed; confirmation must not run.'
    model_path = OUT / 'frozen_training_head.npz'
    assert sha(model_path) == decision['model_sha256']
    assert not (OUT / 'inference_manifest.json').exists(), 'Keep the first frozen confirmation predictions.'
    model = dict(np.load(model_path))
    windows = load(CHECK / 'reserved_windows.json')
    queries = read(CHECK / 'query_model_inputs.csv')
    assert len(queries) == 52
    old = {r['sample_id']: r for r in read(CHECK / 'frozen_predictions.csv') if r['method'] == 'displacement_0.1'}
    context = load(CHECK / 'frozen_component_cache.json')
    batches = read(CHECK / 'inference_batches.csv')
    hashes = load(CHECK / 'source_image_hashes.json')
    modelroot = SOURCE / 'third_party/RacketVision/source/BallTrack'
    sys.path.insert(0, str(modelroot))
    from inference import BallInferencer
    torch.backends.cudnn.benchmark = False
    engine = BallInferencer(str(modelroot / 'configs/tracknetv3_base.py'), str(modelroot / 'checkpoints/balltrack_best.pth'), device='cuda', thre=.3, batchsize=4)
    engine.model.eval().requires_grad_(False)
    pools = {arm: {q['sample_id']: [] for q in queries} for arm in ['native', 'degraded']}
    image_hashes, replay, map_records = {}, [], []
    indices, values, offsets = [], [], [0]
    for window in windows:
        cid = window['candidate_id']
        fd = Path(window['frame_dir'])
        median = np.load(window['median_path'])['median']
        medians = {'global': np.moveaxis(cv2.resize(median, (512, 288)).astype(np.float32) / 255., -1, 0)}
        for arm in pools:
            for i, box in enumerate(TILES):
                medians[arm + str(i)] = np.moveaxis(transform(median, box, arm).astype(np.float32) / 255., -1, 0)
        targets = {int(q['local_frame']): q['sample_id'] for q in queries if q['candidate_id'] == cid}
        keys = sorted(map(int, context[cid]))
        for start in range(0, len(keys), 4):
            times = keys[start:start + 4]
            assert all(json.loads(next(b for b in batches if b['candidate_id'] == cid and int(b['local_frame']) == t)['batch_frames_json']) == times for t in times)
            ims = {}
            for j in sorted({j for t in times for j in range(t - 3, t + 1)}):
                p = fd / f'{j:06d}.jpg'
                digest = sha(p)
                assert digest == hashes[str(p)]
                image_hashes[str(p)] = digest
                ims[j] = cv2.imread(str(p))
                assert ims[j].shape == (1080, 1920, 3)
            batch = [dict(frames=np.stack([cv2.resize(ims[j], (512, 288)) for j in range(t - 3, t + 1)]), median_key='global', row={}) for t in times]
            with torch.inference_mode(), torch.amp.autocast('cuda'):
                outputs, _, _ = engine.model.forward(frames=tensors(batch, medians, engine.device))
            for t, hm in zip(times, outputs[:, 0].float().cpu().numpy(), strict=True):
                assert components(hm) == context[cid][str(t)]
                replay.append(dict(candidate_id=cid, local_frame=t, batch_frames_json=json.dumps(times), exact_component_match=True))
            if not any(t in targets for t in times):
                continue
            for arm in pools:
                for i, box in enumerate(TILES):
                    batch = [dict(frames=np.stack([transform(ims[j], box, arm) for j in range(t - 3, t + 1)]), median_key=arm + str(i), row={}) for t in times]
                    with torch.inference_mode(), torch.amp.autocast('cuda'):
                        outputs, _, _ = engine.model.forward(frames=tensors(batch, medians, engine.device))
                    for t, hm in zip(times, outputs[:, 0].float().cpu().numpy(), strict=True):
                        if t not in targets:
                            continue
                        sid = targets[t]
                        cs = components(hm)
                        pools[arm][sid].extend(mapped(cs, i))
                        ix, val = compact(hm)
                        assert components(expand(ix, val)) == cs
                        indices.append(ix)
                        values.append(val)
                        offsets.append(offsets[-1] + len(ix))
                        map_records.append(dict(sample_id=sid, candidate_id=cid, local_frame=t, arm=arm, tile_index=i, batch_frames_json=json.dumps(times)))
            print('Completed target batch', cid, times, flush=True)
    assert len(replay) == 84 and len(map_records) == 52 * 18
    np.savez_compressed(OUT / 'component_rectangles.npz', indices=np.concatenate(indices), values=np.concatenate(values), offsets=np.array(offsets, dtype=np.int64))
    write(OUT / 'map_records.csv', map_records)
    write(OUT / 'global_replay.csv', replay)
    dump(OUT / 'candidate_pools.json', pools)
    dump(OUT / 'input_image_hashes.json', image_hashes)
    predictions, feature_rows = [], []
    for q in queries:
        sid = q['sample_id']
        c = peak_choice(pools['native'][sid])
        if c:
            x = features(c, pools['native'][sid], pools['degraded'][sid], TILES)
            score = float(predict(model, x[None])[0])
            feature_rows.append(dict(sample_id=sid, **dict(zip(F, x.tolist())), proposal_gate_score=score))
        else:
            score = 0.
        base = old[sid]
        if int(base['visible']):
            actual = {k: base[k] for k in ['visible', 'x_px', 'y_px', 'confidence']}
            origin = 'unchanged_original_temporal'
        elif c and score >= .5:
            actual = dict(visible=1, x_px=c['x_px'], y_px=c['y_px'], confidence=c['bbox_mean'])
            origin = 'accepted_actual_native_tile'
        else:
            actual = dict(visible=0, x_px='', y_px='', confidence=0.)
            origin = 'missing'
        predictions.append(dict(**{k: q[k] for k in ['sample_id', 'case_id', 'window_id', 'candidate_id', 'local_frame', 'source_frame']},
            method='logistic10_frozen', **actual, output_source=origin, proposal_gate_score=score,
            proposal_x_px=c['x_px'] if c else '', proposal_y_px=c['y_px'] if c else '',
            proposal_tile_index=c['tile_index'] if c else -1, proposal_component_index=c['component_index'] if c else -1))
    write(OUT / 'frozen_confirmation_predictions.csv', predictions)
    if feature_rows:
        write(OUT / 'confirmation_candidate_features.csv', feature_rows)
    protected_ok()
    assert sha(model_path) == decision['model_sha256']
    artifacts = ['frozen_confirmation_predictions.csv', 'candidate_pools.json', 'map_records.csv', 'component_rectangles.npz',
        'global_replay.csv', 'input_image_hashes.json', 'full_fit_decision.json', 'frozen_training_head.npz', 'fit_manifest.json', 'protocol.json']
    if feature_rows:
        artifacts.append('confirmation_candidate_features.csv')
    dump(OUT / 'inference_manifest.json', dict(status='predictions_frozen_before_scoring', model_sha256=decision['model_sha256'],
        targets=52, original_contexts_replayed=84, stored_tile_maps=936, no_confirmation_labels_used_in_predictions=True,
        artifact_hashes={n: sha(OUT / n) for n in artifacts}))
    print('Frozen all52 predictions and936 tile maps; no center-label scoring in inference stage.', flush=True)


def evaluate():
    protected_ok()
    assert not (OUT / 'evaluation_report.json').exists(), 'Keep the completed confirmation.'
    manifest = load(OUT / 'inference_manifest.json')
    assert all(sha(OUT / n) == h for n, h in manifest['artifact_hashes'].items())
    labels = read(CHECK / 'completed_review/manual_sequence_labels.csv')
    old = {r['sample_id']: r for r in read(CHECK / 'frozen_predictions.csv') if r['method'] == 'displacement_0.1'}
    candidate = {r['sample_id']: r for r in read(OUT / 'frozen_confirmation_predictions.csv')}
    assert len(labels) == len(candidate) == len(old) == 52
    points, summary, pairs, effects = [], [], [], []
    outcomes = {}
    for method, predictions in [('current_temporal', old), ('logistic10_frozen', candidate)]:
        outcomes[method] = {}
        for window in ['A', 'B', 'C', 'D']:
            group = sorted([g for g in labels if g['window_id'] == window], key=lambda g: int(g['local_frame']))
            counts = Counter()
            longest = run = 0
            for g in group:
                p = predictions[g['sample_id']]
                assert all(p[k] == g[k] for k in ['candidate_id', 'source_frame', 'local_frame'])
                kind, error = classify(g, p)
                outcomes[method][g['sample_id']] = kind
                counts[kind] += 1
                run = run + 1 if kind == 'correct16px' else 0
                longest = max(longest, run)
                points.append(dict(method=method, case_id=g['case_id'], sample_id=g['sample_id'], window_id=window,
                    local_frame=g['local_frame'], source_frame=g['source_frame'], truth_visibility=g['truth_visibility'],
                    phase_status=g['phase_status'], predicted_visible=p['visible'], x_px=p['x_px'], y_px=p['y_px'],
                    outcome=kind, error_px=error if error is not None else ''))
            visible = counts['correct16px'] + counts['missing_visible_ball'] + counts['wrong_position']
            summary.append(dict(method=method, window_id=window,
                **{k: counts[k] for k in ['correct16px', 'missing_visible_ball', 'wrong_position', 'output_when_not_observable', 'no_output_when_not_observable', 'excluded_ball_uncertain', 'excluded_nonplay', 'excluded_phase_uncertain']},
                visible_total=visible, visible_recall16px=counts['correct16px'] / visible if visible else None,
                longest_correct_sampled_run=longest))
            for a, b in zip(group, group[1:]):
                assert int(b['local_frame']) - int(a['local_frame']) == 2
                known = all(g['phase_status'] == 'play' and g['truth_visibility'] == 'visible' for g in [a, b])
                pa, pb = predictions[a['sample_id']], predictions[b['sample_id']]
                present = all(int(p['visible']) for p in [pa, pb])
                error = math.hypot((float(pb['x_px']) - float(pa['x_px'])) - (float(b['truth_x_px']) - float(a['truth_x_px'])),
                    (float(pb['y_px']) - float(pa['y_px'])) - (float(b['truth_y_px']) - float(a['truth_y_px']))) if known and present else ''
                pairs.append(dict(method=method, window_id=window, first_local_frame=a['local_frame'], last_local_frame=b['local_frame'],
                    verified_visible_play_pair=int(known), both_outputs_present=int(present), displacement_error_px=error,
                    both_correct16px=int(known and outcomes[method][a['sample_id']] == outcomes[method][b['sample_id']] == 'correct16px')))
    for g in labels:
        sid = g['sample_id']
        a, b = outcomes['current_temporal'][sid], outcomes['logistic10_frozen'][sid]
        effects.append(dict(sample_id=sid, case_id=g['case_id'], window_id=g['window_id'], current_outcome=a, candidate_outcome=b,
            gained_correct=int(b == 'correct16px' and a != 'correct16px'), lost_correct=int(a == 'correct16px' and b != 'correct16px')))
    totals = {}
    for method in outcomes:
        counts = Counter(outcomes[method].values())
        totals[method] = dict(counts)
        recalls = [s['visible_recall16px'] for s in summary if s['method'] == method and s['visible_recall16px'] is not None]
        totals[method].update(pooled_visible_recall16px=counts['correct16px'] / 45, macro_window_recall16px=sum(recalls) / len(recalls))
    base, prop = totals['current_temporal'], totals['logistic10_frozen']
    assert base['correct16px'] == 33 and base['missing_visible_ball'] == 12
    preserved = all(all(old[sid][k] == candidate[sid][k] for k in ['visible', 'x_px', 'y_px', 'confidence']) for sid in old if int(old[sid]['visible']))
    safe = preserved and not any(e['lost_correct'] for e in effects) and prop.get('correct16px', 0) > base['correct16px'] and all(prop.get(k, 0) <= base.get(k, 0) for k in ['wrong_position', 'output_when_not_observable'])
    write(OUT / 'confirmation_per_frame.csv', points)
    write(OUT / 'confirmation_per_window.csv', summary)
    write(OUT / 'confirmation_adjacent_pairs.csv', pairs)
    write(OUT / 'confirmation_paired_changes.csv', effects)
    report = dict(status='fixed_same_match_confirmation_complete', totals=totals,
        gained_correct=sum(e['gained_correct'] for e in effects), lost_correct=sum(e['lost_correct'] for e in effects),
        original_outputs_preserved=preserved, passes_prespecified_confirmation_guard=safe,
        decision='Eligible for further independent-match/continuous-track evaluation only.' if safe else 'Reject this frozen candidate for operational use. No retuning on these52.',
        automatic_adoption=False, canonical_tracking_changed=False, independent_external_validation=False,
        interpretation='Known same-match development sequences with nearby correlated targets. Exclusions applied equally. No causal,real-time,full-rally or external-match claim.',
        protocol_sha256=sha(OUT / 'protocol.json'), inference_manifest_sha256=sha(OUT / 'inference_manifest.json'))
    protected_ok()
    dump(OUT / 'evaluation_report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['train', 'infer', 'evaluate'])
    args = parser.parse_args()
    {'train': train, 'infer': infer, 'evaluate': evaluate}[args.stage]()
