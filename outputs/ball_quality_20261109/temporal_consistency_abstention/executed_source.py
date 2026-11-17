"""Training-first safety study, then known-case regression; no operational writes."""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT, SOURCE, read, load, write, sha, cv2
from run_manual_ball_finetune import tensors
from compare_ball_candidate_decoders import components
from compare_temporal_candidate_paths import select
from evaluate_observed_ball_fallback import camera_mask
from evaluate_training_phase_scope import classify
from ball_temporal_consistency_rule import decide, WINDOW, MARGIN, RATE, MIN_ANCHOR_SEPARATION

BASE = ROOT.parent / 'ball_quality_20261109'
OUT = BASE / 'temporal_consistency_abstention'
FIXED = BASE / 'frozen_tile_gate_confirmation'
CONT = BASE / 'continuous_tile_gate_review'
PHASE = BASE / 'training_phase_scope_review'
CFG = dict(name='displacement_0.1', order='displacement', strength=.1)


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def protected_ok():
    p = load(OUT / 'protocol.json')
    assert all(sha(Path(f)) == h for f, h in p['protected_input_hashes'].items())
    assert all(sha(Path(f)) == h for f, h in p['helper_hashes'].items())
    assert sha(Path(__file__)) == p['script_sha256']
    return p


def withhold(row, d):
    result = dict(row)
    if not d['keep']:
        result.update(visible=0, x_px=0, y_px=0, confidence=0., output_source='withheld_temporal_contradiction')
    return result


def train():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Do not tune this rule after observing results.'
    label_path = ROOT / 'balltrack_expanded_centers_pilot/expanded_train_labels.csv'
    labels = read(label_path)
    protected_paths = [label_path, FIXED / 'training_predictions.csv', FIXED / 'frozen_training_head.npz',
        FIXED / 'evaluation_report.json', PHASE / 'phase_labels.csv', PHASE / 'evaluation_report.json',
        ROOT / 'ball_temporal_path_comparison/train_displacement_0.1.csv',
        ROOT / 'reviewed_ball_overrides.json', ROOT / 'reviewed_person_overrides.json', ROOT / 'reviewed_tracking_bundles.json',
        CONT / 'continuous_predictions.csv', CONT / 'global_contexts.json', CONT / 'camera_features.json',
        CONT / 'completed_identity_review.json', CONT / 'focus_identity_labels.csv', CONT / 'operational_decision.json',
        ROOT.parent / 'ball_quality_20261104/completed_review/manual_sequence_labels.csv',
        SOURCE / 'third_party/RacketVision/source/BallTrack/checkpoints/balltrack_best.pth']
    plans = []
    for cid in sorted({r['candidate_id'] for r in labels}):
        group = [r for r in labels if r['candidate_id'] == cid]
        frame_dir = Path(json.loads(group[0]['input_frames_json'])[0]).parent
        median = Path(group[0]['median_path'])
        camera = ROOT.parent / 'ball_quality_20261102/temporal_guard/camera_features' / (cid + '.json')
        metadata = frame_dir.parent / 'runs/threshold030_median120/run_metadata.json'
        count = load(metadata)['frame_count']
        targets = sorted(int(r['local_frame']) for r in group)
        anchors = sorted({t + d for t in targets for d in range(-WINDOW, WINDOW + 1) if 3 <= t + d < count})
        context = sorted({a + d for a in anchors for d in range(-8, 9, 2) if 3 <= a + d < count})
        protected_paths += [median, camera, metadata]
        plans.append(dict(candidate_id=cid, target_times=targets, anchor_times=anchors, context_times=context,
            frame_dir=str(frame_dir), median_path=str(median), camera_path=str(camera), frame_count=count))
    helpers = [ROOT / n for n in ['ball_temporal_consistency_rule.py', 'compare_temporal_candidate_paths.py',
        'compare_ball_candidate_decoders.py', 'evaluate_observed_ball_fallback.py', 'run_manual_ball_finetune.py', 'evaluate_training_phase_scope.py']]
    protocol = dict(status='fixed_before_new_training_context_inference_and_rule_results', date_jst='2026-11-09',
        proposal='One non-learned two-anchor contradiction rule; no grid,fitting,or threshold adaptation.',
        parameters=dict(window_source_frames=WINDOW, margin_source_px=MARGIN, rate_px_per_source_frame=RATE,
            minimum_anchor_separation_source_frames=MIN_ANCHOR_SEPARATION),
        parameter_rationale='Reuse old8frame local window,16px tolerance,and64px per2source-frame displacement scale. This repurposing is an exploratory heuristic,not a validated physical speed limit.',
        anchors='Original full-frame fixed temporal observations only. No manual centers,added candidates,recursive filtering or interpolated positions become anchors.',
        rule='Prefer nearest original observation before and after. Existing points require a bracketing pair. Added candidates may use nearest two observations on the only available side,separated by>=2sourceframes. Withhold only if anchors are mutually compatible AND target contradicts BOTH bounds16+32*elapsed_frames. Otherwise retain,including lack of evidence; no claim retained means verified.',
        cuts='Do not compare across camera transition flags. Ambiguous evidence retains prior output.',
        output='Withholding means unknown position,not recovered ball or physically absent ball. No coordinate replacement or interpolation.',
        training='Fixed123training targets and previous full-fit predictions. Recompute contiguous global contexts for original anchor support; no detector/head training. Phase77play/46nonplay already human fixed.',
        training_safety='Retain ALL59correct play targets of previous frozen supplement; do not increase wrong_position OR output_when_not_observable. Report nonplay separately. Failure stops before known-case regression; no alternate radius or variant.',
        controls='Original temporal baseline,previous static supplement,supplement with guard restricted to additions,and guard on existing plus added outputs. Main rule is fixed; controls are ablations,not alternatives selected after results.',
        regression='Only after training safety passes,apply same rule to immutable724continuous predictions and original observations. B275,D515/516 are known development failure cases. No external-performance claim,no automatic adoption.',
        limitations='Exploratory post-failure method,limited negative examples,possible real rapid-turn rejection and wrong anchors. Absence of contradiction is not identity proof.',
        protected_input_hashes={str(p): sha(p) for p in protected_paths}, helper_hashes={str(p): sha(p) for p in helpers},
        plans=plans, script_sha256=sha(Path(__file__)), automatic_adoption=False, new_human_labels=0)
    dump(OUT / 'protocol.json', protocol)
    shutil.copy2(Path(__file__), OUT / 'executed_source.py')
    protected_ok()
    modelroot = SOURCE / 'third_party/RacketVision/source/BallTrack'
    sys.path.insert(0, str(modelroot))
    from inference import BallInferencer
    torch.backends.cudnn.benchmark = False
    engine = BallInferencer(str(modelroot / 'configs/tracknetv3_base.py'), str(modelroot / 'checkpoints/balltrack_best.pth'), device='cuda', thre=.3, batchsize=4)
    engine.model.eval().requires_grad_(False)
    contexts, anchor_tables, cameras, image_hashes = {}, {}, {}, {}
    for plan in plans:
        cid = plan['candidate_id']
        median = np.load(plan['median_path'])['median']
        medians = {'m': np.moveaxis(cv2.resize(median, (512, 288)).astype(np.float32) / 255., -1, 0)}
        mask = camera_mask(load(Path(plan['camera_path']))['changed_fraction'])
        assert len(mask) == plan['frame_count']
        context = {}
        times = plan['context_times']
        for start in range(0, len(times), 4):
            batch = []
            images = {}
            for t in times[start:start + 4]:
                for j in range(t - 3, t + 1):
                    if j in images:
                        continue
                    path = Path(plan['frame_dir']) / f'{j:06d}.jpg'
                    image_hashes[str(path)] = sha(path)
                    images[j] = cv2.imread(str(path))
                    assert images[j].shape == (1080, 1920, 3)
                batch.append(dict(frames=np.stack([cv2.resize(images[j], (512, 288)) for j in range(t - 3, t + 1)]), median_key='m', row={}))
            with torch.inference_mode(), torch.amp.autocast('cuda'):
                outputs, _, _ = engine.model.forward(frames=tensors(batch, medians, engine.device))
            for t, hm in zip(times[start:start + 4], outputs[:, 0].float().cpu().numpy(), strict=True):
                context[t] = components(hm)
        anchors = {}
        for t in plan['anchor_times']:
            c, _ = select(context, t, mask, CFG)
            if c is not None:
                anchors[t] = c
        contexts[cid] = context
        anchor_tables[cid] = anchors
        cameras[cid] = mask
        print('Training original contexts', cid, len(context), 'observed anchors', len(anchors), flush=True)
    dump(OUT / 'training_global_contexts.json', contexts)
    dump(OUT / 'training_anchor_observations.json', anchor_tables)
    dump(OUT / 'training_input_hashes.json', image_hashes)
    phases = {r['sample_id']: r for r in read(PHASE / 'phase_labels.csv')}
    labels_by = {r['sample_id']: r for r in labels}
    raw = read(FIXED / 'training_predictions.csv')
    baseline = read(ROOT / 'ball_temporal_path_comparison/train_displacement_0.1.csv')
    guarded, additions_only, diagnostics, target_replay = [], [], [], []
    for r in raw:
        sid, cid, t = r['sample_id'], r['candidate_id'], int(r['local_frame'])
        original_present = r['output_source'] == 'unchanged_original_temporal'
        point = r if int(r['visible']) else None
        d = decide(t, point, original_present, anchor_tables[cid], cameras[cid])
        guarded.append(withhold(r, d))
        additions_only.append(dict(r) if original_present else withhold(r, d))
        diagnostics.append(dict(sample_id=sid, candidate_id=cid, local_frame=t, phase_status=phases[sid]['phase_status'],
            original_present=int(original_present), previous_visible=r['visible'], **d))
        old_base = next(b for b in baseline if b['sample_id'] == sid)
        new_base = anchor_tables[cid].get(t)
        presence_match = int(old_base['visible']) == int(new_base is not None)
        center_match = presence_match and (new_base is None or (int(old_base['x_px']), int(old_base['y_px'])) == (new_base['x_px'], new_base['y_px']))
        target_replay.append(dict(sample_id=sid, presence_and_center_match=center_match))
    write(OUT / 'training_guarded_predictions.csv', guarded)
    write(OUT / 'training_additions_only_guard.csv', additions_only)
    write(OUT / 'training_guard_diagnostics.csv', diagnostics)
    write(OUT / 'training_anchor_target_numeric_check.csv', target_replay)
    summaries, points = [], []
    methods = [('original_temporal', baseline), ('static_supplement', raw), ('guard_additions_only', additions_only), ('guard_all_outputs', guarded)]
    for name, predictions in methods:
        for phase in ['play', 'nonplay']:
            counts = Counter()
            for r in predictions:
                sid = r['sample_id']
                if phases[sid]['phase_status'] != phase:
                    continue
                outcome, error = classify(labels_by[sid], r)
                counts[outcome] += 1
                points.append(dict(method=name, sample_id=sid, phase_status=phase, outcome=outcome, error_px=error if error is not None else ''))
            summaries.append(dict(method=name, phase_status=phase, **{k: counts[k] for k in ['correct16px', 'wrong_position', 'missing_visible_ball', 'output_when_not_observable', 'no_output_when_not_observable']}))
    write(OUT / 'training_summary.csv', summaries)
    write(OUT / 'training_per_frame_outcomes.csv', points)
    p0 = {r['sample_id']: r for r in points if r['method'] == 'static_supplement' and r['phase_status'] == 'play'}
    p1 = {r['sample_id']: r for r in points if r['method'] == 'guard_all_outputs' and r['phase_status'] == 'play'}
    lost = [sid for sid in p0 if p0[sid]['outcome'] == 'correct16px' and p1[sid]['outcome'] != 'correct16px']
    old_counts = next(s for s in summaries if s['method'] == 'static_supplement' and s['phase_status'] == 'play')
    new_counts = next(s for s in summaries if s['method'] == 'guard_all_outputs' and s['phase_status'] == 'play')
    assert old_counts['correct16px'] == 59
    safe = not lost and all(new_counts[k] <= old_counts[k] for k in ['wrong_position', 'output_when_not_observable'])
    numeric_ok = all(r['presence_and_center_match'] for r in target_replay)
    safe = safe and numeric_ok
    protected_ok()
    assert all(sha(Path(p)) == h for p, h in image_hashes.items())
    decision = dict(status='training_safety_passed' if safe else 'stopped_training_safety_failed', passed=safe,
        previous_play=old_counts, guarded_play=new_counts, lost_correct_sample_ids=lost,
        target_numeric_match_count=sum(r['presence_and_center_match'] for r in target_replay),
        target_count=123, total_withheld=sum(not d['keep'] for d in diagnostics),
        known_failure_regression_permitted=safe, operational_adoption=False,
        scope='Training-only sanity check of one fixed heuristic,not external evaluation.',
        artifacts={n: sha(OUT / n) for n in ['protocol.json', 'training_global_contexts.json', 'training_anchor_observations.json', 'training_input_hashes.json',
            'training_guarded_predictions.csv', 'training_additions_only_guard.csv', 'training_guard_diagnostics.csv', 'training_summary.csv', 'training_per_frame_outcomes.csv']})
    dump(OUT / 'training_decision.json', decision)
    print(json.dumps({k: v for k, v in decision.items() if k != 'artifacts'}, indent=2))


def regression():
    protocol = protected_ok()
    training = load(OUT / 'training_decision.json')
    assert training['passed'], 'Stop after training loss; do not check known failures or tune an alternative.'
    assert all(sha(OUT / n) == h for n, h in training['artifacts'].items())
    assert not (OUT / 'regression_report.json').exists()
    rows = read(CONT / 'continuous_predictions.csv')
    plans = load(CONT / 'protocol.json')['plans']
    context_tables = load(CONT / 'global_contexts.json')
    camera_tables = load(CONT / 'camera_features.json')
    guarded, diagnostics, controls = [], [], []
    for plan in plans:
        cid = plan['candidate_id']
        context = {int(t): c for t, c in context_tables[cid].items()}
        mask = np.zeros(plan['frame_count'], dtype=bool)
        for t, v in camera_tables[cid]['mask']:
            mask[t] = v
        # Every eligible anchor has full selector support inside the stored context.
        support_times = range(plan['review_first'], plan['review_last'] + 1)
        anchors = {}
        for t in support_times:
            c, _ = select(context, t, mask, CFG)
            if c is not None:
                anchors[t] = c
        for r in [r for r in rows if r['candidate_id'] == cid]:
            t = int(r['local_frame'])
            original_present = bool(int(r['original_visible']))
            d = decide(t, r if int(r['visible']) else None, original_present, anchors, mask)
            result = withhold(r, d)
            guarded.append(result)
            controls.append(dict(r) if original_present else withhold(r, d))
            diagnostics.append(dict(sample_id=r['sample_id'], window_id=r['window_id'], local_frame=t,
                previous_visible=r['visible'], guarded_visible=result['visible'], original_present=int(original_present), **d))
    write(OUT / 'continuous_guarded_predictions.csv', guarded)
    write(OUT / 'continuous_additions_only_control.csv', controls)
    write(OUT / 'continuous_guard_diagnostics.csv', diagnostics)
    by = {(r['window_id'], r['local_frame']): r for r in guarded}
    known = read(CONT / 'focus_identity_labels.csv')
    results = []
    for g in known:
        assert g['identity_status'] == 'confirmed_not_rally_ball'
        p = by[g['window_id'], g['local_frame']]
        results.append(dict(window_id=g['window_id'], local_frame=g['local_frame'], source_frame=g['source_frame'],
            wrong_output_withheld=not int(p['visible']), visible_after=p['visible'], human_center_known=False))
    write(OUT / 'known_failure_regression.csv', results)
    # Existing52centers remain development anchors,not additional labels or independent samples.
    gt = read(ROOT.parent / 'ball_quality_20261104/completed_review/manual_sequence_labels.csv')
    old_by = {(r['candidate_id'], r['local_frame']): r for r in rows}
    new_by = {(r['candidate_id'], r['local_frame']): r for r in guarded}
    sparse_counts, losses = {}, []
    from evaluate_reserved_ball_sequences import classify as classify52
    for name, table in [('before_guard', old_by), ('after_guard', new_by)]:
        counts = Counter()
        for g in gt:
            p = table[g['candidate_id'], g['local_frame']]
            kind, _ = classify52(g, p)
            counts[kind] += 1
            if name == 'after_guard':
                previous, _ = classify52(g, old_by[g['candidate_id'], g['local_frame']])
                if previous == 'correct16px' and kind != 'correct16px':
                    losses.append(g['sample_id'])
        sparse_counts[name] = dict(counts)
    report = dict(status='known_development_regression_complete', known_targets=results,
        known_wrong_frames_withheld=sum(r['wrong_output_withheld'] for r in results),
        known_failure_segments=2, no_ball_positions_recovered=True,
        sparse52_development_counts=sparse_counts, lost_correct_sparse_sample_ids=losses,
        continuous_withheld=sum(int(a['visible']) and not int(b['visible']) for a, b in zip(rows, guarded, strict=True)),
        same_frame_positions_never_changed=True, newly_validated_continuous_accuracy=None,
        boundary_note='At4window edges,anchor evidence truncated to stored review targets; unknown support is retained rather than labelled wrong.',
        operational_adoption=False, model_or_canonical_tracking_changed=False,
        interpretation='Known-case diagnostic and52previously examined anchors only. Suppressing known wrong outputs is not recovered localization or external validation.')
    protected_ok()
    dump(OUT / 'regression_report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['train', 'regression'])
    args = parser.parse_args()
    {'train': train, 'regression': regression}[args.stage]()
