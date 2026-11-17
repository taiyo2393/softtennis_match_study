"""All-target frozen-pool upper bounds,never tracking accuracy or pseudo-labels."""
import json
import math
from collections import Counter
from pathlib import Path

from integrate_044_reviewed_tracks import ROOT, read, load, sha, write

BASE = ROOT.parent / 'ball_quality_20261109'
OUT = BASE / 'multicandidate_feasibility'
EF = BASE / 'temporal_guard_next_sequences'
ABCD = ROOT.parent / 'ball_quality_20261104'


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Preserve the first all-target diagnosis.'
    paths = [ROOT / 'balltrack_expanded_centers_pilot/expanded_train_labels.csv',
        BASE / 'training_phase_scope_review/phase_labels.csv', BASE / 'training_tile_detail_comparison/candidate_pools.json',
        BASE / 'temporal_consistency_abstention/training_global_contexts.json',
        BASE / 'frozen_tile_gate_confirmation/training_predictions.csv',
        BASE / 'frozen_tile_gate_confirmation/candidate_pools.json', BASE / 'frozen_tile_gate_confirmation/frozen_confirmation_predictions.csv',
        ABCD / 'completed_review/manual_sequence_labels.csv', ABCD / 'frozen_component_cache.json',
        EF / 'completed_review/manual_sequence_labels.csv', EF / 'global_contexts.json', EF / 'candidate_pools.json',
        EF / 'frozen_predictions.csv', EF / 'evaluation_report.json',
        ROOT / 'reviewed_ball_overrides.json', ROOT / 'reviewed_person_overrides.json', ROOT / 'reviewed_tracking_bundles.json']
    protected = {str(p): sha(p) for p in paths}
    dump(OUT / 'protocol.json', dict(status='fixed_before_all_target_pool_diagnosis', date_jst='2026-11-09',
        scope='All123training targets,all52oldconfirmation targets,all26newconfirmation targets,reported separately;same phase/visibility criteria for success and failure.',
        pools='Saved original global components plus ALL native9tile components. Degraded components are not proposed output positions. No new inference,threshold,fit or candidate search.',
        radius_source_px=16, interpretation='GT-assisted candidate-center coverage is an oracle upper bound,NOT automatic localization accuracy or person/ball identity accuracy. Candidate counts are not distinct physical balls.',
        categories='If chosen static point is incorrect but pool has a center within16px:potential selection limitation. If pool has none:current candidate-generation limitation. No claim a path method can attain the upper bound.',
        data_roles='E/Flabels remain confirmation/development labels,never transferred into training. Looking at their failures now is retrospective diagnosis,not another independent test.',
        no_model_training=True, no_new_labels=True, canonical_tracking_changed=False, protected_input_hashes=protected,
        script_sha256=sha(Path(__file__))))
    train = read(paths[0])
    phases = {r['sample_id']: r['phase_status'] for r in read(BASE / 'training_phase_scope_review/phase_labels.csv')}
    train = [dict(r, phase_status=phases[r['sample_id']]) for r in train]
    old = read(ABCD / 'completed_review/manual_sequence_labels.csv')
    new = read(EF / 'completed_review/manual_sequence_labels.csv')
    sets = [dict(name='training123', labels=train,
        globals=load(BASE / 'temporal_consistency_abstention/training_global_contexts.json'),
        native=load(BASE / 'training_tile_detail_comparison/candidate_pools.json')['native'],
        outputs={r['sample_id']: r for r in read(BASE / 'frozen_tile_gate_confirmation/training_predictions.csv')}),
        dict(name='known_ABCD52', labels=old, globals=load(ABCD / 'frozen_component_cache.json'),
            native=load(BASE / 'frozen_tile_gate_confirmation/candidate_pools.json')['native'],
            outputs={r['sample_id']: r for r in read(BASE / 'frozen_tile_gate_confirmation/frozen_confirmation_predictions.csv')}),
        dict(name='known_EF26', labels=new, globals=load(EF / 'global_contexts.json'), native=load(EF / 'candidate_pools.json')['native'],
            outputs={r['sample_id']: r for r in read(EF / 'frozen_predictions.csv') if r['method'] == 'static_supplement'})]
    per_frame, summary, deficits = [], [], []
    for dataset in sets:
        counts = Counter()
        for g in dataset['labels']:
            sid, cid = g['sample_id'], g['candidate_id']
            global_pool = dataset['globals'][cid][g['local_frame']]
            native = dataset['native'][sid]
            observable = g['phase_status'] == 'play' and g['truth_visibility'] == 'visible'
            if g['phase_status'] not in ['play']:
                counts['excluded_nonplay_or_phase_unknown'] += 1
                continue
            if not observable:
                counts['excluded_no_observable_center'] += 1
                continue
            distance = lambda c: math.hypot(c['x_px'] - float(g['truth_x_px']), c['y_px'] - float(g['truth_y_px']))
            gp = min((distance(c) for c in global_pool), default=None)
            np = min((distance(c) for c in native), default=None)
            union = global_pool + native
            up = min((distance(c) for c in union), default=None)
            output = dataset['outputs'][sid]
            selected_error = math.hypot(float(output['x_px']) - float(g['truth_x_px']), float(output['y_px']) - float(g['truth_y_px'])) if int(output['visible']) else None
            correct = selected_error is not None and selected_error <= 16
            has = up is not None and up <= 16
            category = 'already_selected_correct' if correct else 'near_center_candidate_exists_not_selected' if has else 'no_center_candidate_in_current_union'
            counts['visible_play_targets'] += 1
            counts['selected_correct'] += correct
            counts['global_center_coverage16'] += gp is not None and gp <= 16
            counts['native_center_coverage16'] += np is not None and np <= 16
            counts['union_center_coverage16'] += has
            counts[category] += 1
            row = dict(dataset=dataset['name'], sample_id=sid, case_id=g.get('case_id', ''), candidate_id=cid,
                local_frame=g['local_frame'], source_frame=g['source_frame'], truth_x_px=g['truth_x_px'], truth_y_px=g['truth_y_px'],
                global_components=len(global_pool), native_components=len(native),
                exact_center_union_count=len({(c['x_px'], c['y_px']) for c in union}),
                nearest_global_error_px=gp if gp is not None else '', nearest_native_error_px=np if np is not None else '',
                nearest_union_error_px=up if up is not None else '', selected_error_px=selected_error if selected_error is not None else '',
                selected_visible=output['visible'], category=category, oracle_not_automatic_accuracy=True)
            per_frame.append(row)
            if not correct:
                deficits.append(row)
        summary.append(dict(dataset=dataset['name'], **dict(counts)))
    write(OUT / 'all_visible_play_targets.csv', per_frame)
    write(OUT / 'selected_output_deficits.csv', deficits)
    dump(OUT / 'result.json', dict(status='all_target_diagnosis_complete', summary=summary,
        limitations='Oracle coverage,known same-match data,sparse centers. No new tracking method evaluated;candidate union may contain multiple unrelated objects.',
        no_accuracy_claim_for_candidate_coverage=True, models_labels_and_tracking_unchanged=True))
    assert all(sha(Path(p)) == h for p, h in protected.items())
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
