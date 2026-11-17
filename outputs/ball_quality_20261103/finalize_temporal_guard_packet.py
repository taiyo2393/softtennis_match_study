"""Independently reconcile completed annotation/report and retain a failed guard."""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from integrate_044_reviewed_tracks import ROOT, read, load, write, sha

BASE = ROOT.parent / 'ball_quality_20261109'
OUT = BASE / 'temporal_guard_next_sequences'
METHODS = ['original_temporal', 'static_supplement', 'two_anchor_guard_all_outputs']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    manifest = load(OUT / 'manifest.json')
    report = load(OUT / 'evaluation_report.json')
    assert all(sha(Path(p)) == h for p, h in manifest['protected_input_hashes'].items())
    assert all(sha(OUT / n) == h for n, h in manifest['frozen_artifact_hashes'].items())
    assert all(sha(Path(p)) == h for p, h in load(OUT / 'review_tools.json')['tool_hashes'].items())
    assert report['labels_sha256'] == sha(OUT / 'manual_sequence_labels.csv')
    assert report['events_sha256'] == sha(OUT / 'annotation_events.jsonl')
    assert report['predictions_sha256'] == sha(OUT / 'frozen_predictions.csv')
    labels = read(OUT / 'manual_sequence_labels.csv')
    initial = read(OUT / 'initial_labels_snapshot.csv')
    state = {g['case_id']: g for g in initial}
    events = [json.loads(line) for line in (OUT / 'annotation_events.jsonl').read_text(encoding='utf-8').splitlines() if line]
    for e in events:
        assert state[e['case_id']] == e['before'] and e['model_predictions_hidden'] is True
        state[e['case_id']] = e['after']
    assert [state[g['case_id']] for g in initial] == labels
    assert len(labels) == len({g['sample_id'] for g in labels}) == 26
    assert all(g['reviewer_id'] == 'human_user' and g['review_status'] == 'human_frame_review' and g['annotated_at_utc'] for g in labels)
    by_gt = {g['sample_id']: g for g in labels}
    predictions = {(r['method'], r['sample_id']): r for r in read(OUT / 'frozen_predictions.csv')}
    points = {(r['method'], r['sample_id']): r for r in read(OUT / 'per_frame_evaluation.csv')}
    assert len(predictions) == len(points) == 78
    kinds = {}
    for method in METHODS:
        counts = Counter()
        kinds[method] = {}
        for g in labels:
            assert g['phase_status'] == 'play' and g['truth_visibility'] == 'visible'
            assert 0 <= float(g['truth_x_px']) < 1920 and 0 <= float(g['truth_y_px']) < 1080
            p = predictions[method, g['sample_id']]
            assert all(g[k] == p[k] for k in ['case_id', 'candidate_id', 'local_frame', 'source_frame'])
            if not int(p['visible']):
                kind = 'missing_visible_ball'
            else:
                distance = math.dist((float(g['truth_x_px']), float(g['truth_y_px'])), (float(p['x_px']), float(p['y_px'])))
                kind = 'correct16px' if distance <= 16 else 'wrong_position'
                assert abs(float(points[method, g['sample_id']]['error_px']) - distance) < 1e-10
            assert points[method, g['sample_id']]['outcome'] == kind
            counts[kind] += 1
            kinds[method][g['sample_id']] = kind
        for key in ['correct16px', 'wrong_position', 'missing_visible_ball']:
            assert counts[key] == report['totals'][method][key]
        assert report['totals'][method]['pooled_visible_recall16px'] == counts['correct16px'] / 26
    changes = read(OUT / 'paired_changes.csv')
    losses, gains = [], []
    for row in changes:
        sid, before = row['sample_id'], row['reference_method']
        a, b = kinds[before][sid], kinds[METHODS[2]][sid]
        loss, gain = a == 'correct16px' and b != 'correct16px', a != 'correct16px' and b == 'correct16px'
        assert int(row['loss']) == loss and int(row['gain']) == gain
        if loss:
            losses.append(dict(reference=before, case_id=by_gt[sid]['case_id'], sample_id=sid))
        if gain:
            gains.append(dict(reference=before, case_id=by_gt[sid]['case_id'], sample_id=sid))
    for pair in read(OUT / 'adjacent_sampled_pairs.csv'):
        group = {g['local_frame']: g for g in labels if g['window_id'] == pair['window_id']}
        a, b = group[pair['first_local_frame']], group[pair['last_local_frame']]
        pa, pb = predictions[pair['method'], a['sample_id']], predictions[pair['method'], b['sample_id']]
        if pair['displacement_error_px'] != '':
            error = math.dist((float(pb['x_px']) - float(pa['x_px']), float(pb['y_px']) - float(pa['y_px'])),
                (float(b['truth_x_px']) - float(a['truth_x_px']), float(b['truth_y_px']) - float(a['truth_y_px'])))
            assert abs(error - float(pair['displacement_error_px'])) < 1e-10
    snapshot_hashes = {}
    for p in (OUT / 'completed_review').iterdir():
        assert sha(p) == sha(OUT / p.name)
        snapshot_hashes[p.name] = sha(p)
    # Diagnose the fixed rule's loss without editing GT or exploring new bounds.
    diagnosis = OUT / 'failure_diagnosis'
    diagnosis.mkdir(exist_ok=True)
    from compare_temporal_candidate_paths import select
    cid = 'match1_motion_expand_01_006'
    context = {int(t): cs for t, cs in load(OUT / 'global_contexts.json')[cid].items()}
    mask = np.zeros(300, dtype=bool)
    for t, value in load(OUT / 'camera_features.json')[cid]['mask_pairs']:
        mask[t] = value
    g_by_t = {int(g['local_frame']): g for g in labels if g['candidate_id'] == cid}
    trace = []
    for t in range(154, 163):
        p, status = select(context, t, mask, dict(order='displacement', strength=.1))
        g = g_by_t.get(t)
        trace.append(dict(local_frame=t, predicted_visible=int(p is not None), x_px=p['x_px'] if p else '', y_px=p['y_px'] if p else '',
            human_center_available=int(g is not None), truth_x_px=g['truth_x_px'] if g else '', truth_y_px=g['truth_y_px'] if g else '',
            anchor_for_withheld158=int(t in [157, 159]), withheld_by_guard=int(t == 158), status=status))
    write(diagnosis / 'E154_162_observations.csv', trace)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, key, gtkey in zip(axes, ['x_px', 'y_px'], ['truth_x_px', 'truth_y_px'], strict=True):
        model_rows = [r for r in trace if r['predicted_visible']]
        truth_rows = [r for r in trace if r['human_center_available']]
        ax.plot([r['local_frame'] for r in model_rows], [r[key] for r in model_rows], color='#2679be', marker='o', label='Original observations (line is visual aid)')
        ax.scatter([r['local_frame'] for r in truth_rows], [float(r[gtkey]) for r in truth_rows], marker='+', s=110, linewidths=2, color='#137d42', label='Human centers: even frames only', zorder=4)
        target = next(r for r in trace if r['local_frame'] == 158)
        ax.scatter([158], [target[key]], facecolors='none', edgecolors='#c74740', s=180, linewidths=2, label='Correct output wrongly withheld', zorder=5)
        ax.axvspan(156.85, 157.15, color='#b08c25', alpha=.15)
        ax.axvspan(158.85, 159.15, color='#b08c25', alpha=.15)
        ax.set_ylabel(key + ' (source pixels)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc='upper left')
    axes[1].set_xlabel('Local source frame (157 and159 are unlabelled anchor frames)')
    axes[1].set_xticks(range(154, 163))
    fig.suptitle('E11 / frame158: a correct ball observation rejected by a two-anchor rule')
    fig.tight_layout()
    fig.savefig(diagnosis / 'E11_false_abstention.png', dpi=160)
    fig.savefig(diagnosis / 'E11_false_abstention.pdf')
    plt.close(fig)
    decision = dict(status='two_anchor_guard_rejected_on_reserved_confirmation',
        reason='One correct original and supplemented target(E11/local158)was withheld;wrong localization E10/local156remains. Frozen safety requires no correct losses.',
        lost_correct_target=dict(case_id='E11', local_frame=158, source_frame=40707, original_center_error_px=math.sqrt(20)),
        remaining_wrong_localization=dict(case_id='E10', local_frame=156, source_frame=40705, error_px=math.sqrt(452),
            semantic_identity='Not established; outside16px is a localization error,not proof of a different object.'),
        recovered_missing_target=dict(case_id='E13', local_frame=162, source_frame=40711, supplemented_error_px=math.sqrt(65)),
        head_only_not_adopted=True, head_only_reason='Earlier human review confirmed added wrong-target outputsD515/516;25/26here does not erase that failure.',
        prior_training_and_known_case_results_retained=True, current26_never_training=True,
        threshold_or_window_retuned=False, operational_adoption=False, canonical_tracking_changed=False,
        next='Keep this prospective failure fixed. Examine trustworthiness of neighbouring candidate observations and direction changes;do not select a new radius onE11 or transferE/Flabels into training silently.')
    dump(OUT / 'operational_decision.json', decision)
    dump(OUT / 'completed_verification.json', dict(status='passed', human_targets=26, journal_events=len(events),
        human_labels_and_history_match=True, frozen_outputs_verified=78, reported_outcomes_recomputed=78,
        lost_correct_targets=losses, gained_correct_targets=gains, completed_snapshot_hashes=snapshot_hashes,
        protected_inputs_unchanged=True, operational_adoption=False, no_new_human_labels_written=True,
        script_sha256=sha(Path(__file__)), figure_sha256=sha(diagnosis / 'E11_false_abstention.png')))
    shutil_paths = ['operational_decision.json', 'completed_verification.json']
    import shutil
    for name in shutil_paths:
        shutil.copy2(OUT / name, OUT / 'completed_review' / name)
    print(json.dumps(dict(status='completed_verified_and_guard_rejected', targets=26, events=len(events),
        lost_correct_case='E11', remaining_wrong_position_case='E10', recovered_missing_case='E13',
        no_new_parameter_search=True, canonical_tracking_changed=False), indent=2))


if __name__ == '__main__':
    main()
