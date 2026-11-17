"""Score the3frozen methods only after26genuine human annotations are saved."""
import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from integrate_044_reviewed_tracks import ROOT, read, load, write, sha
from evaluate_reserved_ball_sequences import classify
from serve_temporal_guard_packet import verify_history

OUT = ROOT.parent / 'ball_quality_20261109/temporal_guard_next_sequences'
METHODS = ['original_temporal', 'static_supplement', 'two_anchor_guard_all_outputs']


def main(status_only):
    manifest = load(OUT / 'manifest.json')
    labels = read(OUT / 'manual_sequence_labels.csv')
    pending = sum(r['review_status'] == 'unreviewed' for r in labels)
    assert len(labels) == 26
    verify_history(OUT)
    if pending or status_only:
        print(json.dumps(dict(status='human_annotation_pending' if pending else 'ready_to_evaluate', saved=26 - pending,
            pending=pending, accuracy=None, no_new_scoring=True)))
        return
    assert not (OUT / 'evaluation_report.json').exists(), 'Preserve completed evaluation; revisions need a new snapshot.'
    assert all(sha(Path(p)) == h for p, h in manifest['protected_input_hashes'].items())
    assert all(sha(OUT / n) == h for n, h in manifest['frozen_artifact_hashes'].items())
    assert all(sha(Path(p)) == h for p, h in manifest['helper_hashes'].items())
    initial = {r['sample_id']: r for r in read(OUT / 'initial_labels_snapshot.csv')}
    for g in labels:
        assert g['reviewer_id'] == 'human_user' and g['annotated_at_utc'] and g['review_status'] == 'human_frame_review'
        assert all(g[k] == initial[g['sample_id']][k] for k in ['case_id', 'sample_id', 'window_id', 'candidate_id', 'local_frame', 'source_frame'])
    preds = read(OUT / 'frozen_predictions.csv')
    by = {(p['method'], p['sample_id']): p for p in preds}
    assert len(preds) == len(by) == 78 and sorted({p['method'] for p in preds}) == sorted(METHODS)
    points, summaries, paired, adjacent = [], [], [], []
    kinds = {}
    fields = ['correct16px', 'wrong_position', 'missing_visible_ball', 'output_when_not_observable', 'no_output_when_not_observable',
        'excluded_ball_uncertain', 'excluded_nonplay', 'excluded_phase_uncertain']
    for method in METHODS:
        kinds[method] = {}
        for wid in ['E', 'F']:
            counts = Counter()
            for g in [g for g in labels if g['window_id'] == wid]:
                p = by[method, g['sample_id']]
                assert all(p[k] == g[k] for k in ['case_id', 'candidate_id', 'local_frame', 'source_frame'])
                kind, error = classify(g, p)
                kinds[method][g['sample_id']] = kind
                counts[kind] += 1
                points.append(dict(method=method, sample_id=g['sample_id'], case_id=g['case_id'], window_id=wid,
                    local_frame=g['local_frame'], source_frame=g['source_frame'], phase_status=g['phase_status'],
                    truth_visibility=g['truth_visibility'], visible=p['visible'], x_px=p['x_px'], y_px=p['y_px'], outcome=kind,
                    error_px=error if error is not None else ''))
            visible = sum(counts[k] for k in ['correct16px', 'wrong_position', 'missing_visible_ball'])
            summaries.append(dict(method=method, window_id=wid, **{k: counts[k] for k in fields},
                visible_targets=visible, recall16px=counts['correct16px'] / visible if visible else None))
            group = sorted([g for g in labels if g['window_id'] == wid], key=lambda g: int(g['local_frame']))
            for a, b in zip(group, group[1:]):
                assert int(b['local_frame']) - int(a['local_frame']) == 2
                known = all(g['phase_status'] == 'play' and g['truth_visibility'] == 'visible' for g in [a, b])
                pa, pb = by[method, a['sample_id']], by[method, b['sample_id']]
                present = all(int(p['visible']) for p in [pa, pb])
                error = math.hypot((float(pb['x_px']) - float(pa['x_px'])) - (float(b['truth_x_px']) - float(a['truth_x_px'])),
                    (float(pb['y_px']) - float(pa['y_px'])) - (float(b['truth_y_px']) - float(a['truth_y_px']))) if known and present else ''
                adjacent.append(dict(method=method, window_id=wid, first_local_frame=a['local_frame'], last_local_frame=b['local_frame'],
                    verified_visible_play_pair=int(known), both_outputs_present=int(present), displacement_error_px=error,
                    both_correct16px=int(known and kinds[method][a['sample_id']] == kinds[method][b['sample_id']] == 'correct16px')))
    for g in labels:
        sid = g['sample_id']
        for before in METHODS[:2]:
            a, b = kinds[before][sid], kinds[METHODS[2]][sid]
            paired.append(dict(reference_method=before, sample_id=sid, case_id=g['case_id'], window_id=g['window_id'],
                reference_outcome=a, guarded_outcome=b, gain=int(b == 'correct16px' and a != 'correct16px'),
                loss=int(a == 'correct16px' and b != 'correct16px')))
    totals = {}
    for method in METHODS:
        counts = Counter(kinds[method].values())
        visible = sum(counts[k] for k in ['correct16px', 'wrong_position', 'missing_visible_ball'])
        recalls = [s['recall16px'] for s in summaries if s['method'] == method and s['recall16px'] is not None]
        totals[method] = dict(**{k: counts[k] for k in fields}, visible_targets=visible,
            pooled_visible_recall16px=counts['correct16px'] / visible if visible else None,
            macro_window_recall16px=sum(recalls) / len(recalls) if recalls else None)
    guarded = totals[METHODS[2]]
    static = totals[METHODS[1]]
    original = totals[METHODS[0]]
    no_loss_to_static = not any(r['loss'] for r in paired if r['reference_method'] == METHODS[1])
    no_loss_to_original = not any(r['loss'] for r in paired if r['reference_method'] == METHODS[0])
    errors_not_increased = all(guarded[k] <= original[k] for k in ['wrong_position', 'output_when_not_observable'])
    guard_actions = sum(int(by[METHODS[1], g['sample_id']]['visible']) and not int(by[METHODS[2], g['sample_id']]['visible']) for g in labels)
    report = dict(status='completed_human_annotation_fixed_comparison', totals=totals,
        guard_lost_no_static_correct=no_loss_to_static, guard_lost_no_original_correct=no_loss_to_original,
        pipeline_strict_improvement_guard_passed=no_loss_to_static and no_loss_to_original and errors_not_increased and guarded['correct16px'] > original['correct16px'],
        guard_withheld_target_count=guard_actions, guard_discrimination_not_demonstrated_if_zero_actions=guard_actions == 0,
        phase_counts=dict(Counter(g['phase_status'] for g in labels)),
        visibility_counts=dict(Counter(g['truth_visibility'] for g in labels)),
        labels_sha256=sha(OUT / 'manual_sequence_labels.csv'), events_sha256=sha(OUT / 'annotation_events.jsonl'),
        predictions_sha256=sha(OUT / 'frozen_predictions.csv'), reservation_sha256=sha(OUT / 'reservation.json'),
        automatic_adoption=False, model_changed=False, canonical_tracking_changed=False,
        limitations='Two small same-match development sequences;selection fixed before predictions,prior exposure incomplete,nearby samples correlated. Human phase conditional,not automatic segmentation/external/full-rally performance.')
    write(OUT / 'per_frame_evaluation.csv', points)
    write(OUT / 'per_window_summary.csv', summaries)
    write(OUT / 'paired_changes.csv', paired)
    write(OUT / 'adjacent_sampled_pairs.csv', adjacent)
    snapshot = OUT / 'completed_review'
    snapshot.mkdir()
    (OUT / 'evaluation_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    for n in ['manual_sequence_labels.csv', 'initial_labels_snapshot.csv', 'annotation_events.jsonl', 'frozen_predictions.csv',
        'reservation.json', 'protocol.json', 'evaluation_report.json', 'per_frame_evaluation.csv', 'per_window_summary.csv', 'paired_changes.csv', 'adjacent_sampled_pairs.csv']:
        shutil.copy2(OUT / n, snapshot / n)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', action='store_true')
    main(parser.parse_args().status)
