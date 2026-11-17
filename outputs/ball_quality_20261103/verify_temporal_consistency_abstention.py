"""Reconcile fixed-rule decisions, coordinate preservation and human evidence."""
import json
import math
from collections import Counter
from pathlib import Path

from integrate_044_reviewed_tracks import ROOT, read, load, write, sha

BASE = ROOT.parent / 'ball_quality_20261109'
OUT = BASE / 'temporal_consistency_abstention'
CONT = BASE / 'continuous_tile_gate_review'


def main():
    protocol = load(OUT / 'protocol.json')
    training = load(OUT / 'training_decision.json')
    regression = load(OUT / 'regression_report.json')
    assert all(sha(Path(p)) == h for p, h in protocol['protected_input_hashes'].items())
    assert all(sha(Path(p)) == h for p, h in protocol['helper_hashes'].items())
    assert sha(OUT / 'executed_source.py') == protocol['script_sha256']
    assert all(sha(OUT / n) == h for n, h in training['artifacts'].items())
    assert all(sha(Path(p)) == h for p, h in load(OUT / 'training_input_hashes.json').items())
    initial = read(BASE / 'frozen_tile_gate_confirmation/training_predictions.csv')
    guarded = read(OUT / 'training_guarded_predictions.csv')
    diagnostic = read(OUT / 'training_guard_diagnostics.csv')
    assert len(initial) == len(guarded) == len(diagnostic) == 123
    assert training['total_withheld'] == 0 and initial == guarded
    assert training['previous_play']['correct16px'] == training['guarded_play']['correct16px'] == 59
    assert training['lost_correct_sample_ids'] == [] and training['target_numeric_match_count'] == 123
    raw = read(CONT / 'continuous_predictions.csv')
    new = read(OUT / 'continuous_guarded_predictions.csv')
    controls = read(OUT / 'continuous_additions_only_control.csv')
    ds = read(OUT / 'continuous_guard_diagnostics.csv')
    assert len(raw) == len(new) == len(controls) == len(ds) == 724
    withheld = []
    parameters = protocol['parameters']
    margin, rate = parameters['margin_source_px'], parameters['rate_px_per_source_frame']
    lookup = {(r['candidate_id'], int(r['local_frame'])): r for r in raw}
    for old, after, d in zip(raw, new, ds, strict=True):
        assert old['sample_id'] == after['sample_id'] == d['sample_id']
        if d['keep'] == 'True':
            assert old == after
            continue
        assert int(old['visible']) == 1 and int(after['visible']) == 0
        assert after['output_source'] == 'withheld_temporal_contradiction'
        assert d['reason'] == 'withhold_two_anchor_contradiction'
        t, a, b = int(old['local_frame']), int(d['anchor_a']), int(d['anchor_b'])
        assert 0 < abs(t - a) <= 8 and 0 < abs(t - b) <= 8 and abs(a - b) >= 2
        aa, bb = lookup[old['candidate_id'], a], lookup[old['candidate_id'], b]
        assert int(aa['original_visible']) == int(bb['original_visible']) == 1
        ax, bx = [float(aa[k]) for k in ['original_x_px', 'original_y_px']], [float(bb[k]) for k in ['original_x_px', 'original_y_px']]
        target = [float(old[k]) for k in ['x_px', 'y_px']]
        assert math.dist(ax, bx) <= margin + rate * abs(a - b)
        assert math.dist(target, ax) > margin + rate * abs(t - a)
        assert math.dist(target, bx) > margin + rate * abs(t - b)
        assert abs(math.dist(ax, bx) - float(d['pair_distance_px'])) < 1e-10
        assert abs(math.dist(target, ax) - float(d['distance_a_px'])) < 1e-10
        assert abs(math.dist(target, bx) - float(d['distance_b_px'])) < 1e-10
        withheld.append((old['window_id'], old['local_frame']))
    known = read(CONT / 'focus_identity_labels.csv')
    known_keys = {(g['window_id'], g['local_frame']) for g in known if g['identity_status'] == 'confirmed_not_rally_ball'}
    assert set(withheld) == known_keys == {('B', '275'), ('D', '515'), ('D', '516')}
    assert regression['known_wrong_frames_withheld'] == regression['continuous_withheld'] == 3
    assert regression['sparse52_development_counts']['before_guard'] == regression['sparse52_development_counts']['after_guard']
    assert regression['sparse52_development_counts']['after_guard']['correct16px'] == 40
    assert regression['lost_correct_sparse_sample_ids'] == []
    by_raw = {(r['window_id'], r['local_frame']): r for r in raw}
    by_new = {(r['window_id'], r['local_frame']): r for r in new}
    by_control = {(r['window_id'], r['local_frame']): r for r in controls}
    ablations = []
    for name, table in [('static_supplement', by_raw), ('guard_additions_only', by_control), ('guard_all_outputs', by_new)]:
        ablations.append(dict(method=name, B275_wrong_output_retained=int(table['B', '275']['visible']),
            D515_wrong_output_retained=int(table['D', '515']['visible']), D516_wrong_output_retained=int(table['D', '516']['visible']),
            known_diagnostic_not_accuracy=True))
    write(OUT / 'known_failure_ablation.csv', ablations)
    decision = dict(status='development_candidate_only_requires_new_data',
        training_safety_passed=True, training_withheld_targets=0,
        training_evidence_caveat='No target was withheld on training123,so this check establishes no empirical training-negative discrimination.',
        known_failure_frames_withheld=3, known_failure_segments=2, prior_correct_targets_lost=0,
        returned_ball_positions_at_failure_frames=0, external_validation=False, operational_adoption=False,
        canonical_tracking_changed=False, next='Freeze fresh sequence selection before predictions and human center annotation; evaluate withholding losses and errors together.')
    (OUT / 'development_decision.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
    verification = dict(status='passed', training_rows_identical_to_unguarded=123,
        training_correct_play_targets_retained=59, continuous_rows_verified=724,
        unchanged_continuous_rows=721, withheld_human_confirmed_wrong_rows=3,
        all_retained_coordinates_exact=True, no_recovered_ball_center_claim=True,
        sparse_visible_correct_retained=40, independent_external_validation=False,
        maximum_additional_rule_anchor_window=8, maximum_full_original_model_support_future_frames=16,
        protected_inputs_unchanged=True, script_sha256=sha(Path(__file__)),
        artifacts={p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'verification.json'})
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in verification.items() if k != 'artifacts'}, indent=2))


if __name__ == '__main__':
    main()
