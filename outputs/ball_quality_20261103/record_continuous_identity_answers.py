"""Preserve explicit target-identity answers without inventing ball centers."""
import datetime
import json
import shutil
from pathlib import Path

from integrate_044_reviewed_tracks import ROOT, read, load, write, sha

OUT = ROOT.parent / 'ball_quality_20261109/continuous_tile_gate_review'
FIXED = OUT.parent / 'frozen_tile_gate_confirmation'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    assert not (OUT / 'completed_identity_review.json').exists(), 'Preserve completed answers; amendments require new events.'
    manifest = load(OUT / 'focus_review_manifest.json')
    protocol = load(OUT / 'protocol.json')
    for table in [manifest['protected_input_hashes'], protocol['protected_input_hashes']]:
        assert all(sha(Path(p)) == h for p, h in table.items())
    for video in manifest['videos']:
        assert sha(Path(video['path'])) == video['sha256']
    initial = OUT / 'initial_focus_identity_labels.csv'
    labels_path = OUT / 'focus_identity_labels.csv'
    assert sha(initial) == sha(labels_path) == manifest['initial_labels_sha256']
    rows = read(labels_path)
    assert len(rows) == 3 and all(r['identity_status'] == r['review_status'] == 'unreviewed' for r in rows)
    predictions = {(r['window_id'], int(r['local_frame'])): r for r in read(OUT / 'continuous_predictions.csv')}
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    answers = {('D', 515): (0, '別物体'), ('D', 516): (0, '別物体'), ('B', 275): (1, '別物体')}
    events = []
    for r in rows:
        key = r['window_id'], int(r['local_frame'])
        item, answer = answers[key]
        prediction = predictions[key]
        assert all(r[k] == prediction[k] for k in ['source_frame', 'output_source', 'x_px', 'y_px'])
        old = dict(r)
        r.update(identity_status='confirmed_not_rally_ball', review_status='human_target_identity_review',
            reviewer_id='human_user', reviewed_at_utc=now,
            notes='Explicit answer: 別物体. Target identity only; no ball center, visibility or neighbouring-frame labels inferred.')
        events.append(dict(before=old, after=dict(r), request_call_id='call_407b39dd5c74451ea7af6df4cc11aec8',
            question_item=item, answer_exact=answer, evidence='explicit_user_question_reply',
            question='動画Dの黄色枠（local515・516）で、緑の丸はラリーで使っている実球を囲んでいますか？ 「実球／別物体／判断できない」で教えてください。' if item == 0 else
                '動画Bの黄色枠（local275）で、青の丸はラリーで使っている実球を囲んでいますか？ 「実球／別物体／判断できない」で教えてください。'))
    events_path = OUT / 'focus_identity_events.jsonl'
    assert not events_path.exists()
    events_path.write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in events), encoding='utf-8')
    temporary = labels_path.with_suffix('.pending.csv')
    write(temporary, rows)
    temporary.replace(labels_path)
    replay = {(r['window_id'], r['local_frame']): r for r in read(initial)}
    for e in [json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines()]:
        key = e['before']['window_id'], e['before']['local_frame']
        assert replay[key] == e['before']
        replay[key] = e['after']
    assert read(labels_path) == [replay[r['window_id'], r['local_frame']] for r in read(initial)]
    result = dict(status='three_target_outputs_human_confirmed_not_rally_ball', recorded_at_utc=now,
        request_call_id='call_407b39dd5c74451ea7af6df4cc11aec8',
        original_responses=[dict(question_item=0, answer='別物体', targets=['D515', 'D516']),
            dict(question_item=1, answer='別物体', targets=['B275'])],
        confirmed_targets=[{k: r[k] for k in ['window_id', 'local_frame', 'source_frame', 'output_source', 'x_px', 'y_px', 'identity_status']} for r in rows],
        confirmed_incorrect_output_frames=3, confirmed_added_incorrect_frames=2, confirmed_existing_incorrect_frames=1,
        distinct_reviewed_failure_segments=2,
        interpretation='D515/516 are adjacent observations of one reviewed reacquisition failure; B275 is one existing transient wrong target. Not3independent experiments.',
        scope='Identity of displayed target outputs only; no true ball center,physical ball absence,phase boundary or neighbouring-label propagation.',
        no_accuracy_estimate_for_724frames=True, no_accuracy_estimate_for_72additions=True,
        initial_labels_sha256=sha(initial), labels_sha256=sha(labels_path), events_sha256=sha(events_path),
        frozen_52target_report_unchanged=True, model_or_canonical_tracking_changed=False,
        script_sha256=sha(Path(__file__)))
    dump(OUT / 'completed_identity_review.json', result)
    decision = dict(status='frozen_supplement_not_selected_for_operational_use', date_jst='2026-11-09',
        reason='Continuous inspection confirmed new wrong-target outputs at D515/516. Sparse-anchor safety passed but is insufficient for operational use.',
        operational_adoption=False, candidate_head_sha256=sha(FIXED / 'frozen_training_head.npz'),
        observed_added_failures=[dict(window_id='D', local_frames=[515, 516], source_frames=[154441, 154442])],
        separate_existing_failure=dict(window_id='B', local_frame=275, source_frame=57488,
            inherited_by_supplement=True, limitation='Missing-only supplement cannot correct an already-present wrong output.'),
        previous_52target_results_retained=dict(original_correct=33, candidate_correct=40, visible_targets=45,
            result_scope='Known sparse same-match target positions only; these3newfailure frames are outside those52targets.'),
        training_or_threshold_changed=False, per_frame_exception_added=False, protected_tracking_changed=False,
        next='Diagnose single-frame component evidence versus temporal support using training-only development. Freeze a new rule before checking known failure cases; never call those checks independent validation. Reserve separate sequences/matches for evaluation.',
        evidence_sha256=sha(OUT / 'completed_identity_review.json'))
    dump(OUT / 'operational_decision.json', decision)
    snapshot = OUT / 'completed_identity_snapshot'
    snapshot.mkdir()
    names = ['initial_focus_identity_labels.csv', 'focus_identity_labels.csv', 'focus_identity_events.jsonl',
        'completed_identity_review.json', 'operational_decision.json', 'focus_review_manifest.json']
    for name in names:
        shutil.copy2(OUT / name, snapshot / name)
        assert sha(snapshot / name) == sha(OUT / name)
    assert sha(initial) == manifest['initial_labels_sha256']
    for table in [manifest['protected_input_hashes'], protocol['protected_input_hashes']]:
        assert all(sha(Path(p)) == h for p, h in table.items())
    dump(OUT / 'completed_identity_verification.json', dict(status='passed', explicit_question_answers=2,
        reviewed_target_frames=3, adjacent_Dframes_counted_as_one_failure_segment=True,
        history_replay_exact=True, initial_labels_preserved=True, frozen_inputs_preserved=True,
        snapshots={name: sha(snapshot / name) for name in names}, new_ball_center_labels=0,
        new_visibility_or_phase_labels=0, canonical_tracking_changed=False, model_changed=False))
    print(json.dumps(dict(status=result['status'], decision=decision['status'], confirmed_added_incorrect_frames=2,
        confirmed_existing_incorrect_frames=1, labelled_failure_segments=2, original_results_retained=True,
        ball_center_or_phase_labels_inferred=0), indent=2))


if __name__ == '__main__':
    main()
