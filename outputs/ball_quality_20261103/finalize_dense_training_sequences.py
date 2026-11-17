"""Seal human sequence annotations and export training-only supervision, no fitting."""
import collections
import datetime
import json
import math
import shutil
from pathlib import Path

from integrate_044_reviewed_tracks import ROOT, read, load, write, sha

BASE = ROOT.parent / 'ball_quality_20261109'
OUT = BASE / 'dense_training_sequences'
DONE = OUT / 'completed_review'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def check_state(row):
    assert row['phase_status'] in {'play', 'preparation_or_ended', 'uncertain'}
    assert row['truth_visibility'] in {'visible', 'not_visible', 'uncertain'}
    if row['truth_visibility'] == 'visible':
        x, y = float(row['truth_x_px']), float(row['truth_y_px'])
        assert math.isfinite(x) and math.isfinite(y) and 0 <= x < 1920 and 0 <= y < 1080
    else:
        assert row['truth_x_px'] == row['truth_y_px'] == ''


def main():
    assert not DONE.exists(), 'Preserve completed human review; never overwrite.'
    manifest = load(OUT / 'manifest.json')
    assert all(sha(Path(p)) == h for p, h in manifest['protected_input_hashes'].items())
    assert all(sha(OUT / n) == h for n, h in manifest['frozen_artifact_hashes'].items())
    assert load(OUT / 'verification.json')['status'] == load(OUT / 'http_verification.json')['status'] == 'passed'
    paths = [OUT / n for n in ['manual_sequence_labels.csv', 'initial_labels_snapshot.csv', 'annotation_events.jsonl',
        'manifest.json', 'protocol.json', 'annotation_cases.json', 'query_model_inputs.csv', 'windows.json',
        'source_image_hashes.json', 'verification.json', 'http_verification.json', 'app_verification.json', 'serve_review.py', 'annotation.html']]
    protected = {str(p): sha(p) for p in paths}
    initial, labels = read(OUT / 'initial_labels_snapshot.csv'), read(OUT / 'manual_sequence_labels.csv')
    assert len(initial) == len(labels) == 124 and len({r['sample_id'] for r in labels}) == 124
    state = {r['case_id']: dict(r) for r in initial}
    cases = {r['case_id']: r for r in load(OUT / 'annotation_cases.json')}
    events = [json.loads(line) for line in (OUT / 'annotation_events.jsonl').read_text(encoding='utf-8').splitlines()]
    allowed_changes = {'phase_status', 'truth_visibility', 'truth_x_px', 'truth_y_px', 'reviewer_id', 'annotated_at_utc', 'notes', 'review_status', 'center_source'}
    for e in events:
        cid = e['case_id']
        assert cid in state and state[cid] == e['before'] and e['model_predictions_hidden'] is True
        assert set(e['after']) == set(state[cid])
        assert all(e['before'][k] == e['after'][k] for k in state[cid] if k not in allowed_changes)
        check_state(e['after'])
        assert e['after']['reviewer_id'] == 'human_user' and e['after']['annotated_at_utc']
        datetime.datetime.fromisoformat(e['after']['annotated_at_utc'])
        state[cid] = e['after']
    assert [state[r['case_id']] for r in initial] == labels
    assert all(r['review_status'] != 'unreviewed' for r in labels), 'Human review incomplete.'
    changed = {e['case_id'] for e in events}
    assert changed == {r['case_id'] for r in initial if r['review_status'] == 'unreviewed'}
    assert len(changed) == 120
    for row in labels:
        check_state(row)
        c = cases[row['case_id']]
        assert c['sample_id'] == row['sample_id'] and str(c['source_frame']) == row['source_frame']
        assert sha(Path(c['source_image'])) == c['source_image_sha256']
        assert sha(Path(c['context_video'])) == c['context_video_sha256']
        if c['center_reused']:
            assert row == next(r for r in initial if r['case_id'] == row['case_id'])
        else:
            assert row['review_status'] == 'human_frame_review' and row['center_source'] == 'new_human_frame_review'
    image_hashes = load(OUT / 'source_image_hashes.json')
    assert all(sha(Path(p)) == h for p, h in image_hashes.items())
    summaries, pairs = [], []
    for wid in 'ABCD':
        group = sorted([r for r in labels if r['window_id'] == wid], key=lambda r: int(r['local_frame']))
        assert len(group) == 31
        counts = collections.Counter((r['phase_status'], r['truth_visibility']) for r in group)
        summaries.append(dict(window_id=wid, candidate_id=group[0]['candidate_id'], targets=len(group),
            play_visible=counts['play', 'visible'], play_not_visible=counts['play', 'not_visible'], play_uncertain=counts['play', 'uncertain'],
            nonplay=sum(n for (phase, _), n in counts.items() if phase == 'preparation_or_ended'), phase_uncertain=sum(n for (phase, _), n in counts.items() if phase == 'uncertain')))
        for left, right in zip(group, group[1:]):
            assert int(right['source_frame']) - int(left['source_frame']) == 2
            observable = left['phase_status'] == right['phase_status'] == 'play' and left['truth_visibility'] == right['truth_visibility'] == 'visible'
            distance = math.hypot(float(right['truth_x_px']) - float(left['truth_x_px']), float(right['truth_y_px']) - float(left['truth_y_px'])) if observable else ''
            pairs.append(dict(window_id=wid, left_case_id=left['case_id'], right_case_id=right['case_id'], source_frame_step=2,
                both_human_centers_observable=int(observable), displacement_px=distance, interpretation='image_plane_displacement_not_3d_speed;no_gap_bridging'))
    old_path = ROOT / 'balltrack_expanded_centers_pilot/expanded_train_labels.csv'
    phase_path = BASE / 'training_phase_scope_review/phase_labels.csv'
    old = read(old_path)
    phases = {r['sample_id']: r for r in read(phase_path)}
    queries = {r['sample_id']: r for r in read(OUT / 'query_model_inputs.csv')}
    merged = {}
    for r in old:
        assert r['finetune_role'] == 'train'
        merged[r['sample_id']] = dict(sample_id=r['sample_id'], candidate_id=r['candidate_id'], local_frame=r['local_frame'], source_frame=r['source_frame'],
            finetune_role='train', split_group=r['candidate_id'], phase_status=phases[r['sample_id']]['phase_status'], truth_visibility=r['truth_visibility'],
            truth_x_px=r['truth_x_px'], truth_y_px=r['truth_y_px'], input_frames_json=r['input_frames_json'], input_frame_sha256_json=r['input_frame_sha256_json'],
            median_path=r['median_path'], median_sha256=r['median_sha256'], center_source_file=str(old_path), center_source_file_sha256=sha(old_path),
            phase_source_file=str(phase_path), phase_source_file_sha256=sha(phase_path), annotation_origin='previous_human_training_labels',
            clock='legacy_edited_match1', training_eligible='', exclusion_reason='')
    reused = 0
    for r in labels:
        sid = r['sample_id']
        if sid in merged:
            assert cases[r['case_id']]['center_reused']
            assert all(merged[sid][k] == r[k] for k in ['candidate_id', 'local_frame', 'source_frame', 'phase_status', 'truth_visibility', 'truth_x_px', 'truth_y_px'])
            reused += 1
            continue
        q = queries[sid]
        merged[sid] = dict(sample_id=sid, candidate_id=r['candidate_id'], local_frame=r['local_frame'], source_frame=r['source_frame'],
            finetune_role='train', split_group=r['candidate_id'], phase_status=r['phase_status'], truth_visibility=r['truth_visibility'],
            truth_x_px=r['truth_x_px'], truth_y_px=r['truth_y_px'], input_frames_json=q['input_frames_json'], input_frame_sha256_json=q['input_frame_sha256_json'],
            median_path=q['median_path'], median_sha256=q['median_sha256'], center_source_file=str(DONE / 'manual_sequence_labels.csv'), center_source_file_sha256=protected[str(OUT / 'manual_sequence_labels.csv')],
            phase_source_file=str(DONE / 'manual_sequence_labels.csv'), phase_source_file_sha256=protected[str(OUT / 'manual_sequence_labels.csv')],
            annotation_origin='new_dense_human_training_review', clock='legacy_edited_match1', training_eligible='', exclusion_reason='')
    assert reused == 4 and len(merged) == 243
    confirmation = read(BASE / 'temporal_guard_next_sequences/completed_review/manual_sequence_labels.csv') + read(ROOT.parent / 'ball_quality_20261104/completed_review/manual_sequence_labels.csv')
    assert {r['candidate_id'] for r in confirmation}.isdisjoint(r['candidate_id'] for r in merged.values())
    checked_images = {}
    for row in merged.values():
        eligible = row['phase_status'] == 'play' and row['truth_visibility'] == 'visible'
        row['training_eligible'] = int(eligible)
        row['exclusion_reason'] = '' if eligible else ('not_main_rally' if row['phase_status'] != 'play' else 'center_unobservable_or_uncertain_not_absence_target')
        images, hashes = json.loads(row['input_frames_json']), json.loads(row['input_frame_sha256_json'])
        assert len(images) == len(hashes) == 4
        assert [int(Path(p).stem) for p in images] == list(range(int(row['local_frame'])-3, int(row['local_frame'])+1))
        for p, h in zip(images, hashes, strict=True):
            actual = checked_images.setdefault(p, sha(Path(p)))
            assert actual == h
        assert sha(Path(row['median_path'])) == row['median_sha256']
        if eligible: check_state(row)
    all_rows = sorted(merged.values(), key=lambda r: int(r['source_frame']))
    positive = [r for r in all_rows if r['training_eligible']]
    assert len(positive) == 181 and sum(r['annotation_origin'] == 'new_dense_human_training_review' for r in positive) == 113
    assert all(sha(Path(p)) == h for p, h in protected.items()), 'Review changed during finalization; take no partial snapshot.'
    DONE.mkdir()
    # Keep the preparation verifier separate from the finalization verifier.
    snapshot_names = {str(p): 'preparation_verification.json' if p.name == 'verification.json' else p.name for p in paths}
    for p in paths: shutil.copy2(p, DONE / snapshot_names[str(p)])
    shutil.copy2(Path(__file__), DONE / 'executed_finalize_source.py')
    write(DONE / 'per_window_summary.csv', summaries)
    write(DONE / 'adjacent_human_pairs.csv', pairs)
    write(DONE / 'merged_training_observation_states.csv', all_rows)
    write(DONE / 'training_visible_centers.csv', positive)
    result = dict(status='completed_human_annotations_sealed_training_export_ready', completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reviewed_targets=124, new_targets=120, reused_targets=4, journal_events=len(events), unique_journal_targets=len(changed),
        journal_resaves=len(events)-len(changed), all_targets_play=True, dense_visible_centers=117, dense_unobservable_centers=7,
        per_window=summaries, adjacent_sampled_pairs=len(pairs), pairs_with_two_observable_centers=sum(p['both_human_centers_observable'] for p in pairs),
        previous_play_visible_centers=68, additional_unique_play_visible_centers=113, merged_play_visible_centers=181,
        merged_observation_states=243, merged_play_unobservable_states=16, merged_nonplay_states=46, unique_training_clips=8,
        raw_unobservable_labels_preserved=True, unobservable_converted_to_absence_target=False,
        confirmation_clips_excluded=True, model_inference=False, model_training=False, accuracy_computed=False, operational_adoption=False,
        independent_validation=False, limitations='Four ~1-second sampled training windows in the same known match. Adjacent frames are dependent; no full-rally or unseen-match accuracy established.',
        integrity=dict(history_replay_exact=True, input_images_checked=len(checked_images), dense_source_images_checked=len(image_hashes),
            frozen_inputs_unchanged=True, human_labels_unchanged=True),
        source_snapshot_hashes={snapshot_names[p]: h for p, h in protected.items()},
        export_hashes={n: sha(DONE / n) for n in ['per_window_summary.csv', 'adjacent_human_pairs.csv', 'merged_training_observation_states.csv', 'training_visible_centers.csv']},
        script_sha256=sha(Path(__file__)))
    dump(DONE / 'result.json', result)
    assert all(sha(Path(p)) == h and sha(DONE / snapshot_names[p]) == h for p, h in protected.items())
    assert all(sha(Path(p)) == h for p, h in manifest['protected_input_hashes'].items())
    # Re-read the written exports; no prediction coordinates are exported as human labels.
    reread = read(DONE / 'training_visible_centers.csv')
    assert len(reread) == 181 and len({r['sample_id'] for r in reread}) == 181
    assert all(r['phase_status'] == 'play' and r['truth_visibility'] == 'visible' and r['finetune_role'] == 'train' for r in reread)
    assert not any(any(k.startswith('prediction') for k in r) for r in reread)
    dump(DONE / 'verification.json', dict(status='passed', history_events_verified=len(events), reviewed_targets_verified=124,
        unique_training_centers_verified=181, snapshots_byte_identical=True, exports_reloaded=True, model_predictions_not_used_as_truth=True,
        protected_inputs_unchanged=True, script_sha256=sha(Path(__file__))))
    print(json.dumps({k: v for k, v in result.items() if k not in ['source_snapshot_hashes', 'export_hashes']}, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
