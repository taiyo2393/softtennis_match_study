"""Separate legacy rally-label readiness from position/outcome readiness.

No new human labels are inferred. Absence of an export is not visual absence.
"""
import csv
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent

def read(name):
    with (OUT/name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

rallies = read('rally_label_review.csv')
snapshots = defaultdict(list)
for row in read('serve_phase_position_snapshot_candidates.csv'):
    snapshots[row['candidate_id']].append(row)

result = []
for rally in rallies:
    cid = rally['candidate_id']
    sample = snapshots[cid]
    label_ready = (
        rally['boundary_review_status'] == 'human_confirmed'
        and rally['winner_review_status'] == 'human_confirmed'
        and rally['team_side_mapping_status'] == 'human_confirmed'
        and all(rally[key] for key in ('rally_id','start_source_frame',
                    'end_source_frame','winner_team_id','winner_camera_side','reviewer'))
    )
    exported = len(sample) == 4 and len({r['player_slot'] for r in sample}) == 4
    if sample:
        assert exported, cid
        assert len({r['sample_source_frame'] for r in sample}) == 1, cid
    # Existing export explicitly says last *possible* contact; it is not an
    # exact 30-frame offset from a confirmed contact. Never promote it here.
    timing_ready = bool(sample) and all(
        r['sampling_rule'] == '30_frames_after_confirmed_serve_contact'
        for r in sample)
    ids_ready = bool(sample) and all(
        r['identity_review_status'] == 'human_confirmed' for r in sample)
    points_ready = bool(sample) and all(
        r['position_review_status'] == 'human_confirmed_ground_point' for r in sample)
    calibration_ready = bool(sample) and all(
        r['calibration_status'] == 'independent_transfer_validation_passed' for r in sample)
    gates = {
        'rally_labels': label_ready, 'snapshot_export': exported,
        'sampling_time': timing_ready, 'four_player_ids': ids_ready,
        'four_ground_points': points_ready, 'calibration_transfer': calibration_ready,
    }
    result.append(dict(
        candidate_id=cid,
        legacy_analysis_eligible=rally['analysis_eligible'],
        rally_label_ready=str(label_ready).lower(),
        snapshot_export_present=str(exported).lower(),
        snapshot_source_frame=sample[0]['sample_source_frame'] if sample else '',
        snapshot_timing_confirmed=str(timing_ready).lower(),
        four_player_ids_confirmed=str(ids_ready).lower(),
        four_ground_points_confirmed=str(points_ready).lower(),
        calibration_transfer_confirmed=str(calibration_ready).lower(),
        snapshot_outcome_ready=str(all(gates.values())).lower(),
        trajectory_outcome_status='not_evaluated_against_independent_ground_truth',
        pending_requirements=';'.join(k for k,v in gates.items() if not v),
        note='Missing export means not prepared; not proof players are invisible.'
    ))

with (OUT/'analysis_readiness_audit.csv').open('w',encoding='utf-8-sig',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(result[0]))
    writer.writeheader()
    writer.writerows(result)
print('Candidates:',len(result))
print('Snapshot exports:',sum(r['snapshot_export_present']=='true' for r in result))
print('Rally-label ready:',sum(r['rally_label_ready']=='true' for r in result))
print('Snapshot/outcome ready:',sum(r['snapshot_outcome_ready']=='true' for r in result))
print('Trajectory accuracy: not evaluated')
