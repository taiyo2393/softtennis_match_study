"""Reuse reviewed serve validity, preserving uncertain boundary intervals."""
import csv
import json

from audit_play_starts import SOURCE,WORKSPACE,HANDOFF,OUT,BASE,read,sha
from render_play_start_pilot import render_pilot


def main():
    cid='match1_motion_pilot_016';wid='match1_rally_window_044'
    inventory=read(OUT/'ball_pilot_start_inventory.csv')
    entry=next(r for r in inventory if r['ball_pilot_id']==cid)
    assert entry['rally_window_candidate_id']==wid and entry['serve_validity_status']=='human_confirmed'
    lo=int(entry['effective_contact_candidate_first_source_frame']);hi=int(entry['effective_contact_candidate_last_source_frame'])
    ends=read(HANDOFF/'event_review_ranges.csv')
    end=next(r for r in ends if r['candidate_id']==wid and r['event']=='point_end')
    elo=int(end['first_possible_source_frame']);ehi=int(end['last_possible_source_frame'])
    path=BASE/cid/'balltrack_predictions.csv';raw=read(path);output=[]
    for r in raw:
        frame=int(r['source_frame'])
        if frame<lo:scope='before_effective_serve_includes_first_fault_and_preparation'
        elif frame<=hi:scope='serve_contact_uncertainty'
        elif frame<elo:scope='candidate_main_play_window'
        elif frame<=ehi:scope='point_end_uncertainty'
        else:scope='after_candidate_point_end'
        output.append(dict(r,point_context_id='match1_point_candidate_044',effective_serve_attempt=2,
            serve_validity='human_confirmed_valid',scope_status=scope,in_candidate_main_play_window=int(scope=='candidate_main_play_window'),
            contact_time_status='AI_interval_not_exact_frame_confirmed',point_end_time_status='AI_interval_not_exact_frame_confirmed',
            target_ball_identity_status='not_newly_verified_by_this_scope_export'))
    with (OUT/'044_start_scoped_raw_ball_observations.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    from collections import Counter
    counts=dict(Counter(r['scope_status'] for r in output))
    included=[r for r in output if r['in_candidate_main_play_window']]
    first_attempts=read(HANDOFF/'044_serve_attempt_review.csv')
    # Scope tags must not change the model observations or relabel the fault as a new point.
    assert all(all(new[k]==old[k] for k in old) for old,new in zip(raw,output))
    assert any(r['attempt_number']=='1' and r['attempt_outcome']=='fault' and r['outcome_review_status']=='human_confirmed' for r in first_attempts)
    result=dict(ball_pilot_id=cid,rally_window_candidate_id=wid,
        validity_evidence='044_serve_attempt_review.csv and human_review_log.csv, previously confirmed by user',
        valid_serve_contact_source_frame_interval=[lo,hi],point_end_source_frame_interval=[elo,ehi],
        candidate_main_play_source_frames=[int(included[0]['source_frame']),int(included[-1]['source_frame'])],
        candidate_main_play_local_frames=[int(included[0]['local_frame']),int(included[-1]['local_frame'])],
        counts=counts,raw_observations_unchanged=True,first_fault_preserved_as_same_point_separate_attempt=True,
        source_csv_sha256=sha(path),boundary_intervals_still_provisional=True,
        warning='Scope export, not a new start detector, identity/position evaluation or independent confirmation of exact event frames.')
    (OUT/'044_scope_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    render_pilot(cid,650,900,'044_effective_start')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
