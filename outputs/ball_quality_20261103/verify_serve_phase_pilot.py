"""Check failure-sensitive invariants; not independent detection accuracy."""
import json
from pathlib import Path
import numpy as np
from segment_serve_phases import OUT,phase_filter,context_series
from repair_start_pose_crops import contained
from extract_serve_start_features import sha

def event(t):
    return dict(person_track_id=t,raised_first_local_frame=t,raised_last_local_frame=t+4,
        contact_candidate_first_local_frame=t,contact_candidate_last_local_frame=t+4)

def series(quiet_times,end=200):
    return [dict(local_frame=t,four_player_quiet=t in quiet_times,motion_by_height=[.01]*4 if t in quiet_times else [None]*4) for t in range(0,end+1,2)]

def run(events,rows):
    td={p['person_track_id']:[dict(local_frame=p['raised_first_local_frame'],foot=[0.,0.])] for p in events}
    return phase_filter(events,rows,10.,np.eye(3),td)

def main():
    checks={}
    # Missing measurements never make quiet evidence; a recent ready state is
    # explicitly recorded as old evidence, and expires rather than lasting forever.
    missing=[dict(local_frame=t,people=[],side_counts=[0,0]) for t in range(0,50,2)]
    s,_=context_series(missing,10.);assert not any(r['four_player_quiet'] for r in s)
    checks['missing_four_players_never_quiet']=True
    short=run([event(12)],series(set(range(0,8,2))))
    assert short['accepted_candidates'][0]['four_player_quiet'] is False
    assert short['accepted_candidates'][0]['quiet_evidence_age_sec']>0
    long=run([event(30)],series(set(range(0,8,2))))
    assert not long['accepted_candidates'];checks['ready_hold_is_explicit_and_expires']=True
    # There is no return requirement: a short attempt/ace can still start.
    # Arbitrarily long time alone cannot release the activity lock.
    locked=run([event(12),event(150)],series(set(range(0,12,2))))
    assert len(locked['accepted_candidates'])==1 and locked['candidates'][1]['decision']=='activity_lock'
    checks['no_return_required_and_no_timeout_only_unlock']=True
    quiet=set(range(0,12,2))|set(range(30,56,2))
    retry=run([event(12),event(56)],series(quiet))
    assert len(retry['accepted_candidates'])==2
    assert all(p['outcome']=='unknown' and p['point_id'] is None for p in retry['candidates'])
    checks['quiet_reset_allows_candidate_but_never_infers_fault_or_point_id']=True
    for cid in ['016','002']:
        d=OUT/f'match1_motion_pilot_{cid}'
        features=json.loads((d/'features.json').read_text());p=json.loads((d/'phases.json').read_text())
        assert all(contained(person) for f in features['frames'] for person in f['people'])
        assert p['phase_protocol_sha256']==sha(OUT/'phase_protocol.json')
        assert p['feature_sha256']==sha(d/'features.json')
        assert p['labels_read'] is False and p['serve_outcome_inferred'] is False
    checks['complete_crop_boxes_same_protocol_and_unchanged_feature_inputs']=True
    # A full-court server suppressed in v2 is actually recovered, not merely removed.
    f=json.loads((OUT/'match1_motion_pilot_002/features.json').read_text())['frames'][350]
    assert any(p['origin']=='court' and 1250<(p['box'][0]+p['box'][2])/2<1500 and p['box'][3]>800 for p in f['people'])
    checks['002_complete_near_server_recovered_at_frame700']=True
    (OUT/'verification.json').write_text(json.dumps(dict(checks=checks,independent_accuracy_evaluation=False),indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
