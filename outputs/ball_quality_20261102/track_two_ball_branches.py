"""Two candidate branches with explicit seed provenance and collision abstention."""
import csv
import json
import numpy as np

from track_seeded_ball_branch import associate,render
from extract_multiball_candidates import DEST,ROOT,SOURCE,read,sha


def main():
    assert not (DEST/'two_ball_branches.csv').exists()
    candidates=read(DEST/'candidates.csv');protocol=json.loads((DEST/'protocol.json').read_text())
    start=protocol['start_frame'];end=protocol['end_frame_inclusive']
    review=json.loads((ROOT/'temporal_guard/suppression_audit_20261103/user_reviewed/review.json').read_text(encoding='utf-8'))
    seed_a=(float(review['raw_model_x_px']),float(review['raw_model_y_px']))
    a,events_a=associate(candidates,318,seed_a,start,end)
    # B is model-initialized, NOT a pixel-location annotation from the user.
    choices=[c for c in candidates if int(c['local_frame'])==310 and int(c['rank'])!=int(a[310]['rank'])
             and float(c['peak'])>=.3 and np.hypot(float(c['x_px'])-float(a[310]['x_px']),float(c['y_px'])-float(a[310]['y_px']))>128]
    assert choices
    seed_b=max(choices,key=lambda c:float(c['peak']))
    b,events_b=associate(candidates,310,(float(seed_b['x_px']),float(seed_b['y_px'])),start,end)
    collisions=[]
    for t in sorted(set(a)&set(b)):
        if int(a[t]['rank'])==int(b[t]['rank']):
            collisions.append(t);del a[t];del b[t]
    baseline_path=ROOT.parent/'tracking_quality_20261102/ball_aligned/match1_motion_pilot_002/balltrack_predictions.csv'
    baseline=read(baseline_path)
    queue=read(SOURCE/'data/segments/match1_motion_prediction_pilot_queue.csv')
    window=next(r for r in queue if r['candidate_id']=='match1_motion_pilot_002')['play_window_id']
    output=[]
    for branch,selected,seed,origin in (('A',a,318,'user_confirmed_ball_candidate'),('B',b,310,'model_remaining_component')):
        for t in range(start,end+1):
            c=selected.get(t);raw=baseline[t]
            output.append(dict(candidate_id='match1_motion_pilot_002',play_window_id=window,local_frame=t,
                source_frame=raw['source_frame'],source_timestamp_sec=raw['source_timestamp_sec'],branch=branch,
                visible=int(c is not None),x_px=c['x_px'] if c else 0,y_px=c['y_px'] if c else 0,peak=c['peak'] if c else 0,
                component_rank=c['rank'] if c else '',position_source='observed_heatmap_component' if c else 'missing_no_assignment',
                seed_local_frame=seed,seed_origin=origin,identity_status='association_candidate_not_full_track_reviewed',
                next_served_ball_identity='unresolved',phase='not_independently_annotated',interpolated=0))
    with (DEST/'two_ball_branches.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    user_review=dict(recorded_date='2026-11-03',question='上の002の動画で、緑の丸は左手前の選手の球を追い、右側の別の球へ乗り換えていませんか？',
        answer='球が二球ある',accepted_scope='scene contains two balls',
        full_branch_identity_confirmed=False,precise_positions_annotated=False,
        note='The answer is not treated as confirmation of every green marker or of which ball will be served.')
    (DEST/'user_two_ball_review.json').write_text(json.dumps(user_review,indent=2,ensure_ascii=False),encoding='utf-8')
    video=render(a,candidates,baseline,318,start,end,second_selected=b)
    for row in output:
        if int(row['visible']):
            matches=[c for c in candidates if int(c['local_frame'])==row['local_frame'] and int(c['rank'])==int(row['component_rank'])]
            assert len(matches)==1 and all(str(row[k])==str(matches[0][k]) for k in ('x_px','y_px','peak'))
    assert sha(baseline_path)==protocol['baseline_sha256']
    result=dict(frames=end-start+1,branch_A_observed_frames=len(a),branch_B_observed_frames=len(b),
        both_observed_frames=len(set(a)&set(b)),candidate_assignment_collisions=collisions,
        association_events=dict(A=events_a,B=events_b),seed_A=review,seed_B=seed_b,
        user_confirmation_scope='two balls in scene only, not full trajectories',
        all_positions_are_retained_model_components=True,interpolated_positions=0,old_baseline_unchanged=True,
        next_served_ball_unresolved=True,video=video,
        warning='Observation counts are not identity accuracy. This is a seeded development pilot, not a validated automatic multiball tracker.')
    (DEST/'two_ball_report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('seed_A','seed_B')},indent=2))


if __name__=='__main__':main()
