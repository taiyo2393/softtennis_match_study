"""Persist the user's five real-ball confirmations and one false detection."""
import json
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,read,write,load
from extract_serve_start_features import sha

OUT=ROOT/'ball_044_branch_recovery'
POSITIVE=[891,892,893,960,970];NEGATIVE=[968]

def main():
    dest=OUT/'user_reviewed';dest.mkdir(exist_ok=True);assert not (dest/'review.json').exists()
    rp=ROOT/'reviewed_ball_overrides.json';registry=load(rp);cid='match1_motion_pilot_016';previous=registry['clips'][cid]
    source=Path(previous['reviewed_output']);before=read(source);assert sha(source)==previous['reviewed_output_sha256']
    proposals=OUT/'branch_proposals.csv';render=load(OUT/'render_report.json');assert sha(proposals)==render['proposals_sha256']
    proposed={int(r['local_frame']):r for r in read(proposals)}
    assert sorted(t for t,r in proposed.items() if r['branch_proposal_action']!='unchanged_reviewed_input')==sorted(POSITIVE+NEGATIVE)
    review_id='20261103_044_ball_branch_five_real_one_false'
    review=dict(review_id=review_id,recorded_date='2026-11-03',reviewer='human_user',
        rally_candidate_id='match1_rally_window_044',ball_pilot_id=cid,
        questions=['緑の丸で示した5候補は、実球を囲めていますか？','frame968の橙の丸は、球ですか、それとも誤検出ですか？'],
        answers=['緑の丸で示した5候補は、実球を囲めている','frame968の橙の丸は、誤検出'],
        scope='Five displayed green components enclose the ball; orange raw968 is not a ball. No precise-center or full-track confirmation.',
        confirmed_ball_candidate_local_frames=POSITIVE,confirmed_false_raw_local_frames=NEGATIVE,
        confirmed_observations=[dict(local_frame=t,source_frame=int(before[t]['source_frame']),
            component_xy_px=[float(proposed[t]['branch_proposed_x_px']),float(proposed[t]['branch_proposed_y_px'])],
            component_peak=float(proposed[t]['branch_component_peak'])) for t in POSITIVE],
        rejected_raw_observations=[dict(local_frame=t,source_frame=int(before[t]['source_frame']),
            rejected_xy_px=[float(before[t]['x_px']),float(before[t]['y_px'])]) for t in NEGATIVE],
        physical_ball_visibility_at968='not_established',precise_center_groundtruth_created=False,
        previous_review_record=previous['review_record'],video_sha256=sha(OUT/'branch_review_h264.mp4'),
        proposal_sha256=sha(proposals),independent_accuracy_evaluation=False)
    (dest/'review.json').write_text(json.dumps(review,indent=2,ensure_ascii=False),encoding='utf-8')
    (dest/'previous_reviewed_reference.json').write_text(json.dumps(previous,indent=2),encoding='utf-8')
    result=[]
    for t,r in enumerate(before):
        q=dict(r)
        if t in POSITIVE:
            p=proposed[t]
            q.update(x_px=p['branch_proposed_x_px'],y_px=p['branch_proposed_y_px'],visible='1',confidence='',
                position_source='observed_heatmap_component_user_confirmed_ball',component_rank=p['branch_component_rank'],
                component_peak=p['branch_component_peak'],coordinate_extractor='half_peak_weighted_component_centroid',
                ball_candidate_review='user_confirmed_is_ball',human_review_id=review_id)
        elif t in NEGATIVE:
            q.update(x_px='0',y_px='0',visible='0',confidence='',component_rank='',component_peak='',
                position_source='no_ball_observation_user_rejected_raw_candidate',coordinate_extractor='none_raw_candidate_rejected',
                ball_candidate_review='user_confirmed_raw_candidate_is_not_ball',human_review_id=review_id)
        result.append(q)
    folder=dest/'predictions'/cid;folder.mkdir(parents=True,exist_ok=True);target=folder/'balltrack_predictions.csv';write(target,result)
    saved=read(target);changed=[]
    for t,(a,b) in enumerate(zip(before,saved,strict=True)):
        if a!=b:changed.append(t)
        assert all(a[k]==b[k] for k in ['raw_x_px','raw_y_px','raw_confidence','raw_position_source'])
        if t not in POSITIVE+NEGATIVE:assert a==b
        else:assert b['human_review_id']==review_id
    assert changed==sorted(POSITIVE+NEGATIVE) and len(saved)==1169
    assert saved[968]['visible']=='0' and before[968]['visible']=='1'
    assert saved[891]['visible']=='1' and before[891]['visible']=='0'
    for t in previous['reviewed_changed_local_frames']:assert saved[t]==before[t]
    assert sha(source)==previous['reviewed_output_sha256']
    person=load(ROOT/'reviewed_person_overrides.json')['rallies']['match1_rally_window_044']
    assert sha(Path(person['reviewed_output']))==person['reviewed_output_sha256']
    for name,h in load(ROOT/'play_start_audit/inventory_summary.json')['source_hashes'].items():assert sha(Path(name))==h
    updated=dict(previous,reviewed_output=str(target.resolve()),reviewed_output_sha256=sha(target),
        previous_reviewed_output=str(source),reviewed_changed_local_frames=sorted(previous['reviewed_changed_local_frames']+POSITIVE+NEGATIVE),
        confirmed_ball_candidate_local_frames=sorted(previous.get('confirmed_ball_candidate_local_frames',previous['reviewed_changed_local_frames'])+POSITIVE),
        confirmed_false_raw_local_frames=NEGATIVE,review_id=review_id,review_record=str((dest/'review.json').resolve()),
        review_records=previous.get('review_records',[previous['review_record']])+[str((dest/'review.json').resolve())],
        physical_ball_visibility_at_rejected_frames='not_established')
    registry['clips'][cid]=updated;rp.write_text(json.dumps(registry,indent=2),encoding='utf-8')
    routes_path=ROOT/'serve_phase_fixed_application/tracking_entry_routes.json';routes=load(routes_path)
    entry=routes['primary_tracking_development_input']
    entry['preferred_reviewed_ball_input']='../ball_044_branch_recovery/user_reviewed/predictions/match1_motion_pilot_016/balltrack_predictions.csv'
    entry['reviewed_ball_changes_local_frames']=updated['reviewed_changed_local_frames']
    entry['reviewed_ball_false_raw_local_frames']=NEGATIVE
    routes_path.write_text(json.dumps(routes,indent=2,ensure_ascii=False),encoding='utf-8')
    decision=load(OUT/'decision.json');decision.update(status='five_ball_candidates_and_raw968_rejection_human_confirmed_applied',
        human_ball_candidate_review='user_confirmed_five_green_components_are_ball',
        human_false_detection_review='user_confirmed_raw968_is_false',
        reviewed_output=str(target.resolve()),review_record=str((dest/'review.json').resolve()),
        next='Refresh integrated clip using the reviewed outputs. Preserve remaining gaps and unvalidated position precision.')
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    report=dict(changed_frames=changed,positive_candidates=POSITIVE,rejected_raw=NEGATIVE,
        original_raw_observations_preserved=True,other1163_rows_unchanged=True,old_three_confirmations_preserved=True,
        player_inputs_and_source_review_tables_unchanged=True,interpolated_positions=0,
        full_clip_rows=len(saved),main_play_ball_output_frames=sum(int(saved[t]['visible']) for t in range(739,1141)),
        physical_ball_visibility_at968='unknown',independent_accuracy_measured=False,reviewed_output_sha256=sha(target))
    (dest/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
