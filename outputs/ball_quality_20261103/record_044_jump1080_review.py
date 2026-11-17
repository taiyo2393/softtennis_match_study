"""Apply only the explicitly reviewed orange/blue candidates near1080."""
import json
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,read,write,load,sha

OUT=ROOT/'ball_044_jump1080_audit'
POSITIVE=[1081,1082]
NEGATIVE=[1082,1084,1086,1087,1088]

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    dest=OUT/'user_reviewed';dest.mkdir(exist_ok=True)
    assert not (dest/'review.json').exists(),'Do not overwrite a recorded review.'
    rp=ROOT/'reviewed_ball_overrides.json';registry=load(rp);cid='match1_motion_pilot_016';previous=registry['clips'][cid]
    source=Path(previous['reviewed_output']);before=read(source)
    diagnosis=load(OUT/'diagnosis.json');video=OUT/'jump1080_review_h264.mp4'
    assert sha(source)==previous['reviewed_output_sha256']==diagnosis['input_sha256']
    assert sha(video)==diagnosis['video_sha256']
    assert NEGATIVE==diagnosis['suspect_output_frames'] and POSITIVE==diagnosis['weak_alternative_frames']
    components=read(OUT/'candidate_components.csv')
    blue={t:next(c for c in components if int(c['local_frame'])==t and int(c['rank'])==2) for t in POSITIVE}
    person=load(ROOT/'reviewed_person_overrides.json')['rallies']['match1_rally_window_044']
    protected={str(source):sha(source),str(Path(person['reviewed_output'])):person['reviewed_output_sha256']}
    protected.update(load(ROOT/'play_start_audit/inventory_summary.json')['source_hashes'])
    assert all(sha(Path(p))==h for p,h in protected.items())
    review_id='20261103_044_jump1080_five_false_two_real'
    changed=sorted(set(POSITIVE+NEGATIVE));result=[]
    for t,r in enumerate(before):
        q=dict(r)
        # At1082 reject the orange raw candidate AND retain the distinct blue component.
        if t in POSITIVE:
            c=blue[t]
            q.update(x_px=c['x_px'],y_px=c['y_px'],visible='1',confidence='',
                position_source='observed_heatmap_component_user_confirmed_ball',component_rank=c['rank'],
                component_peak=c['peak'],coordinate_extractor='half_peak_weighted_component_centroid',
                ball_candidate_review='user_confirmed_is_ball',human_review_id=review_id)
        elif t in NEGATIVE:
            q.update(x_px='0',y_px='0',visible='0',confidence='',component_rank='',component_peak='',
                position_source='no_ball_observation_user_rejected_raw_candidate',coordinate_extractor='none_raw_candidate_rejected',
                ball_candidate_review='user_confirmed_raw_candidate_is_not_ball',human_review_id=review_id)
        result.append(q)
    target=dest/'predictions'/cid/'balltrack_predictions.csv';target.parent.mkdir(parents=True,exist_ok=True)
    write(target,result);saved=read(target)
    assert len(saved)==len(before)==1169
    assert [t for t,(a,b) in enumerate(zip(before,saved,strict=True)) if a!=b]==changed
    for t,(a,b) in enumerate(zip(before,saved,strict=True)):
        assert all(a[k]==b[k] for k in ['raw_x_px','raw_y_px','raw_confidence','raw_position_source'])
        if t not in changed:assert a==b
        else:assert b['human_review_id']==review_id and b['interpolated_here']=='0'
    for t in previous['reviewed_changed_local_frames']:assert saved[t]==before[t]
    for t in POSITIVE:
        assert saved[t]['visible']=='1' and saved[t]['x_px']==blue[t]['x_px'] and saved[t]['y_px']==blue[t]['y_px']
    for t in set(NEGATIVE)-set(POSITIVE):assert saved[t]['visible']=='0'
    assert saved[1080]==before[1080] and saved[1096]==before[1096]
    assert all(sha(Path(p))==h for p,h in protected.items())
    review=dict(review_id=review_id,recorded_date='2026-11-03',reviewer='human_user',
        rally_candidate_id='match1_rally_window_044',ball_pilot_id=cid,
        user_answer='橙の丸は一つも実球を囲めていない．青は合ってる',
        scope='Exactly five orange raw candidates rejected and two blue components confirmed. Cyan outputs not newly reviewed.',
        confirmed_ball_candidate_local_frames=POSITIVE,confirmed_false_raw_local_frames=NEGATIVE,
        confirmed_observations=[dict(local_frame=t,source_frame=int(before[t]['source_frame']),component=blue[t]) for t in POSITIVE],
        rejected_raw_observations=[dict(local_frame=t,source_frame=int(before[t]['source_frame']),
            rejected_xy_px=[float(before[t]['x_px']),float(before[t]['y_px'])]) for t in NEGATIVE],
        frame1082_note='The orange raw location is false; the blue rank2 component is a real ball. Positive and negative labels refer to different candidates.',
        physical_ball_visibility_at_rejected_frames={str(t):'ball_confirmed_at_alternative_component' if t==1082 else 'not_established' for t in NEGATIVE},
        video_sha256=sha(video),candidate_components_sha256=sha(OUT/'candidate_components.csv'),
        source_sha256=sha(source),previous_review_record=previous['review_record'],
        precise_center_groundtruth_created=False,independent_accuracy_evaluation=False)
    dump(dest/'review.json',review);dump(dest/'previous_reviewed_reference.json',previous)
    bundle_path=ROOT/'reviewed_tracking_bundles.json'
    dump(dest/'previous_tracking_bundles.json',load(bundle_path))
    updated=dict(previous,reviewed_output=str(target.resolve()),reviewed_output_sha256=sha(target),
        previous_reviewed_output=str(source),reviewed_changed_local_frames=sorted(set(previous['reviewed_changed_local_frames']+changed)),
        confirmed_ball_candidate_local_frames=sorted(set(previous['confirmed_ball_candidate_local_frames']+POSITIVE)),
        confirmed_false_raw_local_frames=sorted(set(previous['confirmed_false_raw_local_frames']+NEGATIVE)),
        review_id=review_id,review_record=str((dest/'review.json').resolve()),
        review_records=previous['review_records']+[str((dest/'review.json').resolve())],
        physical_ball_visibility_at_rejected_frames='See individual review records: at1082 an alternative ball is confirmed; other rejected-frame visibility not established',
        candidate_label_note='A frame can have both a false raw location and a different confirmed ball component.')
    registry['clips'][cid]=updated
    routes_path=ROOT/'serve_phase_fixed_application/tracking_entry_routes.json';routes=load(routes_path)
    entry=routes['primary_tracking_development_input']
    entry['preferred_reviewed_ball_input']='../ball_044_jump1080_audit/user_reviewed/predictions/match1_motion_pilot_016/balltrack_predictions.csv'
    entry['reviewed_ball_changes_local_frames']=updated['reviewed_changed_local_frames']
    entry['reviewed_ball_false_raw_local_frames']=updated['confirmed_false_raw_local_frames']
    dump(rp,registry);dump(routes_path,routes)
    decision=dict(status='user_review_applied',review_record=str((dest/'review.json').resolve()),
        reviewed_output=str(target.resolve()),confirmed_ball_candidates=POSITIVE,rejected_raw_candidates=NEGATIVE,
        global_filter_adopted=False,interpolation_applied=False)
    dump(OUT/'decision.json',decision)
    report=dict(changed_local_frames=changed,confirmed_real_component_frames=POSITIVE,rejected_raw_frames=NEGATIVE,
        added_output_frames=[1081],replaced_output_frames=[1082],withheld_output_frames=[1084,1086,1087,1088],
        unchanged_rows=len(before)-len(changed),all_previous_confirmations_preserved=True,
        original_raw_history_preserved=True,person_inputs_and_source_annotations_unchanged=True,
        full_clip_rows=len(saved),main_play_ball_output_frames=sum(int(saved[t]['visible']) for t in range(739,1141)),
        input_main_play_ball_output_frames=sum(int(before[t]['visible']) for t in range(739,1141)),
        reviewed_output_sha256=sha(target),interpolated_positions=0,independent_accuracy_measured=False)
    dump(dest/'verification.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
