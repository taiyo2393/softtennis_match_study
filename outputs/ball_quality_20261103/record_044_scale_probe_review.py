"""Record candidate-level tiled-inference review, preserving all earlier evidence."""
import json
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha

OUT=ROOT/'ball_044_scale_probe'
POSITIVE=[(t,'A') for t in range(1083,1090)]+[(1091,'B')]
NEGATIVE=[(t,'A') for t in range(1090,1094)]

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    dest=OUT/'user_reviewed';dest.mkdir(exist_ok=True)
    assert not (dest/'review.json').exists(),'Keep existing review snapshots.'
    rp=ROOT/'reviewed_ball_overrides.json';registry=load(rp);cid='match1_motion_pilot_016'
    previous=registry['clips'][cid];source=Path(previous['reviewed_output']);before=read(source)
    render=load(OUT/'render_report.json');video=OUT/'scale_probe_review_h264.mp4';cp=OUT/'displayed_candidates.csv'
    assert sha(source)==previous['reviewed_output_sha256']
    assert sha(video)==render['video_sha256'] and sha(cp)==render['displayed_candidates_sha256']
    candidates={(int(c['local_frame']),c['display_label']):c for c in read(cp)}
    assert set(POSITIVE+NEGATIVE)=={k for k in candidates if 1083<=k[0]<=1093}
    person=load(ROOT/'reviewed_person_overrides.json')['rallies']['match1_rally_window_044']
    protected={str(source):sha(source),str(Path(person['reviewed_output'])):person['reviewed_output_sha256']}
    protected.update(load(ROOT/'play_start_audit/inventory_summary.json')['source_hashes'])
    assert all(sha(Path(p))==h for p,h in protected.items())
    review_id='20261103_044_tiled_eight_real_four_false'
    accepted={t:candidates[(t,label)] for t,label in POSITIVE};result=[]
    for t,r in enumerate(before):
        q=dict(r)
        if t in accepted:
            c=accepted[t]
            assert int(r['visible'])==0,'Only fill confirmed missing observations.'
            q.update(x_px=c['x_px'],y_px=c['y_px'],visible='1',confidence='',
                position_source='observed_tiled_heatmap_component_user_confirmed_ball',
                component_rank=c['rank'],component_peak=c['peak'],
                coordinate_extractor='half_peak_weighted_component_centroid_tile_to_full_image',
                inference_origin='same_frame_fixed_2x_tiled_inference',
                ball_candidate_review='user_confirmed_is_ball',human_review_id=review_id)
        # Rejected experimental components were never in the authoritative output.
        # Keep its missing state and older metadata unchanged at1090/1092/1093.
        result.append(q)
    target=dest/'predictions'/cid/'balltrack_predictions.csv';target.parent.mkdir(parents=True,exist_ok=True)
    write(target,result);saved=read(target);changed=sorted(accepted)
    assert len(saved)==len(before)==1169
    assert [t for t,(a,b) in enumerate(zip(before,saved,strict=True)) if a!=b]==changed
    for t,(a,b) in enumerate(zip(before,saved,strict=True)):
        assert all(a[k]==b[k] for k in ['raw_x_px','raw_y_px','raw_confidence','raw_position_source'])
        if t not in accepted:assert a==b
        else:
            c=accepted[t]
            assert b['x_px']==c['x_px'] and b['y_px']==c['y_px'] and b['interpolated_here']=='0'
    for t in previous['confirmed_ball_candidate_local_frames']:assert saved[t]==before[t]
    for t in (1090,1092,1093):assert saved[t]['visible']=='0' and saved[t]==before[t]
    assert saved[1091]['x_px']==candidates[(1091,'B')]['x_px']!=candidates[(1091,'A')]['x_px']
    assert all(sha(Path(p))==h for p,h in protected.items())
    labelled=[]
    for label,keys in [('user_confirmed_is_ball',POSITIVE),('user_confirmed_is_not_ball',NEGATIVE)]:
        for t,name in keys:
            c=candidates[(t,name)]
            labelled.append(dict(c,source_frame=int(before[t]['source_frame']),review=label,
                human_review_id=review_id,component_inference_view='fixed_2x_tiles',
                component_tile_xywh=[int(c['tile_x']),int(c['tile_y']),960,540]))
    review=dict(review_id=review_id,recorded_date='2026-11-03',reviewer='human_user',
        rally_candidate_id='match1_rally_window_044',ball_pilot_id=cid,
        user_answers=['1083〜1089の青丸は実球を囲めている','1090〜1093は違う．','1091はBが実球'],
        interpretation='The explicit1091-B statement qualifies the broader1090-1093 rejection: reject1091-A, retain1091-B.',
        scope='Displayed tiled candidates1083-1093 only. Other tiled candidates remain unreviewed.',
        confirmed_ball_candidate_local_frames=changed,
        confirmed_observations=[c for c in labelled if c['review']=='user_confirmed_is_ball'],
        rejected_tiled_components=[c for c in labelled if c['review']=='user_confirmed_is_not_ball'],
        physical_ball_visibility_at_rejected_frames={'1090':'not_established','1091':'ball_confirmed_at_B','1092':'not_established','1093':'not_established'},
        video_sha256=sha(video),displayed_candidates_sha256=sha(cp),input_sha256=sha(source),
        previous_review_record=previous['review_record'],precise_center_groundtruth_created=False,
        independent_accuracy_evaluation=False,global_filter_adopted=False)
    dump(dest/'review.json',review);dump(dest/'previous_reviewed_reference.json',previous)
    dump(dest/'previous_tracking_bundles.json',load(ROOT/'reviewed_tracking_bundles.json'))
    write(dest/'candidate_review.csv',labelled)
    rejected_refs=[dict(local_frame=c['local_frame'],display_label=c['display_label'],
        inference_view='fixed_2x_tiles',tile_index=c['tile_index'],component_rank=c['rank'],
        x_px=c['x_px'],y_px=c['y_px'],review_id=review_id) for c in review['rejected_tiled_components']]
    updated=dict(previous,reviewed_output=str(target.resolve()),reviewed_output_sha256=sha(target),
        previous_reviewed_output=str(source),reviewed_changed_local_frames=sorted(set(previous['reviewed_changed_local_frames']+changed)),
        confirmed_ball_candidate_local_frames=sorted(set(previous['confirmed_ball_candidate_local_frames']+changed)),
        confirmed_false_tiled_candidates=previous.get('confirmed_false_tiled_candidates',[])+rejected_refs,
        review_id=review_id,review_record=str((dest/'review.json').resolve()),
        review_records=previous['review_records']+[str((dest/'review.json').resolve())],
        physical_ball_visibility_at_rejected_frames='Candidate-specific; see individual review records. Some rejected raw/tiled locations coexist with confirmed alternative ball components.',
        candidate_label_note='Raw and tiled candidate rejections are separate. A frame can contain a false candidate and a different confirmed ball component.')
    registry['clips'][cid]=updated
    routes_path=ROOT/'serve_phase_fixed_application/tracking_entry_routes.json';routes=load(routes_path)
    entry=routes['primary_tracking_development_input']
    entry['preferred_reviewed_ball_input']='../ball_044_scale_probe/user_reviewed/predictions/match1_motion_pilot_016/balltrack_predictions.csv'
    entry['reviewed_ball_changes_local_frames']=updated['reviewed_changed_local_frames']
    entry['reviewed_ball_false_tiled_candidates']=rejected_refs
    dump(rp,registry);dump(routes_path,routes)
    decision=load(OUT/'decision.json')
    decision.update(status='eight_real_tiled_components_applied_four_false_candidates_recorded',
        automatic_adoption=False,replace_reviewed_output='Only eight confirmed missing observations added; no global tiled replacement',
        review_record=str((dest/'review.json').resolve()),reviewed_output=str(target.resolve()),
        reason='Tiling supplies real missing observations but also false candidates1090-A,1091-A,1092-A,1093-A. Known970 is missed; strongest candidate alone is insufficient.',
        next='Refresh the integration; retain unconfirmed gaps and freeze before testing another rally.')
    dump(OUT/'decision.json',decision)
    report=dict(changed_local_frames=changed,added_observations=8,reviewed_false_tiled_candidates=4,
        unchanged_rows=len(before)-len(changed),earlier_positive_observations_unchanged=True,
        earlier_false_raw_labels_retained=True,original_raw_history_preserved=True,
        person_inputs_and_source_annotations_unchanged=True,full_clip_rows=len(saved),
        main_play_ball_output_frames=sum(int(saved[t]['visible']) for t in range(739,1141)),
        input_main_play_ball_output_frames=sum(int(before[t]['visible']) for t in range(739,1141)),
        reviewed_output_sha256=sha(target),interpolated_positions=0,independent_accuracy_measured=False)
    dump(dest/'verification.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
