"""Append explicit user rejection of463/465, preserving first4review events."""
import copy,json
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,read,load,write,sha

PARENT=ROOT/'ball_040_object_swap_review'
OUT=PARENT/'round2/user_reviewed'
NEW=[463,465]
ALL=[463,464,465,467,470,554]
ID='20261103_040_neighbour463465_not_ball'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def position(r):return tuple(float(r[k]) for k in ['x_px','y_px','visible'])
def extend(rows):
    output=[]
    for t,old in enumerate(rows):
        r=dict(old,review_snapshot_id='040_six_rejected_candidates_round2')
        if t in NEW:
            assert int(old['visible'])==1
            r.update(x_px=0,y_px=0,visible=0,confidence=0,position_source='human_rejected_candidate_no_verified_replacement',
                ball_candidate_review='user_confirmed_is_not_ball_at_displayed_location',human_review_id=ID,
                rejected_candidate_x_px=old['x_px'],rejected_candidate_y_px=old['y_px'],
                precise_center_reviewed_here=0,interpolated_here=0,physical_ball_visibility_reviewed_here=0)
        else:assert all(str(r[k])==str(old[k]) for k in old)
        output.append(r)
    assert [t for t,(a,b) in enumerate(zip(rows,output,strict=True)) if position(a)!=position(b)]==NEW
    return output

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'review.json').exists()
    registry_path=ROOT/'reviewed_ball_overrides.json';before=load(registry_path);registry=copy.deepcopy(before)
    entry=registry['clips']['match1_motion_pilot_015'];previous_entry=copy.deepcopy(entry)
    assert entry['confirmed_false_raw_local_frames']==[464,467,470,554]
    baseline_path=Path(entry['reviewed_output']);experimental_path=Path(entry['experimental_temporal_output'])
    assert sha(baseline_path)==entry['reviewed_output_sha256'] and sha(experimental_path)==entry['experimental_temporal_output_sha256']
    baseline=read(baseline_path);experimental=read(experimental_path);assert len(baseline)==len(experimental)==570
    prior_video=load(PARENT/'render_report.json');assert sha(Path(prior_video['video']))==prior_video['video_sha256']
    queue={int(r['local_frame']):r for r in read(PARENT/'adjacent_review_queue.csv')};assert set(queue)==set(NEW)
    unchanged=[ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',
        ROOT/'balltrack_manual_finetune_pilot/validation_labels.csv',ROOT/'ball_inplay_challenge/manual_center_labels.csv',
        Path(entry['original_baseline']),baseline_path,experimental_path,PARENT/'user_reviewed/review.json']
    protected={str(p):sha(p) for p in unchanged}
    records=read(PARENT/'user_reviewed/candidate_review.csv');new_records=[]
    for t in NEW:
        a,b=baseline[t],experimental[t];q=queue[t]
        assert position(a)==position(b) and str(b['source_frame'])==q['source_frame'] and float(b['x_px'])==float(q['x_px']) and float(b['y_px'])==float(q['y_px'])
        new_records.append(dict(local_frame=t,source_frame=int(b['source_frame']),displayed_x_px=b['x_px'],displayed_y_px=b['y_px'],
            candidate_is_ball=0,reviewer='human_user',human_review_id=ID,threshold030_component_count='',
            existing_selection_status=b['temporal_selection_status'],physical_ball_visibility='not_established',true_ball_center_x_px='',true_ball_center_y_px='',
            comparison_video_time_sec=2*t/59.94005994005994))
    combined=sorted(records+new_records,key=lambda r:int(r['local_frame']))
    review=dict(review_id=ID,recorded_date='2026-11-03',reviewer='human_user',user_answer='463・465の橙はボールじゃない',
        ball_pilot_id='match1_motion_pilot_015',rally_candidate_id='match1_rally_window_040',
        source_video=prior_video['video'],source_video_sha256=prior_video['video_sha256'],
        newly_rejected_local_frames=NEW,cumulative_rejected_local_frames=ALL,
        scope='Only the2amber candidates463/465 now explicitly rejected by user. No neighbour propagation, true center or physical-absence label.',
        prior_review_record=entry['review_record'],prior_registry_sha256=sha(registry_path),
        prior_baseline_sha256=sha(baseline_path),prior_experimental_sha256=sha(experimental_path),
        algorithm_changed=False,training_labels_changed=False)
    dump(OUT/'review.json',review);dump(OUT/'previous_reviewed_ball_registry.json',before)
    write(OUT/'new_candidate_review.csv',new_records);write(OUT/'candidate_review.csv',combined)
    new_base=extend(baseline);new_exp=extend(experimental)
    target_base=OUT/'baseline_balltrack_predictions.csv';target_exp=OUT/'temporal_balltrack_predictions.csv'
    write(target_base,new_base);write(target_exp,new_exp)
    for rows in [new_base,new_exp]:
        assert all(int(rows[t]['visible'])==0 and rows[t]['human_review_id'] for t in ALL)
        assert all(rows[t]['human_review_id']==ID for t in NEW)
    entry.update(reviewed_output=str(target_base.resolve()),reviewed_output_sha256=sha(target_base),
        experimental_temporal_output=str(target_exp.resolve()),experimental_temporal_output_sha256=sha(target_exp),
        reviewed_changed_local_frames=ALL,confirmed_false_raw_local_frames=ALL,review_id=ID,review_record=str((OUT/'review.json').resolve()),
        review_records=previous_entry['review_records']+[str((OUT/'review.json').resolve())],
        previous_reviewed_output=previous_entry['reviewed_output'],previous_reviewed_output_sha256=previous_entry['reviewed_output_sha256'],
        previous_experimental_temporal_output=previous_entry['experimental_temporal_output'],
        previous_experimental_temporal_output_sha256=previous_entry['experimental_temporal_output_sha256'])
    assert all(entry[k]==previous_entry[k] for k in ['automatic_filter_adopted','automatic_temporal_selection_adopted','precise_center_or_full_track_accuracy_established','physical_ball_visibility_at_rejected_frames'])
    assert all(registry['clips'][k]==v for k,v in before['clips'].items() if k!='match1_motion_pilot_015')
    assert all(sha(Path(p))==h for p,h in protected.items())
    pending=registry_path.with_suffix('.pending.json');dump(pending,registry);pending.replace(registry_path)
    report=dict(status='two_explicit_neighbour_rejections_appended',new_rejected_local_frames=NEW,rejected_local_frames=ALL,
        additional_position_changes_per_branch=2,total_rejected_positions=6,remaining564positions_unchanged_from_original_branch=True,
        baseline_output=str(target_base.resolve()),baseline_sha256=sha(target_base),experimental_output=str(target_exp.resolve()),experimental_sha256=sha(target_exp),
        previous_review_and_outputs_unchanged=True,other_registry_entries_unchanged=True,physical_visibility_labels_added=0,center_groundtruth_added=0,
        interpolated_positions=0,algorithm_changes=False,automatic_temporal_adoption=False,
        registry_previous_sha256=review['prior_registry_sha256'],registry_new_sha256=sha(registry_path),protected_input_hashes=protected)
    dump(OUT/'verification.json',report);print(json.dumps({k:v for k,v in report.items() if k!='protected_input_hashes'},indent=2))

if __name__=='__main__':main()
