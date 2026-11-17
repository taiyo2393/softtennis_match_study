"""Record4 explicit false-candidate reports; do not label physical absence.

Canonical baseline=original raw minus exactly4 reviewed false locations.
Experimental branch=previous temporal proposal minus the same4 locations.
No algorithm/weight/threshold tuning or neighbouring-label propagation.
"""
import json,copy
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,read,load,write,sha

OUT=ROOT/'ball_040_object_swap_review/user_reviewed'
FRAMES=[464,467,470,554]
REVIEW_ID='20261103_040_four_other_object_candidates'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def same_xy(a,b):return all(float(a[k])==float(b[k]) for k in ['x_px','y_px','visible'])
def curate(rows,kind):
    result=[]
    for t,old in enumerate(rows):
        r=dict(old)
        if t in FRAMES:
            r.update(x_px=0,y_px=0,visible=0,confidence=0,position_source='human_rejected_candidate_no_verified_replacement')
        r.update(ball_candidate_review='user_confirmed_is_not_ball_at_displayed_location' if t in FRAMES else old.get('ball_candidate_review','not_reviewed_by_this_confirmation'),
            human_review_id=REVIEW_ID if t in FRAMES else old.get('human_review_id',''),
            precise_center_reviewed_here=0 if t in FRAMES else old.get('precise_center_reviewed_here',0),
            interpolated_here=0 if t in FRAMES else old.get('interpolated_here',0),
            rejected_candidate_x_px=old['x_px'] if t in FRAMES else '',rejected_candidate_y_px=old['y_px'] if t in FRAMES else '',
            review_branch=kind,physical_ball_visibility_reviewed_here=0)
        if t not in FRAMES:assert all(str(r[k])==str(old[k]) for k in old)
        result.append(r)
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'review.json').exists(),'Preserve human review event.'
    registry_path=ROOT/'reviewed_ball_overrides.json';registry=load(registry_path);previous=copy.deepcopy(registry)
    assert 'match1_motion_pilot_015' not in registry['clips']
    raw_path=ROOT.parent/'tracking_quality_20261102/ball_aligned/match1_motion_pilot_015/balltrack_predictions.csv'
    branch=ROOT/'temporal_rally_reviews_v2/match1_motion_pilot_015';proposal_path=branch/'balltrack_predictions_proposed.csv'
    verification=load(branch/'verification.json')
    assert sha(proposal_path)==verification['proposal_csv_sha256'] and sha(Path(verification['video']))==verification['sha256']
    protected=load(branch/'verified_inputs.json')['protected_input_hashes']
    assert all(sha(Path(p))==h for p,h in protected.items())
    raw=read(raw_path);proposal=read(proposal_path);cache=load(branch/'full_scope_component_cache.json')['match1_motion_pilot_015']
    assert len(raw)==len(proposal)==570
    records=[]
    for t in FRAMES:
        a,b=raw[t],proposal[t];assert same_xy(a,b) and int(b['visible'])==1 and len(cache[str(t)])==1
        records.append(dict(local_frame=t,source_frame=int(b['source_frame']),displayed_x_px=b['x_px'],displayed_y_px=b['y_px'],
            candidate_is_ball=0,reviewer='human_user',human_review_id=REVIEW_ID,threshold030_component_count=1,
            existing_selection_status=b['temporal_selection_status'],physical_ball_visibility='not_established',
            true_ball_center_x_px='',true_ball_center_y_px='',comparison_video_time_sec=2*t/59.94005994005994))
    review=dict(review_id=REVIEW_ID,recorded_date='2026-11-03',reviewer='human_user',user_answer='464,467,470,554は別の物体に移ってる',
        ball_pilot_id='match1_motion_pilot_015',rally_candidate_id='match1_rally_window_040',
        source_video=verification['video'],source_video_sha256=verification['sha256'],
        scope='Only4 displayed locations at local464/467/470/554 are rejected. No judgement of adjacent frames, physical ball absence or true centers.',
        confirmed_false_raw_local_frames=FRAMES,confirmed_false_proposal_local_frames=FRAMES,
        findings='All4 have exactly1 threshold0.3 component, shared by original and proposed output; not regressions introduced by the4 new temporal rerankings.',
        correction='Exclude only the4 reviewed false candidates; retain missing with physical visibility unknown, no synthetic replacement.',
        canonical_branch='Original raw baseline with4human exclusions, without adopting automatic temporal reranking',
        experimental_branch='Temporal proposal with same4human exclusions; other proposed points stay unreviewed',
        previous_registry_sha256=sha(registry_path),original_raw_sha256=sha(raw_path),original_proposal_sha256=sha(proposal_path),
        independent_full_track_accuracy=False,training_labels_updated=False)
    dump(OUT/'review.json',review);write(OUT/'candidate_review.csv',records);dump(OUT/'previous_reviewed_ball_registry.json',previous)
    baseline=curate(raw,'original_plus_four_human_exclusions');experimental=curate(proposal,'temporal_plus_four_human_exclusions')
    baseline_path=OUT/'baseline_balltrack_predictions.csv';experimental_path=OUT/'temporal_balltrack_predictions.csv'
    write(baseline_path,baseline);write(experimental_path,experimental)
    for old,new in [(raw,baseline),(proposal,experimental)]:
        assert [i for i,(a,b) in enumerate(zip(old,new,strict=True)) if not same_xy(a,b)]==FRAMES
        assert all(not int(new[t]['visible']) and new[t]['true_ball_center_x_px']=='' if 'true_ball_center_x_px' in new[t] else not int(new[t]['visible']) for t in FRAMES)
    registry['clips']['match1_motion_pilot_015']=dict(rally_candidate_id='match1_rally_window_040',
        reviewed_output=str(baseline_path.resolve()),reviewed_output_sha256=sha(baseline_path),original_baseline=str(raw_path),
        reviewed_changed_local_frames=FRAMES,confirmed_ball_candidate_local_frames=[],confirmed_false_raw_local_frames=FRAMES,
        review_id=REVIEW_ID,review_record=str((OUT/'review.json').resolve()),review_records=[str((OUT/'review.json').resolve())],
        automatic_filter_adopted=False,automatic_temporal_selection_adopted=False,precise_center_or_full_track_accuracy_established=False,
        physical_ball_visibility_at_rejected_frames='not_established',
        candidate_label_note='Only displayed raw/proposed coordinates rejected; no absence or neighbouring-frame labels. Experimental temporal branch remains separate.',
        experimental_temporal_output=str(experimental_path.resolve()),experimental_temporal_output_sha256=sha(experimental_path))
    assert all(registry['clips'][k]==v for k,v in previous['clips'].items())
    # Authorised annotation changes this registry only; verify every other old input.
    assert all(sha(Path(p))==h for p,h in protected.items() if Path(p)!=registry_path)
    pending=registry_path.with_suffix('.pending.json');dump(pending,registry);pending.replace(registry_path)
    check=load(registry_path);assert all(check['clips'][k]==v for k,v in previous['clips'].items())
    report=dict(status='four_explicit_false_candidates_recorded_and_excluded',rejected_local_frames=FRAMES,
        baseline_rows=len(baseline),experimental_rows=len(experimental),modified_position_rows_per_branch=4,other566positions_preserved=True,
        candidates_per_rejected_frame=[len(cache[str(t)]) for t in FRAMES],physical_visibility_labels_added=0,center_groundtruth_added=0,
        baseline_output=str(baseline_path.resolve()),baseline_sha256=sha(baseline_path),experimental_output=str(experimental_path.resolve()),experimental_sha256=sha(experimental_path),
        old_raw_and_proposal_unchanged=True,old044and008registry_entries_unchanged=True,person_and_integrated_registries_unchanged=True,
        registry_previous_sha256=review['previous_registry_sha256'],registry_new_sha256=sha(registry_path),
        prior_experiment_hash_notice='Old protocols keep their old registry hash; archived previous registry reproduces it. No historical results edited.',
        algorithm_changes=False,automatic_temporal_adoption=False)
    dump(OUT/'verification.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
