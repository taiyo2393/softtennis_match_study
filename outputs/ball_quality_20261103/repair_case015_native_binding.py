"""Assisted initialization repair; preserve the automatic015 failure unchanged."""
import sys,csv,json,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import cv2
ROOT=Path(__file__).resolve().parent;CASE=ROOT.parent/'ball_quality_20261110/cohort_tracking/case_015';OUT=CASE/'AI_native_binding'
sys.path.insert(0,str(ROOT.parent/'tracking_quality_20261102_stage10'))
import gap_pair_tracking as far
from build_rally_analysis_prototype import pair_features,summarize_prefix
from verify_player_prefix_expansion import calculate_frame,calculate_summary,close
SLOTS=['near_front','near_rear','far_rear','far_front']
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def write(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
    p=load(CASE/'protocol.json');result=load(CASE/'case_result.json');designpath=CASE/'identity_diagnosis/seed_design.json';design=load(designpath)
    assert load(CASE/'AI_visual_findings.json')['known_wrong_person_assignment']
    assert design['native_detector_raw_id']=='3' and design['target_slot']=='near_front' and design['seed_source_frame']==result['initialized_frame']==38548
    assert sha(Path(design['seed_image']))==design['seed_image_sha256']
    rawpath=CASE/'crop_runs/augmented_detections_015.csv';assert sha(rawpath)==design['native_detections_sha256'];raw=read(rawpath);baseline=read(CASE/'four_player_path.csv')
    first,last=p['context_first_source_frame'],p['context_last_source_frame'];init=result['initialized_frame'];native=[r for r in raw if r['track_id']=='3' and init<=int(r['source_frame'])<=last]
    assert len(native)==653 and len({r['source_frame'] for r in native})==653 and all(r['candidate_origin']=='original_full_frame' for r in native)
    assert {int(r['source_frame']) for r in raw if r['track_id']=='3' and 38541<=int(r['source_frame'])<=38548}==set(range(38541,38549))
    old=CASE/'tracking_quality_20261102_stage4/players_015_temporal';summary=load(old/'summary.json');event=load(old/'events.json')[0]
    assert event['seed_ids']['near_front']==1
    protected={str(n):sha(n) for n in [CASE/'protocol.json',CASE/'case_result.json',CASE/'four_player_path.csv',CASE/'verification.json',CASE/'AI_visual_findings.json',CASE/'pre_outcome_covariates.json',designpath,rawpath,old/'summary.json',old/'events.json',Path(far.__file__),Path(__file__)]}
    OUT.mkdir();dump(OUT/'seed_design.json',design)
    q=dict(p);q.update(derived_variant='AI_native_N1_binding_with_retained_far_rerun',AI_identity_binding_added=True,fully_automatic_result=False,full_automatic_baseline_valid=False,algorithm_parameters_changed=False,detector_or_association_numeric_parameters_changed=False,source_of_assignment='AI bound observed original raw3 to N1 at original initialization time; N2 unchanged. Far search recomputed with corrected near exclusions and corrected initialization negative set.',original_automatic_protocol=str(CASE/'protocol.json'),protected_input_hashes=protected);dump(OUT/'protocol.json',q)
    kept=[r for r in baseline if r['player_slot']=='near_rear'];assert len(kept)==653
    corrected=[dict(r,player_slot='near_front',position_source='detected_box',identity_status='AI_native_track_bound_candidate_not_GT',identity_source='AI_seed_F38548_raw3_native_continuity',identity_review_status='AI_image_support_not_independent_human_GT',processing_branch='AI_native_initialization_binding') for r in native]
    near=sorted(kept+corrected,key=lambda r:(int(r['source_frame']),SLOTS.index(r['player_slot'])))
    temp=OUT/'tracking_quality_20261102_stage4/players_015_temporal';temp.mkdir(parents=True);write(temp/'selected_four.csv',near)
    dump(temp/'summary.json',dict(summary,input_csv=str(rawpath),source_warning='Assisted initial N1 binding, not automatic performance.',counts=dict(Counter(r['player_slot'] for r in near))))
    event=dict(event,seed_ids=dict(event['seed_ids'],near_front=3),identity_status='AI_assisted_seed_not_independent_GT')
    dump(temp/'events.json',[event,dict(event='AI_native_initialization_binding',frame=init,slot='near_front',raw_id=3,scope=[init,last])])
    far.OUT=OUT/'tracking_quality_20261102_stage7';far.OUT.mkdir();oldargs=sys.argv;sys.argv=[str(far.__file__),'--tag','015']
    try:far.main()
    finally:sys.argv=oldargs
    farther=read(far.OUT/'players_015_global/far_pair_path.csv');combined=sorted(near+[dict(r,processing_branch='offline_far_search_with_AI_bound_near') for r in farther],key=lambda r:(int(r['source_frame']),SLOTS.index(r['player_slot'])))
    write(OUT/'four_player_path.csv',combined);people=defaultdict(dict);rawmap=defaultdict(list)
    for r in raw:rawmap[int(r['source_frame']),r['track_id']].append(r)
    for r in combined:
        f=int(r['source_frame']);assert r['player_slot'] not in people[f];people[f][r['player_slot']]=r
        assert any(all(r[k]==n[k] for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y','confidence']) for n in rawmap[f,r['track_id']])
    assert [r for r in combined if r['player_slot']=='near_rear']==kept
    assert not any(people[f] for f in range(first,init))
    for ps in people.values():assert len({r['track_id'] for r in ps.values()})==len(ps)
    times=p['requested_input_frames'];cid=p['candidate_id'];features=[dict(rally_candidate_id=cid,relative_source_frame=f-times[0],**pair_features(people[f])) for f in times]
    snap,summ,_=summarize_prefix('015',[dict(r,ball_observation_present=0,ball_x_norm='',ball_y_norm='') for r in features])
    for name,rs in [('prefix_frame_features.csv',features),('snapshot_player_features.csv',[snap]),('prefix_player_summary_features.csv',[summ])]:write(OUT/name,rs)
    expected=[calculate_frame(cid,people,f,f-times[0]) for f in times];checked=0
    for actual,wanted in zip(features,expected):
        for k,v in wanted.items():
            if k!='rally_candidate_id':close(actual[k],v);checked+=1
    es,em=calculate_summary(expected)
    for actual,wanted in [(snap,es),(summ,em)]:
        for k,v in wanted.items():close(actual[k],v);checked+=1
    key=lambda r:tuple(r[k] for k in ['source_frame','player_slot','track_id','x1','y1','x2','y2'])
    far_equal=[key(r) for r in baseline if r['player_slot'].startswith('far_')]==[key(r) for r in combined if r['player_slot'].startswith('far_')]
    video=far.OUT/'players_015_global/global_review_h264.mp4';cap=cv2.VideoCapture(str(video));count=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(1080,1920);count+=1
    cap.release();assert count==last-first+1;assert all(sha(Path(n))==h for n,h in protected.items())
    dump(OUT/'pre_outcome_covariates.json',load(CASE/'pre_outcome_covariates.json'))
    dump(OUT/'verification.json',dict(status='passed',matched_native_boxes=len(combined),recalculated_feature_values=checked,unchanged_N2_observations=len(kept),AI_bound_N1_observations=len(corrected),raw3_missing_after_initialization=[],far_parameters_unchanged=True,far_assignments_identical_to_automatic=far_equal,corrected_seed_negative_set=True,requested_input_frames_unchanged=True,prefix_feature_values_identical_to_automatic=all(read(OUT/n)==read(CASE/n) for n in ['prefix_frame_features.csv','snapshot_player_features.csv','prefix_player_summary_features.csv']),old_sources_unchanged=True,interpolation=False,video_frames_decoded=count,independent_identity_or_position_accuracy_established=False))
    dump(OUT/'case_result.json',dict(result,status='AI_native_initialization_binding_and_retained_far_rerun_constructed',selected_slot_counts=dict(Counter(r['player_slot'] for r in combined)),input_times_with_four_observations=sum(len(people[f])==4 for f in times),full_context_four_observation_frames=sum(len(people[f])==4 for f in range(first,last+1)),AI_identity_binding_added=True,fully_automatic_result=False,phase_audit_status='pending_AI_variant_image_inspection',artifact_hashes={n.name:sha(n) for n in OUT.glob('*.csv')}))
    print(json.dumps(load(OUT/'verification.json'),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
