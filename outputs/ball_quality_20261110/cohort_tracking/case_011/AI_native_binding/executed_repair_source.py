"""AI-assisted case011 native N2 binding and retained far-search rerun.

This is a separate assisted variant, not a general automatic tracking fix.
"""
import csv,hashlib,json,sys
from pathlib import Path
from collections import defaultdict,Counter
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent;CASE=ROOT.parent/'ball_quality_20261110/cohort_tracking/case_011';OUT=CASE/'AI_native_binding'
sys.path.insert(0,str(ROOT.parent/'tracking_quality_20261102_stage10'))
import gap_pair_tracking as far
from build_rally_analysis_prototype import pair_features,summarize_prefix
from verify_player_prefix_expansion import calculate_frame,calculate_summary,close
SLOTS=['near_front','near_rear','far_rear','far_front'];NAMES=['N1','N2','F1','F2'];COLORS=[(80,255,80),(255,190,40),(255,80,240),(0,230,255)]
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def write(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
    OUT.mkdir();p=load(CASE/'protocol.json');original=load(CASE/'case_result.json');cid=p['candidate_id'];first=p['context_first_source_frame'];last=p['context_last_source_frame'];init=original['initialized_frame']
    rawpath=CASE/'crop_runs/augmented_detections_011.csv';raw=read(rawpath);baseline=read(CASE/'four_player_path.csv');native=[r for r in raw if r['track_id']=='1' and init<=int(r['source_frame'])<=last]
    assert len(native)==2882 and all(r['candidate_origin']=='original_full_frame' for r in native)
    original_temporal=CASE/'tracking_quality_20261102_stage4/players_011_temporal';summary=load(original_temporal/'summary.json');event=load(original_temporal/'events.json')[0]
    seed=p['contact_candidate_last_source_frame']-30;assert seed==26873
    protected={str(n):sha(n) for n in [CASE/'protocol.json',CASE/'case_result.json',CASE/'four_player_path.csv',CASE/'verification.json',rawpath,original_temporal/'summary.json',original_temporal/'events.json',Path(far.__file__),Path(__file__)]}
    dump(OUT/'seed_design.json',dict(candidate_id=cid,seed_source_frame=seed,native_detector_raw_id=1,target_slot='near_rear',selection='Existing QA time contact-last-30 and AI correspondence on drift diagnosis. Bind original observed raw1 only, missing18 frames remain empty.',export_scope=[init,last],native_observations=len(native),AI_identity_assistance=True,human_identity_GT=False,fully_automatic_result=False,interpolation=False,baseline_known_wrong_person=True,far_search='Recompute retained offline far search because incorrect near boxes had excluded actual far players. Numerical settings unchanged.'))
    q=dict(p);q.update(derived_variant='AI_native_N2_binding_with_retained_far_rerun',AI_identity_binding_added=True,fully_automatic_result=False,full_automatic_baseline_valid=False,algorithm_parameters_changed=False,detector_or_association_numeric_parameters_changed=False,source_of_assignment='AI bound original raw1 to N2; N1 frozen, far candidates and search recomputed with corrected near exclusion.',original_automatic_protocol=str(CASE/'protocol.json'),protected_input_hashes=protected)
    dump(OUT/'protocol.json',q)
    n1=[r for r in baseline if r['player_slot']=='near_front'];assert len(n1)==2900
    n2=[dict(r,player_slot='near_rear',position_source='detected_box',identity_status='AI_native_track_bound_candidate_not_GT',identity_source='AI_seed_F26873_raw1_native_continuity',identity_review_status='AI_image_support_not_independent_human_GT',processing_branch='AI_native_track_binding') for r in native]
    near=sorted(n1+n2,key=lambda r:(int(r['source_frame']),SLOTS.index(r['player_slot'])))
    temp=OUT/'tracking_quality_20261102_stage4/players_011_temporal';temp.mkdir(parents=True)
    write(temp/'selected_four.csv',near)
    dump(temp/'summary.json',dict(summary,source_warning='Assisted near bindings only; no reuse of original false recovery events.',counts=dict(Counter(r['player_slot'] for r in near)),input_csv=str(rawpath)))
    dump(temp/'events.json',[event,dict(event='AI_native_track_binding',frame=seed,slot='near_rear',raw_id=1,scope=[init,last])])
    far.OUT=OUT/'tracking_quality_20261102_stage7';far.OUT.mkdir();old=sys.argv;sys.argv=[str(far.__file__),'--tag','011']
    try:far.main()
    finally:sys.argv=old
    farther=read(far.OUT/'players_011_global/far_pair_path.csv')
    combined=sorted(near+[dict(r,processing_branch='offline_far_search_with_AI_bound_near') for r in farther],key=lambda r:(int(r['source_frame']),SLOTS.index(r['player_slot'])))
    write(OUT/'four_player_path.csv',combined);people=defaultdict(dict);rawmap=defaultdict(list)
    for r in raw:rawmap[int(r['source_frame']),r['track_id']].append(r)
    for r in combined:
        f=int(r['source_frame']);assert r['player_slot'] not in people[f];people[f][r['player_slot']]=r
        assert any(all(r[k]==x[k] for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y','confidence']) for x in rawmap[f,r['track_id']])
    for ps in people.values():assert len({r['track_id'] for r in ps.values()})==len(ps)
    times=p['requested_input_frames'];features=[dict(rally_candidate_id=cid,relative_source_frame=f-times[0],**pair_features(people[f])) for f in times]
    snap,summ,_=summarize_prefix('011',[dict(r,ball_observation_present=0,ball_x_norm='',ball_y_norm='') for r in features])
    for name,rows in [('prefix_frame_features.csv',features),('snapshot_player_features.csv',[snap]),('prefix_player_summary_features.csv',[summ])]:write(OUT/name,rows)
    expected=[calculate_frame(cid,people,f,f-times[0]) for f in times];checked=0
    for actual,wanted in zip(features,expected):
        for k,v in wanted.items():
            if k!='rally_candidate_id':close(actual[k],v);checked+=1
    es,em=calculate_summary(expected)
    for actual,wanted in [(snap,es),(summ,em)]:
        for k,v in wanted.items():close(actual[k],v);checked+=1
    cap=cv2.VideoCapture(p['source_video']);views=[seed,times[0],times[-1],27222,27223,27237,27385,27514,27854,28140,28286,28881];cells=[]
    for f in views:
        cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1
        for i,s in enumerate(SLOTS):
            r=people[f].get(s)
            if not r:continue
            box=[round(float(r[k])) for k in ['x1','y1','x2','y2']];cv2.rectangle(im,tuple(box[:2]),tuple(box[2:]),COLORS[i],6);cv2.putText(im,f'{NAMES[i]} raw{r["track_id"]}',(box[0],max(35,box[1]-12)),0,1.2,COLORS[i],3)
        thumb=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA);head=np.zeros((34,960,3),np.uint8);cv2.putText(head,f'AI-assisted011 F{f} {len(people[f])}/4 observed',(12,25),0,.8,(255,255,255),1);cells.append(np.vstack([head,thumb]))
    cap.release();cv2.imwrite(str(OUT/'candidate_sequence_grid.jpg'),np.vstack([np.hstack(cells[i:i+3]) for i in range(0,len(cells),3)]))
    video=far.OUT/'players_011_global/global_review_h264.mp4';cap=cv2.VideoCapture(str(video));count=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(1080,1920);count+=1
    cap.release();assert count==last-first+1
    assert all(sha(Path(n))==h for n,h in protected.items())
    v=dict(status='passed',matched_native_boxes=len(combined),recalculated_feature_values=checked,unchanged_N1_observations=len(n1),AI_bound_N2_observations=len(n2),missing_native_N2_frames=[f for f in range(init,last+1) if not any(r['player_slot']=='near_rear' for r in people[f].values())],far_parameters_unchanged=True,requested_input_frames_unchanged=True,prefix_feature_values_identical_to_automatic=all(read(OUT/n)==read(CASE/n) for n in ['prefix_frame_features.csv','snapshot_player_features.csv','prefix_player_summary_features.csv']),old_sources_unchanged=True,interpolation=False,video_frames_decoded=count,independent_identity_or_position_accuracy_established=False)
    dump(OUT/'verification.json',v)
    result=dict(original,status='AI_native_binding_and_retained_far_rerun_constructed',selected_slot_counts=dict(Counter(r['player_slot'] for r in combined)),input_times_with_four_observations=sum(len(people[f])==4 for f in times),full_context_four_observation_frames=sum(len(people[f])==4 for f in range(first,last+1)),AI_identity_binding_added=True,fully_automatic_result=False,phase_audit_status='pending_AI_variant_image_inspection',artifact_hashes={n.name:sha(n) for n in OUT.glob('*.csv')})
    dump(OUT/'case_result.json',result);print(json.dumps(v,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
