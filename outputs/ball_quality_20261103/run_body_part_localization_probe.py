"""Frozen selected-frame pose feasibility probe, never a player-path repair."""
import os,json
from pathlib import Path
from collections import defaultdict,Counter
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,SOURCE,read,write,sha,dump

OUT=BASE/'body_part_localization_probe'
SLOTS=['near_front','near_rear','far_rear','far_front']
CROPS={'court':[0,640,3840,2160],'far_context':[960,720,3072,1440]}
P=dict(imgsz=1920,confidence=.25,nms_iou=.7,device=0,pose_match_iou_min=.5,
       keypoint_confidence=.5,crop_edge_margin_native_pixels=8,
       model_ankle_estimates_never_exported_as_observed_player_positions=True)

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def box(r):return [float(r[k]) for k in ['x1','y1','x2','y2']]
def overlap(a,b,union=False):
    x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]));inter=x*y
    aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1])
    return inter/max(1,aa+bb-inter if union else min(aa,bb))

def one_to_one(scores,minimum=.5):
    """Exact maximum-weight matching for <=4 rows, with unmatched dummy choices."""
    n,m=scores.shape;states={0:(0.,{})}
    for j in range(m):
        updated=dict(states)
        for mask,(value,path) in states.items():
            for i in range(n):
                if mask&(1<<i) or scores[i,j]<minimum:continue
                newmask=mask|(1<<i);candidate=value+float(scores[i,j])-minimum
                if newmask not in updated or candidate>updated[newmask][0]:updated[newmask]=(candidate,dict(path,**{str(i):j}))
        states=updated
    _,(_,chosen)=max(states.items(),key=lambda x:x[1][0])
    return {i:chosen.get(str(i),m+i) for i in range(n)}

def verify_matching():
    from itertools import product
    rng=np.random.default_rng(20261110)
    for n in range(1,5):
        for m in range(0,5):
            for _ in range(5):
                scores=rng.random((n,m));chosen=one_to_one(scores);value=sum(scores[i,j] if j<m else .5 for i,j in chosen.items());best=-1
                for assignment in product(range(-1,m),repeat=n):
                    used=[j for j in assignment if j>=0]
                    if len(used)!=len(set(used)):continue
                    best=max(best,sum(scores[i,j] if j>=0 else .5 for i,j in enumerate(assignment)))
                assert abs(value-best)<1e-9

def case_tables(e):
    near=[r for r in read(Path(e['temporal_folder'])/'selected_four.csv') if r['player_slot'].startswith('near_')]
    far=read(BASE/f'far_identity_guard_development/matched_trial/case_{e["tag"]}/baseline/far_pair_path.csv')
    by=defaultdict(dict)
    for r in near+far:
        f=int(r['source_frame']);assert r['player_slot'] not in by[f];by[f][r['player_slot']]=r
    return by

def make_plan():
    prior=BASE/'far_spatial_guard_development';old=load(prior/'protocol.json');assert load(prior/'final_verification.json')['status']=='passed'
    assert all(sha(Path(n))==h for n,h in old['protected_input_hashes'].items())
    OUT.mkdir();metadata={r['rally_candidate_id'][-3:]:r for r in read(BASE/'current22_player_inputs/rally_metadata.csv')}
    model=SOURCE/'models/yolo11m-pose.pt';source=load(BASE/'cohort_tracking/case_019/protocol.json');protected={str(Path(__file__)):sha(Path(__file__)),str(model):sha(model)}
    for n in [prior/'protocol.json',prior/'final_verification.json',BASE/'current22_player_inputs/publication_hashes.json']:protected[str(n)]=sha(n)
    selected=[]
    extras={'006':[12421,12422],'016':[40625,40626],'019':[46352,46501,46502,46503,46505,46511,46634,46658,46659,46660,46665,46666,46667]}
    for e in old['cases']:
        tag=e['tag'];by=case_tables(e);events=Path(e['temporal_folder'])/'events.json';ev=next(x for x in load(events) if x['event']=='initialized');requests=defaultdict(list);requests[ev['frame']].append('initialization')
        cp=BASE/f'cohort_tracking/case_{tag}/protocol.json'
        if cp.exists():times=load(cp)['requested_input_frames'];protected[str(cp)]=sha(cp)
        else:times=list(range(int(metadata[tag]['input_first_source_frame']),int(metadata[tag]['input_end_source_frame'])+1,2))
        assert len(times)==31
        for i,name in [(0,'prefix_first'),(15,'prefix_middle'),(30,'prefix_last')]:requests[times[i]].append(name)
        ranked_far=[];ranked_near=[]
        for f,group in sorted(by.items()):
            far=[box(r) for s,r in group.items() if s.startswith('far_')];near=[box(r) for s,r in group.items() if s.startswith('near_')]
            if len(far)==2:ranked_far.append((overlap(*far),-f))
            if far and near:ranked_near.append((max(overlap(a,b) for a in far for b in near),-f))
        for ranked,name in [(ranked_far,'maximum_native_far_pair_overlap'),(ranked_near,'maximum_native_far_near_overlap')]:
            if ranked:
                score,minus_f=max(ranked);requests[-minus_f].append(name+f':{score:.6f}')
        for f in extras.get(tag,[]):requests[f].append('retained_known_development_example')
        for f,reason in sorted(requests.items()):
            assert e['first']<=f<=e['last'];selected.append(dict(tag=tag,source_frame=f,reasons=reason,native_observations=list(by[f].values())))
        for n in [events,Path(e['temporal_folder'])/'selected_four.csv',BASE/f'far_identity_guard_development/matched_trial/case_{tag}/baseline/far_pair_path.csv']:protected[str(n)]=sha(n)
    dump(OUT/'protocol.json',dict(purpose='Test whether existing pretrained body parts can separate people inside mixed rectangles. No trajectory modification, GT creation, threshold sweep, or outcome prediction.',cases=old['cases'],selected_frames=selected,selection_fixed_before_pose_outputs=True,selected_after_known_failure_review=True,same_match_development_not_independent_test=True,source_video=source['source_video'],source_video_sha256=source['source_video_sha256'],source_clock='legacy_edited_match1',native_size=[3840,2160],model=str(model),model_sha256=sha(model),parameters=P,crops_native_xyxy=CROPS,crop_rationale='Exactly double the previously used half-resolution court crop and far_court tile. Independent outputs retained; no scale-based winner selected.',threshold_source='Retained pose scripts: conf0.25,IoU0.7,imgsz1920,matchIoU0.5,keypoint0.5,cropedge4px at half resolution.',association='Per-frame exact maximum-IoU one-to-one matching to immutable native player rectangles, unmatched dummy weight0.5. Dynamic programming verified against100 exhaustive small cases. Matching is not verified identity.',visibility_warning='A confident keypoint may be hallucinated or belong to another overlapping person. Neither ankle confidence nor output availability certifies ground contact.',protected_input_hashes=protected,canonical_pipeline_replaced=False,new_human_labels=False))
    return load(OUT/'protocol.json')

def associations(item,mode,people):
    rows=item['native_observations'];valid=[r for r in people if r['eligible_not_crop_truncated']];scores=np.zeros((len(rows),len(valid)))
    for i,r in enumerate(rows):
        for j,d in enumerate(valid):scores[i,j]=overlap(box(r),d['box'],union=True)
    pairs=one_to_one(scores,P['pose_match_iou_min']);out=[]
    for i,r in enumerate(rows):
        j=pairs[i];matched=j<len(valid) and scores[i,j]>=P['pose_match_iou_min'];d=valid[j] if matched else None
        row=dict(tag=item['tag'],source_frame=item['source_frame'],mode=mode,player_slot=r['player_slot'],native_track_id=r['track_id'],native_box_bottom_x=r['bottom_center_x'],native_box_bottom_y=r['bottom_center_y'],pose_matched=int(matched),pose_index=d['pose_index'] if d else '',matching_iou=float(scores[i,j]) if d else '',competing_pose_max_iou=max([scores[i,k] for k in range(len(valid)) if k!=j],default=0),torso_points_above_threshold=0,both_ankles_above_threshold=0,model_torso_x='',model_torso_y='',model_ankles_x='',model_ankles_y='',observed_ground_position_certified=0,identity_certified=0)
        if d:
            kp=np.asarray(d['keypoints']);torso=bool(np.all(kp[[5,6,11,12],2]>=.5));ankles=bool(np.all(kp[[15,16],2]>=.5));row['torso_points_above_threshold']=int(torso);row['both_ankles_above_threshold']=int(ankles)
            if torso:row['model_torso_x'],row['model_torso_y']=kp[[5,6,11,12],:2].mean(axis=0).tolist()
            if ankles:row['model_ankles_x'],row['model_ankles_y']=kp[[15,16],:2].mean(axis=0).tolist()
        out.append(row)
    return out

def main():
    verify_matching();p=make_plan();os.environ['YOLO_CONFIG_DIR']=str(OUT/'ultralytics_config')
    import torch,ultralytics
    from ultralytics import YOLO
    assert torch.cuda.is_available();assert sha(Path(p['source_video']))==p['source_video_sha256'];net=YOLO(p['model']);cap=cv2.VideoCapture(p['source_video']);results=[];matches=[]
    runtime=dict(torch_version=torch.__version__,ultralytics_version=ultralytics.__version__,cuda_device=torch.cuda.get_device_name(0),model_sha256=sha(Path(p['model'])))
    dump(OUT/'runtime.json',runtime)
    for i,item in enumerate(p['selected_frames'],1):
        f=item['source_frame'];cap.set(1,f);ok,im=cap.read();assert ok and im.shape[:2]==(2160,3840) and round(cap.get(1))==f+1
        record=dict(tag=item['tag'],source_frame=f,modes={})
        for mode,rect in CROPS.items():
            l,t,r,b=rect;prediction=net.predict(im[t:b,l:r],imgsz=P['imgsz'],conf=P['confidence'],iou=P['nms_iou'],device=0,verbose=False)[0];people=[]
            for j,(bb,kp) in enumerate(zip(prediction.boxes.data.cpu().numpy(),prediction.keypoints.data.cpu().numpy())):
                bb=bb.copy();kp=kp.copy();bb[[0,2]]+=l;bb[[1,3]]+=t;kp[:,0]+=l;kp[:,1]+=t;eligible=min(bb[0]-l,bb[1]-t,r-bb[2],b-bb[3])>8
                people.append(dict(pose_index=j,box=bb[:4].tolist(),confidence=float(bb[4]),keypoints=kp.tolist(),eligible_not_crop_truncated=bool(eligible)))
            record['modes'][mode]=people;matches.extend(associations(item,mode,people))
        results.append(record)
        if i%10==0 or i==len(p['selected_frames']):print(f'Pose probe {i}/{len(p["selected_frames"])} sourceF{f}',flush=True)
    cap.release();dump(OUT/'pose_predictions.json',dict(frames=results,inferred_not_observed_ground_positions=True));write(OUT/'native_box_to_pose_diagnostics.csv',matches)
    summary=[]
    for tag in sorted({r['tag'] for r in matches}):
        for mode in CROPS:
            rows=[r for r in matches if r['tag']==tag and r['mode']==mode and r['player_slot'].startswith('far_')]
            summary.append(dict(tag=tag,mode=mode,selected_native_far_rows=len(rows),pose_box_matches=sum(r['pose_matched'] for r in rows),torso_sets_above_confidence_threshold=sum(r['torso_points_above_threshold'] for r in rows),ankle_pairs_above_confidence_threshold=sum(r['both_ankles_above_threshold'] for r in rows),identity_or_position_accuracy_established=0))
    write(OUT/'per_case_pose_output_availability_not_accuracy.csv',summary)
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items());dump(OUT/'result.json',dict(status='selected_frame_probe_completed',selected_frames=len(results),inference_calls=len(results)*len(CROPS),native_to_pose_diagnostic_rows=len(matches),parameters_changed_after_outputs=False,canonical_pipeline_changed=False,independent_GT=False,tracking_accuracy_established=False,protected_sources_unchanged=True))
    print(json.dumps(load(OUT/'result.json')),flush=True)

if __name__=='__main__':main()
