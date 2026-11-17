"""Full retained-context comparison with alternate-frame torso evidence."""
import argparse,os,json
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,SOURCE,read,write,sha,dump
from run_far_identity_guard_trial import build_cache,export
from diagnose_initial_pair_appearance import cache as read_cache
from run_body_part_localization_probe import CROPS,SLOTS,load,overlap,one_to_one
from diagnose_pose_torso_appearance import feature,distance
import torso_temporal_association as association

OUT=BASE/'torso_temporal_association_trial'
MODES=['torso_team_soft_cost','torso_team_soft_cost_pose_exclusive']

def prepare():
    old=BASE/'body_part_localization_probe';p=load(old/'protocol.json');assert load(old/'final_verification.json')['status']=='passed';OUT.mkdir();protected=dict(p['protected_input_hashes'])
    for n in [Path(__file__),ROOT/'torso_temporal_association.py',ROOT/'run_far_identity_guard_trial.py',ROOT/'diagnose_pose_torso_appearance.py',ROOT/'run_body_part_localization_probe.py',old/'protocol.json',old/'pose_predictions.json',old/'temporal_demo/pose_predictions.json',old/'final_verification.json',old/'publication_hashes.json']:protected[str(n)]=sha(n)
    cases=[]
    for original in p['cases']:
        e=dict(original);event,near,cache,scores=build_cache(e);schedule={f:['far_context'] for f in range(event['frame'],e['last']+1,2) if cache[f-event['frame']]}
        assert event['frame']-event['window_start']+1==8
        for f in range(event['window_start'],event['frame']+1):schedule[f]=['court','far_context']
        e.update(initialization=event,pose_schedule={str(f):m for f,m in sorted(schedule.items())},full_search_frames=len(cache),scheduled_far_frames=sum('far_context' in m for m in schedule.values()),scheduled_court_frames=8);cases.append(e)
    for n in (BASE/'current22_player_inputs').iterdir():
        if n.is_file():protected[str(n)]=sha(n)
    assert all(sha(Path(n))==h for n,h in protected.items())
    dump(OUT/'protocol.json',dict(cases=cases,modes=MODES,parameters=association.PARAMETERS,far_parameters=association.P,source_video=p['source_video'],source_video_sha256=p['source_video_sha256'],source_clock=p['source_clock'],model=p['model'],model_sha256=p['model_sha256'],model_parameters=p['parameters'],crops_native_xyxy=CROPS,policy_fixed_before_full_context_pose_and_path_outputs=True,same_match_development_after_known_failures_and_pose_probe=True,selection='Same ten retained contexts and all search frames; no control/cohort substitutions.',pose_sampling='At initialization8 frames both scales; subsequently every2 source frames relative to initialization, only where retained native far candidates exist. Unsampled frames receive zero extra cost, not interpolated features.',association_rule='Add0.8*max(0,min_far_team_torso_distance-min_near_or_initial_unselected_torso_distance-0.04) to each native far observation cost. Both far IDs share the same extra cost. No color-based rejection; missing pose/evidence has zero extra cost.',anchor_rule='Retained initialization8 source frames, exact selected native seed IDs. One-to-one pose correspondence IoU>=0.5. Far anchors from far_context, near anchors from court. Initial unselected eligible far-context poses are negative exemplars only when overlap<0.1 with every selected native seed at initialization.',additional_ablation='Second mode also prevents the two chosen native candidates from using the same per-frame pose. It neither requires a pose nor substitutes pose coordinates.',numeric_parameters='Existing weights/margins reused, no sweep. Soft cost is an engineering term, not calibrated identity confidence.',baseline_verification='Replay copied solver with zero extra cost and exclusivity disabled for all ten cases; require exact native paths and cost.',observed_positions='Only original native detection coordinates exported; ankle/torso estimates stay internal diagnostics. Neither original box bottoms nor model keypoints establish measured ground contact.',offline_not_online=True,independent_GT=False,new_outcome_labels=False,canonical_pipeline_replaced=False,protected_input_hashes=protected))
    print('Frozen',len(cases),'contexts;',sum(e['full_search_frames'] for e in cases),'search frames;',sum(e['scheduled_far_frames'] for e in cases),'far pose frames;',sum(e['scheduled_court_frames'] for e in cases),'court pose frames.',flush=True)

def cached_poses():
    prior=BASE/'body_part_localization_probe';cached={}
    for r in load(prior/'pose_predictions.json')['frames']:
        for mode,people in r['modes'].items():cached[r['tag'],r['source_frame'],mode]=people
    for r in load(prior/'temporal_demo/pose_predictions.json')['frames']:
        k=r['tag'],r['source_frame'],'far_context'
        if k in cached:assert cached[k]==r['people']
        else:cached[k]=r['people']
    return cached

def infer():
    p=load(OUT/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items());os.environ['YOLO_CONFIG_DIR']=str(OUT/'ultralytics_config');cached=cached_poses()
    from ultralytics import YOLO
    import torch,ultralytics
    assert torch.cuda.is_available();assert sha(Path(p['model']))==p['model_sha256'];net=YOLO(p['model']);dump(OUT/'runtime.json',dict(torch=torch.__version__,ultralytics=ultralytics.__version__,device=torch.cuda.get_device_name(0),model_sha256=sha(Path(p['model']))));reports=[]
    for e in p['cases']:
        tag=e['tag'];dest=OUT/f'case_{tag}'
        if (dest/'pose_result.json').exists():
            old=load(dest/'pose_result.json');assert all(sha(dest/n)==h for n,h in old['hashes'].items());reports.append(old);print('Retained completed pose cache',tag,flush=True);continue
        dest.mkdir(exist_ok=True);schedule={int(f):m for f,m in e['pose_schedule'].items()};cap=cv2.VideoCapture(p['source_video']);first=min(schedule);last=max(schedule);cap.set(1,first);records=[];boxes=[];keypoints=[];features=[];calls=reused=0;done=0
        for f in range(first,last+1):
            if f not in schedule:assert cap.grab();continue
            ok,im=cap.read();assert ok and im.shape[:2]==(2160,3840) and round(cap.get(1))==f+1
            for mode in schedule[f]:
                reused_here=(tag,f,mode) in cached
                if reused_here:people=cached[tag,f,mode];reused+=1
                else:
                    l,t,r,b=CROPS[mode];pred=net.predict(im[t:b,l:r],imgsz=1920,conf=.25,iou=.7,device=0,verbose=False)[0];people=[];calls+=1
                    for j,(bb,kp) in enumerate(zip(pred.boxes.data.cpu().numpy(),pred.keypoints.data.cpu().numpy())):
                        bb=bb.copy();kp=kp.copy();bb[[0,2]]+=l;bb[[1,3]]+=t;kp[:,0]+=l;kp[:,1]+=t
                        people.append(dict(pose_index=j,box=bb[:4].tolist(),confidence=float(bb[4]),keypoints=kp.tolist(),eligible_not_crop_truncated=bool(min(bb[0]-l,bb[1]-t,r-bb[2],b-bb[3])>8)))
                for person in people:
                    v,count=feature(im,person,'pose_torso_polygon') if person['eligible_not_crop_truncated'] else (None,0);index=len(boxes);boxes.append([*person['box'],person['confidence']]);keypoints.append(person['keypoints']);features.append(v if v is not None else np.full(192,np.nan,np.float32));records.append(dict(source_frame=f,mode=mode,pose_index=person['pose_index'],array_row=index,eligible=int(person['eligible_not_crop_truncated']),feature_pixels=count,torso_available=int(v is not None),prediction_reused=int(reused_here)))
            done+=1
            if done%100==0:print('pose',tag,done,'/',len(schedule),'scheduled source frames',flush=True)
        cap.release();np.savez_compressed(dest/'pose_tensors.npz',boxes=np.asarray(boxes,np.float32),keypoints=np.asarray(keypoints,np.float32),torso_features=np.asarray(features,np.float32));write(dest/'pose_metadata.csv',records);report=dict(tag=tag,scheduled_source_frames=len(schedule),inference_calls=calls,reused_predictions=reused,pose_proposals=len(records),hashes={n:sha(dest/n) for n in ['pose_tensors.npz','pose_metadata.csv']});dump(dest/'pose_result.json',report);reports.append(report);print('Completed pose',tag,report,flush=True)
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items());dump(OUT/'pose_result.json',dict(status='all10_context_pose_caches_completed',cases=reports,new_inference_calls=sum(r['inference_calls'] for r in reports),reused_predictions=sum(r['reused_predictions'] for r in reports),protected_inputs_unchanged=True))

def load_pose(tag):
    dest=OUT/f'case_{tag}';p=load(dest/'pose_result.json');assert all(sha(dest/n)==h for n,h in p['hashes'].items());data=np.load(dest/'pose_tensors.npz');rows=read(dest/'pose_metadata.csv');by=defaultdict(list)
    for r in rows:
        i=int(r['array_row']);by[int(r['source_frame']),r['mode']].append(dict(row=r,box=data['boxes'][i,:4],confidence=data['boxes'][i,4],keypoints=data['keypoints'][i],feature=None if np.isnan(data['torso_features'][i]).all() else data['torso_features'][i],eligible=r['eligible']=='1',pose_index=int(r['pose_index'])))
    return by

def build_gallery(e,event,poses):
    raw,arr,offset=read_cache(e);by=defaultdict(list)
    for r in raw:by[int(r['source_frame'])].append(r)
    anchors={s:[] for s in SLOTS};proof=[]
    for f in range(event['window_start'],event['frame']+1):
        native=[next(r for r in by[f] if int(r['track_id'])==event['seed_ids'][s]) for s in SLOTS]
        for mode in ['court','far_context']:
            people=[r for r in poses[f,mode] if r['eligible']];scores=np.array([[overlap([float(r[k]) for k in ['x1','y1','x2','y2']],pose['box'],union=True) for pose in people] for r in native]).reshape(4,len(people));chosen=one_to_one(scores)
            for i,slot in enumerate(SLOTS):
                if mode!=('far_context' if slot.startswith('far_') else 'court'):continue
                j=chosen[i]
                if j<len(people) and people[j]['feature'] is not None:anchors[slot].append(people[j]['feature']);proof.append(dict(source_frame=f,slot=slot,mode=mode,native_seed_id=event['seed_ids'][slot],pose_index=people[j]['pose_index'],iou=float(scores[i,j])))
    selected=[r for r in by[event['frame']] if int(r['track_id']) in event['seed_ids'].values()];background=[]
    for pose in poses[event['frame'],'far_context']:
        if pose['eligible'] and pose['feature'] is not None and all(overlap(pose['box'],[float(r[k]) for k in ['x1','y1','x2','y2']])<.1 for r in selected):background.append(pose['feature'])
    return anchors,background,proof

def compare():
    p=load(OUT/'protocol.json');assert load(OUT/'pose_result.json')['status']=='all10_context_pose_caches_completed';assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items());reports=[];fields=['source_frame','player_slot','track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y'];signature=lambda rows:[tuple(r[k] for k in fields) for r in rows]
    for e in p['cases']:
        tag=e['tag'];dest=OUT/f'case_{tag}';assert not (dest/'result.json').exists();event,near,cache,_=build_cache(e);path,search=association.solve(cache,event);baseline=export(cache,path);old=BASE/f'far_identity_guard_development/matched_trial/case_{tag}';assert signature(baseline)==signature(read(old/'baseline/far_pair_path.csv'));assert search['best_path_cost']==load(old/'result.json')['modes']['baseline']['path_cost'];folder=dest/'baseline';folder.mkdir();write(folder/'far_pair_path.csv',baseline);bk={tuple(r[k] for k in fields) for r in baseline};poses=load_pose(tag);anchors,background,proof=build_gallery(e,event,poses);write(dest/'initial_pose_anchor_correspondence.csv',proof);dump(dest/'gallery.json',dict(anchors={s:len(a) for s,a in anchors.items()},unselected_initial_pose_features=len(background),independent_GT=False));diagnostics=[];far=anchors['far_rear']+anchors['far_front'];negative=anchors['near_front']+anchors['near_rear']+background
        for offset,items in enumerate(cache):
            f=event['frame']+offset;people=[r for r in poses[f,'far_context'] if r['eligible']]
            for c in items:
                row=c['row'];scores=[overlap([float(row[k]) for k in ['x1','y1','x2','y2']],x['box'],union=True) for x in people];best=int(np.argmax(scores)) if scores else -1;person=people[best] if best>=0 and scores[best]>=.5 else None;v=person['feature'] if person else None;fd=nd='';c['torso_conflict_cost']=0.;c['pose_key']=(f,person['pose_index']) if person else None
                state='unsampled' if str(f) not in e['pose_schedule'] else 'unmatched_pose' if person is None else 'torso_unavailable' if v is None else 'anchor_unavailable' if not far or not negative else 'compared'
                if state=='compared':fd=min(distance(a,v) for a in far);nd=min(distance(a,v) for a in negative);c['torso_conflict_cost']=.8*max(0.,fd-nd-.04)
                diagnostics.append(dict(source_frame=f,native_track_id=row['track_id'],state=state,pose_index=person['pose_index'] if person else '',matching_iou=scores[best] if person else '',far_team_distance=fd,negative_distance=nd,extra_cost=c['torso_conflict_cost'],hard_veto=0,pose_coordinates_exported=0))
        write(dest/'candidate_torso_cost_diagnostics.csv',diagnostics);modes={}
        for mode in MODES:
            exclusive=mode.endswith('pose_exclusive');path,search=association.solve(cache,event,exclusive_pose=exclusive);rows=export(cache,path);folder=dest/mode;folder.mkdir();write(folder/'far_pair_path.csv',rows);nk={tuple(r[k] for k in fields) for r in rows}
            if exclusive:
                for i,chosen in enumerate(path):
                    evidence=[cache[i][j]['pose_key'] for j in chosen if j>=0 and cache[i][j]['pose_key'] is not None];assert len(evidence)==len(set(evidence))
            modes[mode]=dict(far_observations=len(rows),removed_baseline_rows=len(bk-nk),added_rows=len(nk-bk),same_native_path=nk==bk,path_cost=search['best_path_cost']);print('Compared',tag,mode,modes[mode],flush=True)
        report=dict(tag=tag,baseline_path_and_cost_exact=True,search_frames=len(cache),modes=modes);dump(dest/'result.json',report);reports.append(report)
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items());dump(OUT/'result.json',dict(status='ten_context_matched_comparison_completed',cases=reports,baseline_exact_replays=10,canonical_pipeline_changed=False,independent_accuracy_established=False,protected_inputs_unchanged=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','pose','compare'],required=True);args=ap.parse_args();{'prepare':prepare,'pose':infer,'compare':compare}[args.phase]()
