"""Separate observed identity paths from geometric support of bbox-bottom proxies."""
import argparse,os
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
from run_body_part_localization_probe import load,overlap,one_to_one,CROPS,SLOTS
from run_torso_temporal_trial_cached import cached_poses
from build_rally_analysis_prototype import pair_features,summarize_prefix
from verify_player_prefix_expansion import calculate_frame,calculate_summary,close

OUT=BASE/'position_proxy_support_trial'
PRIOR=BASE/'torso_temporal_association_trial'
POLICIES=['retain_native_proxies','mask_native_box_overlap','mask_pose_body_overlap']
BODY=dict(required_keypoints=[0,5,6,11,12],confidence=.5,torso_scale=1.12,head_radius_height=.09,arm_radius_height=.045,leg_radius_height=.065)

def native_table(e):
    near=[r for r in read(Path(e['temporal_folder'])/'selected_four.csv') if r['player_slot'].startswith('near_')];far=read(PRIOR/f'case_{e["tag"]}/torso_team_soft_cost/far_pair_path.csv');by=defaultdict(dict)
    for r in near+far:
        f=int(r['source_frame']);assert r['player_slot'] not in by[f];by[f][r['player_slot']]=r
    return by

def inside_rect(point,row):return float(row['x1'])<=point[0]<=float(row['x2']) and float(row['y1'])<=point[1]<=float(row['y2'])
def rect(row):return [float(row[k]) for k in ['x1','y1','x2','y2']]

def body_supported(person):return bool(np.all(np.asarray(person['keypoints'])[BODY['required_keypoints'],2]>=.5))

def inside_body(point,person):
    """Analytic torso/head/limb approximation, NOT a visible-pixel segmentation."""
    if not body_supported(person):return False
    kp=np.asarray(person['keypoints']);bb=np.asarray(person['box']);height=bb[3]-bb[1];point=np.asarray(point,dtype=float);quad=kp[[5,6,12,11],:2];center=quad.mean(axis=0);hull=cv2.convexHull(np.asarray(center+1.12*(quad-center),np.float32))
    if cv2.pointPolygonTest(hull,tuple(point),False)>=0:return True
    if np.linalg.norm(point-kp[0,:2])<=.09*height:return True
    for a,b,radius in [(5,7,.045),(7,9,.045),(6,8,.045),(8,10,.045),(11,13,.065),(13,15,.065),(12,14,.065),(14,16,.065)]:
        if min(kp[a,2],kp[b,2])<.5:continue
        start=kp[a,:2];delta=kp[b,:2]-start;v=float(np.clip(np.dot(point-start,delta)/max(float(np.dot(delta,delta)),1e-12),0,1))
        if np.linalg.norm(point-(start+v*delta))<=radius*height:return True
    return False

def initial_pose_geometry(tag):
    dest=PRIOR/f'case_{tag}';data=np.load(dest/'pose_tensors.npz');boxes=data['boxes'];kp=data['keypoints'];by=defaultdict(list)
    for r in read(dest/'pose_metadata.csv'):
        i=int(r['array_row']);by[int(r['source_frame']),r['mode']].append(dict(pose_index=int(r['pose_index']),box=boxes[i,:4].tolist(),confidence=float(boxes[i,4]),keypoints=kp[i].tolist(),eligible_not_crop_truncated=r['eligible']=='1'))
    return by

def match_people(group,people):
    rows=list(group.values());valid=[r for r in people if r['eligible_not_crop_truncated']];scores=np.array([[overlap(rect(r),pose['box'],union=True) for pose in valid] for r in rows]).reshape(len(rows),len(valid));chosen=one_to_one(scores);result={}
    for i,row in enumerate(rows):
        j=chosen[i]
        if j<len(valid):result[row['player_slot']]=(valid[j],float(scores[i,j]))
    return result

def prepare():
    p=load(PRIOR/'protocol.json');assert load(PRIOR/'final_verification.json')['status']=='passed';OUT.mkdir();meta={r['rally_candidate_id'][-3:]:r for r in read(BASE/'current22_player_inputs/rally_metadata.csv')};cases=[];protected={str(n):sha(n) for n in [Path(__file__),ROOT/'recover_044_with_pose_occluders.py',ROOT/'run_body_part_localization_probe.py',PRIOR/'protocol.json',PRIOR/'final_verification.json',PRIOR/'publication_hashes.json',BASE/'body_part_localization_probe/pose_predictions.json',BASE/'body_part_localization_probe/temporal_demo/pose_predictions.json']}
    extras={'002':[2319],'006':[12421,12422],'009':[21945,21975],'016':[40625,40626],'019':[46352,46511,46634]}
    for original in p['cases']:
        e=dict(original);tag=e['tag'];table=native_table(e);cp=BASE/f'cohort_tracking/case_{tag}/protocol.json';times=load(cp)['requested_input_frames'] if cp.exists() else list(range(int(meta[tag]['input_first_source_frame']),int(meta[tag]['input_end_source_frame'])+1,2));assert len(times)==31;wanted=set(times)|set(extras.get(tag,[]));risk_frames=[]
        for f,group in sorted(table.items()):
            risks=[]
            for slot,row in group.items():
                if not slot.startswith('far_'):continue
                point=[float(row['bottom_center_x']),float(row['bottom_center_y'])]
                for other,n in group.items():
                    if other!=slot and inside_rect(point,n):risks.append(overlap(rect(row),rect(n)))
            if risks:risk_frames.append((max(risks),-f))
        selected=[]
        if risk_frames:
            selected=sorted({-risk_frames[0][1],-max(risk_frames)[1]});wanted.update(selected)
        if tag=='019':wanted.update(range(46500,46522));wanted.update(range(46625,46642))
        e.update(requested_input_frames=times,extra_control_frames=extras.get(tag,[]),geometric_control_frames=selected,pose_audit_frames=sorted(wanted));cases.append(e)
        for n in [Path(e['temporal_folder'])/'selected_four.csv',PRIOR/f'case_{tag}/torso_team_soft_cost/far_pair_path.csv',PRIOR/f'case_{tag}/pose_tensors.npz',PRIOR/f'case_{tag}/pose_metadata.csv',Path(e['raw_csv'])]:protected[str(n)]=sha(n)
    for n in (BASE/'current22_player_inputs').iterdir():
        if n.is_file():protected[str(n)]=sha(n)
    dump(OUT/'protocol.json',dict(cases=cases,source_video=p['source_video'],source_video_sha256=p['source_video_sha256'],source_clock=p['source_clock'],model=p['model'],model_sha256=p['model_sha256'],model_parameters=p['model_parameters'],crop_native_xyxy=CROPS,body_approximation_parameters=BODY,body_parameter_source='Retain044 pose-occluder torso/head/limb dimensions. Use an analytic geometric approximation, no rectangle fallback and no segmentation/visibility claim.',position='Native detector bbox-bottom center in image coordinates. No ground-plane/contact/physical-speed claim.',pose_correspondence='Exact per-frame one-to-one maximum-IoU>=0.5 matching to the frozen native observations. Preferred far_context for far slots and court for near slots. No fallback-scale selection.',pose_coverage='Re-use all prior available pose geometry; request BOTH scales at every fixed early31 time, at two retained019 neighborhoods and at preselected control frames. No shifts, interpolation or nearest-frame substitution.',flags='Native bbox-bottom inside another selected native rectangle; or inside a confident matched OTHER-player pose body approximation. These indicate model/geometric overlap risk, not verified occlusion or position error.',sensitivity_policies=POLICIES,policy_meaning='Hypothetical position-only masking. Identity rows always preserved. Absence of pose/body support is unassessed, never certified correct. Body risk can only be tested where an OTHER-body approximation exists.',confidence_warning='Confident ankles can be inferred behind occluders or belong to a different person. All certified-identity/position fields remain0.',new_numeric_distance_thresholds=False,new_pose_coordinates_exported_as_observed_positions=False,policy_fixed_before_new_pose_or_sensitivity_outputs=True,same_match_development_not_independent_evaluation=True,canonical_pipeline_replaced=False,protected_input_hashes=protected))
    print('Frozen',len(cases),'cases,',sum(len(e['pose_audit_frames']) for e in cases),'requested frame pairs; all310 fixed input times retained.',flush=True)

def infer():
    p=load(OUT/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items());old=cached_poses();os.environ['YOLO_CONFIG_DIR']=str(OUT/'ultralytics_config')
    from ultralytics import YOLO
    net=YOLO(p['model']);assert sha(Path(p['model']))==p['model_sha256'];reports=[]
    for e in p['cases']:
        tag=e['tag'];dest=OUT/f'case_{tag}';dest.mkdir();dense=initial_pose_geometry(tag);wanted=set(e['pose_audit_frames']);cap=cv2.VideoCapture(p['source_video']);cap.set(1,min(wanted));records=[];boxes=[];joints=[];calls=reused=0
        for f in range(min(wanted),max(wanted)+1):
            if f not in wanted:assert cap.grab();continue
            ok,im=cap.read();assert ok and round(cap.get(1))==f+1
            for mode,region in CROPS.items():
                if (f,mode) in dense:people=dense[f,mode];provenance='retained_full_context_cache';reused+=1
                elif (tag,f,mode) in old:people=old[tag,f,mode];provenance='retained_probe_or_temporal_demo';reused+=1
                else:
                    l,t,r,b=region;pred=net.predict(im[t:b,l:r],imgsz=1920,conf=.25,iou=.7,device=0,verbose=False)[0];people=[];calls+=1;provenance='new_fixed_position_audit'
                    for j,(bb,kp) in enumerate(zip(pred.boxes.data.cpu().numpy(),pred.keypoints.data.cpu().numpy())):
                        bb=bb.copy();kp=kp.copy();bb[[0,2]]+=l;bb[[1,3]]+=t;kp[:,0]+=l;kp[:,1]+=t;people.append(dict(pose_index=j,box=bb[:4].tolist(),confidence=float(bb[4]),keypoints=kp.tolist(),eligible_not_crop_truncated=bool(min(bb[0]-l,bb[1]-t,r-bb[2],b-bb[3])>8)))
                for person in people:
                    i=len(boxes);boxes.append([*person['box'],person['confidence']]);joints.append(person['keypoints']);records.append(dict(source_frame=f,mode=mode,pose_index=person['pose_index'],array_row=i,eligible=int(person['eligible_not_crop_truncated']),prediction_provenance=provenance))
        cap.release();np.savez_compressed(dest/'pose_geometry.npz',boxes=np.asarray(boxes,np.float32),keypoints=np.asarray(joints,np.float32));write(dest/'pose_metadata.csv',records);report=dict(tag=tag,requested_frame_pairs=len(wanted),new_inference_calls=calls,reused_predictions=reused,pose_proposals=len(records),hashes={n:sha(dest/n) for n in ['pose_geometry.npz','pose_metadata.csv']});dump(dest/'pose_result.json',report);reports.append(report);print(report,flush=True)
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items());dump(OUT/'pose_result.json',dict(status='complete',cases=reports,new_inference_calls=sum(r['new_inference_calls'] for r in reports),reused_predictions=sum(r['reused_predictions'] for r in reports)))

def all_pose_geometry(tag):
    by=initial_pose_geometry(tag);old=cached_poses()
    for (t,f,mode),people in old.items():
        if t==tag and (f,mode) not in by:by[f,mode]=people
    dest=OUT/f'case_{tag}';data=np.load(dest/'pose_geometry.npz');boxes=data['boxes'];kp=data['keypoints'];new=defaultdict(list)
    for r in read(dest/'pose_metadata.csv'):
        i=int(r['array_row']);new[int(r['source_frame']),r['mode']].append(dict(pose_index=int(r['pose_index']),box=boxes[i,:4].tolist(),confidence=float(boxes[i,4]),keypoints=kp[i].tolist(),eligible_not_crop_truncated=r['eligible']=='1'))
    by.update(new);return by

def support_for_frame(tag,f,group,poses):
    matches={};evaluated={}
    for mode in CROPS:
        evaluated[mode]=(f,mode) in poses
        for slot,value in match_people(group,poses.get((f,mode),[])).items():
            if mode==('far_context' if slot.startswith('far_') else 'court'):matches[slot]=value
    rows=[]
    for slot in SLOTS:
        row=group.get(slot);mode='far_context' if slot.startswith('far_') else 'court';result=dict(tag=tag,source_frame=f,player_slot=slot,native_observation_present=int(row is not None),native_track_id=row['track_id'] if row else '',native_proxy_x=row['bottom_center_x'] if row else '',native_proxy_y=row['bottom_center_y'] if row else '',native_proxy_in_other_boxes='',native_proxy_in_other_pose_bodies='',other_pose_bodies_evaluated='',pose_frame_evaluated=int(evaluated[mode]),own_pose_matched=0,own_pose_index='',own_pose_iou='',own_body_approximation_available=0,both_model_ankles_above_threshold=0,model_ankle_mid_x='',model_ankle_mid_y='',native_minus_model_ankle_dx_over_native_height='',native_minus_model_ankle_dy_over_native_height='',native_proxy_to_model_ankle_distance_over_native_height='',native_height_over_model_pose_height='',geometry_box_overlap_risk=0,geometry_body_overlap_risk=0,position_support_state='native_missing',identity_certified=0,ground_position_certified=0)
        if row is None:rows.append(result);continue
        point=np.array([float(row['bottom_center_x']),float(row['bottom_center_y'])]);others=[s for s,r in group.items() if s!=slot and inside_rect(point,r)];bodies=[s for s,(pose,iou) in matches.items() if s!=slot and body_supported(pose)];inside=[s for s in bodies if inside_body(point,matches[s][0])];result.update(native_proxy_in_other_boxes='|'.join(others),native_proxy_in_other_pose_bodies='|'.join(inside),other_pose_bodies_evaluated='|'.join(bodies),geometry_box_overlap_risk=int(bool(others)),geometry_body_overlap_risk=int(bool(inside)))
        if slot in matches:
            person,iou=matches[slot];kp=np.asarray(person['keypoints']);height=float(row['y2'])-float(row['y1']);ankles=np.all(kp[[15,16],2]>=.5);result.update(own_pose_matched=1,own_pose_index=person['pose_index'],own_pose_iou=iou,own_body_approximation_available=int(body_supported(person)),both_model_ankles_above_threshold=int(ankles),native_height_over_model_pose_height=height/(person['box'][3]-person['box'][1]))
            if ankles:
                mid=kp[[15,16],:2].mean(axis=0);delta=(point-mid)/height;result.update(model_ankle_mid_x=mid[0],model_ankle_mid_y=mid[1],native_minus_model_ankle_dx_over_native_height=delta[0],native_minus_model_ankle_dy_over_native_height=delta[1],native_proxy_to_model_ankle_distance_over_native_height=float(np.linalg.norm(delta)))
        state='other_pose_body_overlap_risk' if inside else 'other_native_box_overlap_only' if others else 'model_ankles_available_not_verified' if result['both_model_ankles_above_threshold'] else 'pose_matched_ankles_unavailable' if result['own_pose_matched'] else 'no_matching_pose' if evaluated[mode] else 'pose_not_evaluated';result['position_support_state']=state;rows.append(result)
    return rows

def audit():
    p=load(OUT/'protocol.json');assert load(OUT/'pose_result.json')['status']=='complete';assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items());summaries=[];checks=0
    for e in p['cases']:
        tag=e['tag'];dest=OUT/f'case_{tag}';table=native_table(e);poses=all_pose_geometry(tag);diagnostics=[];by={}
        for f in range(e['initialization']['frame'],e['last']+1):
            rows=support_for_frame(tag,f,table[f],poses);diagnostics.extend(rows);by[f]={r['player_slot']:r for r in rows}
        write(dest/'position_proxy_support.csv',diagnostics);write(dest/'fixed_prefix_position_support.csv',[r for r in diagnostics if r['source_frame'] in e['requested_input_frames']]);reports=[]
        for policy in POLICIES:
            features=[];masked_rows=0
            for f in e['requested_input_frames']:
                native=table[f];kept={s:r for s,r in native.items() if policy=='retain_native_proxies' or not by[f][s]['geometry_box_overlap_risk' if policy=='mask_native_box_overlap' else 'geometry_body_overlap_risk']};masked_rows+=len(native)-len(kept);frame=dict(rally_candidate_id=f'match1_rally_window_{tag}',relative_source_frame=f-e['requested_input_frames'][0],**pair_features(kept));expected=calculate_frame(frame['rally_candidate_id'],{f:kept},f,frame['relative_source_frame'])
                for k,value in expected.items():
                    if k!='rally_candidate_id':close(frame[k],value);checks+=1
                features.append(frame)
            snapshot,summary,_=summarize_prefix(tag,[dict(r,ball_observation_present=0,ball_x_norm='',ball_y_norm='') for r in features]);expected_s,expected_m=calculate_summary(features)
            for actual,expected in [(snapshot,expected_s),(summary,expected_m)]:
                for k,value in expected.items():close(actual[k],value);checks+=1
            folder=dest/policy;folder.mkdir();write(folder/'diagnostic_prefix_frame_features.csv',features);write(folder/'diagnostic_snapshot_features.csv',[snapshot]);write(folder/'diagnostic_prefix_summary.csv',[summary]);reports.append(dict(tag=tag,policy=policy,requested_times=31,position_proxies_masked=masked_rows,near_pair_proxy_available_times=sum(r['near_pair_valid'] for r in features),far_pair_proxy_available_times=sum(r['far_pair_valid'] for r in features),four_proxy_times=sum(r['near_pair_valid'] and r['far_pair_valid'] for r in features),not_an_accuracy_estimate=1))
        write(dest/'sensitivity_summary.csv',reports);summaries.extend(reports)
    write(OUT/'fixed_prefix_mask_sensitivity_not_accuracy.csv',summaries);assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items());dump(OUT/'result.json',dict(status='position_support_and_sensitivity_complete',cases=10,search_frames=7019,sidecar_rows=7019*4,fixed_input_times=310,sensitivity_policies=POLICIES,recalculated_feature_values_checked=checks,native_ID_paths_changed=False,mask_policies_adopted=False,independent_position_accuracy_established=False));print(load(OUT/'result.json'),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','pose','audit'],required=True);args=ap.parse_args();{'prepare':prepare,'pose':infer,'audit':audit}[args.phase]()
