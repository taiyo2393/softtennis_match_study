"""Matched offline association, using cached poses and separate missing positions."""
import sys,json
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
from run_near_prefix_initialization_trial import load
from run_position_proxy_support_head_independent import all_pose_geometry
from run_body_part_localization_probe import one_to_one,overlap
from diagnose_initial_pair_appearance import cache
from partial_body_identity import BodyIdentity,body_feature,PARAMETERS

OUT=BASE/'partial_body_identity_trial'
PRIOR=BASE/'camera_valid_near_prefix_trial'
SLOTS=['near_front','near_rear','far_rear','far_front']
MODES=['unmasked_torso','exclude_near_rectangles']


def rect(r):return [float(r[k]) for k in ['x1','y1','x2','y2']]


def match(rows,people):
    scores=np.array([[overlap(rect(r),p['box'],union=True) for p in people] for r in rows]).reshape(len(rows),len(people))
    return {i:j for i,j in one_to_one(scores).items() if j<len(people)}


def prepare():
    assert not OUT.exists();p=load(PRIOR/'protocol.json');cases=[]
    protected={str(n):sha(n) for n in [Path(__file__),ROOT/'partial_body_identity.py',
        ROOT/'run_body_part_localization_probe.py',ROOT/'run_position_proxy_support_head_independent.py',
        PRIOR/'protocol.json',PRIOR/'publication_hashes.json',BASE/'far_initialization_diagnostic_006/publication_hashes.json',
        BASE/'body_part_localization_probe/pose_predictions.json',BASE/'body_part_localization_probe/temporal_demo/pose_predictions.json']}
    for old in p['cases']:
        e=dict(old);tag=e['tag'];poses=all_pose_geometry(tag)
        event=load(Path(e['temporal_folder'])/'events.json')[0];start=event['frame'];end=max(e['requested_input_frames'])
        schedule={f for f,mode in poses if mode=='far_context' and start<=f<=end}
        schedule.update(range(event['window_start'],start+1));schedule.update(e['requested_input_frames'])
        assert all((f,'far_context') in poses for f in schedule)
        e.update(initialization=event,feature_frames=sorted(schedule));cases.append(e)
        for n in [Path(e['raw_csv']),Path(e['cache'])/'cached_detections.csv',Path(e['cache'])/'appearance_cache.npz',
            Path(e['temporal_folder'])/'events.json',PRIOR/f'case_{tag}/candidate_native_observations.csv',
            BASE/f'torso_temporal_association_trial/case_{tag}/pose_tensors.npz',
            BASE/f'torso_temporal_association_trial/case_{tag}/pose_metadata.csv',
            BASE/f'position_proxy_support_trial/case_{tag}/pose_geometry.npz',
            BASE/f'position_proxy_support_trial/case_{tag}/pose_metadata.csv']:
            protected[str(n)]=sha(n)
    for n in (BASE/'current22_player_inputs').iterdir():
        if n.is_file():protected[str(n)]=sha(n)
    OUT.mkdir()
    dump(OUT/'protocol.json',dict(cases=cases,source_video=p['source_video'],source_video_sha256=p['source_video_sha256'],
        parameters=PARAMETERS,modes=MODES,
        feature='HSV torso polygon from cached shoulders/hips confidence>=.5, at least12 pixels. Second mode subtracts all current selected near-player native rectangles. No fallback feature. Confidence and mask support are not visible-pixel truth.',
        anchors='Same inherited common initialization8 source frames/native seed IDs, IoU>=.5 one-to-one pose correspondence. Far torso anchors from far_context; near negatives from court. Unselected initial poses require min-area overlap<.1 with all four native seeds.',
        association='Conservative2-person joint assignment with inherited appearance/negative/motion/size constants. Frozen galleries, no new feature learning, no frame/ID-specific rules. Two assignments must be different poses and have overlap<=.55.',
        temporal='Forward through all cached far-context frames from common initialization to fixed input end. For006, separately reverse from common initialization through all earlier fixed input times, using the same gallery and motion rules. No online claim or interpolation.',
        output='Body identity hypotheses only, one row per fixed time and far slot, including unsupported. analysis_x/analysis_y always blank. Pose box and torso centroid are internal diagnostics, never exported as observed foot/ground positions.',
        comparison='Existing native paths are model correspondence controls, not person ground truth. Counts and agreements cannot establish identity accuracy. No existing paths replaced.',
        fixed_times_per_case=31,fixed_contexts=10,methods_fixed_before_new_features_and_associations=True,
        same_match_development=True,new_pose_inference=False,offline_future_anchors_for_006=True,
        canonical_pipeline_replaced=False,independent_GT_added=False,protected_input_hashes=protected))
    print('Frozen10 contexts;',sum(len(e['feature_frames']) for e in cases),'feature frames.',flush=True)


def features():
    p=load(OUT/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    cap=cv2.VideoCapture(p['source_video']);reports=[]
    for e in p['cases']:
        tag=e['tag'];dest=OUT/f'case_{tag}';dest.mkdir();poses=all_pose_geometry(tag)
        near=defaultdict(list)
        for r in read(PRIOR/f'case_{tag}/candidate_native_observations.csv'):
            if r['player_slot'].startswith('near_'):near[int(r['source_frame'])].append(rect(r))
        records=[];vectors=[];wanted=set(e['feature_frames']);init=e['initialization']
        cap.set(1,min(wanted))
        for f in range(min(wanted),max(wanted)+1):
            if f not in wanted:assert cap.grab();continue
            ok,im=cap.read();assert ok and round(cap.get(1))==f+1
            modes=['far_context','court'] if init['window_start']<=f<=init['frame'] else ['far_context']
            for mode in modes:
                for person in poses[f,mode]:
                    for policy in MODES:
                        masks=near[f] if policy=='exclude_near_rectangles' and mode=='far_context' else []
                        v,count,original=body_feature(im,person,masks)
                        index=len(vectors);vectors.append(v if v is not None else np.full(192,np.nan,np.float32))
                        records.append(dict(source_frame=f,mode=mode,pose_index=person['pose_index'],policy=policy,
                            array_row=index,pixels=count,original_pixels=original,feature_available=int(v is not None)))
        np.savez_compressed(dest/'body_features.npz',features=np.asarray(vectors,np.float32))
        write(dest/'feature_metadata.csv',records)
        report=dict(tag=tag,source_images_decoded=len(wanted),body_feature_rows=len(records),
            available=sum(r['feature_available'] for r in records),new_pose_inference_calls=0)
        dump(dest/'feature_result.json',report);reports.append(report);print(report,flush=True)
    cap.release();dump(OUT/'feature_result.json',dict(cases=reports))


def compare():
    p=load(OUT/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    all_results=[];all_comparisons=[];reports=[]
    for e in p['cases']:
        tag=e['tag'];dest=OUT/f'case_{tag}';poses=all_pose_geometry(tag);event=e['initialization'];start=event['frame']
        data=np.load(dest/'body_features.npz')['features'];feats={}
        for r in read(dest/'feature_metadata.csv'):
            v=data[int(r['array_row'])];feats[int(r['source_frame']),r['mode'],int(r['pose_index']),r['policy']]=None if np.isnan(v).all() else v
        raw,_,_=cache(e);by=defaultdict(list)
        for r in raw:by[int(r['source_frame'])].append(r)
        baseline=defaultdict(dict)
        for r in read(PRIOR/f'case_{tag}/candidate_native_observations.csv'):baseline[int(r['source_frame'])][r['player_slot']]=r
        mode_reports={}
        for policy in MODES:
            anchors={s:[] for s in SLOTS};negative=[];initial_people={};proof=[]
            for f in range(event['window_start'],start+1):
                native=[next(r for r in by[f] if int(r['track_id'])==event['seed_ids'][s]) for s in SLOTS]
                for mode in ['court','far_context']:
                    people=[r for r in poses[f,mode] if r['eligible_not_crop_truncated']]
                    matches=match(native,people)
                    for i,j in matches.items():
                        slot=SLOTS[i]
                        if mode!=('far_context' if slot.startswith('far_') else 'court'):continue
                        person=people[j];v=feats[f,mode,person['pose_index'],policy]
                        if v is not None:anchors[slot].append(v)
                        if f==start and slot.startswith('far_'):initial_people[slot]=person
                        proof.append(dict(policy=policy,source_frame=f,slot=slot,native_seed=event['seed_ids'][slot],pose_index=person['pose_index'],feature_available=int(v is not None)))
                    if mode=='far_context' and f==start:
                        for person in people:
                            v=feats[f,mode,person['pose_index'],policy]
                            if v is not None and all(overlap(person['box'],rect(r))<.1 for r in native):negative.append(v)
            negative+=anchors['near_front']+anchors['near_rear']
            outputs={start:{slot:dict(person=person,cost=0.,margin=0.) for slot,person in initial_people.items() if anchors[slot]}}
            sequences=[sorted(f for f in e['feature_frames'] if f>start),sorted([f for f in e['requested_input_frames'] if f<start],reverse=True)]
            for sequence in sequences:
                tracker=BodyIdentity(anchors,negative,initial_people,start)
                for f in sequence:
                    people=[r for r in poses[f,'far_context'] if r['eligible_not_crop_truncated']]
                    near=[r for slot,r in baseline[f].items() if slot.startswith('near_')]
                    near_indices={people[j]['pose_index'] for j in match(near,people).values()}
                    vectors=[feats[f,'far_context',person['pose_index'],policy] for person in people]
                    outputs[f]=tracker.step(f,people,vectors,near_indices)
            results=[]
            for f in e['requested_input_frames']:
                assigned=outputs.get(f,{})
                for slot in ['far_rear','far_front']:
                    value=assigned.get(slot);person=value['person'] if value else None
                    ref=baseline[f].get(slot)
                    correspondence=overlap(rect(ref),person['box'],union=True) if ref and person else None
                    result=dict(tag=tag,policy=policy,source_frame=f,player_slot=slot,
                        body_ID_candidate=int(person is not None),pose_index=person['pose_index'] if person else '',
                        body_box_json=json.dumps(person['box']) if person else '',
                        identity_status='offline_body_hypothesis_not_GT' if person else 'unsupported',
                        analysis_x='',analysis_y='',position_status='not_estimated_body_only_trial',
                        association_cost=value['cost'] if value else '',assignment_margin=value['margin'] if value else '',
                        baseline_native_present=int(ref is not None),baseline_native_id=ref['track_id'] if ref else '',
                        baseline_same_slot_box_iou=correspondence if correspondence is not None else '',
                        uses_future_common_anchor=int(f<start),independent_identity_verified=0)
                    results.append(result);all_results.append(result)
            write(dest/f'{policy}_fixed_body_hypotheses.csv',results)
            write(dest/f'{policy}_anchor_proof.csv',proof)
            dump(dest/f'{policy}_gallery.json',dict(anchor_counts={s:len(v) for s,v in anchors.items()},negative_count=len(negative)))
            mode_reports[policy]=dict(requested_slot_times=62,body_candidates=sum(r['body_ID_candidate'] for r in results),
                candidates_where_baseline_missing=sum(r['body_ID_candidate'] and not r['baseline_native_present'] for r in results),
                compared_to_existing_native=sum(r['body_ID_candidate'] and r['baseline_native_present'] for r in results),
                native_same_slot_iou_at_least_half=sum(r['baseline_same_slot_box_iou']!='' and r['baseline_same_slot_box_iou']>=.5 for r in results),
                position_coordinates_created=0)
            print(tag,policy,mode_reports[policy],flush=True)
        report=dict(tag=tag,modes=mode_reports);reports.append(report);dump(dest/'result.json',report)
    write(OUT/'all_fixed_body_hypotheses.csv',all_results)
    dump(OUT/'result.json',dict(cases=reports,fixed_contexts=10,fixed_times=310,
        body_hypothesis_rows=len(all_results),independent_accuracy_established=False,
        position_coordinates_created=0,canonical_pipeline_replaced=False))
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())


if __name__=='__main__':{'prepare':prepare,'features':features,'compare':compare}[sys.argv[1]]()
