"""Matched far-crop candidate augmentation, with exact unchanged baseline replay."""
import sys,os,json
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from audit_match2_F1_missing import OUT,cached_case,TARGETS,PARENT,load
from run_fixed_cohort_player_case_v3 import ROOT,BASE,SOURCE,read,write,sha,dump
from calibrated_camera_temporal_tracker import calibrated_temporal_class,CameraGateMap
from confirmed_continuity_guard import adaptive,old
from match_frame_coordinate_adapter import FrameCoordinates
from prepare_match2_external_validation import VIDEO,FPS

DEST=OUT/'crop_trial'
MODES=['legacy_image_crop','court_margin_transferred_crop']


def project(H,points):
    q=np.column_stack([np.asarray(points),np.ones(len(points))])@np.linalg.inv(np.asarray(H)).T
    return q[:,:2]/q[:,2:]


def prepare():
    assert not DEST.exists();p=load(OUT/'diagnostic_protocol.json');H=np.array(p['image_to_court_homography_canonical'])
    legacy=SOURCE/'config/match1_court_plane_calibration.json';oldH=np.array(load(legacy)['image_to_court_homography'])
    old_far=project(oldH,[[0,1],[1,1]])/[3840,2160];new_far=project(H,[[0,1],[1,1]])/[3840,2160]
    old_net=project(oldH,[[0,.5],[1,.5]])/[3840,2160];new_net=project(H,[[0,.5],[1,.5]])/[3840,2160]
    original=np.array([.18,.43,.82,.68])
    transferred=[new_far[:,0].min()-(old_far[:,0].min()-.18),
                 new_far[:,1].min()-(old_far[:,1].min()-.43),
                 new_far[:,0].max()+(.82-old_far[:,0].max()),
                 new_net[:,1].max()+(.68-old_net[:,1].max())]
    assert 0<transferred[0]<transferred[2]<1 and 0<transferred[1]<transferred[3]<1
    crops={MODES[0]:[round(v*s) for v,s in zip(original,[3840,2160,3840,2160])],
           MODES[1]:[round(v*s) for v,s in zip(transferred,[3840,2160,3840,2160])]}
    files=[Path(__file__),OUT/'diagnostic_protocol.json',OUT/'diagnostic_result.json',ROOT/'calibrated_camera_temporal_tracker.py',
        ROOT/'audit_match2_F1_missing.py',SOURCE/'models/yolo11m.pt',legacy,
        Path(old.previous.__file__),Path(old.a.__file__),PARENT/'publication_hashes.json']
    DEST.mkdir();dump(DEST/'protocol.json',dict(cases=p['cases'],source_video=p['source_video'],source_video_sha256=p['source_video_sha256'],
        image_to_court_homography_canonical=H.tolist(),modes=MODES,crops_canonical_xyxy=crops,
        crop_normalized={MODES[0]:original.tolist(),MODES[1]:transferred},
        transferred_rule='Preserve legacy image-fraction margins around projected far-baseline x extents, far-baseline top and net-ground bottom. New/legacy calibration only; no person boxes, target-frame positions or parameter sweep used to choose crop.',
        warning='Image-fraction padding is an engineering camera adaptation, not a 3D body-height projection. Both ROIs are tested and retained. This development follows4 known missing frames, not independent test accuracy.',
        controls='All939 line-supported frames of both retained main-camera segments. No per-frame targeted crop, no replacement windows. Source899 frames per clip including cuts retained for replay.',
        full_frame_diagnostic_frames=p['diagnostic_neighborhood_frames'],
        full_frame_diagnostic='Same full-frame YOLO.predict before ByteTrack, at the11 target/neighborhood frames. Anonymous boxes saved only as detector-stage diagnostic; never inserted as persistent track IDs.',
        detector=dict(model=str(SOURCE/'models/yolo11m.pt'),conf=.25,imgsz=1280,tracker='bytetrack.yaml',device=0),
        proposal_rules='Prior3-canonical-pixel crop-edge exclusion,50-canonical-pixel minimum height, legacy far-foot band0.49..0.665 transferred with existing camera mapper. Duplicates:IoU>0.35 OR foot distance<0.22*min height and0.5<height ratio<2. Applied to full-frame and previously added candidates.',
        IDs='Crop ByteTrack IDs use1,000,000 offset within each clip/mode and camera segment; not manual person bindings. Full-frame IDs preserved. No anonymous per-frame boxes promoted to stable IDs.',
        association='Same frozen calibrated temporal tracker,8-frame initialization/recovery, same camera-reset mask and appearance extraction. Only candidate pool changes. No position interpolation or forced missing-frame replacement.',
        baseline_replay='Reproduce original selected native boxes/slots/costs before interpreting augmented results.',
        handover_and_initialization_changes_reported=True,all_added_coordinates_actual_crop_detection_boxes=True,
        independent_GT_added=False,policy_fixed_before_any_new_crop_or_full_predict_outputs=True,
        protected_input_hashes={str(n):sha(n) for n in files}))
    print('Frozen crop ROIs',crops,';all939 eligible frames; no targeted per-frame crop.',flush=True)


def run():
    p=load(DEST/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    assert sha(VIDEO)==p['source_video_sha256']
    os.environ['YOLO_CONFIG_DIR']=str(DEST/'ultralytics_config')
    from ultralytics import YOLO
    import torch,ultralytics
    assert torch.cuda.is_available();dump(DEST/'runtime.json',dict(torch=torch.__version__,ultralytics=ultralytics.__version__,device=torch.cuda.get_device_name(0)))
    adapter=FrameCoordinates(1920,1080);H=np.array(p['image_to_court_homography_canonical']);oldH=np.array(load(SOURCE/'config/match1_court_plane_calibration.json')['image_to_court_homography'])
    mapper=CameraGateMap(H,oldH);factory=calibrated_temporal_class(adaptive.AdaptiveNearTracker,H,oldH)
    diagnostics=set(p['full_frame_diagnostic_frames']);reports=[]
    for e in p['cases']:
        tag=e['clip_id'];folder=DEST/tag;folder.mkdir();by,vectors,line=cached_case(e)
        crop_models={m:YOLO(p['detector']['model']) for m in MODES};full_model=YOLO(p['detector']['model'])
        trackers={m:None for m in ['baseline']+MODES};output={m:[] for m in trackers};events={m:[] for m in trackers}
        crop_rows={m:[] for m in MODES};proposal_log={m:[] for m in MODES};native_prediction=[];availability=[]
        cap=cv2.VideoCapture(str(VIDEO));cap.set(1,e['first_source_frame']);eligible_count=0
        for f in range(e['first_source_frame'],e['last_source_frame']+1):
            if not line[f]:
                assert cap.grab()
                for m,tr in trackers.items():
                    if tr is not None:events[m].extend(tr.events)
                    trackers[m]=None
                for m in trackers:availability.append(dict(source_frame=f,mode=m,line_supported=0,assigned_slots=0,four_candidates=0))
                continue
            ok,im=cap.read();assert ok and round(cap.get(1))==f+1
            if trackers['baseline'] is None:
                trackers={m:factory() for m in trackers}
            original=by[f];original_features=[vectors[f,str(r['track_id'])] for r in original]
            canonical=adapter.image(im);pools={'baseline':original};feature_pools={'baseline':original_features}
            if f in diagnostics:
                pred=full_model.predict(canonical,classes=[0],conf=.25,imgsz=1280,device=0,verbose=False)[0]
                for j,b in enumerate(pred.boxes):
                    x1,y1,x2,y2=b.xyxy[0].cpu().tolist()
                    r=dict(source_frame=f,anonymous_detection_index=j,x1=x1/2,y1=y1/2,x2=x2/2,y2=y2/2,
                        bottom_center_x=(x1+x2)/4,bottom_center_y=y2/2,confidence=float(b.conf.item()),persistent_person_ID_assigned=False)
                    r['maximum_iou_with_saved_ByteTrack_boxes']=max([old.a.box_iou(dict(r),adapter.row(v,False)) for v in original] or [0.])
                    native_prediction.append(r)
            for mode in MODES:
                l,t,r,b=p['crops_canonical_xyxy'][mode];crop=canonical[t:b,l:r]
                pred=crop_models[mode].track(crop,persist=True,classes=[0],conf=.25,imgsz=1280,tracker='bytetrack.yaml',device=0,verbose=False)[0]
                added=[];added_features=[]
                for box in pred.boxes:
                    if box.id is None:continue
                    x1,y1,x2,y2=box.xyxy[0].cpu().tolist();border=min(x1,y1,crop.shape[1]-x2,crop.shape[0]-y2)
                    row=dict(source_frame=f,timestamp_sec=float(f/FPS),track_id=1000000+int(box.id.item()),x1=x1+l,y1=y1+t,x2=x2+l,y2=y2+t,
                        bottom_center_x=(x1+x2)/2+l,bottom_center_y=y2+t,confidence=float(box.conf.item()),candidate_origin=mode+'_extra')
                    q=mapper.legacy_foot(row);height=old.a.height(row)
                    duplicates=[]
                    for prior in original+added:
                        iou=old.a.box_iou(row,prior);dist=float(np.linalg.norm(old.a.foot(row)-old.a.foot(prior)))/min(height,old.a.height(prior))
                        ratio=height/max(1,old.a.height(prior))
                        if iou>.35 or (dist<.22 and .5<ratio<2):duplicates.append((iou,str(prior['track_id'])))
                    reasons=[]
                    if border<3:reasons.append('crop_border')
                    if height<50:reasons.append('height')
                    if q is None or not .49<q[1]/2160<.665:reasons.append('camera_mapped_far_band')
                    if duplicates:reasons.append('duplicate')
                    accepted=not reasons
                    proposal_log[mode].append(dict(adapter.row(row,False),border_distance_canonical=border,
                        accepted=int(accepted),reasons=';'.join(reasons),duplicate_raw_ids=';'.join(v[1] for v in sorted(duplicates,reverse=True)),
                        maximum_duplicate_iou=max([v[0] for v in duplicates] or [0.])))
                    if accepted:
                        assert all(int(v['track_id'])!=row['track_id'] for v in original+added)
                        added.append(row);added_features.append(old.a.appearance(im,row));crop_rows[mode].append(adapter.row(row,False))
                pools[mode]=original+added;feature_pools[mode]=original_features+added_features
            for mode,tr in trackers.items():
                current=pools[mode];assigned=tr.step(f,current,feature_pools[mode])
                for slot,j,margin,cost,origin in assigned:
                    output[mode].append(dict(adapter.row(current[j],False),player_slot=old.a.SLOTS[slot],identity_source=origin,
                        association_cost=float(cost),alternative_margin=float(margin),position_source='actual_detection_bbox_bottom_image_proxy',
                        independent_person_verified=0,independent_position_verified=0))
                availability.append(dict(source_frame=f,mode=mode,line_supported=1,assigned_slots=len(assigned),four_candidates=int(len(assigned)==4)))
            eligible_count+=1
            if eligible_count%180==0:print(tag,'crop eligible',eligible_count,flush=True)
        cap.release();del full_model,crop_models
        for m,tr in trackers.items():
            if tr is not None:events[m].extend(tr.events)
        expected=read(PARENT/tag/'frozen_tracker_reset_on_line_failure_native_observations.csv')
        baseline=output['baseline'];assert len(expected)==len(baseline)
        fields=['source_frame','track_id','player_slot','x1','y1','x2','y2','bottom_center_x','bottom_center_y']
        for a,b in zip(baseline,expected):
            assert all(str(a[k])==str(b[k]) if k in ['track_id','player_slot'] else abs(float(a[k])-float(b[k]))<1e-9 for k in fields)
            assert np.allclose([a['association_cost'],a['alternative_margin']],[float(b['association_cost']),float(b['alternative_margin'])],atol=1e-9,rtol=0)
        if native_prediction:write(folder/'full_predict_before_ByteTrack_diagnostic.csv',native_prediction)
        write(folder/'all_frame_availability_not_accuracy.csv',availability)
        results={}
        baseline_keys={(int(r['source_frame']),r['player_slot'],str(r['track_id']),*[float(r[k]) for k in ['x1','y1','x2','y2']]) for r in baseline}
        for mode in output:
            dest=folder/mode;dest.mkdir()
            if output[mode]:write(dest/'selected_native_observations.csv',output[mode])
            dump(dest/'events.json',events[mode]);a=[v for v in availability if v['mode']==mode]
            actual={(int(r['source_frame']),r['player_slot'],str(r['track_id']),*[float(r[k]) for k in ['x1','y1','x2','y2']]) for r in output[mode]}
            results[mode]=dict(observations=len(output[mode]),four_candidate_frames=sum(v['four_candidates'] for v in a),line_supported_frames=sum(v['line_supported'] for v in a),
                removed_baseline_rows=len(baseline_keys-actual),added_or_changed_rows=len(actual-baseline_keys),
                observations_at_known_missing_frames=[dict(source_frame=int(r['source_frame']),player_slot=r['player_slot'],raw_id=r['track_id'],origin=r['candidate_origin']) for r in output[mode] if int(r['source_frame']) in TARGETS and r['player_slot']=='far_rear'])
            if mode in MODES:
                if crop_rows[mode]:write(dest/'added_native_crop_candidates.csv',crop_rows[mode])
                if proposal_log[mode]:write(dest/'all_crop_proposals_with_rejections.csv',proposal_log[mode])
                results[mode].update(crop_proposals=len(proposal_log[mode]),accepted_extra_candidates=len(crop_rows[mode]))
            print(tag,mode,results[mode],flush=True)
        report=dict(clip_id=tag,baseline_native_path_and_costs_reproduced=True,eligible_crop_frames=eligible_count,modes=results)
        dump(folder/'result.json',report);reports.append(report)
    dump(DEST/'result.json',dict(cases=reports,exact_baseline_replays=2,known_missing_frames_used_for_development=True,
        independent_accuracy_established=False,coordinate_interpolation=False,previous_results_or_labels_changed=False))
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items()) and sha(VIDEO)==p['source_video_sha256']


if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
