"""Prospective fixed-time transfer of frozen full/crop candidate association.

Not an independent match or an accuracy benchmark without person/position GT.
Full and crop detector passes are sequential to isolate ByteTrack ID counters.
"""
import os, sys, json, shutil
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np
from run_fixed_cohort_player_case_v3 import ROOT, BASE, SOURCE, read, write, sha, dump
from prepare_match2_external_validation import VIDEO, FPS, load
from calibrated_camera_temporal_tracker import calibrated_temporal_class, CameraGateMap
from confirmed_continuity_guard import adaptive, old
from match_frame_coordinate_adapter import FrameCoordinates
from fit_match2_court_geometry_centers import fit_line_centers
import fit_match2_court_geometry as geometry

OUT=BASE/'match2_unseen_crop_transfer'
PRIOR=BASE/'match2_F1_missing_development'
MODES=['full_only','full_plus_frozen_camera_crop']

def save(p, rows):
    if rows: write(p,rows)

def prepare():
    assert not OUT.exists()
    previous=load(PRIOR/'crop_trial/protocol.json');cases=[]
    for i,t in enumerate([1200,1800,2400],1):
        first=round(t*FPS);last=round((t+10)*FPS)-1
        cases.append(dict(clip_id=f'match2_unseen_{i:03}',nominal_start_sec=t,duration_sec=10,
            first_source_frame=first,last_source_frame=last,source_frames=last-first+1,
            fixed_review_source_frames=[round((t+d)*FPS) for d in [1,5,9]],
            fixed_availability_source_frames=[round(v) for v in np.linspace(first,last,31)],
            selection='Elapsed20/30/40 minutes,10 seconds each. Fixed before any new raw previews, person predictions or line eligibility. Never replace unsuccessful windows.',
            rally_or_in_play_status='unknown; camera tracking stress windows, not labeled rallies'))
    files=[Path(__file__),PRIOR/'publication_hashes.json',PRIOR/'crop_trial/protocol.json',
        ROOT/'calibrated_camera_temporal_tracker.py',ROOT/'match_frame_coordinate_adapter.py',
        ROOT/'fit_match2_court_geometry.py',ROOT/'fit_match2_court_geometry_centers.py',
        Path(old.__file__),Path(old.previous.__file__),Path(old.a.__file__),Path(adaptive.__file__),
        SOURCE/'config/match1_court_plane_calibration.json',SOURCE/'models/yolo11m.pt']
    files.extend(n for n in (BASE/'current22_player_inputs').iterdir() if n.is_file())
    OUT.mkdir()
    dump(OUT/'protocol.json',dict(cases=cases,source_video=str(VIDEO),source_video_sha256=previous['source_video_sha256'],
        image_to_court_homography_canonical=previous['image_to_court_homography_canonical'],
        crop_canonical_xyxy=previous['crops_canonical_xyxy']['court_margin_transferred_crop'],
        detector=previous['detector'],modes=MODES,
        association='Unmodified calibrated temporal class, fixed appearance and motion/height/margin gates.8-frame initialization/recovery. Both arms consume identical full-frame stream. Only extra frozen crop candidates differ.',
        camera='Same4-line center support test every native frame. No recalibration, hysteresis or threshold tuning. Clear assignment state and pending initialization on any unsupported frame; segment-local identities only.',
        crop='Frozen prior new-camera ROI,3px canonical border exclusion,50px minimum canonical height, camera-mapped far band0.49..0.665. DedupIoU>0.35 or foot distance<0.22*min height with0.5<height ratio<2. Same rules as prior. Crop ByteTrack reset on line failure. Sequential model passes avoid global counter interference.',
        identity_scope='clip:segment:slot only; F1/F2/N1/N2 are not ground-truth identity or cross-cut identity links.',
        position_contract='Store actual native box and image-bottom proxy separately. A proxy projected by H is never a validated foot position. measured_court_x/y remain empty without independent placement/reference; no interpolation. Do not feed upper-body bbox bottoms into rally-position/velocity analyses.',
        stage_scope='Camera tracking stress test only; no ball inference, rally/score labels or outcome features. Non-rally frames may be present, never count them as labeled in-play evidence.',
        evaluation='Report all fixed windows and all source-frame denominators, camera support, initialization and missing observations. Compare exact preserved/changed selected observations. Identity switches/false positive rates remain unmeasurable without independent references. AI review flags are diagnostic, not GT.',
        exposure='Same match already used for camera/crop development. No person inference known in these time windows. Raw previews/cadence at1800/2700 exist in original registration, so these are not fully blinded external test data. No numerical tuning permitted after outputs.',
        policy_fixed_before_new_predictions=True,coordinate_imputation=False,independent_GT_added=False,
        protected_input_hashes={str(n):sha(n) for n in files}))
    print('Frozen20/30/40min x10s; two arms, same source frames; no new outputs inspected.',flush=True)

def detect():
    p=load(OUT/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    assert sha(VIDEO)==p['source_video_sha256']
    geometry.fit_line=fit_line_centers
    os.environ['YOLO_CONFIG_DIR']=str(OUT/'ultralytics_config')
    from ultralytics import YOLO
    import torch,ultralytics
    assert torch.cuda.is_available()
    dump(OUT/'runtime.json',dict(torch=torch.__version__,ultralytics=ultralytics.__version__,device=torch.cuda.get_device_name(0)))
    adapter=FrameCoordinates(1920,1080);H=np.array(p['image_to_court_homography_canonical'])
    mapper=CameraGateMap(H,np.array(load(SOURCE/'config/match1_court_plane_calibration.json')['image_to_court_homography']))
    for e in p['cases']:
        dest=OUT/e['clip_id'];dest.mkdir();raw=[];meta=[];vectors=[];line=[]
        detector=YOLO(p['detector']['model']);cap=cv2.VideoCapture(str(VIDEO));cap.set(1,e['first_source_frame'])
        for f in range(e['first_source_frame'],e['last_source_frame']+1):
            ok,im=cap.read();assert ok and round(cap.get(1))==f+1
            proposal,diag=geometry.geometry(im)
            line.append(dict(source_frame=f,line_supported=int(proposal is not None),failed_lines=';'.join(k for k,v in diag.items() if not v['accepted'])))
            if f in e['fixed_review_source_frames']:cv2.imwrite(str(dest/f'raw_F{f}.png'),im)
            pred=detector.track(adapter.image(im),persist=True,classes=[0],conf=.25,imgsz=1280,tracker='bytetrack.yaml',device=0,verbose=False)[0]
            for b in pred.boxes:
                if b.id is None:continue
                x1,y1,x2,y2=b.xyxy[0].cpu().tolist()
                r=dict(source_frame=f,timestamp_sec=f/float(FPS),track_id=int(b.id.item()),x1=x1,y1=y1,x2=x2,y2=y2,
                    bottom_center_x=(x1+x2)/2,bottom_center_y=y2,confidence=float(b.conf.item()),candidate_origin='full_frame')
                v=old.a.appearance(im,r);raw.append(adapter.row(r,False))
                meta.append(dict(source_frame=f,track_id=r['track_id'],array_row=len(vectors)))
                vectors.append(v if v is not None else np.full((2,192),np.nan,np.float32))
            if (f-e['first_source_frame'])%240==0:print(e['clip_id'],'full',f,flush=True)
        cap.release();del detector
        save(dest/'native_full_frame_detections.csv',raw);save(dest/'appearance_metadata.csv',meta)
        write(dest/'line_support.csv',line);np.savez_compressed(dest/'appearance_features.npz',features=np.asarray(vectors,np.float32))
        # Full pass is finished before crop tracker initialization: independent ID namespace.
        cropmodel=YOLO(p['detector']['model']);by=defaultdict(list)
        for r in raw:by[int(r['source_frame'])].append(adapter.row(r,True))
        lines={r['source_frame']:r['line_supported'] for r in line}
        extra=[];logs=[];extra_features=[];extra_meta=[];pending_reset=False
        cap=cv2.VideoCapture(str(VIDEO));cap.set(1,e['first_source_frame'])
        for f in range(e['first_source_frame'],e['last_source_frame']+1):
            if not lines[f]:assert cap.grab();pending_reset=True;continue
            ok,im=cap.read();assert ok and round(cap.get(1))==f+1
            if pending_reset and hasattr(cropmodel,'predictor') and cropmodel.predictor is not None and hasattr(cropmodel.predictor,'trackers'):
                for tr in cropmodel.predictor.trackers:tr.reset()
            pending_reset=False
            l,t,r,b=p['crop_canonical_xyxy'];crop=adapter.image(im)[t:b,l:r]
            pred=cropmodel.track(crop,persist=True,classes=[0],conf=.25,imgsz=1280,tracker='bytetrack.yaml',device=0,verbose=False)[0]
            added=[]
            for box in pred.boxes:
                if box.id is None:continue
                x1,y1,x2,y2=box.xyxy[0].cpu().tolist();border=min(x1,y1,crop.shape[1]-x2,crop.shape[0]-y2)
                row=dict(source_frame=f,timestamp_sec=f/float(FPS),track_id=1000000+int(box.id.item()),x1=x1+l,y1=y1+t,x2=x2+l,y2=y2+t,
                    bottom_center_x=(x1+x2)/2+l,bottom_center_y=y2+t,confidence=float(box.conf.item()),candidate_origin='frozen_camera_crop_extra')
                q=mapper.legacy_foot(row);height=old.a.height(row);reasons=[]
                if border<3:reasons.append('crop_border')
                if height<50:reasons.append('height')
                if q is None or not .49<q[1]/2160<.665:reasons.append('camera_mapped_far_band')
                duplicate=[]
                for v in by[f]+added:
                    iou=old.a.box_iou(row,v);distance=np.linalg.norm(old.a.foot(row)-old.a.foot(v))/min(height,old.a.height(v))
                    ratio=height/max(1,old.a.height(v))
                    if iou>.35 or (distance<.22 and .5<ratio<2):duplicate.append(v['track_id'])
                if duplicate:reasons.append('duplicate')
                logs.append(dict(adapter.row(row,False),accepted=int(not reasons),reasons=';'.join(reasons)))
                if not reasons:
                    added.append(row);extra.append(adapter.row(row,False));v=old.a.appearance(im,row)
                    extra_meta.append(dict(source_frame=f,track_id=row['track_id'],array_row=len(extra_features)))
                    extra_features.append(v if v is not None else np.full((2,192),np.nan,np.float32))
            if (f-e['first_source_frame'])%240==0:print(e['clip_id'],'crop',f,flush=True)
        cap.release();del cropmodel
        save(dest/'crop_proposals.csv',logs);save(dest/'native_extra_crop_candidates.csv',extra);save(dest/'extra_appearance_metadata.csv',extra_meta)
        np.savez_compressed(dest/'extra_appearance_features.npz',features=np.asarray(extra_features,np.float32))
        dump(dest/'detection_stage_complete.json',dict(source_frames=len(line),line_supported_frames=sum(v['line_supported'] for v in line),
            full_candidates=len(raw),extra_crop_candidates=len(extra),crop_proposals=len(logs),counter_isolation='Sequential passes; crop tracker reset at each geometry interruption. IDs segment-local.'))
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())

def cached(folder,stem,metadata,features):
    by=defaultdict(list);vec={};adapter=FrameCoordinates(1920,1080)
    if not (folder/stem).exists():return by,vec
    for r in read(folder/stem):by[int(r['source_frame'])].append(adapter.row(r,True))
    values=np.load(folder/features)['features']
    for r in read(folder/metadata):
        v=values[int(r['array_row'])];vec[int(r['source_frame']),str(r['track_id'])]=None if np.isnan(v).all() else v
    return by,vec

def associate():
    p=load(OUT/'protocol.json');adapter=FrameCoordinates(1920,1080)
    H=np.array(p['image_to_court_homography_canonical']);oldH=np.array(load(SOURCE/'config/match1_court_plane_calibration.json')['image_to_court_homography'])
    factory=calibrated_temporal_class(adaptive.AdaptiveNearTracker,H,oldH);results=[]
    for e in p['cases']:
        folder=OUT/e['clip_id'];full,fv=cached(folder,'native_full_frame_detections.csv','appearance_metadata.csv','appearance_features.npz')
        extras,ev=cached(folder,'native_extra_crop_candidates.csv','extra_appearance_metadata.csv','extra_appearance_features.npz')
        line={int(r['source_frame']):r['line_supported']=='1' for r in read(folder/'line_support.csv')}
        rows={m:[] for m in MODES};events={m:[] for m in MODES};trs=None;segment=0;availability=[]
        for f in range(e['first_source_frame'],e['last_source_frame']+1):
            if not line[f]:
                if trs is not None:
                    for m,tr in trs.items():events[m].extend(dict(v,segment_id=segment) for v in tr.events)
                trs=None
            elif trs is None:segment+=1;trs={m:factory() for m in MODES}
            for m in MODES:
                assigned=[]
                if line[f]:
                    current=full[f]+(extras[f] if m==MODES[1] else [])
                    vectors=[(ev if r['candidate_origin']=='frozen_camera_crop_extra' else fv)[f,str(r['track_id'])] for r in current]
                    assigned=trs[m].step(f,current,vectors)
                    for slot,j,margin,cost,origin in assigned:
                        r=adapter.row(current[j],False)
                        # Deliberately do not create a measured physical position from an image rectangle.
                        rows[m].append(dict(r,player_slot=old.a.SLOTS[slot],identity_namespace=f'{e["clip_id"]}:S{segment:03}',
                            identity_source=origin,association_cost=float(cost),alternative_margin=float(margin),
                            position_source='unverified_image_bbox_bottom_proxy',measured_court_x='',measured_court_y='',
                            position_measurement_validated=0,independent_person_verified=0,independent_position_verified=0))
                availability.append(dict(source_frame=f,mode=m,line_supported=int(line[f]),segment_id=segment if line[f] else '',assigned_slots=len(assigned),four_candidates=int(len(assigned)==4),fixed_sample=int(f in e['fixed_availability_source_frames'])))
        if trs is not None:
            for m,tr in trs.items():events[m].extend(dict(v,segment_id=segment) for v in tr.events)
        write(folder/'all_frame_availability_not_accuracy.csv',availability);counts={}
        for m in MODES:
            (folder/m).mkdir();save(folder/m/'selected_native_observations.csv',rows[m]);dump(folder/m/'events.json',events[m])
            counts[m]=dict(observations=len(rows[m]),four_candidate_frames=sum(r['four_candidates'] for r in availability if r['mode']==m),
                initialized_segments=sum(v['event']=='initialized' for v in events[m]),
                crop_selected_observations=sum(r['candidate_origin']=='frozen_camera_crop_extra' for r in rows[m]))
        def k(r):return (int(r['source_frame']),r['player_slot'],str(r['track_id']),*[float(r[x]) for x in ['x1','y1','x2','y2']])
        a={k(r) for r in rows[MODES[0]]};b={k(r) for r in rows[MODES[1]]}
        changed=[dict(source_frame=v[0],player_slot=v[1],track_id=v[2],change='baseline_removed_or_changed',x1=v[3],y1=v[4],x2=v[5],y2=v[6]) for v in sorted(a-b)]
        changed.extend(dict(source_frame=v[0],player_slot=v[1],track_id=v[2],change='augmented_added_or_changed',x1=v[3],y1=v[4],x2=v[5],y2=v[6]) for v in sorted(b-a))
        save(folder/'changed_selected_observations.csv',changed)
        report=dict(clip_id=e['clip_id'],source_frames=e['source_frames'],line_supported_frames=sum(line.values()),camera_segments=segment,
            modes=counts,baseline_observations_preserved=len(a&b),baseline_removed_or_changed=len(a-b),augmented_added_or_changed=len(b-a),
            independently_established_identity_switches=None,independent_accuracy_established=False,validated_physical_positions=0)
        dump(folder/'result.json',report);results.append(report);print(report,flush=True)
    dump(OUT/'result.json',dict(cases=results,unseen_person_tracking_windows=3,same_match_development_transfer=True,
        phase_labels_created=False,independent_accuracy_established=False,validated_physical_positions=0))
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())

if __name__=='__main__':{'prepare':prepare,'detect':detect,'associate':associate}[sys.argv[1]]()
