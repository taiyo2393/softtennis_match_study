"""Prospective first-point camera-gate feasibility, not final ID accuracy."""
import sys,os,json,shutil
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,SOURCE,read,write,sha,dump
from prepare_match2_external_validation import VIDEO,FPS,load
from calibrated_camera_temporal_tracker import calibrated_temporal_class,old
from pair_initialization_guard import pair_initialization_class
from confirmed_continuity_guard import adaptive
from run_far_identity_guard_trial import case_plans
from diagnose_initial_pair_appearance import cache
from match_frame_coordinate_adapter import FrameCoordinates

OUT=BASE/'match2_camera_transfer_pilot'
SETUP=BASE/'match2_camera_setup'
LEGACY=SOURCE/'config/match1_court_plane_calibration.json'


def prepare():
    assert not OUT.exists();geo=load(SETUP/'geometry_center_trial/result.json')['anchor_geometry']
    a=FrameCoordinates(1920,1080);H=a.native_homography_in_canonical_coordinates(geo['image_to_court_homography'])
    first,last=round(205*FPS),round(209*FPS)-1
    files=[Path(__file__),ROOT/'calibrated_camera_temporal_tracker.py',ROOT/'match_frame_coordinate_adapter.py',
        ROOT/'pair_initialization_guard.py',ROOT/'court_initialization_guard.py',Path(old.__file__),Path(old.a.__file__),
        Path(old.previous.__file__),Path(adaptive.__file__),LEGACY,SOURCE/'models/yolo11m.pt',
        SETUP/'geometry_center_trial/protocol.json',SETUP/'geometry_center_trial/result.json',SETUP/'first_point_raw/result.json',
        BASE/'match2_external_validation/publication_hashes.json']
    controls=case_plans()
    for e in controls:files.extend([Path(e['raw_csv']),Path(e['cache'])/'cached_detections.csv',Path(e['cache'])/'appearance_cache.npz'])
    files.extend(n for n in (BASE/'current22_player_inputs').iterdir() if n.is_file())
    OUT.mkdir();dump(OUT/'protocol.json',dict(source_video=str(VIDEO),source_video_sha256=load(BASE/'match2_external_validation/source_registration.json')['source_video_sha256'],
        first_source_frame=first,last_source_frame=last,nominal_interval_sec=[205,209],
        selection='First scored-point main-view excerpt, chosen from raw200..211s inspection before any match2 person detection. Partial point context, not exact serve-to-end annotation.',
        exact_serve_or_point_end_human_confirmed=False,score_winner_used=False,
        native_dimensions=[1920,1080],canonical_dimensions=[3840,2160],
        image_to_court_homography_canonical=H.tolist(),image_to_court_homography_native=geo['image_to_court_homography'],
        calibration_source_frame=54006,calibration_future_raw_image_used=True,
        calibration_warning='Provided AI/white-line calibration from a later raw main-view frame; no person outputs or tracking labels. Offline camera setup, not automatic online calibration or surveyed truth.',
        modes=['resolution_only_legacy_gates','court_mapped_camera_gates'],
        comparison='Identical full-frameYOLO11m/ByteTrack detections and color features. First arm uses legacy image-region thresholds with new court envelope. Second maps only seed/side gates via new/old court planes. Motion, box overlap, appearance, temporal confirmation and real output coordinates unchanged.',
        detector=dict(model=str(SOURCE/'models/yolo11m.pt'),confidence=.25,imgsz=1280,device=0,tracker='bytetrack.yaml'),
        appearance='Native1080 image is exactly the half-resolution image expected for canonical4K box features. No invented clothing features or future player IDs.',
        output='All exported boxes/foot proxies converted back to original1080 coordinates. Missing remains missing, no interpolation.',
        limitations='Full-frame temporal feasibility only: no crop augmentation, offline far beam solver or torso auxiliary recovery. Do not call this the complete best pipeline or an independent tracking accuracy result.',
        development_control_replay='Same ten cached contexts, original calibration mapped to itself. Require matching assignments and numerical costs for original vs camera-adapted temporal trackers.',
        controls=controls,policy_fixed_before_any_match2_person_predictions=True,
        independent_person_or_position_GT=False,
        protected_input_hashes={str(n):sha(n) for n in files}))
    print('Frozen2 camera-gate arms,first-point205..209s and10 old-context controls.',flush=True)


def controls():
    p=load(OUT/'protocol.json');H=np.array(load(LEGACY)['image_to_court_homography']);reports=[]
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    for e in p['controls']:
        raw,features,offset=cache(e);base=adaptive.AdaptiveNearTracker if e.get('adaptive_size',True) else old.TemporalTracker
        prior=pair_initialization_class(base,H)();new=calibrated_temporal_class(base,H,H)();assigned=0
        for i,f in enumerate(range(e['first'],e['last']+1)):
            rows=raw[offset[i]:offset[i+1]];vectors=[None if np.isnan(v).all() else v for v in features[offset[i]:offset[i+1]]]
            a=prior.step(f,rows,vectors);b=new.step(f,rows,vectors)
            assert len(a)==len(b),(e['tag'],f)
            for x,y in zip(a,b):
                assert x[:2]==y[:2] and x[4]==y[4] and np.allclose(x[2:4],y[2:4],rtol=0,atol=1e-9),(e['tag'],f,x,y)
            assigned+=len(a)
        assert prior.initialized_frame==new.initialized_frame
        report=dict(tag=e['tag'],all_source_frames=e['last']-e['first']+1,matching_assignments=assigned,
            initialization_frame=new.initialized_frame,assignments_and_costs_match=True,old_camera_identity_transform=True)
        reports.append(report);print(report,flush=True)
    dump(OUT/'old_context_regression.json',dict(status='passed',cases=reports,contexts=10,
        numeric_tolerance=1e-9,independent_accuracy_claim=False))


def transfer():
    p=load(OUT/'protocol.json');assert load(OUT/'old_context_regression.json')['status']=='passed'
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    assert sha(VIDEO)==p['source_video_sha256']
    os.environ['YOLO_CONFIG_DIR']=str(OUT/'ultralytics_config')
    from ultralytics import YOLO
    import torch,ultralytics
    assert torch.cuda.is_available()
    dump(OUT/'runtime.json',dict(torch=torch.__version__,ultralytics=ultralytics.__version__,device=torch.cuda.get_device_name(0)))
    adapter=FrameCoordinates(1920,1080);H=np.array(p['image_to_court_homography_canonical']);oldH=np.array(load(LEGACY)['image_to_court_homography'])
    trackers=dict(resolution_only_legacy_gates=pair_initialization_class(adaptive.AdaptiveNearTracker,H)(),
                  court_mapped_camera_gates=calibrated_temporal_class(adaptive.AdaptiveNearTracker,H,oldH)())
    cap=cv2.VideoCapture(str(VIDEO));cap.set(1,p['first_source_frame']);model=YOLO(p['detector']['model'])
    raw=[];outputs={k:[] for k in trackers};availability=[]
    for f in range(p['first_source_frame'],p['last_source_frame']+1):
        ok,im=cap.read();assert ok and im.shape[:2]==(1080,1920) and round(cap.get(1))==f+1
        canonical=adapter.image(im)
        pred=model.track(canonical,persist=True,classes=[0],conf=.25,imgsz=1280,tracker='bytetrack.yaml',device=0,verbose=False)[0]
        current=[];vectors=[]
        for box in pred.boxes:
            if box.id is None:continue
            x1,y1,x2,y2=box.xyxy[0].cpu().tolist()
            r=dict(source_frame=f,timestamp_sec=float(f/FPS),track_id=int(box.id.item()),x1=x1,y1=y1,x2=x2,y2=y2,
                bottom_center_x=(x1+x2)/2,bottom_center_y=y2,confidence=float(box.conf.item()),candidate_origin='full_frame_only')
            current.append(r);vectors.append(old.a.appearance(im,r));raw.append(adapter.row(r,to_canonical=False))
        for mode,tr in trackers.items():
            assigned=tr.step(f,current,vectors)
            for slot,j,margin,cost,origin in assigned:
                outputs[mode].append(dict(adapter.row(current[j],to_canonical=False),player_slot=old.a.SLOTS[slot],
                    identity_source=origin,assignment_cost=float(cost),alternative_margin=float(margin),
                    position_source='native_image_detected_box_bottom_proxy',independent_ID_verified=0,independent_position_verified=0))
            availability.append(dict(source_frame=f,mode=mode,assigned_slots=len(assigned),four_slot_candidates=int(len(assigned)==4)))
        if (f-p['first_source_frame'])%60==0:print('match2 frame',f,flush=True)
    cap.release();write(OUT/'native_full_frame_detections.csv',raw);write(OUT/'frame_availability_not_accuracy.csv',availability)
    reports={}
    for mode,tr in trackers.items():
        dest=OUT/mode;dest.mkdir()
        if outputs[mode]:write(dest/'selected_native_observations.csv',outputs[mode])
        dump(dest/'events.json',tr.events)
        a=[r for r in availability if r['mode']==mode]
        reports[mode]=dict(initialized_frame=tr.initialized_frame,observations=len(outputs[mode]),
            four_candidate_frames=sum(r['four_slot_candidates'] for r in a),total_frames=len(a),
            independent_accuracy_established=False)
        print(mode,reports[mode],flush=True)
    dump(OUT/'result.json',dict(modes=reports,raw_native_boxes=len(raw),source_clip_unchanged=True,
        method_or_threshold_tuning_after_match2_predictions=False,independent_accuracy_claim=False))
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items()) and sha(VIDEO)==p['source_video_sha256']


if __name__=='__main__':{'prepare':prepare,'controls':controls,'transfer':transfer}[sys.argv[1]]()
