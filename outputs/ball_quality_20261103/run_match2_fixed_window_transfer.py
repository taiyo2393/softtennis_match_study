"""Keep preselected windows, frozen association, and explicit camera resets."""
import sys,os,json
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,SOURCE,read,write,sha,dump
from prepare_match2_external_validation import VIDEO,FPS,load
from calibrated_camera_temporal_tracker import calibrated_temporal_class
from confirmed_continuity_guard import adaptive,old
from match_frame_coordinate_adapter import FrameCoordinates
from fit_match2_court_geometry_centers import fit_line_centers
import fit_match2_court_geometry as geometry_module

OUT=BASE/'match2_fixed_window_transfer'
PILOT=BASE/'match2_camera_transfer_pilot'
MODES=['frozen_tracker_no_camera_reset','frozen_tracker_reset_on_line_failure']


def prepare():
    assert not OUT.exists();pilot=load(PILOT/'protocol.json');pre=load(BASE/'match2_external_validation/preflight_protocol.json')
    files=[Path(__file__),ROOT/'calibrated_camera_temporal_tracker.py',ROOT/'match_frame_coordinate_adapter.py',
        ROOT/'fit_match2_court_geometry.py',ROOT/'fit_match2_court_geometry_centers.py',
        PILOT/'protocol.json',PILOT/'publication_hashes.json',PILOT/'old_context_regression.json',
        BASE/'match2_external_validation/preflight_protocol.json',BASE/'match2_external_validation/publication_hashes.json',
        BASE/'match2_camera_setup/publication_hashes.json',SOURCE/'models/yolo11m.pt',
        SOURCE/'config/match1_court_plane_calibration.json',Path(old.__file__),Path(old.previous.__file__),Path(old.a.__file__),Path(adaptive.__file__)]
    files.extend(n for n in (BASE/'current22_player_inputs').iterdir() if n.is_file())
    cases=[]
    for e in pre['cases']:
        cases.append(dict(e,processing_role='preplay_raw_inspection_only_no_person_inference' if e['nominal_start_sec']==60 else 'full_prespecified_window_including_view_changes',
            phase_evidence='Raw90..240s chronology suggests first scored point around205..210s; this60s window is preparation footage, AI assessment not independent phase GT.' if e['nominal_start_sec']==60 else 'Rally phase unconfirmed; do not infer all frames are in play.'))
    OUT.mkdir();dump(OUT/'protocol.json',dict(cases=cases,modes=MODES,source_video=pilot['source_video'],source_video_sha256=pilot['source_video_sha256'],
        detector=pilot['detector'],image_to_court_homography_canonical=pilot['image_to_court_homography_canonical'],
        association='Exact previous court-mapped temporal class, constants, galleries and image features. No numeric retuning after first match2 outputs. Full-frame only; no far crop/beam/body auxiliary.',
        shared_inputs='One identical native/canonical detection stream and appearance vectors supplied to both arms. ByteTrack persists within each clip and restarts per clip, identically for both arms.',
        camera_flag='Exact prior4 painted-line support tests on each native frame, same coarse endpoints and thresholds. This is main-view geometry compatibility, not certified camera/visibility/in-play GT. No per-frame homography updates.',
        reset_policy='Additional control-flow arm: if any line unsupported, emit nothing and discard slot state and pending initialization. Next supported frame starts a new namespace and ordinary8-source-frame confirmation. No hysteresis/threshold tuning; false line failures may harm availability and are retained.',
        identity_scope='No claims of cross-camera or cross-clip person identity. Reset-arm output IDs include clip and segment prefix; same N1/F1 text in a new segment does not bind the same person. No automatic cross-cut ReID.',
        preplay_policy='Keep the frozen60..75s case with31 unknown sampling rows per mode; do not run preplay player/ball inference or replace the window. It is not counted as a missed player test. All3 cases retain93 fixed times per mode.',
        selection='Exactly all three original15s windows. No replacement, shifting, extension or source frame rate conversion. Two intervals receive person inference; one preparation window remains explicit excluded/unknown.',
        native_geometry='Original1080 native boxes only; no positions on camera-unsupported frames in reset arm; no backfill or inferred foot points.',
        study_limits='Additional cases from the same new match, not two more independent matches. Counts are availability, not identity accuracy. The camera policy is tested after raw-camera setup but fixed before these two clips person outputs.',
        policy_fixed_before_new_window_outputs=True,person_identity_GT_added=False,current22_changed=False,
        protected_input_hashes={str(n):sha(n) for n in files}))
    print('Frozen original3 windows,2 person-inference clips,2 camera-control arms; association unchanged.',flush=True)


def run():
    p=load(OUT/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    assert sha(VIDEO)==p['source_video_sha256']
    geometry_module.fit_line=fit_line_centers
    os.environ['YOLO_CONFIG_DIR']=str(OUT/'ultralytics_config')
    from ultralytics import YOLO
    import torch,ultralytics
    assert torch.cuda.is_available()
    dump(OUT/'runtime.json',dict(torch=torch.__version__,ultralytics=ultralytics.__version__,device=torch.cuda.get_device_name(0)))
    adapter=FrameCoordinates(1920,1080);H=np.array(p['image_to_court_homography_canonical'])
    oldH=np.array(load(SOURCE/'config/match1_court_plane_calibration.json')['image_to_court_homography'])
    factory=calibrated_temporal_class(adaptive.AdaptiveNearTracker,H,oldH)
    reports=[];all_availability=[]
    for e in p['cases']:
        tag=e['clip_id'];dest=OUT/tag;dest.mkdir()
        if e['processing_role'].startswith('preplay'):
            report=dict(clip_id=tag,status='preserved_preplay_excluded_AI_phase_assessment',source_frames=e['source_frames'],person_inference_calls=0,not_a_tracking_failure=True)
            dump(dest/'result.json',report);reports.append(report)
            for f in e['availability_sample_source_frames']:
                for mode in MODES:all_availability.append(dict(clip_id=tag,source_frame=f,mode=mode,status='preplay_not_processed',line_geometry_supported='',segment_id='',assigned_slots='',four_slot_candidates='',requested_fixed_time=1))
            print(report,flush=True);continue
        # New detector tracker for each source window. Shared between both arms.
        detector=YOLO(p['detector']['model']);cap=cv2.VideoCapture(str(VIDEO));cap.set(1,e['first_source_frame'])
        continuous=factory();reset=None;segment=0;active_first=None;segments=[];raw=[];rows={m:[] for m in MODES};availability=[];line_rows=[];feature_records=[];feature_vectors=[]
        wanted=set(e['availability_sample_source_frames'])
        for f in range(e['first_source_frame'],e['last_source_frame']+1):
            ok,im=cap.read();assert ok and im.shape[:2]==(1080,1920) and round(cap.get(1))==f+1
            fit,diag=geometry_module.geometry(im);supported=fit is not None
            line_rows.append(dict(source_frame=f,line_geometry_supported=int(supported),
                failed_lines=';'.join(k for k,v in diag.items() if not v['accepted']),
                minimum_bin_coverage=min(v['bin_coverage'] for v in diag.values())))
            pred=detector.track(adapter.image(im),persist=True,classes=[0],conf=.25,imgsz=1280,tracker='bytetrack.yaml',device=0,verbose=False)[0]
            current=[];vectors=[]
            for box in pred.boxes:
                if box.id is None:continue
                x1,y1,x2,y2=box.xyxy[0].cpu().tolist()
                r=dict(source_frame=f,timestamp_sec=float(f/FPS),track_id=int(box.id.item()),x1=x1,y1=y1,x2=x2,y2=y2,
                    bottom_center_x=(x1+x2)/2,bottom_center_y=y2,confidence=float(box.conf.item()),candidate_origin='full_frame_only')
                v=old.a.appearance(im,r);current.append(r);vectors.append(v);raw.append(adapter.row(r,False))
                feature_records.append(dict(source_frame=f,track_id=r['track_id'],array_row=len(feature_vectors),appearance_available=int(v is not None)))
                feature_vectors.append(v if v is not None else np.full((2,192),np.nan,np.float32))
            assigned0=continuous.step(f,current,vectors)
            if supported:
                if reset is None:segment+=1;reset=factory();active_first=f
                assigned1=reset.step(f,current,vectors)
                namespace=f'{tag}:S{segment:03}'
            else:
                assigned1=[];namespace=''
                if reset is not None:
                    segments.append(dict(segment_id=f'{tag}:S{segment:03}',first_frame=active_first,last_frame=f-1,events=reset.events))
                reset=None;active_first=None
            for mode,assigned,prefix in [(MODES[0],assigned0,f'{tag}:continuous'),(MODES[1],assigned1,namespace)]:
                for slot,j,margin,cost,origin in assigned:
                    rows[mode].append(dict(adapter.row(current[j],False),player_slot=old.a.SLOTS[slot],
                        identity_namespace=prefix,scoped_candidate_id=prefix+':'+old.a.SLOTS[slot],
                        line_geometry_supported=int(supported),identity_source=origin,
                        association_cost=float(cost),alternative_margin=float(margin),
                        cross_camera_person_identity_verified=0,position_source='native_bbox_bottom_image_proxy'))
                availability.append(dict(clip_id=tag,source_frame=f,mode=mode,status='processed',
                    line_geometry_supported=int(supported),segment_id=prefix,assigned_slots=len(assigned),four_slot_candidates=int(len(assigned)==4),requested_fixed_time=int(f in wanted)))
            if (f-e['first_source_frame'])%180==0:print(tag,'frame',f,'/',e['last_source_frame'],'line',supported,flush=True)
        cap.release();del detector
        if reset is not None:segments.append(dict(segment_id=f'{tag}:S{segment:03}',first_frame=active_first,last_frame=e['last_source_frame'],events=reset.events))
        write(dest/'native_full_frame_detections.csv',raw);write(dest/'appearance_metadata.csv',feature_records)
        np.savez_compressed(dest/'appearance_features.npz',features=np.asarray(feature_vectors,np.float32))
        write(dest/'line_geometry_support.csv',line_rows);write(dest/'frame_availability_not_accuracy.csv',availability)
        dump(dest/'reset_segments.json',segments);dump(dest/'continuous_events.json',continuous.events)
        modes={}
        for mode in MODES:
            if rows[mode]:write(dest/f'{mode}_native_observations.csv',rows[mode])
            a=[r for r in availability if r['mode']==mode];eligible=[r for r in a if r['line_geometry_supported']]
            modes[mode]=dict(native_observations=len(rows[mode]),four_candidate_frames=sum(r['four_slot_candidates'] for r in a),total_source_frames=len(a),
                line_supported_frames=len(eligible),four_candidates_on_line_supported_frames=sum(r['four_slot_candidates'] for r in eligible),
                emitted_rows_on_line_unsupported_frames=sum(not r['line_geometry_supported'] for r in rows[mode]),
                initializations=len([v for s in segments for v in s['events'] if v['event']=='initialized']) if mode==MODES[1] else len([v for v in continuous.events if v['event']=='initialized']))
        report=dict(clip_id=tag,status='completed_frozen_methods',source_frames=e['source_frames'],line_supported_frames=sum(r['line_geometry_supported'] for r in line_rows),
            reset_line_segments=len(segments),modes=modes,independent_accuracy_established=False)
        dump(dest/'result.json',report);reports.append(report);all_availability.extend(availability);print(report,flush=True)
    write(OUT/'all_frame_availability_not_accuracy.csv',all_availability)
    fixed=[r for r in all_availability if r['requested_fixed_time']];assert len(fixed)==3*31*2
    write(OUT/'all93_fixed_times_both_modes.csv',fixed)
    dump(OUT/'result.json',dict(cases=reports,original_windows_retained=3,person_inference_windows=2,
        original_fixed_times_per_mode=93,tracked_source_frames=sum(r['source_frames'] for r in reports if r['status']=='completed_frozen_methods'),
        association_thresholds_or_calibration_changed=False,cross_cut_identity_reconnected=False,
        independent_accuracy_established=False,previous_work_changed=False))
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items()) and sha(VIDEO)==p['source_video_sha256']


if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
