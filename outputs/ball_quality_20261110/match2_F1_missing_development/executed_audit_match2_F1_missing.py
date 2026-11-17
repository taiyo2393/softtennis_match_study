"""Cached exact replay and candidate-level reasons, never new person truth."""
import sys,json
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,SOURCE,read,write,sha,dump
from run_match2_fixed_window_transfer import MODES
from calibrated_camera_temporal_tracker import calibrated_temporal_class,CameraGateMap
from confirmed_continuity_guard import adaptive,old
from match_frame_coordinate_adapter import FrameCoordinates
from prepare_match2_external_validation import VIDEO,load

OUT=BASE/'match2_F1_missing_development'
PARENT=BASE/'match2_fixed_window_transfer'
TARGETS=[54328,54330,54538,54544]


def cached_case(e):
    dest=PARENT/e['clip_id'];raw=read(dest/'native_full_frame_detections.csv');by=defaultdict(list)
    adapter=FrameCoordinates(1920,1080)
    for r in raw:by[int(r['source_frame'])].append(adapter.row(r,True))
    array=np.load(dest/'appearance_features.npz')['features'];vectors={}
    for r in read(dest/'appearance_metadata.csv'):
        value=array[int(r['array_row'])];vectors[int(r['source_frame']),str(r['track_id'])]=None if np.isnan(value).all() else value
    line={int(r['source_frame']):r['line_geometry_supported']=='1' for r in read(dest/'line_geometry_support.csv')}
    return by,vectors,line


def prepare():
    assert not OUT.exists();p=load(PARENT/'protocol.json');cases=p['cases'][1:]
    files=[Path(__file__),PARENT/'protocol.json',PARENT/'publication_hashes.json',ROOT/'calibrated_camera_temporal_tracker.py',
        Path(old.__file__),Path(old.a.__file__),Path(old.previous.__file__),Path(adaptive.__file__),SOURCE/'config/match1_court_plane_calibration.json']
    for e in cases:
        dest=PARENT/e['clip_id']
        files.extend(dest/n for n in ['native_full_frame_detections.csv','appearance_features.npz','appearance_metadata.csv',
            'line_geometry_support.csv',MODES[1]+'_native_observations.csv','verification.json'])
    files.extend(n for n in (BASE/'current22_player_inputs').iterdir() if n.is_file())
    OUT.mkdir();dump(OUT/'diagnostic_protocol.json',dict(cases=cases,source_video=p['source_video'],source_video_sha256=p['source_video_sha256'],
        image_to_court_homography_canonical=p['image_to_court_homography_canonical'],
        target_frames=TARGETS,diagnostic_neighborhood_frames=sorted({f+d for f in TARGETS for d in [-1,0,1]}),
        purpose='Exact cached replay of unchanged reset-arm tracker in both windows; log F1 candidate costs and gates at4 known missing times and immediate neighbors.',
        not_independent_evaluation=True,person_or_position_GT_created=False,outputs_or_thresholds_changed=False,
        protected_input_hashes={str(n):sha(n) for n in files}))


def replay():
    p=load(OUT/'diagnostic_protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    H=np.array(p['image_to_court_homography_canonical']);oldH=np.array(load(SOURCE/'config/match1_court_plane_calibration.json')['image_to_court_homography'])
    mapper=CameraGateMap(H,oldH);factory=calibrated_temporal_class(adaptive.AdaptiveNearTracker,H,oldH)
    audit=[];states=[];reports=[]
    wanted=set(p['diagnostic_neighborhood_frames'])
    for e in p['cases']:
        by,vectors,line=cached_case(e);tr=None;actual=[]
        for f in range(e['first_source_frame'],e['last_source_frame']+1):
            if not line[f]:tr=None;continue
            if tr is None:tr=factory()
            rows=by[f];features=[vectors[f,str(r['track_id'])] for r in rows]
            for s,st in tr.states.items():st['_audit_slot']=s
            if f in wanted and 'far_rear' in tr.states:
                st=tr.states['far_rear'];prior=st['row']
                states.append(dict(source_frame=f,last_observed_frame=st['frame'],last_F1_raw_id=prior['track_id'],
                    previous_box_native=[float(prior[k])/2 for k in ['x1','y1','x2','y2']],
                    current_same_raw_ID_present=int(any(str(r['track_id'])==str(prior['track_id']) for r in rows))))
            original=old.previous.continuity_cost
            def logged(state,row,feature,frame,use_appearance=True):
                value=original(state,row,feature,frame,use_appearance)
                if frame in wanted and state.get('_audit_slot')=='far_rear':
                    h=old.a.height(row);oldh=old.a.height(state['row']);gap=frame-state['frame'];ratio=h/oldh
                    predicted=old.a.foot(state['row'])+state['velocity']*min(gap,12)
                    residual=float(np.linalg.norm(old.a.foot(row)-predicted))/max(h,oldh)
                    app=old.a.app_distance(state['template'],feature)
                    negative=min((old.a.app_distance(n,feature) for n in state['negatives']),default=None)
                    same=int(row['track_id'])==int(state['row']['track_id']);bypass=same and gap<=2 and residual<.4
                    reasons=[]
                    if h<.5*state['reference_height']:reasons.append('reference_height_guard')
                    if not .45<=ratio<=2.2:reasons.append('successive_height_ratio')
                    if gap>90:reasons.append('gap_limit')
                    if not mapper.allowed_continuation('far',row):reasons.append('camera_side_gate')
                    if residual>min(2.,.55+.025*gap):reasons.append('motion_gate')
                    if app>.86:reasons.append('appearance_gate')
                    if not bypass and negative is not None and negative+.04<app:reasons.append('negative_appearance')
                    audit.append(dict(source_frame=frame,candidate_raw_id=row['track_id'],previous_F1_raw_id=state['row']['track_id'],
                        candidate_box_native=[float(row[k])/2 for k in ['x1','y1','x2','y2']],candidate_native_height=h/2,
                        reference_native_height=state['reference_height']/2,same_raw_ID=int(same),gap_source_frames=gap,
                        motion_residual_height=residual,appearance_distance=app,negative_distance=negative,
                        continuity_cost=float(value),failed_gate_reasons=';'.join(reasons),
                        candidate_valid_cost=int(value<1.05),independent_person_match_verified=0))
                return value
            old.previous.continuity_cost=logged
            try:assigned=tr.step(f,rows,features)
            finally:old.previous.continuity_cost=original
            selected={str(rows[j]['track_id']):old.a.SLOTS[s] for s,j,_,_,_ in assigned}
            if f in wanted:
                for r in audit:
                    if r['source_frame']==f:r['assigned_slot']=selected.get(str(r['candidate_raw_id']),'')
            for slot,j,margin,cost,origin in assigned:
                r=rows[j];actual.append((f,old.a.SLOTS[slot],str(r['track_id']),tuple(float(r[k])/2 for k in ['x1','y1','x2','y2']),origin,float(cost),float(margin)))
        expected=read(PARENT/e['clip_id']/(MODES[1]+'_native_observations.csv'))
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):
            assert a[:3]==(int(b['source_frame']),b['player_slot'],str(b['track_id']))
            assert np.allclose(a[3],[float(b[k]) for k in ['x1','y1','x2','y2']],atol=1e-9,rtol=0)
            assert a[4]==b['identity_source'] and np.allclose(a[5:],[float(b['association_cost']),float(b['alternative_margin'])],atol=1e-9,rtol=0)
        report=dict(clip_id=e['clip_id'],exact_native_path_replay=True,matching_observations=len(actual));reports.append(report);print(report,flush=True)
    write(OUT/'F1_candidate_gate_audit.csv',audit);dump(OUT/'F1_prior_state_audit.json',dict(states=states))
    missing=[r for r in states if r['source_frame'] in TARGETS]
    assert len(missing)==4 and all(r['current_same_raw_ID_present']==0 for r in missing)
    dump(OUT/'diagnostic_result.json',dict(cases=reports,missing_F1_frames=TARGETS,
        missing_frames_without_previous_F1_raw_ID=4,
        conclusion='The previously accepted F1 raw-ID box is absent from the saved ByteTrack candidate stream at all4 times. This does not yet establish a YOLO detector miss: framewise detections or another rawID may remain.',
        model_or_thresholds_modified=False,independent_accuracy_claim=False))
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())


if __name__=='__main__':{'prepare':prepare,'replay':replay}[sys.argv[1]]()
