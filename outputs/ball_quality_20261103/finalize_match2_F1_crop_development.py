"""Review actual detections and verify development-only augmentation provenance."""
from pathlib import Path
from collections import defaultdict, Counter
import json, shutil, subprocess
import cv2
import numpy as np
from run_match2_far_crop_development import DEST, MODES
from audit_match2_F1_missing import OUT, PARENT, TARGETS
from run_fixed_cohort_player_case_v3 import ROOT, read, write, sha, dump
from prepare_match2_external_validation import VIDEO, FPS

CASE = 'match2_fixed_clip_003'
MODE = MODES[1]
COLORS = {'far_rear': (225,80,200), 'far_front': (40,220,245),
          'near_front': (100,220,50), 'near_rear': (230,150,50)}
LABELS = dict(far_rear='F1',far_front='F2',near_front='N1',near_rear='N2')

def key(r):
    return (int(r['source_frame']),r['player_slot'],str(r['track_id']),
            *[float(r[k]) for k in ['x1','y1','x2','y2']])

def frame_at(cap,f):
    cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1
    return im

def boxed(im, rows, offset=(0,0)):
    im=im.copy()
    for r in rows:
        xy=[round(float(r[k])) for k in ['x1','y1','x2','y2']]
        x1,y1,x2,y2=xy; ox,oy=offset
        c=COLORS[r['player_slot']]
        cv2.rectangle(im,(x1-ox,y1-oy),(x2-ox,y2-oy),c,1)
        cv2.putText(im,LABELS[r['player_slot']],(x1-ox,max(12,y1-oy-4)),cv2.FONT_HERSHEY_SIMPLEX,.4,c,1,cv2.LINE_AA)
    return im

def render():
    rows=read(DEST/CASE/MODE/'selected_native_observations.csv');by=defaultdict(list)
    for r in rows:by[int(r['source_frame'])].append(r)
    cap=cv2.VideoCapture(str(VIDEO));panels=[]
    for f in TARGETS:
        im=frame_at(cap,f);cv2.imwrite(str(OUT/f'raw_F{f}.png'),im)
        x1,y1,x2,y2=(650,405,1050,635) if f<54400 else (620,405,1020,635)
        raw=im[y1:y2,x1:x2];annotated=boxed(raw,by[f],(x1,y1))
        raw=cv2.resize(raw,(600,345));annotated=cv2.resize(annotated,(600,345))
        header=np.zeros((38,1200,3),np.uint8)
        cv2.putText(header,f'F{f}: raw image | selected boxes; pink = added F1 candidate',(12,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
        panels.append(np.vstack([header,np.hstack([raw,annotated])]))
    cap.release();cv2.imwrite(str(OUT/'four_missing_frames_review.jpg'),np.vstack(panels))
    # Both slow-motion views retain native aspect ratio, every source frame, no interpolation.
    manifest=[]
    for name,start,end,roi in [('overlap_far_pair',54312,54345,(630,390,1130,690)),
                               ('overlap_near_player',54524,54556,(590,390,1090,690))]:
        cap=cv2.VideoCapture(str(VIDEO));cap.set(1,start)
        path=OUT/(name+'.mp4')
        proc=subprocess.Popen(['C:/ffmpeg/bin/ffmpeg.exe','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1200x430','-r',str(FPS/8),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        for f in range(start,end+1):
            ok,im=cap.read();assert ok
            x1,y1,x2,y2=roi;raw=im[y1:y2,x1:x2];ann=boxed(raw,by[f],(x1,y1))
            header=np.zeros((70,1200,3),np.uint8)
            cv2.putText(header,f'Source F{f}  |  1/8 speed  |  raw left, candidates right',(12,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(header,'Pink=F1; yellow=F2; green=N1. Box bottom is NOT verified foot position.',(12,55),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
            out=np.vstack([header,np.hstack([cv2.resize(raw,(600,360)),cv2.resize(ann,(600,360))])])
            proc.stdin.write(out.tobytes());manifest.append(dict(video=path.name,display_frame=f-start,source_frame=f,source_timestamp_sec=f/FPS))
        proc.stdin.close();err=proc.stderr.read();assert proc.wait()==0,err
        cap.release()
    write(OUT/'video_source_frame_map.csv',manifest)

def verify():
    p=json.loads((DEST/'protocol.json').read_text(encoding='utf-8'))
    q=json.loads((OUT/'diagnostic_protocol.json').read_text(encoding='utf-8'))
    for protocol in [p,q]:
        assert all(sha(Path(n))==h for n,h in protocol['protected_input_hashes'].items())
    old_hashes=json.loads((PARENT/'publication_hashes.json').read_text(encoding='utf-8'))
    assert all(sha(PARENT/n)==h for n,h in old_hashes.items())
    counter_audit=[];added=[];preserved=0;origin_checked=0
    for case in ['match2_fixed_clip_002',CASE]:
        b=read(DEST/case/'baseline/selected_native_observations.csv');a=read(DEST/case/MODE/'selected_native_observations.csv')
        bs={key(r) for r in b};aa={key(r) for r in a};assert bs<=aa;preserved+=len(bs)
        actual=read(DEST/case/MODE/'added_native_crop_candidates.csv')
        candidates={(int(r['source_frame']),str(r['track_id']),*[float(r[k]) for k in ['x1','y1','x2','y2']]) for r in actual}
        for r in a:
            if key(r) not in bs:
                assert (int(r['source_frame']),str(r['track_id']),*[float(r[k]) for k in ['x1','y1','x2','y2']]) in candidates
                added.append(r);origin_checked+=1
        initial=[]
        for mode in MODES:
            rr=read(DEST/case/mode/'all_crop_proposals_with_rejections.csv');first=min(int(r['source_frame']) for r in rr)
            ids=sorted(int(r['track_id'])-1000000 for r in rr if int(r['source_frame'])==first)
            assert ids==list(range(1,max(ids)+1))
            counts=Counter((r['source_frame'],r['track_id']) for r in rr);assert max(counts.values())==1
            initial.append(max(ids))
        assert initial[1]>=initial[0]
        counter_audit.append(dict(clip_id=case,legacy_initial_allocated_max=initial[0],mapped_initial_allocated_max=initial[1],duplicate_ID_within_frame=False))
    assert len(added)==4 and sorted(int(r['source_frame']) for r in added)==TARGETS
    write(OUT/'four_added_actual_native_detections.csv',added)
    dump(OUT/'verification.json',dict(parent_publication_files_unchanged=len(old_hashes),all_protected_inputs_unchanged=True,
        baseline_paths_and_costs_exactly_replayed_in_two_cases=True,baseline_observations_preserved=preserved,
        added_selected_observations=origin_checked,all_added_boxes_exact_actual_same_frame_crop_detections=True,
        interpolation=False,manual_ID_assignment=False,independent_accuracy_established=False,
        counter_audit=counter_audit,
        shared_ID_counter_note='Installed ByteTrack uses a class-global counter. Second model resets it once on initialization. Its initial allocated maximum is >= first model maximum in both cases, so subsequent global increments cannot reuse any prior ID inside either model. persist=True returns existing trackers and skips resets thereafter. IDs are local to each clip/mode, never identities across runs. Shared counter remains a maintenance risk: future concurrent-model code should isolate counters or use sequential passes.'))

def publish():
    assert (OUT/'verification.json').exists()
    added=read(OUT/'four_added_actual_native_detections.csv')
    review=[]
    for r in added:
        f=int(r['source_frame'])
        review.append(dict(source_frame=f,player_slot='far_rear',reviewer='AI assistant; not independent GT',
            identity_visual_interpretation='Box surrounds the same magenta-shirt far player in these neighboring frames',
            overlapping_player='F2' if f<54400 else 'N1',
            bbox_bottom_position_risk='Not independently validated' if f<54400 else 'Upper-body detection; bottom ends near waist, not feet',
            independent_identity_label_created=0,independent_foot_label_created=0,
            validated_court_position_available=0,position_imputation_performed=0))
    write(OUT/'AI_position_risk_review.csv',review)
    dump(OUT/'AI_visual_review_correction.json',dict(
        supersedes_statement_in=str(PARENT/'AI_visual_review.json'),
        correction='Previous statement that all4 gaps occurred during overlap with near N1 was too broad. Native zooms show F54328/54330 overlap with far F2; F54538/54544 overlap with near N1. Prior immutable report retained.',
        evidence='four_missing_frames_review.jpg and both native-aspect slow-motion clips',
        person_reacquisition_interpretation='All4 new boxes visually surround far F1; this is AI review, not independent human truth.',
        coordinate_limitation='At54538 and54544 the F1 box covers upper body only; a detected person does not imply a measured foot location. No court-plane coordinate is validated by this audit.',
        old_tracker_behavior='At54328/54330 only a competing F2 candidate could be given finite F1 cost; one-to-one matching keeps it assigned to F2. At54538/54544 no valid F1 candidate survives. Relaxing assignment gates would not supply missing image evidence.',
        full_predict_diagnostic='F54328/54330/54544 anonymous full-frame prediction has a box overlapping the crop F1 box (IoU0.418/0.708/0.611), absent from saved ByteTrack candidates. F54538 has no comparable box (maxIoU0.048). These stage comparisons suggest candidate filtering in3 cases and a full-frame detection miss in1; replayed anonymous predictions are not an exact internal trace or independent person labels.',
        independent_accuracy_established=False))
    for n in ['audit_match2_F1_missing.py','run_match2_far_crop_development.py','finalize_match2_F1_crop_development.py']:
        shutil.copy2(ROOT/n,OUT/('executed_'+n))
    import importlib.util
    package=Path(importlib.util.find_spec('ultralytics').origin).parent
    dump(OUT/'tracker_implementation_hashes.json',{str(package/n):sha(package/n) for n in ['trackers/track.py','trackers/byte_tracker.py','trackers/basetrack.py']})
    dump(OUT/'publication_hashes.json',{str(n.relative_to(OUT)):sha(n) for n in sorted(OUT.rglob('*'))
        if n.is_file() and n.name!='publication_hashes.json' and '__pycache__' not in n.parts})
    print('Published development audit; preserved3696 observations, added4 candidate detections; no independent accuracy or foot-position claim.')

if __name__=='__main__':
    import sys
    {'render':render,'verify':verify,'publish':publish}[sys.argv[1]]()
