"""Native observation provenance and full-window comparison media, not GT."""
from collections import defaultdict, Counter
from pathlib import Path
import json, shutil, subprocess, sys
import cv2
import numpy as np
from run_match2_unseen_crop_transfer import OUT, PRIOR, MODES
from run_fixed_cohort_player_case_v3 import ROOT, read, write, dump, sha
from prepare_match2_external_validation import VIDEO, FPS, load

COLORS={'far_rear':(225,80,200),'far_front':(40,220,245),'near_front':(100,220,50),'near_rear':(230,150,50)}
LABELS=dict(far_rear='F1',far_front='F2',near_front='N1',near_rear='N2')

def grouped(rows):
    by=defaultdict(list)
    for r in rows:by[int(r['source_frame'])].append(r)
    return by

def annotate(im,rows):
    im=im.copy()
    for r in rows:
        x1,y1,x2,y2=[round(float(r[k])) for k in ['x1','y1','x2','y2']]
        c=COLORS[r['player_slot']];cv2.rectangle(im,(x1,y1),(x2,y2),c,3)
        cv2.putText(im,LABELS[r['player_slot']],(x1,max(20,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.8,c,2,cv2.LINE_AA)
    return im

def render():
    p=load(OUT/'protocol.json');maps=[]
    for e in p['cases']:
        folder=OUT/e['clip_id'];by={m:grouped(read(folder/m/'selected_native_observations.csv')) if (folder/m/'selected_native_observations.csv').exists() else defaultdict(list) for m in MODES}
        lines={int(r['source_frame']):r['line_supported']=='1' for r in read(folder/'line_support.csv')}
        cap=cv2.VideoCapture(str(VIDEO));cap.set(1,e['first_source_frame']);screens=[]
        proc=subprocess.Popen(['C:/ffmpeg/bin/ffmpeg.exe','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1280x430','-r',str(float(FPS)/2),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'comparison.mp4')],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        n=0
        for f in range(e['first_source_frame'],e['last_source_frame']+1):
            ok,im=cap.read();assert ok and round(cap.get(1))==f+1
            if (f-e['first_source_frame'])%2 and f not in e['fixed_review_source_frames']:continue
            header=np.zeros((70,1280,3),np.uint8)
            cv2.putText(header,f'{e["clip_id"]}  source F{f}   main-view support={int(lines[f])}',(12,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(header,'Full-frame only',(12,55),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(header,'+ frozen far crop (candidate boxes; no foot measurement)',(650,55),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
            panel=np.hstack([cv2.resize(annotate(im,by[m][f]),(640,360)) for m in MODES]);panel=np.vstack([header,panel])
            if f in e['fixed_review_source_frames']:
                cv2.imwrite(str(folder/f'comparison_F{f}.jpg'),panel);screens.append(panel)
            if not (f-e['first_source_frame'])%2:
                proc.stdin.write(panel.tobytes());maps.append(dict(clip_id=e['clip_id'],display_frame=n,source_frame=f,source_timestamp_sec=f/float(FPS)));n+=1
        proc.stdin.close();err=proc.stderr.read();assert proc.wait()==0,err;cap.release()
        assert len(screens)==3;cv2.imwrite(str(folder/'three_fixed_times.jpg'),np.vstack(screens))
    write(OUT/'video_source_frame_map.csv',maps)

def verify():
    p=load(OUT/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    assert sha(VIDEO)==p['source_video_sha256']
    priorhash=load(PRIOR/'publication_hashes.json');assert all(sha(PRIOR/n)==h for n,h in priorhash.items())
    evidence=[]
    for e in p['cases']:
        folder=OUT/e['clip_id'];lines={int(r['source_frame']):r['line_supported']=='1' for r in read(folder/'line_support.csv')}
        full=read(folder/'native_full_frame_detections.csv') if (folder/'native_full_frame_detections.csv').exists() else []
        extra=read(folder/'native_extra_crop_candidates.csv') if (folder/'native_extra_crop_candidates.csv').exists() else []
        def native_key(r):return (int(r['source_frame']),str(r['track_id']),*[float(r[k]) for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y']])
        for mode in MODES:
            pool={native_key(r) for r in full+(extra if mode==MODES[1] else [])}
            file=folder/mode/'selected_native_observations.csv';rows=read(file) if file.exists() else []
            for r in rows:
                assert native_key(r) in pool and lines[int(r['source_frame'])]
                assert r['measured_court_x']==r['measured_court_y']=='' and r['position_measurement_validated']=='0'
            c=Counter((r['source_frame'],r['track_id']) for r in rows);assert not c or max(c.values())==1
            slots=Counter((r['source_frame'],r['player_slot']) for r in rows);assert not slots or max(slots.values())==1
            evidence.append(dict(clip_id=e['clip_id'],mode=mode,selected_observations=len(rows),all_same_frame_real_candidate_boxes=True,
                all_geometry_unsupported_outputs_absent=True,one_to_one_assignment=True,measured_position_values=0))
        av=read(folder/'all_frame_availability_not_accuracy.csv');assert len(av)==e['source_frames']*2
        assert sum(int(r['fixed_sample']) for r in av)==62
        cap=cv2.VideoCapture(str(folder/'comparison.mp4'));expected=len(range(e['first_source_frame'],e['last_source_frame']+1,2));assert round(cap.get(7))==expected
        cap.set(1,expected//2);ok,im=cap.read();assert ok and im.shape==(430,1280,3);cap.release()
    dump(OUT/'verification.json',dict(prior_development_publication_unchanged=len(priorhash),protected_inputs_and_source_hash_unchanged=True,
        all_three_fixed_windows_retained=True,all93_fixed_times_in_both_arms=True,provenance_checks=evidence,
        no_manual_ID_mapping=True,no_interpolated_positions=True,no_human_truth_created=True,independent_accuracy_established=False))

def diagnose():
    p=load(OUT/'protocol.json');e=p['cases'][1];d=OUT/e['clip_id']
    changes=read(d/'changed_selected_observations.csv');by=grouped(read(d/MODES[1]/'selected_native_observations.csv'))
    cap=cv2.VideoCapture(str(VIDEO));panels=[]
    for c in changes:
        f=int(c['source_frame']);cap.set(1,f);ok,im=cap.read();assert ok
        raw=im[410:630,880:1200];ann=annotate(im,by[f])[410:630,880:1200]
        h=np.zeros((30,640,3),np.uint8)
        cv2.putText(h,f'F{f} added {LABELS[c["player_slot"]]}: raw | candidates',(8,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
        panels.append(np.vstack([h,np.hstack([raw,ann])]))
    cap.release();assert len(panels)==8
    cv2.imwrite(str(OUT/'all_eight_added_review.jpg'),np.vstack([np.hstack(panels[i:i+2]) for i in range(0,8,2)]))
    missing=[];counts=[]
    for e in p['cases']:
        d=OUT/e['clip_id'];by=grouped(read(d/MODES[1]/'selected_native_observations.csv'));first=min(by);counter=Counter()
        for r in read(d/'all_frame_availability_not_accuracy.csv'):
            f=int(r['source_frame'])
            if r['mode']!=MODES[1] or r['line_supported']!='1' or f<first:continue
            present={v['player_slot'] for v in by[f]}
            for slot in LABELS:
                if slot not in present:
                    missing.append(dict(clip_id=e['clip_id'],source_frame=f,player_slot=slot,status='post_initialization_no_selected_candidate',independent_visibility_label='unknown'));counter[slot]+=1
        counts.append(dict(clip_id=e['clip_id'],post_initialization_missing_by_slot=dict(counter),initialization_first_output=first))
    write(OUT/'all_remaining_post_initialization_missing.csv',missing)
    dump(OUT/'post_initialization_missing_summary.json',counts)
    e=p['cases'][2];d=OUT/e['clip_id'];by=grouped(read(d/MODES[1]/'selected_native_observations.csv'))
    times=[r['source_frame'] for r in missing if r['clip_id']==e['clip_id'] and r['player_slot']=='near_rear'];runs=[]
    for f in times:
        if not runs or f!=runs[-1][-1]+1:runs.append([f])
        else:runs[-1].append(f)
    frames=sorted({f for rr in runs for f in [rr[0]-1,rr[0],rr[len(rr)//2],rr[-1],min(rr[-1]+1,e['last_source_frame'])]})
    cap=cv2.VideoCapture(str(VIDEO));panels=[]
    for f in frames:
        cap.set(1,f);ok,im=cap.read();assert ok
        h=np.zeros((40,1340,3),np.uint8);cv2.putText(h,f'Source F{f}: right edge, raw | selected (blue=N2)',(12,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
        panels.append(np.vstack([h,np.hstack([im[510:1080,1250:1920],annotate(im,by[f])[510:1080,1250:1920]])]))
    cap.release();cv2.imwrite(str(OUT/'N2_remaining_gap_review.jpg'),np.vstack(panels))
    dump(OUT/'remaining_missing_inventory.json',dict(clip_id=e['clip_id'],slot='near_rear',post_initialization_missing_observations=len(times),contiguous_runs=[dict(first=min(v),last=max(v),frames=len(v)) for v in runs],review_frames=frames,independent_visibility_label_created=False))

def publish():
    assert (OUT/'verification.json').exists() and (OUT/'AI_visual_review.json').exists() and (OUT/'README.md').exists()
    for n in ['run_match2_unseen_crop_transfer.py','finalize_match2_unseen_crop_transfer.py']:
        shutil.copy2(ROOT/n,OUT/('executed_'+n))
    dump(OUT/'publication_hashes.json',{str(n.relative_to(OUT)):sha(n) for n in sorted(OUT.rglob('*')) if n.is_file() and n.name!='publication_hashes.json' and '__pycache__' not in n.parts})
    print('Published3 retained windows; measured positions and independent tracking accuracy remain unestablished.')

if __name__=='__main__':{'render':render,'verify':verify,'diagnose':diagnose,'publish':publish}[sys.argv[1]]()
