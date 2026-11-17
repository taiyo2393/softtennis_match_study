"""Nine training-only candidates, awaiting object identity rather than target match.

Distance from a clicked target is only a query cue. Another real ball remains
a ball for this question; no negative labels are automatically manufactured.
"""
import json,math,subprocess
from pathlib import Path
from collections import Counter
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2,np

OUT=ROOT/'ball_appearance_training_review'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def crop96(image,x,y):
    # Preserve native source pixels in lossless PNG; outside-image area is padded.
    padded=cv2.copyMakeBorder(image,48,48,48,48,cv2.BORDER_REFLECT_101)
    x,y=round(x),round(y);result=padded[y:y+96,x:x+96];assert result.shape[:2]==(96,96)
    return result

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'manifest.json').exists()
    train_path=ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv';train=read(train_path)
    cache_path=ROOT/'ball_temporal_path_comparison/train_component_cache.json';cache=load(cache_path)
    groups={r['candidate_id'] for r in train};assert groups=={f'match1_motion_pilot_{i:03d}' for i in [1,3,4,5,6,7,9,10]}
    protected={str(p):sha(p) for p in [train_path,cache_path,ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',
        ROOT/'ball_inplay_challenge/manual_center_labels.csv',ROOT/'ball_040_object_swap_review/round2/user_reviewed/candidate_review.csv',
        ROOT/'balltrack_manual_finetune_pilot/validation_labels.csv',ROOT/'balltrack_manual_finetune_pilot/audit_labels.csv']}
    protocol=dict(status='fixed_before_new_object_labels',role='training_only_never_evaluation',training_clips=sorted(groups),
        positive_source='102visible user-clicked centers in existing training120+newA/C/D3. Not new independent ball-enclosure checks.',
        review_pool='ALLthreshold0.3 components >=64source pixels from visible manual target, on training clips only',
        label_caution='Far from target is not generic non-ball. Multiple real balls possible, especially before play. Every actual ball counts positive regardless of target ID.',
        object_labels='ball / not_ball / uncertain; none inferred from distance or old empty reviewer fields',
        patches='Raw96x96 native source pixels saved PNG, reflect padding if needed; marker/header never enter training patches',
        media='Target image with neutral candidate ring; context +/-24source frames at0.5x, target held. No manual target marker or presumed class displayed.',
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    (OUT/'positive_patch96').mkdir();(OUT/'candidate_patch96').mkdir()
    positives=[];queries=[]
    for r in train:
        if r['truth_visibility']!='visible':continue
        cid=r['candidate_id'];t=int(r['local_frame']);base=SOURCE/'data/balltrack/racketvision/match1'/cid
        image_path=base/'frames'/f'{t:06d}.jpg';assert str(image_path)==json.loads(r['input_frames_json'])[-1]
        assert sha(image_path)==json.loads(r['input_frame_sha256_json'])[-1]
        image=cv2.imread(str(image_path));assert image.shape[:2]==(1080,1920)
        tx,ty=float(r['truth_x_px']),float(r['truth_y_px']);positive_path=OUT/'positive_patch96'/f'{r["sample_id"]}.png'
        assert cv2.imwrite(str(positive_path),crop96(image,tx,ty))
        positives.append(dict(sample_id=r['sample_id'],candidate_id=cid,local_frame=t,source_frame=r['source_frame'],
            label='ball_at_manual_center',label_source='existing_user_clicked_center',source_image=str(image_path),source_image_sha256=sha(image_path),
            x_px=tx,y_px=ty,raw_patch96=str(positive_path.resolve()),patch_sha256=sha(positive_path),split_group=f'legacy_match1:{cid}'))
        for c in cache[cid][str(t)]:
            distance=math.hypot(c['x_px']-tx,c['y_px']-ty)
            if distance<64:continue
            sample_id=f'{cid}_frame_{t:06d}_component{c["component_index"]}'
            patch=OUT/'candidate_patch96'/f'{sample_id}.png';assert cv2.imwrite(str(patch),crop96(image,c['x_px'],c['y_px']))
            queries.append(dict(sample_id=sample_id,candidate_id=cid,local_frame=t,source_frame=int(r['source_frame']),
                component_index=c['component_index'],candidate_x_px=c['x_px'],candidate_y_px=c['y_px'],candidate_peak=c['component_peak'],
                sampling_target_distance_px=distance,manual_target_x_px=tx,manual_target_y_px=ty,
                source_image=str(image_path),source_image_sha256=sha(image_path),raw_patch96=str(patch.resolve()),patch_sha256=sha(patch),
                split_group=f'legacy_match1:{cid}',object_label='unreviewed',reviewer_id='',reviewed_at_utc='',notes=''))
    queries.sort(key=lambda r:(r['candidate_id'],r['local_frame'],r['component_index']))
    assert len(positives)==102 and len(queries)==9
    assert {q['candidate_id'] for q in queries}<=groups
    cases=[];labels=[]
    for i,q in enumerate(queries):
        case_id=chr(65+i);t=q['local_frame'];cid=q['candidate_id'];frame_dir=Path(q['source_image']).parent
        meta=load(frame_dir.parent/'runs/threshold030_median120/run_metadata.json');video=OUT/f'context_{case_id}_{cid[-3:]}_{t}_h264.mp4'
        lo=max(0,t-24);hi=min(meta['frame_count']-1,t+24);fps=meta['fps']
        cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x800','-r',str(fps),'-i','pipe:0',
            '-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
        count=0
        with (OUT/f'context_{case_id}_render.log').open('wb') as log:
            proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
            for j in range(lo,hi+1):
                im=cv2.imread(str(frame_dir/f'{j:06d}.jpg'));assert im.shape[:2]==(1080,1920)
                frame=np.full((800,1280,3),18,np.uint8);frame[80:]=cv2.resize(im,(1280,720),interpolation=cv2.INTER_AREA)
                cv2.putText(frame,f'{case_id} | pilot{cid[-3:]} local{j} | TARGET {t} | 0.5x',(12,29),0,.74,(240,240,240),1,cv2.LINE_AA)
                cv2.putText(frame,'TARGET - judge the object inside the blue ring' if j==t else 'Motion context only; ring appears on the exact target frame',(12,63),0,.64,(100,230,255) if j==t else (220,220,220),1,cv2.LINE_AA)
                if j==t:
                    p=(round(q['candidate_x_px']*2/3),80+round(q['candidate_y_px']*2/3));cv2.circle(frame,p,10,(255,205,25),2,cv2.LINE_AA)
                    cv2.rectangle(frame,(1,80),(1278,798),(60,230,255),3)
                for _ in range(32 if j==t else 2):proc.stdin.write(frame.tobytes());count+=1
            proc.stdin.close();assert proc.wait()==0
        cap=cv2.VideoCapture(str(video));n=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape[:2]==(800,1280);n+=1
        cap.release();assert n==count
        case=dict(q,case_id=case_id,context_video=str(video.resolve()),context_video_sha256=sha(video),decoded_video_frames=n,duration_sec=n/fps)
        cases.append(case)
        labels.append({k:case[k] for k in ['case_id','sample_id','candidate_id','local_frame','source_frame','component_index','candidate_x_px','candidate_y_px',
            'object_label','reviewer_id','reviewed_at_utc','notes']})
    write(OUT/'existing_positive_centers.csv',positives);write(OUT/'candidate_review_labels.csv',labels);dump(OUT/'cases.json',cases)
    assert all(sha(Path(p))==h for p,h in protected.items())
    manifest=dict(status='nine_training_object_queries_ready_labels_pending',cases=9,existing_manual_center_positive_patches=102,
        training_only=True,unreviewed_candidates=9,automatic_negative_labels=0,eval_six_candidates_not_included=True,
        cases_sha256=sha(OUT/'cases.json'),initial_labels_sha256=sha(OUT/'candidate_review_labels.csv'),
        positive_table_sha256=sha(OUT/'existing_positive_centers.csv'),decoded_video_frames=sum(c['decoded_video_frames'] for c in cases),
        protected_input_hashes=protected)
    dump(OUT/'manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k!='protected_input_hashes'},indent=2))

if __name__=='__main__':main()
