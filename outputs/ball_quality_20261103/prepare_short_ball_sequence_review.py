"""Freeze two small temporal evaluation windows; no new model or invented GT."""
import json,sys,subprocess
from pathlib import Path
import numpy as np
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2
import compare_temporal_candidate_paths as temporal

OUT=ROOT/'ball_short_sequence_evaluation'
CONFIG=dict(name='displacement_0.1',order='displacement',strength=.1)
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Do not overwrite fixed query windows or human labels.'
    challenge_path=ROOT/'ball_inplay_challenge/manual_center_labels.csv';anchors=sorted(read(challenge_path),key=lambda r:r['candidate_id'])
    anchors=[anchors[0],anchors[-1]] # first/last ordered clip; not selected by new errors
    assert [(r['candidate_id'][-3:],r['local_frame']) for r in anchors]==[('012','303'),('020','171')]
    assert all(r['phase_status']=='play' and r['phase_review_status']=='human_confirmed' and r['truth_visibility']=='visible' for r in anchors)
    old_path=ROOT/'balltrack_manual_finetune_pilot/validation_labels.csv';old=read(old_path)
    known={(r['candidate_id'],int(r['local_frame'])):r for r in old}
    known.update({(r['candidate_id'],int(r['local_frame'])):r for r in anchors})
    inventory_path=ROOT/'play_start_audit/ball_pilot_start_inventory.csv';inventory={r['ball_pilot_id']:r for r in read(inventory_path)}
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth'
    protected={str(p):sha(p) for p in [challenge_path,old_path,inventory_path,weight,
        SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv',ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',
        ROOT/'ball_temporal_path_comparison/training_selection.json',ROOT/'ball_appearance_training_review/candidate_review_labels.csv',
        ROOT/'ball_training_only_appearance_comparison/decision.json']}
    protocol=dict(status='fixed_before_window_inference_or_new_labels',role='known_same_match_validation_only_never_training',
        windows='First and last source clips among four already human-confirmed-play challenge anchors, ordered by clip ID. +/-12 source frames, stride2: 13 observations per window.',
        selection_uses_new_error_locations=False,anchor_caveat='Anchor time only is confirmed play. Neighbours require their own phase/visibility annotation. Earlier challenge selection enriched model disagreements; not population sampling.',
        clock='Legacy edited match1 cache; never map integer frames to redownloaded match1_original',
        methods=['original_threshold0.3_bbox_area','previously_frozen_displacement_0.1'],temporal_config=CONFIG,
        freeze='Predict both methods on ALL26target times before new labels; existing clicked centers only reused at exact same source frame.',
        annotation='Two existing play/center anchors reused; two other old centers need only phase review;22centers new. Click play-ball center or explicitly choose visible-state/phase uncertainty. No predictions shown.',
        no_interpolation=True,no_new_training=True,no_parameter_search=True,
        evaluation='Report each window separately. Visible-frame errors16px, missing/wrong/false outputs, adjacent verified pairs and long gaps in sampled timeline. Uncertain/nonplay excluded with reasons, not absent-ball negatives.',
        continuity='Consecutive stride2 observations, ~29.97Hz. No claim about odd frames or whole-rally accuracy. Do not bridge uncertain/missing frames or count repeated-frame pairs as independent.',
        no_automatic_adoption=True,protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    rows=[];labels=[];cases=[];windows=[];annotation_fields=['truth_visibility','truth_x_px','truth_y_px','phase_status','reviewer_id','annotated_at_utc','notes']
    for wi,anchor in enumerate(anchors):
        wid=chr(65+wi);cid=anchor['candidate_id'];t0=int(anchor['local_frame']);entry=inventory[cid]
        base=SOURCE/'data/balltrack/racketvision/match1'/cid;mp=base/'runs/threshold030_median120/median.npz';meta=load(base/'runs/threshold030_median120/run_metadata.json')
        times=list(range(t0-12,t0+13,2));windows.append(dict(window_id=wid,candidate_id=cid,anchor_local_frame=t0,
            first_local_frame=times[0],last_local_frame=times[-1],sampled_frames=13,stride=2,fps=meta['fps'],span_sec=24/meta['fps']))
        for order,t in enumerate(times,1):
            sample_id=f'{cid}_frame_{t:06d}';paths=[base/'frames'/f'{j:06d}.jpg' for j in range(t-3,t+1)]
            assert all(p.exists() for p in paths);source_frame=int(entry['clip_first_source_frame'])+t
            row=dict(sample_id=sample_id,candidate_id=cid,local_frame=t,source_frame=source_frame,window_id=wid,case_id=f'{wid}{order:02d}',
                truth_visibility='unreviewed',truth_x_px='',truth_y_px='',input_frames_json=json.dumps([str(p) for p in paths]),
                input_frame_sha256_json=json.dumps([sha(p) for p in paths]),median_path=str(mp),median_sha256=sha(mp))
            rows.append(row);reuse=known.get((cid,t));is_anchor=t==t0
            state=dict(truth_visibility='unreviewed',truth_x_px='',truth_y_px='',phase_status='unknown',reviewer_id='',annotated_at_utc='',notes='')
            if reuse:
                assert int(reuse['source_frame'])==source_frame
                state.update({k:reuse.get(k,'') for k in annotation_fields});state['phase_status']='play' if is_anchor else 'unknown'
                state['notes']=reuse.get('notes','')
            label={k:row[k] for k in ['sample_id','candidate_id','local_frame','source_frame','window_id','case_id']}
            label.update(state,center_source='existing_human_click' if reuse else 'pending_new_human_click',
                center_source_file=str(challenge_path if is_anchor else old_path) if reuse else '',
                center_source_file_sha256=sha(challenge_path if is_anchor else old_path) if reuse else '',
                review_status='reused_confirmed_anchor' if is_anchor else 'unreviewed')
            labels.append(label)
            if not is_anchor:
                cases.append(dict(case_id=row['case_id'],sample_id=sample_id,window_id=wid,candidate_id=cid,local_frame=t,source_frame=source_frame,
                    source_image=str(paths[-1]),source_image_sha256=sha(paths[-1]),center_reused=bool(reuse),
                    reused_x=state['truth_x_px'] if reuse else '',reused_y=state['truth_y_px'] if reuse else '',
                    context_video=str(OUT/f'context_{row["case_id"]}_h264.mp4')))
    assert len(rows)==26 and len(cases)==24 and sum(bool(c['center_reused']) for c in cases)==2
    write(OUT/'query_model_inputs.csv',rows);write(OUT/'manual_sequence_labels.csv',labels);dump(OUT/'windows.json',windows)
    sys.path.insert(0,str(modelroot));from inference import BallInferencer
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    temporal.OUT=OUT;contexts,masks=temporal.get_contexts(infer,rows,'frozen',protected)
    preds=temporal.predictions(rows,contexts,masks,None,'short_sequence')+temporal.predictions(rows,contexts,masks,CONFIG,'short_sequence')
    write(OUT/'frozen_predictions.csv',preds)
    for c in cases:
        cid=c['candidate_id'];t=c['local_frame'];frame_dir=Path(c['source_image']).parent
        meta=load(frame_dir.parent/'runs/threshold030_median120/run_metadata.json');lo=max(0,t-24);hi=min(meta['frame_count']-1,t+24)
        fps=meta['fps'];output=Path(c['context_video']);count=0
        cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x800','-r',str(fps),'-i','pipe:0',
            '-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
        with (OUT/f'context_{c["case_id"]}_render.log').open('wb') as log:
            proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
            for j in range(lo,hi+1):
                p=frame_dir/f'{j:06d}.jpg';protected[str(p)]=sha(p);im=cv2.imread(str(p));assert im.shape==(1080,1920,3)
                frame=np.full((800,1280,3),18,np.uint8);frame[80:]=cv2.resize(im,(1280,720),interpolation=cv2.INTER_AREA)
                cv2.putText(frame,f'{c["case_id"]} | pilot{cid[-3:]} local{j} | TARGET {t} | 0.5x',(12,29),0,.74,(240,240,240),1,cv2.LINE_AA)
                if j==t:cv2.rectangle(frame,(2,81),(1277,797),(0,225,255),3);cv2.putText(frame,'TARGET FRAME',(12,61),0,.72,(0,225,255),2,cv2.LINE_AA)
                for _ in range(32 if j==t else 2):proc.stdin.write(frame.tobytes());count+=1
            proc.stdin.close();assert proc.wait()==0
        cap=cv2.VideoCapture(str(output));decoded=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(800,1280,3);decoded+=1
        cap.release();assert decoded==count
        c.update(context_video_sha256=sha(output),decoded_video_frames=decoded,duration_sec=count/fps)
    dump(OUT/'annotation_cases.json',cases)
    assert all(sha(Path(p))==h for p,h in protected.items())
    manifest=dict(status='two_fixed_windows_ready_new_labels_pending',windows=2,target_times=26,annotation_cases=24,
        new_centers_pending=22,old_centers_phase_only=2,old_confirmed_anchors_reused=2,model_predictions=52,
        predictions_sha256=sha(OUT/'frozen_predictions.csv'),cases_sha256=sha(OUT/'annotation_cases.json'),
        initial_labels_sha256=sha(OUT/'manual_sequence_labels.csv'),decoded_video_frames=sum(c['decoded_video_frames'] for c in cases),
        protected_input_hashes=protected,new_training=False,automatic_adoption=False)
    dump(OUT/'manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k!='protected_input_hashes'},indent=2),flush=True)
if __name__=='__main__':main()
