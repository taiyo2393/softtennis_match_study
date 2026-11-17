"""Freeze original-model056predictions for a second assisted-time case;no adaptation."""
import hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,read,load,write,sha,cv2,np
from run_manual_ball_finetune import tensors
from calibrate_point_supervised_detector import compact,expand

BASE=ROOT.parent/'ball_quality_20261109'
OUT=BASE/'rally_repair_056'
PARENT=BASE/'rally_repair_pilot_044_v2'
FIRST,LAST=154078,154880
CORE_FIRST,CORE_LAST=154087,154839
TARGETS=list(range(154088,154839,2))


def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def stream_sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def replace_once(s,a,b):
    assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)


def clock_check(source):
    cap=cv2.VideoCapture(str(source));assert cap.get(cv2.CAP_PROP_FRAME_WIDTH)==3840 and cap.get(cv2.CAP_PROP_FRAME_HEIGHT)==2160
    assert abs(cap.get(cv2.CAP_PROP_FPS)-60000/1001)<1e-6
    cells=[]
    for f in range(154079,154088):
        cap.set(1,f);ok,im=cap.read();assert ok
        tile=cv2.resize(im[700:1750,750:1550],(400,525));cv2.rectangle(tile,(0,0),(220,36),(0,0,0),-1);cv2.putText(tile,f'F{f}',(10,27),0,.8,(255,255,255),2);cells.append(tile)
    sheet=np.vstack([np.hstack(cells[i:i+3]) for i in range(0,9,3)]);ok,b=cv2.imencode('.jpg',sheet);assert ok
    contact=ROOT.parent/'frontcourt_handoff_20260927/056_contact_every_frame_review.jpg';assert b.tobytes()==contact.read_bytes()
    cells=[]
    for f in range(154865,154899,3):
        cap.set(1,f);ok,im=cap.read();assert ok
        tile=cv2.resize(im[350:1450,1150:2450],(540,600));cv2.rectangle(tile,(0,0),(160,30),(0,0,0),-1);cv2.putText(tile,str(f),(5,23),0,.7,(255,255,255),2);cells.append(tile)
    sheet=np.vstack([np.hstack(cells[i:i+4]) for i in range(0,12,4)]);ok,b=cv2.imencode('.jpg',sheet);assert ok
    finish=ROOT.parent/'frontcourt_handoff_20260927/056_finish_4k_0.jpg';assert b.tobytes()==finish.read_bytes();cap.release()
    return dict(status='passed',contact_frames=list(range(154079,154088)),finish_frames=list(range(154865,154899,3)),
        contact_sheet_sha256=sha(contact),finish_sheet_sha256=sha(finish),both_sheets_byte_identical=True,
        conclusion='Raw match1.mp4 matches legacy056clock;not a certificate of exact serve/end event boundaries.')


def make_tools():
    source=(ROOT/'serve_rally_repair_pilot.py').read_text(encoding='utf-8')
    source=replace_once(source,'assert len(cases)==201','assert len(cases)==376')
    source=replace_once(source,'targets=201,pending=195','targets=376,pending=376')
    prefix='import sys\nsys.path.insert(0,'+repr(str(ROOT))+')\n'
    (OUT/'serve_review.py').write_text(prefix+source,encoding='utf-8')
    html=(ROOT/'rally_repair_pilot.html').read_text(encoding='utf-8')
    html=replace_once(html,'<title>044の見直し時間を測る</title>','<title>056の見直し時間を測る</title>')
    html=replace_once(html,'<h1>044：見直し・修正にかかる時間を測る</h1>','<h1>056：見直し・修正にかかる時間を測る</h1>')
    html=replace_once(html,'主プレー候補区間の201時点を2フレーム間隔で確認します。以前の手動中心6点は入力済みで、新しく確認するのは195時点です。サーブの有効性や足の接地は再確認しません。',
        '主プレー候補区間の376時点を2フレーム間隔で確認します。今回は全て未入力です。サーブ有効・2バウンドで終了という以前の確認を引き継ぎ、再質問しません。')
    html=replace_once(html,'水色は先行修正を含む現在の球出力です。','水色は今回固定した元モデルの未修正出力です。')
    html=replace_once(html,'`044 ／ frame ${c.local_frame} ／ ${index+1} / ${cases.length}`','`056 ／ 元frame ${c.source_frame}（local ${c.local_frame}）／ ${index+1} / ${cases.length}`')
    (OUT/'review.html').write_text(html,encoding='utf-8')
    shutil.copy2(PARENT/'start_review.ps1',OUT/'start_review.ps1')
    finalizer=(ROOT/'finalize_rally_repair_pilot.py').read_text(encoding='utf-8')
    finalizer=replace_once(finalizer,"OUT=ROOT.parent/'ball_quality_20261109/rally_repair_pilot_044_v2'","OUT=ROOT.parent/'ball_quality_20261109/rally_repair_056'")
    finalizer=re.sub(r'\b201\b','376',finalizer);finalizer=re.sub(r'\b195\b','376',finalizer)
    finalizer=replace_once(finalizer,"rally_candidate_id='match1_rally_window_044'","rally_candidate_id='match1_rally_window_056'")
    finalizer=replace_once(finalizer,'candidate_scope_source_frames=402,candidate_scope_seconds=402/','candidate_scope_source_frames=753,candidate_scope_seconds=753/')
    finalizer=replace_once(finalizer,'reused_centers_unchanged=6','reused_centers_unchanged=0')
    finalizer=replace_once(finalizer,'Remaining review/edit work after historical corrections;instructions/setup excluded;not total raw-automation-to-final cost.',
        'Review/edit of an unpatched original-model baseline;instructions/setup excluded;not a from-scratch timing comparison.')
    finalizer=replace_once(finalizer,'One previously inspected,partly repaired same-match candidate interior;not representative labor or accuracy benchmark.',
        'One known same-match candidate interior with unpatched original-model ball baseline;not representative labor or independent accuracy benchmark.')
    (OUT/'finalize_review.py').write_text(prefix+finalizer,encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists(),'Preserve frozen outputs and future human answers.'
    source=SOURCE/'data/raw/match1.mp4';source_hash=stream_sha(source);source_stat=source.stat()
    clock=clock_check(source);dump(OUT/'source_clock_check.json',clock)
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth'
    paths=[weight,ROOT/'serve_rally_repair_pilot.py',ROOT/'rally_repair_pilot.html',ROOT/'finalize_rally_repair_pilot.py',
        PARENT/'start_review.ps1',PARENT/'completed_review/result.json',PARENT/'completed_review/verification.json',
        ROOT.parent/'frontcourt_handoff_20260927/human_review_log.csv',ROOT.parent/'frontcourt_handoff_20260927/rally_label_review.csv',
        ROOT.parent/'frontcourt_handoff_20260927/boundary_review_queue.csv',ROOT.parent/'frontcourt_handoff_20260927/event_review_ranges.csv',
        ROOT.parent/'frontcourt_handoff_20260927/056_contact_every_frame_review.jpg',ROOT.parent/'frontcourt_handoff_20260927/056_finish_4k_0.jpg',
        ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json']
    protected={str(p):sha(p) for p in paths}
    protocol=dict(status='fixed_before_new_model_predictions_or_review',date_jst='2026-11-09',rally='match1_rally_window_056',ball_clip='match1_rally_056_raw_context',
        purpose='Second assisted usability/time case,then stop collecting further timed cases for an initial assessment. Not a new detector-improvement trial.',
        selection='056 has pre-existing human opening-serve-valid and two-bounces-end labels;selected without viewing new detector output. No new questions on validity or end type.',
        interval=dict(context_source_frames=[FIRST,LAST],candidate_core_source_frames=[CORE_FIRST,CORE_LAST],contact_bracket=[154081,154086],end_bracket=[154840,154880],
            target_source_frames=[TARGETS[0],TARGETS[-1]],target_stride=2,targets=len(TARGETS)),
        boundary_status='Inherited AI intervals,not exact confirmed event frames. Conservative core excludes both uncertainty brackets;not full-rally/all-source-frame accuracy.',
        source_video=str(source),source_video_sha256=source_hash,source_video_bytes=source_stat.st_size,source_fps=60000/1001,source_wh=[3840,2160],clock='legacy_edited_match1',
        clock_validation='Both old contact/end JPEG sheets regenerated byte-identically from current raw video;redownloaded match1_original not used.',
        frame_cache='Raw4K->1920x1080 INTER_AREA,saveJPEGquality95;decode those SAME files for both model and review. No person/circle overlays in inference inputs.',
        detector='Original unchanged TrackNet checkpoint,global512x288BILINEAR,BGRmedian+t-3..t,/255,CUDA AMP,batch4,strictthreshold0.3,unchanged largest-bbox decoder. No rejected weights,tiles,new classifier,interpolation,or manually selected candidate.',
        background='Pixelwise uint8native1920median of120uniformly sampled cached context frames;resize512x288. Fixed offline whole-context input,not causal real-time processing.',
        first_three_context_frames='No prediction because full4framecontext is unavailable;all376targets lie later and have complete context.',
        baseline='New original-model output with no human position overrides. Thus different input status from already repaired044;do not call a difference in time a controlled method effect.',
        reuse='No precise human ball centers for these targets are available in the earlier20ball clips. Prior person/serve/event confirmations are not ball-center labels.',
        annotation='Same prediction-assisted workflow and timer as044. Accept,click edit,not_visible,uncertain,nonplay. Accepted predictions are not independent precision GT;do not auto-fit review answers.',
        timing='Explicit start/pause,focus/visibility pause,heartbeat5s,15sper-gap cap with excess recorded,60sidle pause. Historical setup and source/model processing excluded from operator time.',
        decision='After044and056report cases separately,not population mean or time-savings claim. Stop additional annotation requests until these results are assessed.',
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)),new_training=False,canonical_tracking_changed=False)
    dump(OUT/'protocol.json',protocol);shutil.copy2(Path(__file__),OUT/'executed_prepare_source.py')
    frame_dir=OUT/'frames';frame_dir.mkdir();images={};cap=cv2.VideoCapture(str(source));cap.set(1,FIRST)
    for source_frame in range(FIRST,LAST+1):
        ok,im=cap.read();assert ok and im.shape==(2160,3840,3)
        im=cv2.resize(im,(1920,1080),interpolation=cv2.INTER_AREA);p=frame_dir/f'{source_frame-FIRST:06d}.jpg'
        assert cv2.imwrite(str(p),im,[cv2.IMWRITE_JPEG_QUALITY,95]);images[str(p)]=sha(p)
    cap.release();dump(OUT/'source_image_hashes.json',images)
    sample_ids=np.unique(np.linspace(0,LAST-FIRST,120,dtype=int));assert len(sample_ids)==120
    median=np.empty((1080,1920,3),np.uint8)
    # Stripwise computation preserves exact pixelwise median while bounding memory.
    for y in range(0,1080,120):
        strip=np.stack([cv2.imread(str(frame_dir/f'{i:06d}.jpg'))[y:y+120] for i in sample_ids])
        median[y:y+120]=np.median(strip,axis=0).astype(np.uint8)
    np.savez_compressed(OUT/'median.npz',median=median)
    dump(OUT/'median_samples.json',dict(local_frame_indexes=sample_ids.tolist(),source_frames=(sample_ids+FIRST).tolist(),median_sha256=sha(OUT/'median.npz')))
    sys.path.insert(0,str(modelroot));from inference import BallInferencer
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    engine=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    engine.model.eval().requires_grad_(False)
    medians={'m':np.moveaxis(cv2.resize(median,(512,288)).astype(np.float32)/255.,-1,0)}
    predictions=[];idxs=[];values=[];offsets=[0];records=[]
    for t in range(3):predictions.append(dict(candidate_id='match1_rally_056_raw_context',local_frame=t,source_frame=FIRST+t,source_timestamp_sec=(FIRST+t)/(60000/1001),x_px=0,y_px=0,visible=0,confidence=0.,position_source='context_only_not_inferred',input_first_source_frame='',input_last_source_frame=''))
    for lo in range(3,LAST-FIRST+1,4):
        times=list(range(lo,min(lo+4,LAST-FIRST+1)));batch=[]
        for t in times:
            frames=np.stack([cv2.resize(cv2.imread(str(frame_dir/f'{j:06d}.jpg')),(512,288)) for j in range(t-3,t+1)])
            batch.append(dict(frames=frames,median_key='m',row={}))
        with torch.inference_mode(),torch.amp.autocast('cuda'):output,_,_=engine.model.forward(frames=tensors(batch,medians,engine.device))
        for t,hm in zip(times,output[:,0].float().cpu().numpy(),strict=True):
            x,y,w,h,conf=engine._predict_location(hm);px,py=int((x+w/2)*1920/512),int((y+h/2)*1080/288)
            predictions.append(dict(candidate_id='match1_rally_056_raw_context',local_frame=t,source_frame=FIRST+t,source_timestamp_sec=(FIRST+t)/(60000/1001),x_px=px,y_px=py,
                visible=int(px!=0 or py!=0),confidence=float(conf),position_source='original_model_observed',input_first_source_frame=FIRST+t-3,input_last_source_frame=FIRST+t))
            idx,v=compact(hm);assert engine._predict_location(expand(idx,v))==engine._predict_location(hm)
            idxs.append(idx);values.append(v);offsets.append(offsets[-1]+len(idx));records.append(dict(map_index=len(records),local_frame=t,source_frame=FIRST+t,batch_frames_json=json.dumps(times)))
        if (lo-3)%160==0:print('056 baseline',lo,'/',LAST-FIRST+1,flush=True)
    assert len(predictions)==803 and len(records)==800
    write(OUT/'baseline_all_frames.csv',predictions);write(OUT/'map_records.csv',records)
    np.savez_compressed(OUT/'component_rectangles.npz',indices=np.concatenate(idxs),values=np.concatenate(values),offsets=np.array(offsets,dtype=np.int64))
    by_source={int(r['source_frame']):r for r in predictions};cases=[];labels=[];baseline=[]
    for number,f in enumerate(TARGETS,1):
        b=by_source[f];cid=f'R{number:03d}';p=frame_dir/f'{f-FIRST:06d}.jpg'
        cases.append(dict(case_id=cid,local_frame=f-FIRST,source_frame=f,source_image=str(p),source_image_sha256=sha(p),prediction_visible=b['visible'],prediction_x_px=float(b['x_px']),prediction_y_px=float(b['y_px']),prior_candidate_review='',prior_review_id='',reused_manual_center=False,video_target_time_sec=(f-FIRST+.25)/(60000/1001)))
        labels.append(dict(case_id=cid,local_frame=f-FIRST,source_frame=f,review_status='unreviewed',decision='',x_px='',y_px='',coordinate_source='',independent_new_center_gt=0,prior_center_source_file='',prior_center_source_sha256='',reviewer_id='',reviewed_at_utc='',notes=''));baseline.append(b)
    write(OUT/'review_labels.csv',labels);shutil.copy2(OUT/'review_labels.csv',OUT/'initial_labels_snapshot.csv');write(OUT/'baseline_predictions.csv',baseline);dump(OUT/'cases.json',cases)
    silent=OUT/'context_silent.mp4';cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x800','-r',str(60000/1001),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(silent)]
    with (OUT/'render.log').open('wb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for b in predictions:
            im=cv2.imread(str(frame_dir/f'{b["local_frame"]:06d}.jpg'));frame=np.full((800,1280,3),18,np.uint8);frame[80:]=cv2.resize(im,(1280,720),interpolation=cv2.INTER_AREA)
            if b['visible']:cv2.circle(frame,(round(b['x_px']*2/3),80+round(b['y_px']*2/3)),8,(255,220,60),2,cv2.LINE_AA)
            cv2.putText(frame,f'056 | source {b["source_frame"]} | local {b["local_frame"]} | unpatched original model',(12,29),0,.65,(240,240,240),1,cv2.LINE_AA)
            cv2.putText(frame,'Cyan: input candidate, not ground truth | no interpolation',(12,61),0,.64,(210,220,230),1,cv2.LINE_AA);proc.stdin.write(frame.tobytes())
        proc.stdin.close();assert proc.wait()==0
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-i',str(silent),'-ss',str(FIRST/(60000/1001)),'-i',str(source),'-map','0:v:0','-map','1:a:0?','-t',str(803/(60000/1001)),'-c:v','copy','-c:a','aac','-movflags','+faststart',str(OUT/'context.mp4')],check=True)
    make_tools()
    summary=dict(candidate_core_source_frames=[CORE_FIRST,CORE_LAST],candidate_core_frames=753,candidate_core_seconds=753/(60000/1001),review_sample_times=376,review_sample_output_times=sum(c['prediction_visible'] for c in cases),reused_precise_centers=0,new_review_targets=376,output_counts_are_not_accuracy=True,baseline_has_prior_human_position_repairs=False)
    dump(OUT/'initial_readiness.json',summary)
    assert source.stat().st_size==source_stat.st_size and stream_sha(source)==source_hash
    assert all(sha(Path(p))==h for p,h in protected.items())
    frozen={n:sha(OUT/n) for n in ['protocol.json','executed_prepare_source.py','source_clock_check.json','source_image_hashes.json','median.npz','median_samples.json','baseline_all_frames.csv','map_records.csv','component_rectangles.npz','baseline_predictions.csv','cases.json','initial_labels_snapshot.csv','context.mp4','review.html','serve_review.py','start_review.ps1','finalize_review.py','initial_readiness.json']}
    dump(OUT/'manifest.json',dict(status='original_model_fixed_human_review_pending',targets=376,pending=376,protected_input_hashes=protected,frozen_artifact_hashes=frozen))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
