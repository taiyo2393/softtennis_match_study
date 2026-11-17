"""Freeze old methods on four temporally separate sequences; no new labels."""
import json, sys, subprocess, shutil
from pathlib import Path
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT, SOURCE, load, read, write, sha, cv2
from run_manual_ball_finetune import tensors
from compare_ball_candidate_decoders import components
from compare_temporal_candidate_paths import select, OFFSETS
from evaluate_observed_ball_fallback import camera_mask

OUT = ROOT.parent / 'ball_quality_20261104'
CFG = dict(name='displacement_0.1', order='displacement', strength=.1)

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    assert not (OUT / 'sequence_protocol.json').exists(), 'Preserve frozen experiment.'
    inventory = load(OUT / 'inventory_manifest.json')
    assert sha(OUT / 'reserved_windows.json') == inventory['reserved_windows_sha256']
    windows = load(OUT / 'reserved_windows.json')
    modelroot = SOURCE / 'third_party/RacketVision/source/BallTrack'
    weights = modelroot / 'checkpoints/balltrack_best.pth'
    protected = {str(p): sha(p) for p in [weights, OUT / 'reserved_windows.json', OUT / 'inventory_manifest.json',
        ROOT / 'reviewed_ball_overrides.json', ROOT / 'reviewed_person_overrides.json', ROOT / 'reviewed_tracking_bundles.json',
        SOURCE / 'annotations/balltrack_groundtruth_match1_pilot.csv',
        ROOT / 'ball_temporal_path_comparison/training_selection.json',
        ROOT / 'compare_temporal_candidate_paths.py', ROOT / 'compare_ball_candidate_decoders.py',
        ROOT / 'evaluate_observed_ball_fallback.py', ROOT.parent / 'ball_quality_20261102/temporal_consensus_guard.py']}
    selected = load(ROOT / 'ball_temporal_path_comparison/training_selection.json')['chosen_configuration']
    assert selected == CFG
    protocol = dict(status='fixed_before_new_predictions_or_annotations', created_date_jst='2026-11-04',
        reservation_sha256=sha(OUT / 'reserved_windows.json'), windows=4, targets_per_window=13,
        selection='Already reserved chronological strata of expanded clips, clip midpoint +/-12sourceframes step2; no scores/images used.',
        model='Original TrackNet weights only; no rejected fine-tuned model and no new candidate model.',
        methods=['bbox_area', 'displacement_0.1'], config=CFG, threshold=.3,
        inputs='Original median + BGRt-3..t,512x288,/255,CUDA AMP,batch4 ascending target/context order within clip.',
        offline='Median comes from full source clip; temporal selector uses up to8future frames. Not real-time/causal.',
        camera='Exact older heuristic: reduced-color8 JPEG,160x90 AREA,rows8:82,int16,12frame lag,meanBGR absdiff>24 fraction>.25; expand12before4after.',
        camera_partial_computation='Compute all scores needed for masks over the full context interval, including 12frame image lag and expansion support; outside region is uncomputed, not evidence of no cut.',
        annotation='52 target centers/phase states pending human review; predictions hidden. No assumed in-play/visibility.',
        comparison='Per-window phase-filtered center errors16px, missing/wrong/output-on-human-not-visible, paired gains/losses, valid consecutive sampled-pair displacement errors and correct runs. Report exclusions and macro windows; no frame-independence significance test.',
        operational_policy='No automatic adoption; keep all canonical ball/person tracking outputs unchanged.',
        scope=inventory['limitations'], protected_input_hashes=protected, script_sha256=sha(Path(__file__)))
    dump(OUT / 'sequence_protocol.json', protocol)
    shutil.copy2(Path(__file__), OUT / 'executed_prepare_source.py')
    old_hashes = set()
    for role in ['train', 'validation', 'audit']:
        for r in read(ROOT / 'balltrack_manual_finetune_pilot' / (role + '_labels.csv')):
            old_hashes.update(json.loads(r['input_frame_sha256_json']))
    queries, labels, cases, batches = [], [], [], []
    inputs_hashed, contexts, masks, camera_records, medians = {}, {}, {}, {}, {}
    sys.path.insert(0, str(modelroot))
    from inference import BallInferencer
    torch.backends.cudnn.benchmark = False
    infer = BallInferencer(str(modelroot / 'configs/tracknetv3_base.py'), str(weights), device='cuda', thre=.3, batchsize=4)
    infer.model.eval().requires_grad_(False)
    for w in windows:
        cid, wid, center = w['candidate_id'], w['window_id'], w['center_local_frame']
        fd, mp = Path(w['frame_dir']), Path(w['median_path'])
        protected[str(mp)] = sha(mp)
        protected[w['metadata_path']] = sha(Path(w['metadata_path']))
        medians[str(mp)] = np.moveaxis(cv2.resize(np.load(mp)['median'], (512,288)).astype(np.float32)/255, -1, 0)
        targets = list(range(center-12, center+13, 2))
        times = sorted({t+d for t in targets for d in OFFSETS})
        assert min(times) >= 3 and max(times) < w['frame_count']
        # For a mask index j, all activating score indexes are [j-4,j+12].
        score_first, score_last = min(times)-4, max(times)+12
        image_first = score_first-12
        small = {}
        for j in range(image_first, score_last+1):
            p = fd / f'{j:06d}.jpg'
            inputs_hashed[str(p)] = sha(p)
            im = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_8)
            small[j] = cv2.resize(im, (160,90), interpolation=cv2.INTER_AREA)[8:82].astype(np.int16)
        scores = np.zeros(w['frame_count'])
        for j in range(score_first, score_last+1):
            scores[j] = float((np.abs(small[j]-small[j-12]).mean(2)>24).mean())
        masks[cid] = camera_mask(scores)
        camera_records[cid] = dict(computed_score_first=score_first, computed_score_last=score_last,
            computed_image_first=image_first, valid_mask_first=min(times), valid_mask_last=max(times),
            score_pairs=[[j,float(scores[j])] for j in range(score_first,score_last+1)],
            mask_at_context=[[j,int(masks[cid][j])] for j in times], outside_computed_region='unknown_not_no_cut')
        contexts[cid] = {}
        for start in range(0,len(times),4):
            batch = []
            for t in times[start:start+4]:
                images=[]
                for j in range(t-3,t+1):
                    p=fd/f'{j:06d}.jpg';h=sha(p)
                    assert h not in old_hashes, 'Input duplicates a previous labelled input.'
                    inputs_hashed[str(p)] = h
                    im=cv2.imread(str(p));assert im.shape==(1080,1920,3)
                    images.append(cv2.resize(im,(512,288)))
                batch.append(dict(frames=np.stack(images),median_key=str(mp),row={}))
            with torch.inference_mode(),torch.amp.autocast('cuda'):
                maps,_,_=infer.model.forward(frames=tensors(batch,medians,infer.device))
            for t,hm in zip(times[start:start+4],maps[:,0].float().cpu().numpy(),strict=True):
                contexts[cid][t]=components(hm)
                batches.append(dict(candidate_id=cid,local_frame=t,batch_first_local_frame=times[start],batch_frames_json=json.dumps(times[start:start+4])))
        for order,t in enumerate(targets,1):
            case_id=f'{wid}{order:02d}';sample_id=f'{cid}_frame_{t:06d}'
            paths=[fd/f'{j:06d}.jpg' for j in range(t-3,t+1)]
            q=dict(sample_id=sample_id,case_id=case_id,window_id=wid,candidate_id=cid,local_frame=t,
                source_frame=w['first_source_frame']+t,truth_visibility='unreviewed',truth_x_px='',truth_y_px='',
                input_frames_json=json.dumps([str(p) for p in paths]),input_frame_sha256_json=json.dumps([sha(p) for p in paths]),
                median_path=str(mp),median_sha256=sha(mp))
            queries.append(q)
            label={k:q[k] for k in ['sample_id','candidate_id','local_frame','source_frame','window_id','case_id']}
            label.update(truth_visibility='unreviewed',truth_x_px='',truth_y_px='',phase_status='unknown',reviewer_id='',annotated_at_utc='',notes='',
                center_source='pending_new_human_click',center_source_file='',center_source_file_sha256='',review_status='unreviewed')
            labels.append(label)
            cases.append(dict(case_id=case_id,sample_id=sample_id,window_id=wid,candidate_id=cid,local_frame=t,source_frame=q['source_frame'],
                source_image=str(paths[-1]),source_image_sha256=sha(paths[-1]),center_reused=False,reused_x='',reused_y='',
                context_video=str(OUT/f'context_{case_id}_h264.mp4')))
        print('Frozen model candidates:',cid,len(times),flush=True)
    preds=[]
    for method,cfg in [('bbox_area',None),('displacement_0.1',CFG)]:
        for q in queries:
            c,status=select(contexts[q['candidate_id']],q['local_frame'],masks[q['candidate_id']],cfg)
            preds.append(dict(**{k:q[k] for k in ['sample_id','case_id','window_id','candidate_id','local_frame','source_frame']},
                method=method,x_px=c['x_px'] if c else '',y_px=c['y_px'] if c else '',visible=int(c is not None),
                confidence=c['bbox_mean'] if c else 0,selection_status=status,selected_component_index=c['component_index'] if c else -1))
    write(OUT/'query_model_inputs.csv',queries);write(OUT/'manual_sequence_labels.csv',labels)
    write(OUT/'frozen_predictions.csv',preds);write(OUT/'inference_batches.csv',batches)
    dump(OUT/'frozen_component_cache.json',contexts);dump(OUT/'camera_features.json',camera_records)
    # Render only unmarked source context for annotation; target identity is a border/header.
    for number,c in enumerate(cases):
        w=next(w for w in windows if w['window_id']==c['window_id']);fd=Path(w['frame_dir']);t=c['local_frame']
        output=Path(c['context_video']);count=0
        cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x800','-r',str(w['fps']),'-i','pipe:0',
            '-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
        with (OUT/f'render_{c["case_id"]}.log').open('wb') as log:
            proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
            for j in range(t-24,t+25):
                p=fd/f'{j:06d}.jpg';inputs_hashed[str(p)]=sha(p);im=cv2.imread(str(p));assert im.shape==(1080,1920,3)
                frame=np.full((800,1280,3),18,np.uint8);frame[80:]=cv2.resize(im,(1280,720),interpolation=cv2.INTER_AREA)
                cv2.putText(frame,f'{c["case_id"]} | expand{c["candidate_id"][-3:]} local{j} | TARGET {t} | 0.5x',(12,29),0,.74,(240,240,240),1,cv2.LINE_AA)
                if j==t:
                    cv2.rectangle(frame,(2,81),(1277,797),(0,225,255),3)
                    cv2.putText(frame,'TARGET FRAME - no model predictions shown',(12,61),0,.72,(0,225,255),2,cv2.LINE_AA)
                for repeat in range(32 if j==t else 2):proc.stdin.write(frame.tobytes());count+=1
            proc.stdin.close();assert proc.wait()==0
        c.update(context_video_sha256=sha(output),decoded_video_frames=count,duration_sec=count/w['fps'])
        if (number+1)%13==0:print('Unmarked annotation videos:',number+1,'/',len(cases),flush=True)
    dump(OUT/'annotation_cases.json',cases);dump(OUT/'source_image_hashes.json',inputs_hashed)
    template=(ROOT/'ball_short_sequence_evaluation/annotation.html').read_text(encoding='utf-8')
    template=template.replace('2場面の前後約0.4秒','4場面の各約0.4秒')
    template=template.replace('既存の回答2件は表示を省略し、別の2件はプレー区分だけ確認します。','準備中や終了後なら、その状態を選んでください。全52件は一度に終えなくても保存して再開できます。')
    template=template.replace('／ pilot${c.candidate_id.slice(-3)}','／ expand${c.candidate_id.slice(-3)}')
    template=template.replace("'保存済み 24 / 24'","`保存済み ${cases.length} / ${cases.length}`")
    (OUT/'annotation.html').write_text(template,encoding='utf-8')
    assert all(sha(Path(p))==h for p,h in protected.items())
    assert all(sha(Path(p))==h for p,h in inputs_hashed.items())
    frozen={str(OUT/name):sha(OUT/name) for name in ['sequence_protocol.json','reserved_windows.json','query_model_inputs.csv',
        'frozen_predictions.csv','inference_batches.csv','frozen_component_cache.json','camera_features.json','annotation_cases.json',
        'source_image_hashes.json','annotation.html','executed_prepare_source.py']}
    dump(OUT/'manifest.json',dict(status='frozen_predictions_and_blind_review_ready_human_labels_pending',windows=4,target_times=52,annotation_cases=52,
        model_predictions=104,predictions_sha256=sha(OUT/'frozen_predictions.csv'),cases_sha256=sha(OUT/'annotation_cases.json'),
        initial_labels_sha256=sha(OUT/'manual_sequence_labels.csv'),protected_input_hashes=protected,frozen_artifact_hashes=frozen,
        inference_context_frames=sum(len(v) for v in contexts.values()),new_training=False,new_labels=0,operational_changes=False))
    print('Prepared52 pending labels and104 frozen predictions. No accuracy computed.',flush=True)

if __name__=='__main__':main()
