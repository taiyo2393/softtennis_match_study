"""Replay original batches for gain/loss mechanisms; figures/videos, no tuning."""
import json,sys,math,subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2
from run_manual_ball_finetune import tensors
from compare_ball_candidate_decoders import components,choose

OUT=ROOT/'ball_balanced_gain_loss_diagnosis';VALID=ROOT/'ball_balanced_fixed_validation';BASE=ROOT/'ball_temporal_path_comparison'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def circle_values(hm,x,y):
    yy,xx=np.mgrid[:288,:512];mask=(xx*3.75-float(x))**2+(yy*3.75-float(y))**2<=16**2
    assert mask.any();return hm[mask]
def peak_record(hm,tx,ty):
    vals=circle_values(hm,tx,ty);y,x=np.unravel_index(int(np.argmax(hm)),hm.shape)
    return dict(global_peak=float(hm[y,x]),global_peak_source_x=x*3.75,global_peak_source_y=y*3.75,
        peak_within16px=float(vals.max()),pixels_over03_within16px=int((vals>.3).sum()),
        full_component_count=len(components(hm)))
def draw_gt(im,x,y,color=(255,255,255),size=10):
    cv2.drawMarker(im,(round(x),round(y)),(0,0,0),cv2.MARKER_CROSS,size+2,5,cv2.LINE_AA)
    cv2.drawMarker(im,(round(x),round(y)),color,cv2.MARKER_CROSS,size,2,cv2.LINE_AA)
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Keep mechanism diagnosis; no cherry-picked new settings.'
    changes=read(VALID/'changed_label_outcomes.csv');assert len(changes)==2
    source=SOURCE/'third_party/RacketVision/source/BallTrack';weight=source/'checkpoints/balltrack_best.pth';snapshot=VALID/'candidate_frozen_delta.pth'
    protected={str(p):sha(p) for p in [weight,snapshot,VALID/'protocol.json',VALID/'result.json',VALID/'verification.json',VALID/'frozen_selection.json',VALID/'operational_decision.json',VALID/'changed_label_outcomes.csv',
        ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv']}
    protocol=dict(status='fixed_before_replay_and_visualization',cases='ALL changed label outcomes from previous one-candidate validation: gain015/285 and loss002/832, no other case selection',
        replay='For every requested t-8..t+8step2, reproduce its ORIGINAL sorted per-clip batch4 using saved context keys. Same JPEGs,median,inputt-3..t,CUDA AMP.',
        threshold=.3,models=['original_weights','already_fixed_balanced_snapshot'],no_new_weights=True,no_threshold_search=True,
        measures='Whole-map peak,peak within16source-px of existing manual center,threshold component count and closest bbox center; compare saved selection vs raw.',
        causal_scope='Direct mechanism of changed output availability only. Does not isolate the neural-training root cause or prove a globally useful new method.',
        media='Same native128x128 source crop centered at humanGT,3x nearest magnification,shared0..1heatmap color scale with exact source-grid remapping. No per-model normalization.',
        context_video='Original full-frame aspect ratio1280x720+80header,0.5x,+/-24source frames,target held30extra displayframes; GT andmodelmarkers ONLY at target. No interpolated GT/trajectory.',
        unchanged_decision='Prior audit failure remains; never patch only002/832 or lower threshold using this audit case.',
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    sys.path.insert(0,str(source));from inference import BallInferencer
    infer=BallInferencer(str(source/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4);net=infer.model
    net.eval().requires_grad_(False);initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};delta=torch.load(snapshot,map_location='cpu',weights_only=False)
    cases=[];batch_log=[];maps={};measure=[];image_hashes={};medians={};frame_cache={};replayed=0
    for k,r in enumerate(changes):
        case_id=chr(65+k);role=r['role'];cid=r['candidate_id'];target=int(r['local_frame']);base=SOURCE/'data/balltrack/racketvision/match1'/cid;fd=base/'frames'
        old_path=BASE/f'{role}_component_cache.json';new_path=VALID/f'{role}_component_cache.json'
        for p in [old_path,new_path,BASE/f'{role}_selected.csv',VALID/f'{role}_selected.csv',BASE/f'{role}_context_image_hashes.json',VALID/f'{role}_context_image_hashes.json']:protected[str(p)]=sha(p)
        cache_old=load(old_path)[cid];cache_new=load(new_path)[cid];keys=sorted(map(int,cache_old));assert keys==sorted(map(int,cache_new))
        requested=[target+d for d in range(-8,9,2)];assert all(t in keys for t in requested)
        batch_starts=sorted({(keys.index(t)//4)*4 for t in requested});mp=base/'runs/threshold030_median120/median.npz';protected[str(mp)]=sha(mp)
        medians[str(mp)]=np.moveaxis(cv2.resize(np.load(mp)['median'],(512,288)).astype(np.float32)/255.,-1,0)
        expected_hashes=[load(BASE/f'{role}_context_image_hashes.json'),load(VALID/f'{role}_context_image_hashes.json')]
        for model_name,state in [('original',initial),('balanced',dict(initial,**delta['trainable_state_dict']))]:
            net.load_state_dict(state,strict=True);net.eval()
            for start in batch_starts:
                times=keys[start:start+4];batch=[]
                for t in times:
                    ims=[];paths=[]
                    for j in range(t-3,t+1):
                        p=fd/f'{j:06d}.jpg';h=sha(p);assert all(table[str(p)]==h for table in expected_hashes);image_hashes[str(p)]=h
                        if str(p) not in frame_cache:frame_cache[str(p)]=cv2.imread(str(p))
                        im=frame_cache[str(p)];assert im.shape==(1080,1920,3);ims.append(cv2.resize(im,(512,288)));paths.append(str(p))
                    batch.append(dict(frames=np.stack(ims),median_key=str(mp),row={}));batch_log.append(dict(case_id=case_id,model=model_name,batch_start_index=start,batch_frames_json=json.dumps(times),target_local_frame=t,input_paths_json=json.dumps(paths)))
                with torch.inference_mode(),torch.amp.autocast('cuda'):p,_,_=net.forward(frames=tensors(batch,medians,infer.device))
                for t,hm in zip(times,p[:,0].float().cpu().numpy()):
                    expected=(cache_old if model_name=='original' else cache_new)[str(t)];actual=components(hm);assert actual==expected,(case_id,model_name,t)
                    replayed+=1
                    if t in requested:maps[case_id,model_name,t]=hm
            for t in requested:
                hm=maps[case_id,model_name,t];ps=peak_record(hm,float(r['truth_x']),float(r['truth_y']));cs=components(hm)
                near=min((math.hypot(c['x_px']-float(r['truth_x']),c['y_px']-float(r['truth_y'])) for c in cs),default=None)
                measure.append(dict(case_id=case_id,model=model_name,local_frame=t,source_frame=int(r['source_frame'])+t-target,is_target=int(t==target),
                    **ps,closest_component_center_error_px=near,gt_scope='Exact center confirmed ONLY at target; elsewhere score probe uses fixed target location, not moving GT.'))
        old_row=next(x for x in read(BASE/f'{role}_selected.csv') if x['sample_id']==r['sample_id']);new_row=next(x for x in read(VALID/f'{role}_selected.csv') if x['sample_id']==r['sample_id'])
        assert old_row['truth_x_px']==new_row['truth_x_px']==r['truth_x'] and old_row['truth_y_px']==new_row['truth_y_px']==r['truth_y']
        meta_path=base/'runs/threshold030_median120/run_metadata.json';protected[str(meta_path)]=sha(meta_path);meta=load(meta_path)
        cases.append(dict(case_id=case_id,role=role,candidate_id=cid,target_local_frame=target,source_frame=int(r['source_frame']),manual_x=float(r['truth_x']),manual_y=float(r['truth_y']),
            source_image=str(fd/f'{target:06d}.jpg'),source_image_sha256=sha(fd/f'{target:06d}.jpg'),old_output=old_row,new_output=new_row,frame_dir=str(fd),fps=meta['fps'],frame_count=meta['frame_count']))
    write(OUT/'replay_batches.csv',batch_log);write(OUT/'response_probes.csv',measure);dump(OUT/'cases.json',cases)
    np.savez_compressed(OUT/'replayed_heatmaps.npz',case_ids=np.array([k[0] for k in maps]),model_names=np.array([k[1] for k in maps]),local_frames=np.array([k[2] for k in maps]),heatmaps=np.stack(list(maps.values())).astype(np.float16))
    font_path=Path(r'C:\Windows\Fonts\arial.ttf');font=ImageFont.truetype(str(font_path),20) if font_path.exists() else ImageFont.load_default()
    big=ImageFont.truetype(str(font_path),24) if font_path.exists() else font
    canvas=Image.new('RGB',(1200,1020),(18,26,34));draw=ImageDraw.Draw(canvas)
    draw.text((18,10),'Matched gain/loss | same source pixels,128px crop x3 | heatmap scale0..1',font=big,fill='white')
    for row,c in enumerate(cases):
        target=c['target_local_frame'];tx,ty=c['manual_x'],c['manual_y'];x0=round(tx)-64;y0=round(ty)-64
        assert 0<=x0 and x0+128<=1920 and 0<=y0 and y0+128<=1080
        im=cv2.imread(c['source_image']);crop=im[y0:y0+128,x0:x0+128].copy();top=55+row*450
        draw.text((18,top),f'{c["case_id"]} | {c["candidate_id"][-3:]}/{target} | {"gain" if row==0 else "loss"}',font=big,fill='white')
        for col,model in enumerate(['source','original','balanced']):
            left=16+col*396;header=top+34
            if model=='source':panel=crop.copy();title='Source crop / manual center';detail='White cross = previous human click'
            else:
                hm=maps[c['case_id'],model,target];yy,xx=np.mgrid[y0:y0+128,x0:x0+128]
                q=cv2.remap(hm,(xx/3.75).astype(np.float32),(yy/3.75).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                color=cv2.applyColorMap(np.clip(q*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO);panel=cv2.addWeighted(crop,.45,color,.55,0)
                rec=next(r for r in measure if r['case_id']==c['case_id'] and r['model']==model and r['is_target'])
                title='Original model' if model=='original' else 'Balanced model'
                detail=f'peak16={rec["peak_within16px"]:.6f}; components={rec["full_component_count"]}'
                output=c['old_output'] if model=='original' else c['new_output']
                if int(output['visible']):cv2.circle(panel,(round(float(output['x_px'])-x0),round(float(output['y_px'])-y0)),8,(0,170,255) if model=='original' else (255,220,0),1,cv2.LINE_AA)
            draw_gt(panel,tx-x0,ty-y0,size=8);tile=cv2.resize(panel,(384,384),interpolation=cv2.INTER_NEAREST)
            draw.text((left,header),title,font=font,fill='white');draw.text((left,header+24),detail,font=font,fill=(205,220,230));canvas.paste(Image.fromarray(cv2.cvtColor(tile,cv2.COLOR_BGR2RGB)),(left,header+52))
    gradient=np.linspace(0,255,1152).astype(np.uint8)[None,:];bar=cv2.applyColorMap(np.repeat(gradient,12,axis=0),cv2.COLORMAP_TURBO)
    canvas.paste(Image.fromarray(cv2.cvtColor(bar,cv2.COLOR_BGR2RGB)),(18,962));draw.text((18,978),'0.0',font=font,fill='white');draw.text((332,978),'0.3 threshold',font=font,fill='white');draw.text((1110,978),'1.0',font=font,fill='white');draw.line((18+round(1152*.3),958,18+round(1152*.3),977),fill='white',width=2)
    canvas.save(OUT/'paired_evidence.png')
    all_frames=[];video=OUT/'gain_loss_context_h264.mp4';fps=cases[0]['fps'];assert all(c['fps']==fps for c in cases)
    cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x800','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    expected=0;video_records=[]
    with (OUT/'video_render.log').open('wb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for c in cases:
            t=c['target_local_frame'];lo=max(0,t-24);hi=min(c['frame_count']-1,t+24);start=expected
            for j in range(lo,hi+1):
                p=Path(c['frame_dir'])/f'{j:06d}.jpg';image_hashes[str(p)]=sha(p);image=cv2.imread(str(p));assert image.shape==(1080,1920,3)
                if j==t:
                    draw_gt(image,c['manual_x'],c['manual_y'],size=20)
                    for key,color,radius in [('old_output',(0,170,255),12),('new_output',(255,220,0),18)]:
                        r=c[key]
                        if int(r['visible']):cv2.circle(image,(round(float(r['x_px'])),round(float(r['y_px']))),radius,color,2,cv2.LINE_AA)
                frame=np.full((800,1280,3),18,np.uint8);frame[80:]=cv2.resize(image,(1280,720),interpolation=cv2.INTER_AREA)
                cv2.putText(frame,f'{c["case_id"]}: pilot{c["candidate_id"][-3:]} local{j} | target{t} | original/balanced comparison',(10,29),0,.72,(240,240,240),1,cv2.LINE_AA)
                cv2.putText(frame,'TARGET ONLY: white=manual center; orange=original; cyan=balanced. No marker elsewhere.',(10,61),0,.67,(220,220,220),1,cv2.LINE_AA)
                if j==t:cv2.rectangle(frame,(2,81),(1277,797),(0,225,255),3)
                for _ in range(32 if j==t else 2):proc.stdin.write(frame.tobytes());expected+=1
            video_records.append(dict(case_id=c['case_id'],first_display_frame=start,last_display_frame=expected-1,source_local_range=[lo,hi]))
        proc.stdin.close();assert proc.wait()==0
    cap=cv2.VideoCapture(str(video));decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(800,1280,3);decoded+=1
    cap.release();assert decoded==expected
    target_records=[r for r in measure if r['is_target']];write(OUT/'target_response_summary.csv',target_records)
    conclusions=[]
    for c in cases:
        old=next(r for r in target_records if r['case_id']==c['case_id'] and r['model']=='original');new=next(r for r in target_records if r['case_id']==c['case_id'] and r['model']=='balanced')
        mechanism='response_crosses_threshold_into_detection' if old['full_component_count']==0 and new['full_component_count']>0 else 'response_drops_below_threshold_into_missing' if old['full_component_count']>0 and new['full_component_count']==0 else 'other'
        conclusions.append(dict(case_id=c['case_id'],candidate_id=c['candidate_id'],local_frame=c['target_local_frame'],old_peak16=old['peak_within16px'],new_peak16=new['peak_within16px'],
            old_global_peak=old['global_peak'],new_global_peak=new['global_peak'],old_components=old['full_component_count'],new_components=new['full_component_count'],mechanism=mechanism,
            selector_not_responsible_for_target_presence=True if mechanism!='other' else False))
    assert all(sha(Path(p))==h for p,h in protected.items()) and all(sha(Path(p))==h for p,h in image_hashes.items())
    dump(OUT/'source_image_hashes.json',image_hashes)
    result=dict(status='matched_batch_gain_loss_mechanisms_verified',conclusions=conclusions,replayed_frame_outputs_matching_saved_components=replayed,
        requested_model_heatmaps=len(maps),video=dict(path=str(video),sha256=sha(video),display_frames=decoded,duration_sec=decoded/fps,segments=video_records),
        figure=dict(path=str(OUT/'paired_evidence.png'),sha256=sha(OUT/'paired_evidence.png'),source_crop128px=True,shared_heatmap_scale=[0,1]),
        protected_inputs_unchanged=True,no_threshold_change=True,no_new_training=True,no_new_labels=True,operational_outputs_changed=False,prior_audit_rejection_preserved=True,
        protected_input_hashes=protected)
    dump(OUT/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='protected_input_hashes'},indent=2))
if __name__=='__main__':main()
