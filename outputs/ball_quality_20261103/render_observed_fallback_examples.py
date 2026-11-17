"""Observed fallback proposals: four labelled gains and one labelled error.

Surrounding frames are unlabelled, not auto-confirmed by the target annotation.
Keep rejected algorithm as a diagnostic proposal; do not update any registry.
"""
import json,sys,subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2
from evaluate_observed_ball_fallback import decide,camera_mask

OUT=ROOT/'balltrack_observed_fallback'

def main():
    assert not (OUT/'video_report.json').exists()
    training=ROOT/'balltrack_manual_finetune_pilot';manifest=load(training/'data_manifest.json')
    checkpoint=Path(load(training/'checkpoint_selection.json')['selected_checkpoint'])
    rawroot=ROOT.parent/'tracking_quality_20261102/ball_aligned'
    earlier=read(OUT/'validation_decisions.csv')
    targets=[r for r in earlier if r['threshold']=='0.3' and r['decision']=='add_observed_finetuned_candidate']
    assert len(targets)==5
    labels={r['sample_id']:r for r in read(training/'validation_labels.csv')}
    rows=[];cases=[];raw={};masks={};medians={};cache={}
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';sys.path.insert(0,str(modelroot));from inference import BallInferencer
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(checkpoint),device='cuda',thre=.3,batchsize=4)
    for target in targets:
        cid=target['candidate_id'];t=int(target['local_frame']);base=SOURCE/'data/balltrack/racketvision/match1'/cid
        if cid not in raw:
            raw[cid]=read(rawroot/cid/'balltrack_predictions.csv')
            masks[cid]=camera_mask(load(ROOT.parent/'ball_quality_20261102/temporal_guard/camera_features'/(cid+'.json'))['changed_fraction'])
            median=np.load(base/'runs/threshold030_median120/median.npz')['median']
            medians[cid]=np.moveaxis(cv2.resize(median,(512,288)).astype(np.float32)/255.,-1,0)
        start,end=t-15,t+15;assert 0<=start<end<len(raw[cid]);samples=[]
        for j in range(start-3,end+1):
            im=cv2.imread(str(base/'frames'/f'{j:06d}.jpg'));assert im is not None
            cache[cid,j]=np.moveaxis(cv2.resize(im,(512,288)).astype(np.float32)/255.,-1,0)
        for j in range(start,end+1):samples.append((cid,j))
        for first in range(0,len(samples),4):
            batch=samples[first:first+4]
            data=np.stack([np.concatenate([medians[cid][None],np.stack([cache[cid,k] for k in range(j-3,j+1)])],axis=0).reshape(15,288,512) for cid,j in batch])
            with torch.inference_mode(),torch.amp.autocast('cuda'):
                maps,_,_=infer.model.forward(frames=torch.from_numpy(data).to(infer.device))
            for (cid,j),hm in zip(batch,maps[:,0].float().cpu().numpy()):
                x,y,w,h,conf=infer._predict_location(hm);px=int((x+w/2)*1920/512);py=int((y+h/2)*1080/288)
                proposal=dict(x_px=px,y_px=py,visible=int(px!=0 or py!=0),confidence=conf)
                d=decide(raw[cid],j,proposal,masks[cid]);p=raw[cid][j];add=d['decision']=='add_observed_finetuned_candidate'
                label=labels[target['sample_id']] if j==t else None
                rows.append(dict(candidate_id=cid,local_frame=j,target_frame=t,original_x=p['x_px'],original_y=p['y_px'],original_visible=p['visible'],
                    finetuned_x=px,finetuned_y=py,finetuned_visible=proposal['visible'],
                    output_x=px if add else p['x_px'],output_y=py if add else p['y_px'],output_visible=proposal['visible'] if add else p['visible'],
                    proposed_addition=int(add),gate_decision=d['decision'],manual_center_x=label['truth_x_px'] if label else '',
                    manual_center_y=label['truth_y_px'] if label else '',new_human_review_status='existing_manual_target' if label else 'not_annotated_in_this_preview',interpolated=0))
        cases.append(dict(candidate_id=cid,target_frame=t,start=start,end=end,sample_id=target['sample_id']))
    write(OUT/'continuous_diagnostic_proposals.csv',rows)
    by={(r['candidate_id'],r['local_frame']):r for r in rows}
    saved={(r['candidate_id'],int(r['local_frame'])):r for r in read(training/'calibration/validation_threshold30.csv')}
    gains=0;errors=0
    for c in cases:
        r=by[c['candidate_id'],c['target_frame']];old=saved[c['candidate_id'],c['target_frame']]
        assert abs(float(r['finetuned_x'])-float(old['x_px']))<=2 and abs(float(r['finetuned_y'])-float(old['y_px']))<=2
        c['target_error_px']=float(np.hypot(float(r['output_x'])-float(r['manual_center_x']),float(r['output_y'])-float(r['manual_center_y'])))
        c['target_matches_manual16px']=c['target_error_px']<=16;gains+=c['target_matches_manual16px'];errors+=not c['target_matches_manual16px']
    assert gains==4 and errors==1
    final=OUT/'fallback_gains_and_error_h264.mp4';fps=59.94005994005994;encoded=0
    cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1920x660','-r',str(fps),'-i','pipe:0',
        '-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(final)]
    with (OUT/'video_render.log').open('wb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for i,c in enumerate(cases,1):
            cid=c['candidate_id'];target=by[cid,c['target_frame']]
            points=np.array([[float(target['output_x']),float(target['output_y'])],[float(target['manual_center_x']),float(target['manual_center_y'])]])
            center=(points.min(0)+points.max(0))/2;left=max(0,min(1440,round(center[0])-240));top=max(0,min(810,round(center[1])-135))
            c['crop_xywh']=[left,top,480,270]
            for t in range(c['start'],c['end']+1):
                im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{t:06d}.jpg'));assert im is not None
                canvas=np.full((660,1920,3),22,np.uint8);canvas[120:,:960]=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)
                canvas[120:,960:]=cv2.resize(im[top:top+270,left:left+480],(960,540),interpolation=cv2.INTER_LINEAR)
                r=by[cid,t]
                def circle(x,y,color):
                    cv2.circle(canvas,(round(x*.5),120+round(y*.5)),7,color,2,cv2.LINE_AA)
                    if left<=x<left+480 and top<=y<top+270:cv2.circle(canvas,(960+round((x-left)*2),120+round((y-top)*2)),14,color,2,cv2.LINE_AA)
                if int(r['original_visible']):circle(float(r['original_x']),float(r['original_y']),(255,255,0))
                if r['proposed_addition']:circle(float(r['output_x']),float(r['output_y']),(255,150,60))
                if t==c['target_frame']:
                    x,y=float(r['manual_center_x']),float(r['manual_center_y'])
                    for px,py in [(round(x*.5),120+round(y*.5)),(960+round((x-left)*2),120+round((y-top)*2))]:
                        cv2.line(canvas,(px-10,py),(px+10,py),(80,255,100),2);cv2.line(canvas,(px,py-10),(px,py+10),(80,255,100),2)
                cv2.putText(canvas,f'{i}/5 | pilot{cid[-3:]} local {t} | target {c["target_frame"]} | 0.5x + target hold | NOT ADOPTED',(16,29),0,.76,(245,245,245),1,cv2.LINE_AA)
                cv2.putText(canvas,'Cyan=original; BLUE=new observed fallback; green cross=manual center ONLY at target',(16,67),0,.68,(235,235,235),1,cv2.LINE_AA)
                cv2.putText(canvas,('Target matches manual16px' if c['target_matches_manual16px'] else 'Target is WRONG: racket-area candidate')+' | Surrounding frames have no new groundtruth',(16,106),0,.65,(80,255,100) if c['target_matches_manual16px'] else (90,150,255),1,cv2.LINE_AA)
                repeats=2+(30 if t==c['target_frame'] else 0)
                if t==c['target_frame']:cv2.imwrite(str(OUT/f'proposal_{cid[-3:]}_{t}.jpg'),canvas)
                for _ in range(repeats):proc.stdin.write(canvas.tobytes());encoded+=1
        proc.stdin.close();assert proc.wait()==0
    cap=cv2.VideoCapture(str(final));n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(660,1920);n+=1
    cap.release();assert n==encoded==460
    assert all(sha(Path(p))==h for p,h in manifest['input_hashes'].items())
    report=dict(video=str(final.resolve()),video_sha256=sha(final),decoded_frames=n,duration_sec=n/fps,
        cases=cases,manual_target_agreements=4,manual_target_wrong_locations=1,
        unlabelled_surrounding_frames_not_counted_as_accuracy=True,original_observations_preserved=True,
        no_interpolation=True,automatic_adoption=False,reviewed_registry_and_source_labels_unchanged=True,
        source_aspect_preserved=True,crop_uniform_enlargement=2)
    (OUT/'video_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))

if __name__=='__main__':main()
