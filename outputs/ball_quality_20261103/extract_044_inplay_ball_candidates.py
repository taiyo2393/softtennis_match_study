"""Extract all unchanged-model ball components in the existing044 play scope."""
import csv,json,sys
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'ball_quality_20261102'))
from extract_weak_candidates import components
from extract_multiball_candidates import SOURCE,read,window,sha

OUT=ROOT/'inplay_ball_044'

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'candidates.csv').exists()
    scopepath=ROOT/'play_start_audit/044_scope_report.json';scope=json.loads(scopepath.read_text())
    start,end=scope['candidate_main_play_local_frames'];cid=scope['ball_pilot_id']
    base=SOURCE/'data/balltrack/racketvision/match1'/cid
    baseline_path=ROOT.parent/'tracking_quality_20261102/ball_aligned'/cid/'balltrack_predictions.csv'
    baseline=read(baseline_path);meta=json.loads((base/'runs/threshold030_median120/run_metadata.json').read_text())
    ballroot=SOURCE/'third_party/RacketVision/source/BallTrack';sys.path.insert(0,str(ballroot))
    from inference import BallInferencer
    checkpoint=ballroot/'checkpoints/balltrack_best.pth'
    protocol=dict(pilot=cid,rally='044',local_frames=[start,end],fps=meta['fps'],
        boundary_status='Existing AI contact/end intervals; second serve validity previously human confirmed',
        model='unchanged RacketVision MS-TrackNetV3',checkpoint_sha256=sha(checkpoint),
        input='BGR JPEG t-3..t, median120, resize512x288, divide255',
        components='peak floor0.10, top6, half-peak weighted centroid',
        baseline_sha256=sha(baseline_path),scope_sha256=sha(scopepath),script_sha256=sha(Path(__file__)),
        development_only=True,ball_identity_annotations_used=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    infer=BallInferencer(str(ballroot/'configs/tracknetv3_base.py'),str(checkpoint),device='cuda',thre=.3,batchsize=8)
    median=np.load(base/'runs/threshold030_median120/median.npz')['median']
    median=np.moveaxis(cv2.resize(median,(infer.width,infer.height))/255.,-1,0)
    cache={}
    for t in range(start-3,end+1):
        im=cv2.imread(str(base/'frames'/f'{t:06d}.jpg'));assert im is not None
        cache[t]=np.moveaxis(cv2.resize(im,(infer.width,infer.height)).astype(np.float32)/255.,-1,0)
    candidates=[];checks=[];heatmaps=[]
    for first in range(start,end+1,8):
        times=list(range(first,min(end+1,first+8)))
        batch=np.array([np.concatenate([median[None],np.stack([cache[j] for j in window(t)])],axis=0).reshape(-1,infer.height,infer.width) for t in times])
        with torch.inference_mode(),torch.amp.autocast('cuda'):
            predicted,_,_=infer.model.forward(frames=torch.from_numpy(batch).float().to(infer.device))
        for t,hm in zip(times,predicted[:,0].float().cpu().numpy()):
            x,y,w,h,conf=infer._predict_location(hm)
            px=int((x+w/2)*1920/infer.width);py=int((y+h/2)*1080/infer.height)
            present=int(px!=0 or py!=0);old=baseline[t]
            delta=float(np.hypot(px-float(old['x_px']),py-float(old['y_px'])))
            assert present==int(old['visible']) and delta<=2.,(t,present,old,delta)
            checks.append(dict(local_frame=t,delta_px=delta,visible=present))
            for rank,c in enumerate(components(hm,1920,1080),1):candidates.append(dict(local_frame=t,rank=rank,**c))
            heatmaps.append(hm.astype(np.float16))
        if (first-start)%80==0:print('044',first,'/',end,flush=True)
    with (OUT/'candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(candidates[0]));writer.writeheader();writer.writerows(candidates)
    np.savez_compressed(OUT/'heatmaps.npz',heatmaps=np.stack(heatmaps),local_frames=np.arange(start,end+1))
    report=dict(frames=end-start+1,baseline_outputs=sum(int(baseline[t]['visible']) for t in range(start,end+1)),
        components=len(candidates),baseline_reproduction_checks=checks,
        warning='Model components and temporal consistency are not independently verified ball centers.')
    (OUT/'extraction_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='baseline_reproduction_checks'},indent=2))

if __name__=='__main__':main()
