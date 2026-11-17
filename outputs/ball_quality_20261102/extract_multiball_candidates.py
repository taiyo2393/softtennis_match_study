"""Retain multiple observed heatmap components before selecting a target ball."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
DEST=ROOT.parent/'ball_quality_20261103/multiball_pilot_002'
sys.path.insert(0,str(ROOT.parent/'tracking_quality_20261102'))
from ball_alignment import SOURCE,read,window
from extract_weak_candidates import components


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    DEST.mkdir(parents=True,exist_ok=True);assert not (DEST/'candidates.csv').exists()
    cid='match1_motion_pilot_002';start,end=280,360
    base=SOURCE/'data/balltrack/racketvision/match1'/cid
    baseline_path=ROOT.parent/'tracking_quality_20261102/ball_aligned'/cid/'balltrack_predictions.csv'
    rows=read(baseline_path)
    ballroot=SOURCE/'third_party/RacketVision/source/BallTrack';sys.path.insert(0,str(ballroot))
    from inference import BallInferencer
    checkpoint=ballroot/'checkpoints/balltrack_best.pth'
    protocol=dict(candidate_id=cid,start_frame=start,end_frame_inclusive=end,
        model='existing MS-TrackNetV3, unchanged weights',checkpoint_sha256=sha(checkpoint),
        input='cached BGR JPEG frames t-3..t + original median120 background; resize512x288; divide255',
        component_floor=.1,top_k=6,component_position='half-peak weighted centroid, same as prior weak-component experiment',
        selection='No ball chosen at extraction. Candidate rank is per-frame score order, never persistent identity.',
        baseline_sha256=sha(baseline_path),purpose='Two-real-ball counterexample; development only, not new held-out accuracy.')
    (DEST/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    infer=BallInferencer(str(ballroot/'configs/tracknetv3_base.py'),str(checkpoint),device='cuda',thre=.3,batchsize=8)
    median=np.load(base/'runs/threshold030_median120/median.npz')['median']
    median=np.moveaxis(cv2.resize(median,(infer.width,infer.height))/255.,-1,0)
    cache={}
    for f in range(start-3,end+1):
        im=cv2.imread(str(base/'frames'/f'{f:06d}.jpg'));assert im is not None
        cache[f]=np.moveaxis(cv2.resize(im,(infer.width,infer.height)).astype(np.float32)/255.,-1,0)
    candidates=[];checks=[];heatmaps=[]
    for batchstart in range(start,end+1,8):
        frames=list(range(batchstart,min(end+1,batchstart+8)))
        data=np.array([np.concatenate([median[None],np.stack([cache[j] for j in window(t)])],axis=0).reshape(-1,infer.height,infer.width) for t in frames])
        tensor=torch.from_numpy(data).float().to(infer.device)
        with torch.inference_mode(),torch.amp.autocast('cuda'):
            output,_,_=infer.model.forward(frames=tensor)
        for t,hm in zip(frames,output[:,0].float().cpu().numpy()):
            x,y,w,h,conf=infer._predict_location(hm)
            px=int((x+w/2)*1920/infer.width);py=int((y+h/2)*1080/infer.height)
            present=int(px!=0 or py!=0);old=rows[t]
            delta=float(np.hypot(px-float(old['x_px']),py-float(old['y_px'])))
            assert present==int(old['visible']) and delta<=2.,(t,present,old,delta)
            checks.append(dict(local_frame=t,baseline_delta_px=delta,baseline_visible=present))
            for rank,c in enumerate(components(hm,1920,1080),1):
                candidates.append(dict(candidate_id=cid,local_frame=t,rank=rank,**c))
            heatmaps.append(hm.astype(np.float16))
    with (DEST/'candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(candidates[0]));writer.writeheader();writer.writerows(candidates)
    np.savez_compressed(DEST/'heatmaps.npz',heatmaps=np.stack(heatmaps),local_frames=np.arange(start,end+1))
    result=dict(frames=end-start+1,components=len(candidates),multiple_component_frames=len({r['local_frame'] for r in candidates if r['rank']>1}),
        baseline_reproduction_checks=checks,annotation_coordinates_used=False,
        warning='Multiple heatmap components are candidate outputs, not verified balls or target identities.')
    (DEST/'extraction_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='baseline_reproduction_checks'},indent=2))


if __name__=='__main__':main()
