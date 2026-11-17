"""Infer weak heatmap components only where aligned baseline has no output.

No labels are read. Reuse cached JPEGs and exactly the existing median/model.
Correctly aligned input ends at output frame. No interpolation or smoothing.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'tracking_quality_20261102'))
from ball_alignment import SOURCE,read,window

P=dict(heatmap_floor=.10,components_per_frame=6,batch_size=16,
       component_position='half-peak weighted centroid',context='t-3,t-2,t-1,t',
       cache_resized_frames=96)


def components(hm,width,height):
    count,labels,stats,_=cv2.connectedComponentsWithStats((hm>P['heatmap_floor']).astype(np.uint8),8)
    result=[]
    for k in range(1,count):
        x,y,w,h,area=stats[k]
        patch=hm[y:y+h,x:x+w]
        mask=labels[y:y+h,x:x+w]==k
        peak=float(patch[mask].max())
        weights=np.where(mask & (patch>=max(P['heatmap_floor'],.5*peak)),patch,0.)
        total=float(weights.sum())
        if total<=0:continue
        yy,xx=np.indices(patch.shape)
        cx=x+float((weights*xx).sum())/total
        cy=y+float((weights*yy).sum())/total
        result.append(dict(x_px=cx*width/hm.shape[1],y_px=cy*height/hm.shape[0],
                           peak=peak,area=int(area),mean=float(patch[mask].mean())))
    return sorted(result,key=lambda r:r['peak'],reverse=True)[:P['components_per_frame']]


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['validation','test'],required=True);args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    ballroot=SOURCE/'third_party/RacketVision/source/BallTrack'
    sys.path.insert(0,str(ballroot));from inference import BallInferencer
    checkpoint=ballroot/'checkpoints/balltrack_best.pth'
    infer=BallInferencer(str(ballroot/'configs/tracknetv3_base.py'),str(checkpoint),device='cuda',thre=.30,batchsize=P['batch_size'])
    plan=dict(parameters=P,phase=args.phase,model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              note='No label coordinates used for candidate inference; baseline-present frames are unchanged.')
    (OUT/f'inference_protocol_{args.phase}.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    cids=range(1,11) if args.phase=='validation' else range(11,21)
    for number in cids:
        cid=f'match1_motion_pilot_{number:03d}'
        dest=OUT/'weak_candidates'/cid;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'metadata.json').exists():
            print(cid,'cached',flush=True);continue
        base=SOURCE/'data/balltrack/racketvision/match1'/cid
        oldpath=OUT.parent/'tracking_quality_20261102/ball_aligned'/cid/'balltrack_predictions.csv'
        rows=read(oldpath)
        metadata=json.loads((base/'runs/threshold030_median120/run_metadata.json').read_text(encoding='utf-8'))
        width,height=metadata['width'],metadata['height']
        target=[i for i,r in enumerate(rows) if not int(r['visible'])]
        present=[i for i,r in enumerate(rows) if int(r['visible'])]
        sentinels=[present[j] for j in (0,len(present)//2,len(present)-1)] if present else []
        requested=sorted(set(target+sentinels))
        median=np.load(base/'runs/threshold030_median120/median.npz')['median']
        median=np.moveaxis(cv2.resize(median,(infer.width,infer.height))/255.,-1,0)
        cache=OrderedDict();output=[];checks=[]

        def frame(i):
            if i not in cache:
                im=cv2.imread(str(base/'frames'/f'{i:06d}.jpg'))
                if im is None:raise RuntimeError(f'Missing frame {cid}:{i}')
                cache[i]=np.moveaxis(cv2.resize(im,(infer.width,infer.height)).astype(np.float32)/255.,-1,0)
                if len(cache)>P['cache_resized_frames']:cache.popitem(last=False)
            return cache[i]

        for start in range(0,len(requested),P['batch_size']):
            ids=requested[start:start+P['batch_size']]
            data=np.array([np.concatenate([median[None],np.stack([frame(j) for j in window(i)])],axis=0)
                           .reshape(-1,infer.height,infer.width) for i in ids])
            tensor=torch.from_numpy(data).float().to(infer.device)
            with torch.inference_mode():
                if infer.device.type=='cuda':
                    with torch.amp.autocast('cuda'):maps,_,_=infer.model.forward(frames=tensor)
                else:maps,_,_=infer.model.forward(frames=tensor)
            for i,hm in zip(ids,maps.float().cpu().numpy()[:,0]):
                if i in sentinels:
                    x,y,w,h,confidence=infer._predict_location(hm)
                    cx=int((x+w/2)*width/infer.width);cy=int((y+h/2)*height/infer.height)
                    delta=float(np.hypot(cx-float(rows[i]['x_px']),cy-float(rows[i]['y_px'])))
                    assert delta<=2.,(cid,i,delta)
                    checks.append(dict(local_frame=i,baseline_coordinate_delta_px=delta))
                if not int(rows[i]['visible']):
                    for rank,c in enumerate(components(hm,width,height),1):
                        output.append(dict(candidate_id=cid,local_frame=i,source_frame=rows[i]['source_frame'],rank=rank,**c))
        fields=['candidate_id','local_frame','source_frame','rank','x_px','y_px','peak','area','mean']
        with (dest/'candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(output)
        report=dict(candidate_id=cid,frames=len(rows),baseline_missing_frames=len(target),
            weak_components=len(output),baseline_checks=checks,parameters=P,
            baseline_csv_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),fps=metadata['fps'],
            width=width,height=height,note='Weak components are not verified ball detections or calibrated probabilities.')
        (dest/'metadata.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(cid,'missing',len(target),'weak components',len(output),flush=True)


if __name__=='__main__':main()
