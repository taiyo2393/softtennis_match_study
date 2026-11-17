"""Diagnose only original vs predeclared epoch2/.3; no decoder changes."""
import json,math,sys
from pathlib import Path
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,read,write,load,sha,cv2
from run_manual_ball_finetune import prepare
from run_hard_case_ball_finetune import infer_maps

out=ROOT/'balltrack_expanded_centers_pilot'
assert not (out/'new_center_heatmap_diagnostics.csv').exists()
medians={};samples=prepare(read(out/'expanded_train_labels.csv')[-3:],medians)
modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';sys.path.insert(0,str(modelroot))
from inference import BallInferencer
infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(modelroot/'checkpoints/balltrack_best.pth'),device='cuda',thre=.3,batchsize=4)
results=[]
for family,path in [('original',modelroot/'checkpoints/balltrack_best.pth'),('expanded_epoch2',out/'expanded_epoch02.pth')]:
    infer.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'],strict=True)
    for s,hm in zip(samples,infer_maps(infer,samples,medians),strict=True):
        r=s['row'];tx=float(r['truth_x_px']);ty=float(r['truth_y_px'])
        yy,xx=np.ogrid[:288,:512];region=(xx*1920/512-tx)**2+(yy*1080/288-ty)**2<=16**2
        contours,_=cv2.findContours((hm>.3).astype('uint8')*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        rects=[cv2.boundingRect(c) for c in contours]
        errors=[math.hypot((x+w/2)*1920/512-tx,(y+h/2)*1080/288-ty) for x,y,w,h in rects]
        x,y,w,h,conf=infer._predict_location(hm)
        results.append(dict(sample_id=r['sample_id'],family=family,target_region_peak=float(hm[region].max()),
            target_region_above03_pixels=int((hm[region]>.3).sum()),component_count=len(rects),
            nearest_component_center_error_px=min(errors) if errors else '',component_within16px=int(bool(errors) and min(errors)<=16),
            selected_x_px=int((x+w/2)*1920/512),selected_y_px=int((y+h/2)*1080/288),
            selected_bbox_area_grid=w*h,selected_mean_score=conf,checkpoint_sha256=sha(path)))
write(out/'new_center_heatmap_diagnostics.csv',results)
assert all(sha(Path(p))==h for p,h in load(out/'protocol.json')['protected_input_hashes'].items())
print(json.dumps(results,indent=2))
