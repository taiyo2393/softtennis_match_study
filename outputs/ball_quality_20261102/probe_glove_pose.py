"""Feasibility probe of a wrist-associated risk flag, not a new accuracy claim."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

SOURCE=Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study')
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'glove_guard'
os.environ['YOLO_CONFIG_DIR']=str(OUT/'ultralytics_config')


def main():
    from ultralytics import YOLO
    checkpoint=SOURCE/'models/yolo11m-pose.pt';net=YOLO(str(checkpoint))
    targets=[('006',899,1233.,513.),('001',16,960.,567.)]
    reports=[]
    for cid,t,bx,by in targets:
        name=f'match1_motion_pilot_{cid}'
        im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1'/name/'frames'/f'{t:06d}.jpg'))
        assert im is not None
        left=max(0,min(im.shape[1]-384,round(bx)-192));top=max(0,min(im.shape[0]-384,round(by)-128))
        result=net.predict(im[top:top+384,left:left+384],imgsz=960,conf=.15,iou=.7,device=0,verbose=False)[0]
        people=[]
        for box,keypoint in zip(result.boxes.data.cpu().numpy(),result.keypoints.data.cpu().numpy()):
            x1,y1,x2,y2=box[:4]+np.array([left,top,left,top]);h=y2-y1
            xa,xb=max(0,int(x1)),min(im.shape[1],int(x2))
            ya,yb=max(0,int(y1+.2*h)),min(im.shape[0],int(y1+.6*h))
            patch=im[ya:yb,xa:xb]
            hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV) if patch.size else np.zeros((1,1,3),np.uint8)
            # Blue/cyan clothing is a match-specific context cue, not a universal official classifier.
            blue=float(((hsv[:,:,0]>=80)&(hsv[:,:,0]<=115)&(hsv[:,:,1]>=65)&(hsv[:,:,2]>=55)).mean())
            wrists=[]
            for k in (9,10):
                x,y,conf=keypoint[k];x+=left;y+=top
                wrists.append(dict(x=float(x),y=float(y),confidence=float(conf),candidate_distance_by_person_height=float(np.hypot(x-bx,y-by)/h)))
            people.append(dict(box=[float(v) for v in (x1,y1,x2,y2)],confidence=float(box[4]),blue_torso_fraction=blue,wrists=wrists))
        reports.append(dict(candidate_id=name,local_frame=t,candidate_xy=[bx,by],people=people))
        cv2.imwrite(str(OUT/f'pose_probe_roi_{cid}_{t}.jpg'),result.plot())
    report=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                inference=dict(imgsz=960,roi_size=384,person_conf=.15,iou=.7),samples=reports)
    (OUT/'pose_probe_roi.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps([dict(candidate_id=r['candidate_id'],people=[dict(blue=p['blue_torso_fraction'],wrists=p['wrists']) for p in r['people']]) for r in reports],indent=2))


if __name__=='__main__':main()
