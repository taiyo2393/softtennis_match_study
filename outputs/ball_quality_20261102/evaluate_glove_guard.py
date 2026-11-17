"""Match-specific blue-torso raised-wrist guard. Preserve raw predictions."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'glove_guard'
os.environ['YOLO_CONFIG_DIR']=str(OUT/'ultralytics_config')
sys.path.insert(0,str(ROOT.parent/'tracking_quality_20261102'))
from ball_alignment import SOURCE,read,evaluate
LABEL=SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv'
BASE=ROOT.parent/'tracking_quality_20261102/ball_aligned'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def init():
    assert not (OUT/'guard_protocol.json').exists()
    save('guard_protocol.json',dict(
        version=1,hypothesis='Suppress a ball candidate attached to a raised wrist of a blue-torso person, to reject line-judge gloves in this match.',
        status='match-specific experimental guard, not universal semantic official recognition',
        context='Features use frame and baseline candidate only, no truth coordinates or phase seeds.',
        pose_model='yolo11m-pose.pt',pose_sha256=sha(SOURCE/'models/yolo11m-pose.pt'),roi=[384,384],
        roi_anchor='candidate x-192,y-128, clamped to source image',imgsz=960,person_conf=.15,iou=.7,
        torso='polygon shoulders5,6 and hips12,11; every keypoint confidence >=0.4',
        blue_hsv=dict(h_min=80,h_max=115,s_min=65,v_min=55),min_blue_fraction=.5,
        wrist_min_conf=.6,shoulder_min_conf=.5,raised_wrist='wrist_y <= corresponding_shoulder_y+0.05*person_height',
        distance_thresholds=[0.,.08,.12,.16],
        selection='Most nonlocalized validation outputs suppressed with zero lost localized validation TP; ties fewer total suppressions then smaller distance.',
        test='Freeze validation selection before test inference; no retuning on already seen same-match test.',
        label_sha256=sha(LABEL),warning='Legacy label provenance unverified, and visible balls exist in some not_visible labels. No independent accuracy claim.'))
    print('guard protocol recorded',flush=True)


def features(net,im,bx,by):
    height,width=im.shape[:2];left=max(0,min(width-384,round(bx)-192));top=max(0,min(height-384,round(by)-128))
    result=net.predict(im[top:top+384,left:left+384],imgsz=960,conf=.15,iou=.7,device=0,verbose=False)[0]
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    candidates=[]
    for box,kp in zip(result.boxes.data.cpu().numpy(),result.keypoints.data.cpu().numpy()):
        x1,y1,x2,y2=box[:4]+np.array([left,top,left,top]);h=float(y2-y1)
        kp=kp.copy();kp[:,0]+=left;kp[:,1]+=top
        if h<=0 or min(float(kp[k,2]) for k in (5,6,11,12))<.4:continue
        xa,xb=max(0,math.floor(x1)),min(width,math.ceil(x2));ya,yb=max(0,math.floor(y1)),min(height,math.ceil(y2))
        mask=np.zeros((yb-ya,xb-xa),np.uint8)
        poly=np.array([[kp[k,0]-xa,kp[k,1]-ya] for k in (5,6,12,11)],dtype=np.int32)
        cv2.fillConvexPoly(mask,cv2.convexHull(poly),1)
        colors=hsv[ya:yb,xa:xb];inside=mask.astype(bool)
        if inside.sum()<4:continue
        blue=float((((colors[:,:,0]>=80)&(colors[:,:,0]<=115)&(colors[:,:,1]>=65)&(colors[:,:,2]>=55))&inside).sum()/inside.sum())
        for wrist,shoulder in ((9,5),(10,6)):
            wx,wy,wc=kp[wrist];sx,sy,sc=kp[shoulder]
            candidates.append(dict(distance_by_person_height=float(np.hypot(wx-bx,wy-by)/h),blue_torso_fraction=blue,
                wrist_conf=float(wc),shoulder_conf=float(sc),raised_wrist=bool(wy<=sy+.05*h),
                wrist_x=float(wx),wrist_y=float(wy),person_height=h))
    eligible=[c for c in candidates if c['blue_torso_fraction']>=.5 and c['wrist_conf']>=.6 and c['shoulder_conf']>=.5 and c['raised_wrist']]
    nearest=min(eligible,key=lambda c:c['distance_by_person_height']) if eligible else None
    return dict(nearest_eligible_wrist=nearest,people_detected=len(result.boxes),wrist_candidates=candidates)


def infer(phase):
    assert not (OUT/f'{phase}_guard_features.json').exists()
    if phase=='test':assert (OUT/'guard_selection.json').exists()
    from ultralytics import YOLO
    net=YOLO(str(SOURCE/'models/yolo11m-pose.pt'))
    labels=[r for r in read(LABEL) if r['evaluation_split']==phase]
    baselines={cid:read(BASE/cid/'balltrack_predictions.csv') for cid in sorted({r['candidate_id'] for r in labels})}
    output=[];started=time.perf_counter()
    for i,r in enumerate(labels):
        t=int(r['local_frame']);cid=r['candidate_id'];p=baselines[cid][t]
        if not int(p['visible']):continue
        im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{t:06d}.jpg'));assert im is not None
        output.append(dict(candidate_id=cid,local_frame=t,**features(net,im,float(p['x_px']),float(p['y_px']))))
        if len(output)%20==0:print(phase,'candidate features',len(output),flush=True)
    save(f'{phase}_guard_features.json',dict(frames=output,seconds=time.perf_counter()-started,
        protocol_sha256=sha(OUT/'guard_protocol.json'),script_sha256=sha(Path(__file__))))
    print(phase,len(output),'visible baseline candidates processed',flush=True)


def apply(labels,threshold,phase):
    data=json.loads((OUT/f'{phase}_guard_features.json').read_text())
    feats={(r['candidate_id'],r['local_frame']):r for r in data['frames']}
    pred={};removed=[]
    for cid in sorted({r['candidate_id'] for r in labels}):
        pred[cid]={int(r['local_frame']):dict(r,raw_x_px=r['x_px'],raw_y_px=r['y_px'],raw_visible=r['visible'],glove_guard_flag=0) for r in read(BASE/cid/'balltrack_predictions.csv')}
    for r in labels:
        cid=r['candidate_id'];t=int(r['local_frame']);p=pred[cid][t]
        if not int(p['visible']):continue
        nearest=feats[cid,t]['nearest_eligible_wrist']
        if threshold>0 and nearest and nearest['distance_by_person_height']<=threshold:
            p.update(visible=0,x_px=0,y_px=0,glove_guard_flag=1,position_source='withheld_blue_raised_wrist_candidate')
            removed.append(dict(candidate_id=cid,local_frame=t,raw_xy=[float(p['raw_x_px']),float(p['raw_y_px'])],**nearest))
    return pred,removed


def score(phase):
    protocol=json.loads((OUT/'guard_protocol.json').read_text());assert sha(LABEL)==protocol['label_sha256']
    labels=[r for r in read(LABEL) if r['evaluation_split']==phase]
    base,_=apply(labels,0.,phase);bm=evaluate(labels,base,phase)
    if phase=='validation':
        assert not (OUT/'guard_selection.json').exists() and not (OUT/'test_guard_features.json').exists()
        reports=[]
        for threshold in protocol['distance_thresholds']:
            pred,removed=apply(labels,threshold,phase);m=evaluate(labels,pred,phase)
            tp=m['counts'].get('localized_tp_16px',0);bad=m['counts']['pred_visible']-tp
            reports.append(dict(threshold=threshold,metrics=m,suppressed=removed,
                lost_localized_tp=bm['counts']['localized_tp_16px']-tp,
                removed_nonlocalized=(bm['counts']['pred_visible']-bm['counts']['localized_tp_16px'])-bad))
        best=max([r for r in reports if r['lost_localized_tp']==0],key=lambda r:(r['removed_nonlocalized'],-len(r['suppressed']),-r['threshold']))
        save('guard_selection.json',dict(selected_threshold=best['threshold'],baseline=bm,selected=best,validation_grid=reports,protocol_sha256=sha(OUT/'guard_protocol.json')))
        print(json.dumps(dict(threshold=best['threshold'],removed_nonlocalized=best['removed_nonlocalized'],lost_tp=best['lost_localized_tp'],suppressed=best['suppressed']),indent=2))
    else:
        choice=json.loads((OUT/'guard_selection.json').read_text());assert sha(OUT/'guard_protocol.json')==choice['protocol_sha256']
        pred,removed=apply(labels,choice['selected_threshold'],phase);m=evaluate(labels,pred,phase)
        save('guard_test_results.json',dict(baseline=bm,guard=m,suppressed=removed,threshold=choice['selected_threshold'],selection_sha256=sha(OUT/'guard_selection.json'),
            warning='Reused same-match development test, label conflicts unresolved. No external accuracy claim or adopted-output replacement.'))
        print(json.dumps(dict(baseline=bm,guard=m,suppressed=removed),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step',choices=['init','infer-validation','score-validation','infer-test','score-test']);a=p.parse_args()
    if a.step=='init':init()
    elif a.step.startswith('infer-'):infer(a.step.removeprefix('infer-'))
    else:score(a.step.removeprefix('score-'))
