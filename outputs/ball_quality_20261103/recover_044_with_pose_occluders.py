"""Keep recovery gates fixed; approximate occluders with current-frame poses."""
import json,os
from pathlib import Path
import cv2
import numpy as np
import propose_044_occlusion_recovery as recovery
from diagnose_044_person_gaps import setup,GAPS
from track_044_far_pair import ROOT,FIRST,NEAR,a,image_at
from extract_serve_start_features import SOURCE,sha

OUT=ROOT/'person_044_pose_recovery'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'serve_start_pilot/ultralytics_config')
PRIMITIVES={}

def iou(a,b):
    x1,y1,x2,y2=a;u1,v1,u2,v2=b
    inter=max(0,min(x2,u2)-max(x1,u1))*max(0,min(y2,v2)-max(y1,v1))
    return inter/max(1,(x2-x1)*(y2-y1)+(u2-u1)*(v2-v1)-inter)

def build_masks():
    OUT.mkdir(exist_ok=True)
    path=OUT/'near_pose_primitives.json'
    if path.exists():
        data=json.loads(path.read_text());return {int(k):v for k,v in data['frames'].items()}
    from ultralytics import YOLO
    model=SOURCE/'models/yolo11m-pose.pt';net=YOLO(str(model))
    _,_,by,_,_=setup();frames={};matched=0;fallback=0
    times=sorted({t for _,first,last in GAPS for t in range(first,last+1)})
    for t in times:
        f=FIRST+t;im=image_at(f)
        pred=net.predict(im[320:],imgsz=1920,conf=.25,iou=.7,device=0,verbose=False)[0]
        poses=[]
        for box,kp in zip(pred.boxes.data.cpu().numpy(),pred.keypoints.data.cpu().numpy()):
            box=box[:4].copy();kp=kp.copy();box[[1,3]]+=320;kp[:,1]+=320
            poses.append((box,kp))
        items=[];used=set()
        for near in [r for r in by[f] if int(r['track_id']) in NEAR.values()]:
            box=np.array([float(near[k])*.5 for k in ['x1','y1','x2','y2']])
            choices=sorted([(iou(box,b),i,b,k) for i,(b,k) in enumerate(poses) if i not in used],key=lambda q:q[0],reverse=True)
            valid=bool(choices and choices[0][0]>=.5 and np.all(choices[0][3][[0,5,6,11,12],2]>=.5))
            if valid:
                _,j,b,k=choices[0];used.add(j);matched+=1
                items.append(dict(raw_near_id=near['track_id'],mode='current_frame_pose_body_approximation',
                    raw_box=box.tolist(),pose_box=b.tolist(),keypoints=k.tolist()))
            else:
                fallback+=1;items.append(dict(raw_near_id=near['track_id'],mode='conservative_box_fallback',raw_box=box.tolist()))
        frames[t]=items
    path.write_text(json.dumps(dict(frames=frames,model_sha256=sha(model),imgsz=1920,confidence=.25,
        pose_iou_match_min=.5,required_keypoint_confidence=.5,pose_matches=matched,box_fallbacks=fallback,
        independent_occlusion_groundtruth=False),indent=2),encoding='utf-8')
    return frames

def mask_for(row):
    mask=np.zeros((1080,1920),np.uint8)
    for p in PRIMITIVES[int(row['source_frame'])-FIRST]:
        box=np.asarray(p['raw_box']);h=box[3]-box[1]
        if p['mode']=='conservative_box_fallback':
            l,t,r,b=np.rint(box).astype(int);cv2.rectangle(mask,(l,t),(r,b),255,-1);continue
        k=np.asarray(p['keypoints']);quad=k[[5,6,12,11],:2];center=quad.mean(axis=0)
        quad=np.rint(center+1.12*(quad-center)).astype(np.int32);cv2.fillConvexPoly(mask,quad,255)
        cv2.circle(mask,tuple(np.rint(k[0,:2]).astype(int)),max(2,round(.09*h)),255,-1)
        for j,l,radius in [(5,7,.045),(7,9,.045),(6,8,.045),(8,10,.045),
                           (11,13,.065),(13,15,.065),(12,14,.065),(14,16,.065)]:
            if min(k[j,2],k[l,2])<.5:continue
            p1=tuple(np.rint(k[j,:2]).astype(int));p2=tuple(np.rint(k[l,:2]).astype(int));rad=max(2,round(radius*h))
            cv2.line(mask,p1,p2,255,2*rad);cv2.circle(mask,p1,rad,255,-1);cv2.circle(mask,p2,rad,255,-1)
    return mask

def visible_feature(im,row,occluders):
    occlusion=mask_for(row)
    x1,y1,x2,y2=[float(row[k])*.5 for k in ['x1','y1','x2','y2']];w=x2-x1;h=y2-y1
    parts=[];fractions=[];counts=[]
    for lo,hi in [(.18,.52),(.52,.78)]:
        l=max(0,round(x1+.15*w));r=min(im.shape[1],round(x2-.15*w));t=max(0,round(y1+lo*h));b=min(im.shape[0],round(y1+hi*h))
        patch=im[t:b,l:r]
        if not patch.size:return None,[],[]
        mask=(occlusion[t:b,l:r]==0).astype(np.uint8)*255
        count=int(np.count_nonzero(mask));counts.append(count);fractions.append(count/max(1,mask.size))
        if count<recovery.P['min_part_pixels']:parts.append(None);continue
        hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
        hist=cv2.calcHist([hsv],[0,1,2],mask,[12,4,4],[0,180,0,256,0,256]).ravel()
        parts.append(hist/max(float(hist.sum()),1.))
    return parts,fractions,counts

def main():
    global PRIMITIVES
    PRIMITIVES=build_masks();recovery.OUT=OUT;recovery.visible_feature=visible_feature
    recovery.main()
    protocol=json.loads((OUT/'protocol.json').read_text())
    protocol.update(mask_method='Pose torso polygon, head disc, limb capsules; bounding rectangle fallback if pose unreliable',
        wrapper_sha256=sha(Path(__file__)),pose_primitives_sha256=sha(OUT/'near_pose_primitives.json'),
        mask_parameters=dict(torso_polygon_scale=1.12,head_radius_height=.09,arm_radius_height=.045,leg_radius_height=.065),
        gating_parameters_unchanged_from_box_mask_prototype=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')

if __name__=='__main__':main()
