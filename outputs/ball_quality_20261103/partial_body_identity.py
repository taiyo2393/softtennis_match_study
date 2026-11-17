"""Offline body-only identity hypotheses. Never exports a foot/ground position."""
import cv2
import math
import numpy as np
from confirmed_continuity_guard import old
from diagnose_pose_torso_appearance import distance

PARAMETERS=dict(keypoint_confidence=.5,min_feature_pixels=12,pose_match_iou=.5,
    appearance_max=.65,negative_margin=.06,missing_cost=1.05,
    motion_weight=.62,appearance_weight=.75,scale_weight=.15,
    max_gap_source_frames=90,source='Existing tracker/torso feature constants reused; no sweep.',
    feature_fallback=False,foot_position_export=False)


def body_feature(image,person,near_boxes=()):
    """Pose torso polygon minus near rectangles; model confidence != visibility."""
    bb=np.asarray(person['box']);kp=np.asarray(person['keypoints'])
    if not person['eligible_not_crop_truncated'] or np.any(kp[[5,6,11,12],2]<.5):return None,0,0
    l,t,r,b=np.rint(bb).astype(int);l=max(0,l);t=max(0,t);r=min(image.shape[1],r);b=min(image.shape[0],b)
    if l>=r or t>=b:return None,0,0
    mask=np.zeros((b-t,r-l),np.uint8)
    polygon=np.rint(kp[[5,6,12,11],:2]-[l,t]).astype(np.int32)
    cv2.fillConvexPoly(mask,cv2.convexHull(polygon),255);original=int(np.count_nonzero(mask))
    for rect in near_boxes:
        a,c,d,e=np.rint(rect).astype(int)
        if min(r,d)>max(l,a) and min(b,e)>max(t,c):
            mask[max(0,c-t):min(b,e)-t,max(0,a-l):min(r,d)-l]=0
    count=int(np.count_nonzero(mask))
    if count<12:return None,count,original
    hsv=cv2.cvtColor(image[t:b,l:r],cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1,2],mask,[12,4,4],[0,180,0,256,0,256]).ravel()
    hist/=max(float(hist.sum()),1)
    return hist,count,original


def body_point(person):
    # Internal motion cue only, never interpreted as court/foot coordinates.
    return np.asarray(person['keypoints'])[[5,6,11,12],:2].mean(axis=0)


class BodyIdentity:
    def __init__(self,anchors,negatives,initial_people,frame):
        self.anchors=anchors;self.negatives=negatives;self.states={}
        for slot,person in initial_people.items():
            if anchors.get(slot):self.states[slot]=dict(person=person,point=body_point(person),
                height=float(person['box'][3]-person['box'][1]),frame=frame,velocity=np.zeros(2))

    def step(self,frame,people,features,near_pose_indices):
        slots=['far_rear','far_front'];costs=np.full((2,len(people)),1e6)
        for i,slot in enumerate(slots):
            if slot not in self.states:continue
            st=self.states[slot];gap=abs(frame-st['frame'])
            if not 0<gap<=90:continue
            direction=1 if frame>st['frame'] else -1
            for j,(person,feature) in enumerate(zip(people,features)):
                if feature is None or person['pose_index'] in near_pose_indices:continue
                height=person['box'][3]-person['box'][1]
                if not .5<=height/st['height']<=2.4:continue
                app=min(distance(a,feature) for a in self.anchors[slot])
                if app>.65:continue
                if self.negatives and min(distance(n,feature) for n in self.negatives)+.06<app:continue
                predicted=st['point']+direction*st['velocity']*min(gap,12)
                residual=float(np.linalg.norm(body_point(person)-predicted))/max(height,st['height'])
                if residual>min(2.,.55+.025*gap):continue
                costs[i,j]=.75*app+.62*residual+.15*abs(math.log(height/st['height']))
        assigned=old.a.joint_assignment(costs,missing_cost=1.05);result={}
        if len(assigned)==2:
            a,b=[people[j]['box'] for _,j,_,_ in assigned]
            from run_body_part_localization_probe import overlap
            if overlap(a,b)>.55:return result
        for i,j,margin,value in assigned:
            slot=slots[i];person=people[j];st=self.states[slot];delta=frame-st['frame']
            velocity=(body_point(person)-st['point'])/delta
            speed=float(np.linalg.norm(velocity));limit=.10*st['height']
            if speed>limit:velocity*=limit/speed
            st.update(person=person,point=body_point(person),frame=frame,velocity=.75*st['velocity']+.25*velocity)
            result[slot]=dict(person=person,cost=value,margin=margin)
        return result
