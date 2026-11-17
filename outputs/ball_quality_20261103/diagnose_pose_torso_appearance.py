"""Matched-pose rectangle versus torso-mask color comparison, diagnostic only."""
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
from run_body_part_localization_probe import OUT,SLOTS,load,overlap

DEST=OUT/'torso_appearance_diagnostic'
POLICIES=['pose_box_upper_rectangle','pose_torso_polygon']

def feature(im,person,policy):
    bb=np.asarray(person['box']);kp=np.asarray(person['keypoints']);l,t,r,b=np.rint(bb).astype(int);l=max(0,l);t=max(0,t);r=min(im.shape[1],r);b=min(im.shape[0],b)
    if l>=r or t>=b:return None,0
    mask=np.zeros((b-t,r-l),np.uint8)
    if policy=='pose_torso_polygon':
        if np.any(kp[[5,6,11,12],2]<.5):return None,0
        quad=np.rint(kp[[5,6,12,11],:2]-[l,t]).astype(np.int32);cv2.fillConvexPoly(mask,cv2.convexHull(quad),255)
    else:
        w=bb[2]-bb[0];h=bb[3]-bb[1];xy=np.rint([bb[0]+.15*w,bb[1]+.18*h,bb[2]-.15*w,bb[1]+.52*h]).astype(int);cv2.rectangle(mask,(xy[0]-l,xy[1]-t),(xy[2]-l,xy[3]-t),255,-1)
    count=int(np.count_nonzero(mask))
    if count<12:return None,count
    hsv=cv2.cvtColor(im[t:b,l:r],cv2.COLOR_BGR2HSV);hist=cv2.calcHist([hsv],[0,1,2],mask,[12,4,4],[0,180,0,256,0,256]).ravel();hist/=max(float(hist.sum()),1)
    return hist,count

def distance(a,b):return float(cv2.compareHist(a,b,cv2.HISTCMP_BHATTACHARYYA))

def main():
    p=load(OUT/'protocol.json');pred=load(OUT/'pose_predictions.json');matches=read(OUT/'native_box_to_pose_diagnostics.csv');DEST.mkdir();protected={str(n):sha(n) for n in [OUT/'protocol.json',OUT/'pose_predictions.json',OUT/'native_box_to_pose_diagnostics.csv',Path(__file__)]}
    dump(DEST/'protocol.json',dict(purpose='Compare only the color-sampling region on the same matched pose proposals. Not a replay of the original8-frame-anchor tracker.',policies=POLICIES,selected_frames_unchanged=74,initial_gallery='One fixed initialization frame per case; far slots from far_context, near slots from court. Both feature policies use exactly the same pose proposals. Missing anchor or pose stays unavailable.',negative_gallery='Eligible unassigned initialization far_context poses with minimum-area overlap<0.1 against every initialized native player box. Not certified nonplayers.',histogram='HSV12x4x4 normalized; Bhattacharyya distance. Rectangle uses15..85percent width,18..52percent height of the pose box. Torso is convex hull of shoulders and hips, each confidence>=0.5.',min_pixels=12,disagreement_margin=.04,margin_source='Retained appearance margin; no calibration or sweep.',fixed_before_color_comparison_outputs=True,after_same_match_pose_image_inspection=True,canonical_pipeline_replaced=False,position_values_not_created=True,protected_input_hashes=protected))
    matchby={(r['tag'],int(r['source_frame']),r['mode'],r['player_slot']):r for r in matches};frames={(r['tag'],r['source_frame']):r for r in pred['frames']};features={};counts={};cap=cv2.VideoCapture(p['source_video'])
    for item in p['selected_frames']:
        tag,f=item['tag'],item['source_frame'];cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1
        for mode,people in frames[tag,f]['modes'].items():
            for person in people:
                if not person['eligible_not_crop_truncated']:continue
                for policy in POLICIES:
                    k=tag,f,mode,person['pose_index'],policy;features[k],counts[k]=feature(im,person,policy)
    cap.release();galleries={};gallery_report=[]
    for e in p['cases']:
        tag=e['tag'];item=next(r for r in p['selected_frames'] if r['tag']==tag and 'initialization' in r['reasons']);f=item['source_frame'];galleries[tag]={}
        for policy in POLICIES:
            gallery=[]
            for slot in SLOTS:
                mode='far_context' if slot.startswith('far_') else 'court';m=matchby.get((tag,f,mode,slot))
                if m and m['pose_matched']=='1':
                    v=features.get((tag,f,mode,int(m['pose_index']),policy))
                    if v is not None:gallery.append((slot,v))
            for person in frames[tag,f]['modes']['far_context']:
                if not person['eligible_not_crop_truncated'] or any(overlap(person['box'],[float(n[k]) for k in ['x1','y1','x2','y2']])>=.1 for n in item['native_observations']):continue
                v=features.get((tag,f,'far_context',person['pose_index'],policy))
                if v is not None:gallery.append((f'initial_unselected_pose{person["pose_index"]}',v))
            galleries[tag][policy]=gallery;gallery_report.append(dict(tag=tag,policy=policy,initial_frame=f,roles=[role for role,v in gallery]))
    dump(DEST/'initial_gallery_availability.json',dict(galleries=gallery_report,certified_identities=False))
    rows=[]
    for m in matches:
        if not m['player_slot'].startswith('far_'):continue
        tag,f,mode,slot=m['tag'],int(m['source_frame']),m['mode'],m['player_slot']
        for policy in POLICIES:
            k=(tag,f,mode,int(m['pose_index']),policy) if m['pose_matched']=='1' else None;v=features.get(k);gallery=galleries[tag][policy];own=next((a for role,a in gallery if role==slot),None);ready=v is not None and own is not None
            result=dict(tag=tag,source_frame=f,mode=mode,player_slot=slot,native_track_id=m['native_track_id'],pose_index=m['pose_index'],policy=policy,pose_matched=int(m['pose_matched']),feature_pixels=counts.get(k,0),own_anchor_available=int(own is not None),comparison_available=int(ready),own_distance='',best_competing_role='',best_competing_distance='',own_minus_competing_distance='',would_disagree_with_native_slot='',identity_certified=0)
            if ready:
                result['own_distance']=distance(own,v);competing=sorted((distance(a,v),role) for role,a in gallery if role!=slot)
                if competing:
                    d,role=competing[0];result.update(best_competing_role=role,best_competing_distance=d,own_minus_competing_distance=result['own_distance']-d,would_disagree_with_native_slot=int(d+.04<result['own_distance']))
            rows.append(result)
    write(DEST/'appearance_comparison_diagnostic.csv',rows);write(DEST/'known_examples.csv',[r for r in rows if (r['tag'],r['source_frame'],r['player_slot']) in {('006',12421,'far_rear'),('006',12422,'far_rear'),('016',40625,'far_rear'),('016',40626,'far_rear'),('019',46501,'far_rear'),('019',46511,'far_rear'),('019',46634,'far_rear'),('019',46665,'far_front'),('019',46666,'far_front')}])
    assert all(sha(Path(n))==h for n,h in protected.items());dump(DEST/'result.json',dict(status='completed_diagnostic_only',comparison_rows=len(rows),available_rows=sum(r['comparison_available'] for r in rows),mask_disagreements_not_errors=sum(r['would_disagree_with_native_slot']==1 for r in rows),protected_inputs_unchanged=True,tracker_modified=False,accuracy_established=False));print(load(DEST/'result.json'))

if __name__=='__main__':main()
