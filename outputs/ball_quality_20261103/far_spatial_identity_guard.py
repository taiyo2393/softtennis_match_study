"""Development candidate veto requiring both appearance and spatial support.

Initial nonplayer rectangles require a fixed camera. Predicted near rectangles
are internal exclusion regions only; no predicted player coordinates are output.
"""
import numpy as np
from far_identity_guard import original,solve,part_distance

PARAMETERS=dict(appearance_negative_margin=.04,
    spatial_overlap_min=original.P['pair_overlap_max'],
    near_prediction_max_gap=original.t.PARAMETERS['recovery_source_frames'],
    near_velocity_cap_fraction_of_last_height=.10,
    upper_patch_x_fraction=[.15,.85],upper_patch_y_fraction=[.18,.52],
    background_rectangles_from_initialization=True,
    predicted_rectangles_exported=False,requires_fixed_camera=True)

def upper_patch(row):
    x1,y1,x2,y2=[float(row[k]) for k in ['x1','y1','x2','y2']];w=x2-x1;h=y2-y1
    return dict(x1=x1+.15*w,x2=x1+.85*w,y1=y1+.18*h,y2=y1+.52*h)

def near_regions(frame,observed,history):
    for row in observed:
        slot=row['player_slot'];seq=history.setdefault(slot,[])
        if not seq or int(seq[-1]['source_frame'])!=frame:seq.append(row);del seq[:-2]
    regions=[]
    for slot,seq in history.items():
        last=seq[-1];age=frame-int(last['source_frame'])
        if not 0<=age<=PARAMETERS['near_prediction_max_gap']:continue
        region=dict(last);kind='observed'
        if age:
            velocity=np.zeros(2)
            if len(seq)>1:
                delta=int(last['source_frame'])-int(seq[-2]['source_frame']);assert delta>0
                velocity=(original.a.foot(last)-original.a.foot(seq[-2]))/delta
                speed=np.linalg.norm(velocity);cap=.10*original.a.height(last)
                if speed>cap:velocity*=cap/speed
            dx,dy=velocity*age
            for k in ['x1','x2','bottom_center_x']:region[k]=float(last[k])+dx
            for k in ['y1','y2','bottom_center_y']:region[k]=float(last[k])+dy
            kind='predicted_exclusion_only'
        regions.append(dict(slot=slot,row=region,age=age,kind=kind))
    return regions

def gate(row,feature,anchors,background,regions,include_near=True):
    own=[min(original.a.app_distance(a,feature) for a in anchors[s]) for s in ['far_rear','far_front']]
    own_upper=[min(part_distance(a,feature,0) for a in anchors[s]) for s in ['far_rear','far_front']]
    patch=upper_patch(row);allowed=[];diagnostic=[]
    for i in range(2):
        bg=[];nr=[]
        for exemplar in background:
            overlap=original.t.overlap_fraction(patch,exemplar['row']);distance=part_distance(exemplar['feature'],feature,0)
            if overlap>PARAMETERS['spatial_overlap_min'] and distance+PARAMETERS['appearance_negative_margin']<own_upper[i]:
                bg.append(dict(negative_id=exemplar['row']['track_id'],overlap=overlap,upper_distance=distance))
        if include_near:
            for region in regions:
                overlap=original.t.overlap_fraction(row,region['row']);distance=min(original.a.app_distance(a,feature) for a in anchors[region['slot']])
                if overlap>PARAMETERS['spatial_overlap_min'] and distance+PARAMETERS['appearance_negative_margin']<own[i]:
                    nr.append(dict(near_slot=region['slot'],overlap=overlap,distance=distance,age=region['age'],kind=region['kind']))
        allowed.append(not(bg or nr));diagnostic.append(dict(own_distance=own[i],own_upper=own_upper[i],background_witnesses=bg,near_witnesses=nr))
    return allowed,diagnostic
