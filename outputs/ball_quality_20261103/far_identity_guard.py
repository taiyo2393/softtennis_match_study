"""Retained beam solver with an explicit per-identity candidate gate.

Beam mechanics copied from the retained solver; only identity_allowed veto is new.
No frame numbers, raw person IDs or outcome labels enter the policy.
"""
import sys,math
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'tracking_quality_20261102_stage10'))
import gap_pair_tracking as original
P=original.P
a=original.a
t=original.t
PARAMETERS=dict(identity_appearance_max=P['appearance_max'],negative_margin=.04,
    upper_body_negative_margin=.04,interpolation=False,
    near_templates_from_initial_confirmation=True,background_templates_from_initialization=True)

def part_distance(x,y,part):
    if x is None or y is None:return 1.
    return float(cv2.compareHist(x[part],y[part],cv2.HISTCMP_BHATTACHARYYA))

def gate(feature,anchors,background,upper=False):
    slots=['far_rear','far_front']
    own=[min(a.app_distance(f,feature) for f in anchors[s]) for s in slots]
    near=anchors['near_front']+anchors['near_rear']
    near_score=min(a.app_distance(f,feature) for f in near)
    own_upper=[min(part_distance(f,feature,0) for f in anchors[s]) for s in slots]
    competitors=background+near
    negative_upper=min((part_distance(f,feature,0) for f in competitors),default=1.)
    allowed=[];reasons=[]
    for i in range(2):
        rejected=[]
        if own[i]>PARAMETERS['identity_appearance_max']:rejected.append('slot_appearance_limit')
        if near_score+PARAMETERS['negative_margin']<own[i]:rejected.append('closer_to_initial_near_player')
        if upper and negative_upper+PARAMETERS['upper_body_negative_margin']<own_upper[i]:rejected.append('upper_clothing_closer_to_nonplayer_or_near_player')
        allowed.append(not rejected);reasons.append('|'.join(rejected))
    return allowed,dict(own=own,near=near_score,own_upper=own_upper,negative_upper=negative_upper,reasons=reasons)

def solve(cache,event):
    last_use={c['tracklet_key']:f for f,items in enumerate(cache) for c in items}
    ids = [event['seed_ids'][s] for s in ('far_rear','far_front')]
    chosen = [next(i for i,c in enumerate(cache[0]) if int(c['row']['track_id'])==tid) for tid in ids]
    points = np.array([cache[0][j]['point'] for j in chosen])
    heights = np.array([cache[0][j]['height'] for j in chosen])
    first = dict(cost=0.,pos=points,vel=np.zeros((2,2)),last=points.copy(),gap=[0,0],ids=ids,
                 parent=-1,chosen=chosen,owners={cache[0][j]['tracklet_key']:i for i,j in enumerate(chosen)})
    layers = [[first]]
    for frame,items in enumerate(cache[1:],1):
        proposals = []
        for parent,node in enumerate(layers[-1]):
            opts = []
            predicted = node['pos']+node['vel']
            for i in range(2):
                choices = [(P['missing_cost'],-1)]
                scale = heights[i]*(P['uncertainty_height_base']+P['uncertainty_per_missing_frame']*node['gap'][i])
                for j,c in enumerate(items):
                    if not c.get('identity_allowed',[True,True])[i]:continue
                    if c['stable_tracklet'] and node['owners'].get(c['tracklet_key'],i)!=i:continue
                    if not .5<=c['height']/heights[i]<=2.4:continue
                    dist = float(np.linalg.norm(c['point']-predicted[i]))/scale
                    if dist>P['max_motion_normalized']:continue
                    value = (.25+P['observation_feature_weight']*c['appearance'][i]
                             -.15*float(c['row']['confidence'])
                             +P['motion_weight']*min(dist,3.)+.12*abs(math.log(c['height']/heights[i]))
                             +(P['detector_id_change_cost'] if int(c['row']['track_id'])!=node['ids'][i] else 0.))
                    choices.append((value,j))
                opts.append(sorted(choices)[:P['options_per_person']])
            for ca,j in opts[0]:
                for cb,k in opts[1]:
                    if j>=0 and k>=0 and (j==k or t.overlap_fraction(items[j]['row'],items[k]['row'])>P['pair_overlap_max']):continue
                    pos=predicted.copy();vel=node['vel'].copy();last=node['last'].copy()
                    owners=dict(node['owners'])
                    gaps=list(node['gap']);newids=list(node['ids'])
                    for i,index in enumerate((j,k)):
                        if index<0:
                            gaps[i]+=1;vel[i]*=P['velocity_retention_missing']
                        else:
                            c=items[index];delta=(c['point']-last[i])/(gaps[i]+1)
                            if c['stable_tracklet']:owners[c['tracklet_key']]=i
                            vel[i]=.8*vel[i]+.2*delta
                            speed=np.linalg.norm(vel[i]);limit=.10*heights[i]
                            if speed>limit:vel[i]*=limit/speed
                            pos[i]=c['point'];last[i]=c['point'];gaps[i]=0;newids[i]=int(c['row']['track_id'])
                    owners={key:owner for key,owner in owners.items() if last_use[key]>frame}
                    proposals.append(dict(cost=node['cost']+ca+cb,pos=pos,vel=vel,last=last,gap=gaps,ids=newids,parent=parent,chosen=[j,k],owners=owners))
        proposals.sort(key=lambda n:n['cost'])
        retained=[];seen=set()
        for n in proposals:
            key=(tuple(n['chosen']),tuple(np.round(n['pos'].ravel()/32).astype(int)),
                 tuple(np.round(n['vel'].ravel()/2).astype(int)),tuple(min(g,200) for g in n['gap']),tuple(sorted(n['owners'].items())))
            if key in seen:continue
            retained.append(n);seen.add(key)
            if len(retained)>=P['beam_width']:break
        assert retained
        layers.append(retained)
        if frame%240==0:print('global search',frame,'/',len(cache),flush=True)
    # Offline backtracking uses later evidence to choose earlier assignments.
    path=[];index=0
    for layer in reversed(layers):
        node=layer[index];path.append(node['chosen']);index=node['parent']
    return path[::-1],dict(best_path_cost=layers[-1][0]['cost'],surviving_final_hypotheses=len(layers[-1]),
                          caution='Beam search is approximate; path score is not calibrated identity confidence.')

