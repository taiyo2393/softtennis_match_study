"""Retained solver plus bounded team-conflict cost and optional pose exclusivity.

Zero-cost, exclusivity-disabled replay must reproduce every retained baseline.
This is an offline development solver, not an identity-confidence estimator.
"""
import math,numpy as np
from far_identity_guard import original,P,a,t
PARAMETERS=dict(torso_negative_margin=.04,torso_cost_weight=P['observation_feature_weight'],pose_match_iou_min=.5,pose_stride_source_frames=2,near_and_far_anchor_source_frames=8,hard_color_veto=False,far_team_pooled=True,coordinate_replacement=False)

def solve(cache,event,exclusive_pose=False):
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
                             +(P['detector_id_change_cost'] if int(c['row']['track_id'])!=node['ids'][i] else 0.)
                             +float(c.get('torso_conflict_cost',0.)))
                    choices.append((value,j))
                opts.append(sorted(choices)[:P['options_per_person']])
            for ca,j in opts[0]:
                for cb,k in opts[1]:
                    if j>=0 and k>=0 and (j==k or t.overlap_fraction(items[j]['row'],items[k]['row'])>P['pair_overlap_max']):continue
                    if exclusive_pose and j>=0 and k>=0 and items[j].get('pose_key') is not None and items[j].get('pose_key')==items[k].get('pose_key'):continue
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

