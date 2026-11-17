"""Offline joint two-player beam search with retained identity permutations.

The final path is backtracked after seeing the complete interval. It is a
development proposal, not a verified identity solution. No missing-coordinate
interpolation is exported. Original detections and experiments are unchanged.
"""
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'tracking_quality_20261102_stage4'))
import temporal_tracker as t
a = t.a
P = dict(beam_width=48,options_per_person=4,appearance_max=.70,missing_cost=.90,
         uncertainty_height_base=.25,uncertainty_per_missing_frame=.03,
         max_motion_normalized=6.,velocity_retention_missing=.98,
         observation_feature_weight=.8,motion_weight=.35,detector_id_change_cost=.04,
         pair_overlap_max=.55,stable_tracklet_min_observations=8,
         tracklet_split_gap=4,tracklet_jump_base_height=.40,tracklet_jump_per_frame_height=.30,
         scope='offline_development_not_independent_evaluation')


def write(path,rows):
    if rows:
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w = csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def tracklet_index(raw):
    groups=defaultdict(list)
    for r in raw:
        if int(r['track_id'])>=0:groups[int(r['track_id'])].append(r)
    index={};counts=defaultdict(int)
    for tid,rows in groups.items():
        rows=sorted(rows,key=lambda r:int(r['source_frame']))
        segment=0;previous=None
        for r in rows:
            frame=int(r['source_frame'])
            if previous is not None:
                delta=frame-int(previous['source_frame'])
                distance=np.linalg.norm(a.foot(r)-a.foot(previous))/max(a.height(r),a.height(previous),1)
                ratio=a.height(r)/max(1,a.height(previous))
                if (delta>P['tracklet_split_gap'] or not .5<=ratio<=2.
                    or distance>P['tracklet_jump_base_height']+P['tracklet_jump_per_frame_height']*delta):
                    segment+=1
            key=f'{tid}:{segment}'
            index[(frame,tid)]=key;counts[key]+=1;previous=r
    return index,counts


def candidates(tag):
    old = OUT.parent/'tracking_quality_20261102_stage4'/f'players_{tag}_temporal'
    summary = json.loads((old/'summary.json').read_text(encoding='utf-8'))
    event = json.loads((old/'events.json').read_text(encoding='utf-8'))[0]
    init = event['frame']
    raw = a.read(Path(summary['input_csv']))
    segments,segment_counts=tracklet_index(raw)
    byframe = defaultdict(list)
    for r in raw:
        if int(r['track_id']) >= 0 and float(r['confidence']) >= .25:
            byframe[int(r['source_frame'])].append(r)
    previous = a.read(old/'selected_four.csv')
    near = defaultdict(list)
    for r in previous:
        if r['player_slot'].startswith('near_'):
            near[int(r['source_frame'])].append(r)
    anchors = {}
    cap = cv2.VideoCapture(str(a.SOURCE/'data/raw/match1.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,event['window_start'])
    negatives = []
    for frame in range(event['window_start'],init+1):
        ok,im = cap.read();assert ok
        im = cv2.resize(im,(1920,1080))
        for slot in ('far_rear','far_front'):
            r = next(r for r in byframe[frame] if int(r['track_id']) == event['seed_ids'][slot])
            anchors.setdefault(slot,[]).append(a.appearance(im,r))
        if frame == init:
            chosen = [r for r in byframe[frame] if int(r['track_id']) in event['seed_ids'].values()]
            negatives = [a.appearance(im,r) for r in byframe[frame]
                         if int(r['track_id']) not in event['seed_ids'].values()
                         and all(t.overlap_fraction(r,s) < .1 for s in chosen)]
    cap.set(cv2.CAP_PROP_POS_FRAMES,init)
    cache = []
    for frame in range(init,summary['last_frame']+1):
        ok,im = cap.read();assert ok
        im = cv2.resize(im,(1920,1080))
        items = []
        for r in byframe[frame]:
            x,y = a.foot(r)/[3840,2160]
            if not (.25<x<.75 and .52<y<.665):continue
            # Exclude boxes mostly explained by a tracked near-side person.
            if any(t.overlap_fraction(r,n) > .55 for n in near[frame]):continue
            feat = a.appearance(im,r)
            ds = [min(a.app_distance(f,feat) for f in anchors[s]) for s in ('far_rear','far_front')]
            if min(ds)>P['appearance_max']:continue
            if negatives and min(a.app_distance(f,feat) for f in negatives)+.04<min(ds):continue
            key=segments[(frame,int(r['track_id']))]
            items.append(dict(row=r,appearance=ds,point=a.foot(r),height=a.height(r),
                              tracklet_key=key,stable_tracklet=segment_counts[key]>=P['stable_tracklet_min_observations']))
        # Retain exact identity anchors, even if an overlap veto rejects them.
        if frame == init:
            for i,slot in enumerate(('far_rear','far_front')):
                tid = event['seed_ids'][slot]
                if not any(int(x['row']['track_id'])==tid for x in items):
                    r = next(r for r in byframe[frame] if int(r['track_id'])==tid)
                    key=segments[(frame,tid)]
                    items.append(dict(row=r,appearance=[0.,0.],point=a.foot(r),height=a.height(r),
                                      tracklet_key=key,stable_tracklet=segment_counts[key]>=P['stable_tracklet_min_observations']))
        cache.append(items)
    cap.release()
    return summary,event,near,cache


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


def main():
    import argparse,subprocess
    p=argparse.ArgumentParser(__doc__);p.add_argument('--tag',default='048');args=p.parse_args()
    dest=OUT/f'players_{args.tag}_global';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'summary.json').exists(),'Choose a new experiment directory'
    (dest/'protocol.json').write_text(json.dumps(dict(parameters=P,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    summary,event,near,cache=candidates(args.tag)
    path,search=solve(cache,event)
    assigned=[]
    for offset,choices in enumerate(path):
        for i,index in enumerate(choices):
            if index<0:continue
            row=cache[offset][index]['row']
            assigned.append(dict(row,player_slot=('far_rear','far_front')[i],
                position_source='detected_box',identity_status='offline_path_candidate_requires_review',
                tracklet_key=cache[offset][index]['tracklet_key'],stable_tracklet=int(cache[offset][index]['stable_tracklet'])))
    write(dest/'far_pair_path.csv',assigned)
    byframe=defaultdict(list)
    for r in assigned:byframe[int(r['source_frame'])].append(r)
    cap=cv2.VideoCapture(str(a.SOURCE/'data/raw/match1.mp4'));fps=cap.get(cv2.CAP_PROP_FPS)
    cap.set(1,summary['first_frame'])
    temp=dest/'intermediate.mp4';w=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,1080));assert w.isOpened()
    for f in range(summary['first_frame'],summary['last_frame']+1):
        ok,im=cap.read();assert ok;im=cv2.resize(im,(1920,1080))
        for r in near[f]+byframe[f]:
            i=a.SLOTS.index(r['player_slot']);x1,y1,x2,y2=[round(float(r[k])*.5) for k in ('x1','y1','x2','y2')]
            cv2.rectangle(im,(x1,y1),(x2,y2),a.COLORS[i],2)
            cv2.putText(im,f'{a.NAMES[i]} / raw {r["track_id"]}',(x1,max(22,y1-8)),0,.65,a.COLORS[i],2)
        cv2.rectangle(im,(0,1010),(1920,1080),(0,0,0),-1)
        cv2.putText(im,f'{args.tag} F{f} | joint search + continuous-tracklet identity constraint | candidates',(15,1038),0,.60,(255,255,255),1)
        cv2.putText(im,'Future frames choose the full path. Missing frames remain empty. Same detector boxes.',(15,1070),0,.60,(255,255,255),1)
        w.write(im)
    w.release();cap.release()
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-v','error','-y','-i',str(temp),'-ss',str(summary['first_frame']/fps),'-t',str(summary['frames']/fps),'-i',str(a.SOURCE/'data/raw/match1.mp4'),'-map','0:v:0','-map','1:a:0?','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(dest/'global_review_h264.mp4')],check=True)
    report=dict(tag=args.tag,parameters=P,search=search,far_detection_assignments=len(assigned),
                frames_with_both_far=sum(len(rs)==2 for rs in byframe.values()),frames=summary['frames'],
                warning='New development candidate; no human identity accuracy measured. Near tracks reused from stage4.')
    (dest/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
