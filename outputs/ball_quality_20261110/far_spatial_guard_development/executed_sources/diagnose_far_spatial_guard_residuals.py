"""Explain rejected/retained native boxes without tuning the frozen policy."""
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import BASE,read,write,sha,dump
from run_far_identity_guard_trial import load
from diagnose_initial_pair_appearance import cache as read_cache
import far_spatial_identity_guard as g

OUT=BASE/'far_spatial_guard_development'

def main():
    p=load(OUT/'protocol.json');e=next(e for e in p['cases'] if e['tag']=='019')
    ev=next(x for x in load(Path(e['temporal_folder'])/'events.json') if x['event']=='initialized')
    raw,array,_=read_cache(e);features={(int(r['source_frame']),r['track_id']):(None if np.isnan(a).all() else a) for r,a in zip(raw,array)}
    byframe=defaultdict(list)
    for r in raw:byframe[int(r['source_frame'])].append(r)
    anchors={s:[features[f,str(t)] for f in range(ev['window_start'],ev['frame']+1)] for s,t in ev['seed_ids'].items()}
    chosen=[r for r in byframe[ev['frame']] if int(r['track_id']) in ev['seed_ids'].values()]
    background=[r for r in byframe[ev['frame']] if int(r['track_id']) not in ev['seed_ids'].values() and all(g.original.t.overlap_fraction(r,s)<.1 for s in chosen)]
    near=defaultdict(list)
    for r in read(Path(e['temporal_folder'])/'selected_four.csv'):
        if r['player_slot'].startswith('near_'):near[int(r['source_frame'])].append(r)
    history={};regions={}
    for f in range(ev['frame'],e['last']+1):regions[f]=g.near_regions(f,near[f],history)
    base=read(BASE/'far_identity_guard_development/matched_trial/case_019/baseline/far_pair_path.csv')
    wanted={46501,46502,46503,46505,46511,46634,46665,46666};diagnostics=[]
    for r in base:
        f=int(r['source_frame'])
        if f not in wanted or (f<46665 and r['player_slot']!='far_rear') or (f>=46665 and r['player_slot']!='far_front'):continue
        feature=features[f,r['track_id']];own=min(g.part_distance(a,feature,0) for a in anchors[r['player_slot']]);bg=[];nr=[]
        for b in background:
            overlap=g.original.t.overlap_fraction(g.upper_patch(r),b)
            if overlap>0:
                distance=g.part_distance(features[ev['frame'],b['track_id']],feature,0)
                bg.append(dict(initial_raw_id=b['track_id'],upper_region_overlap=overlap,negative_upper_distance=distance,own_upper_distance=own,appearance_margin=own-distance,appearance_veto=distance+.04<own,spatial_veto=overlap>.55))
        for reg in regions[f]:
            overlap=g.original.t.overlap_fraction(r,reg['row']);distance=min(g.original.a.app_distance(a,feature) for a in anchors[reg['slot']]);own_mean=min(g.original.a.app_distance(a,feature) for a in anchors[r['player_slot']])
            nr.append(dict(near_slot=reg['slot'],kind=reg['kind'],age=reg['age'],overlap=overlap,near_distance=distance,own_distance=own_mean,appearance_veto=distance+.04<own_mean,spatial_veto=overlap>.55))
        diagnostics.append(dict(source_frame=f,slot=r['player_slot'],raw_id=r['track_id'],overlapping_initial_negative_regions=bg,near_regions=nr,not_independent_GT=True))
    dump(OUT/'residual_mechanism_diagnostic.json',dict(rows=diagnostics,no_new_thresholds_or_solver_runs=True,interpretation='46511 spatially overlaps an initial negative but does not meet the same-exemplar upper appearance margin.46665/46666 near-rear past-box extrapolation fails the overlap criterion. These are algorithm diagnostics, not recovered ground truth.'))
    geometry=[]
    for f in range(46656,46668):
        for r in near[f]:
            if r['player_slot']=='near_rear':geometry.append(dict(source_frame=f,observed=1,**{k:r[k] for k in ['track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y']}))
        if not any(r['player_slot']=='near_rear' for r in near[f]):geometry.append(dict(source_frame=f,observed=0))
    write(OUT/'near_rear_pre_gap_native_geometry.csv',geometry)
    c=load(BASE/'cohort_tracking/case_019/protocol.json');cap=cv2.VideoCapture(c['source_video']);cells=[]
    for f in [46658,46659,46660,46665]:
        cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1
        for reg in regions[f]:
            if reg['slot']!='near_rear':continue
            r=reg['row'];bb=[round(float(r[k])) for k in ['x1','y1','x2','y2']];color=(255,80,240) if reg['age'] else (255,190,40)
            cv2.rectangle(im,tuple(bb[:2]),tuple(bb[2:]),color,5)
        crop=cv2.resize(im[1000:1600,1400:2500],(880,480));canvas=np.zeros((540,880,3),np.uint8);canvas[60:540]=crop
        label='past-box prediction ONLY; not an output position' if f==46665 else 'observed near N2 rectangle; bottom is a proxy'
        cv2.putText(canvas,f'019 F{f}: {label}',(8,30),0,.52,(255,255,255),1);cells.append(canvas)
    cap.release();image=OUT/'near_gap_prediction_diagnostic.jpg';assert cv2.imwrite(str(image),np.vstack([np.hstack(cells[:2]),np.hstack(cells[2:])]))
    dump(OUT/'near_gap_prediction_image.json',dict(source_frames=[46658,46659,46660,46665],crop_native_xyxy=[1400,1000,2500,1600],display_size=[880,480],aspect_preserved=True,image_sha256=sha(image),AI_inspection_not_human_GT=True))
    print('Saved8 probe diagnostics and near-gap geometry without tuning.')

if __name__=='__main__':main()
