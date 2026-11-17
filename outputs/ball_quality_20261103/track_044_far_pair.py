"""Reinitialize044's far pair near the effective service; reuse joint search."""
import json,sys
from collections import defaultdict,Counter
from pathlib import Path
import cv2
import numpy as np
from extract_serve_start_features import ROOT,SOURCE,sha

sys.path.insert(0,str(ROOT.parent/'tracking_quality_20261102_stage10'))
import gap_pair_tracking as g
a=g.a
OUT=ROOT/'person_044_reinit'
INIT=115360;END=115775;FIRST=114635
SEEDS={'far_rear':422,'far_front':20};NEAR={'near_front':2,'near_rear':1}

def image_at(f):
    im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_016/frames'/f'{f-FIRST:06d}.jpg'))
    assert im is not None and im.shape[:2]==(1080,1920)
    return im

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'summary.json').exists()
    source=ROOT.parent/'frontcourt_handoff_20260927/full_rally_044/bytetrack.csv'
    raw=a.read(source);by=defaultdict(list)
    for r in raw:
        if int(r['track_id'])>=0 and float(r['confidence'])>=.25:by[int(r['source_frame'])].append(r)
    seed_rows={s:next(r for r in by[INIT] if int(r['track_id'])==tid) for s,tid in SEEDS.items()}
    anchors={s:[] for s in SEEDS};anchor_rows=[]
    for frame in range(INIT-20,INIT+1):
        im=image_at(frame)
        for slot,seed in seed_rows.items():
            candidates=[r for r in by[frame] if .6<=a.height(r)/a.height(seed)<=1.7
                and np.linalg.norm(a.foot(r)-a.foot(seed))<=160]
            if not candidates:continue
            r=min(candidates,key=lambda r:np.linalg.norm(a.foot(r)-a.foot(seed)))
            feature=a.appearance(im,r)
            if feature is not None:
                anchors[slot].append(feature);anchor_rows.append(dict(r,anchor_slot=slot))
    assert all(len(v)>=8 for v in anchors.values())
    initial=image_at(INIT)
    chosen=list(seed_rows.values())+[r for r in by[INIT] if int(r['track_id']) in NEAR.values()]
    negatives=[a.appearance(initial,r) for r in by[INIT] if int(r['track_id']) not in set(SEEDS.values())|set(NEAR.values())
        and all(g.t.overlap_fraction(r,s)<.1 for s in chosen)]
    negatives=[f for f in negatives if f is not None]
    segments,counts=g.tracklet_index(raw);cache=[];near=defaultdict(list)
    for f in range(INIT,END+1):
        for r in by[f]:
            slot=next((s for s,tid in NEAR.items() if int(r['track_id'])==tid),None)
            if slot:near[f].append(dict(r,player_slot=slot,position_source='detected_box',identity_status='legacy_near_id_unreviewed_in_this_clip'))
        im=image_at(f);items=[]
        for r in by[f]:
            x,y=a.foot(r)/[3840,2160]
            if not (.20<x<.90 and .49<y<.665):continue
            if any(g.t.overlap_fraction(r,n)>.55 for n in near[f]):continue
            feat=a.appearance(im,r)
            ds=[min(a.app_distance(z,feat) for z in anchors[s]) for s in ('far_rear','far_front')]
            if min(ds)>g.P['appearance_max']:continue
            if negatives and min(a.app_distance(z,feat) for z in negatives)+.04<min(ds):continue
            key=segments[(f,int(r['track_id']))]
            items.append(dict(row=r,appearance=ds,point=a.foot(r),height=a.height(r),tracklet_key=key,
                stable_tracklet=counts[key]>=g.P['stable_tracklet_min_observations']))
        if f==INIT:
            for slot,tid in SEEDS.items():
                if not any(int(c['row']['track_id'])==tid for c in items):
                    r=seed_rows[slot];key=segments[(f,tid)]
                    items.append(dict(row=r,appearance=[0.,0.],point=a.foot(r),height=a.height(r),
                        tracklet_key=key,stable_tracklet=counts[key]>=g.P['stable_tracklet_min_observations']))
        cache.append(items)
    protocol=dict(rally_candidate='044',ball_pilot='match1_motion_pilot_016',source_frame_interval=[INIT,END],
        local_frame_interval=[INIT-FIRST,END-FIRST],source_clock='legacy edited match1',
        raw_detector_source=str(source),raw_source_sha256=sha(source),search_parameters=g.P,
        search_code_sha256=sha(Path(g.__file__)),runner_sha256=sha(Path(__file__)),
        seed_ids=SEEDS,near_ids=NEAR,seed_role_meaning='F1=far-side server at initialization; F2=far-side partner. Names remain identities after movement.',
        initialization='Spatially selected detector seeds near known service, not manual identity ground truth',
        far_candidate_roi=dict(x=[.20,.90],y=[.49,.665]),anchor_window_source_frames=[INIT-20,INIT],
        negative_templates=len(negatives),interpolation=False,independent_accuracy_claim=False,
        difference_from_prior_candidate_extraction='Reinitialize near effective service; extend coarse lateral ROI for the far player running toward the right sideline; retain exact existing joint solver parameters.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    g.write(OUT/'initialization_anchors.csv',anchor_rows)
    path,search=g.solve(cache,dict(seed_ids=SEEDS));far=[];allrows=[];tracklets=defaultdict(set)
    for offset,choices in enumerate(path):
        f=INIT+offset
        for i,j in enumerate(choices):
            if j<0:continue
            item=cache[offset][j];r=item['row'];slot=('far_rear','far_front')[i]
            row=dict(r,local_frame=f-FIRST,player_slot=slot,position_source='detected_box',
                identity_status='joint_path_candidate_requires_review',tracklet_key=item['tracklet_key'])
            far.append(row);tracklets[item['tracklet_key']].add(slot)
        # Common schema for observed near and far rows; no missing coordinates.
        for r in near[f]:allrows.append(dict(r,local_frame=f-FIRST,tracklet_key='legacy_near_'+r['track_id']))
    allrows+=far;allrows.sort(key=lambda r:(int(r['source_frame']),r['player_slot']))
    g.write(OUT/'far_pair_path.csv',far);g.write(OUT/'observed_player_paths.csv',allrows)
    assert all(len(owners)==1 for owners in tracklets.values())
    lookup={(int(r['source_frame']),int(r['track_id'])):r for r in raw}
    for r in allrows:
        original=lookup[(int(r['source_frame']),int(r['track_id']))]
        assert all(r[k]==original[k] for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y','confidence'])
    for f in range(INIT,END+1):
        rows=[r for r in far if int(r['source_frame'])==f]
        assert len({r['track_id'] for r in rows})==len(rows)
    mainfar=[r for r in far if 115374<=int(r['source_frame'])<=END]
    mainall=[r for r in allrows if 115374<=int(r['source_frame'])<=END]
    byfar=Counter(int(r['source_frame']) for r in mainfar);byall=Counter(int(r['source_frame']) for r in mainall)
    summary=dict(search=search,total_output_frames=END-INIT+1,main_play_frames=402,
        far_assignments_in_main=len(mainfar),frames_with_two_far_candidate_boxes=sum(v==2 for v in byfar.values()),
        frames_with_four_candidate_boxes=sum(v==4 for v in byall.values()),
        slots={s:dict(rows=sum(r['player_slot']==s for r in mainall),raw_ids=dict(Counter(r['track_id'] for r in mainall if r['player_slot']==s))) for s in a.SLOTS},
        missing_local_frames={s:[f-FIRST for f in range(115374,END+1) if not any(int(r['source_frame'])==f and r['player_slot']==s for r in mainall)] for s in a.SLOTS},
        accuracy_not_measured=True,identity_review='pending',
        warning='Counts describe candidate availability, not correct identity or position. Future frames select the full offline path.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(dict(all_output_boxes_from_same_frame_original_detections=True,
        no_predicted_or_interpolated_positions_exported=True,no_detection_assigned_to_two_far_identities=True,
        continuous_tracklet_owner_consistent=True,source_detector_csv_unchanged=sha(source)==protocol['raw_source_sha256'],
        source_frame_to_local_offset=FIRST,identity_accuracy_verified=False),indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='missing_local_frames'},indent=2),flush=True)

if __name__=='__main__':main()
