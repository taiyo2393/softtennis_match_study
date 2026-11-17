"""Select four seeded court players and stitch only unmistakable ID handovers.

The four initial boxes are tied to this one inspected clip. Reusing these seeds
for another rally, camera angle, or court change would be invalid.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
READY=ROOT/'data/segments/analysis_ready_main_camera_segments.csv'

def read(path):
    with path.open(newline='', encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))


def overlap(a, b):
    ax1, ay1, ax2, ay2 = [float(a[k]) for k in ('x1','y1','x2','y2')]
    bx1, by1, bx2, by2 = [float(b[k]) for k in ('x1','y1','x2','y2')]
    o = max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1))
    union = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-o
    return o/union if union else 0


def missing_runs(frames):
    if not frames:
        return []
    runs=[]
    start=prev=frames[0]
    for f in frames[1:]:
        if f != prev+1:
            runs.append([start,prev])
            start=f
        prev=f
    runs.append([start,prev])
    return runs


def seed_by_court_geometry(rows, first, width, height):
    """Use this match's fixed-view court geometry to seed two players per end."""
    first_rows=[r for r in rows if int(r['source_frame'])==first and int(r['track_id'])>=0]
    near=[];far=[]
    for r in first_rows:
        x=float(r['bottom_center_x'])/width
        y=float(r['bottom_center_y'])/height
        box_height=(float(r['y2'])-float(r['y1']))/height
        if y>=.61 and box_height>=.08:
            near.append(r)
        elif .55<=y<.61 and .30<=x<=.63 and box_height>=.05:
            far.append(r)
    near=sorted(near,key=lambda r:float(r['confidence']),reverse=True)[:2]
    far=sorted(far,key=lambda r:float(r['confidence']),reverse=True)[:2]
    if len(near)!=2 or len(far)!=2:
        raise RuntimeError(f'Cannot automatically seed four players at F{first}: near={len(near)} far={len(far)}')
    near=sorted(near,key=lambda r:float(r['bottom_center_y']))
    far=sorted(far,key=lambda r:float(r['bottom_center_y']))
    return {'near_front':int(near[0]['track_id']),'near_rear':int(near[1]['track_id']),
            'far_rear':int(far[0]['track_id']),'far_front':int(far[1]['track_id'])}


def analysis_ready_end(first):
    matches=[int(r['end_source_frame']) for r in read(READY)
             if r['video_id']=='match1' and r['usable_for_analysis'].lower()=='true'
             and int(r['start_source_frame'])<=first<=int(r['end_source_frame'])]
    if len(matches)!=1:
        raise RuntimeError(f'Expected exactly one analysis-ready interval for F{first}, found {len(matches)}')
    return matches[0]


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--frame-width',type=int,default=3840)
    p.add_argument('--frame-height',type=int,default=2160)
    p.add_argument('--mode',choices=('both','bytetrack','botsort_reid'),default='both')
    args=p.parse_args()
    summary=[]
    modes=('bytetrack','botsort_reid') if args.mode=='both' else (args.mode,)
    for mode in modes:
        rows=read(args.pilot/f'{mode}.csv')
        frames=sorted({int(r['source_frame']) for r in rows})
        if not frames:
            raise RuntimeError(f'Empty tracker output: {mode}')
        first=frames[0]
        clip_end=analysis_ready_end(first)
        frames=[f for f in frames if f<=clip_end]
        rows=[r for r in rows if int(r['source_frame'])<=clip_end]
        tracks=defaultdict(dict)
        for row in rows:
            tid=int(row['track_id'])
            if tid>=0:
                tracks[tid][int(row['source_frame'])]=row
        seeds=seed_by_court_geometry(rows,first,args.frame_width,args.frame_height)
        if not all(first in tracks[tid] for tid in seeds.values()):
            raise RuntimeError('Inspected four-player seeds missing from the first frame')
        assigned=set(seeds.values())
        chains={slot:[tid] for slot,tid in seeds.items()}
        stitch_events=[]
        for slot,chain in chains.items():
            while True:
                old_id=chain[-1]
                old=tracks[old_id]
                end=max(old)
                # A new ID may duplicate the old box for one frame, as at F519.
                candidates=[]
                for new_id,new in tracks.items():
                    if new_id in assigned:
                        continue
                    start=min(new)
                    gap=start-end-1
                    if not end-1 <= start <= end+8 or len(new)<(10 if gap>1 else 3):
                        continue
                    junction=max(start,min(end,max(old)))
                    a=old.get(junction) or old[end]
                    b=new.get(junction) or new[start]
                    score=overlap(a,b)
                    foot_gap=((float(a['bottom_center_x'])-float(b['bottom_center_x']))**2+
                              (float(a['bottom_center_y'])-float(b['bottom_center_y']))**2)**.5
                    old_height=float(a['y2'])-float(a['y1'])
                    new_height=float(b['y2'])-float(b['y1'])
                    if gap<=1 and score>=.55 and foot_gap<=.25*old_height:
                        candidates.append((score,new_id,start,'overlap_or_short_gap'))
                    elif 1<gap<=7 and foot_gap<=.5*old_height and .75<=new_height/old_height<=1.4 and float(b['confidence'])>=.35:
                        candidates.append((1-foot_gap/old_height,new_id,start,'gap_relinked_review'))
                if len(candidates)!=1:
                    break
                score,new_id,start,reason=candidates[0]
                chain.append(new_id)
                assigned.add(new_id)
                stitch_events.append(dict(slot=slot,old_id=old_id,new_id=new_id,start_frame=start,
                                          link_type=reason,link_score=round(score,3)))
        selected=[]
        mode_summary={'mode':mode,'seed_frame':first,'analysis_ready_end_frame':clip_end,
                      'seed_ids':seeds,'stitches':stitch_events,'slots':{}}
        for slot,chain in chains.items():
            observed=0
            gaps=[]
            for frame in frames:
                found=[tracks[tid][frame] for tid in chain if frame in tracks[tid]]
                # At a transition, prefer the earlier trusted ID in the shared frame.
                if found:
                    row=found[0].copy()
                    row['player_slot']=slot
                    linked=next((s for s in stitch_events if s['slot']==slot and s['new_id']==int(row['track_id'])),None)
                    row['selection_status']=linked['link_type'] if linked else 'seed_track'
                    selected.append(row)
                    observed+=1
                else:
                    gaps.append(frame)
            mode_summary['slots'][slot]={'raw_track_ids':chain,'observed_frames':observed,'total_frames':len(frames),'missing_runs':missing_runs(gaps)}
        fields=list(rows[0])+['player_slot','selection_status']
        with (args.pilot/f'{mode}_selected_four.csv').open('w',newline='',encoding='utf-8-sig') as file:
            writer=csv.DictWriter(file,fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
        summary.append(mode_summary)
        print(json.dumps(mode_summary,ensure_ascii=False))
    (args.pilot/'four_player_selection_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
