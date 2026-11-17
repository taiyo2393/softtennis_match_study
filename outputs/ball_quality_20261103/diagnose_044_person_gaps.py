"""Explain candidate rejection during reviewed044 paths' four missing runs."""
import json
from collections import defaultdict,Counter
from pathlib import Path
import cv2
import numpy as np
from track_044_far_pair import ROOT,OUT as BASE,SOURCE,INIT,FIRST,SEEDS,NEAR,a,g,image_at
from extract_serve_start_features import sha

OUT=ROOT/'person_044_gap_audit'
GAPS=[('far_front',796,835),('far_rear',898,912),('far_rear',982,999),('far_rear',1102,1115)]

def setup():
    proto=json.loads((BASE/'protocol.json').read_text())
    raw=a.read(Path(proto['raw_detector_source']));by=defaultdict(list)
    for r in raw:
        if int(r['track_id'])>=0 and float(r['confidence'])>=.25:by[int(r['source_frame'])].append(r)
    anchors=defaultdict(list)
    for r in a.read(BASE/'initialization_anchors.csv'):
        feat=a.appearance(image_at(int(r['source_frame'])),r)
        if feat is not None:anchors[r['anchor_slot']].append(feat)
    initial=image_at(INIT)
    chosen=[r for r in by[INIT] if int(r['track_id']) in set(SEEDS.values())|set(NEAR.values())]
    negative=[a.appearance(initial,r) for r in by[INIT] if int(r['track_id']) not in set(SEEDS.values())|set(NEAR.values()) and all(g.t.overlap_fraction(r,s)<.1 for s in chosen)]
    return proto,raw,by,anchors,[n for n in negative if n is not None]

def main():
    OUT.mkdir(exist_ok=True)
    proto,raw,by,anchors,negatives=setup()
    reviewed=json.loads((ROOT/'reviewed_person_overrides.json').read_text())['rallies']['match1_rally_window_044']
    paths=a.read(Path(reviewed['reviewed_output']));lookup={(int(r['local_frame']),r['player_slot']):r for r in paths}
    details=[];summaries=[]
    for slot,start,end in GAPS:
        before=lookup[(start-1,slot)];after=lookup[(end+1,slot)]
        counters=Counter();representatives=[]
        for t in range(start,end+1):
            f=FIRST+t;im=image_at(f);near=[r for r in by[f] if int(r['track_id']) in NEAR.values()]
            # Reference only for finding diagnostic crops/candidates, never exported
            # as a player coordinate or used to fill a gap.
            w=(t-(start-1))/(end-start+2);reference=(1-w)*a.foot(before)+w*a.foot(after)
            scale=max(a.height(before),a.height(after));items=[]
            for r in by[f]:
                if np.linalg.norm(a.foot(r)-reference)>1.5*scale:continue
                x,y=a.foot(r)/[3840,2160];feat=a.appearance(im,r)
                ds=[min(a.app_distance(z,feat) for z in anchors[s]) for s in ('far_rear','far_front')]
                neg=min(a.app_distance(z,feat) for z in negatives) if negatives else 1.
                overlap=max([g.t.overlap_fraction(r,n) for n in near] or [0.])
                reasons=[]
                if not (.20<x<.90 and .49<y<.665):reasons.append('outside_coarse_far_roi')
                if overlap>.55:reasons.append('near_overlap_veto')
                if min(ds)>g.P['appearance_max']:reasons.append('appearance_gate')
                if neg+.04<min(ds):reasons.append('negative_template_veto')
                items.append(dict(row=r,normalized_distance_to_reference=float(np.linalg.norm(a.foot(r)-reference)/scale),
                    appearance=ds,negative_distance=neg,near_overlap_fraction=overlap,
                    candidate_filter_reasons=reasons,passes_candidate_filters=not reasons))
            items.sort(key=lambda q:q['normalized_distance_to_reference'])
            # This nearest object is diagnostic, not an identity annotation.
            nearest=items[0] if items else None
            state='candidate_passes_but_unassigned' if any(i['passes_candidate_filters'] for i in items) else 'only_rejected_candidates' if items else 'no_nearby_detection'
            counters[state]+=1
            if nearest:
                for reason in nearest['candidate_filter_reasons']:counters['nearest_'+reason]+=1
            detail=dict(slot=slot,local_frame=t,source_frame=f,diagnostic_status=state,candidates=items,
                reference_is_interpolated_for_query_only=True)
            details.append(detail)
            if t in [start,(start+end)//2,end]:representatives.append(detail)
        summaries.append(dict(slot=slot,missing_local_interval=[start,end],frames=end-start+1,
            diagnostic_counts=dict(counters),representatives=representatives,
            human_occlusion_or_visibility_annotation='not_established'))
        # Equal-aspect fixed crop per gap, both original image and nearest raw boxes.
        points=np.array([a.foot(before),a.foot(after)])*.5;center=np.mean(points,axis=0)
        left=int(np.clip(round(center[0]-210),0,1500));top=int(np.clip(round(center[1]-230),0,720))
        tiles=[]
        for t in [start-1,start,(start+end)//2,end,end+1]:
            f=FIRST+t;im=image_at(f);crop=im[top:top+360,left:left+420].copy()
            rr=next((d for d in details if d['slot']==slot and d['local_frame']==t),None)
            if rr:
                for i,item in enumerate(rr['candidates'][:3]):
                    r=item['row'];x1,y1,x2,y2=[float(r[k])*.5 for k in ['x1','y1','x2','y2']]
                    c=(0,210,255) if item['candidate_filter_reasons'] else (80,255,80)
                    cv2.rectangle(crop,(round(x1-left),round(y1-top)),(round(x2-left),round(y2-top)),c,1)
                    cv2.putText(crop,str(r['track_id']),(max(0,round(x1-left)),max(20,round(y1-top)-4)),0,.5,c,1)
            tile=np.full((405,420,3),22,np.uint8);tile[45:]=crop
            cv2.putText(tile,f'{slot} local {t}',(8,19),0,.5,(235,235,235),1)
            cv2.putText(tile,'orange=rejected, green=passes filters',(8,38),0,.43,(235,235,235),1)
            tiles.append(tile)
        cv2.imwrite(str(OUT/f'{slot}_{start}_{end}.jpg'),np.concatenate(tiles,axis=1))
    (OUT/'diagnostics.json').write_text(json.dumps(dict(gaps=summaries,frames=details,
        source_paths_sha256=sha(Path(reviewed['reviewed_output'])),
        limitation='Diagnostic candidate proximity and overlap are not independent visibility/occlusion or identity labels. No positions changed.'),indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in s.items() if k!='representatives'} for s in summaries],indent=2))

if __name__=='__main__':main()
