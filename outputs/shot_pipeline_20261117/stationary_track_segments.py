"""Offline disjoint stationary runs; each interval uses its own median.

Search earliest qualifying 2s seed, extend right until invalid, resume after
that run. Missing/nonfinite observations count as excursions. Endpoints must
be inliers; no full-track outside gate or camera-line gate is used.
"""
from collections import defaultdict
import math
import numpy as np

RULES = dict(min_stationary_seconds=2., median_u_outside=[-.05,1.05],
    stationary_radius_m=.7,max_excursion_seconds=.2,full_track_stationary_fraction=.30,
    interval_padding_seconds=.5,court_dimensions_m=[10.97,23.77],
    duration='inclusive source-frame span / exact FPS; track existence includes gaps',
    search='earliest qualifying 2s seed, extend right while valid; disjoint intervals',
    median='componentwise median of finite observed positions in the interval',
    missing='missing/nonfinite positions count toward consecutive excursion limit',
    camera_support='all ByteTrack observations; no camera-line gate',
    padding='ceil(.5 * FPS) source frames each side, only for interval exclusion')

def interval_evidence(xy,first,last,fps):
    part=xy[first:last+1]
    finite=np.isfinite(part).all(axis=1)
    if not finite.any():return None
    center=np.median(part[finite],axis=0)
    if not (center[0]<-.05 or center[0]>1.05):return None
    radius=np.full(len(part),np.inf)
    radius[finite]=np.linalg.norm((part[finite]-center)*[10.97,23.77],axis=1)
    inside=radius<=.7
    if not inside[0]:return None
    bad=np.flatnonzero(~inside)
    groups=np.split(bad,np.flatnonzero(np.diff(bad)>1)+1) if len(bad) else []
    longest=max(map(len,groups),default=0)
    if longest>math.floor(.2*fps+1e-9):return None
    return dict(median_u=float(center[0]),median_v=float(center[1]),
        observed_frames=int(finite.sum()),inlier_frames=int(inside.sum()),
        excursion_frames=int((~inside).sum()),max_excursion_frames=longest,
        within_0_7m_fraction=float(inside.mean()),ends_inside=bool(inside[-1]))

def stationary_runs(frames,full_track_stats,fps):
    tracks=defaultdict(list)
    for frame in frames:
        for row,uv in zip(frame['candidates'],frame['uv']):
            tracks[row['track_id']].append((frame['source_frame'],*uv))
    result=[]
    minimum=math.ceil(2*fps-1e-9);padding=math.ceil(.5*fps-1e-9)
    for stat in full_track_stats:
        values=tracks[stat['track_id']]
        if not values:continue
        first,last=int(values[0][0]),int(values[-1][0])
        xy=np.full((last-first+1,2),np.nan)
        for frame,u,v in values:xy[int(frame)-first]=u,v
        runs=[];start=0
        while start+minimum<=len(xy):
            end=start+minimum-1
            evidence=interval_evidence(xy,start,end,fps)
            if evidence is None or not evidence['ends_inside']:
                start+=1;continue
            best_end,best=end,evidence
            for extension in range(end+1,len(xy)):
                candidate=interval_evidence(xy,start,extension,fps)
                if candidate is None:break
                if candidate['ends_inside']:best_end,best=extension,candidate
            best.pop('ends_inside')
            runs.append(dict(first_frame=first+start,last_frame=first+best_end,
                duration_frames=best_end-start+1,duration_seconds=(best_end-start+1)/fps,**best))
            start=best_end+1
        if not runs:continue
        stationary_frames=sum(r['duration_frames'] for r in runs)
        fraction=stationary_frames/len(xy)
        mode='whole_track' if fraction>=.30 else 'interval'
        ranges=([[first,last]] if mode=='whole_track' else
            [[max(first,r['first_frame']-padding),min(last,r['last_frame']+padding)] for r in runs])
        merged=[]
        for a,b in ranges:
            if merged and a<=merged[-1][1]+1:merged[-1][1]=max(b,merged[-1][1])
            else:merged.append([a,b])
        result.append(dict(point=stat['point'],track_id=stat['track_id'],
            exclusion_mode=mode,stationary_runs=runs,exclusion_ranges=merged,
            existence_frames=len(xy),existence_seconds=len(xy)/fps,
            stationary_frames=stationary_frames,stationary_seconds=stationary_frames/fps,
            stationary_fraction=fraction,
            reason=f'{len(runs)} outside-u stationary interval(s) >=2s; '
                f'{stationary_frames}/{len(xy)} frames = {fraction:.3%}; '+
                ('>=30%: exclude whole track' if mode=='whole_track' else
                 '<30%: exclude intervals with +/-0.5s padding')))
    return result

def is_excluded(track_id,frame,exclusions):
    return any(a<=frame<=b for a,b in exclusions.get(track_id,[]))
