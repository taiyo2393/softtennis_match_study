"""Two-pass offline person retracking. No ball inference or legacy input writes.

Pass 1 is the detection/geometry/appearance part of legacy track_players, copied
without changing its parameters or coordinate conventions. Pass 2 replays the
same calibrated semantic tracker with whole-track or interval exclusion masks.
The default stationary-run mode uses interval medians and the 30 percent rule;
full-track retains the prior rule only as a reproducible reference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from track_points import load_scopes, INVENTORY, PROXY_FIELDS, write_csv, summarize_point, SUMMARY_FIELDS
sys.path.insert(0, str(ROOT/'outputs/ball_quality_20261111'))
import build_match2_player_ball_review as legacy
common = legacy.common
sys.path.insert(0, str(common.OLD))
import numpy as np
import cv2
from stationary_track_segments import stationary_runs, is_excluded, RULES as RUN_RULES
import fit_match2_court_geometry as geometry
from fit_match2_court_geometry_centers import fit_line_centers
from calibrated_camera_temporal_tracker import calibrated_temporal_class
from confirmed_continuity_guard import adaptive, old
from match_frame_coordinate_adapter import FrameCoordinates

RULES = dict(min_observed_seconds=2., median_u_outside=[-.05, 1.05],
             stationary_radius_m=.7, min_stationary_fraction=.9,
             court_dimensions_m=[10.97,23.77],
             duration='observed frame count / (60000/1001), not first-to-last span',
             position='native box bottom center, same projection convention as legacy')


def dump(path, obj):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def collect(point, scope, dest, force=False, replay_only=False):
    protocol = common.load(legacy.PILOT/'protocol.json')
    config = protocol['detector']
    cache = dest/'candidate_cache.pkl'
    signature = dict(point=point, scope=list(scope), detector=config,
                     video=protocol['source_video'], protocol_sha256=common.sha(legacy.PILOT/'protocol.json'),
                     implementation_sha256=common.sha(Path(__file__)))
    if cache.exists() and not force:
        with cache.open('rb') as f:
            saved=pickle.load(f)
        matching_inputs=all(saved['signature'].get(k)==v for k,v in signature.items() if k!='implementation_sha256')
        if saved['signature']==signature or (replay_only and matching_inputs):
            print(f'Point {point}: reusing {len(saved["frames"])} cached frames',flush=True)
            return saved
    if replay_only:
        raise ValueError(f'Point {point}: matching detection cache required for --replay-only')
    os.environ['YOLO_CONFIG_DIR']=str(dest/'ultralytics_config')
    from ultralytics import YOLO
    import ultralytics
    import torch
    assert torch.cuda.is_available(), 'CUDA required'
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    geometry.fit_line=fit_line_centers
    H=np.asarray(protocol['image_to_court_homography_canonical'])
    adapter=FrameCoordinates(1920,1080)
    detector=YOLO(config['model'])
    capture=cv2.VideoCapture(protocol['source_video'])
    first,start,last=scope
    assert capture.isOpened() and capture.set(cv2.CAP_PROP_POS_FRAMES,first)
    assert abs(capture.get(cv2.CAP_PROP_FPS)-common.FPS)<.001
    frames=[]
    started=time.perf_counter()
    try:
        for frame in range(first,last+1):
            ok,image=capture.read()
            assert ok and image.shape[:2]==(1080,1920)
            assert round(capture.get(cv2.CAP_PROP_POS_FRAMES))==frame+1
            fit,_=geometry.geometry(image)
            pred=detector.track(adapter.image(image),persist=True,classes=[0],
                                conf=config['confidence'],imgsz=config['imgsz'],
                                tracker=config['tracker'],device=config['device'],verbose=False)[0]
            candidates,appearance,uv=[],[],[]
            # Retain ALL ByteTrack candidates, including unsupported-camera frames.
            for box in pred.boxes:
                if box.id is None:
                    continue
                x1,y1,x2,y2=box.xyxy[0].cpu().tolist()
                row=dict(source_frame=frame,timestamp_sec=frame/common.FPS,
                         track_id=int(box.id.item()),x1=x1,y1=y1,x2=x2,y2=y2,
                         bottom_center_x=(x1+x2)/2,bottom_center_y=y2,
                         confidence=float(box.conf.item()),candidate_origin='full_frame_only')
                candidates.append(row)
                appearance.append(old.a.appearance(image,row))
                uv.append(common.project(H,row['bottom_center_x'],row['bottom_center_y']))
            frames.append(dict(source_frame=frame,camera_line_supported=fit is not None,
                               candidates=candidates,appearance=appearance,uv=uv))
            if (frame-first)%120==0 or frame==last:
                elapsed=time.perf_counter()-started
                print(f'Pass 1 point {point}: F{frame}/{last}, {frame-first+1}/{last-first+1}, '
                      f'{(frame-first+1)/max(elapsed,.001):.1f} fps, line={fit is not None}',flush=True)
    finally:
        capture.release()
    saved=dict(signature=signature,frames=frames,seconds=time.perf_counter()-started,
               runtime=dict(torch=torch.__version__,ultralytics=ultralytics.__version__))
    temp=cache.with_suffix('.tmp')
    with temp.open('wb') as f:
        pickle.dump(saved,f,protocol=pickle.HIGHEST_PROTOCOL)
    temp.replace(cache)
    return saved


def classify(point, frames):
    tracks=defaultdict(list)
    for f in frames:
        for row,uv in zip(f['candidates'],f['uv']):
            if np.isfinite(uv).all():
                tracks[row['track_id']].append((f['source_frame'],*uv))
    stats=[]
    for tid,values in sorted(tracks.items()):
        data=np.array(values)
        center=np.median(data[:,1:],axis=0)
        radius=np.linalg.norm((data[:,1:]-center)*[10.97,23.77],axis=1)
        seconds=len(data)/common.FPS
        fraction=float(np.mean(radius<=.7))
        duration_ok=seconds>=2
        outside=bool(center[0]<-.05 or center[0]>1.05)
        stable=fraction>=.9
        stats.append(dict(point=point,track_id=tid,first_frame=int(data[0,0]),last_frame=int(data[-1,0]),
                          observed_frames=len(data),observed_seconds=seconds,
                          span_seconds=float((data[-1,0]-data[0,0])/common.FPS),
                          median_u=float(center[0]),median_v=float(center[1]),
                          within_0_7m_fraction=fraction,radius_p90_m=float(np.quantile(radius,.9)),
                          duration_pass=duration_ok,outside_pass=outside,stationary_pass=stable,
                          excluded=duration_ok and outside and stable,
                          reason='; '.join([f'observed={seconds:.3f}s >=2: {duration_ok}',
                                           f'median_u={center[0]:.6f} outside [-.05,1.05]: {outside}',
                                           f'within .7m={fraction:.3%} >=90%: {stable}'])))
    return stats


def replay(point, scope, saved, excluded):
    protocol=common.load(legacy.PILOT/'protocol.json')
    H=np.asarray(protocol['image_to_court_homography_canonical'])
    legacy_H=np.asarray(common.load(common.CAL)['image_to_court_homography'])
    make_tracker=calibrated_temporal_class(adaptive.AdaptiveNearTracker,H,legacy_H)
    adapter=FrameCoordinates(1920,1080)
    tracker,segment=None,0
    rows,frames,events=[],[],[]
    for f in saved['frames']:
        frame=f['source_frame']
        assigned=[]
        if not f['camera_line_supported']:
            tracker=None
        else:
            if tracker is None:
                tracker=make_tracker()
                segment+=1
                events.append(dict(point=point,segment=segment,start=frame,events=tracker.events))
            keep=[i for i,c in enumerate(f['candidates'])
                  if not (is_excluded(c['track_id'],frame,excluded) if isinstance(excluded,dict)
                          else c['track_id'] in excluded)]
            # Copy because the legacy tracker is allowed to mutate its inputs.
            candidates=copy.deepcopy([f['candidates'][i] for i in keep])
            appearance=copy.deepcopy([f['appearance'][i] for i in keep])
            assigned=tracker.step(frame,candidates,appearance)
            if frame>=scope[1]:
                for slot,candidate,margin,cost,origin in assigned:
                    row=candidates[candidate]
                    u,v=common.project(H,row['bottom_center_x'],row['bottom_center_y'])
                    rows.append(dict(point=point,segment=segment,**adapter.row(row,to_canonical=False),
                                     player_slot=old.a.SLOTS[slot],court_proxy_x=u,court_proxy_y=v,
                                     assignment_margin=float(margin),assignment_cost=float(cost),identity_source=origin))
        frames.append(dict(point=point,source_frame=frame,in_core=int(frame>=scope[1]),
                           camera_line_supported=int(f['camera_line_supported']),
                           segment=segment if f['camera_line_supported'] else '',assigned_slots=len(assigned)))
    return rows,frames,events


def compare(point, before, after, kind):
    # Match per source frame and semantic slot, report missing separately.
    def index(rows):
        return {(int(r['source_frame']),r['player_slot']):r for r in rows}
    a,b=index(before),index(after)
    stats=[]
    for slot in common.SLOTS:
        ka={k for k in a if k[1]==slot};kb={k for k in b if k[1]==slot}
        distances=[];box_changed=0
        for key in sorted(ka&kb):
            x,y=a[key],b[key]
            distances.append(float(np.linalg.norm([(float(x['court_proxy_x'])-float(y['court_proxy_x']))*10.97,
                                                    (float(x['court_proxy_y'])-float(y['court_proxy_y']))*23.77])))
            box_changed+=any(float(x[k])!=float(y[k]) for k in ('x1','y1','x2','y2'))
        stats.append(dict(point=point,comparison=kind,player_slot=slot,common_frames=len(distances),
                          position_difference_ge_0_5m=sum(d>=.5 for d in distances),
                          max_difference_m=max(distances,default=0.),box_changed_frames=box_changed,
                          removed_frames=len(ka-kb),added_frames=len(kb-ka)))
    return stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points',type=int,nargs='+',default=[7,8,9,10,11])
    parser.add_argument('--force-detect',action='store_true')
    parser.add_argument('--replay-only',action='store_true',help='Reuse cached detections with identical source, scope and detector settings.')
    parser.add_argument('--stationarity-mode',choices=['full-track','stationary-run'],default='stationary-run',
                        help='stationary-run applies interval medians and the 30 percent rule.')
    args=parser.parse_args()
    if args.force_detect and args.replay_only:parser.error('--force-detect and --replay-only are incompatible')
    rules=RUN_RULES if args.stationarity_mode=='stationary-run' else RULES
    output=HERE/'tracking_v2'
    output.mkdir(exist_ok=True)
    (output/'.gitignore').write_text('*/candidate_cache.pkl\n*/ultralytics_config/\n*/player_court_proxies.csv\n*/baseline_replay.csv\n*/official_candidates.json\ntrack_statistics.csv\n*/strict_reference_player_court_proxies.csv\n*/strict_reference_official_candidates.json\nstrict_reference_track_statistics.csv\n',encoding='utf-8')
    with (output/'.gitignore').open('a',encoding='utf-8') as stream:
        stream.write('protected_all_before_stationary_run.json\n')
    if args.stationarity_mode=='stationary-run':
        for name in ['comparison.csv','track_statistics.csv','verification.json','summary.csv']:
            source=output/name
            target=output/('strict_reference_'+name)
            if source.exists() and not target.exists():shutil.copy2(source,target)
    all_stats,all_comparisons,summaries=[],[],[]
    protected={str(p):common.sha(p) for directory in (HERE/'tracking',HERE/'pose')
               for p in directory.rglob('*') if p.is_file()}
    protected[str(ROOT/'outputs/ball_quality_20261111/build_match2_player_ball_review.py')]=common.sha(Path(legacy.__file__))
    dump(output/'input_sha256_before.json',protected)
    segment_candidates=[]
    for point,scope in load_scopes(INVENTORY,args.points).items():
        dest=output/f'point_{point:02d}'
        dest.mkdir(exist_ok=True)
        saved=collect(point,scope,dest,args.force_detect,args.replay_only)
        stats=classify(point,saved['frames'])
        if args.stationarity_mode=='stationary-run':
            additional=stationary_runs(saved['frames'],stats,common.FPS)
            segment_candidates+=additional
            by_id={s['track_id']:s for s in stats}
            for stat in stats:
                stat['full_track_excluded']=stat['excluded']
                stat['excluded']=False
                stat.update(exclusion_basis='none',exclusion_mode='none',stationary_fraction=0.,
                            stationary_seconds=0.,existence_seconds=(stat['last_frame']-stat['first_frame']+1)/common.FPS,
                            stationary_runs=[],exclusion_ranges=[])
            for extra in additional:
                stat=by_id[extra['track_id']]
                stat['excluded']=True
                stat['exclusion_basis']='stationary_interval_rule'
                stat.update({k:v for k,v in extra.items() if k not in ('point','track_id','existence_frames','stationary_frames')})
            for name in ['player_court_proxies.csv','official_candidates.json','protocol.json']:
                source=dest/name
                target=dest/('strict_reference_'+name)
                if source.exists() and not target.exists():shutil.copy2(source,target)
        excluded={s['track_id'] for s in stats if s['excluded']}
        exclusion_mask=({s['track_id']:s['exclusion_ranges'] for s in stats if s['excluded']}
                        if args.stationarity_mode=='stationary-run' else excluded)
        all_stats+=stats
        for s in stats:
            if s['excluded']:
                print(f'EXCLUDE point {point} ID{s["track_id"]}: {s["reason"]}',flush=True)
        if not excluded:
            print(f'Point {point}: no official candidates',flush=True)
        baseline,_,_=replay(point,scope,saved,set())
        original=common.read(HERE/f'tracking/point_{point:02d}/player_court_proxies.csv')
        control=compare(point,original,baseline,'original_vs_unfiltered_replay')
        write_csv(dest/'baseline_replay.csv',PROXY_FIELDS,baseline)
        rows,frames,events=replay(point,scope,saved,exclusion_mask)
        comparisons=control+compare(point,original,rows,'original_vs_v2')+compare(point,baseline,rows,'unfiltered_vs_v2')
        all_comparisons+=comparisons
        write_csv(dest/'player_court_proxies.csv',PROXY_FIELDS,rows)
        write_csv(dest/'player_frame_availability.csv',list(frames[0]),frames)
        dump(dest/'player_segment_events.json',events)
        dump(dest/'official_candidates.json',dict(rules=rules,tracks=stats,excluded_track_ids=sorted(excluded)))
        dump(dest/'protocol.json',dict(source_video=saved['signature']['video'],source_fps='60000/1001',
                                      scope=scope,detector=saved['signature']['detector'],runtime=saved['runtime'],
                                      official_exclusion_rules=rules,excluded_track_ids=sorted(excluded),
                                      exclusion_ranges=exclusion_mask if isinstance(exclusion_mask,dict) else {},
                                      stationarity_mode=args.stationarity_mode,
                                      cache_signature=saved['signature'],baseline_replay_comparison=control))
        summaries.append(summarize_point(point,scope[0],scope[2],dest))
        write_csv(output/'comparison.csv',list(comparisons[0]),all_comparisons)
        write_csv(output/'track_statistics.csv',list(stats[0]),all_stats)
        write_csv(output/'summary.csv',SUMMARY_FIELDS,summaries)
        print(f'Pass 2 point {point} complete: excluded={sorted(excluded)}; differences='+
              str([(r['player_slot'],r['position_difference_ge_0_5m']) for r in comparisons if r['comparison']=='original_vs_v2']),flush=True)
    assert all(common.sha(Path(p))==h for p,h in protected.items()), 'Protected input changed'
    dump(output/'verification.json',dict(protected_inputs_unchanged=len(protected),
                                        excluded_candidates=[s for s in all_stats if s['excluded']],rules=rules,
                                        stationarity_mode=args.stationarity_mode,segment_candidates=segment_candidates))


if __name__=='__main__':
    main()
