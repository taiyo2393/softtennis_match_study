"""Frozen2x scale diagnostic, not an adopted tracker or independent benchmark.

Nine uniformly resized, overlapping960x540 source tiles use the unchanged
four-frame TrackNet+median model. Candidate extraction is unchanged. Reviewed
coordinates are loaded only after inference and never choose a tile or component.
"""
import json,sys,time
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np,torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha

sys.path.insert(0,str(ROOT.parent/'ball_quality_20261102'))
from extract_weak_candidates import components

OUT=ROOT/'ball_044_scale_probe'
TILES=[(x,y,960,540) for y in (0,270,540) for x in (0,480,960)]
EXTRA=(773,784,891,892,893,960,968,970,1103)
TIMES=sorted(set(range(1068,1107))|set(EXTRA))

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Preserve the fixed diagnostic snapshot.'
    cid='match1_motion_pilot_016';base=SOURCE/'data/balltrack/racketvision/match1'/cid
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack'
    checkpoint=modelroot/'checkpoints/balltrack_best.pth'
    tracked=[ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json']
    protected={str(p):sha(p) for p in tracked}
    manifest=load(tracked[0])['clips'][cid];ballpath=Path(manifest['reviewed_output'])
    protected[str(ballpath)]=sha(ballpath)
    protocol=dict(status='fixed_before_inference',model='unchanged RacketVision MS-TrackNetV3',
        checkpoint_sha256=sha(checkpoint),script_sha256=sha(Path(__file__)),
        target_local_frames=TIMES,selection_scope='User-reported1080 vicinity plus previously reviewed candidate frames; development cases, not a test set',
        tiles_xywh=TILES,tile_selection='All9 fixed image tiles for every requested frame; no label-selected tile',
        source_image_wh=[1920,1080],network_wh=[512,288],tile_scale_gain=2,
        candidate_extractor='unchanged floor0.10, top6 per tile, half-peak weighted centroid',
        model_input='BGR t-3..t plus corresponding cropped median120, bilinear resize, /255',
        comparison='Cached unchanged full-frame candidates versus all tiled candidates',
        deduplication_for_count='Within12 source-image pixels, keep highest peak; no ball-identity inference',
        descriptive_match_radius_px=24,coordinate_reference='Previously reviewed ball-enclosing component centers; no precise-center independent GT',
        automatic_adoption=False,missing_positions_filled=False,independent_accuracy_evaluation=False,
        input_hashes=protected)
    dump(OUT/'protocol.json',protocol)
    sys.path.insert(0,str(modelroot))
    from inference import BallInferencer
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(checkpoint),device='cuda',thre=.3,batchsize=8)
    median=np.load(base/'runs/threshold030_median120/median.npz')['median']
    assert median.shape[:2]==(1080,1920),median.shape
    image_cache={}
    for t in sorted({j for t in TIMES for j in range(t-3,t+1)}):
        im=cv2.imread(str(base/'frames'/f'{t:06d}.jpg'));assert im is not None
        image_cache[t]=im
    rows=[];hm_rows=[];jobs=[(t,k) for t in TIMES for k in range(len(TILES))]
    medians=[]
    for x,y,w,h in TILES:
        medians.append(np.moveaxis(cv2.resize(median[y:y+h,x:x+w],(512,288)).astype(np.float32)/255.,-1,0))
    start=time.monotonic()
    for first in range(0,len(jobs),8):
        batch=jobs[first:first+8];inputs=[]
        for t,k in batch:
            x,y,w,h=TILES[k]
            frames=np.stack([np.moveaxis(cv2.resize(image_cache[j][y:y+h,x:x+w],(512,288)).astype(np.float32)/255.,-1,0) for j in range(t-3,t+1)])
            inputs.append(np.concatenate([medians[k][None],frames],axis=0).reshape(-1,288,512))
        tensor=torch.from_numpy(np.stack(inputs)).to(infer.device)
        with torch.inference_mode(),torch.amp.autocast('cuda'):
            predictions,_,_=infer.model.forward(frames=tensor)
        for (t,k),hm in zip(batch,predictions[:,0].float().cpu().numpy()):
            x,y,w,h=TILES[k]
            hm_rows.append(hm.astype(np.float16))
            for rank,c in enumerate(components(hm,w,h),1):
                rows.append(dict(local_frame=t,tile_index=k,tile_x=x,tile_y=y,rank=rank,
                    x_px=c['x_px']+x,y_px=c['y_px']+y,peak=c['peak'],area=c['area'],mean=c['mean']))
        if first%80==0:print('tiled probe',first,'/',len(jobs),flush=True)
    write(OUT/'tiled_components.csv',rows)
    np.savez_compressed(OUT/'tile_heatmaps.npz',heatmaps=np.stack(hm_rows),
        local_frames=np.array([t for t,k in jobs]),tile_indices=np.array([k for t,k in jobs]))
    # Inference is complete; only now use reviewed coordinates for descriptive matching.
    full=[c for c in read(ROOT/'inplay_ball_044/candidates.csv') if int(c['local_frame']) in TIMES]
    ball=read(ballpath);labelled=[]
    for t in manifest['confirmed_ball_candidate_local_frames']:
        b=ball[t];labelled.append(dict(local_frame=t,label='confirmed_ball_component',x_px=float(b['x_px']),y_px=float(b['y_px'])))
    for review_path in manifest['review_records']:
        review=load(Path(review_path))
        for r in review.get('rejected_raw_observations',[]):
            labelled.append(dict(local_frame=r['local_frame'],label='confirmed_false_raw_component',x_px=r['rejected_xy_px'][0],y_px=r['rejected_xy_px'][1]))
    by_full=defaultdict(list);by_tile=defaultdict(list)
    for c in full:by_full[int(c['local_frame'])].append(c)
    for c in rows:by_tile[int(c['local_frame'])].append(c)
    comparisons=[]
    for lab in labelled:
        t=lab['local_frame'];p=np.array([lab['x_px'],lab['y_px']]);r=dict(lab)
        for name,pool in [('full',by_full[t]),('tiles',by_tile[t])]:
            near=[c for c in pool if np.linalg.norm(np.array([float(c['x_px']),float(c['y_px'])])-p)<=24]
            near.sort(key=lambda c:float(c['peak']),reverse=True)
            c=near[0] if near else None
            r.update({name+'_component_within24px':int(c is not None),name+'_nearby_peak':float(c['peak']) if c else '',
                name+'_nearby_x_px':c['x_px'] if c else '',name+'_nearby_y_px':c['y_px'] if c else '',
                name+'_component_count':len(pool)})
        comparisons.append(r)
    write(OUT/'reviewed_case_comparison.csv',comparisons)
    counts=[]
    for t in TIMES:
        unique=[]
        for c in sorted(by_tile[t],key=lambda c:float(c['peak']),reverse=True):
            if not any(np.hypot(float(c['x_px'])-float(q['x_px']),float(c['y_px'])-float(q['y_px']))<=12 for q in unique):unique.append(c)
        counts.append(dict(local_frame=t,full_components=len(by_full[t]),tile_components=len(by_tile[t]),tile_unique_components=len(unique)))
    write(OUT/'candidate_counts.csv',counts)
    positives=[r for r in comparisons if r['label']=='confirmed_ball_component']
    negatives=[r for r in comparisons if r['label']=='confirmed_false_raw_component']
    report=dict(frames=len(TIMES),tile_inference_jobs=len(jobs),inference_seconds=time.monotonic()-start,
        reviewed_positive_examples=len(positives),full_positive_nearby_components=sum(r['full_component_within24px'] for r in positives),
        tile_positive_nearby_components=sum(r['tiles_component_within24px'] for r in positives),
        reviewed_false_examples=len(negatives),full_false_nearby_components=sum(r['full_component_within24px'] for r in negatives),
        tile_false_nearby_components=sum(r['tiles_component_within24px'] for r in negatives),
        full_component_count=sum(r['full_components'] for r in counts),tile_unique_component_count=sum(r['tile_unique_components'] for r in counts),
        comparison_is_development_component_proximity_only=True,independent_accuracy_established=False,
        warning='More components near reviewed examples does not imply better single-ball tracking. Neighborhood matches may hit another object; compare image evidence before adoption.')
    assert all(sha(Path(p))==h for p,h in protected.items())
    report['reviewed_inputs_unchanged']=True
    dump(OUT/'report.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
