"""Training-only fixed native/degraded tile comparison; unchanged detector."""
import json,sys,time,math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2
from run_manual_ball_finetune import tensors,metrics
from compare_ball_candidate_decoders import components,choose
from run_hard_case_ball_finetune import good_ids,err
from calibrate_point_supervised_detector import compact,expand

OUT=ROOT.parent/'ball_quality_20261109/training_tile_detail_comparison'
BASE=ROOT/'ball_temporal_path_comparison'
TILES=[(x,y,960,540) for y in [0,270,540] for x in [0,480,960]]
ARMS=['native','degraded']

def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

def transform(image,box,arm):
    x,y,w,h=box;crop=image[y:y+h,x:x+w]
    assert crop.shape[:2]==(540,960)
    if arm=='degraded':crop=cv2.resize(crop,(256,144),interpolation=cv2.INTER_LINEAR)
    return cv2.resize(crop,(512,288),interpolation=cv2.INTER_LINEAR)

def mapped(cs,tile_index):
    x,y,w,h=TILES[tile_index]
    return [dict(c,tile_index=tile_index,tile_x=x,tile_y=y,
                 x_px=int((c['x']+c['w']/2)*w/512+x),y_px=int((c['y']+c['h']/2)*h/288+y)) for c in cs]

def peak_choice(pool):
    return max(pool,key=lambda c:(c['component_peak'],c['bbox_area'],-c.get('tile_index',-1),-c['component_index'])) if pool else None

def prediction(g,c,method):
    return dict(**{k:g[k] for k in ['sample_id','candidate_id','local_frame','source_frame','truth_visibility','truth_x_px','truth_y_px']},
        method=method,x_px=c['x_px'] if c else 0,y_px=c['y_px'] if c else 0,visible=int(c is not None),
        confidence=c['bbox_mean'] if c else 0,component_peak=c['component_peak'] if c else 0,
        tile_index=c.get('tile_index',-1) if c else -1,component_index=c['component_index'] if c else -1)

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists(),'No post-result settings changes.'
    train_path=ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv';rows=read(train_path)
    assert len(rows)==123 and Counter(r['truth_visibility'] for r in rows)==Counter(visible=102,not_visible=21)
    split=load(ROOT/'balltrack_manual_finetune_pilot/data_manifest.json')
    assert {int(r['candidate_id'][-3:]) for r in rows}==set(split['train_clips'])
    context=load(BASE/'train_component_cache.json');expected_hashes=load(BASE/'train_context_image_hashes.json')
    strong=read(BASE/'train_displacement_0.1.csv');strong_by={r['sample_id']:r for r in strong}
    assert good_ids(strong) and len(good_ids(strong))==66 and err(metrics(strong))==4
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth'
    paths=[weight,train_path,BASE/'train_component_cache.json',BASE/'train_context_image_hashes.json',BASE/'train_displacement_0.1.csv',
        BASE/'train_bbox_area.csv',ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',
        SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv',ROOT.parent/'ball_quality_20261104/completed_review/manual_sequence_labels.csv',
        ROOT.parent/'ball_quality_20261104/frozen_predictions.csv',ROOT.parent/'ball_quality_20261104/evaluation_report.json']
    protected={str(p):sha(p) for p in paths};plans=[]
    for cid in sorted({r['candidate_id'] for r in rows}):
        group=[r for r in rows if r['candidate_id']==cid];keys=sorted(map(int,context[cid]));targets={int(r['local_frame']) for r in group}
        assert targets<=set(keys)
        batches=[keys[start:start+4] for start in sorted({keys.index(t)//4*4 for t in targets})]
        base=SOURCE/'data/balltrack/racketvision/match1'/cid
        mp=base/'runs/threshold030_median120/median.npz';protected[str(mp)]=sha(mp)
        assert all(r['median_path']==str(mp) and r['median_sha256']==sha(mp) for r in group)
        plans.append(dict(candidate_id=cid,targets=sorted(targets),batches=batches,median_path=str(mp),frame_dir=str(base/'frames')))
    protocol=dict(status='fixed_before_tiled_inference',date_jst='2026-11-09',training_only=True,labels=123,visible=102,not_visible=21,
        weights='Original TrackNet unchanged',threshold=.3,tiles_xywh=TILES,network_wh=[512,288],
        arms=dict(native='Same960x540 crop ->512x288 bilinear',degraded='Same960x540 crop ->256x144 ->512x288,both bilinear'),
        interpretation='Matched crop field of view and final object scale; down/up control tests lost spatial detail plus resampling. Native tiles versus full frame also change context, so not resolution alone.',
        model_input='BGRt-3..t plus the identically transformed median,/255,CUDA AMP,batch4. All9tiles for all labels,never GT-selected.',
        replay='For every training target, reproduce its original global4context batch and assert exact candidate dictionary equality; same time batch used per tile and arm.',
        pooling='All components strictly above0.3,no topk. Select highest component peak,then bbox area,then lower tile and component indexes. Duplicated overlap components are retained; candidate counts are not distinct balls.',
        coordinates='Actual tile bbox center int((x+w/2)*960/512+tile_x),same for y. No interpolation,oracle GT center or score threshold tuning.',
        controls='Saved original area and strongest frozen temporal66correct/4errors; also global peak-ranking control on the same original candidates.',
        policies='Raw peak selection is diagnostic. Primary candidate is missing-only fallback onto frozen temporal output, preserving every existing coordinate/confidence. Test native and degraded equally.',
        eligibility='Require all66old correct and errors<=4,strictly more correct. Choose max correct then fewer errors,ties native before degraded. If none,stop. No new validation or audit inference in this training-only experiment.',
        oracle='Candidate-center coverage within16px is an upper bound using training GT; never call it final detector accuracy.',
        labels_not_used='New52/oldvalidation/audit/challenge/short labels not read for fitting or comparison; new52files are hashed only for preservation.',
        storage='Exact half-precision values inside every threshold0.3component rectangle; enough to reconstruct all geometry and bbox/component scores at0.3. Subthreshold outside is not retained.',
        no_training=True,no_automatic_adoption=True,plans=plans,protected_input_hashes=protected,
        script_sha256=sha(Path(__file__)),helper_hashes={str(p):sha(p) for p in [ROOT/'compare_ball_candidate_decoders.py',ROOT/'calibrate_point_supervised_detector.py',ROOT/'run_manual_ball_finetune.py']})
    dump(OUT/'protocol.json',protocol)
    sys.path.insert(0,str(modelroot));from inference import BallInferencer
    torch.backends.cudnn.benchmark=False
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    infer.model.eval().requires_grad_(False)
    image_hashes={};candidate_cache={a:{r['sample_id']:[] for r in rows} for a in ARMS}
    idxs=[];values=[];offsets=[0];map_records=[];replay_log=[];cursor=0;started=time.monotonic();forward_calls=0
    for plan in plans:
        cid=plan['candidate_id'];fd=Path(plan['frame_dir']);mp=Path(plan['median_path']);median=np.load(mp)['median']
        assert median.shape[:2]==(1080,1920)
        medians={'global':np.moveaxis(cv2.resize(median,(512,288)).astype(np.float32)/255.,-1,0)}
        for arm in ARMS:
            for tile,box in enumerate(TILES):medians[arm+str(tile)]=np.moveaxis(transform(median,box,arm).astype(np.float32)/255.,-1,0)
        target_by={int(r['local_frame']):r['sample_id'] for r in rows if r['candidate_id']==cid}
        for bindex,times in enumerate(plan['batches']):
            images={}
            for j in sorted({j for t in times for j in range(t-3,t+1)}):
                p=fd/f'{j:06d}.jpg';h=sha(p);assert expected_hashes[str(p)]==h;image_hashes[str(p)]=h
                images[j]=cv2.imread(str(p));assert images[j].shape==(1080,1920,3)
            batch=[dict(frames=np.stack([cv2.resize(images[j],(512,288)) for j in range(t-3,t+1)]),median_key='global',row={}) for t in times]
            with torch.inference_mode(),torch.amp.autocast('cuda'):
                output,_,_=infer.model.forward(frames=tensors(batch,medians,infer.device))
            forward_calls+=1
            for t,hm in zip(times,output[:,0].float().cpu().numpy(),strict=True):
                assert components(hm)==context[cid][str(t)],(cid,t,'Global replay mismatch')
                replay_log.append(dict(candidate_id=cid,local_frame=t,batch_frames_json=json.dumps(times),original_components_match=True))
            for arm in ARMS:
                for tile,box in enumerate(TILES):
                    batch=[dict(frames=np.stack([transform(images[j],box,arm) for j in range(t-3,t+1)]),median_key=arm+str(tile),row={}) for t in times]
                    with torch.inference_mode(),torch.amp.autocast('cuda'):
                        output,_,_=infer.model.forward(frames=tensors(batch,medians,infer.device))
                    forward_calls+=1
                    for t,hm in zip(times,output[:,0].float().cpu().numpy(),strict=True):
                        if t not in target_by:continue
                        sid=target_by[t];cs=components(hm);converted=mapped(cs,tile);candidate_cache[arm][sid].extend(converted)
                        idx,v=compact(hm);assert components(expand(idx,v))==cs
                        idxs.append(idx);values.append(v);cursor+=len(idx);offsets.append(cursor)
                        map_records.append(dict(map_index=len(map_records),sample_id=sid,candidate_id=cid,local_frame=t,arm=arm,tile_index=tile,
                            batch_frames_json=json.dumps(times),source_component_count=len(cs),whole_map_peak=float(hm.max())))
            if (bindex+1)%5==0:print(cid,'original target batches',bindex+1,'/',len(plan['batches']),'forward calls',forward_calls,flush=True)
        print('Finished',cid,'elapsed_sec',round(time.monotonic()-started,1),flush=True)
    assert len(map_records)==123*2*9
    np.savez_compressed(OUT/'component_rectangles.npz',indices=np.concatenate(idxs),values=np.concatenate(values),offsets=np.array(offsets,dtype=np.int64))
    write(OUT/'map_records.csv',map_records);write(OUT/'global_replay.csv',replay_log)
    dump(OUT/'candidate_pools.json',candidate_cache);dump(OUT/'input_image_hashes.json',image_hashes)
    raw={};summaries=[];pool_coverage=[];base_ids=good_ids(strong)
    # All model outputs are now frozen. GT is used only in these training metrics.
    global_area=[prediction(g,choose(context[g['candidate_id']][g['local_frame']],'bbox_area'),'global_area') for g in rows]
    old_area={r['sample_id']:r for r in read(BASE/'train_bbox_area.csv')}
    assert all(all(float(r[k])==float(old_area[r['sample_id']][k]) for k in ['visible','x_px','y_px','confidence']) for r in global_area)
    raw['global_area']=global_area
    raw['global_peak']=[prediction(g,peak_choice(context[g['candidate_id']][g['local_frame']]),'global_peak') for g in rows]
    for arm in ARMS:raw[arm+'_peak']=[prediction(g,peak_choice(candidate_cache[arm][g['sample_id']]),arm+'_peak') for g in rows]
    eligible=[]
    for arm in ARMS:
        fallback=[]
        for g,proposal in zip(rows,raw[arm+'_peak'],strict=True):
            old=strong_by[g['sample_id']]
            q=dict(proposal,method=arm+'_missing_only')
            if int(old['visible']):
                q.update({k:old[k] for k in ['visible','x_px','y_px','confidence']});q.update(tile_index=-1,component_index=old['selected_component_index'])
                q['output_source']='unchanged_original_temporal'
            else:q['output_source']='observed_tile_candidate' if int(proposal['visible']) else 'missing'
            fallback.append(q)
        raw[arm+'_missing_only']=fallback
    for name,rs in raw.items():
        metric=metrics(rs);ids=good_ids(rs);lost=sorted(base_ids-ids);errors=err(metric)
        arm=name.split('_')[0] if name.startswith(('native','degraded')) else None
        upper=0
        for g in rows:
            pool=candidate_cache[arm][g['sample_id']] if arm else context[g['candidate_id']][g['local_frame']]
            near=False
            if g['truth_visibility']=='visible':near=any(math.hypot(c['x_px']-float(g['truth_x_px']),c['y_px']-float(g['truth_y_px']))<=16 for c in pool)
            upper+=int(near)
            pool_coverage.append(dict(method=name,sample_id=g['sample_id'],truth_visibility=g['truth_visibility'],component_count=len(pool),
                candidate_center_within16px=int(near) if g['truth_visibility']=='visible' else '',oracle_not_selected_accuracy=True))
        safe=not lost and errors<=4;strict=safe and len(ids)>66
        summary=dict(method=name,correct=len(ids),errors=errors,wrong_position=metric['counts']['wrong_localization'],
            output_when_not_observable=metric['counts']['presence_fp'],missing_visible=metric['counts']['presence_fn'],
            old_correct_lost=len(lost),lost_sample_ids=';'.join(lost),own_pool_center_upper_bound=upper,
            meets_safety=int(safe),strict_improvement=int(strict))
        summaries.append(summary);write(OUT/(name+'_predictions.csv'),rs)
        if name.endswith('_missing_only') and strict:eligible.append(summary)
    write(OUT/'training_summary.csv',summaries);write(OUT/'candidate_coverage.csv',pool_coverage)
    eligible.sort(key=lambda s:(-s['correct'],s['errors'],0 if s['method'].startswith('native') else 1))
    choice=eligible[0]['method'] if eligible else None
    assert all(sha(Path(p))==h for p,h in protected.items()) and all(sha(Path(p))==h for p,h in image_hashes.items())
    result=dict(status='training_only_fixed_comparison_completed',strong_original=dict(correct=66,errors=4),summary=summaries,
        chosen_training_proposal=choice,selection_uses_new52_or_validation_labels=False,new_training=False,
        inference_seconds=time.monotonic()-started,model_forward_calls=forward_calls,stored_tile_maps=len(map_records),
        original_context_outputs_reproduced=len(replay_log),new_validation_inference=False,canonical_tracking_changed=False,
        decision='Stop: no training-eligible tiled fallback.' if choice is None else 'Training-eligible proposal only; requires separate frozen validation protocol before any use.',
        protected_input_hashes=protected)
    dump(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='protected_input_hashes'},indent=2),flush=True)

if __name__=='__main__':main()
