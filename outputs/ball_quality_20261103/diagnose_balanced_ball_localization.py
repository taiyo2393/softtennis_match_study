"""TRAIN-ONLY response/candidate/selection diagnosis on frozen balanced models.

No retraining or evaluation-role application of diagnostic checkpoints.
"""
import argparse,json,sys,math
from pathlib import Path
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2
import compare_temporal_candidate_paths as temporal
from compare_ball_candidate_decoders import components,choose
from calibrate_point_supervised_detector import components_at,expand
from run_manual_ball_finetune import prepare,tensors,metrics
from run_hard_case_ball_finetune import good_ids,err

OUT=ROOT/'ball_balanced_localization_diagnosis';PARENT=ROOT/'ball_point_training_factor_diagnosis';BASE=ROOT/'ball_temporal_path_comparison'
CONDITIONS=['balanced_preserve','balanced_no_preserve'];THRESHOLDS=[.1,.2,.3,.4,.5,.6,.7,.8,.9,.95];CFG=dict(name='displacement_0.1',order='displacement',strength=.1)
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def compact_floor(hm):
    keep=np.zeros((288,512),bool)
    for c in components_at(hm,.1):keep[c['y']:c['y']+c['h'],c['x']:c['x']+c['w']]=True
    idx=np.flatnonzero(keep).astype(np.int32);v=hm.ravel()[idx].astype(np.float16)
    assert np.array_equal(v.astype(np.float32),hm.ravel()[idx]);return idx,v
def activation_near(hm,x,y,threshold):
    qx=float(x)/3.75;qy=float(y)/3.75;r=16/3.75
    x0=max(0,math.floor(qx-r));x1=min(512,math.ceil(qx+r)+1);y0=max(0,math.floor(qy-r));y1=min(288,math.ceil(qy+r)+1)
    yy,xx=np.mgrid[y0:y1,x0:x1];circle=(xx-qx)**2+(yy-qy)**2<=r*r
    return bool(((hm[y0:y1,x0:x1]>threshold)&circle).any())
def self_test():
    rng=np.random.default_rng(20261103)
    hm=rng.uniform(0,.08,(288,512)).astype(np.float16).astype(np.float32)
    hm[30:55,40:80]=rng.uniform(.08,1,(25,40)).astype(np.float16).astype(np.float32);hm[40:43,50:54]=np.float16(.03)
    hm[150:154,300:304]=np.float16(.75);idx,v=compact_floor(hm);recon=expand(idx,v)
    for t in THRESHOLDS:
        assert components_at(hm,t)==components_at(recon,t)
        for x,y in [(180,160),(1129,568),(0,0),(1919,1079)]:assert activation_near(hm,x,y,t)==activation_near(recon,x,y,t)
    hm=np.zeros((288,512),np.float32);hm[100,100]=1
    assert activation_near(hm,375,375,.3) and activation_near(hm,390,375,.3) and not activation_near(hm,392,375,.3)
    print('Passed exact candidate preservation at>=0.1 and true-center16px activation geometry, including holes and image edges. Synthetic maps only.')
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
    parent=load(PARENT/'result.json');train_path=ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv';rows=read(train_path)
    source=SOURCE/'third_party/RacketVision/source/BallTrack';weight=source/'checkpoints/balltrack_best.pth'
    paths=[weight,train_path,PARENT/'protocol.json',PARENT/'result.json',PARENT/'verification.json',BASE/'train_displacement_0.1.csv',BASE/'train_bbox_area.csv',
        ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv']
    for condition in CONDITIONS:paths.extend([Path(parent['models'][condition]['path']),PARENT/f'{condition}_training123_raw.csv'])
    protected={str(p):sha(p) for p in paths}
    protocol=dict(status='fixed_before_balanced_localization_inference',purpose='Training-only localization diagnosis of two frozen balanced diagnostic models. Not model selection for external evaluation or deployment.',
        conditions=CONDITIONS,thresholds=THRESHOLDS,threshold_reason='Balanced training lowered many responses; include predeclared0.1/0.2 as well as earlier0.3..0.95 before inspecting new maps.',
        input='Original causal median+BGRt-3..t,512x288. Production-style CUDA AMP for all maps. No new weights.',
        precision_replay='Separately repeat FP32 target-only raw probe to verify prior factor diagnostic. AMP vsFP32 differences are recorded and not claimed as model improvement.',
        decoders=['max_bbox_area','existing_displacement0.1'],selector_context='Old offline offsets-8..8stride2,cut/short-gap behavior unchanged; future8source frames.',
        causal_categories='For human-visible ball: selected correct16px; candidate center<=16px but not selected; response>threshold within16px source circle but no component center<=16px; no response>threshold within that circle.',
        category_caution='Activation and candidate availability are not observed true-ball recovery. Nonvisible labels have no true center; no circle test or physical-absence inference.',
        cache='All pixels within bounding rectangles of floor0.1components, including values belowfloor inside rectangles; exact atthreshold>=0.1.',
        diagnostics='All20settings on same123TRAINING labels. Report original strongest66/4 correctness preservation, tradeoffs and candidate limits. No validation/challenge/short/six/audit inference.',
        followup='Any train-feasible setting remains a diagnostic proposal requiring a separately fixed independent evaluation; no automatic promotion of diagnostic checkpoints.',
        no_retraining=True,no_new_human_labels=True,no_model_selection_for_deployment=True,no_interpolation=True,
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    sys.path.insert(0,str(source));from inference import BallInferencer
    infer=BallInferencer(str(source/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4);net=infer.model
    net.eval().requires_grad_(False);initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    baseline=read(BASE/'train_displacement_0.1.csv');good=good_ids(baseline);assert len(good)==66 and err(metrics(baseline))==4
    medians={};samples=prepare(rows,medians);summaries=[];mechanisms=[];replays=[];cache_records=[]
    for condition in CONDITIONS:
        record=parent['models'][condition];assert sha(Path(record['path']))==record['sha256'];delta=torch.load(record['path'],map_location='cpu',weights_only=False)
        assert delta['meta']['diagnostic_only'];state=dict(initial);state.update(delta['trainable_state_dict']);net.load_state_dict(state,strict=True);net.eval()
        old={r['sample_id']:r for r in read(PARENT/f'{condition}_training123_raw.csv')}
        with torch.inference_mode():
            for first in range(0,len(samples),4):
                batch=samples[first:first+4];p,_,_=net.forward(frames=tensors(batch,medians,infer.device))
                for r,hm in zip([s['row'] for s in batch],p[:,0].cpu().numpy()):
                    c=choose(components(hm),'bbox_area');x,y=(c['x_px'],c['y_px']) if c else (0,0);a=old[r['sample_id']]
                    same=(x,y,int(c is not None))==(int(a['x_px']),int(a['y_px']),int(a['visible']))
                    replays.append(dict(condition=condition,sample_id=r['sample_id'],same=int(same),old_x=a['x_px'],old_y=a['y_px'],new_x=x,new_y=y));assert same
        jobs=[]
        for cid in sorted({r['candidate_id'] for r in rows}):
            p=SOURCE/'data/balltrack/racketvision/match1'/cid/'runs/threshold030_median120/run_metadata.json';protected[str(p)]=sha(p);meta=load(p)
            times=sorted({int(r['local_frame'])+d for r in rows if r['candidate_id']==cid for d in temporal.OFFSETS if 3<=int(r['local_frame'])+d<meta['frame_count']});jobs.extend((cid,t) for t in times)
        idxs=[];vals=[];offsets=[0];cursor=0
        def capture(hm):
            nonlocal cursor
            idx,v=compact_floor(hm);recon=expand(idx,v);assert components(hm)==components(recon)
            idxs.append(idx);vals.append(v);offsets.append(offsets[-1]+len(idx));cursor+=1;return components(hm)
        original=temporal.components;temporal.components=capture;temporal.OUT=OUT
        try:_,masks=temporal.get_contexts(infer,rows,condition+'_train',protected)
        finally:temporal.components=original
        assert cursor==len(jobs)
        cache=OUT/f'{condition}_compact_maps.npz';np.savez_compressed(cache,clips=np.array([cid for cid,t in jobs]),frames=np.array([t for cid,t in jobs]),
            offsets=np.array(offsets,dtype=np.int64),indices=np.concatenate(idxs),values=np.concatenate(vals))
        cache_records.append(dict(condition=condition,path=str(cache),sha256=sha(cache),context_frames=len(jobs)))
        hm_by={(cid,t):expand(idx,v) for (cid,t),idx,v in zip(jobs,idxs,vals)}
        for threshold in THRESHOLDS:
            ctx={}
            for (cid,t),hm in hm_by.items():ctx.setdefault(cid,{})[t]=components_at(hm,threshold)
            raw=temporal.predictions(rows,ctx,masks,None,'training_diagnostic');selected=temporal.predictions(rows,ctx,masks,CFG,'training_diagnostic')
            for method,pred in [('raw',raw),('temporal',selected)]:
                for r in pred:r['condition']=condition;r['threshold']=threshold
                write(OUT/f'{condition}_threshold{round(threshold*100):02d}_{method}.csv',pred)
                m=metrics(pred);lost=sorted(good-good_ids(pred));counts=dict(correct=0,selection_failure=0,response_without_localized_center=0,no_near_response=0)
                for truth,r in zip(rows,pred):
                    cid=truth['candidate_id'];t=int(truth['local_frame']);cs=ctx[cid][t]
                    if truth['truth_visibility']!='visible':
                        mechanisms.append(dict(condition=condition,threshold=threshold,method=method,sample_id=truth['sample_id'],truth_visibility=truth['truth_visibility'],
                            category='output_on_nonvisible_label' if int(r['visible']) else 'no_output_on_nonvisible_label',candidate_count=len(cs),nearest_center_error_px='',activation_near_truth='',
                            prior_correct_lost=int(truth['sample_id'] in lost)));continue
                    dist=min((math.hypot(c['x_px']-float(truth['truth_x_px']),c['y_px']-float(truth['truth_y_px'])) for c in cs),default=None)
                    activated=activation_near(hm_by[cid,t],truth['truth_x_px'],truth['truth_y_px'],threshold)
                    right=bool(int(r['visible']) and math.hypot(float(r['x_px'])-float(truth['truth_x_px']),float(r['y_px'])-float(truth['truth_y_px']))<=16)
                    category='correct' if right else 'selection_failure' if dist is not None and dist<=16 else 'response_without_localized_center' if activated else 'no_near_response'
                    counts[category]+=1;mechanisms.append(dict(condition=condition,threshold=threshold,method=method,sample_id=truth['sample_id'],truth_visibility='visible',category=category,
                        candidate_count=len(cs),nearest_center_error_px=dist,activation_near_truth=int(activated),prior_correct_lost=int(truth['sample_id'] in lost)))
                assert sum(counts.values())==102
                summaries.append(dict(condition=condition,threshold=threshold,method=method,correct=len(good_ids(pred)),errors=err(m),missing_visible=m['counts']['presence_fn'],
                    original_correct_lost=len(lost),lost_sample_ids=';'.join(lost),candidate_center_upper_bound=counts['correct']+counts['selection_failure'],
                    **{k:counts[k] for k in ['selection_failure','response_without_localized_center','no_near_response']},
                    meets_original_training_safety=int(not lost and err(m)<=4),strict_training_improvement=int(not lost and err(m)<=4 and (len(good_ids(pred))>66 or err(m)<4))))
        print('DIAGNOSED',condition,flush=True)
    write(OUT/'training_localization_summary.csv',summaries);write(OUT/'training_failure_mechanisms.csv',mechanisms);write(OUT/'fp32_probe_reproduction.csv',replays)
    assert all(sha(Path(path))==h for path,h in protected.items())
    result=dict(status='frozen_balanced_training_localization_diagnosis_complete',summary=summaries,cache=cache_records,
        original_baseline=dict(correct=66,errors=4),train_safe_temporal_settings=[r for r in summaries if r['method']=='temporal' and r['meets_original_training_safety']],
        strict_train_improving_temporal_settings=[r for r in summaries if r['method']=='temporal' and r['strict_training_improvement']],
        fp32_prior_probe_reproduced=len(replays),protected_input_hashes=protected,protected_inputs_unchanged=True,
        no_validation_or_audit=True,no_best_model_selection_for_deployment=True,retraining=False,new_human_labels=0,automatic_adoption=False,operational_outputs_changed=False,
        interpretation='Activation availability, localized candidate centers, selected positions and same-data fit are distinct. Results only diagnose these trained8clips, not independent accuracy.')
    dump(OUT/'result.json',result)
    print(json.dumps([r for r in summaries if r['method']=='temporal'],indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    else:main()
