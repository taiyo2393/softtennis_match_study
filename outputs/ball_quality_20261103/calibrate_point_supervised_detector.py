"""Training-only threshold selection on fixed point-trained final models.

Preserve only original floor0.3 component rectangles: exact decoding at>=0.3.
"""
import argparse,json,sys,math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,read,load,write,sha,cv2
import compare_temporal_candidate_paths as temporal
from compare_ball_candidate_decoders import components as original_components
from run_manual_ball_finetune import metrics
from run_hard_case_ball_finetune import good_ids,err

OUT=ROOT/'ball_point_detector_calibration';MODELS=ROOT/'ball_point_supervised_detector_fp32';BASE=ROOT/'ball_temporal_path_comparison';DATA=ROOT/'balltrack_manual_finetune_pilot';SHORT=ROOT/'ball_short_sequence_evaluation'
THRESHOLDS=[.3,.4,.5,.6,.7,.8,.9,.95];ARMS=['positive_control','point_negative'];CFG=dict(name='displacement_0.1',order='displacement',strength=.1)
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def components_at(hm,threshold):
    binary=(hm>threshold).astype('uint8');contours,_=cv2.findContours(binary*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);out=[]
    for i,contour in enumerate(contours):
        x,y,w,h=cv2.boundingRect(contour);mask=np.zeros_like(binary);cv2.drawContours(mask,[contour],-1,1,-1);pixels=hm[(mask!=0)&(binary!=0)]
        out.append(dict(component_index=i,x=x,y=y,w=w,h=h,bbox_area=w*h,component_peak=float(pixels.max()),bbox_mean=float(hm[y:y+h,x:x+w].mean()),
            component_mean=float(pixels.mean()),component_pixels=len(pixels),x_px=int((x+w/2)*1920/512),y_px=int((y+h/2)*1080/288)))
    return out
def compact(hm):
    keep=np.zeros(hm.shape,bool)
    for c in original_components(hm):keep[c['y']:c['y']+c['h'],c['x']:c['x']+c['w']]=True
    idx=np.flatnonzero(keep).astype(np.int32);values=hm.ravel()[idx].astype(np.float16)
    assert np.array_equal(values.astype(np.float32),hm.ravel()[idx]),'Do not silently quantize a non-AMP heatmap.'
    return idx,values
def expand(idx,values):
    hm=np.zeros((288,512),np.float32);hm.ravel()[idx]=values;return hm
def self_test():
    rng=np.random.default_rng(20261103)
    for _ in range(3):
        hm=rng.uniform(0,.25,(288,512)).astype(np.float16).astype(np.float32)
        hm[30:60,40:100]=rng.uniform(.25,1,(30,60)).astype(np.float16).astype(np.float32)
        hm[150:165,300:320]=.75;hm[155:160,305:309]=np.float16(.1)
        idx,values=compact(hm);recon=expand(idx,values)
        assert original_components(hm)==components_at(hm,.3)
        for threshold in THRESHOLDS:assert components_at(hm,threshold)==components_at(recon,threshold)
    hm=np.full((288,512),np.float16(.25),np.float32);idx,v=compact(hm);assert len(idx)==0 and all(not components_at(expand(idx,v),t) for t in THRESHOLDS)
    print('Passed exact floor0.3 reproduction and exact high-threshold geometry/means after compact reconstruction, including holes, split components and empty maps. Synthetic maps only.')

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Preserve threshold trial; no extra grid after results.'
    models=load(MODELS/'trained_models.json');weight=SOURCE/'third_party/RacketVision/source/BallTrack/checkpoints/balltrack_best.pth';modelroot=weight.parent.parent
    train_path=ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv'
    paths=[weight,MODELS/'protocol.json',MODELS/'trained_models.json',MODELS/'decision.json',MODELS/'verification.json',train_path,
        DATA/'validation_labels.csv',DATA/'audit_labels.csv',BASE/'train_displacement_0.1.csv',BASE/'validation_selected.csv',BASE/'challenge_selected.csv',BASE/'audit_selected.csv',
        ROOT/'ball_inplay_challenge/manual_center_labels.csv',ROOT/'ball_inplay_challenge/query_model_inputs.csv',SHORT/'manual_sequence_labels.csv',SHORT/'frozen_predictions.csv',SHORT/'query_model_inputs.csv',
        ROOT/'ball_040_object_swap_review/round2/user_reviewed/candidate_review.csv',ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',
        SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv']
    for arm in ARMS:
        p=Path(models['checkpoints'][arm]['path']);assert sha(p)==models['checkpoints'][arm]['sha256'];paths.append(p)
        paths.extend([MODELS/f'{arm}_train_temporal.csv',MODELS/f'{arm}_train_component_cache.json'])
    protected={str(p):sha(p) for p in paths}
    protocol=dict(status='fixed_before_new_heatmap_extraction_or_threshold_selection',
        motivation='The fixed0.3 point-trained trial overactivated regions. New threshold-only development study; old failed protocol/results remain intact.',
        models='Only the two already fixed final epoch3 deltas; exclude diagnostic-only9image model. No retraining/epoch choice.',arms=ARMS,thresholds=THRESHOLDS,
        decoder='Original contours,bbox centers and fixed displacement0.1. Only binary threshold changes. Causal t-3..t inputs, offline selector future8sourceframes.',
        training='123training labels only. Include original strongest baseline66correct,4errors. Feasible iff all66correct retained and errors<=4. Max correct,then min errors; exact ties baseline then control then listed threshold order.',
        primary_claim='No requirement to prefer negative-point model. If control selected, do not attribute benefit to9negative labels.',
        evaluation='Freeze ONE strictly improving feasible model/threshold before validation. If no improvement feasible, stop before all new validation/challenge/short/six/audit inference. Never substitute after failure.',
        audit='Only one chosen setting and only if validation/challenge/short lose no original correct, add no errors, and improve validation.',
        preservation='No new teacher labels or positions. Output only observed same-frame component centers. Empty means unknown, not physical absence.',
        cache='Keep full original probability values inside union of ALLfloor0.3component bounding rectangles, including subthreshold values inside rectangles. This preserves exact candidates AND bbox means for thresholds>=0.3. Zero outside.',
        scope='Known same-match repeatedly inspected development data. Numeric choice is training-only, not independent external validation. Short26 contains4reused centers and2challenge anchors.',
        automatic_adoption=False,new_human_labels=0,interpolation=0,protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    sys.path.insert(0,str(modelroot));from inference import BallInferencer
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4);net=infer.model;initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};net.eval().requires_grad_(False)
    def load_arm(arm):
        delta=torch.load(models['checkpoints'][arm]['path'],map_location='cpu',weights_only=False);assert delta['meta']['source_weight_sha256']==sha(weight)
        state=dict(initial);state.update(delta['trainable_state_dict']);net.load_state_dict(state,strict=True);net.eval()
    def collect(rows,role,arm):
        jobs=[]
        for cid in sorted({r['candidate_id'] for r in rows}):
            p=SOURCE/'data/balltrack/racketvision/match1'/cid/'runs/threshold030_median120/run_metadata.json';protected[str(p)]=sha(p);meta=load(p)
            times=sorted({int(r['local_frame'])+d for r in rows if r['candidate_id']==cid for d in temporal.OFFSETS if 3<=int(r['local_frame'])+d<meta['frame_count']})
            jobs.extend((cid,t) for t in times)
        indices=[];values=[];offsets=[0];cursor=0
        def capture(hm):
            nonlocal cursor
            idx,v=compact(hm);cs=original_components(hm)
            assert cs==original_components(expand(idx,v))
            indices.append(idx);values.append(v);offsets.append(offsets[-1]+len(idx));cursor+=1;return cs
        original=temporal.components;temporal.components=capture;temporal.OUT=OUT
        try:floor,masks=temporal.get_contexts(infer,rows,arm+'_'+role,protected)
        finally:temporal.components=original
        assert cursor==len(jobs)
        cache=OUT/f'{arm}_{role}_compact_heatmaps.npz'
        np.savez_compressed(cache,clips=np.array([j[0] for j in jobs]),local_frames=np.array([j[1] for j in jobs]),offsets=np.array(offsets,dtype=np.int64),
            indices=np.concatenate(indices),values=np.concatenate(values))
        return jobs,indices,values,masks,floor
    def decode_context(jobs,indices,values,threshold):
        result={}
        for (cid,t),idx,v in zip(jobs,indices,values):result.setdefault(cid,{})[t]=components_at(expand(idx,v),threshold)
        return result
    train=read(train_path);baseline=read(BASE/'train_displacement_0.1.csv');bm=metrics(baseline);original_good=good_ids(baseline);assert len(original_good)==66 and err(bm)==4
    best=(66,-4);chosen=None;grid=[];reproduction=[]
    for arm in ARMS:
        load_arm(arm);jobs,idx,values,masks,floor=collect(train,'train',arm)
        previous={r['sample_id']:r for r in read(MODELS/f'{arm}_train_temporal.csv')}
        for threshold in THRESHOLDS:
            ctx=decode_context(jobs,idx,values,threshold);pred=temporal.predictions(train,ctx,masks,CFG,'train')
            for r in pred:r['arm']=arm;r['threshold']=threshold
            if threshold==.3:
                for r in pred:
                    old=previous[r['sample_id']];same=(r['x_px'],r['y_px'],r['visible'])==(int(old['x_px']),int(old['y_px']),int(old['visible']))
                    reproduction.append(dict(arm=arm,sample_id=r['sample_id'],same=int(same)));assert same,(arm,r['sample_id'])
            name=f'{arm}_train_threshold{round(threshold*100):02d}.csv';write(OUT/name,pred)
            m=metrics(pred);lost=sorted(original_good-good_ids(pred));feasible=not lost and err(m)<=4
            grid.append(dict(arm=arm,threshold=threshold,correct=len(good_ids(pred)),errors=err(m),correct_lost=len(lost),lost_sample_ids=';'.join(lost),feasible=int(feasible),missing_visible=m['counts']['presence_fn']))
            score=(len(good_ids(pred)),-err(m))
            if feasible and score>best:best=score;chosen=dict(arm=arm,threshold=threshold,metrics=m,predictions_file=name,predictions_sha256=sha(OUT/name))
    write(OUT/'training_threshold_grid.csv',grid);write(OUT/'threshold03_reproduction.csv',reproduction)
    selection=dict(chosen=chosen,training_baseline=bm,grid=grid,validation_used=False,challenge_used=False,short_used=False,six_false_used=False,audit_used=False,choice_frozen=True)
    dump(OUT/'training_selection.json',selection);print('TRAIN SELECTION',json.dumps(selection),flush=True)
    result=dict(status='training_only_threshold_selection_complete',training=selection,validation=None,challenge=None,short_sequence=None,six_false=None,audit=None)
    def evaluate(rows,role,reference):
        arm=chosen['arm'];jobs,idx,values,masks,_=collect(rows,role,arm);ctx=decode_context(jobs,idx,values,chosen['threshold']);pred=temporal.predictions(rows,ctx,masks,CFG,role)
        for r in pred:r['arm']=arm;r['threshold']=chosen['threshold']
        write(OUT/f'{role}_selected.csv',pred);pm=metrics(pred);old=metrics(reference);lost=sorted(good_ids(reference)-good_ids(pred));gained=sorted(good_ids(pred)-good_ids(reference))
        result[role]=dict(baseline=old,proposed=pm,original_correct_lost=lost,original_correct_gained=gained,nonregression=not lost and err(pm)<=err(old))
        return pred
    if chosen:
        load_arm(chosen['arm']);evaluate(read(DATA/'validation_labels.csv'),'validation',read(BASE/'validation_selected.csv'))
        gt={r['sample_id']:r for r in read(ROOT/'ball_inplay_challenge/manual_center_labels.csv')};rows=[dict(r,**{k:gt[r['sample_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px']}) for r in read(ROOT/'ball_inplay_challenge/query_model_inputs.csv')]
        evaluate(rows,'challenge',read(BASE/'challenge_selected.csv'))
        gt={r['sample_id']:r for r in read(SHORT/'manual_sequence_labels.csv')};rows=[dict(r,**{k:gt[r['sample_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px']}) for r in read(SHORT/'query_model_inputs.csv')]
        reference=[dict(r,**{k:gt[r['sample_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px']}) for r in read(SHORT/'frozen_predictions.csv') if r['method']=='displacement_0.1'];evaluate(rows,'short_sequence',reference)
        false=read(ROOT/'ball_040_object_swap_review/round2/user_reviewed/candidate_review.csv');template=next(r for r in read(DATA/'validation_labels.csv') if r['candidate_id']=='match1_motion_pilot_015');rows=[]
        for r in false:
            t=int(r['local_frame']);fd=SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_015/frames';ps=[fd/f'{j:06d}.jpg' for j in range(t-3,t+1)]
            rows.append(dict(sample_id=f'match1_motion_pilot_015_frame_{t:06d}',candidate_id='match1_motion_pilot_015',local_frame=t,source_frame=r['source_frame'],
                input_frames_json=json.dumps([str(p) for p in ps]),input_frame_sha256_json=json.dumps([sha(p) for p in ps]),median_path=template['median_path'],median_sha256=template['median_sha256'],truth_visibility='unlabelled',truth_x_px='',truth_y_px=''))
        jobs,idx,values,masks,_=collect(rows,'six_false',chosen['arm']);ctx=decode_context(jobs,idx,values,chosen['threshold']);pred=temporal.predictions(rows,ctx,masks,CFG,'six_false');outcomes=[]
        for f,r in zip(false,pred):
            d=math.hypot(float(r['x_px'])-float(f['displayed_x_px']),float(r['y_px'])-float(f['displayed_y_px'])) if int(r['visible']) else None
            status='position_unknown' if not int(r['visible']) else 'same_known_false_point' if d==0 else 'near_known_false_point_unverified' if d<=16 else 'different_unverified_candidate'
            outcomes.append(dict(local_frame=r['local_frame'],x_px=r['x_px'],y_px=r['y_px'],visible=r['visible'],status=status,true_center='unknown',physical_visibility='not_labelled'))
        write(OUT/'six_false_outcomes.csv',outcomes);result['six_false']=dict(Counter(r['status'] for r in outcomes))
        passes=all(result[role]['nonregression'] for role in ['validation','challenge','short_sequence'])
        benefit=len(result['validation']['original_correct_gained'])>0 or err(result['validation']['proposed'])<2
        if passes and benefit:evaluate(read(DATA/'audit_labels.csv'),'audit',read(BASE/'audit_selected.csv'))
    assert all(sha(Path(p))==h for p,h in protected.items())
    result.update(status='fixed_model_training_calibration_trial_complete',protected_input_hashes=protected,protected_inputs_unchanged=True,
        model_weights_unchanged=True,retraining=False,new_human_labels=0,interpolation=0,automatic_adoption=False,operational_outputs_changed=False,independent_generalization_established=False)
    dump(OUT/'decision.json',result);print(json.dumps({k:v for k,v in result.items() if k!='protected_input_hashes'},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    else:main()
