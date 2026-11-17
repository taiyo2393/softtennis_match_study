"""Training-selected two-input candidate policies. No model/threshold fitting."""
import argparse,json,sys,math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2
from compare_ball_candidate_decoders import components,choose
import compare_temporal_candidate_paths as temporal
from evaluate_observed_ball_fallback import camera_mask
from run_manual_ball_finetune import metrics
from run_hard_case_ball_finetune import good_ids,err

OUT=ROOT/'ball_dual_stride_path_comparison';BASE=ROOT/'ball_temporal_path_comparison';PREV=ROOT/'ball_input_stride_comparison'
CFG=dict(name='displacement_0.1',order='displacement',strength=.1)
POLICIES=['union_path','missing_only_union_path','missing_only_stride2_path','missing_only_stride2_area']
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def contexts(path):return {cid:{int(t):cs for t,cs in times.items()} for cid,times in load(path).items()}
def pool(a,b):
    groups={}
    for stride,cs in [(1,a),(2,b)]:
        for c in cs:
            key=(c['x_px'],c['y_px']);item=dict(c,input_stride=stride,original_component_index=c['component_index'])
            groups.setdefault(key,[]).append(item)
    result=[]
    for key,observations in groups.items():
        winner=max(observations,key=lambda c:(c['component_peak'],-c['input_stride'],-c['original_component_index']))
        result.append(dict(winner,component_index=len(result),observed_in_strides=sorted({c['input_stride'] for c in observations})))
    return result

def pick(s1,s2,t,mask,policy):
    baseline,status=temporal.select(s1,t,mask,CFG)
    if policy is None:return baseline,'baseline:'+status
    if policy.startswith('missing_only') and baseline is not None:return baseline,'preserve_existing_output'
    times=sorted(j for j in s1 if j in {t+d for d in temporal.OFFSETS})
    if mask[min(times):max(times)+1].any():return baseline,'camera_guard_preserves_baseline'
    if policy=='missing_only_stride2_area':
        c=choose(s2[t],'bbox_area')
        return (dict(c,input_stride=2,original_component_index=c['component_index']) if c else None),'stride2_missing_fill' if c else 'still_missing'
    context={j:pool(s1[j],s2[j]) for j in times} if policy in ['union_path','missing_only_union_path'] else {j:s2[j] for j in times}
    chosen,how=temporal.select(context,t,mask,CFG)
    if how!='observed_path_selection':return baseline,'short_or_empty_path_preserves_baseline'
    if chosen is not None and policy=='missing_only_stride2_path':chosen=dict(chosen,input_stride=2,original_component_index=chosen['component_index'])
    return chosen,'observed_dual_path' if chosen is not None else 'still_missing'

def self_test():
    f=lambda x,peak=.8,index=0:dict(x_px=x,y_px=100,component_peak=peak,bbox_area=9,component_index=index,bbox_mean=peak)
    p=pool([f(100,.7)],[f(100,.9),f(500,.6,1)])
    assert len(p)==2 and p[0]['x_px']==100 and p[0]['input_stride']==2 and p[0]['observed_in_strides']==[1,2]
    assert all((c['x_px'],c['y_px']) in {(100,100),(500,100)} for c in p)
    mask=np.zeros(30,bool);a={8:[f(100)],10:[f(110)],12:[f(120)]};b={8:[f(800,.99)],10:[f(900,.99)],12:[f(1000,.99)]}
    for policy in POLICIES[1:]:assert pick(a,b,10,mask,policy)[0]['x_px']==110
    a[10]=[];b={8:[f(100)],10:[f(110)],12:[f(120)]}
    for policy in POLICIES:assert pick(a,b,10,mask,policy)[0]['x_px']==110
    mask[9]=True
    assert all(pick(a,b,10,mask,policy)[0] is None for policy in POLICIES)
    print('Passed exact-center pooling/provenance, no fabricated coordinates, preservation, actual missing fills and camera abstention. Synthetic data only.')

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Preserve fixed methods and training selection.'
    train_path=ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv';DATA=ROOT/'balltrack_manual_finetune_pilot';SHORT=ROOT/'ball_short_sequence_evaluation'
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth'
    protected={str(p):sha(p) for p in [train_path,DATA/'validation_labels.csv',DATA/'audit_labels.csv',weight,
        ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',
        SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv',ROOT/'ball_inplay_challenge/manual_center_labels.csv',
        ROOT/'ball_inplay_challenge/query_model_inputs.csv',SHORT/'manual_sequence_labels.csv',SHORT/'frozen_predictions.csv',SHORT/'query_model_inputs.csv',
        ROOT/'ball_040_object_swap_review/round2/user_reviewed/candidate_review.csv',PREV/'decision.json',PREV/'verification.json']}
    for role,cache,ref in [('train',BASE/'train_component_cache.json',BASE/'train_displacement_0.1.csv'),('validation',BASE/'validation_component_cache.json',BASE/'validation_selected.csv'),
        ('challenge',BASE/'challenge_component_cache.json',BASE/'challenge_selected.csv'),('short_sequence',SHORT/'frozen_component_cache.json',SHORT/'frozen_predictions.csv'),
        ('audit',BASE/'audit_component_cache.json',BASE/'audit_selected.csv')]:
        for p in [cache,ref]:protected[str(p)]=sha(p)
        if role!='audit':
            for p in [PREV/f'{role}_stride2_heatmaps.npz',PREV/f'{role}_stride2.csv']:protected[str(p)]=sha(p)
    for p in [ROOT/'temporal_rally_reviews_v2/match1_motion_pilot_015/full_scope_component_cache.json',PREV/'six_false_stride2_heatmaps.npz',PREV/'six_false_stride2.csv']:protected[str(p)]=sha(p)
    protocol=dict(status='fixed_before_dual_context_inference_or_training_selection',policies=POLICIES,
        inherited='Same original weights, threshold0.3, median+BGR512x288; causal target-aligned stride1/2 input; prior displacement0.1 remains numerically unchanged.',
        candidate_pool='All actual same-frame candidates from both inputs. Deduplicate ONLY exact source center; highest peak wins, then stride1. No coordinate average or tolerance tuning.',
        path='Old9observations offsets-8..8step2, emission-log(peak), .1*min(displacement/64,4); offline future8source frames. Component indices are local, not temporal identity.',
        policies_detail=dict(union_path='Select pooled path; cut or target segment<3 observations preserves previous stride1 temporal baseline.',
            missing_only_union_path='Keep every existing previous stride1 temporal output. Only fill missing target from pooled path if segment>=3 and no camera guard.',
            missing_only_stride2_path='Same existing-output preservation; fill from stride2-only path, only segment>=3 and no camera guard.',
            missing_only_stride2_area='Same existing-output preservation; fill from stride2 target max bbox area unless camera guard; intentionally unguarded by path support as comparator.'),
        baseline='Prior stride1 displacement0.1. Preserve all previously saved baseline coordinates after exact cache reproduction.',
        training_selection='123training labels only. Retain ALL66baseline-correct and errors<=4; maximise correct, then minimise errors, then changed outputs; exact ties baseline then listed policy order. No parameter sweep.',
        evaluation='Freeze ONE training-selected improving policy; validation120 then challenge4 and short26 separately, no substitute after failure. Report original-correct losses and extra errors.',
        six_false='Only chosen policy diagnostic on before-manual-removal inputs; never fit on6 labels. Missing-only policies cannot remove an existing false output, by construction.',
        audit='Only if selected policy passes all validation/challenge/short nonregression and improves validation; otherwise no audit.',
        known_data_warning='Design motivated by known same-match failures and repeated development data; training-only numeric choice is not independent external testing. Short26 overlaps4oldcenters and2challenge anchors.',
        no_retraining=True,no_new_human_labels=True,no_interpolation=True,automatic_adoption=False,
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    sys.path.insert(0,str(modelroot));from inference import BallInferencer
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    image_hashes={};inputs=[];medians={}
    def get(rows,role,old_path):
        s1=contexts(old_path);s2={};masks={};cached={}
        hp=PREV/f'{role}_stride2_heatmaps.npz';rp=PREV/f'{role}_stride2.csv'
        if hp.exists():
            hs=np.load(hp);rs=read(rp);assert list(hs['sample_ids'])==[r['sample_id'] for r in rs]
            cached={(r['candidate_id'],int(r['local_frame'])):hm.astype(np.float32) for r,hm in zip(rs,hs['heatmaps'])}
        reused=total=0
        for cid in sorted({r['candidate_id'] for r in rows}):
            targets=[int(r['local_frame']) for r in rows if r['candidate_id']==cid];times=sorted({j for t in targets for j in s1[cid] if j in {t+d for d in temporal.OFFSETS}})
            base=SOURCE/'data/balltrack/racketvision/match1'/cid;mp=base/'runs/threshold030_median120/median.npz'
            protected[str(mp)]=sha(mp)
            if str(mp) not in medians:medians[str(mp)]=np.moveaxis(cv2.resize(np.load(mp)['median'],(512,288)).astype(np.float32)/255.,-1,0)
            cp=ROOT.parent/'ball_quality_20261102/temporal_guard/camera_features'/f'{cid}.json';protected[str(cp)]=sha(cp);masks[cid]=camera_mask(load(cp)['changed_fraction'])
            s2[cid]={};todo=[]
            for t in times:
                if (cid,t) in cached:s2[cid][t]=components(cached[cid,t]);reused+=1
                else:todo.append(t)
            for start in range(0,len(todo),4):
                batch=[];bt=todo[start:start+4]
                for t in bt:
                    ims=[];paths=[];js=[max(0,t-6),max(0,t-4),max(0,t-2),t]
                    for j in js:
                        p=base/'frames'/f'{j:06d}.jpg'
                        if str(p) not in image_hashes:image_hashes[str(p)]=sha(p)
                        im=cv2.imread(str(p));assert im.shape==(1080,1920,3);ims.append(cv2.resize(im,(512,288)));paths.append(str(p))
                    inputs.append(dict(role=role,candidate_id=cid,local_frame=t,input_stride=2,input_frames=json.dumps(js),image_paths=json.dumps(paths),image_sha256=json.dumps([image_hashes[p] for p in paths])))
                    seq=np.moveaxis(np.stack(ims).astype(np.float32)/255.,-1,1);batch.append(np.concatenate([medians[str(mp)][None],seq]).reshape(15,288,512))
                with torch.inference_mode(),torch.amp.autocast('cuda'):
                    prediction,_,_=infer.model.forward(frames=torch.from_numpy(np.stack(batch)).to(infer.device))
                for t,hm in zip(bt,prediction[:,0].float().cpu().numpy(),strict=True):s2[cid][t]=components(hm)
            total+=len(times)
        dump(OUT/f'{role}_stride2_context_components.json',s2);dump(OUT/f'{role}_cache_record.json',dict(context_frames=total,reused_previous_target_maps=reused,new_context_frames=total-reused))
        print('CONTEXT',role,total,'reused',reused,flush=True);return s1,s2,masks
    def predict(rows,ctx,policy,role):
        s1,s2,masks=ctx;result=[]
        for r in rows:
            cid=r['candidate_id'];t=int(r['local_frame']);c,status=pick(s1[cid],s2[cid],t,masks[cid],policy)
            x,y=(c['x_px'],c['y_px']) if c else (0,0)
            if c:assert any((x,y)==(q['x_px'],q['y_px']) for q in s1[cid][t]+s2[cid][t])
            result.append(dict(sample_id=r['sample_id'],candidate_id=cid,local_frame=t,source_frame=r['source_frame'],role=role,method=policy or 'baseline',
                truth_visibility=r.get('truth_visibility',''),truth_x_px=r.get('truth_x_px',''),truth_y_px=r.get('truth_y_px',''),
                x_px=x,y_px=y,visible=int(c is not None),confidence=c['bbox_mean'] if c else 0.,decision=status,
                input_stride=c.get('input_stride',1) if c else '',interpolated=0))
        return result
    def reproduce(rows,ctx,ref,role):
        baseline=predict(rows,ctx,None,role);lookup={r['sample_id']:r for r in ref}
        assert all((int(r['visible']),float(r['x_px']),float(r['y_px']))==(int(lookup[r['sample_id']]['visible']),float(lookup[r['sample_id']]['x_px']),float(lookup[r['sample_id']]['y_px'])) for r in baseline),role
        write(OUT/f'{role}_baseline.csv',baseline);return baseline
    train=read(train_path);ctx=get(train,'train',BASE/'train_component_cache.json');baseline=reproduce(train,ctx,read(BASE/'train_displacement_0.1.csv'),'train');bm=metrics(baseline)
    assert len(good_ids(baseline))==66 and err(bm)==4
    chosen=None;best=(66,-4,0);grid=[]
    for policy in POLICIES:
        pred=predict(train,ctx,policy,'train');pm=metrics(pred);lost=sorted(good_ids(baseline)-good_ids(pred));changed=sum((a['visible'],a['x_px'],a['y_px'])!=(b['visible'],b['x_px'],b['y_px']) for a,b in zip(baseline,pred))
        eligible=not lost and err(pm)<=4;score=(len(good_ids(pred)),-err(pm),-changed)
        write(OUT/f'train_{policy}.csv',pred);grid.append(dict(policy=policy,correct=len(good_ids(pred)),errors=err(pm),correct_lost=len(lost),lost_ids=';'.join(lost),changed_outputs=changed,eligible=int(eligible)))
        if eligible and score>best:best=score;chosen=policy
    selection=dict(chosen=chosen,baseline=bm,grid=grid,validation_used=False,challenge_used=False,short_used=False,six_false_used=False,selection_frozen=True)
    write(OUT/'training_comparison.csv',grid);dump(OUT/'training_selection.json',selection);print('SELECT',json.dumps(selection),flush=True)
    result=dict(status='training_selection_complete',training=selection,validation=None,challenge=None,short_sequence=None,six_false=None,audit=None)
    def evaluate(rows,role,cache,ref):
        ctx=get(rows,role,cache);a=reproduce(rows,ctx,ref,role);b=predict(rows,ctx,chosen,role);write(OUT/f'{role}_selected.csv',b)
        am,pm=metrics(a),metrics(b);lost=sorted(good_ids(a)-good_ids(b));gains=sorted(good_ids(b)-good_ids(a))
        result[role]=dict(baseline=am,proposed=pm,correct_lost=lost,correct_gained=gains,nonregression=not lost and err(pm)<=err(am))
        return a,b
    if chosen:
        evaluate(read(DATA/'validation_labels.csv'),'validation',BASE/'validation_component_cache.json',read(BASE/'validation_selected.csv'))
        labels={r['sample_id']:r for r in read(ROOT/'ball_inplay_challenge/manual_center_labels.csv')}
        rows=[dict(r,**{k:labels[r['sample_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px']}) for r in read(ROOT/'ball_inplay_challenge/query_model_inputs.csv')]
        evaluate(rows,'challenge',BASE/'challenge_component_cache.json',read(BASE/'challenge_selected.csv'))
        labels={r['sample_id']:r for r in read(SHORT/'manual_sequence_labels.csv')}
        rows=[dict(r,**{k:labels[r['sample_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px']}) for r in read(SHORT/'query_model_inputs.csv')]
        reference=[dict(r,**{k:labels[r['sample_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px']}) for r in read(SHORT/'frozen_predictions.csv') if r['method']=='displacement_0.1']
        evaluate(rows,'short_sequence',SHORT/'frozen_component_cache.json',reference)
        false=read(ROOT/'ball_040_object_swap_review/round2/user_reviewed/candidate_review.csv');rows=[dict(sample_id=f'match1_motion_pilot_015_frame_{int(r["local_frame"]):06d}',candidate_id='match1_motion_pilot_015',local_frame=r['local_frame'],source_frame=r['source_frame']) for r in false]
        ctx=get(rows,'six_false',ROOT/'temporal_rally_reviews_v2/match1_motion_pilot_015/full_scope_component_cache.json');before=predict(rows,ctx,None,'six_false');after=predict(rows,ctx,chosen,'six_false');outcomes=[]
        for r,a,b in zip(false,before,after):
            assert (float(a['x_px']),float(a['y_px']))==(float(r['displayed_x_px']),float(r['displayed_y_px']))
            same=bool(int(b['visible']) and (b['x_px'],b['y_px'])==(a['x_px'],a['y_px']))
            state='same_known_false_point' if same else 'different_unverified_candidate' if b['visible'] else 'position_unknown'
            outcomes.append(dict(local_frame=r['local_frame'],known_false_x=r['displayed_x_px'],known_false_y=r['displayed_y_px'],proposed_x=b['x_px'],proposed_y=b['y_px'],output_present=b['visible'],status=state,true_center='unknown',physical_visibility='not_labelled'))
        write(OUT/'six_false_outcomes.csv',outcomes);result['six_false']=dict(status_counts=dict(Counter(r['status'] for r in outcomes)),missing_only_policy_cannot_remove_original_errors=chosen.startswith('missing_only'))
        passes=all(result[role]['nonregression'] for role in ['validation','challenge','short_sequence'])
        if passes and (len(result['validation']['correct_gained'])>0 or err(result['validation']['proposed'])<err(result['validation']['baseline'])):
            evaluate(read(DATA/'audit_labels.csv'),'audit',BASE/'audit_component_cache.json',read(BASE/'audit_selected.csv'))
    if inputs:write(OUT/'new_context_input_manifest.csv',inputs)
    dump(OUT/'new_context_image_hashes.json',image_hashes)
    assert all(sha(Path(p))==h for p,h in protected.items()) and all(sha(Path(p))==h for p,h in image_hashes.items())
    result.update(status='dual_input_training_selected_trial_complete',protected_input_hashes=protected,source_images_unchanged=True,
        automatic_adoption=False,operational_outputs_changed=False,new_labels=0,interpolation=0,independent_generalization_established=False)
    dump(OUT/'decision.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    else:main()
