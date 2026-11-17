"""Fixed final checkpoints, same frozen temporal decoder; no selection grid."""
import json,sys,math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha
import compare_temporal_candidate_paths as temporal
from run_manual_ball_finetune import prepare,tensors,metrics
from run_hard_case_ball_finetune import good_ids,err
from train_point_supervised_ball_detector_fp32 import OUT,point_value

BASE=ROOT/'ball_temporal_path_comparison';DATA=ROOT/'balltrack_manual_finetune_pilot';SHORT=ROOT/'ball_short_sequence_evaluation';CFG=dict(name='displacement_0.1',order='displacement',strength=.1)
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    assert not (OUT/'evaluation_protocol.json').exists(),'Preserve fixed evaluation; no model substitution after results.'
    protocol=load(OUT/'protocol.json');trained=load(OUT/'trained_models.json')
    assert trained['status']=='two_final_epoch3_models_frozen_before_evaluation'
    protected=dict(protocol['protected_input_hashes'])
    for p in [OUT/'protocol.json',OUT/'trained_models.json',OUT/'training_history.json',OUT/'positive_training_manifest.csv',OUT/'negative_point_manifest.csv']:
        protected[str(p)]=sha(p)
    for arm,c in trained['checkpoints'].items():assert sha(Path(c['path']))==c['sha256'];protected[c['path']]=c['sha256']
    train=read(ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv');val=read(DATA/'validation_labels.csv')
    gt={r['sample_id']:r for r in read(ROOT/'ball_inplay_challenge/manual_center_labels.csv')}
    challenge=[dict(r,**{k:gt[r['sample_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px']}) for r in read(ROOT/'ball_inplay_challenge/query_model_inputs.csv')]
    gt={r['sample_id']:r for r in read(SHORT/'manual_sequence_labels.csv')}
    short=[dict(r,**{k:gt[r['sample_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px']}) for r in read(SHORT/'query_model_inputs.csv')]
    refs=dict(train=read(BASE/'train_displacement_0.1.csv'),validation=read(BASE/'validation_selected.csv'),challenge=read(BASE/'challenge_selected.csv'),
        short_sequence=[dict(r,**{k:gt[r['sample_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px']}) for r in read(SHORT/'frozen_predictions.csv') if r['method']=='displacement_0.1'])
    for role,p in [('train',BASE/'train_displacement_0.1.csv'),('validation',BASE/'validation_selected.csv'),('challenge',BASE/'challenge_selected.csv'),('short_sequence',SHORT/'frozen_predictions.csv')]:protected[str(p)]=sha(p)
    ep=dict(status='specified_before_final_model_evaluation',primary='point_negative_epoch3',control='positive_control_epoch3',
        selection='None: both final checkpoints, original threshold0.3 and old displacement0.1; no new values chosen.',
        roles=['train123','validation120','challenge4','short26'],role_counts_not_combined=True,
        raw_and_temporal_both_saved=True,six_false_not_training=True,audit_primary_only_if_all_nonregression_and_validation_improves=True,
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'evaluation_protocol.json',ep)
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth';sys.path.insert(0,str(modelroot));from inference import BallInferencer
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    net=infer.model;initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};net.eval().requires_grad_(False);temporal.OUT=OUT
    result=dict(status='fixed_point_supervision_evaluation_in_progress',arms={},original_current={role:metrics(rows) for role,rows in refs.items()},audit=None)
    summary=[];medians={};samples=prepare(read(OUT/'positive_training_manifest.csv'),medians);point_scores=[]
    neg=read(OUT/'negative_point_manifest.csv');neg_by={r['positive_sample_id']:r for r in neg}
    # No body/camera phase labels are manufactured; six queries have unknown true centers.
    false=read(ROOT/'ball_040_object_swap_review/round2/user_reviewed/candidate_review.csv');template=next(r for r in val if r['candidate_id']=='match1_motion_pilot_015');six=[]
    for r in false:
        t=int(r['local_frame']);frame_dir=SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_015/frames';paths=[frame_dir/f'{j:06d}.jpg' for j in range(t-3,t+1)]
        six.append(dict(sample_id=f'match1_motion_pilot_015_frame_{t:06d}',candidate_id='match1_motion_pilot_015',local_frame=t,source_frame=r['source_frame'],
            input_frames_json=json.dumps([str(p) for p in paths]),input_frame_sha256_json=json.dumps([sha(p) for p in paths]),
            median_path=template['median_path'],median_sha256=template['median_sha256'],truth_visibility='unlabelled',truth_x_px='',truth_y_px=''))
    def load_arm(arm):
        entry=trained['checkpoints'][arm];delta=torch.load(entry['path'],map_location='cpu',weights_only=False)
        assert delta['meta']['epoch']==3 and delta['meta']['source_weight_sha256']==sha(weight)
        state=dict(initial);state.update(delta['trainable_state_dict']);net.load_state_dict(state,strict=True);net.eval()
        assert all(torch.equal(v.detach().cpu(),initial[k]) for k,v in net.state_dict().items() if k not in delta['trainable_state_dict'])
    def evaluate(rows,role,arm):
        prefix=arm+'_'+role;ctx,masks=temporal.get_contexts(infer,rows,prefix,protected)
        outputs={}
        for method,cfg in [('raw',None),('temporal',CFG)]:
            pred=temporal.predictions(rows,ctx,masks,cfg,role);write(OUT/f'{prefix}_{method}.csv',pred);outputs[method]=pred
        if role!='six_false':
            baseline=refs[role];pm=metrics(outputs['temporal']);lost=sorted(good_ids(baseline)-good_ids(outputs['temporal']));gains=sorted(good_ids(outputs['temporal'])-good_ids(baseline))
            result['arms'][arm][role]=dict(raw=metrics(outputs['raw']),temporal=pm,original_correct_lost=lost,original_correct_gained=gains,
                nonregression=not lost and err(pm)<=err(metrics(baseline)))
            for method,pred in outputs.items():
                m=metrics(pred);summary.append(dict(arm=arm,role=role,method=method,correct=m['counts']['localized_tp_16px'],errors=err(m),
                    missing_visible=m['counts']['presence_fn'],original_current_correct=metrics(baseline)['counts']['localized_tp_16px'],original_current_errors=err(metrics(baseline)),
                    original_correct_lost=len(good_ids(baseline)-good_ids(pred))))
        return outputs
    for arm in ['positive_control','point_negative']:
        load_arm(arm);result['arms'][arm]={}
        for first in range(0,len(samples),4):
            batch=samples[first:first+4]
            with torch.inference_mode(),torch.amp.autocast('cuda'):p,_,_=net.forward(frames=tensors(batch,medians,infer.device))
            for j,s in enumerate(batch):
                r=s['row'];pmap=p[j:j+1].float();point_scores.append(dict(arm=arm,sample_id=r['sample_id'],kind='manual_positive',probability=float(point_value(pmap,r['truth_x_px'],r['truth_y_px']))))
                if r['sample_id'] in neg_by:
                    n=neg_by[r['sample_id']];point_scores.append(dict(arm=arm,sample_id=r['sample_id'],kind='confirmed_non_ball',probability=float(point_value(pmap,n['candidate_x_px'],n['candidate_y_px']))))
        for role,rows in [('train',train),('validation',val),('challenge',challenge),('short_sequence',short)]:evaluate(rows,role,arm)
        outputs=evaluate(six,'six_false',arm);six_results=[]
        for method,preds in outputs.items():
            for f,p in zip(false,preds):
                d=math.hypot(float(p['x_px'])-float(f['displayed_x_px']),float(p['y_px'])-float(f['displayed_y_px'])) if int(p['visible']) else None
                state='position_unknown' if not int(p['visible']) else 'same_known_false_point' if d==0 else 'near_known_false_point_unverified' if d<=16 else 'different_unverified_candidate'
                six_results.append(dict(arm=arm,method=method,local_frame=p['local_frame'],x_px=p['x_px'],y_px=p['y_px'],visible=p['visible'],distance_to_known_false_point=d,status=state,true_ball_center='unknown',physical_visibility='not_labelled'))
        write(OUT/f'{arm}_six_false_outcomes.csv',six_results);result['arms'][arm]['six_false']={method:dict(Counter(r['status'] for r in six_results if r['method']==method)) for method in ['raw','temporal']}
    write(OUT/'final_training_point_scores.csv',point_scores)
    primary=result['arms']['point_negative'];passes=all(primary[role]['nonregression'] for role in ['train','validation','challenge','short_sequence'])
    benefit=primary['validation']['temporal']['counts']['localized_tp_16px']>81 or err(primary['validation']['temporal'])<2
    if passes and benefit:
        load_arm('point_negative');refs['audit']=read(BASE/'audit_selected.csv');protected[str(BASE/'audit_selected.csv')]=sha(BASE/'audit_selected.csv')
        evaluate(read(DATA/'audit_labels.csv'),'audit','point_negative');result['audit']=result['arms']['point_negative']['audit']
    for arm,entry in trained['checkpoints'].items():assert sha(Path(entry['path']))==entry['sha256']
    assert all(sha(Path(p))==h for p,h in protected.items())
    write(OUT/'comparison_summary.csv',summary)
    result.update(status='fixed_final_point_supervision_comparison_complete',protected_input_hashes=protected,protected_inputs_unchanged=True,
        original_weights_unchanged=True,new_human_labels=0,interpolation=0,automatic_adoption=False,operational_outputs_changed=False,independent_generalization_established=False)
    dump(OUT/'decision.json',result)
    print(json.dumps(summary,indent=2),flush=True)
    print('SIX',json.dumps({arm:result['arms'][arm]['six_false'] for arm in result['arms']}),flush=True)
if __name__=='__main__':main()

