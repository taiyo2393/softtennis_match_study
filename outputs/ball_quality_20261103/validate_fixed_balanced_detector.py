"""New follow-up: ONE frozen training-selected balanced model on known holdouts.

No retraining, no alternate checkpoint after failure, no operational adoption.
"""
import json,sys,math,shutil
from pathlib import Path
from collections import Counter
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha
import compare_temporal_candidate_paths as temporal
from run_manual_ball_finetune import metrics
from run_hard_case_ball_finetune import good_ids,err

OUT=ROOT/'ball_balanced_fixed_validation';DIAG=ROOT/'ball_balanced_localization_diagnosis';FACT=ROOT/'ball_point_training_factor_diagnosis';BASE=ROOT/'ball_temporal_path_comparison';DATA=ROOT/'balltrack_manual_finetune_pilot';SHORT=ROOT/'ball_short_sequence_evaluation'
CFG=dict(name='displacement_0.1',order='displacement',strength=.1)
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Preserve fixed validation; never substitute after results.'
    diag=load(DIAG/'result.json');assert load(DIAG/'verification.json')['status']=='passed'
    candidates=diag['strict_train_improving_temporal_settings'];assert candidates
    chosen=sorted(candidates,key=lambda r:(-r['correct'],r['errors'],r['condition']!='balanced_preserve',r['threshold']))[0]
    assert chosen['condition']=='balanced_preserve' and chosen['threshold']==.3 and chosen['correct']==66 and chosen['errors']==2 and chosen['original_correct_lost']==0
    record=load(FACT/'result.json')['models'][chosen['condition']];source_delta=Path(record['path']);assert sha(source_delta)==record['sha256']
    snapshot=OUT/'candidate_frozen_delta.pth';shutil.copyfile(source_delta,snapshot);assert sha(snapshot)==record['sha256']
    selection=dict(condition=chosen['condition'],threshold=chosen['threshold'],selector=CFG,training_metrics=chosen,
        choice='One strict-training-improving condition; max correct then min errors, preserve-enabled then smaller threshold on ties.',
        source_diagnostic_model=str(source_delta),source_diagnostic_model_sha256=record['sha256'],validation_snapshot=str(snapshot),snapshot_sha256=sha(snapshot),
        role_transition='Prior experiments remain training-only and weights retain diagnostic_only metadata. This NEW protocol freezes this candidate for development validation only, not deployment.',
        validation_labels_used_for_this_choice=False,alternate_model_after_failure=False)
    dump(OUT/'frozen_selection.json',selection)
    source=SOURCE/'third_party/RacketVision/source/BallTrack';weight=source/'checkpoints/balltrack_best.pth'
    paths=[DIAG/'protocol.json',DIAG/'result.json',DIAG/'verification.json',DIAG/'balanced_preserve_threshold30_temporal.csv',
        FACT/'protocol.json',FACT/'result.json',FACT/'verification.json',source_delta,snapshot,weight,DATA/'validation_labels.csv',DATA/'audit_labels.csv',
        BASE/'validation_selected.csv',BASE/'challenge_selected.csv',BASE/'audit_selected.csv',ROOT/'ball_inplay_challenge/manual_center_labels.csv',ROOT/'ball_inplay_challenge/query_model_inputs.csv',
        SHORT/'manual_sequence_labels.csv',SHORT/'query_model_inputs.csv',SHORT/'frozen_predictions.csv',
        ROOT/'ball_040_object_swap_review/round2/user_reviewed/candidate_review.csv',ROOT/'ball_point_supervised_detector_fp32/positive_training_manifest.csv',
        ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv']
    protected={str(p):sha(p) for p in paths}
    protocol=dict(status='new_followup_fixed_before_validation_inference',selection=selection,
        scope='Known same-match clips excluded from gradient training, but repeatedly inspected development labels. NOT independent external testing.',
        inputs='Original causal median+BGRt-3..t512x288; production CUDA AMP; fixed0.3 and existing displacement0.1 with future8sourceframes.',
        primary='First compare fixed validation120 against unchanged original strongest temporal81correct,2errors. No family/threshold/epoch fallback after failure.',
        failure_gate='If validation loses any original-correct point or adds errors, stop before challenge4,short26,audit. Six oldfalse-candidate diagnostic allowed regardless, never used for model choice.',
        pass_gate='After validation nonregression, evaluate challenge4 and short26 separately. Audit only if both pass and validation improves or at least one of6knownfalse queries becomes position-unknown.',
        six_false='Infer BEFORE manual exclusions. No output is position unknown, shifted output remains unverified. No new true-center or absence label.',
        no_retraining=True,no_parameter_search=True,automatic_adoption=False,operational_outputs_changed=False,
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    sys.path.insert(0,str(source));from inference import BallInferencer
    infer=BallInferencer(str(source/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4);net=infer.model
    initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};delta=torch.load(snapshot,map_location='cpu',weights_only=False)
    assert delta['meta']['diagnostic_only'] and delta['meta']['source_weight_sha256']==sha(weight)
    assert len(delta['trainable_state_dict'])==8 and all(k.startswith('up_block_3.') or k.startswith('predictor.') for k in delta['trainable_state_dict'])
    state=dict(initial);state.update(delta['trainable_state_dict']);net.load_state_dict(state,strict=True);net.eval().requires_grad_(False);temporal.OUT=OUT
    assert all(torch.equal(net.state_dict()[k].detach().cpu(),v) for k,v in initial.items() if k not in delta['trainable_state_dict'])
    result=dict(status='fixed_balanced_development_validation_in_progress',selection=selection,validation=None,challenge=None,short_sequence=None,six_false=None,audit=None)
    def predict(rows,role):
        contexts,masks=temporal.get_contexts(infer,rows,role,protected);raw=temporal.predictions(rows,contexts,masks,None,role);selected=temporal.predictions(rows,contexts,masks,CFG,role)
        write(OUT/f'{role}_raw.csv',raw);write(OUT/f'{role}_selected.csv',selected);return raw,selected
    def evaluate(rows,role,baseline):
        raw,pred=predict(rows,role);bm=metrics(baseline);pm=metrics(pred);lost=sorted(good_ids(baseline)-good_ids(pred));gains=sorted(good_ids(pred)-good_ids(baseline))
        result[role]=dict(baseline=bm,proposed=pm,raw=metrics(raw),original_correct_lost=lost,original_correct_gained=gains,nonregression=not lost and err(pm)<=err(bm))
        print('RESULT',role,json.dumps(result[role]),flush=True)
    val=read(DATA/'validation_labels.csv');evaluate(val,'validation',read(BASE/'validation_selected.csv'))
    false=read(ROOT/'ball_040_object_swap_review/round2/user_reviewed/candidate_review.csv');template=next(r for r in val if r['candidate_id']=='match1_motion_pilot_015');six=[]
    for f in false:
        t=int(f['local_frame']);fd=SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_015/frames';ps=[fd/f'{j:06d}.jpg' for j in range(t-3,t+1)]
        six.append(dict(sample_id=f'match1_motion_pilot_015_frame_{t:06d}',candidate_id='match1_motion_pilot_015',local_frame=t,source_frame=f['source_frame'],input_frames_json=json.dumps([str(p) for p in ps]),input_frame_sha256_json=json.dumps([sha(p) for p in ps]),
            median_path=template['median_path'],median_sha256=template['median_sha256'],truth_visibility='unlabelled',truth_x_px='',truth_y_px=''))
    raw,pred=predict(six,'six_false');outcomes=[]
    for f,r in zip(false,pred):
        d=math.hypot(float(r['x_px'])-float(f['displayed_x_px']),float(r['y_px'])-float(f['displayed_y_px'])) if int(r['visible']) else None
        status='position_unknown' if not int(r['visible']) else 'same_known_false_point' if d==0 else 'near_known_false_point_unverified' if d<=16 else 'different_unverified_candidate'
        outcomes.append(dict(local_frame=r['local_frame'],source_frame=r['source_frame'],known_false_x=f['displayed_x_px'],known_false_y=f['displayed_y_px'],x_px=r['x_px'],y_px=r['y_px'],visible=r['visible'],status=status,true_ball_center='unknown',physical_visibility='not_labelled'))
    write(OUT/'six_false_outcomes.csv',outcomes);result['six_false']=dict(Counter(r['status'] for r in outcomes))
    if result['validation']['nonregression']:
        gt={r['sample_id']:r for r in read(ROOT/'ball_inplay_challenge/manual_center_labels.csv')};rows=[dict(r,**{k:gt[r['sample_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px']}) for r in read(ROOT/'ball_inplay_challenge/query_model_inputs.csv')]
        evaluate(rows,'challenge',read(BASE/'challenge_selected.csv'))
        gt={r['sample_id']:r for r in read(SHORT/'manual_sequence_labels.csv')};rows=[dict(r,**{k:gt[r['sample_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px']}) for r in read(SHORT/'query_model_inputs.csv')]
        baseline=[dict(r,**{k:gt[r['sample_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px']}) for r in read(SHORT/'frozen_predictions.csv') if r['method']=='displacement_0.1'];evaluate(rows,'short_sequence',baseline)
        benefit=len(result['validation']['original_correct_gained'])>0 or err(result['validation']['proposed'])<2 or result['six_false'].get('position_unknown',0)>0
        if result['challenge']['nonregression'] and result['short_sequence']['nonregression'] and benefit:evaluate(read(DATA/'audit_labels.csv'),'audit',read(BASE/'audit_selected.csv'))
    assert all(sha(Path(p))==h for p,h in protected.items())
    result.update(status='fixed_balanced_development_validation_complete',protected_input_hashes=protected,protected_inputs_unchanged=True,weights_unchanged=True,
        retraining=False,new_human_labels=0,automatic_adoption=False,operational_outputs_changed=False,interpolation=0,independent_external_accuracy_established=False)
    dump(OUT/'result.json',result);print('DONE',json.dumps({k:result[k] for k in ['status','six_false','challenge','short_sequence','audit']}),flush=True)
if __name__=='__main__':main()
