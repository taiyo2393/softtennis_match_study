"""One training-only hard-example reweighting trial with matched controls.

Same120 labelled inputs/order/180 updates/decoder parameters as the old trial.
Two training frames receive loss weight8, using their existing manual Gaussian
targets. No new non-ball class or validation/audit examples enter training.
Select both old ordinary and new weighted checkpoints under identical guards.
"""
import json,random,sys,math
from pathlib import Path
import numpy as np,torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha
import run_manual_ball_finetune as helper
from run_manual_ball_finetune import prepare,tensors,metrics

OUT=ROOT/'balltrack_hard_case_finetune_pilot'
DATA=ROOT/'balltrack_manual_finetune_pilot'
SEED=20261103
THRESHOLDS=[.3,.4,.5,.6,.7]

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def good_ids(rows):
    return {r['sample_id'] for r in rows if r['truth_visibility']=='visible' and int(r['visible']) and
        math.hypot(float(r['x_px'])-float(r['truth_x_px']),float(r['y_px'])-float(r['truth_y_px']))<=16}
def err(m):return m['counts']['wrong_localization']+m['counts']['presence_fp']

def infer_maps(infer,samples,medians):
    infer.model.eval();maps=[]
    with torch.inference_mode():
        for first in range(0,len(samples),4):
            data=tensors(samples[first:first+4],medians,infer.device)
            with torch.amp.autocast('cuda'):prediction,_,_=infer.model.forward(frames=data)
            maps.extend(prediction[:,0].float().cpu().numpy())
    return maps

def decode(infer,maps,samples,threshold,family,epoch,role):
    infer.thre=threshold;rows=[]
    for hm,s in zip(maps,samples,strict=True):
        r=s['row'];x,y,w,h,conf=infer._predict_location(hm)
        px=int((x+w/2)*1920/512);py=int((y+h/2)*1080/288)
        rows.append(dict(sample_id=r['sample_id'],candidate_id=r['candidate_id'],local_frame=r['local_frame'],source_frame=r['source_frame'],
            truth_visibility=r['truth_visibility'],truth_x_px=r['truth_x_px'],truth_y_px=r['truth_y_px'],
            x_px=px,y_px=py,visible=int(px!=0 or py!=0),confidence=conf,family=family,epoch=epoch,threshold=threshold,role=role))
    return rows

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Never tune a completed trial in place.'
    manifest=load(DATA/'data_manifest.json');hardfile=DATA/'training_hard_cases/far_wrong_target_predictions.csv';hardcases=read(hardfile)
    assert len(hardcases)==2 and all(r['role']=='training_only' for r in hardcases)
    hard_ids={r['sample_id'] for r in hardcases};train_rows=read(DATA/'train_labels.csv')
    assert hard_ids<={r['sample_id'] for r in train_rows} and len(train_rows)==120
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';initial_weight=modelroot/'checkpoints/balltrack_best.pth'
    old_history=load(DATA/'training_history.json')
    ordinary={r['epoch']:Path(r['checkpoint']) for r in old_history if r['epoch']}
    assert set(ordinary)=={1,2,3}
    protected=dict(manifest['input_hashes']);protected[str(hardfile)]=sha(hardfile);protected[str(initial_weight)]=sha(initial_weight)
    for r in old_history:
        if r['epoch']:
            assert sha(Path(r['checkpoint']))==r['checkpoint_sha256'];protected[r['checkpoint']]=r['checkpoint_sha256']
    assert all(sha(Path(p))==h for p,h in protected.items())
    protocol=dict(status='fixed_before_training_and_matched_control_selection',seed=SEED,epochs=3,batch_size=2,
        learning_rate=1e-4,optimizer='Adam',weight_decay=0.,gradient_clip_norm=1.,
        trainable=['up_block_3','predictor'],frozen='Other weights and allBatchNorm running statistics',
        hard_sample_ids=sorted(hard_ids),hard_frame_loss_weight=8.,other_frame_weight=1.,
        normalisation='Batch mean of(per-frame sourceWBCELoss*weight) divided by global training mean weight134/120, not per-batch weight sum',
        labels='Original user-clicked Gaussian targets, sigma3.5; existing not_visible targets unchanged. Hard cases are wrong target locations, not new generic non-ball labels.',
        train_inputs_and_order='Same120frames,np.default_rng(20261103) permutations,3epochs,180updates as ordinary trial. No sampling/augmentation changes.',
        matched_control='Reuse all3 previously saved ordinary checkpoints; identical threshold and checkpoint selection to the new weighted family',
        thresholds=THRESHOLDS,selection='Original model included. Retain every original correct validation sample and wrong_position+presenceFP<=3; among feasible candidates maxTP16px,then fewer errors,then earlier epoch. Ordinary/original wins family ties.',
        audit='Read/evaluate60audit labels only after global validation choice is saved. No model,threshold or family selection on audit.',
        caveat='Validation/audit clips were seen in previous trials. Clip disjointness is enforced; this is same-match development, not an external test.',
        extra_hyperparameter_search=False,automatic_adoption=False,
        dataset_manifest_sha256=sha(DATA/'data_manifest.json'),hard_inventory_sha256=sha(hardfile),
        script_sha256=sha(Path(__file__)),helper_code_sha256=sha(ROOT/'run_manual_ball_finetune.py'),protected_input_hashes=protected)
    dump(OUT/'protocol.json',protocol)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.backends.cudnn.benchmark=False
    sys.path.insert(0,str(modelroot));from inference import BallInferencer
    from model.loss_utils import WBCELoss
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(initial_weight),device='cuda',thre=.3,batchsize=4)
    net=infer.model;initial_state={n:t.detach().cpu().clone() for n,t in net.state_dict().items()}
    params=[];names=[]
    for name,p in net.named_parameters():
        enabled=name.startswith('up_block_3.') or name.startswith('predictor.');p.requires_grad_(enabled)
        if enabled:params.append(p);names.append(name)
    medians={};train=prepare(train_rows,medians);validation=prepare(read(DATA/'validation_labels.csv'),medians)
    baseline_maps=infer_maps(infer,validation,medians);baseline_rows=decode(infer,baseline_maps,validation,.3,'original',0,'validation')
    write(OUT/'validation_original.csv',baseline_rows);baseline=metrics(baseline_rows);original_ids=good_ids(baseline_rows)
    assert len(original_ids)==80 and err(baseline)==3
    bests={name:dict(family='original',epoch=0,threshold=.3,checkpoint=str(initial_weight),checkpoint_sha256=sha(initial_weight),metrics=baseline,correct_sample_ids=sorted(original_ids)) for name in ['ordinary','hard_weighted']}
    summaries=[]
    def score_family(family,epoch,path):
        maps=infer_maps(infer,validation,medians)
        for threshold in THRESHOLDS:
            rows=decode(infer,maps,validation,threshold,family,epoch,'validation');m=metrics(rows);ids=good_ids(rows);lost=original_ids-ids
            feasible=not lost and err(m)<=3
            summary=dict(family=family,epoch=epoch,threshold=threshold,tp16=m['counts']['localized_tp_16px'],errors=err(m),wrong_positions=m['counts']['wrong_localization'],
                nonvisible_outputs=m['counts']['presence_fp'],baseline_correct_frames_lost=len(lost),lost_sample_ids=';'.join(sorted(lost)),feasible=int(feasible))
            summaries.append(summary);write(OUT/f'validation_{family}_epoch{epoch:02d}_threshold{int(threshold*100):02d}.csv',rows)
            best=bests[family]
            if feasible and (summary['tp16'],-summary['errors'])>(best['metrics']['counts']['localized_tp_16px'],-err(best['metrics'])):
                bests[family]=dict(family=family,epoch=epoch,threshold=threshold,checkpoint=str(path.resolve()),checkpoint_sha256=sha(path),metrics=m,correct_sample_ids=sorted(ids))
        print('validation',family,'epoch',epoch,'grid',json.dumps([r for r in summaries if r['family']==family and r['epoch']==epoch]),flush=True)
    # Train new family from identical original initialisation.
    optimizer=torch.optim.Adam(params,lr=1e-4,weight_decay=0.);scaler=torch.amp.GradScaler('cuda');generator=np.random.default_rng(SEED)
    global_mean_weight=(len(train)+7*len(hard_ids))/len(train);history=[]
    for epoch in range(1,4):
        net.eval();order=generator.permutation(len(train));losses=[]
        for first in range(0,len(order),2):
            batch=[train[i] for i in order[first:first+2]];data,target=tensors(batch,medians,infer.device,targets=True)
            weights=torch.tensor([8. if s['row']['sample_id'] in hard_ids else 1. for s in batch],device=infer.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):prediction,_,_=net.forward(frames=data)
            per_frame=WBCELoss(prediction.float(),target,reduce=False)
            assert per_frame.shape==weights.shape
            loss=(per_frame*weights).mean()/global_mean_weight;assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
            if first%40==0:print('hard weighted epoch',epoch,'train',first,'/',len(train),'loss',losses[-1],flush=True)
        checkpoint=OUT/f'hard_weighted_epoch{epoch:02d}.pth'
        torch.save(dict(state_dict={n:t.detach().cpu() for n,t in net.state_dict().items()},meta=dict(epoch=epoch,protocol_sha256=sha(OUT/'protocol.json'))),checkpoint)
        history.append(dict(epoch=epoch,mean_weighted_loss=float(np.mean(losses)),checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha(checkpoint)))
        score_family('hard_weighted',epoch,checkpoint);dump(OUT/'training_history.json',history)
    assert all(torch.equal(net.state_dict()[n].detach().cpu(),t) for n,t in initial_state.items() if n not in names)
    # Old checkpoints get the same joint selection; do not credit weighting for a calibration gain.
    for epoch,path in ordinary.items():
        net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        score_family('ordinary',epoch,path)
    write(OUT/'validation_selection_grid.csv',summaries)
    ordinary_best=bests['ordinary'];weighted_best=bests['hard_weighted']
    selection=weighted_best if (weighted_best['metrics']['counts']['localized_tp_16px'],-err(weighted_best['metrics']))>(ordinary_best['metrics']['counts']['localized_tp_16px'],-err(ordinary_best['metrics'])) else ordinary_best
    dump(OUT/'validation_selection.json',dict(family_choices=bests,selected=selection,audit_used_for_selection=False,automatic_adoption=False))
    # Audit chosen family ONLY; other families are not evaluated and selected on audit.
    audit=prepare(read(DATA/'audit_labels.csv'),medians)
    net.load_state_dict(initial_state,strict=True);baseline_audit_maps=infer_maps(infer,audit,medians)
    audit_base_rows=decode(infer,baseline_audit_maps,audit,.3,'original',0,'audit');audit_base=metrics(audit_base_rows)
    if selection['family']!='original':
        net.load_state_dict(torch.load(selection['checkpoint'],map_location='cpu',weights_only=False)['state_dict'],strict=True)
        audit_maps=infer_maps(infer,audit,medians)
    else:audit_maps=baseline_audit_maps
    audit_rows=decode(infer,audit_maps,audit,selection['threshold'],selection['family'],selection['epoch'],'audit');audit_result=metrics(audit_rows)
    write(OUT/'audit_original.csv',audit_base_rows);write(OUT/'audit_selected.csv',audit_rows)
    audit_lost=sorted(good_ids(audit_base_rows)-good_ids(audit_rows))
    assert all(sha(Path(p))==h for p,h in protected.items())
    for role,h in manifest['partition_hashes'].items():assert sha(DATA/f'{role}_labels.csv')==h
    result=dict(status='fixed_hard_case_trial_and_matched_selection_complete',automatic_adoption=False,
        original_validation=baseline,ordinary_selected=ordinary_best,hard_weighted_selected=weighted_best,validation_selected=selection,
        audit_original=audit_base,audit_selected=audit_result,audit_original_correct_frames_lost=audit_lost,
        audit_error_count_increased=err(audit_result)>err(audit_base),
        audit_passes_nonregression=not audit_lost and err(audit_result)<=err(audit_base),
        selection_used_validation_only=True,training_hard_cases_only=True,same_training_updates_and_inputs_as_control=True,
        original_models_labels_reviewed_outputs_unchanged=True,frozen_parameters_and_BN_buffers_unchanged_during_training=True,
        independent_external_accuracy=False,
        scope='One fixed multiplier8 on2 training cases; no multiplier search. Same-match selected sparse labels, including preparation/transitions.')
    dump(OUT/'decision.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
