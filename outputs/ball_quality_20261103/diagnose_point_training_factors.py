"""Matched2x2 TRAIN-ONLY diagnosis: negative exposure frequency x preservation.

Equal optimizer updates and exact total per-point/per-image loss weight dose.
"""
import argparse,json,sys,math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,read,load,write,sha
from train_point_supervised_ball_detector_fp32 import point_value,support_mask,preservation_loss,reset_seed,SEED
from run_manual_ball_finetune import prepare,tensors,metrics
from compare_ball_candidate_decoders import components,choose

OUT=ROOT/'ball_point_training_factor_diagnosis';PAIRED=ROOT/'ball_point_supervised_detector_fp32'
CONDITIONS=[('rare_preserve','rare',1.),('rare_no_preserve','rare',0.),('balanced_preserve','balanced',1.),('balanced_no_preserve','balanced',0.)]
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def schedules(count,negative_indices):
    assert count==102 and len(negative_indices)==9
    rng=np.random.default_rng(SEED);rare=np.concatenate([rng.permutation(count) for _ in range(3)]).reshape(153,2)
    rng=np.random.default_rng(SEED);paired=np.concatenate([rng.permutation(negative_indices) for _ in range(17)])
    rng=np.random.default_rng(SEED+1);generic=np.concatenate([rng.permutation(count),rng.permutation(count)])[:153]
    balanced=np.stack([paired,generic],axis=1);out={}
    for name,slots in [('rare',rare),('balanced',balanced)]:
        counts=np.bincount(slots.ravel(),minlength=count);positive_weights=3./counts
        enabled=np.isin(slots,negative_indices) if name=='rare' else np.tile([True,False],(153,1))
        neg_weight=102/9 if name=='rare' else 2.
        pdose=np.zeros(count);ndose=np.zeros(count)
        for ids,on in zip(slots,enabled):
            for i,e in zip(ids,on):pdose[i]+=positive_weights[i];ndose[i]+=neg_weight if e else 0.
        assert np.allclose(pdose,3.,rtol=0,atol=1e-12)
        assert np.allclose(ndose[negative_indices],34.,rtol=0,atol=1e-12) and np.isclose(ndose.sum(),306.)
        out[name]=dict(slots=slots,enabled=enabled,positive_weights=positive_weights,negative_weight=neg_weight,
            positive_dose=pdose,negative_dose=ndose,negative_exposure_slots=int(enabled.sum()),negative_exposure_batches=int(enabled.any(1).sum()),image_counts=counts)
    return out
def self_test():
    s=schedules(102,np.array([0,1,3,20,40,55,70,90,101]))
    assert s['rare']['negative_exposure_slots']==27 and s['balanced']['negative_exposure_slots']==153
    assert s['balanced']['negative_exposure_batches']==153
    assert np.array_equal(s['rare']['slots'].ravel()[:102],np.random.default_rng(SEED).permutation(102))
    assert np.allclose(s['rare']['positive_dose'],s['balanced']['positive_dose']) and np.allclose(s['rare']['negative_dose'],s['balanced']['negative_dose'])
    print('Passed deterministic102-image schedules, exact matched per-image positive/preservation dose, exact9-point negative dose, and one negative pair per balanced batch. Synthetic indices only.')
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Preserve factor diagnosis; no selecting new coefficients after results.'
    positives=read(PAIRED/'positive_training_manifest.csv');negatives=read(PAIRED/'negative_point_manifest.csv');all_labels=read(ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv')
    assert len(positives)==102 and len(negatives)==9
    neg_by={r['positive_sample_id']:r for r in negatives};negative_indices=np.array([i for i,r in enumerate(positives) if r['sample_id'] in neg_by]);schedule=schedules(102,negative_indices)
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth'
    paths=[weight,PAIRED/'protocol.json',PAIRED/'positive_training_manifest.csv',PAIRED/'negative_point_manifest.csv',PAIRED/'original_teacher_maps.npz',
        PAIRED/'point_negative_epoch03_delta.pth',PAIRED/'trained_models.json',PAIRED/'verification.json',ROOT/'ball_point_detector_calibration/decision.json',
        ROOT/'ball_point_supervision_sanity/result.json',ROOT/'ball_point_supervision_sanity/point_scores.csv',
        ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',
        ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv',ROOT/'train_point_supervised_ball_detector_fp32.py']
    protected={str(p):sha(p) for p in paths}
    protocol=dict(status='fixed_before_factor_training',purpose='Training diagnosis ONLY: separate negative-exposure frequency and original-response preservation. No validation, model selection or deployment candidate.',
        conditions=CONDITIONS,unique_positive_images=102,unique_negative_points=9,source_clips=8,negative_clips=5,
        updates_per_condition=153,batch_size=2,image_slots_per_condition=306,seed=SEED,learning_rate=1e-4,optimizer='Adam',precision='FP32',gradient_clip=1.,
        trainable=['up_block_3','predictor'],frozen='All other weights and ALLBatchNorm buffers;eval always',
        rare='Exact old3permutations of102frames,51batches/epoch; each positive image3times, each of9negative points3times; negative multiplier102/9.',
        balanced='153batches: one paired-frame slot cycling9each17times + one generic-positive slot cyclingall102. Negative loss only in paired slot, multiplier2. Same image may occupy both slots; no BatchNorm update.',
        positive_importance='Each positive image loss weight3/(its total exposure count). Thus every positive center total weight3, and every preservation image total weight3 when enabled.',
        negative_importance='Every negative point total weight34 in both schedules; total positive weight306 and negative weight306. Equal weighted dose does not imply identical clipped/Adam trajectories: frequency is the factor.',
        preservation='Prior implementation unchanged, coefficient0or1 only. Same original AMP soft teacher and sparse point support masks in all conditions; teacher not GT.',
        labels='Point labels only; no nonvisible training frames or dense background GT. Adjacent21nonvisible labels are used only in TRAINING-SET post-fit raw-output diagnosis.',
        final_probe='Final153updates only, fixed0.3. Positive/negative point response, within-image pair ordering and per-frame point fit. Raw training123 output also reported to separate point fit from detector precision.',
        reproduction='Compare rare_preserve final weight tensors with old point_negative153update model; record exact/numerical differences, no checkpoint substitution.',
        inference='Fixed original causal median+BGRt-3..t,512x288; FP32 probe for all four and original. No new threshold/crop/stride choice.',
        no_validation_or_audit=True,no_best_model_selection=True,no_new_human_labels=True,no_automatic_adoption=True,
        inference_warning='Training probes are same data being fit, not held-out performance. Single seed and small selected negative corpus limit causal conclusions to this controlled run.',
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    schedule_rows=[];dose_rows=[]
    for name,s in schedule.items():
        for step,(ids,on) in enumerate(zip(s['slots'],s['enabled']),1):
            for slot,(i,e) in enumerate(zip(ids,on)):
                schedule_rows.append(dict(schedule=name,step=step,slot=slot,sample_id=positives[i]['sample_id'],positive_weight=s['positive_weights'][i],
                    negative_label_exposed=int(e),negative_weight=s['negative_weight'] if e else 0.))
        for i,r in enumerate(positives):dose_rows.append(dict(schedule=name,sample_id=r['sample_id'],exposures=int(s['image_counts'][i]),positive_total_weight=s['positive_dose'][i],negative_total_weight=s['negative_dose'][i]))
    write(OUT/'scheduled_batches.csv',schedule_rows);write(OUT/'per_image_loss_dose.csv',dose_rows)
    reset_seed();sys.path.insert(0,str(modelroot));from inference import BallInferencer
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4);net=infer.model
    initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};params=[];names=set()
    for name,param in net.named_parameters():
        enabled=name.startswith('up_block_3.') or name.startswith('predictor.');param.requires_grad_(enabled)
        if enabled:params.append(param);names.add(name)
    medians={};all_samples=prepare(all_labels,medians);all_by={s['row']['sample_id']:s for s in all_samples};samples=[all_by[r['sample_id']] for r in positives]
    with np.load(PAIRED/'original_teacher_maps.npz') as cache:
        assert list(cache['sample_ids'])==[r['sample_id'] for r in positives];teacher=torch.from_numpy(cache['heatmaps'].astype(np.float32))
    masks=torch.stack([support_mask([(r['truth_x_px'],r['truth_y_px'])]+[(neg_by[r['sample_id']]['candidate_x_px'],neg_by[r['sample_id']]['candidate_y_px'])] if r['sample_id'] in neg_by else [(r['truth_x_px'],r['truth_y_px'])])[None] for r in positives])
    point_rows=[];summary=[];history=[];model_records={}
    def probe(condition):
        raw=[];local=[];footprints=[]
        with torch.inference_mode():
            for first in range(0,len(all_samples),4):
                batch=all_samples[first:first+4];p,_,_=net.forward(frames=tensors(batch,medians,infer.device))
                for j,sample in enumerate(batch):
                    r=sample['row'];hm=p[j,0].cpu().numpy();c=choose(components(hm),'bbox_area');x,y=(c['x_px'],c['y_px']) if c else (0,0)
                    raw.append(dict(sample_id=r['sample_id'],candidate_id=r['candidate_id'],local_frame=r['local_frame'],source_frame=r['source_frame'],
                        truth_visibility=r['truth_visibility'],truth_x_px=r['truth_x_px'],truth_y_px=r['truth_y_px'],x_px=x,y_px=y,visible=int(c is not None)))
                    footprints.append(float((hm>.3).mean()))
                    if r['truth_visibility']=='visible':
                        pos=float(point_value(p[j:j+1],r['truth_x_px'],r['truth_y_px']));local.append(dict(condition=condition,sample_id=r['sample_id'],kind='positive',score=pos))
                        if r['sample_id'] in neg_by:
                            n=neg_by[r['sample_id']];negative=float(point_value(p[j:j+1],n['candidate_x_px'],n['candidate_y_px']));local.append(dict(condition=condition,sample_id=r['sample_id'],kind='negative',score=negative))
        write(OUT/f'{condition}_training123_raw.csv',raw);point_rows.extend(local)
        pos={r['sample_id']:r['score'] for r in local if r['kind']=='positive'};neg={r['sample_id']:r['score'] for r in local if r['kind']=='negative'}
        m=metrics(raw)
        record=dict(condition=condition,training_positive_above03=sum(v>.3 for v in pos.values()),training_negative_above03=sum(v>.3 for v in neg.values()),
            correct_point_pairs=sum(pos[k]>.3 and v<=.3 for k,v in neg.items()),true_point_above_negative=sum(pos[k]>v for k,v in neg.items()),
            mean_positive_response=float(np.mean(list(pos.values()))),mean_negative_response=float(np.mean(list(neg.values()))),
            raw_training_correct=m['counts']['localized_tp_16px'],raw_training_errors=m['counts']['wrong_localization']+m['counts']['presence_fp'],
            raw_training_outputs=m['counts']['pred_visible'],mean_threshold_pixel_fraction=float(np.mean(footprints)))
        summary.append(record);return record
    net.eval();probe('original_fp32')
    for condition,exposure,strength in CONDITIONS:
        reset_seed();net.load_state_dict(initial,strict=True);net.eval();optimizer=torch.optim.Adam(params,lr=1e-4,weight_decay=0.);s=schedule[exposure]
        for step,(ids,on) in enumerate(zip(s['slots'],s['enabled']),1):
            batch=[samples[i] for i in ids];data=tensors(batch,medians,infer.device);q=teacher[ids].to(infer.device);mask=masks[ids].to(infer.device);optimizer.zero_grad(set_to_none=True)
            p,_,_=net.forward(frames=data);pos=[];neg=[]
            for j,(i,e) in enumerate(zip(ids,on)):
                r=positives[i];a=point_value(p[j:j+1],r['truth_x_px'],r['truth_y_px']).clamp(1e-6,1-1e-6);pos.append(-a.log())
                if e:
                    n=neg_by[r['sample_id']];b=point_value(p[j:j+1],n['candidate_x_px'],n['candidate_y_px']).clamp(1e-6,1-1e-6);neg.append(-torch.log1p(-b))
                else:neg.append(p[j].sum()*0.)
            pos_vector=torch.stack(pos);negative_vector=torch.stack(neg)
            if exposure=='rare':
                positive_loss=pos_vector.mean();preserve=preservation_loss(p,q,mask);negative_loss=(102/9)*negative_vector.mean()
            else:
                w=torch.tensor(s['positive_weights'][ids],device=infer.device,dtype=torch.float32)
                positive_loss=(w*pos_vector).mean();per_preserve=torch.stack([preservation_loss(p[j:j+1],q[j:j+1],mask[j:j+1]) for j in range(len(ids))]);preserve=(w*per_preserve).mean();negative_loss=2.*negative_vector.mean()
            loss=positive_loss+preserve+1.*negative_loss if strength==1. else positive_loss+0.*preserve+1.*negative_loss
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm);optimizer.step()
            history.append(dict(condition=condition,step=step,positive_loss=float(positive_loss.detach()),weighted_preservation_loss=float(preserve.detach()),weighted_negative_loss=float(negative_loss.detach()),
                total_loss=float(loss.detach()),negative_slots=int(on.sum()),gradient_norm_before_clip=float(norm),gradient_clipped=int(float(norm)>1.)))
            if step%51==0:print(condition,'step',step,'loss',history[-1]['total_loss'],flush=True)
        assert all(torch.equal(net.state_dict()[k].detach().cpu(),v) for k,v in initial.items() if k not in names)
        delta={k:net.state_dict()[k].detach().cpu().clone() for k in sorted(names)};path=OUT/f'{condition}_diagnostic_delta.pth'
        torch.save(dict(trainable_state_dict=delta,meta=dict(condition=condition,steps=153,diagnostic_only=True,protocol_sha256=sha(OUT/'protocol.json'),source_weight_sha256=sha(weight))),path)
        replication=None
        if condition=='rare_preserve':
            old=torch.load(PAIRED/'point_negative_epoch03_delta.pth',map_location='cpu',weights_only=False)['trainable_state_dict']
            replication=dict(all_tensors_bitwise_equal=all(torch.equal(delta[k],old[k]) for k in delta),max_abs_weight_difference=max(float((delta[k]-old[k]).abs().max()) for k in delta))
        model_records[condition]=dict(path=str(path),sha256=sha(path),frozen_state_unchanged=True,replication_of_previous=replication)
        print('PROBE',json.dumps(probe(condition)),flush=True)
        write(OUT/'training_history.csv',history);write(OUT/'point_responses.csv',point_rows);write(OUT/'training_diagnostic_summary.csv',summary)
    assert all(sha(Path(p))==h for p,h in protected.items())
    contrasts=[];records={r['condition']:r for r in summary}
    for factor,a,b in [('frequency_with_preservation','rare_preserve','balanced_preserve'),('frequency_without_preservation','rare_no_preserve','balanced_no_preserve'),
        ('preservation_with_rare_exposure','rare_no_preserve','rare_preserve'),('preservation_with_balanced_exposure','balanced_no_preserve','balanced_preserve')]:
        contrasts.append(dict(contrast=factor,from_condition=a,to_condition=b,
            delta_negative_above03=records[b]['training_negative_above03']-records[a]['training_negative_above03'],delta_positive_above03=records[b]['training_positive_above03']-records[a]['training_positive_above03'],
            delta_correct_point_pairs=records[b]['correct_point_pairs']-records[a]['correct_point_pairs'],delta_raw_training_errors=records[b]['raw_training_errors']-records[a]['raw_training_errors']))
    write(OUT/'factor_contrasts.csv',contrasts)
    result=dict(status='matched_training_only_factor_diagnosis_complete',summary=summary,contrasts=contrasts,models=model_records,
        weighted_positive_mass_per_condition=306.,weighted_negative_mass_per_condition=306.,updates_per_condition=153,unique_training_images=102,negative_point_pairs=9,
        negative_exposure_batches={k:s['negative_exposure_batches'] for k,s in schedule.items()},
        protected_inputs_unchanged=True,no_validation_or_audit=True,no_best_model_selection=True,new_human_labels=0,automatic_adoption=False,operational_outputs_changed=False,
        interpretation='Training response/fit and same-training raw-output diagnostics only. Equal doses isolate scheduling frequency and preservation switch within this one run; no external tracking benefit established.')
    dump(OUT/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='models'},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    else:main()
