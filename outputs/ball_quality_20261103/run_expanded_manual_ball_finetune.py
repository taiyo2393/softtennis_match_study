"""Fixed matched trial: three new human centers, post-play B excluded.

Both arms have123 slots,3 epochs,186 optimizer steps and the same clip/visibility
exposure and loss weights. Control extra slots repeat old human labels from the
same clips; expansion replaces ONLY those slots with the new clicked centers.
"""
import json,math,random,sys,time
from pathlib import Path
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha
from run_manual_ball_finetune import prepare,tensors,metrics
from run_hard_case_ball_finetune import infer_maps,decode,good_ids,err

ANNOT=ROOT/'ball_training_case_expansion'
DATA=ROOT/'balltrack_manual_finetune_pilot'
OUT=ROOT/'balltrack_expanded_centers_pilot'
SEED=20261103
THRESHOLDS=[.3,.4,.5,.6,.7]
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def reset_seed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False
def harmonize(rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    return [{k:r.get(k,'') for k in fields} for r in rows]

def ingest():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Preserve completed trials; no retuning in place.'
    manifest=load(DATA/'data_manifest.json');annotation_manifest=load(ANNOT/'annotation_manifest.json')
    protected={**manifest['input_hashes'],**annotation_manifest['protected_input_hashes']}
    for role,h in manifest['partition_hashes'].items():
        p=DATA/f'{role}_labels.csv';assert sha(p)==h;protected[str(p)]=h
    for name in ['manual_center_labels.csv','annotation_events.jsonl','annotation_cases.json']:
        p=ANNOT/name;protected[str(p)]=sha(p)
    assert sha(ANNOT/'annotation_cases.json')==annotation_manifest['cases_sha256']
    assert all(sha(Path(p))==h for p,h in protected.items())
    cases=load(ANNOT/'annotation_cases.json');labels=read(ANNOT/'manual_center_labels.csv')
    assert len(cases)==len(labels)==4 and len({r['sample_id'] for r in labels})==4
    cases={c['sample_id']:c for c in cases}
    events=[json.loads(line) for line in (ANNOT/'annotation_events.jsonl').read_text(encoding='utf-8').splitlines() if line]
    latest={e['sample_id']:e for e in events}
    old_train=read(DATA/'train_labels.csv');validation=read(DATA/'validation_labels.csv');audit=read(DATA/'audit_labels.csv')
    inventory={r['ball_pilot_id']:r for r in read(ROOT/'play_start_audit/ball_pilot_start_inventory.csv')}
    train_clips={r['candidate_id'] for r in old_train}
    other_clips={r['candidate_id'] for r in validation+audit}
    other_hashes={h for r in validation+audit for h in json.loads(r['input_frame_sha256_json'])}
    old_ids={r['sample_id'] for r in old_train+validation+audit}
    accepted=[];additions=[]
    for r in labels:
        c=cases[r['sample_id']];cid=r['candidate_id'];t=int(r['local_frame'])
        assert latest[r['sample_id']]['after']==r
        assert latest[r['sample_id']]['model_prediction_hidden'] and latest[r['sample_id']]['context_video_available']
        assert r['reviewer_id']=='human_user' and r['annotated_at_utc']
        assert r['truth_visibility']=='visible' and 0<=float(r['truth_x_px'])<1920 and 0<=float(r['truth_y_px'])<1080
        assert all(math.isfinite(float(r[k])) for k in ['truth_x_px','truth_y_px'])
        assert r['sample_id'] not in old_ids and cid in train_clips and cid not in other_clips
        assert t==c['local_frame'] and int(r['source_frame'])==c['source_frame']==int(inventory[cid]['clip_first_source_frame'])+t
        assert sha(Path(c['source_image']))==c['source_image_sha256']
        # Explicit free-text user evidence takes precedence over untouched optional dropdown.
        excluded=c['case_id']=='B'
        if excluded:assert 'プレー終了後' in r['notes']
        accepted.append(dict(**r,case_id=c['case_id'],resolved_phase_status='post_play_user_note' if excluded else 'unknown',
            included_in_this_training=int(not excluded),exclusion_reason='User explicitly says post-play and unnecessary to track' if excluded else '',
            original_target_error_px=math.hypot(float(r['prediction_x_px'])-float(r['truth_x_px']),float(r['prediction_y_px'])-float(r['truth_y_px']))))
        if excluded:continue
        template=next(x for x in old_train if x['candidate_id']==cid)
        frame_dir=Path(c['source_image']).parent;frames=[frame_dir/f'{j:06d}.jpg' for j in range(t-3,t+1)]
        hashes=[sha(p) for p in frames]
        assert not set(hashes)&other_hashes and str(frames[-1])==c['source_image']
        additions.append(dict(**r,finetune_role='train',split_group=template['split_group'],
            input_frames_json=json.dumps([str(p) for p in frames]),input_frame_sha256_json=json.dumps(hashes),
            median_path=template['median_path'],median_sha256=template['median_sha256'],input_ends_at_labelled_frame=1))
    assert len(additions)==3 and {cases[r['sample_id']]['case_id'] for r in additions}=={'A','C','D'}
    write(OUT/'accepted_center_annotations.csv',accepted)
    write(OUT/'expanded_train_labels.csv',harmonize(old_train+additions))
    # Same clip and visible-label counts in both arms, without using validation outcomes.
    chooser=np.random.default_rng(SEED+1);duplicates=[]
    for r in additions:
        pool=[x for x in old_train if x['candidate_id']==r['candidate_id'] and x['truth_visibility']=='visible']
        duplicates.append(pool[int(chooser.integers(len(pool)))])
    write(OUT/'control_train_slots.csv',old_train+duplicates)
    hardfile=DATA/'training_hard_cases/far_wrong_target_predictions.csv'
    hard_ids={r['sample_id'] for r in read(hardfile)}
    assert len(hard_ids)==2 and hard_ids<={r['sample_id'] for r in old_train}
    protected[str(hardfile)]=sha(hardfile)
    report=dict(status='four_human_centers_ingested_three_used',human_center_count=4,training_added=3,
        excluded_case='B',excluded_reason='User note explicitly says post-play; no need to track it',
        raw_annotations_preserved=True,all_saved_events_match_csv=True,source_clocks_checked=True,
        new_training_input_hashes_disjoint_from_validation_and_audit=True,phase_of_other_three='unknown; no in-play claim',
        selected_active_learning_cases_not_evaluation=True,labels_sha256=sha(ANNOT/'manual_center_labels.csv'),
        event_log_sha256=sha(ANNOT/'annotation_events.jsonl'),expanded_labels_sha256=sha(OUT/'expanded_train_labels.csv'))
    dump(OUT/'annotation_ingestion.json',report)
    return old_train,additions,duplicates,validation,hard_ids,protected,report

def main():
    started=time.monotonic()
    old_train,additions,duplicates,val_rows,hard_ids,protected,ingestion=ingest()
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth'
    protected[str(weight)]=sha(weight)
    protocol=dict(status='fixed_before_training',seed=SEED,epochs=3,batch_size=2,steps_per_arm=186,
        base_unique_training_labels=120,expanded_unique_training_labels=123,slots_per_arm=123,
        extra_control_labels=[r['sample_id'] for r in duplicates],extra_expansion_labels=[r['sample_id'] for r in additions],
        training_order='Same permutation of123 positions for each arm, reset seed and original weights per arm',
        loss_weights='Both: old hard2 weight8, other base118 weight1, last3 slots weight8; mean-normalise by158/123. Fixed8 reused, not tuned.',
        control='Last3 slots repeat old visible manual labels from the same3 clips; expansion uses new A/C/D centers. Same clip/visibility exposure and update count.',
        excluded='B visible center retained as annotation but excluded because user states post-play',
        learning_rate=1e-4,optimizer='Adam',gradient_clip_norm=1.,trainable=['up_block_3','predictor'],
        all_other_parameters_and_batchnorm_buffers_frozen=True,input='median+BGRt-3..t,512x288,/255',
        target='Same sigma3.5 manual center Gaussian; not_visible zero map',augmentation=False,
        thresholds=THRESHOLDS,selection='Original included. Every original correct validation sample must remain correct; wrong_position+presenceFP<=3. Then max correct16px, fewer errors, earlier epoch. Control wins family ties.',
        audit='Only selected changed model evaluated after validation selection; if original selected, reuse unchanged original audit result, no repeated audit search.',
        scope='Fixed120 sparse validation labels include preparation/transitions; main-play-only scope unavailable here. No rally-only or external accuracy claim.',
        stop='One paired trial; no new thresholds, multipliers, seeds or fallback rules in response to results. No automatic operational adoption.',
        protected_input_hashes=protected,annotation_ingestion_sha256=sha(OUT/'annotation_ingestion.json'),
        script_sha256=sha(Path(__file__)),helper_sha256=sha(ROOT/'run_manual_ball_finetune.py'),
        decoding_helper_sha256=sha(ROOT/'run_hard_case_ball_finetune.py'))
    dump(OUT/'protocol.json',protocol)
    reset_seed();sys.path.insert(0,str(modelroot))
    from inference import BallInferencer
    from model.loss_utils import WBCELoss
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    net=infer.model;initial={n:t.detach().cpu().clone() for n,t in net.state_dict().items()}
    params=[];trainable=set()
    for name,p in net.named_parameters():
        enabled=name.startswith('up_block_3.') or name.startswith('predictor.');p.requires_grad_(enabled)
        if enabled:params.append(p);trainable.add(name)
    medians={};base=prepare(old_train,medians);extra=prepare(additions,medians);validation=prepare(val_rows,medians)
    by_id={s['row']['sample_id']:s for s in base}
    control=base+[by_id[r['sample_id']] for r in duplicates];expanded=base+extra
    sample_weights=np.array([8. if s['row']['sample_id'] in hard_ids else 1. for s in base]+[8.]*3,dtype=np.float32)
    assert sample_weights.sum()==158 and len(control)==len(expanded)==123
    baseline_rows=decode(infer,infer_maps(infer,validation,medians),validation,.3,'original',0,'validation')
    baseline=metrics(baseline_rows);original_ids=good_ids(baseline_rows)
    assert len(original_ids)==80 and err(baseline)==3
    previous={r['sample_id']:r for r in read(ROOT/'balltrack_hard_case_finetune_pilot/validation_original.csv')}
    assert all(int(r['visible'])==int(previous[r['sample_id']]['visible']) and
        abs(float(r['x_px'])-float(previous[r['sample_id']]['x_px']))<=2 and abs(float(r['y_px'])-float(previous[r['sample_id']]['y_px']))<=2 for r in baseline_rows)
    write(OUT/'validation_original.csv',baseline_rows)
    original=dict(family='original',epoch=0,threshold=.3,checkpoint=str(weight),checkpoint_sha256=sha(weight),metrics=baseline)
    bests={f:dict(original) for f in ['control','expanded']};grid=[];history=[]
    for family,samples in [('control',control),('expanded',expanded)]:
        reset_seed();net.load_state_dict(initial,strict=True);optimizer=torch.optim.Adam(params,lr=1e-4,weight_decay=0.)
        scaler=torch.amp.GradScaler('cuda');generator=np.random.default_rng(SEED);updates=0
        for epoch in range(1,4):
            net.eval();order=generator.permutation(len(samples));losses=[]
            for first in range(0,len(order),2):
                ids=order[first:first+2];batch=[samples[i] for i in ids];inputs,targets=tensors(batch,medians,infer.device,targets=True)
                weights=torch.from_numpy(sample_weights[ids]).to(infer.device);optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):prediction,_,_=net.forward(frames=inputs)
                losses_frame=WBCELoss(prediction.float(),targets,reduce=False)
                assert losses_frame.shape==weights.shape
                loss=(losses_frame*weights).mean()/float(sample_weights.mean());assert torch.isfinite(loss)
                scaler.scale(loss).backward();scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(params,1.)
                assert torch.isfinite(norm)
                scaler.step(optimizer);scaler.update();updates+=1;losses.append(float(loss.detach()))
                if first%40==0:print(family,'epoch',epoch,'slot',first,'loss',losses[-1],flush=True)
            path=OUT/f'{family}_epoch{epoch:02d}.pth'
            torch.save(dict(state_dict={n:t.detach().cpu() for n,t in net.state_dict().items()},meta=dict(family=family,epoch=epoch,protocol_sha256=sha(OUT/'protocol.json'))),path)
            maps=infer_maps(infer,validation,medians)
            for threshold in THRESHOLDS:
                rows=decode(infer,maps,validation,threshold,family,epoch,'validation');m=metrics(rows);lost=original_ids-good_ids(rows)
                feasible=not lost and err(m)<=3
                grid.append(dict(family=family,epoch=epoch,threshold=threshold,correct=m['counts']['localized_tp_16px'],
                    errors=err(m),wrong_position=m['counts']['wrong_localization'],presence_fp=m['counts']['presence_fp'],
                    original_correct_lost=len(lost),lost_sample_ids=';'.join(sorted(lost)),feasible=int(feasible)))
                write(OUT/f'validation_{family}_epoch{epoch:02d}_threshold{int(threshold*100):02d}.csv',rows)
                best=bests[family]
                if feasible and (m['counts']['localized_tp_16px'],-err(m))>(best['metrics']['counts']['localized_tp_16px'],-err(best['metrics'])):
                    bests[family]=dict(family=family,epoch=epoch,threshold=threshold,checkpoint=str(path.resolve()),checkpoint_sha256=sha(path),metrics=m)
            history.append(dict(family=family,epoch=epoch,loss=float(np.mean(losses)),updates=updates,checkpoint=str(path.resolve()),checkpoint_sha256=sha(path)))
            dump(OUT/'training_history.json',history)
            print('validation',family,epoch,json.dumps([r for r in grid if r['family']==family and r['epoch']==epoch]),flush=True)
        assert updates==186
        assert all(torch.equal(net.state_dict()[n].detach().cpu(),t) for n,t in initial.items() if n not in trainable)
    write(OUT/'validation_selection_grid.csv',grid)
    choice=bests['control']
    if (bests['expanded']['metrics']['counts']['localized_tp_16px'],-err(bests['expanded']['metrics']))>(choice['metrics']['counts']['localized_tp_16px'],-err(choice['metrics'])):
        choice=bests['expanded']
    dump(OUT/'validation_selection.json',dict(family_choices=bests,selected=choice,audit_used_for_selection=False,automatic_adoption=False))
    old_audit=metrics(read(ROOT/'balltrack_hard_case_finetune_pilot/audit_original.csv'))
    if choice['family']=='original':
        audit_result=old_audit;audit_lost=[];audit_mode='Original retained; previously computed unchanged original audit reused'
    else:
        audit=prepare(read(DATA/'audit_labels.csv'),medians)
        net.load_state_dict(torch.load(choice['checkpoint'],map_location='cpu',weights_only=False)['state_dict'],strict=True)
        rows=decode(infer,infer_maps(infer,audit,medians),audit,choice['threshold'],choice['family'],choice['epoch'],'audit')
        write(OUT/'audit_selected.csv',rows);audit_result=metrics(rows)
        audit_lost=sorted(good_ids(read(ROOT/'balltrack_hard_case_finetune_pilot/audit_original.csv'))-good_ids(rows))
        audit_mode='Selected checkpoint only, after immutable validation choice'
    assert all(sha(Path(p))==h for p,h in protected.items())
    result=dict(status='fixed_three_center_expansion_comparison_complete',human_annotations=4,new_training_centers=3,
        excluded_post_play='B',original_validation=baseline,matched_epoch2_threshold03=[r for r in grid if r['epoch']==2 and r['threshold']==.3],
        family_choices=bests,validation_selected=choice,audit_original=old_audit,audit_selected=audit_result,audit_mode=audit_mode,
        audit_original_correct_lost=audit_lost,audit_nonregression=not audit_lost and err(audit_result)<=err(old_audit),
        operational_model_changed=False,automatic_adoption=False,original_labels_models_reviewed_outputs_unchanged=True,
        frozen_parameters_and_batchnorm_buffers_unchanged=True,training_slots_steps_weights_matched=True,
        scope='Sparse same-match development; not in-play-only or independent external accuracy',elapsed_seconds=time.monotonic()-started)
    dump(OUT/'decision.json',result);print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
