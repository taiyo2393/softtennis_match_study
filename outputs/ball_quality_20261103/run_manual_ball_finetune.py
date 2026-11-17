"""One fixed supervised fine-tuning pilot on manually clicked source centers.

Train8clips120labels, choose on8otherclips120labels, then audit4clips60labels.
Only final decoder block and predictor learn; all BN running statistics fixed.
No source weights, labels, reviewed overrides or inference thresholds changed.
"""
import json,sys,time,random,math
from pathlib import Path
from collections import Counter
import numpy as np,torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2

OUT=ROOT/'balltrack_manual_finetune_pilot'
SEED=20261103
EPOCHS=3

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def prepare(rows,medians):
    samples=[]
    for i,r in enumerate(rows):
        images=[]
        for path,h in zip(json.loads(r['input_frames_json']),json.loads(r['input_frame_sha256_json']),strict=True):
            assert sha(Path(path))==h
            im=cv2.imread(path);assert im is not None and im.shape[:2]==(1080,1920)
            images.append(cv2.resize(im,(512,288)))
        mp=r['median_path']
        if mp not in medians:
            assert sha(Path(mp))==r['median_sha256']
            med=np.load(mp)['median'];assert med.shape[:2]==(1080,1920)
            medians[mp]=np.moveaxis(cv2.resize(med,(512,288)).astype(np.float32)/255.,-1,0)
        samples.append(dict(row=r,frames=np.stack(images),median_key=mp))
        if (i+1)%30==0:print('input cache',r['finetune_role'],i+1,'/',len(rows),flush=True)
    return samples

def tensors(samples,medians,device,targets=False):
    inputs=[];ys=[]
    for s in samples:
        images=np.moveaxis(s['frames'].astype(np.float32)/255.,-1,1)
        inputs.append(np.concatenate([medians[s['median_key']][None],images],axis=0).reshape(15,288,512))
        if targets:
            r=s['row']
            if r['truth_visibility']=='visible':
                x=int(float(r['truth_x_px'])*512/1920);y=int(float(r['truth_y_px'])*288/1080)
                yy,xx=np.ogrid[:288,:512]
                target=np.exp(-.5*((xx-x)**2+(yy-y)**2)/(3.5**2)).astype(np.float32)
            else:target=np.zeros((288,512),np.float32)
            ys.append(target[None])
    x=torch.from_numpy(np.stack(inputs)).to(device)
    return (x,torch.from_numpy(np.stack(ys)).to(device)) if targets else x

def metrics(rows):
    counts=Counter();errors=[]
    for r in rows:
        real=r['truth_visibility']=='visible';pred=int(r['visible'])==1
        counts['frames']+=1;counts['truth_visible']+=real;counts['pred_visible']+=pred
        if real and pred:
            error=math.hypot(float(r['x_px'])-float(r['truth_x_px']),float(r['y_px'])-float(r['truth_y_px']))
            errors.append(error);counts['presence_tp']+=1
            counts['localized_tp_16px']+=error<=16;counts['wrong_localization']+=error>16
        elif real:counts['presence_fn']+=1
        elif pred:counts['presence_fp']+=1
        else:counts['presence_tn']+=1
    for name in ['localized_tp_16px','wrong_localization','presence_fp','presence_fn']:counts[name]+=0
    return dict(counts=dict(counts),localization_recall_16px=counts['localized_tp_16px']/max(1,counts['truth_visible']),
        localization_precision_16px=counts['localized_tp_16px']/max(1,counts['pred_visible']),
        mean_error_both_visible_px=float(np.mean(errors)) if errors else None)

def key(m):
    c=m['counts'];return(c['localized_tp_16px'],-c['wrong_localization']-c['presence_fp'])

def evaluate(infer,samples,medians,epoch,role):
    infer.model.eval();rows=[]
    with torch.inference_mode():
        for first in range(0,len(samples),4):
            batch=samples[first:first+4];data=tensors(batch,medians,infer.device)
            with torch.amp.autocast('cuda'):maps,_,_=infer.model.forward(frames=data)
            for s,hm in zip(batch,maps[:,0].float().cpu().numpy()):
                x,y,w,h,conf=infer._predict_location(hm)
                px=int((x+w/2)*1920/512);py=int((y+h/2)*1080/288)
                r=s['row'];rows.append(dict(sample_id=r['sample_id'],candidate_id=r['candidate_id'],local_frame=r['local_frame'],
                    source_frame=r['source_frame'],role=role,epoch=epoch,truth_visibility=r['truth_visibility'],
                    truth_x_px=r['truth_x_px'],truth_y_px=r['truth_y_px'],x_px=px,y_px=py,visible=int(px!=0 or py!=0),confidence=conf,
                    temporal_alignment='input_ends_at_output_frame'))
    write(OUT/f'{role}_epoch{epoch:02d}_predictions.csv',rows)
    return metrics(rows),rows

def main():
    assert not (OUT/'training_protocol.json').exists(),'One fixed trial; never overwrite a completed plan.'
    manifest=load(OUT/'data_manifest.json')
    assert all(sha(Path(p))==h for p,h in manifest['input_hashes'].items())
    for role,h in manifest['partition_hashes'].items():assert sha(OUT/f'{role}_labels.csv')==h
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth'
    initial_hash=sha(weight)
    protocol=dict(status='fixed_before_training',seed=SEED,epochs=EPOCHS,batch_size=2,validation_batch_size=4,
        optimizer='Adam',learning_rate=1e-4,weight_decay=0.,gradient_clip_norm=1.,
        trainable=['up_block_3','predictor'],frozen='All other parameters and ALL BatchNorm running statistics',
        loss='Unchanged source WBCELoss on final heatmap, FP32 reduction; sigma3.5 Gaussian centered at int(manual_x*512/1920), int(manual_y*288/1080); stored not_visible ->zero heatmap',
        input='median+BGRt-3..t, /255,512x288; same source JPEGs as manual click interface; no off-by-one shift',
        transforms='No mixup,flips,crops,rotations,augmentation or threshold tuning',
        model_inference_threshold=.3,decoder='unchanged infer._predict_location largest component bounding-box center',
        localization_radius_source_px=16.,
        partition_manifest_sha256=sha(OUT/'data_manifest.json'),manual_provenance_sha256=sha(OUT/'manual_annotation_provenance.json'),
        checkpoint_selection='Include epoch0 original. Max validation localizedTP16px, then fewer(wrong_localization+presenceFP), ties earliest. Audit labels never used for selection.',
        audit='Only after epoch selection, evaluate60labels on002/008/016/018. Those clips do not enter training or validation.',
        phase_scope='Stored manual labels include preparation and transitions; aggregate is not in-play-only accuracy',
        data_reuse_warning='These clips have been seen in previous experiments; clip separation prevents frame leakage but is not untouched external testing',
        automatic_adoption=False,source_weights_sha256=initial_hash,script_sha256=sha(Path(__file__)),
        checkpoint_format='New local full state_dict snapshots; never modify pretrained checkpoint or reviewed output registries')
    dump(OUT/'training_protocol.json',protocol)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False
    sys.path.insert(0,str(modelroot));from inference import BallInferencer
    from model.loss_utils import WBCELoss
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    net=infer.model;net.eval()
    initial_state={name:t.detach().cpu().clone() for name,t in net.state_dict().items()}
    names=[];params=[]
    for name,p in net.named_parameters():
        enabled=name.startswith('up_block_3.') or name.startswith('predictor.')
        p.requires_grad_(enabled)
        if enabled:names.append(name);params.append(p)
    dump(OUT/'trainable_parameters.json',dict(names=names,count=sum(p.numel() for p in params),batchnorm_running_statistics_fixed=True))
    medians={};train=prepare(read(OUT/'train_labels.csv'),medians);validation=prepare(read(OUT/'validation_labels.csv'),medians)
    baseline,rows=evaluate(infer,validation,medians,0,'validation')
    # The fresh original model must reproduce the timing-aligned baseline at annotated frames.
    base=ROOT.parent/'tracking_quality_20261102/ball_aligned';baseline_tables={}
    for r in rows:
        cid=r['candidate_id']
        if cid not in baseline_tables:baseline_tables[cid]=read(base/cid/'balltrack_predictions.csv')
        old=baseline_tables[cid][int(r['local_frame'])]
        assert int(old['visible'])==r['visible'] and math.hypot(float(old['x_px'])-r['x_px'],float(old['y_px'])-r['y_px'])<=2
    history=[dict(epoch=0,validation=baseline,training_loss=None)]
    best_epoch=0;best_score=key(baseline);best_file=weight;best_hash=initial_hash
    optimizer=torch.optim.Adam(params,lr=1e-4,weight_decay=0.)
    scaler=torch.amp.GradScaler('cuda');clock=time.monotonic();generator=np.random.default_rng(SEED)
    print('epoch0 validation',json.dumps(baseline),flush=True)
    for epoch in range(1,EPOCHS+1):
        net.eval();order=generator.permutation(len(train));losses=[]
        for first in range(0,len(order),2):
            batch=[train[i] for i in order[first:first+2]];data,target=tensors(batch,medians,infer.device,targets=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):prediction,_,_=net.forward(frames=data)
            loss=WBCELoss(prediction.float(),target)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
            if first%40==0:print('epoch',epoch,'train',first,'/',len(train),'loss',losses[-1],flush=True)
        score,_=evaluate(infer,validation,medians,epoch,'validation')
        snapshot=OUT/f'epoch{epoch:02d}.pth'
        torch.save(dict(state_dict={name:t.detach().cpu() for name,t in net.state_dict().items()},
            meta=dict(epoch=epoch,source_weights_sha256=initial_hash,training_protocol_sha256=sha(OUT/'training_protocol.json'))),snapshot)
        if key(score)>best_score:
            best_score=key(score);best_epoch=epoch;best_file=snapshot;best_hash=sha(snapshot)
        history.append(dict(epoch=epoch,training_loss=float(np.mean(losses)),validation=score,checkpoint=str(snapshot.resolve()),checkpoint_sha256=sha(snapshot)))
        dump(OUT/'training_history.json',history)
        print('epoch',epoch,'validation',json.dumps(score),'selected',best_epoch,flush=True)
    # Frozen parameters AND BN buffers must not have changed.
    frozen_names=[name for name in initial_state if name not in names]
    assert all(torch.equal(net.state_dict()[name].detach().cpu(),initial_state[name]) for name in frozen_names)
    selection=dict(selected_epoch=best_epoch,selected_checkpoint=str(best_file.resolve()),selected_checkpoint_sha256=best_hash,
        validation_selection_key=list(best_score),baseline_key=list(key(baseline)),audit_results_used=False,
        automatic_adoption=False,source_model_unchanged=sha(weight)==initial_hash)
    dump(OUT/'checkpoint_selection.json',selection)
    # After the choice is immutable, read/cache audit rows and evaluate original+selected.
    audit=prepare(read(OUT/'audit_labels.csv'),medians)
    net.load_state_dict(initial_state,strict=True)
    audit_base,_=evaluate(infer,audit,medians,0,'audit_original')
    if best_epoch:
        selected_state=torch.load(best_file,map_location='cpu',weights_only=False)['state_dict']
        net.load_state_dict(selected_state,strict=True)
    audit_selected,_=evaluate(infer,audit,medians,best_epoch,'audit_selected')
    assert all(sha(Path(p))==h for p,h in manifest['input_hashes'].items())
    assert sha(weight)==initial_hash and sha(best_file)==best_hash
    for role,h in manifest['partition_hashes'].items():assert sha(OUT/f'{role}_labels.csv')==h
    selected_validation=next(r['validation'] for r in history if r['epoch']==best_epoch)
    decision=dict(status='fixed_manual_supervised_pilot_complete_not_adopted',automatic_adoption=False,
        selected_epoch=best_epoch,validation_original=baseline,validation_selected=selected_validation,
        audit_original=audit_base,audit_selected=audit_selected,
        audit_key_nonworse=key(audit_selected)>=key(audit_base),
        source_and_reviewed_outputs_unchanged=True,frozen_parameters_and_batchnorm_buffers_unchanged=True,
        train_validation_audit_clips_disjoint=True,training_seconds=time.monotonic()-clock,
        manual_annotation_source_confirmed=True,independent_external_accuracy=False,
        next='Review validation/audit changes before any operational adoption. Do not retune on audit or overwrite the reviewed ball/player outputs.')
    dump(OUT/'decision.json',decision);print(json.dumps(decision,indent=2))

if __name__=='__main__':main()
