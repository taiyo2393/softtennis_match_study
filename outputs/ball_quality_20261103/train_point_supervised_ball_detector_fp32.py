"""Matched positive-only vs positive+confirmed-negative point training.

Unlabelled pixels have only original-model preservation, never background GT.
"""
import argparse,json,sys,random,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2
from run_manual_ball_finetune import prepare,tensors

OUT=ROOT/'ball_point_supervised_detector_fp32';DATA=ROOT/'balltrack_manual_finetune_pilot';REVIEW=ROOT/'ball_appearance_training_review';SEED=20261103
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def reset_seed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.backends.cudnn.benchmark=False
def coordinates(x,y,width=512,height=288):
    return max(0.,min(width-1.,float(x)*width/1920)),max(0.,min(height-1.,float(y)*height/1080))
def point_value(hm,x,y):
    qx,qy=coordinates(x,y,hm.shape[-1],hm.shape[-2])
    grid=torch.tensor([[[[2*qx/(hm.shape[-1]-1)-1,2*qy/(hm.shape[-2]-1)-1]]]],device=hm.device,dtype=torch.float32)
    return F.grid_sample(hm,grid,mode='bilinear',align_corners=True)[0,0,0,0]
def support_mask(points):
    mask=torch.ones((288,512),dtype=torch.bool)
    for x,y in points:
        qx,qy=coordinates(x,y)
        for ix in {math.floor(qx),math.ceil(qx)}:
            for iy in {math.floor(qy),math.ceil(qy)}:mask[iy,ix]=False
    return mask
def preservation_loss(pred,teacher,mask):
    # Soft original-model outputs are a prior, not verified ball/background labels.
    pixel=F.binary_cross_entropy(pred.clamp(1e-6,1-1e-6),teacher,reduction='none')
    values=[]
    for i in range(len(pred)):
        known=mask[i,0];q=teacher[i,0];loss=pixel[i,0];terms=[]
        for chosen in [known&(q>=.1),known&(q<.1)]:
            terms.append(loss[chosen].mean() if chosen.any() else loss.sum()*0.)
        values.append(.5*(terms[0]+terms[1]))
    return torch.stack(values).mean()

def self_test():
    hm=torch.zeros((1,1,288,512),dtype=torch.float32);hm[0,0,100,100]=1.
    assert abs(float(point_value(hm,375,375))-1)<1e-5
    mask=support_mask([(375,375)]);assert (~mask).sum()==1 and not mask[100,100]
    assert coordinates(1919,1079)==(511.,287.) and coordinates(0,0)==(0.,0.)
    mask=support_mask([(376,376)]);assert (~mask).sum()==4
    p=torch.full((1,1,288,512),.2,requires_grad=True);q=p.detach().clone();m=torch.ones_like(p,dtype=torch.bool)
    loss=preservation_loss(p,q,m);loss.backward();assert torch.isfinite(loss) and p.grad.abs().max()<1e-6
    p=torch.full((1,1,288,512),.5,requires_grad=True);v=point_value(p,376,376);(-torch.log(v)).backward()
    assert (p.grad!=0).sum()<=4 and (p.grad!=0).sum()>0
    print('Passed bilinear point geometry, exact four-pixel support, boundaries, soft-target stationarity and sparse positive gradient. Synthetic tensors only.')

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Preserve fixed trial; no retuning in place.'
    train_path=ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv';rows=[r for r in read(train_path) if r['truth_visibility']=='visible']
    assert len(rows)==102 and len({r['sample_id'] for r in rows})==102
    by_time={(r['candidate_id'],int(r['local_frame'])):r for r in rows};negatives=read(REVIEW/'candidate_review_labels.csv');cases=load(REVIEW/'cases.json')
    assert len(negatives)==9 and all(r['object_label']=='not_ball' and r['reviewer_id']=='human_user' and r['reviewed_at_utc'] for r in negatives)
    completed=load(REVIEW/'completed_review.json');assert sha(REVIEW/'candidate_review_labels.csv')==completed['labels_sha256']
    assert sha(REVIEW/'annotation_events.jsonl')==completed['events_sha256']
    neg_by=defaultdict(list);neg_manifest=[]
    for r in negatives:
        cid=r['candidate_id'];t=int(r['local_frame']);truth=by_time[cid,t];c=next(c for c in cases if c['sample_id']==r['sample_id'])
        x,y=float(r['candidate_x_px']),float(r['candidate_y_px'])
        assert math.hypot(x-float(truth['truth_x_px']),y-float(truth['truth_y_px']))>=64
        assert int(r['source_frame'])==int(truth['source_frame'])==c['source_frame']
        assert sha(Path(c['source_image']))==c['source_image_sha256']
        neg_by[truth['sample_id']].append((x,y));neg_manifest.append(dict(r,positive_sample_id=truth['sample_id'],supervision='only_bilinear_center_target0'))
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth'
    paths=[train_path,REVIEW/'candidate_review_labels.csv',REVIEW/'annotation_events.jsonl',REVIEW/'cases.json',REVIEW/'completed_review.json',weight,
        ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',
        SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv',DATA/'validation_labels.csv',DATA/'audit_labels.csv',
        ROOT/'ball_inplay_challenge/manual_center_labels.csv',ROOT/'ball_short_sequence_evaluation/manual_sequence_labels.csv',
        ROOT/'ball_040_object_swap_review/round2/user_reviewed/candidate_review.csv',ROOT/'run_manual_ball_finetune.py',modelroot/'model/tracknet_v3.py']
    protected={str(p):sha(p) for p in paths}
    protocol=dict(status='fixed_before_teacher_extraction_or_training',student_precision='FP32 with GradScaler disabled; matched numerical repair after original AMP first backward overflowed before any updates',teacher_precision='original CUDA AMP inference unchanged',seed=SEED,arms=['positive_control','point_negative'],primary_arm='point_negative',
        training_inputs=102,positive_manual_centers=102,confirmed_negative_points=9,negative_clips=5,not_visible_frames_used_for_training=0,
        point_supervision='Bilinear sampled model probability at manual center: target1. Explicit confirmed non-ball center: target0. No Gaussian map or whole-frame background/absence target.',
        preservation='Original frozen model probability at all other pixels: soft-target BCE, equally weight original q>=0.1 and q<0.1 groups per frame. Exclude only bilinear support pixels of positive and negative points. Same mask both arms.',
        loss='Mean positive point BCE + mean balanced preservation BCE + [0control/1primary]*(102/9)*batch-mean sum negative point BCE; overall epoch negative/positive mass balanced. Weights fixed1, no sweep.',
        teacher_caution='Teacher is a preservation prior, not ground truth; may preserve old mistakes. No claim that unknown pixels are ball-free.',
        matched_control='Same original weights,102inputs,permutations,point masks,teacher,153updates per arm. ONLY explicit negative-loss coefficient differs.',
        epochs=3,batch_size=2,steps_per_epoch=51,optimizer='Adam',learning_rate=1e-4,weight_decay=0.,gradient_clip_norm=1.,
        trainable=['up_block_3','predictor'],frozen='All other weights and ALL BatchNorm running buffers; model always eval',
        input='Original median+BGR t-3..t,512x288,/255. No stride2,tiles,augmentations.',
        checkpoint='Only FINAL epoch3 evaluated; no checkpoint or threshold selection; save trainable-weight deltas tied to source hash.',
        evaluation='Both fixed final arms, unchanged threshold0.3 and existing displacement0.1 selector; train123 and validation120, challenge4, short26 separately. Raw decoding also retained as diagnostic.',
        six_false='Both models on6previously confirmed false candidates BEFORE manual exclusion; true centers unknown. Never train on6.',
        audit='Primary arm only, if all original-correct points and error counts in train/validation/challenge/short are preserved, plus validation improves; never choose control/another epoch after failure.',
        repeated_data_warning='All same-match known development data, selected negatives9; no unseen-match generalization claim. Short26 includes4reused centers and2challenge anchors.',
        no_dense_background_gt=True,new_human_labels=0,hyperparameter_search=False,automatic_adoption=False,
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol);write(OUT/'negative_point_manifest.csv',neg_manifest);write(OUT/'positive_training_manifest.csv',rows)
    reset_seed();sys.path.insert(0,str(modelroot));from inference import BallInferencer
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    net=infer.model;initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};params=[];trainable=set()
    for name,param in net.named_parameters():
        enabled=name.startswith('up_block_3.') or name.startswith('predictor.');param.requires_grad_(enabled)
        if enabled:params.append(param);trainable.add(name)
    medians={};samples=prepare(rows,medians);teacher=[]
    with torch.inference_mode():
        for start in range(0,len(samples),4):
            with torch.amp.autocast('cuda'):p,_,_=net.forward(frames=tensors(samples[start:start+4],medians,infer.device))
            teacher.extend(p.detach().cpu().float())
    teacher=torch.stack(teacher);assert teacher.shape==(102,1,288,512)
    np.savez_compressed(OUT/'original_teacher_maps.npz',sample_ids=np.array([r['sample_id'] for r in rows]),heatmaps=teacher.numpy().astype(np.float16))
    masks=torch.stack([support_mask([(r['truth_x_px'],r['truth_y_px'])]+neg_by[r['sample_id']])[None] for r in rows]);point_before=[]
    for i,r in enumerate(rows):
        point_before.append(dict(sample_id=r['sample_id'],kind='manual_positive',source_x=r['truth_x_px'],source_y=r['truth_y_px'],original_probability=float(point_value(teacher[i:i+1],r['truth_x_px'],r['truth_y_px']))))
        for x,y in neg_by[r['sample_id']]:point_before.append(dict(sample_id=r['sample_id'],kind='confirmed_non_ball',source_x=x,source_y=y,original_probability=float(point_value(teacher[i:i+1],x,y))))
    write(OUT/'original_training_point_scores.csv',point_before);history=[];checkpoints={}
    for family,coefficient in [('positive_control',0.),('point_negative',1.)]:
        reset_seed();net.load_state_dict(initial,strict=True);optimizer=torch.optim.Adam(params,lr=1e-4,weight_decay=0.);scaler=torch.amp.GradScaler('cuda',enabled=False);generator=np.random.default_rng(SEED);updates=0
        for epoch in range(1,4):
            net.eval();order=generator.permutation(len(samples));loss_log=[]
            for first in range(0,len(order),2):
                indexes=order[first:first+2];batch=[samples[i] for i in indexes];data=tensors(batch,medians,infer.device);q=teacher[indexes].to(infer.device);mask=masks[indexes].to(infer.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda',enabled=False):p,_,_=net.forward(frames=data)
                p=p.float();pos=[];neg=[]
                for j,i in enumerate(indexes):
                    r=rows[i];positive=point_value(p[j:j+1],r['truth_x_px'],r['truth_y_px']).clamp(1e-6,1-1e-6);pos.append(-positive.log())
                    terms=[-torch.log1p(-point_value(p[j:j+1],x,y).clamp(1e-6,1-1e-6)) for x,y in neg_by[r['sample_id']]]
                    neg.append(torch.stack(terms).sum() if terms else p[j].sum()*0.)
                positive_loss=torch.stack(pos).mean();negative_loss=(102/9)*torch.stack(neg).mean();preserve=preservation_loss(p,q,mask)
                loss=positive_loss+preserve+coefficient*negative_loss;assert torch.isfinite(loss)
                scaler.scale(loss).backward();scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
                scaler.step(optimizer);scaler.update();updates+=1
                loss_log.append([float(x.detach()) for x in [loss,positive_loss,preserve,negative_loss]])
                if first%40==0:print(family,'epoch',epoch,'slot',first,'loss',loss_log[-1],flush=True)
            means=np.mean(loss_log,axis=0).tolist();history.append(dict(family=family,epoch=epoch,updates=updates,loss=means[0],positive_loss=means[1],preservation_loss=means[2],balanced_negative_loss=means[3]))
            dump(OUT/'training_history.json',history)
        assert updates==153
        assert all(torch.equal(net.state_dict()[name].detach().cpu(),value) for name,value in initial.items() if name not in trainable)
        delta={name:net.state_dict()[name].detach().cpu().clone() for name in sorted(trainable)};path=OUT/f'{family}_epoch03_delta.pth'
        torch.save(dict(trainable_state_dict=delta,meta=dict(family=family,epoch=3,updates=153,source_weight_sha256=sha(weight),protocol_sha256=sha(OUT/'protocol.json'))),path)
        # Reconstruct final model from original+delta; frozen state must remain exact.
        reconstructed=dict(initial);reconstructed.update(torch.load(path,map_location='cpu',weights_only=False)['trainable_state_dict'])
        assert all(torch.equal(net.state_dict()[name].detach().cpu(),value) for name,value in reconstructed.items())
        checkpoints[family]=dict(path=str(path),sha256=sha(path),epoch=3,updates=153,trainable_parameter_tensors=len(trainable),frozen_state_unchanged=True)
    assert all(sha(Path(p))==h for p,h in protected.items())
    dump(OUT/'trained_models.json',dict(status='two_final_epoch3_models_frozen_before_evaluation',checkpoints=checkpoints,source_weight_sha256=sha(weight),protected_inputs_unchanged=True))
    print(json.dumps(checkpoints,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    else:main()

