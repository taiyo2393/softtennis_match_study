"""Can the existing head fit nine known positive/negative point pairs at all?

Training-only diagnostic, never an evaluated deployment candidate.
"""
import json,sys,math
from pathlib import Path
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha
from run_manual_ball_finetune import prepare,tensors
from train_point_supervised_ball_detector_fp32 import point_value,reset_seed

OUT=ROOT/'ball_point_supervision_sanity';PAIRED=ROOT/'ball_point_supervised_detector_fp32'
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
    negative=read(PAIRED/'negative_point_manifest.csv');positive={r['sample_id']:r for r in read(PAIRED/'positive_training_manifest.csv')}
    rows=[positive[r['positive_sample_id']] for r in negative];assert len(rows)==9 and len({r['sample_id'] for r in rows})==9
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth'
    protected={str(p):sha(p) for p in [weight,PAIRED/'negative_point_manifest.csv',PAIRED/'positive_training_manifest.csv',PAIRED/'decision.json',PAIRED/'verification.json',
        ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json']}
    protocol=dict(status='fixed_before_training_only_sanity_test',motivation='Paired full-data point-negative training did not suppress even its9supervised negative centers below0.3. Check learnability before asking for new annotation.',
        images=9,positive_points=9,negative_points=9,source_clips=5,input='Original causal median+BGRt-3..t,512x288',
        loss='Equal per-frame positive-center BCE and negative-center BCE, each0.5; all9frames have BOTHlabels. No teacher/preservation, no full-frame background/absence GT.',
        trainable=['up_block_3','predictor'],frozen='All other weights and allBatchNorm buffers',precision='FP32',
        steps=100,batch_size=3,learning_rate=1e-4,optimizer='Adam',gradient_clip=1.,seed=20261103,
        checkpoints='Final100steps only; no early stop or step/threshold tuning',
        success_measure='Training-only bilinear positive response>0.3 and negative response<=0.3. NOT object rejection, full detector output, generalization, or independent validation.',
        no_validation_or_audit=True,no_model_selection=True,no_deployment_candidate=True,
        rationale='This intentionally removes full-data sampling and preservation constraints; it is a diagnostic of fitting, not a matched ablation identifying which constraint caused failure.',
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');write(OUT/'training_labels.csv',[dict(p,negative_x=n['candidate_x_px'],negative_y=n['candidate_y_px']) for p,n in zip(rows,negative)])
    reset_seed();sys.path.insert(0,str(modelroot));from inference import BallInferencer
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=3);net=infer.model
    initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};params=[];names=set()
    for name,p in net.named_parameters():
        enabled=name.startswith('up_block_3.') or name.startswith('predictor.');p.requires_grad_(enabled)
        if enabled:params.append(p);names.add(name)
    medians={};samples=prepare(rows,medians);optimizer=torch.optim.Adam(params,lr=1e-4);generator=np.random.default_rng(20261103);order=[];history=[]
    def scores(step):
        data=[]
        with torch.inference_mode():
            for start in range(0,9,3):
                p,_,_=net.forward(frames=tensors(samples[start:start+3],medians,infer.device))
                for j in range(len(p)):
                    i=start+j;r=rows[i];n=negative[i]
                    data.append(dict(step=step,sample_id=r['sample_id'],positive=float(point_value(p[j:j+1],r['truth_x_px'],r['truth_y_px'])),negative=float(point_value(p[j:j+1],n['candidate_x_px'],n['candidate_y_px']))))
        return data
    net.eval();initial_scores=scores(0)
    for step in range(1,101):
        if len(order)<3:order.extend(generator.permutation(9).tolist())
        ids=order[:3];order=order[3:];batch=[samples[i] for i in ids];optimizer.zero_grad(set_to_none=True)
        p,_,_=net.forward(frames=tensors(batch,medians,infer.device));terms=[]
        for j,i in enumerate(ids):
            r=rows[i];n=negative[i];pos=point_value(p[j:j+1],r['truth_x_px'],r['truth_y_px']).clamp(1e-6,1-1e-6)
            neg=point_value(p[j:j+1],n['candidate_x_px'],n['candidate_y_px']).clamp(1e-6,1-1e-6);terms.append(.5*(-pos.log()-torch.log1p(-neg)))
        loss=torch.stack(terms).mean();assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm);optimizer.step()
        history.append(dict(step=step,loss=float(loss.detach()),gradient_norm_before_clip=float(norm)))
        if step%20==0:print('sanity step',step,'loss',history[-1]['loss'],flush=True)
    final_scores=scores(100);write(OUT/'point_scores.csv',initial_scores+final_scores);write(OUT/'training_history.csv',history)
    assert all(torch.equal(net.state_dict()[k].detach().cpu(),v) for k,v in initial.items() if k not in names)
    torch.save(dict(trainable_state_dict={k:net.state_dict()[k].detach().cpu() for k in sorted(names)},meta=dict(diagnostic_only=True,steps=100,protocol_sha256=sha(OUT/'protocol.json'))),OUT/'diagnostic_only_delta.pth')
    assert all(sha(Path(p))==h for p,h in protected.items())
    result=dict(status='training_only_sanity_complete',images=9,steps=100,
        initial_positive_above03=sum(r['positive']>.3 for r in initial_scores),initial_negative_above03=sum(r['negative']>.3 for r in initial_scores),
        final_positive_above03=sum(r['positive']>.3 for r in final_scores),final_negative_above03=sum(r['negative']>.3 for r in final_scores),
        final_correct_point_pairs=sum(r['positive']>.3 and r['negative']<=.3 for r in final_scores),
        mean_final_positive=float(np.mean([r['positive'] for r in final_scores])),mean_final_negative=float(np.mean([r['negative'] for r in final_scores])),
        protected_inputs_unchanged=True,validation_or_audit_used=False,no_model_selection=True,automatic_adoption=False,operational_outputs_changed=False,
        interpretation='Training-point fit only. Removing preservation and changing exposure simultaneously does not identify a single cause of the full-data failure.')
    (OUT/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
