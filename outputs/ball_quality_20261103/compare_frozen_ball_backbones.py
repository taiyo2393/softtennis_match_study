"""Retrospective grouped comparison of hand features and frozen ImageNet features.

No downloads, no backbone fine tuning, no parameter sweep, no adopted changes.
Compare a common fixed linear head under the original016-only split and under
leave-one-source-clip-out. This is repeatedly seen, selected same-match data.
"""
import json
from pathlib import Path
from collections import Counter
import numpy as np,torch,torchvision
from torchvision.models import resnet18,ResNet18_Weights
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha,cv2
from fit_frozen_ball_appearance_probe import features as handcrafted

OUT=ROOT/'ball_pretrained_feature_comparison'
WEIGHTS=Path(r'C:\Users\ta239\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth')
BASE_GROUP='legacy_match1:match1_motion_pilot_016'

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def fit(x,y):
    assert set(y)=={0.,1.}
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.
    X=torch.from_numpy((x-mean)/scale);Y=torch.from_numpy(y)
    sw=torch.from_numpy(np.where(y==1,len(y)/(2*y.sum()),len(y)/(2*(1-y).sum())))
    w=torch.zeros(X.shape[1],dtype=torch.float64,requires_grad=True);bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
    optimiser=torch.optim.LBFGS([w,bias],lr=1.,max_iter=300,line_search_fn='strong_wolfe',tolerance_grad=1e-9,tolerance_change=1e-12)
    def closure():
        optimiser.zero_grad()
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(X@w+bias,Y,reduction='none')*sw).mean()+.5*w.square().sum()
        loss.backward();return loss
    optimiser.step(closure)
    return dict(mean=mean,scale=scale,weights=w.detach().numpy(),bias=np.array(float(bias.detach())))

def predict(model,x):
    logits=((x-model['mean'])/model['scale'])@model['weights']+model['bias']
    return torch.sigmoid(torch.from_numpy(logits)).numpy()

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Preserve grouped comparison snapshots.'
    cp=ROOT/'ball_candidate_cross_rally/user_reviewed/all_reviewed_candidate_cases.csv'
    corpus=read(cp);assert len(corpus)==36
    assert sha(cp)==load(cp.parent/'corpus_manifest.json')['all_cases_csv_sha256']
    labels=np.array([r['label']=='confirmed_ball_enclosed' for r in corpus],dtype=np.float64)
    groups=np.array([r['split_group'] for r in corpus]);group_names=sorted(set(groups))
    assert labels.sum()==21 and len(group_names)==4
    protected_paths=[cp,ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',
        ROOT/'ball_candidate_cross_rally/appearance_probe_v2/model.npz',ROOT/'ball_candidate_cross_rally/appearance_probe_v2/frozen_predictions.csv']
    protected={str(p):sha(p) for p in protected_paths}
    for r in corpus:
        assert sha(Path(r['patch96_path']))==r['patch96_sha256']
    protocol=dict(status='fixed_before_new_feature_extraction_and_fitting',
        scope='Retrospective comparison on36 explicitly reviewed, selected same-match candidates; all labels previously seen',
        corpus_sha256=sha(cp),class_counts=dict(real=21,false=15),groups=group_names,
        methods=['handcrafted672','frozen_resnet18_512'],
        handcrafted_source=str(ROOT/'fit_frozen_ball_appearance_probe.py'),
        handcrafted_code_sha256=sha(ROOT/'fit_frozen_ball_appearance_probe.py'),
        pretrained_weights=str(WEIGHTS),pretrained_weights_sha256=sha(WEIGHTS),pretrained_weights_kind='Torchvision ResNet18 IMAGENET1K_V1, cached local weights',
        torchvision_version=torchvision.__version__,backbone='eval,all parameters frozen,fc replaced with identity,512 global-average features',
        deep_input='Central32x32 native pixels from each96px patch, BGR->RGB; official weights.transforms():bilinear short-side256,center224, /255, ImageNet mean/std',
        image_context_warning='The32px input has only original detail; resizing is not recovery of lost pixels. Hand and deep preprocessing differ, so this is a method comparison, not a causal isolation of pretraining.',
        training='Same feature-wise training-only standardisation, class-balanced logistic mean loss +0.5*L2(weights);CPUfloat64,LBFGS300,strong_wolfe,zero initialisation',
        threshold=.5,score_semantics='Uncalibrated sigmoid scores',
        regimes=['train016_only_check_other8','leave_one_source_clip_out'],
        split_unit='Entire source clip; frame/patch random splits forbidden',
        hyperparameter_search=False,fine_tuning=False,new_download=False,
        automatic_adoption=False,minimum_safety='Any confirmed real-candidate loss or zero false-candidate rejection prevents adoption from this check alone; passing still requires new untouched evaluation',
        official_reference='https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html',
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    model=resnet18(weights=None)
    model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)
    model.fc=torch.nn.Identity();model.eval().requires_grad_(False)
    device=torch.device('cuda');model.to(device)
    transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
    images=[]
    for r in corpus:
        im=cv2.imread(r['patch96_path']);assert im.shape[:2]==(96,96)
        rgb=im[32:64,32:64,::-1].copy()
        images.append(transform(torch.from_numpy(rgb).permute(2,0,1)))
    learned=[]
    with torch.inference_mode():
        for start in range(0,len(images),12):
            learned.append(model(torch.stack(images[start:start+12]).to(device)).cpu().numpy())
    learned=np.concatenate(learned).astype(np.float64);assert learned.shape==(36,512) and np.isfinite(learned).all()
    manual=np.stack([handcrafted(Path(r['patch96_path'])) for r in corpus]);assert manual.shape==(36,672)
    matrices={'handcrafted672':manual,'frozen_resnet18_512':learned}
    np.savez_compressed(OUT/'features.npz',case_ids=np.array([r['case_id'] for r in corpus]),
        groups=groups,handcrafted672=manual,frozen_resnet18_512=learned)
    folds=[('train016_only_check_other8','other_clips',groups==BASE_GROUP,groups!=BASE_GROUP)]
    folds += [('leave_one_source_clip_out',g,groups!=g,groups==g) for g in group_names]
    predictions=[];folds_meta=[]
    for regime,heldout,train,test in folds:
        assert not (set(groups[train])&set(groups[test]))
        assert not (set(r['original_image_sha256'] for r,m in zip(corpus,train) if m)&set(r['original_image_sha256'] for r,m in zip(corpus,test) if m))
        for method,x in matrices.items():
            fitted=fit(x[train],labels[train]);scores=predict(fitted,x[test]);test_indices=np.where(test)[0]
            filename=f'{regime}__{heldout.replace(":","_")}__{method}.npz'
            np.savez_compressed(OUT/filename,**fitted)
            folds_meta.append(dict(regime=regime,heldout=heldout,method=method,train_cases=int(train.sum()),
                check_cases=int(test.sum()),train_groups=sorted(set(groups[train])),check_groups=sorted(set(groups[test])),
                fitted_head_file=filename,fitted_head_sha256=sha(OUT/filename)))
            for index,score in zip(test_indices,scores):
                r=corpus[index]
                predictions.append(dict(regime=regime,heldout_group=heldout,method=method,case_id=r['case_id'],
                    local_frame=r['local_frame'],ball_pilot_id=r['ball_pilot_id'],group=r['split_group'],
                    candidate_is_ball=int(labels[index]),uncalibrated_ball_score=float(score),keep=int(score>=.5)))
    # The hand-feature method under the old fixed split must reproduce its old frozen predictions.
    old=read(ROOT/'ball_candidate_cross_rally/appearance_probe_v2/frozen_predictions.csv')
    old_scores={r['case_id']:float(r['uncalibrated_ball_score']) for r in old}
    rerun=[r for r in predictions if r['regime']=='train016_only_check_other8' and r['method']=='handcrafted672']
    max_difference=max(abs(r['uncalibrated_ball_score']-old_scores[r['case_id'].split(':',1)[1]]) for r in rerun)
    assert max_difference<1e-6,max_difference
    write(OUT/'grouped_predictions.csv',predictions)
    summaries=[]
    for regime in protocol['regimes']:
        for method in protocol['methods']:
            rows=[r for r in predictions if r['regime']==regime and r['method']==method]
            real=[r for r in rows if r['candidate_is_ball']];false=[r for r in rows if not r['candidate_is_ball']]
            summaries.append(dict(regime=regime,method=method,reviewed_real_total=len(real),
                reviewed_real_kept=sum(r['keep'] for r in real),reviewed_false_total=len(false),
                reviewed_false_kept=sum(r['keep'] for r in false),
                missed_real_candidates=';'.join(r['case_id'] for r in real if not r['keep']),
                kept_false_candidates=';'.join(r['case_id'] for r in false if r['keep'])))
    write(OUT/'summary.csv',summaries)
    assert all(sha(Path(p))==h for p,h in protected.items())
    assert sha(WEIGHTS)==protocol['pretrained_weights_sha256']
    for p,h in load(ROOT/'play_start_audit/inventory_summary.json')['source_hashes'].items():assert sha(Path(p))==h
    dump(OUT/'folds.json',folds_meta)
    decision=dict(status='retrospective_grouped_comparison_complete_not_adopted',automatic_adoption=False,
        summary=summaries,original_hand_probe_reproduced_max_score_difference=max_difference,
        protected_inputs_and_labels_unchanged=True,backbone_weights_unchanged=True,
        no_frame_random_split=True,independent_accuracy_established=False,
        warning='Small selected, repeatedly inspected same-match cases and very unequal clip groups. No model is validated for full-rally automatic use by these results alone.')
    dump(OUT/'decision.json',decision);print(json.dumps(decision,indent=2))

if __name__=='__main__':main()
