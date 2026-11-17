"""One small image-only diagnostic classifier, fixed before new008 reviews.

No sklearn installation or external model weights; use gradient histograms+colour and
deterministic regularised logistic fitting in existing PyTorch. Not adopted.
"""
import json
from pathlib import Path
import torch
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha,cv2,np

OUT=ROOT/'ball_candidate_cross_rally/appearance_probe_v2'

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def gradient_histogram(gray):
    # Fixed unsigned9-bin gradient histograms,4x4 cells,2x2-cell L2-Hys blocks.
    # Linear interpolation in orientation; hard cell membership, no spatial interpolation.
    gy,gx=np.gradient(gray.astype(np.float64)/255.)
    magnitude=np.hypot(gx,gy);angle=np.mod(np.arctan2(gy,gx),np.pi)*9/np.pi
    lo=np.floor(angle).astype(int)%9;fraction=angle-np.floor(angle)
    cell=gray.shape[0]//4;hist=np.zeros((4,4,9))
    for y in range(4):
        for x in range(4):
            region=np.s_[y*cell:(y+1)*cell,x*cell:(x+1)*cell]
            bins=lo[region].ravel();m=magnitude[region].ravel();f=fraction[region].ravel()
            hist[y,x]=np.bincount(bins,weights=m*(1-f),minlength=9)+np.bincount((bins+1)%9,weights=m*f,minlength=9)
    blocks=[]
    for y in range(3):
        for x in range(3):
            b=hist[y:y+2,x:x+2].ravel();b=b/np.sqrt(np.sum(b*b)+1e-10)
            b=np.minimum(b,.2);b=b/np.sqrt(np.sum(b*b)+1e-10);blocks.extend(b)
    return np.array(blocks)

def features(path):
    im=cv2.imread(str(path));assert im is not None and im.shape[:2]==(96,96)
    values=[]
    for size in (16,32):
        p=im[48-size//2:48+size//2,48-size//2:48+size//2]
        gray=p[:,:,0]*.114+p[:,:,1]*.587+p[:,:,2]*.299
        values.extend(gradient_histogram(gray))
    for size in (4,8,16,32):
        p=im[48-size//2:48+size//2,48-size//2:48+size//2].astype(np.float64)/255.
        values.extend(p.mean((0,1)));values.extend(p.std((0,1)))
    values=np.asarray(values,dtype=np.float64);assert values.shape==(672,) and np.isfinite(values).all()
    return values

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Do not tune/overwrite the fixed check.'
    train_path=ROOT/'ball_044_reviewed_appearance_cases/cases.csv'
    eval_path=ROOT/'ball_candidate_cross_rally/cases.csv'
    cases_manifest=load(eval_path.parent/'case_manifest.json')
    train=read(train_path);evaluation=read(eval_path)
    assert len(train)==28 and len(evaluation)==8
    assert {r['split_group'] for r in train}=={'legacy_match1:rally044'}
    assert not ({r['split_group'] for r in train}&{r['split_group'] for r in evaluation})
    assert not ({r['original_image_sha256'] for r in train}&{r['original_image_sha256'] for r in evaluation})
    protected=cases_manifest['input_hashes']
    assert all(sha(Path(p))==h for p,h in protected.items())
    protocol=dict(status='fixed_before_training_and_new008_answers',training_rows=28,
        training_label_counts=dict(real=18,false=10),training_groups=['legacy_match1:rally044'],
        prediction_cases=[r['case_id'] for r in evaluation],new_answers_pending=[r['case_id'] for r in evaluation if r['label']=='pending_user_confirmation'],
        features='672 image-only values: NumPy unsigned9-bin gradient histograms at16/32px (4x4cells,2x2cell blocks,L2-Hys clip0.2,orientation interpolation,hard cell membership), BGR mean/std for4/8/16/32px central squares',
        excluded_features=['absolute_xy','timestamp','frame_id','source_clip_id','detector_score','temporal_trace','review_outcome_of_prediction_cases'],
        preprocessing='Standardise each feature using training mean/std only; constants use scale1',
        model='Linear logistic with intercept; weighted binary cross entropy mean plus0.5*L2(weight)',
        regularisation_lambda=1.,class_weights='N/(2*N_class)',score_threshold=.5,
        optimiser='CPU float64, zero initialisation, LBFGS lr1/max_iter300, strong_wolfe; no parameter selection',
        score_semantics='Sigmoid model score, not a calibrated correctness probability',
        comparator='Keep all original detector candidates, equivalent to no added appearance rejection',
        no_row_random_split=True,no_augmentations=True,automatic_adoption=False,
        caveat='One training rally and tiny selected same-match checks. This is an exploratory appearance baseline, not a general accuracy estimate.',
        train_table_sha256=sha(train_path),prediction_table_sha256=sha(eval_path),script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    x=np.stack([features(Path(r['patch96_path'])) for r in train])
    y=np.array([r['label']=='confirmed_ball_enclosed' for r in train],dtype=np.float64)
    assert y.sum()==18 and (1-y).sum()==10
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.
    X=torch.from_numpy((x-mean)/scale);Y=torch.from_numpy(y)
    sample_weights=torch.from_numpy(np.where(y==1,len(y)/(2*y.sum()),len(y)/(2*(1-y).sum())))
    w=torch.zeros(X.shape[1],dtype=torch.float64,requires_grad=True);bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([w,bias],lr=1.,max_iter=300,line_search_fn='strong_wolfe',tolerance_grad=1e-9,tolerance_change=1e-12)
    def closure():
        optimizer.zero_grad()
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(X@w+bias,Y,reduction='none')*sample_weights).mean()+.5*w.square().sum()
        loss.backward();return loss
    optimizer.step(closure)
    weights=w.detach().numpy();intercept=float(bias.detach());model=OUT/'model.npz'
    np.savez_compressed(model,weights=weights,bias=np.array(intercept),mean=mean,scale=scale)
    saved=np.load(model);assert np.array_equal(saved['weights'],weights)
    rows=[]
    for r in evaluation:
        f=features(Path(r['patch96_path']));logit=float(((f-mean)/scale)@weights+intercept)
        score=float(torch.sigmoid(torch.tensor(logit,dtype=torch.float64)));keep=int(score>=.5)
        rows.append(dict(case_id=r['case_id'],display_label=r['display_label'],ball_pilot_id=r['ball_pilot_id'],
            local_frame=r['local_frame'],split_group=r['split_group'],label_at_prediction=r['label'],
            image_only_logit=logit,uncalibrated_ball_score=score,appearance_keep=keep,
            no_added_filter_keep=1,model_sha256=sha(model)))
    write(OUT/'frozen_predictions.csv',rows)
    regression=[r for r in rows if r['label_at_prediction']!='pending_user_confirmation']
    report=dict(status='predictions_saved_before_new_human_answers',model_sha256=sha(model),
        frozen_predictions_sha256=sha(OUT/'frozen_predictions.csv'),training_features=672,training_cases=28,
        new008_cases_pending=4,prior_regression_cases=4,
        prior_real_candidates_retained=sum(r['appearance_keep'] for r in regression if r['label_at_prediction']=='confirmed_ball_enclosed'),
        prior_real_total=sum(r['label_at_prediction']=='confirmed_ball_enclosed' for r in regression),
        prior_false_candidates_retained=sum(r['appearance_keep'] for r in regression if r['label_at_prediction']=='confirmed_not_ball_enclosed'),
        prior_false_total=sum(r['label_at_prediction']=='confirmed_not_ball_enclosed' for r in regression),
        image_hashes_disjoint_between_train_and_check=True,training_and_check_groups_disjoint=True,
        all_registries_and_old_source_files_unchanged=True,automatic_adoption=False,
        evaluation_warning='These4 are previously known safety cases: preparation ball and hands near edits. Counts are not in-play performance.')
    assert all(sha(Path(p))==h for p,h in protected.items())
    for p,h in load(ROOT/'play_start_audit/inventory_summary.json')['source_hashes'].items():assert sha(Path(p))==h
    dump(OUT/'report.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
