"""Fixed image-only candidate rejection using training8clips, 102+9 human labels.

Never fit on previous audit36 or validation6 false candidates. No threshold grid.
"""
import json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2
from fit_frozen_ball_appearance_probe import features as handcrafted
from compare_frozen_ball_backbones import fit,predict,WEIGHTS
from prepare_training_appearance_review import crop96
from run_manual_ball_finetune import metrics
from run_hard_case_ball_finetune import good_ids,err

OUT=ROOT/'ball_training_only_appearance_comparison'
REVIEW=ROOT/'ball_appearance_training_review'
BASE=ROOT/'ball_temporal_path_comparison'
METHODS=['handcrafted672','frozen_resnet18_512']
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

def verified_training():
    manifest=load(REVIEW/'manifest.json');cases=load(REVIEW/'cases.json');labels=read(REVIEW/'candidate_review_labels.csv')
    assert sha(REVIEW/'cases.json')==manifest['cases_sha256']
    assert sha(REVIEW/'existing_positive_centers.csv')==manifest['positive_table_sha256']
    assert len(labels)==9 and {r['case_id'] for r in labels}=={c['case_id'] for c in cases}
    events=[json.loads(s) for s in (REVIEW/'annotation_events.jsonl').read_text(encoding='utf-8').splitlines() if s]
    replay={};lookup={r['case_id']:r for r in labels}
    for event in events:
        cid=event['case_id'];before,after=event['before'],event['after']
        assert cid in lookup and before['case_id']==after['case_id']==cid
        assert all(before[k]==after[k] for k in before if k not in ['object_label','reviewer_id','reviewed_at_utc','notes'])
        if cid in replay:assert replay[cid]==before
        else:assert before['object_label']=='unreviewed' and before['reviewer_id']==''
        assert after['object_label'] in ['ball','not_ball','uncertain'] and after['reviewer_id']=='human_user' and after['reviewed_at_utc']
        replay[cid]=after
    assert replay==lookup and not any(r['object_label']=='unreviewed' for r in labels)
    rows=[]
    for r in read(REVIEW/'existing_positive_centers.csv'):
        rows.append(dict(r,label=1,label_source='existing_user_clicked_center'))
    for c in cases:
        r=lookup[c['case_id']]
        if r['object_label']=='uncertain':continue
        rows.append(dict(c,label=int(r['object_label']=='ball'),label_source='new_human_object_judgment'))
    assert len(rows)==111 and sum(r['label'] for r in rows)==102
    for r in rows:
        assert sha(Path(r['raw_patch96']))==r['patch_sha256']
        assert sha(Path(r['source_image']))==r['source_image_sha256']
    dump(REVIEW/'completed_review.json',dict(status='all_nine_human_labels_verified',counts=dict(Counter(r['object_label'] for r in labels)),
        labels_sha256=sha(REVIEW/'candidate_review_labels.csv'),events_sha256=sha(REVIEW/'annotation_events.jsonl'),
        events=len(events),training_ready_cases=len(rows),manual_positive_centers=102,
        automatic_labels_added=0,physical_absence_labels_added=0,registry_changed=False))
    return rows

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Preserve fixed comparison. No result-driven reruns.'
    torch.set_num_threads(4);torch.manual_seed(20261103)
    train=verified_training();groups=np.array([r['split_group'] for r in train]);y=np.array([r['label'] for r in train],dtype=np.float64)
    protected={str(p):sha(p) for p in [REVIEW/'candidate_review_labels.csv',REVIEW/'annotation_events.jsonl',REVIEW/'cases.json',REVIEW/'existing_positive_centers.csv',
        ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',
        SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv',WEIGHTS,
        ROOT/'fit_frozen_ball_appearance_probe.py',ROOT/'compare_frozen_ball_backbones.py']}
    for p,h in load(REVIEW/'manifest.json')['protected_input_hashes'].items():assert sha(Path(p))==h;protected[p]=h
    for role in ['validation','challenge','audit']:p=BASE/f'{role}_selected.csv';protected[str(p)]=sha(p)
    model=Path(load(BASE/'protocol.json')['original_model']);protected[str(model)]=sha(model)
    false_path=ROOT/'ball_040_object_swap_review/round2/user_reviewed/candidate_review.csv';protected[str(false_path)]=sha(false_path)
    protocol=dict(status='fixed_before_features_fitting_or_new_predictions',training_cases=111,manual_center_positives=102,confirmed_non_ball_negatives=9,
        training_clip_groups=sorted(set(groups)),negative_clip_groups=sorted(set(groups[y==0])),
        label_scope='Candidate non-ball is object-specific. No physical-absence labels; uncertain excluded. Other actual balls positive.',
        methods=METHODS,hand_features='Existing fixed gradient/color672 on central native16/32 plus mean/std4/8/16/32',
        resnet='Locally cached ImageNet ResNet18 V1, frozen eval incl BN, global-average512; central32 RGB, weights.transforms() official resize256/crop224/normalisation',
        head='Training-only mean/std; class-balanced logistic mean BCE +0.5*L2(weight); float64 CPU, zero init, LBFGS300 strong_wolfe; intercept unpenalised',
        threshold=.5,hyperparameter_search=False,augmentation=False,backbone_finetune=False,downloads=False,
        scores='Uncalibrated sigmoid; not a correctness probability',
        internal_check='Leave one entire training clip out, fit8heads per feature method. Diagnostic only, no model/threshold selection.',
        fit='Final two fixed heads fit ALL111training examples only; same clip/image excluded from every prediction role',
        application='Gate already chosen temporal candidate only; score<0.5 ->position unknown, not physical absence; no coordinate changes or replacement',
        evaluation='Both fixed methods on validation120 and challenge4; six separately confirmed false candidates scored BEFORE manual removal. No tuning or choosing alternate feature scales.',
        audit='Only a method preserving ALL81 validation correct and ALL4 challenge correct, adding no errors, and reducing validation errors or rejecting >=1 of six false candidates; otherwise no audit.',
        adoption='No automatic registry/output changes even if passing; this known same-match developmental comparison cannot establish external generalisation.',
        centering_limitation='Positive training patches use manual centers; inference uses detector centers. Off-center localization and unbalanced negative contexts can impair generalization.',
        previous_audit36_used_for_training=False,protected_input_hashes=protected,weights_sha256=sha(WEIGHTS),script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    write(OUT/'training_manifest.csv',[{k:r[k] for k in ['sample_id','candidate_id','local_frame','source_frame','split_group','source_image','source_image_sha256','raw_patch96','patch_sha256','label','label_source']} for r in train])
    backbone=resnet18(weights=None);backbone.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)
    backbone.fc=torch.nn.Identity();backbone.eval().requires_grad_(False).to('cuda')
    transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
    def extract(rows,name):
        tensors=[];paths=[];meta=[];folder=OUT/f'{name}_patches';folder.mkdir()
        for i,r in enumerate(rows):
            if 'raw_patch96' in r:patch=cv2.imread(r['raw_patch96']);path=Path(r['raw_patch96'])
            else:
                p=SOURCE/'data/balltrack/racketvision/match1'/r['candidate_id']/'frames'/f'{int(r["local_frame"]):06d}.jpg'
                im=cv2.imread(str(p));assert im.shape[:2]==(1080,1920)
                patch=crop96(im,float(r['x_px']),float(r['y_px']));path=folder/f'{i:04d}.png';assert cv2.imwrite(str(path),patch)
                meta.append(dict(sample_id=r['sample_id'],source_image=str(p),source_image_sha256=sha(p),crop_x=r['x_px'],crop_y=r['y_px'],raw_patch96=str(path),patch_sha256=sha(path)))
            assert patch.shape==(96,96,3);paths.append(path)
            rgb=patch[32:64,32:64,::-1].copy();tensors.append(transform(torch.from_numpy(rgb).permute(2,0,1)))
        deep=[]
        with torch.inference_mode():
            for start in range(0,len(rows),16):deep.append(backbone(torch.stack(tensors[start:start+16]).to('cuda')).cpu().numpy())
        features=dict(handcrafted672=np.stack([handcrafted(p) for p in paths]),frozen_resnet18_512=np.concatenate(deep).astype(np.float64))
        assert all(np.isfinite(x).all() for x in features.values())
        np.savez_compressed(OUT/f'{name}_features.npz',sample_ids=np.array([r['sample_id'] for r in rows]),**features)
        if meta:write(OUT/f'{name}_patch_manifest.csv',meta)
        return features
    features=extract(train,'training');fold_rows=[];fold_meta=[];fitted={}
    for method,x in features.items():
        for group in sorted(set(groups)):
            fitting=groups!=group;check=groups==group
            assert not ({train[i]['source_image_sha256'] for i in np.where(fitting)[0]}&{train[i]['source_image_sha256'] for i in np.where(check)[0]})
            head=fit(x[fitting],y[fitting]);scores=predict(head,x[check]);filename=f'fold_{group[-3:]}_{method}.npz';np.savez_compressed(OUT/filename,**head)
            fold_meta.append(dict(method=method,heldout=group,train_count=int(fitting.sum()),check_count=int(check.sum()),head=filename,head_sha256=sha(OUT/filename)))
            for i,s in zip(np.where(check)[0],scores):fold_rows.append(dict(method=method,heldout=group,sample_id=train[i]['sample_id'],label=int(y[i]),score=float(s),keep=int(s>=.5)))
        fitted[method]=fit(x,y);path=OUT/f'final_{method}.npz';np.savez_compressed(path,**fitted[method]);saved=dict(np.load(path))
        assert np.allclose(predict(saved,x),predict(fitted[method],x),rtol=0,atol=1e-12)
    write(OUT/'grouped_training_predictions.csv',fold_rows);dump(OUT/'folds.json',fold_meta)
    internal=[]
    for method in METHODS:
        rs=[r for r in fold_rows if r['method']==method]
        internal.append(dict(method=method,real_total=102,real_kept=sum(r['keep'] for r in rs if r['label']),false_total=9,false_kept=sum(r['keep'] for r in rs if not r['label'])))
    dump(OUT/'training_result.json',dict(internal_clip_holdout=internal,full_models_frozen=True,final_model_hashes={k:sha(OUT/f'final_{k}.npz') for k in METHODS},evaluation_labels_used_for_fitting=False))
    print('TRAIN',json.dumps(internal),flush=True)
    train_ids=set(groups);train_hashes={r['source_image_sha256'] for r in train}
    def evaluate(role,allowed):
        original=read(BASE/f'{role}_selected.csv');assert not ({'legacy_match1:'+r['candidate_id'] for r in original}&train_ids)
        observed=[r for r in original if int(r['visible'])];fs=extract(observed,role)
        patch_manifest=read(OUT/f'{role}_patch_manifest.csv');assert not ({r['source_image_sha256'] for r in patch_manifest}&train_hashes)
        bm=metrics(original);result={}
        for method in allowed:
            scores=predict(fitted[method],fs[method]);lookup={r['sample_id']:float(s) for r,s in zip(observed,scores)};pred=[]
            for r in original:
                q=dict(r,appearance_method=method,appearance_score=lookup.get(r['sample_id'],''),appearance_decision='already_missing' if not int(r['visible']) else 'keep')
                if int(r['visible']) and lookup[r['sample_id']]<.5:q.update(x_px=0,y_px=0,visible=0,confidence=0.,appearance_decision='position_unknown')
                assert not int(q['visible']) or (q['x_px'],q['y_px'])==(r['x_px'],r['y_px'])
                pred.append(q)
            write(OUT/f'{role}_{method}.csv',pred);pm=metrics(pred);lost=sorted(good_ids(original)-good_ids(pred))
            result[method]=dict(baseline=bm,proposed=pm,correct_lost=lost,nonregression=not lost and err(pm)<=err(bm),
                withheld=sum(int(a['visible']) and not int(b['visible']) for a,b in zip(original,pred)))
        return result
    result=dict(training_clip_holdout=internal,validation=evaluate('validation',METHODS),challenge=evaluate('challenge',METHODS))
    assert all(v['baseline']['counts']['localized_tp_16px']==81 for v in result['validation'].values())
    assert all(v['baseline']['counts']['localized_tp_16px']==4 for v in result['challenge'].values())
    # Score actual displayed candidates before the human manual exclusions.
    false=read(false_path);old=read(ROOT/'temporal_rally_reviews_v2/match1_motion_pilot_015/balltrack_predictions_proposed.csv');false_queries=[]
    for r in false:
        t=int(r['local_frame']);b=old[t];assert int(b['visible']) and (float(b['x_px']),float(b['y_px']))==(float(r['displayed_x_px']),float(r['displayed_y_px']))
        assert r['candidate_is_ball']=='0';false_queries.append(dict(b))
    fs=extract(false_queries,'six_false');assert not ({r['source_image_sha256'] for r in read(OUT/'six_false_patch_manifest.csv')}&train_hashes)
    outcomes=[];result['six_false']={}
    for method in METHODS:
        scores=predict(fitted[method],fs[method]);result['six_false'][method]=dict(total=6,withheld=int((scores<.5).sum()),retained=int((scores>=.5).sum()))
        for r,s in zip(false_queries,scores):outcomes.append(dict(method=method,local_frame=r['local_frame'],x_px=r['x_px'],y_px=r['y_px'],score=float(s),withheld=int(s<.5),physical_visibility='not_labelled',true_center='unknown'))
    write(OUT/'six_false_predictions.csv',outcomes)
    eligible=[method for method in METHODS if result['validation'][method]['nonregression'] and result['challenge'][method]['nonregression'] and
        (err(result['validation'][method]['proposed'])<2 or result['six_false'][method]['withheld']>0)]
    result['audit']=evaluate('audit',eligible) if eligible else None
    assert all(sha(Path(p))==h for p,h in protected.items())
    result.update(status='fixed_training_only_appearance_comparison_complete',automatic_adoption=False,operational_outputs_changed=False,
        audit_eligible_methods=eligible,protected_inputs_unchanged=True,physical_absence_labels_added=0,interpolation=0,
        independent_generalization_established=False)
    dump(OUT/'decision.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
