"""Post-trial provenance, leakage, scores and protected-state checks."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
from integrate_044_reviewed_tracks import ROOT,read,load,write,sha
from compare_frozen_ball_backbones import predict

OUT=ROOT/'ball_training_only_appearance_comparison'
def main():
    p=load(OUT/'protocol.json');d=load(OUT/'decision.json');t=load(OUT/'training_result.json');train=read(OUT/'training_manifest.csv')
    assert len(train)==111 and Counter(r['label'] for r in train)=={'1':102,'0':9}
    hashes={r['source_image_sha256'] for r in train};cids={r['candidate_id'] for r in train}
    for r in train:
        assert sha(Path(r['source_image']))==r['source_image_sha256']
        assert sha(Path(r['raw_patch96']))==r['patch_sha256']
    for role in ['validation','challenge','six_false']:
        meta=read(OUT/f'{role}_patch_manifest.csv')
        for r in meta:
            assert sha(Path(r['source_image']))==r['source_image_sha256'] and r['source_image_sha256'] not in hashes
            assert sha(Path(r['raw_patch96']))==r['patch_sha256']
        if role!='six_false':assert not ({r['candidate_id'] for r in read(ROOT/f'ball_temporal_path_comparison/{role}_selected.csv')}&cids)
    folds=load(OUT/'folds.json');assert len(folds)==16
    for f in folds:assert sha(OUT/f['head'])==f['head_sha256']
    fs=np.load(OUT/'training_features.npz');fit_rows=[];fit_summary=[]
    for method in p['methods']:
        path=OUT/f'final_{method}.npz';assert sha(path)==t['final_model_hashes'][method]
        model=dict(np.load(path));scores=predict(model,fs[method]);ys=np.array([int(r['label']) for r in train]);pred=scores>=.5
        for r,s in zip(train,scores):fit_rows.append(dict(method=method,sample_id=r['sample_id'],label=r['label'],score=float(s),keep=int(s>=.5)))
        fit_summary.append(dict(method=method,in_sample_real_kept=int(pred[ys==1].sum()),in_sample_false_kept=int(pred[ys==0].sum()),not_a_generalization_estimate=True))
    write(OUT/'in_sample_predictions.csv',fit_rows)
    assert d['audit'] is None and d['audit_eligible_methods']==[] and not any(OUT.glob('audit_*'))
    assert not d['automatic_adoption'] and not d['operational_outputs_changed']
    assert all(sha(Path(path))==h for path,h in p['protected_input_hashes'].items())
    grouped=read(OUT/'grouped_training_predictions.csv');clip_summary=[]
    for method in p['methods']:
        for group in sorted({r['split_group'] for r in train}):
            rows=[r for r in grouped if r['method']==method and r['heldout']==group]
            clip_summary.append(dict(method=method,heldout=group,real_total=sum(r['label']=='1' for r in rows),real_kept=sum(r['label']=='1' and r['keep']=='1' for r in rows),
                false_total=sum(r['label']=='0' for r in rows),false_kept=sum(r['label']=='0' and r['keep']=='1' for r in rows)))
    write(OUT/'per_clip_training_holdout.csv',clip_summary)
    summary=[]
    for method in p['methods']:
        v=d['validation'][method];c=d['challenge'][method];s=d['six_false'][method]
        summary.append(dict(method=method,validation_correct_before=81,validation_correct_after=v['proposed']['counts']['localized_tp_16px'],
            validation_wrong_before=2,validation_wrong_after=v['proposed']['counts']['wrong_localization']+v['proposed']['counts']['presence_fp'],
            correct_lost=len(v['correct_lost']),challenge_correct=c['proposed']['counts']['localized_tp_16px'],six_false_withheld=s['withheld'],six_false_retained=s['retained']))
    write(OUT/'comparison_summary.csv',summary)
    report=dict(status='passed',human_object_labels=9,confirmed_not_ball=9,uncertain=0,manual_center_positives=102,
        training_clips=8,negative_clips=5,grouped_heads_verified=16,training_and_evaluation_images_disjoint=True,
        training_and_evaluation_clips_disjoint=True,all_source_images_and_patches_verified=True,
        protected_inputs_unchanged=True,final_heads_unchanged=True,audit_not_run=True,
        in_sample_only=fit_summary,summary=summary)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
