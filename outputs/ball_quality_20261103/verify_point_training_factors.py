"""Check matched exposure doses, exact old-model replication and training scope."""
import json,math
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT,load,read,sha
from diagnose_point_training_factors import OUT,PAIRED,CONDITIONS

def main():
    assert not (OUT/'verification.json').exists()
    protocol=load(OUT/'protocol.json');result=load(OUT/'result.json')
    assert sha(ROOT/'diagnose_point_training_factors.py')==protocol['script_sha256']
    positives=read(PAIRED/'positive_training_manifest.csv');negative=read(PAIRED/'negative_point_manifest.csv');positive_ids={r['sample_id'] for r in positives};negative_ids={r['positive_sample_id'] for r in negative}
    assert len(positive_ids)==102 and len(negative_ids)==9
    for r in positives:
        assert r['truth_visibility']=='visible'
        for path,h in zip(json.loads(r['input_frames_json']),json.loads(r['input_frame_sha256_json']),strict=True):assert sha(Path(path))==h
    batches=read(OUT/'scheduled_batches.csv');dose_check=[]
    for exposure in ['rare','balanced']:
        rows=[r for r in batches if r['schedule']==exposure];assert len(rows)==306
        steps=Counter(int(r['step']) for r in rows);assert set(steps)==set(range(1,154)) and set(steps.values())=={2}
        pdose=defaultdict(float);ndose=defaultdict(float);neg_steps=set();occurrence=Counter()
        for r in rows:
            sid=r['sample_id'];assert sid in positive_ids;pw=float(r['positive_weight']);nw=float(r['negative_weight']);pdose[sid]+=pw;occurrence[sid]+=1
            if int(r['negative_label_exposed']):assert sid in negative_ids;ndose[sid]+=nw;neg_steps.add(int(r['step']))
            else:assert nw==0
        assert set(pdose)==positive_ids and all(abs(v-3)<1e-10 for v in pdose.values())
        assert set(ndose)==negative_ids and all(abs(v-34)<1e-10 for v in ndose.values())
        assert abs(sum(pdose.values())-306)<1e-9 and abs(sum(ndose.values())-306)<1e-9
        dose_check.append(dict(schedule=exposure,updates=153,slots=306,unique_images=102,positive_total_weight=sum(pdose.values()),negative_total_weight=sum(ndose.values()),negative_exposed_batches=len(neg_steps),
            min_image_exposure=min(occurrence.values()),max_image_exposure=max(occurrence.values())))
        if exposure=='balanced':assert len(neg_steps)==153 and all(sum(int(r['negative_label_exposed']) for r in rows if int(r['step'])==s)==1 for s in steps)
    history=read(OUT/'training_history.csv');assert len(history)==612;gradient_summary=[]
    for condition,_,_ in CONDITIONS:
        rows=[r for r in history if r['condition']==condition];assert [int(r['step']) for r in rows]==list(range(1,154))
        assert all(math.isfinite(float(r[k])) for r in rows for k in ['positive_loss','weighted_preservation_loss','weighted_negative_loss','total_loss','gradient_norm_before_clip'])
        for group in ['with_negative','without_negative']:
            sub=[r for r in rows if (int(r['negative_slots'])>0)==(group=='with_negative')]
            if sub:gradient_summary.append(dict(condition=condition,batches=group,steps=len(sub),mean_norm_before_clip=float(np.mean([float(r['gradient_norm_before_clip']) for r in sub])),
                clipped_steps=sum(int(r['gradient_clipped']) for r in sub),caution='Overall gradient norm does not identify separate positive/negative gradient contributions.'))
        record=result['models'][condition];assert sha(Path(record['path']))==record['sha256']
        delta=torch.load(record['path'],map_location='cpu',weights_only=False);assert delta['meta']['steps']==153 and delta['meta']['diagnostic_only']
        assert len(delta['trainable_state_dict'])==8 and all((name.startswith('up_block_3.') or name.startswith('predictor.')) and 'running_' not in name for name in delta['trainable_state_dict'])
        assert all(torch.isfinite(value).all() for value in delta['trainable_state_dict'].values())
    old=torch.load(PAIRED/'point_negative_epoch03_delta.pth',map_location='cpu',weights_only=False)['trainable_state_dict']
    replicate=torch.load(result['models']['rare_preserve']['path'],map_location='cpu',weights_only=False)['trainable_state_dict']
    assert old.keys()==replicate.keys() and all(torch.equal(old[k],replicate[k]) for k in old)
    scores=read(OUT/'point_responses.csv');assert len(scores)==555
    for c in ['original_fp32']+[r[0] for r in CONDITIONS]:
        rs=[r for r in scores if r['condition']==c];assert sum(r['kind']=='positive' for r in rs)==102 and sum(r['kind']=='negative' for r in rs)==9
        assert all(math.isfinite(float(r['score'])) and 0<=float(r['score'])<=1 for r in rs)
        raw=read(OUT/f'{c}_training123_raw.csv');assert len(raw)==123 and len({r['candidate_id'] for r in raw})==8
    assert all(sha(Path(path))==h for path,h in protocol['protected_input_hashes'].items())
    assert result['no_validation_or_audit'] and result['no_best_model_selection'] and not result['automatic_adoption']
    report=dict(status='passed',matched_dose=dose_check,total_updates=612,original_condition_reproduced_all8tensors_bitwise=True,
        frozen_state_unchanged_all_conditions=True,source_inputs_and_teacher_unchanged=True,protected_inputs_unchanged=True,
        training_only_point_scores=555,no_new_validation_or_audit=True,no_best_model_selection=True,
        gradient_diagnostics=gradient_summary,
        frequency_factor_scope='Frequency condition includes paired-frame repetition and inverse-exposure step weights, with exact overall per-point/image weight dose held equal. It does not identify one optimizer mechanism in isolation.',
        same_seed_training_diagnosis_only=True,operational_outputs_changed=False,automatic_adoption=False)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
