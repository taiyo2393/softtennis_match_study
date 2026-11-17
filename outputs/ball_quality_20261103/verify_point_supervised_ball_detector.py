"""Matched-arm/state/data provenance checks and point-supervision diagnostics."""
import json,math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT,read,load,write,sha
from train_point_supervised_ball_detector_fp32 import OUT,support_mask

def main():
    assert not (OUT/'verification.json').exists()
    protocol=load(OUT/'protocol.json');trained=load(OUT/'trained_models.json');decision=load(OUT/'decision.json');ep=load(OUT/'evaluation_protocol.json')
    assert sha(ROOT/'train_point_supervised_ball_detector_fp32.py')==protocol['script_sha256']
    assert sha(ROOT/'evaluate_point_supervised_ball_detector.py')==ep['script_sha256']
    assert protocol['no_dense_background_gt'] and protocol['not_visible_frames_used_for_training']==0
    assert protocol['epochs']==3 and protocol['steps_per_epoch']==51 and not protocol['hyperparameter_search']
    rows=read(OUT/'positive_training_manifest.csv');negs=read(OUT/'negative_point_manifest.csv');assert len(rows)==102 and len(negs)==9
    assert all(r['truth_visibility']=='visible' for r in rows)
    train_clips={r['candidate_id'] for r in rows};train_hashes={h for r in rows for h in json.loads(r['input_frame_sha256_json'])}
    neg_by={r['positive_sample_id']:r for r in negs};mask_records=[]
    for r in rows:
        for p,h in zip(json.loads(r['input_frames_json']),json.loads(r['input_frame_sha256_json']),strict=True):assert sha(Path(p))==h
        points=[(r['truth_x_px'],r['truth_y_px'])]
        if r['sample_id'] in neg_by:
            n=neg_by[r['sample_id']];assert n['object_label']=='not_ball' and n['reviewer_id']=='human_user'
            assert n['candidate_id']==r['candidate_id'] and n['source_frame']==r['source_frame'];points.append((n['candidate_x_px'],n['candidate_y_px']))
        mask=support_mask(points)
        assert (~mask).sum()<=4*len(points) and mask.sum()>288*512-9
        mask_records.append(dict(sample_id=r['sample_id'],point_count=len(points),excluded_bilinear_support_pixels=int((~mask).sum()),all_other_pixels_have_only_soft_teacher_prior=True))
    history=load(OUT/'training_history.json');assert len(history)==6
    for arm in ['positive_control','point_negative']:
        hs=[r for r in history if r['family']==arm];assert [r['updates'] for r in hs]==[51,102,153] and [r['epoch'] for r in hs]==[1,2,3]
        assert all(math.isfinite(r[k]) for r in hs for k in ['loss','positive_loss','preservation_loss','balanced_negative_loss'])
        entry=trained['checkpoints'][arm];assert sha(Path(entry['path']))==entry['sha256']
        delta=torch.load(entry['path'],map_location='cpu',weights_only=False)
        assert delta['meta']['updates']==153 and delta['meta']['epoch']==3 and delta['meta']['source_weight_sha256']==trained['source_weight_sha256']
        assert len(delta['trainable_state_dict'])==8 and all((name.startswith('up_block_3.') or name.startswith('predictor.')) and 'running_' not in name and 'num_batches' not in name for name in delta['trainable_state_dict'])
        assert all(torch.isfinite(v).all() for v in delta['trainable_state_dict'].values())
        for role in ['validation','challenge','short_sequence']:
            hashes=load(OUT/f'{arm}_{role}_context_image_hashes.json')
            assert not train_hashes&set(hashes.values())
            assert all(sha(Path(p))==h for p,h in hashes.items())
            pred=read(OUT/f'{arm}_{role}_temporal.csv');assert not train_clips&{r['candidate_id'] for r in pred}
        hashes=load(OUT/f'{arm}_train_context_image_hashes.json');assert all(sha(Path(p))==h for p,h in hashes.items())
        cache=load(OUT/f'{arm}_train_component_cache.json')
        for method in ['raw','temporal']:
            for r in read(OUT/f'{arm}_train_{method}.csv'):
                if int(r['visible']):assert any(float(r['x_px'])==c['x_px'] and float(r['y_px'])==c['y_px'] for c in cache[r['candidate_id']][r['local_frame']])
    for path,h in decision['protected_input_hashes'].items():assert sha(Path(path))==h
    original=read(OUT/'original_training_point_scores.csv');final=read(OUT/'final_training_point_scores.csv');point_summary=[]
    for arm,source in [('original',original),('positive_control',[r for r in final if r['arm']=='positive_control']),('point_negative',[r for r in final if r['arm']=='point_negative'])]:
        key='original_probability' if arm=='original' else 'probability'
        for kind,count in [('manual_positive',102),('confirmed_non_ball',9)]:
            values=[float(r[key]) for r in source if r['kind']==kind];assert len(values)==count
            point_summary.append(dict(arm=arm,kind=kind,points=count,mean_sampled_probability=float(np.mean(values)),points_above_detector_threshold03=sum(v>.3 for v in values),
                interpretation='Bilinear sampled training point response, not number of retained/rejected objects or independent performance.'))
    write(OUT/'training_point_score_summary.csv',point_summary);write(OUT/'point_support_mask_audit.csv',mask_records)
    old=ROOT/'ball_point_supervised_detector';a=np.load(old/'original_teacher_maps.npz');b=np.load(OUT/'original_teacher_maps.npz')
    assert np.array_equal(a['sample_ids'],b['sample_ids']) and np.array_equal(a['heatmaps'],b['heatmaps'])
    if decision['audit'] is None:assert not list(OUT.glob('point_negative_audit_*'))
    report=dict(status='passed',training_frames=102,positive_points=102,negative_points=9,not_visible_training_frames=0,
        matched_updates_per_arm=153,final_epoch_only=3,trainable_parameter_tensors_per_arm=8,
        source_weights_and_registries_unchanged=True,train_evaluation_clips_and_inputs_disjoint=True,
        teacher_arrays_unchanged_after_numerical_repair=True,sparse_point_support_masks_verified=True,
        all_training_outputs_have_actual_same_frame_component=True,training_point_responses=point_summary,
        no_new_human_labels=True,no_dense_background_ground_truth=True,soft_teacher_is_not_ground_truth=True,
        automatic_adoption=False,operational_outputs_changed=False,independent_generalization_established=False)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
