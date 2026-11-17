"""Verify the one-candidate follow-up and preserve its failed audit decision."""
import json,math
from pathlib import Path
import torch
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha
from run_manual_ball_finetune import metrics
from run_hard_case_ball_finetune import good_ids,err

OUT=ROOT/'ball_balanced_fixed_validation';BASE=ROOT/'ball_temporal_path_comparison';DIAG=ROOT/'ball_balanced_localization_diagnosis';SHORT=ROOT/'ball_short_sequence_evaluation';DATA=ROOT/'balltrack_manual_finetune_pilot'
def label_result(r):
    if not int(r['visible']):return 'missing_visible_ball' if r['truth_visibility']=='visible' else 'no_output_on_nonvisible_label'
    if r['truth_visibility']!='visible':return 'output_on_nonvisible_label'
    return 'correct16px' if math.hypot(float(r['x_px'])-float(r['truth_x_px']),float(r['y_px'])-float(r['truth_y_px']))<=16 else 'wrong_position'
def main():
    assert not (OUT/'verification.json').exists()
    p=load(OUT/'protocol.json');r=load(OUT/'result.json');selection=load(OUT/'frozen_selection.json')
    assert sha(ROOT/'validate_fixed_balanced_detector.py')==p['script_sha256'] and selection==r['selection']==p['selection']
    assert selection['condition']=='balanced_preserve' and selection['threshold']==.3 and selection['training_metrics']['original_correct_lost']==0
    assert sha(Path(selection['source_diagnostic_model']))==selection['snapshot_sha256']==sha(Path(selection['validation_snapshot']))
    delta=torch.load(selection['validation_snapshot'],map_location='cpu',weights_only=False)
    assert delta['meta']['diagnostic_only'] and delta['meta']['condition']=='balanced_preserve' and delta['meta']['steps']==153
    assert len(delta['trainable_state_dict'])==8 and all(k.startswith('up_block_3.') or k.startswith('predictor.') for k in delta['trainable_state_dict'])
    training=read(ROOT/'ball_point_supervised_detector_fp32/positive_training_manifest.csv');cids={x['candidate_id'] for x in training};hashes={h for x in training for h in json.loads(x['input_frame_sha256_json'])}
    source_short={x['sample_id']:x for x in read(SHORT/'manual_sequence_labels.csv')}
    references=dict(validation=read(BASE/'validation_selected.csv'),challenge=read(BASE/'challenge_selected.csv'),audit=read(BASE/'audit_selected.csv'),
        short_sequence=[dict(x,**{k:source_short[x['sample_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px']}) for x in read(SHORT/'frozen_predictions.csv') if x['method']=='displacement_0.1'])
    summary=[];changes=[];all_outputs=0
    for role in ['validation','challenge','short_sequence','audit']:
        old=references[role];new=read(OUT/f'{role}_selected.csv');before={x['sample_id']:x for x in old};ctx=load(OUT/f'{role}_component_cache.json');images=load(OUT/f'{role}_context_image_hashes.json')
        assert not hashes&set(images.values()) and not cids&{x['candidate_id'] for x in new}
        assert all(sha(Path(path))==h for path,h in images.items())
        for x in new:
            a=before[x['sample_id']]
            assert all(x[k]==a[k] for k in ['truth_visibility','truth_x_px','truth_y_px','candidate_id','local_frame','source_frame'])
            if int(x['visible']):
                assert any(c['x_px']==float(x['x_px']) and c['y_px']==float(x['y_px']) and abs(c['bbox_mean']-float(x['confidence']))<1e-12 for c in ctx[x['candidate_id']][x['local_frame']]);all_outputs+=1
            prior,after=label_result(a),label_result(x)
            if prior!=after:
                changes.append(dict(role=role,sample_id=x['sample_id'],candidate_id=x['candidate_id'],local_frame=x['local_frame'],source_frame=x['source_frame'],
                    before=prior,after=after,old_x=a['x_px'],old_y=a['y_px'],new_x=x['x_px'],new_y=x['y_px'],truth_x=x['truth_x_px'],truth_y=x['truth_y_px']))
        bm=metrics(old);nm=metrics(new);assert bm==r[role]['baseline'] and nm==r[role]['proposed']
        lost=sorted(good_ids(old)-good_ids(new));gained=sorted(good_ids(new)-good_ids(old));assert lost==r[role]['original_correct_lost'] and gained==r[role]['original_correct_gained']
        summary.append(dict(role=role,frames=len(new),correct_before=len(good_ids(old)),correct_after=len(good_ids(new)),errors_before=err(bm),errors_after=err(nm),correct_lost=len(lost),correct_gained=len(gained)))
    six=read(OUT/'six_false_outcomes.csv');assert len(six)==6 and all(x['true_ball_center']=='unknown' and x['physical_visibility']=='not_labelled' for x in six)
    assert all(sha(Path(path))==h for path,h in r['protected_input_hashes'].items())
    assert r['validation']['nonregression'] and r['challenge']['nonregression'] and r['short_sequence']['nonregression'] and not r['audit']['nonregression']
    write(OUT/'comparison_summary.csv',summary);write(OUT/'changed_label_outcomes.csv',changes)
    operational=dict(status='not_adopted',reason='Audit lost one original-correct visible ball after earlier gates passed; six known false-candidate neighbourhoods still have output.',
        audit_lost=r['audit']['original_correct_lost'],validation_gained=r['validation']['original_correct_gained'],
        no_alternate_model_or_threshold_after_failure=True,retained_model='Original weights and previous reviewed tracking references',
        no_automatic_adoption=True,operational_outputs_changed=False,independent_external_accuracy_established=False)
    (OUT/'operational_decision.json').write_text(json.dumps(operational,indent=2),encoding='utf-8')
    report=dict(status='passed',one_frozen_training_selected_candidate=True,source_diagnostic_weights_preserved=True,
        train_clips_and_input_images_disjoint_from_followup=True,actual_same_frame_candidate_outputs_verified=all_outputs,
        original_labels_and_registries_unchanged=True,summary=summary,label_outcome_changes=changes,
        six_outputs_near_prior_false_locations_without_new_truth=True,audit_failure_preserved=True,no_post_audit_tuning=True,
        new_human_labels=0,interpolation=0,automatic_adoption=False,operational_outputs_changed=False)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
