"""Provenance/preservation checks and training error decomposition for two trials."""
import json,math
from pathlib import Path
from collections import Counter
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha
from run_manual_ball_finetune import metrics
from run_hard_case_ball_finetune import good_ids,err

OUT=ROOT/'ball_dual_stride_path_comparison';GUARD=ROOT/'ball_dual_stride_anchor_guard';BASE=ROOT/'ball_temporal_path_comparison'
def category(r):
    if not int(r['visible']):return 'missing_visible_ball' if r['truth_visibility']=='visible' else 'no_output_not_visible_label'
    if r['truth_visibility']!='visible':return 'output_on_not_visible_label'
    return 'correct16px' if math.hypot(float(r['x_px'])-float(r['truth_x_px']),float(r['y_px'])-float(r['truth_y_px']))<=16 else 'wrong_position'
def main():
    assert not (OUT/'verification.json').exists()
    decision=load(OUT/'decision.json');guard=load(GUARD/'training_selection.json');protocol=load(OUT/'protocol.json');gp=load(GUARD/'training_protocol.json')
    assert decision['training']['chosen'] is None and not guard['training_pass']
    assert all(decision[k] is None for k in ['validation','challenge','short_sequence','six_false','audit'])
    assert all(not list(OUT.glob(f'{role}_*')) for role in ['validation','challenge','short_sequence','six_false','audit'])
    assert sha(ROOT/'compare_dual_stride_ball_paths.py')==protocol['script_sha256']
    assert sha(ROOT/'train_anchored_dual_stride_fallback.py')==gp['script_sha256']
    for protected in [decision['protected_input_hashes'],gp['protected_input_hashes']]:assert all(sha(Path(p))==h for p,h in protected.items())
    for p,h in load(OUT/'new_context_image_hashes.json').items():assert sha(Path(p))==h
    # Cached target maps retain their prior verified input images too.
    for p,h in load(ROOT/'ball_input_stride_comparison/input_image_hashes.json').items():assert sha(Path(p))==h
    s1={cid:{int(t):cs for t,cs in times.items()} for cid,times in load(BASE/'train_component_cache.json').items()}
    s2={cid:{int(t):cs for t,cs in times.items()} for cid,times in load(OUT/'train_stride2_context_components.json').items()}
    baseline=read(OUT/'train_baseline.csv');lookup={r['sample_id']:r for r in baseline};old=read(BASE/'train_displacement_0.1.csv')
    assert all((r['sample_id'],r['x_px'],r['y_px'],r['visible'])==(s['sample_id'],s['x_px'],s['y_px'],s['visible']) for r,s in zip(baseline,old))
    labels={r['sample_id']:r for r in read(ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv')}
    changes=[];checks=[]
    files=[(method,OUT/f'train_{method}.csv') for method in protocol['policies']]+[('anchored_missing_only_union_path',GUARD/'train_guarded_predictions.csv')]
    for method,path in files:
        rows=read(path);assert len(rows)==123
        preserved=added=actual=0
        for r in rows:
            b=lookup[r['sample_id']];cid=r['candidate_id'];t=int(r['local_frame'])
            assert (r['truth_visibility'],r['truth_x_px'],r['truth_y_px'])==(b['truth_visibility'],b['truth_x_px'],b['truth_y_px'])
            if int(r['visible']):
                source=s1[cid][t] if r['input_stride']=='1' else s2[cid][t]
                assert any(float(r['x_px'])==c['x_px'] and float(r['y_px'])==c['y_px'] and abs(float(r['confidence'])-c['bbox_mean'])<1e-12 for c in source)
                actual+=1
            same=(r['x_px'],r['y_px'],r['visible'],r['confidence'])==(b['x_px'],b['y_px'],b['visible'],b['confidence'])
            if int(b['visible']) and same:preserved+=1
            if method!='union_path' and int(b['visible']):assert same
            changed=(r['x_px'],r['y_px'],r['visible'])!=(b['x_px'],b['y_px'],b['visible'])
            if not int(b['visible']) and int(r['visible']):added+=1
            if changed:
                changes.append(dict(method=method,sample_id=r['sample_id'],candidate_id=cid,local_frame=t,source_frame=r['source_frame'],
                    previous_outcome=category(b),new_outcome=category(r),x_px=r['x_px'],y_px=r['y_px'],truth_visibility=r['truth_visibility'],
                    truth_x_px=r['truth_x_px'],truth_y_px=r['truth_y_px'],existing_label_notes=labels[r['sample_id']].get('notes',''),
                    existing_phase_status=labels[r['sample_id']].get('phase_status',''),inferred_phase='not_inferred',interpolated=0))
        pm=metrics(rows);checks.append(dict(method=method,correct=len(good_ids(rows)),errors=err(pm),prior_correct_lost=len(good_ids(baseline)-good_ids(rows)),
            baseline_present=70,existing_output_exactly_preserved=preserved,added_outputs=added,outputs_with_same_frame_source_provenance=actual))
    write(OUT/'training_change_diagnostics.csv',changes)
    write(OUT/'training_policy_verification.csv',checks)
    unguarded={r['sample_id']:r for r in read(OUT/'train_missing_only_union_path.csv')}
    guarded=read(GUARD/'train_guarded_predictions.csv');blocked=[]
    for r in guarded:
        a=unguarded[r['sample_id']]
        if int(a['visible']) and not int(r['visible']):blocked.append(dict(sample_id=r['sample_id'],discarded_candidate_outcome=category(a),anchor_gate_status=r['anchor_gate_status'],
            discarded_x_px=a['x_px'],discarded_y_px=a['y_px'],anchor_residual_px=r['anchor_residual_px']))
    if blocked:write(GUARD/'discarded_candidate_diagnostics.csv',blocked)
    result=dict(status='passed',baseline_reproduction_rows=123,training_baseline_correct=66,training_baseline_errors=4,
        policy_checks=checks,guard_discarded_candidates=blocked,guard_rejected_correct_candidates=sum(r['discarded_candidate_outcome']=='correct16px' for r in blocked),
        no_new_evaluation_predictions=True,no_new_human_labels=True,no_interpolation=True,all_outputs_have_actual_same_frame_source=True,
        protected_inputs_unchanged=True,operational_outputs_changed=False,automatic_adoption=False)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');(GUARD/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    print('New errors for missing-only pooled path:')
    for r in changes:
        if r['method']=='missing_only_union_path' and r['previous_outcome']=='missing_visible_ball' and r['new_outcome']=='wrong_position' or r['method']=='missing_only_union_path' and r['new_outcome']=='output_on_not_visible_label':print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
