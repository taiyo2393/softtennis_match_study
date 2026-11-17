"""Verify actual observed selections and report sparse evaluation changes."""
import json,math
from pathlib import Path
from collections import Counter
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha
from run_manual_ball_finetune import metrics
from run_hard_case_ball_finetune import good_ids,err

OUT=ROOT/'ball_temporal_path_comparison'
def status(r):
    if not int(r['visible']):return 'missing' if r['truth_visibility']=='visible' else 'correct_absent'
    if r['truth_visibility']!='visible':return 'false_presence'
    return 'correct' if math.hypot(float(r['x_px'])-float(r['truth_x_px']),float(r['y_px'])-float(r['truth_y_px']))<=16 else 'wrong_position'

def main():
    assert not (OUT/'post_trial_verification.json').exists()
    selection=load(OUT/'training_selection.json');decision=load(OUT/'decision.json');config=selection['chosen_configuration']
    assert config and not selection['validation_labels_used'] and not selection['challenge_labels_used']
    summary=[];changes=[]
    for role in ['train','validation','challenge','audit']:
        selected_file=OUT/(f'train_{config["name"]}.csv' if role=='train' else f'{role}_selected.csv')
        if not selected_file.exists():continue
        a=read(OUT/f'{role}_bbox_area.csv');b=read(selected_file);cache=load(OUT/f'{role}_component_cache.json')
        assert len(a)==len(b)
        for x,y in zip(a,b,strict=True):
            assert x['sample_id']==y['sample_id'] and x['visible']==y['visible']
            cs=cache[y['candidate_id']][str(y['local_frame'])]
            if int(y['visible']):
                assert any((c['x_px'],c['y_px'],c['component_index'])==(int(y['x_px']),int(y['y_px']),int(y['selected_component_index'])) for c in cs)
            else:assert not cs
            if (x['x_px'],x['y_px'])!=(y['x_px'],y['y_px']):
                def error(r):return math.hypot(float(r['x_px'])-float(r['truth_x_px']),float(r['y_px'])-float(r['truth_y_px'])) if r['truth_visibility']=='visible' else ''
                changes.append(dict(role=role,sample_id=x['sample_id'],candidate_id=x['candidate_id'],local_frame=x['local_frame'],source_frame=x['source_frame'],
                    before=status(x),after=status(y),truth_x_px=x['truth_x_px'],truth_y_px=x['truth_y_px'],
                    before_x=x['x_px'],before_y=x['y_px'],after_x=y['x_px'],after_y=y['y_px'],
                    before_error_px=error(x),after_error_px=error(y),selection_status=y['selection_status']))
        bm=metrics(a);pm=metrics(b);lost=sorted(good_ids(a)-good_ids(b))
        assert not lost and err(pm)<=err(bm)
        summary.append(dict(role=role,frames=len(a),before_correct=len(good_ids(a)),after_correct=len(good_ids(b)),before_wrong_or_false=err(bm),after_wrong_or_false=err(pm),
            original_correct_lost=len(lost),missing_before=bm['counts']['presence_fn'],missing_after=pm['counts']['presence_fn'],
            false_presence_before=bm['counts']['presence_fp'],false_presence_after=pm['counts']['presence_fp'],
            selection_status_counts=json.dumps(dict(Counter(r['selection_status'] for r in b)))))
    write(OUT/'changed_observations.csv',changes);write(OUT/'comparison_summary.csv',summary)
    protected=load(OUT/'verified_inputs.json')['protected_input_hashes'];assert all(sha(Path(p))==h for p,h in protected.items())
    report=dict(status='fixed_selected_temporal_candidate_rule_passes_sparse_development_checks',chosen_configuration=config,
        role_summaries=summary,all_outputs_verified_original_target_frame_components=True,all_presence_and_missing_flags_preserved=True,
        original_correct_lost=0,validation_parameters_fixed_on_training_only=True,post_evaluation_tuning=False,
        missing_recovery_claim=False,continuous_tracking_accuracy_claim=False,independent_external_accuracy=False,
        operating_on='Raw original model candidates; existing human-reviewed ball overrides were not replaced',
        operational_registries_unchanged=True,protected_inputs_unchanged=True,
        interpretation='Promising local candidate selection prototype. Old validation/audit previously seen; challenge4same-match actively selected near older frames. No pooling of sample counts across roles.',
        next='Review temporal output on clips and retain explicit human corrections when integrating. Do not infer all context frames are correct from sparse targets.')
    (OUT/'post_trial_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
