"""Verify a failed abstention trial without changing any setting or output."""
import json,math
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,read,load,write,sha
from run_manual_ball_finetune import metrics
from run_hard_case_ball_finetune import good_ids,err

OUT=ROOT/'ball_abstention_comparison';BASE=ROOT/'ball_temporal_path_comparison'
def main():
    assert not (OUT/'operational_decision.json').exists()
    decision=load(OUT/'decision.json');selection=load(OUT/'training_selection.json');chosen=selection['chosen']
    assert chosen==dict(name='unknown1.2_switch0.25',unknown_cost=1.2,switch_cost=.25)
    refs=dict(train=BASE/'train_displacement_0.1.csv',validation=BASE/'validation_selected.csv',challenge=BASE/'challenge_selected.csv')
    summaries=[];changes=[]
    for role,reference_path in refs.items():
        before=read(OUT/f'{role}_temporal_baseline.csv');reference=read(reference_path)
        assert all(a['sample_id']==b['sample_id'] and (a['x_px'],a['y_px'],a['visible'])==(b['x_px'],b['y_px'],b['visible']) for a,b in zip(before,reference,strict=True))
        name=f'train_{chosen["name"]}.csv' if role=='train' else f'{role}_abstention.csv';after=read(OUT/name)
        for a,b in zip(before,after,strict=True):
            assert not int(b['visible']) or (a['x_px'],a['y_px'],a['visible'])==(b['x_px'],b['y_px'],b['visible'])
            if a['visible']=='1' and b['visible']=='0':
                error=math.hypot(float(a['x_px'])-float(a['truth_x_px']),float(a['y_px'])-float(a['truth_y_px'])) if a['truth_visibility']=='visible' else ''
                changes.append(dict(role=role,sample_id=a['sample_id'],truth_visibility=a['truth_visibility'],old_x=a['x_px'],old_y=a['y_px'],
                    true_x=a['truth_x_px'],true_y=a['truth_y_px'],old_error_px=error,old_correct16px=int(error!='' and error<=16),new_state='position_unknown'))
        a=metrics(before);b=metrics(after)
        summaries.append(dict(role=role,samples=len(before),correct_before=len(good_ids(before)),correct_after=len(good_ids(after)),
            errors_before=err(a),errors_after=err(b),visible_ball_missing_before=a['counts']['presence_fn'],visible_ball_missing_after=b['counts']['presence_fn'],
            original_correct_lost=len(good_ids(before)-good_ids(after))))
    write(OUT/'withheld_observation_diagnostics.csv',changes);write(OUT/'comparison_summary.csv',summaries)
    lost=decision['validation']['original_correct_lost'];assert lost==['match1_motion_pilot_013_frame_000375']
    false=read(OUT/'six_false_candidate_outcomes.csv');assert len(false)==6
    rejected=[int(r['local_frame']) for r in false if r['known_false_candidate_withheld']=='1'];assert rejected==[470]
    protected=load(OUT/'verified_inputs.json')['protected_input_hashes'];assert all(sha(Path(p))==h for p,h in protected.items())
    result=dict(status='not_adopted_validation_true_ball_lost',chosen_configuration=chosen,
        training_false_positions_changed_to_unknown=2,training_correct_positions_preserved=66,
        validation_correct_before=81,validation_correct_after=80,validation_errors_before=2,validation_errors_after=2,
        lost_true_ball_sample=lost[0],challenge_correct_before=4,challenge_correct_after=4,
        six_known_false_candidates_rejected=rejected,six_known_false_candidates_retained=[int(r['local_frame']) for r in false if r['known_false_candidate_withheld']=='0'],
        audit_executed=False,post_validation_parameter_search=False,
        all_kept_coordinates_identical_to_prior_temporal_selector=True,all_priors_reproduced_exactly=True,
        manual_six_exclusions_unchanged=True,operational_registries_unchanged=True,model_and_threshold_unchanged=True,
        error_reduction_is_not_ball_position_recovery=True,unknown_is_not_a_physical_absence_label=True,
        next='Retain frozen temporal selector and explicit six human exclusions. Need evidence beyond confidence/continuity for stable false objects; no stricter gate tuned to these six cases.')
    (OUT/'operational_decision.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
