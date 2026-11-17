"""Verify training-only localization taxonomy, precision replay and provenance."""
import json,math
from pathlib import Path
from collections import Counter
import numpy as np
from integrate_044_reviewed_tracks import ROOT,read,load,sha,write
from diagnose_balanced_ball_localization import OUT,PARENT,CONDITIONS,THRESHOLDS
from calibrate_point_supervised_detector import components_at,expand
from run_hard_case_ball_finetune import good_ids,err
from run_manual_ball_finetune import metrics

def main():
    assert not (OUT/'verification.json').exists()
    protocol=load(OUT/'protocol.json');result=load(OUT/'result.json');replay=read(OUT/'fp32_probe_reproduction.csv')
    assert sha(ROOT/'diagnose_balanced_ball_localization.py')==protocol['script_sha256']
    assert len(replay)==246 and all(r['same']=='1' for r in replay)
    assert all(sha(Path(path))==h for path,h in result['protected_input_hashes'].items())
    summary=read(OUT/'training_localization_summary.csv');mechanisms=read(OUT/'training_failure_mechanisms.csv')
    assert len(summary)==40 and len(mechanisms)==4920
    baseline=read(ROOT/'ball_temporal_path_comparison/train_displacement_0.1.csv');good=good_ids(baseline)
    verified_nodes=verified_outputs=0;examples=[];pareto=[]
    for condition in CONDITIONS:
        entry=next(r for r in result['cache'] if r['condition']==condition);assert sha(Path(entry['path']))==entry['sha256']
        with np.load(entry['path']) as c:
            clips=c['clips'];frames=c['frames'];offsets=c['offsets'];indices=c['indices'];values=c['values']
        assert len(frames)==1107 and len(offsets)==1108 and len(indices)==len(values)==int(offsets[-1]) and np.isfinite(values).all()
        floor=load(OUT/f'{condition}_train_component_cache.json');target_maps={}
        targets={(r['candidate_id'],int(r['local_frame'])) for r in baseline}
        for i,(cid,t) in enumerate(zip(clips,frames)):
            start,stop=offsets[i:i+2];hm=expand(indices[start:stop],values[start:stop]);assert components_at(hm,.3)==floor[str(cid)][str(int(t))];verified_nodes+=1
            if (str(cid),int(t)) in targets:target_maps[str(cid),int(t)]=hm
        assert len(target_maps)==123
        hashes=load(OUT/f'{condition}_train_context_image_hashes.json');assert all(sha(Path(p))==h for p,h in hashes.items())
        for threshold in THRESHOLDS:
            target_candidates={key:components_at(hm,threshold) for key,hm in target_maps.items()};upper=[];nonlocal_response=[];missing_response=[]
            for method in ['raw','temporal']:
                rows=read(OUT/f'{condition}_threshold{round(threshold*100):02d}_{method}.csv');assert len(rows)==123
                for r in rows:
                    if int(r['visible']):
                        candidates=target_candidates[r['candidate_id'],int(r['local_frame'])]
                        assert any(c['x_px']==float(r['x_px']) and c['y_px']==float(r['y_px']) and abs(c['bbox_mean']-float(r['confidence']))<1e-12 for c in candidates);verified_outputs+=1
                s=next(r for r in summary if r['condition']==condition and float(r['threshold'])==threshold and r['method']==method)
                ms=[r for r in mechanisms if r['condition']==condition and float(r['threshold'])==threshold and r['method']==method]
                assert len(ms)==123;visible=[r for r in ms if r['truth_visibility']=='visible'];other=[r for r in ms if r['truth_visibility']!='visible'];assert len(visible)==102 and len(other)==21
                count=Counter(r['category'] for r in visible);assert count['correct']==int(s['correct'])
                for key in ['selection_failure','response_without_localized_center','no_near_response']:assert count[key]==int(s[key])
                assert sum(count.values())==102 and int(s['candidate_center_upper_bound'])==int(s['correct'])+int(s['selection_failure'])
                assert all(r['activation_near_truth']=='' and r['nearest_center_error_px']=='' for r in other)
                lost=good-good_ids(rows);m=metrics(rows);assert len(lost)==int(s['original_correct_lost']) and err(m)==int(s['errors'])
                upper.append(int(s['candidate_center_upper_bound']));nonlocal_response.append(int(s['response_without_localized_center']));missing_response.append(int(s['no_near_response']))
                if method=='temporal':pareto.append(dict(condition=condition,threshold=threshold,correct=int(s['correct']),errors=int(s['errors']),prior_correct_lost=len(lost),candidate_upper=int(s['candidate_center_upper_bound'])))
            assert upper[0]==upper[1] and nonlocal_response[0]==nonlocal_response[1] and missing_response[0]==missing_response[1]
    for r in pareto:
        r['pareto_correct_vs_errors']=int(not any(s['correct']>=r['correct'] and s['errors']<=r['errors'] and (s['correct']>r['correct'] or s['errors']<r['errors']) for s in pareto))
    write(OUT/'training_temporal_tradeoffs.csv',pareto)
    assert result['no_validation_or_audit'] and result['no_best_model_selection_for_deployment'] and not result['automatic_adoption']
    report=dict(status='passed',fp32_prior_raw_predictions_reproduced=246,compact_amp_context_nodes_reproduced=verified_nodes,
        settings=20,decoder_rows=40,per_frame_mechanism_rows=4920,outputs_with_actual_same_frame_candidate=verified_outputs,
        candidate_upper_bounds_consistent_across_decoders=True,nonvisible_labels_have_no_invented_center_or_activation=True,
        train_safe_temporal_settings=result['train_safe_temporal_settings'],strict_training_proposals=result['strict_train_improving_temporal_settings'],
        protected_inputs_and_weights_unchanged=True,no_new_human_labels=True,no_validation_or_audit=True,
        no_best_model_selection_for_deployment=True,automatic_adoption=False,operational_outputs_changed=False,
        limitation='Same training clips and repeated development. Safe training settings, if any, are proposals only; activation, candidate position and identity are separate claims.')
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
