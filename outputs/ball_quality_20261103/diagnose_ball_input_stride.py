"""Read saved heatmaps only; candidate upper bounds and provenance, no fitting."""
import json,math
from pathlib import Path
from collections import Counter
import numpy as np
from integrate_044_reviewed_tracks import ROOT,read,load,write,sha
from compare_ball_candidate_decoders import components,choose
OUT=ROOT/'ball_input_stride_comparison'
def main():
    assert not (OUT/'verification.json').exists(),'Preserve diagnostic snapshot.'
    decision=load(OUT/'decision.json');protocol=load(OUT/'protocol.json');manifest=read(OUT/'input_manifest.csv')
    assert protocol['input_strides']==[1,2] and protocol['threshold']==.3 and protocol['fitting'] is False
    assert sha(ROOT/'compare_ball_input_stride.py')==protocol['script_sha256']
    for r in manifest:
        frames=json.loads(r['input_local_frames']);t=int(r['local_frame']);s=int(r['input_stride'])
        assert frames==[max(0,t-3*s),max(0,t-2*s),max(0,t-s),t]
    original_hashes=load(OUT/'input_image_hashes.json')
    assert all(sha(Path(p))==h for p,h in original_hashes.items())
    assert all(sha(Path(p))==h for p,h in decision['protected_input_hashes'].items())
    roles=['train','validation','challenge','short_sequence'];observations=[];inventories=[];summaries=[];reproduction_count=0
    for role in roles:
        for stride in [1,2]:
            rows=read(OUT/f'{role}_stride{stride}.csv');cache=np.load(OUT/f'{role}_stride{stride}_heatmaps.npz')
            assert list(cache['sample_ids'])==[r['sample_id'] for r in rows]
            for r,hm in zip(rows,cache['heatmaps']):
                cs=components(hm.astype(np.float32));picked=choose(cs,'bbox_area');assert int(r['visible'])==int(picked is not None)
                if picked:assert (picked['x_px'],picked['y_px'])==(int(r['x_px']),int(r['y_px']))
                reproduction_count+=1
                true=r['truth_visibility']=='visible'
                distances=[math.hypot(c['x_px']-float(r['truth_x_px']),c['y_px']-float(r['truth_y_px'])) for c in cs] if true else []
                near=min(distances,default=None)
                selected_good=bool(picked and true and math.hypot(picked['x_px']-float(r['truth_x_px']),picked['y_px']-float(r['truth_y_px']))<=16)
                mechanism='correct' if selected_good else 'selection_failure_with_real_candidate' if true and near is not None and near<=16 else 'no_candidate_near_manual_center' if true else 'not_visible_label_no_center'
                lost=r['sample_id'] in decision['results'][role]['prior_correct_lost'] if stride==2 else False
                observations.append(dict(role=role,stride=stride,sample_id=r['sample_id'],candidate_id=r['candidate_id'],local_frame=r['local_frame'],
                    truth_visible=int(true),candidate_count=len(cs),nearest_candidate_error_px=near,selected_correct16px=int(selected_good),
                    real_candidate_available16px=int(true and near is not None and near<=16),mechanism=mechanism,prior_temporal_correct_lost=int(lost)))
                for c in cs:inventories.append(dict(role=role,stride=stride,sample_id=r['sample_id'],**c))
            items=[q for q in observations if q['role']==role and q['stride']==stride]
            summaries.append(dict(role=role,stride=stride,truth_visible=sum(q['truth_visible'] for q in items),
                candidate_upper_bound16px=sum(q['real_candidate_available16px'] for q in items),selected_correct16px=sum(q['selected_correct16px'] for q in items),
                selection_failures=sum(q['mechanism']=='selection_failure_with_real_candidate' for q in items),
                no_candidate_near_truth=sum(q['mechanism']=='no_candidate_near_manual_center' for q in items)))
    write(OUT/'candidate_mechanisms.csv',observations);write(OUT/'candidate_inventory.csv',inventories);write(OUT/'candidate_availability_summary.csv',summaries)
    lost=[q for q in observations if q['prior_temporal_correct_lost']];write(OUT/'prior_correct_lost_mechanisms.csv',lost)
    cadence=read(OUT/'input_image_change_proxy.csv');proxy=[]
    for s in [1,2]:
        values=[float(r['mean_court_mad']) for r in cadence if r['role']=='train' and int(r['stride'])==s]
        proxy.append(dict(stride=s,training_examples=len(values),median_court_rgb_mean_absolute_change=float(np.median(values)),
            caveat='Encoded image-change proxy only; not an independently established motion-update frame rate.'))
    result=dict(status='saved_heatmaps_and_causal_input_alignment_verified',input_rows=len(manifest),decoded_rows_reproduced=reproduction_count,
        fresh_stride1_matches_old_raw=decision['stride1_reproduction'],protected_inputs_unchanged=True,source_image_hashes_verified=len(original_hashes),
        candidate_summary=summaries,prior_correct_lost_mechanisms=lost,training_image_change_proxy=proxy,
        six_false=decision['six_false_status_counts'],audit_not_run=not decision['audit_performed'],
        operational_outputs_changed=False,new_human_labels=0,additional_parameter_selection=False,
        input_timing_fix_proven=False,unseen_match_accuracy_established=False)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
