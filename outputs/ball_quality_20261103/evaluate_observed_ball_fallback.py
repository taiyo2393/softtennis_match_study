"""Validation-only missing-output fallback from the fixed fine-tuned model.

Never changes an existing original detection. Candidates must be actual model
outputs and supported by nearby original observations on both sides. The line
is a gate only; no interpolated coordinates are exported.
"""
import json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha
from run_manual_ball_finetune import metrics

OUT=ROOT/'balltrack_observed_fallback'
P=dict(max_missing_run_frames=8,linear_support_radius_px=16.,
    thresholds=[.3,.4,.5,.6,.7],camera_changed_fraction=.25,camera_expand_before=12,camera_expand_after=4,
    camera_feature='Existing160x90 frame difference vs12frames earlier,meanBGR difference>24; not an independent camera-cut label',
    original_detection_policy='Preserve all original coordinates/confidences/presence, including existing mistakes',
    candidate_policy='Add actual fine-tuned same-frame output only to an original missing row',
    automatic_adoption=False)

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def camera_mask(scores):
    result=np.zeros(len(scores),bool)
    for t,s in enumerate(scores):
        if s>P['camera_changed_fraction']:result[max(0,t-12):min(len(scores),t+5)]=True
    return result

def xy(r):return np.array([float(r['x_px']),float(r['y_px'])])

def decide(raw,t,candidate,cut_mask):
    result=dict(local_frame=t,decision='original_detection_preserved',before_anchor='',after_anchor='',
        missing_run_frames='',reference_x_px='',reference_y_px='',candidate_residual_px='',camera_flag=int(cut_mask[t]))
    if int(raw[t]['visible']):return result
    if not int(candidate['visible']):result['decision']='no_finetuned_observation';return result
    if cut_mask[t]:result['decision']='camera_transition_abstain';return result
    before=next((i for i in range(t-1,max(-1,t-P['max_missing_run_frames']-2),-1) if int(raw[i]['visible'])),None)
    after=next((i for i in range(t+1,min(len(raw),t+P['max_missing_run_frames']+2)) if int(raw[i]['visible'])),None)
    if before is None or after is None:result['decision']='missing_two_sided_anchors';return result
    result.update(before_anchor=before,after_anchor=after,missing_run_frames=after-before-1)
    if after-before-1>P['max_missing_run_frames']:result['decision']='original_gap_too_long';return result
    if cut_mask[before:after+1].any():result['decision']='camera_transition_in_anchor_interval';return result
    fraction=(t-before)/(after-before);reference=xy(raw[before])*(1-fraction)+xy(raw[after])*fraction
    distance=float(np.linalg.norm(xy(candidate)-reference))
    result.update(reference_x_px=float(reference[0]),reference_y_px=float(reference[1]),candidate_residual_px=distance)
    result['decision']='add_observed_finetuned_candidate' if distance<=P['linear_support_radius_px'] else 'candidate_outside_support_gate'
    return result

def evaluate_samples(labels,raw_tables,candidate_tables,masks,threshold):
    result=[];diagnostics=[]
    for label in labels:
        cid=label['candidate_id'];t=int(label['local_frame']);raw=raw_tables[cid]
        candidate=candidate_tables[(cid,t)];d=decide(raw,t,candidate,masks[cid]);d.update(sample_id=label['sample_id'],candidate_id=cid,threshold=threshold)
        output=dict(raw[t])
        if d['decision']=='add_observed_finetuned_candidate':
            output.update({k:candidate[k] for k in ['x_px','y_px','visible','confidence']})
            assert not int(raw[t]['visible']) and int(output['visible'])
        else:assert output==raw[t]
        result.append(dict(sample_id=label['sample_id'],candidate_id=cid,local_frame=t,source_frame=label['source_frame'],
            truth_visibility=label['truth_visibility'],truth_x_px=label['truth_x_px'],truth_y_px=label['truth_y_px'],
            x_px=output['x_px'],y_px=output['y_px'],visible=output['visible'],confidence=output['confidence'],
            output_source='observed_finetuned_missing_fallback' if d['decision']=='add_observed_finetuned_candidate' else 'unchanged_original',interpolated=0))
        diagnostics.append(d)
    return metrics(result),result,diagnostics

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Keep completed selection protocols.'
    training=ROOT/'balltrack_manual_finetune_pilot';manifest=load(training/'data_manifest.json')
    selection=load(training/'checkpoint_selection.json');checkpoint=Path(selection['selected_checkpoint'])
    assert selection['selected_epoch']==2 and sha(checkpoint)==selection['selected_checkpoint_sha256']
    assert all(sha(Path(p))==h for p,h in manifest['input_hashes'].items())
    labels=read(training/'validation_labels.csv');assert len(labels)==120
    base=ROOT.parent/'tracking_quality_20261102/ball_aligned';camera=ROOT.parent/'ball_quality_20261102/temporal_guard/camera_features'
    hashes=dict(manifest['input_hashes']);raw_tables={};masks={}
    for cid in sorted({r['candidate_id'] for r in labels}):
        rp=base/cid/'balltrack_predictions.csv';cp=camera/(cid+'.json');data=load(cp)
        assert data['baseline_sha256']==sha(rp)
        raw_tables[cid]=read(rp);assert len(data['changed_fraction'])==len(raw_tables[cid])
        masks[cid]=camera_mask(data['changed_fraction']);hashes[str(rp)]=sha(rp);hashes[str(cp)]=sha(cp)
    for threshold in P['thresholds']:
        path=training/'calibration'/f'validation_threshold{int(threshold*100):02d}.csv';assert path.exists();hashes[str(path)]=sha(path)
    protocol=dict(status='fixed_before_fallback_selection',parameters=P,checkpoint_sha256=sha(checkpoint),
        validation_partition_sha256=sha(training/'validation_labels.csv'),
        source='Reuse already saved validation outputs; no target coordinates or labels used by decide()',
        rationale='Do not replace good original observations; test whether actual new observations can fill short, supported gaps',
        gate_reuse='16px and short-gap constraints reuse prior observed-candidate ideas; no new geometry sweep. Camera masking only suppresses new additions, never original observations.',
        selection='Original included. Max localizedTP16px subject to original-correct IDs retained and wrong_position+presenceFP<=original3; ties fewer errors, then fewer additions, then original.',
        audit='Apply the selected policy without changes to60 manual labels, then inspect continuous output before any adoption',
        scope='Repeatedly inspected same-match development data, not independent external testing',
        interpolation_used_as_output=False,protected_input_hashes=hashes,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    baseline_rows=[]
    for r in labels:
        p=raw_tables[r['candidate_id']][int(r['local_frame'])]
        baseline_rows.append(dict(sample_id=r['sample_id'],truth_visibility=r['truth_visibility'],truth_x_px=r['truth_x_px'],truth_y_px=r['truth_y_px'],
            x_px=p['x_px'],y_px=p['y_px'],visible=p['visible']))
    baseline=metrics(baseline_rows);assert baseline['counts']['localized_tp_16px']==80
    def correct_ids(rows):
        return {r['sample_id'] for r in rows if r['truth_visibility']=='visible' and int(r['visible']) and math.hypot(float(r['x_px'])-float(r['truth_x_px']),float(r['y_px'])-float(r['truth_y_px']))<=16}
    original_correct=correct_ids(baseline_rows);budget=baseline['counts']['wrong_localization']+baseline['counts']['presence_fp']
    summaries=[];best=None;best_key=(80,-budget,0);all_diagnostics=[]
    for threshold in P['thresholds']:
        path=training/'calibration'/f'validation_threshold{int(threshold*100):02d}.csv'
        candidates={(r['candidate_id'],int(r['local_frame'])):r for r in read(path)}
        m,rows,diagnostics=evaluate_samples(labels,raw_tables,candidates,masks,threshold)
        ids=correct_ids(rows);added=[r for r in rows if r['output_source']=='observed_finetuned_missing_fallback']
        error_count=m['counts']['wrong_localization']+m['counts']['presence_fp']
        feasible=not original_correct-ids and error_count<=budget
        scores=(m['counts']['localized_tp_16px'],-error_count,-len(added))
        if feasible and scores>best_key:best_key=scores;best=dict(threshold=threshold,metrics=m,rows=rows,added=added)
        summaries.append(dict(threshold=threshold,localized_tp_16px=m['counts']['localized_tp_16px'],
            wrong_positions=m['counts']['wrong_localization'],nonvisible_outputs=m['counts']['presence_fp'],
            additions=len(added),original_correct_frames_lost=len(original_correct-ids),feasible=feasible,
            added_samples=[r['sample_id'] for r in added]))
        write(OUT/f'validation_fallback_threshold{int(threshold*100):02d}.csv',rows)
        all_diagnostics.extend(diagnostics)
    write(OUT/'validation_decisions.csv',all_diagnostics)
    result=dict(status='fallback_validation_selection_complete',selected_policy='observed_short_gap_fallback' if best else 'original',
        selected_threshold=best['threshold'] if best else None,baseline=baseline,selected_validation=best['metrics'] if best else baseline,
        threshold_results=summaries,validation_added_samples=[r['sample_id'] for r in best['added']] if best else [],
        automatic_adoption=False,audit_used_for_selection=False,
        meaning='Existing original observations are unchanged; any addition is a real same-frame model coordinate, not an interpolated point',
        scope='Selected, repeatedly used same-match development labels')
    assert all(sha(Path(p))==h for p,h in hashes.items())
    dump(OUT/'selection.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
