"""One training-only revision after four dual-input policies failed safety.

Reuse previously specified two-sided short-gap gate unchanged. No new tuning.
"""
import json
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha
from evaluate_observed_ball_fallback import decide,P,camera_mask
from run_manual_ball_finetune import metrics
from run_hard_case_ball_finetune import good_ids,err

OUT=ROOT/'ball_dual_stride_anchor_guard';PARENT=ROOT/'ball_dual_stride_path_comparison'
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'training_protocol.json').exists(),'Preserve this fixed training-only revision.'
    parent=load(PARENT/'decision.json');assert parent['training']['chosen'] is None and parent['validation'] is None
    baseline_path=PARENT/'train_baseline.csv';candidate_path=PARENT/'train_missing_only_union_path.csv'
    baseline=read(baseline_path);proposed=read(candidate_path);assert len(baseline)==len(proposed)==123
    paths=[PARENT/'protocol.json',PARENT/'decision.json',PARENT/'training_selection.json',baseline_path,candidate_path,
        ROOT/'evaluate_observed_ball_fallback.py',ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json']
    protected={str(p):sha(p) for p in paths}
    for cid in sorted({r['candidate_id'] for r in baseline}):
        for p in [ROOT.parent/'tracking_quality_20261102/ball_aligned'/cid/'balltrack_predictions.csv',ROOT.parent/'ball_quality_20261102/temporal_guard/camera_features'/f'{cid}.json']:
            protected[str(p)]=sha(p)
    assert P['max_missing_run_frames']==8 and P['linear_support_radius_px']==16
    protocol=dict(status='fixed_before_guard_training_evaluation',revision_reason='All four predefined dual-input policies preserved old correct points but added training errors. Evaluation inference was not run.',
        candidate_policy='Already fixed missing_only_union_path: actual candidate from stride1/stride2 pooled path. Preserve all existing stride1 temporal outputs.',
        guard='Reuse evaluate_observed_ball_fallback.decide unmodified; original full raw stride1 observations are anchors; short gap<=8source frames, both sides, line residual<=16source pixels, existing camera guard.',
        guard_values=dict(max_missing_run_frames=8,linear_support_radius_px=16),
        geometry='Line is a gate only, never a predicted/interpolated output coordinate. Source candidate remains actual same-frame observation.',
        selection='Single guarded variant. Need retain all66training correct, errors<=4 and correct>66 or errors<4. No gate/threshold sweep or case-specific exceptions.',
        if_pass='Freeze guarded recipe before any new validation inference; evaluate against original baseline on validation120, challenge4, short26; audit only after full nonregression and validation benefit.',
        independence='Known same-match train development. Revision follows training result; not preregistered before the initial four policies.',
        validation_used=False,challenge_used=False,short_used=False,six_false_used=False,new_labels=0,automatic_adoption=False,
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    (OUT/'training_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    raw={cid:read(ROOT.parent/'tracking_quality_20261102/ball_aligned'/cid/'balltrack_predictions.csv') for cid in {r['candidate_id'] for r in baseline}}
    masks={cid:camera_mask(load(ROOT.parent/'ball_quality_20261102/temporal_guard/camera_features'/f'{cid}.json')['changed_fraction']) for cid in raw}
    outputs=[];gate_rows=[]
    for a,b in zip(baseline,proposed):
        assert a['sample_id']==b['sample_id'];cid=a['candidate_id'];t=int(a['local_frame'])
        q=dict(a,method='anchored_missing_only_union_path',anchor_gate_status='unchanged_baseline',anchor_before='',anchor_after='',anchor_residual_px='')
        if not int(a['visible']) and int(b['visible']):
            assert not int(raw[cid][t]['visible'])
            d=decide(raw[cid],t,b,masks[cid]);gate_rows.append(dict(sample_id=a['sample_id'],candidate_id=cid,**d))
            if d['decision']=='add_observed_finetuned_candidate':
                for k in ['x_px','y_px','visible','confidence','input_stride']:q[k]=b[k]
                q['decision']='add_actual_dual_input_candidate_with_fixed_anchors'
            q.update(anchor_gate_status=d['decision'],anchor_before=d['before_anchor'],anchor_after=d['after_anchor'],anchor_residual_px=d['candidate_residual_px'])
        if int(a['visible']):assert (q['x_px'],q['y_px'],q['visible'],q['confidence'])==(a['x_px'],a['y_px'],a['visible'],a['confidence'])
        if int(q['visible']) and not int(a['visible']):assert (q['x_px'],q['y_px'])==(b['x_px'],b['y_px'])
        outputs.append(q)
    write(OUT/'train_guarded_predictions.csv',outputs)
    if gate_rows:write(OUT/'training_gate_decisions.csv',gate_rows)
    bm=metrics(baseline);pm=metrics(outputs);lost=sorted(good_ids(baseline)-good_ids(outputs));gained=sorted(good_ids(outputs)-good_ids(baseline))
    passes=not lost and err(pm)<=4 and (len(good_ids(outputs))>66 or err(pm)<4)
    assert all(sha(Path(p))==h for p,h in protected.items())
    result=dict(status='single_anchored_training_revision_complete',baseline=bm,proposed=pm,correct_lost=lost,correct_gained=gained,
        training_pass=passes,eligible_for_frozen_validation=passes,added_outputs=sum(not int(a['visible']) and int(b['visible']) for a,b in zip(baseline,outputs)),
        gate_status_counts={s:sum(r['decision']==s for r in gate_rows) for s in sorted({r['decision'] for r in gate_rows})},
        validation_used=False,challenge_used=False,short_used=False,six_false_used=False,automatic_adoption=False,operational_outputs_changed=False,protected_inputs_unchanged=True)
    (OUT/'training_selection.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
