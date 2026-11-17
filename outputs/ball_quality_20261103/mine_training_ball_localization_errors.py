"""Mine far-away predictions from TRAINING clips only; do not invent ball classes.

Manual target elsewhere means a wrong localization of that annotated target.
It does not certify that a different real ball cannot exist at the prediction.
"""
import json,math
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha

OUT=ROOT/'balltrack_manual_finetune_pilot/training_hard_cases'

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'manifest.json').exists()
    parent=OUT.parent;source=parent/'train_labels.csv';labels=read(source)
    manifest=load(parent/'data_manifest.json');assert sha(source)==manifest['partition_hashes']['train']
    assert len(labels)==120 and {int(r['candidate_id'].rsplit('_',1)[1]) for r in labels}==set(manifest['train_clips'])
    inputs={str(source):sha(source)};base=ROOT.parent/'tracking_quality_20261102/ball_aligned';tables={};cases=[];targets=[]
    for r in labels:
        if r['truth_visibility']!='visible':continue
        cid=r['candidate_id'];t=int(r['local_frame'])
        if cid not in tables:
            p=base/cid/'balltrack_predictions.csv';tables[cid]=read(p);inputs[str(p)]=sha(p)
        pred=tables[cid][t]
        if not int(pred['visible']):continue
        error=math.hypot(float(pred['x_px'])-float(r['truth_x_px']),float(pred['y_px'])-float(r['truth_y_px']))
        if error<=64:continue
        cases.append(dict(sample_id=r['sample_id'],candidate_id=cid,local_frame=t,source_frame=r['source_frame'],
            role='training_only',manual_target_x=r['truth_x_px'],manual_target_y=r['truth_y_px'],
            wrong_target_candidate_x=pred['x_px'],wrong_target_candidate_y=pred['y_px'],target_distance_px=error,
            derived_label='wrong_location_for_this_manual_target',physical_object_class='not_established',
            phase='not_established',potential_other_ball='not_ruled_out',training_ready_for_dense_target_loss=1,
            training_ready_for_generic_non_ball_classification=0,
            point_target_source='user_confirmed_manual_center_table',baseline_prediction_sha256=inputs[str(base/cid/'balltrack_predictions.csv')]))
    if cases:write(OUT/'far_wrong_target_predictions.csv',cases)
    report=dict(status='training_only_far_wrong_target_inventory',train_samples=120,train_clips=manifest['train_clips'],
        candidates=len(cases),source_distance_min_exclusive_px=64,
        mining_rule='Stored visible manual target plus original model output>64sourcepixels away; never read validation/audit labels for mining',
        object_semantics='Wrong position for annotated target, not a claim of no physical ball at that position',
        next='Use explicit training-side target losses or sampling of these examples in a new, separately specified trial; do not copy known validation failures into training.',
        original_targets_unchanged=True,validation_and_audit_not_mined=True,model_not_trained_in_this_step=True,
        input_hashes=inputs)
    assert all(sha(Path(p))==h for p,h in inputs.items())
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='input_hashes'},indent=2))

if __name__=='__main__':main()
