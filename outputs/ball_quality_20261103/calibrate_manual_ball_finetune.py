"""One explicit validation-only calibration after the fixed0.3 training trial.

Preserve the training trial. Select only if every baseline-correct validation
frame is retained and wrong-position+false-output count is no higher than3.
The previously seen60-frame audit is checked afterwards, never used for tuning.
"""
import json,sys,math
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha
from run_manual_ball_finetune import prepare,tensors,evaluate,metrics
import run_manual_ball_finetune as training
import torch

OUT=ROOT/'balltrack_manual_finetune_pilot/calibration'

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def correct_ids(rows):
    return {r['sample_id'] for r in rows if r['truth_visibility']=='visible' and int(r['visible']) and
        math.hypot(float(r['x_px'])-float(r['truth_x_px']),float(r['y_px'])-float(r['truth_y_px']))<=16}

def errors(m):return m['counts']['wrong_localization']+m['counts']['presence_fp']

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
    parent=OUT.parent;manifest=load(parent/'data_manifest.json');selection=load(parent/'checkpoint_selection.json')
    assert selection['selected_epoch']==2
    checkpoint=Path(selection['selected_checkpoint']);assert sha(checkpoint)==selection['selected_checkpoint_sha256']
    weight=SOURCE/'third_party/RacketVision/source/BallTrack/checkpoints/balltrack_best.pth'
    baseline_rows=read(parent/'validation_epoch00_predictions.csv');baseline=metrics(baseline_rows);original_correct=correct_ids(baseline_rows)
    assert baseline['counts']['localized_tp_16px']==80 and errors(baseline)==3
    thresholds=[.3,.4,.5,.6,.7]
    protocol=dict(status='fixed_before_calibration_evaluation',checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        threshold_grid=thresholds,comparison_baseline='Original pretrained checkpoint,threshold0.3',
        validation_only_constraints='Retain all80 originally correct validation sample IDs and wrong-position+nonvisible-output count<=3',
        selection='Original model included; among feasible candidates maximise localizedTP16px,then fewer errors. Keep original on exact tie; no audit tuning.',
        calibration_reason='Fine tuning at unchanged0.3 increased correct detections and false outputs. This is a separate development calibration, not a change to the original fixed training experiment.',
        data_reuse='Validation and audit have been seen previously; this remains development, not independent external evaluation',
        automatic_adoption=False,source_sha256=sha(weight),validation_partition_sha256=sha(parent/'validation_labels.csv'),
        audit_partition_sha256=sha(parent/'audit_labels.csv'),script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';sys.path.insert(0,str(modelroot))
    from inference import BallInferencer
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(checkpoint),device='cuda',thre=.3,batchsize=4)
    medians={};validation=prepare(read(parent/'validation_labels.csv'),medians)
    # Cache outputs once, apply the same decoder at each fixed threshold.
    maps=[]
    with torch.inference_mode():
        for first in range(0,len(validation),4):
            data=tensors(validation[first:first+4],medians,infer.device)
            with torch.amp.autocast('cuda'):hm,_,_=infer.model.forward(frames=data)
            maps.extend(hm[:,0].float().cpu().numpy())
    summaries=[];best=dict(kind='original',threshold=.3,metrics=baseline,correct_ids=original_correct,rows=baseline_rows)
    for threshold in thresholds:
        infer.thre=threshold;rows=[]
        for sample,hm in zip(validation,maps,strict=True):
            r=sample['row'];x,y,w,h,conf=infer._predict_location(hm)
            px=int((x+w/2)*1920/512);py=int((y+h/2)*1080/288)
            rows.append(dict(sample_id=r['sample_id'],candidate_id=r['candidate_id'],local_frame=r['local_frame'],
                source_frame=r['source_frame'],truth_visibility=r['truth_visibility'],truth_x_px=r['truth_x_px'],truth_y_px=r['truth_y_px'],
                x_px=px,y_px=py,visible=int(px!=0 or py!=0),confidence=conf,threshold=threshold))
        write(OUT/f'validation_threshold{int(threshold*100):02d}.csv',rows)
        m=metrics(rows);ids=correct_ids(rows);lost=sorted(original_correct-ids)
        feasible=not lost and errors(m)<=errors(baseline)
        summaries.append(dict(threshold=threshold,localized_tp_16px=m['counts']['localized_tp_16px'],
            wrong_positions=m['counts']['wrong_localization'],nonvisible_outputs=m['counts']['presence_fp'],
            original_correct_frames_lost=len(lost),lost_sample_ids=lost,feasible=feasible))
        candidate_key=(m['counts']['localized_tp_16px'],-errors(m));best_key=(best['metrics']['counts']['localized_tp_16px'],-errors(best['metrics']))
        if feasible and candidate_key>best_key:best=dict(kind='fine_tuned',threshold=threshold,metrics=m,correct_ids=ids,rows=rows)
    result=dict(selected_model=best['kind'],selected_threshold=best['threshold'],validation_selected=best['metrics'],
        original_correct_frame_loss=len(original_correct-best['correct_ids']),validation_original=baseline,
        threshold_results=summaries,audit_used_for_selection=False,automatic_adoption=False)
    dump(OUT/'selection.json',result)
    # Only now evaluate the chosen condition on the60-label audit.
    if best['kind']=='original':
        infer.model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=False)['state_dict'],strict=True)
    infer.thre=best['threshold'];audit=prepare(read(parent/'audit_labels.csv'),medians)
    old_training_out=training.OUT;training.OUT=OUT
    selected_audit,_=evaluate(infer,audit,medians,selection['selected_epoch'] if best['kind']=='fine_tuned' else 0,'audit_calibrated')
    training.OUT=old_training_out
    result.update(audit_selected=selected_audit,audit_original=load(parent/'decision.json')['audit_original'],
        original_checkpoint_and_reviewed_outputs_unchanged=True,
        conclusion='No automatic adoption; retain the original if no threshold satisfies the predefined validation non-regression constraints.')
    assert all(sha(Path(p))==h for p,h in manifest['input_hashes'].items())
    assert sha(checkpoint)==protocol['checkpoint_sha256'] and sha(weight)==protocol['source_sha256']
    dump(OUT/'decision.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
