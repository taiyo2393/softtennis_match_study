"""Post-trial diagnostics, with no model/settings selection or adoption."""
import json,math,sys
from pathlib import Path
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha
from run_manual_ball_finetune import prepare,metrics
from run_hard_case_ball_finetune import infer_maps,decode

OUT=ROOT/'balltrack_expanded_centers_pilot'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def status(r):
    if not int(r['visible']):return 'missing' if r['truth_visibility']=='visible' else 'correct_absent'
    if r['truth_visibility']!='visible':return 'false_presence'
    return 'correct' if math.hypot(float(r['x_px'])-float(r['truth_x_px']),float(r['y_px'])-float(r['truth_y_px']))<=16 else 'wrong_position'

def main():
    decision=load(OUT/'decision.json');protocol=load(OUT/'protocol.json')
    assert not (OUT/'post_trial_diagnostics.json').exists()
    control=read(OUT/'validation_control_epoch02_threshold30.csv');expanded=read(OUT/'validation_expanded_epoch02_threshold30.csv')
    original=read(OUT/'validation_original.csv');base={r['sample_id']:r for r in original}
    changed=[]
    for a,b in zip(control,expanded,strict=True):
        assert a['sample_id']==b['sample_id']
        if status(a)!=status(b):changed.append(dict(sample_id=a['sample_id'],original=status(base[a['sample_id']]),control=status(a),expanded=status(b),
            truth_x=a['truth_x_px'],truth_y=a['truth_y_px'],control_x=a['x_px'],control_y=a['y_px'],expanded_x=b['x_px'],expanded_y=b['y_px']))
    if changed:write(OUT/'validation_control_expansion_status_changes.csv',changed)
    # New training rows are diagnostic ONLY, not held-out performance or model selection.
    rows=read(OUT/'expanded_train_labels.csv')[-3:];assert len(rows)==3
    medians={};samples=prepare(rows,medians)
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';sys.path.insert(0,str(modelroot))
    from inference import BallInferencer
    weight=modelroot/'checkpoints/balltrack_best.pth'
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    configs=[('original',0,weight)]+[(r['family'],r['epoch'],Path(r['checkpoint'])) for r in load(OUT/'training_history.json')]
    fit=[];fit_summary=[]
    for family,epoch,path in configs:
        state=torch.load(path,map_location='cpu',weights_only=False)['state_dict'];infer.model.load_state_dict(state,strict=True)
        pred=decode(infer,infer_maps(infer,samples,medians),samples,.3,family,epoch,'new_training_centers_diagnostic_only')
        for r in pred:
            r['status16px']=status(r)
            r['center_error_px']=math.hypot(r['x_px']-float(r['truth_x_px']),r['y_px']-float(r['truth_y_px'])) if r['visible'] else ''
        fit.extend(pred);fit_summary.append(dict(family=family,epoch=epoch,correct=sum(r['status16px']=='correct' for r in pred),statuses=[r['status16px'] for r in pred]))
    write(OUT/'new_training_centers_fit_diagnostic.csv',fit)
    grid=read(OUT/'validation_selection_grid.csv')
    assert all(sha(Path(p))==h for p,h in protocol['protected_input_hashes'].items())
    summary=dict(validation_phase_scope='Includes preparation/transitions; not in-play-only',
        matched_epoch2_threshold03=decision['matched_epoch2_threshold03'],
        status_changes_control_to_expanded=changed,new_three_training_centers_fit=fit_summary,
        fitting_diagnostic_used_for_selection=False,feasible_configuration_count=sum(int(r['feasible']) for r in grid),
        strictly_improved_feasible_configurations=sum(int(r['feasible']) and (int(r['correct']),-int(r['errors']))>(80,-3) for r in grid),
        candidate_settings_checked=30,operational_original_retained=decision['validation_selected']['family']=='original',
        original_and_new_labels_models_reviewed_registries_unchanged=True,
        interpretation='New3 training targets do not constitute held-out evidence; assess their fit separately from validation. No more threshold/multiplier searches in this trial.')
    dump(OUT/'post_trial_diagnostics.json',summary);print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
