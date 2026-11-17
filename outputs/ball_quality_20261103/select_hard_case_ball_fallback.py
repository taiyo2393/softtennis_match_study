"""Apply unchanged observed-gap policy to matched ordinary/weighted families.

Select on validation only, no new geometry/appearance/camera parameter tuning.
Both families receive the same3epochs x5threshold grid. Old source retained.
"""
import json,math
from pathlib import Path
import numpy as np
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha
from evaluate_observed_ball_fallback import P,camera_mask,evaluate_samples
from run_manual_ball_finetune import metrics

OUT=ROOT/'balltrack_hard_case_finetune_pilot/fallback'

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
    family_dir=OUT.parent;data=ROOT/'balltrack_manual_finetune_pilot';m=load(data/'data_manifest.json')
    protected=dict(m['input_hashes']);labels=read(data/'validation_labels.csv')
    base=ROOT.parent/'tracking_quality_20261102/ball_aligned';camera=ROOT.parent/'ball_quality_20261102/temporal_guard/camera_features'
    raw={};masks={}
    for cid in sorted({r['candidate_id'] for r in labels}):
        path=base/cid/'balltrack_predictions.csv';cp=camera/(cid+'.json');c=load(cp)
        assert c['baseline_sha256']==sha(path)
        raw[cid]=read(path);masks[cid]=camera_mask(c['changed_fraction']);protected[str(path)]=sha(path);protected[str(cp)]=sha(cp)
    for family in ['ordinary','hard_weighted']:
        for epoch in range(1,4):
            for threshold in P['thresholds']:
                p=family_dir/f'validation_{family}_epoch{epoch:02d}_threshold{int(threshold*100):02d}.csv';protected[str(p)]=sha(p)
    assert all(sha(Path(p))==h for p,h in protected.items())
    protocol=dict(status='fixed_before_matched_fallback_selection',families=['ordinary','hard_weighted'],epochs=[1,2,3],thresholds=P['thresholds'],
        frozen_gate_parameters=P,source_gate_script_sha256=sha(ROOT/'evaluate_observed_ball_fallback.py'),
        trial_reason='Hard-weighted validation counts improved over ordinary weights, but full replacement still increases errors. Reuse the previous gap policy, do not change its gates.',
        selection='Original included. All original correct frames retained; wrong+presenceFP<=3. MaxTP16,then fewer errors,then fewer additions,ordinary wins ties. Both families get identical grids.',
        source_data='Saved predictions only; no validation labels in gate logic',automatic_adoption=False,
        audit_used_for_selection=False,scope='Retrospective same-match development after multiple prior experiments; not independent external evaluation',
        protected_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    baseline=[]
    for label in labels:
        r=raw[label['candidate_id']][int(label['local_frame'])]
        baseline.append(dict(sample_id=label['sample_id'],truth_visibility=label['truth_visibility'],truth_x_px=label['truth_x_px'],truth_y_px=label['truth_y_px'],x_px=r['x_px'],y_px=r['y_px'],visible=r['visible']))
    bm=metrics(baseline)
    def correct_ids(rows):return {r['sample_id'] for r in rows if r['truth_visibility']=='visible' and int(r['visible']) and math.hypot(float(r['x_px'])-float(r['truth_x_px']),float(r['y_px'])-float(r['truth_y_px']))<=16}
    original_ids=correct_ids(baseline);assert len(original_ids)==80
    best=dict(family='original',epoch=0,threshold=.3,metrics=bm,added_samples=[]);best_key=(80,-3,0);summary=[];family_best={}
    for family in ['ordinary','hard_weighted']:
        fb=dict(best) if best['family']=='original' else dict(family='original',epoch=0,threshold=.3,metrics=bm,added_samples=[])
        fk=(80,-3,0)
        for epoch in range(1,4):
            for threshold in P['thresholds']:
                path=family_dir/f'validation_{family}_epoch{epoch:02d}_threshold{int(threshold*100):02d}.csv'
                candidates={(r['candidate_id'],int(r['local_frame'])):r for r in read(path)}
                score,rows,diagnostics=evaluate_samples(labels,raw,candidates,masks,threshold)
                ids=correct_ids(rows);error=score['counts']['wrong_localization']+score['counts']['presence_fp']
                added=[r['sample_id'] for r in rows if r['output_source']=='observed_finetuned_missing_fallback']
                feasible=not(original_ids-ids) and error<=3;candidate_key=(score['counts']['localized_tp_16px'],-error,-len(added))
                item=dict(family=family,epoch=epoch,threshold=threshold,tp16=score['counts']['localized_tp_16px'],errors=error,additions=len(added),
                    original_correct_lost=len(original_ids-ids),feasible=int(feasible),added_samples=';'.join(added))
                summary.append(item)
                write(OUT/f'validation_{family}_epoch{epoch:02d}_threshold{int(threshold*100):02d}.csv',rows)
                if feasible and candidate_key>fk:
                    fk=candidate_key;fb=dict(family=family,epoch=epoch,threshold=threshold,metrics=score,added_samples=added)
                if feasible and candidate_key>best_key:
                    best_key=candidate_key;best=dict(family=family,epoch=epoch,threshold=threshold,metrics=score,added_samples=added)
                    write(OUT/'selected_validation.csv',rows);write(OUT/'selected_validation_decisions.csv',diagnostics)
        family_best[family]=fb
    write(OUT/'validation_grid.csv',summary)
    result=dict(status='matched_fallback_validation_complete',baseline=bm,selected=best,family_choices=family_best,
        audit_used_for_selection=False,automatic_adoption=False,original_observations_unchanged=True,
        any_addition_uses_same_frame_model_coordinates=True,geometry_and_camera_parameters_unchanged=True,
        scope='Selected sparse same-match development samples, not full-rally tracking validation')
    assert all(sha(Path(p))==h for p,h in protected.items())
    dump(OUT/'selection.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
