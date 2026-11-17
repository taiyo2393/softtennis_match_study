"""Verify training-only selection, exact sparse cache and fixed source states."""
import json,math
from pathlib import Path
from collections import Counter
import numpy as np
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha
from calibrate_point_supervised_detector import OUT,MODELS,THRESHOLDS,components_at,expand
from compare_ball_candidate_decoders import components

def main():
    assert not (OUT/'verification.json').exists()
    p=load(OUT/'protocol.json');d=load(OUT/'decision.json');selection=load(OUT/'training_selection.json')
    assert sha(OUT/'executed_source.py')==p['script_sha256']
    assert selection['chosen'] is None and all(d[k] is None for k in ['validation','challenge','short_sequence','six_false','audit'])
    for role in ['validation','challenge','short_sequence','six_false','audit']:assert not list(OUT.glob(f'*_{role}_compact_heatmaps.npz')) and not (OUT/f'{role}_selected.csv').exists()
    assert all(sha(Path(path))==h for path,h in d['protected_input_hashes'].items())
    grid=read(OUT/'training_threshold_grid.csv');assert len(grid)==16 and all(r['feasible']=='0' for r in grid)
    repro=read(OUT/'threshold03_reproduction.csv');assert len(repro)==246 and all(r['same']=='1' for r in repro)
    sparse_records=[];verified_nodes=0
    for arm in ['positive_control','point_negative']:
        path=OUT/f'{arm}_train_compact_heatmaps.npz'
        # Decompress each NPZ member once, rather than once for every context frame.
        with np.load(path) as cache:
            clips=cache['clips'];frames=cache['local_frames'];offsets=cache['offsets'];all_indices=cache['indices'];all_values=cache['values']
        context=load(OUT/f'{arm}_train_component_cache.json');assert len(offsets)==len(frames)+1
        assert len(all_values)==len(all_indices)==int(offsets[-1])
        for i,(cid,t) in enumerate(zip(clips,frames)):
            start,stop=offsets[i:i+2];idx=all_indices[start:stop];v=all_values[start:stop];hm=expand(idx,v)
            assert len(idx)==len(set(idx.tolist())) and np.isfinite(v).all()
            assert components_at(hm,.3)==context[str(cid)][str(int(t))]
            verified_nodes+=1
        hashes=load(OUT/f'{arm}_train_context_image_hashes.json');assert all(sha(Path(path))==h for path,h in hashes.items())
        sparse_records.append(dict(arm=arm,context_frames=len(frames),retained_values=len(all_values),file_bytes=path.stat().st_size,sha256=sha(path)))
    # Point-order diagnostic on known9training pairs only, never candidate-object accuracy.
    original=read(MODELS/'original_training_point_scores.csv');final=read(MODELS/'final_training_point_scores.csv');pairs=[];pair_summary=[]
    for arm,rows in [('original',original),('positive_control',[r for r in final if r['arm']=='positive_control']),('point_negative',[r for r in final if r['arm']=='point_negative'])]:
        field='original_probability' if arm=='original' else 'probability';positive={r['sample_id']:float(r[field]) for r in rows if r['kind']=='manual_positive'}
        sub=[]
        for r in rows:
            if r['kind']!='confirmed_non_ball':continue
            a,b=positive[r['sample_id']],float(r[field]);sub.append(dict(arm=arm,sample_id=r['sample_id'],true_center_response=a,non_ball_center_response=b,
                true_above_non_ball=int(a>b),margin=a-b,single_threshold_can_separate_this_point_pair=int(a>b)))
        assert len(sub)==9;pairs.extend(sub);pair_summary.append(dict(arm=arm,point_pairs=9,true_above_non_ball=sum(r['true_above_non_ball'] for r in sub),
            mean_point_response_margin=float(np.mean([r['margin'] for r in sub])),not_object_detection_accuracy=True))
    write(OUT/'training_point_pair_order.csv',pairs)
    # Training tradeoffs are retained; do not call infeasible settings worse on every metric.
    frontier=[]
    candidates=[dict(arm='original_baseline',threshold=.3,correct=66,errors=4,correct_lost=0)]+[
        dict(arm=r['arm'],threshold=float(r['threshold']),correct=int(r['correct']),errors=int(r['errors']),correct_lost=int(r['correct_lost'])) for r in grid]
    for r in candidates:
        r['pareto_by_correct_and_errors']=int(not any(s['correct']>=r['correct'] and s['errors']<=r['errors'] and (s['correct']>r['correct'] or s['errors']<r['errors']) for s in candidates))
        frontier.append(r)
    write(OUT/'training_tradeoffs.csv',frontier)
    report=dict(status='passed',model_weights_unchanged=True,retraining=False,thresholds=THRESHOLDS,training_grid_rows=16,feasible_settings=0,
        threshold03_predictions_reproduced=246,sparse_context_nodes_reproduced=verified_nodes,sparse_cache=sparse_records,
        no_new_evaluation_inference=True,protected_inputs_unchanged=True,point_order_diagnostic=pair_summary,
        source_snapshot_sha256=sha(OUT/'executed_source.py'),current_script_sha256=sha(ROOT/'calibrate_point_supervised_detector.py'),
        source_fix_note='After the run completed, only embedded synthetic fixture FP32 0.1 was rounded to FP16. Original executed source is preserved. No inference/selection code or result changed.',
        automatic_adoption=False,operational_outputs_changed=False,new_human_labels=0,interpolation=0,independent_generalization_established=False)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
