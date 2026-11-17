"""Select conservative weak-component recovery on validation labels only."""
import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'tracking_quality_20261102'))
from ball_alignment import SOURCE,read,evaluate

VARIANTS=[dict(name=f'gap{gap}_peak{peak:.2f}',max_gap=gap,min_peak=peak,radius_px=16.,
               all_missing_frames_supported=True) for gap in (4,8,12) for peak in (.10,.15)]


def recover(rows,components,config):
    result=[dict(r,position_source='baseline_model_observed' if int(r['visible']) else 'missing',
                 recovery_anchor_before='',recovery_anchor_after='',
                 baseline_threshold=r.get('threshold','0.30'),recovery_peak_min='') for r in rows]
    if config is None:return result
    byframe=defaultdict(list)
    for c in components:byframe[int(c['local_frame'])].append(c)
    anchors=[i for i,r in enumerate(rows) if int(r['visible'])]
    for left,right in zip(anchors,anchors[1:]):
        gap=right-left-1
        if not 0<gap<=config['max_gap']:continue
        proposals=[]
        for f in range(left+1,right):
            fraction=(f-left)/(right-left)
            x=float(rows[left]['x_px'])*(1-fraction)+float(rows[right]['x_px'])*fraction
            y=float(rows[left]['y_px'])*(1-fraction)+float(rows[right]['y_px'])*fraction
            options=[(math.hypot(float(c['x_px'])-x,float(c['y_px'])-y),c) for c in byframe[f]
                     if float(c['peak'])>=config['min_peak']]
            options.sort(key=lambda p:p[0])
            if not options or options[0][0]>config['radius_px']:break
            proposals.append((f,options[0][1]))
        if len(proposals)!=gap:continue
        for f,c in proposals:
            result[f].update(x_px=c['x_px'],y_px=c['y_px'],visible=1,confidence=c['peak'],
                position_source='weak_heatmap_temporally_supported',
                threshold=.10,recovery_peak_min=config['min_peak'],
                recovery_anchor_before=left,recovery_anchor_after=right)
    return result


def load_phase(phase,config):
    predictions={};added=0
    numbers=range(1,11) if phase=='validation' else range(11,21)
    for n in numbers:
        cid=f'match1_motion_pilot_{n:03d}'
        rows=read(ROOT.parent/'tracking_quality_20261102/ball_aligned'/cid/'balltrack_predictions.csv')
        cs=read(ROOT/'weak_candidates'/cid/'candidates.csv') if config is not None else []
        rs=recover(rows,cs,config)
        predictions[cid]={int(r['local_frame']):r for r in rs}
        added+=sum(r['position_source']=='weak_heatmap_temporally_supported' for r in rs)
    return predictions,added


def output_phase(phase,config):
    import csv
    pred,added=load_phase(phase,config)
    for cid,byframe in pred.items():
        rows=list(byframe.values());dest=ROOT/'selected_predictions'/cid;dest.mkdir(parents=True,exist_ok=True)
        with (dest/'balltrack_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return pred,added


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['select-validation','evaluate-test'],required=True);args=p.parse_args()
    labelpath=SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv'
    if args.phase=='select-validation':
        assert not (ROOT/'test_results.json').exists(),'Do not retune after examining this test'
        plan=dict(variants=VARIANTS,selection='Maximum localized validation TP with no added nonlocalized labeled outputs; ties prefer fewer added candidates',
                  tolerance_px=16,image_size=[1920,1080],interpolation=False)
        (ROOT/'selection_protocol.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
        labels=[r for r in read(labelpath) if r['evaluation_split']=='validation']
        baseline,_=load_phase('validation',None)
        bm=evaluate(labels,baseline,'validation')
        baseline_bad=bm['counts']['pred_visible']-bm['counts']['localized_tp_16px']
        reports=[];selected=None;best_tp=bm['counts']['localized_tp_16px'];best_added=0
        for config in VARIANTS:
            pred,added=load_phase('validation',config)
            metrics=evaluate(labels,pred,'validation')
            bad=metrics['counts']['pred_visible']-metrics['counts']['localized_tp_16px']
            eligible=bad<=baseline_bad
            tp=metrics['counts']['localized_tp_16px']
            reports.append(dict(config=config,added_frames_all_clips=added,metrics=metrics,
                                nonlocalized_outputs_on_labeled_frames=bad,eligible=eligible))
            if eligible and (tp>best_tp or (tp==best_tp and added<best_added)):
                selected=config;best_tp=tp;best_added=added
        selection=dict(config=selected,baseline_metrics=bm,validation_candidates=reports,
            label_sha256=hashlib.sha256(labelpath.read_bytes()).hexdigest(),
            warning='Existing labels have unverified independent annotation provenance. No test labels used for selection.')
        (ROOT/'selected_method.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
        output_phase('validation',selected)
        print(json.dumps(dict(selected=selected,baseline=bm,chosen_true_positives=best_tp,added_frames=best_added)),flush=True)
    else:
        selection_path=ROOT/'selected_method.json';selection=json.loads(selection_path.read_text(encoding='utf-8'))
        labels=[r for r in read(labelpath) if r['evaluation_split']=='test']
        baseline,_=load_phase('test',None)
        pred,added=output_phase('test',selection['config'])
        result=dict(selected_method_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
            selected_config=selection['config'],baseline=evaluate(labels,baseline,'test'),
            recovered=evaluate(labels,pred,'test'),added_frames_all_clips=added,
            warning='Previously used same-match test and provenance-unverified labels; not a new independent external test.')
        (ROOT/'test_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
