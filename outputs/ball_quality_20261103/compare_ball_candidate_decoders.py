"""One fixed comparison of component ranking with identical model/maps/centers.

Validation chooses among area, peak, bounding-box mean and component mean.
Only a strictly improving non-regressing choice proceeds to the fixed audit.
No retraining, new threshold, interpolation, learned scorer, or operational writes.
"""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2
from run_manual_ball_finetune import prepare,metrics
from run_hard_case_ball_finetune import infer_maps,good_ids,err

OUT=ROOT/'ball_candidate_decoder_comparison'
DATA=ROOT/'balltrack_manual_finetune_pilot'
METHODS=['bbox_area','component_peak','bbox_mean','component_mean']

def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def components(hm):
    binary=(hm>.3).astype('uint8');contours,_=cv2.findContours(binary*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for i,contour in enumerate(contours):
        x,y,w,h=cv2.boundingRect(contour)
        mask=np.zeros_like(binary);cv2.drawContours(mask,[contour],-1,1,-1)
        pixels=hm[(mask!=0)&(binary!=0)]
        assert len(pixels)>0
        candidates.append(dict(component_index=i,x=x,y=y,w=w,h=h,bbox_area=w*h,
            component_peak=float(pixels.max()),bbox_mean=float(hm[y:y+h,x:x+w].mean()),component_mean=float(pixels.mean()),
            component_pixels=len(pixels),x_px=int((x+w/2)*1920/512),y_px=int((y+h/2)*1080/288)))
    return candidates

def choose(candidates,method):
    assert method in METHODS
    if not candidates:return None
    # Existing area and contour ordering remain deterministic secondary tie breakers.
    return max(candidates,key=lambda c:(c[method],c['bbox_area'],-c['component_index']))

def self_test():
    blank=np.zeros((288,512),np.float32)
    assert all(choose(components(blank),m) is None for m in METHODS)
    hm=blank.copy();hm[10:20,10:20]=.4;hm[50:53,50:53]=.8
    cs=components(hm);assert len(cs)==2
    assert choose(cs,'bbox_area')['bbox_area']==100
    assert all(choose(cs,m)['bbox_area']==9 for m in METHODS[1:])
    equal=[dict(cs[0],component_index=0),dict(cs[0],component_index=1)]
    assert all(choose(equal,m)['component_index']==0 for m in METHODS)
    hm[:]=.3;assert components(hm)==[] # Threshold remains strictly greater than.3.
    print('Passed: empty maps, deterministic ties, ranking, fixed threshold and unchanged centers.')

def classify(r):
    if not int(r['visible']):return 'missing' if r['truth_visibility']=='visible' else 'correct_absent'
    if r['truth_visibility']!='visible':return 'false_presence'
    return 'correct' if math.hypot(float(r['x_px'])-float(r['truth_x_px']),float(r['y_px'])-float(r['truth_y_px']))<=16 else 'wrong_position'

def evaluate_role(infer,samples,medians,role):
    maps=infer_maps(infer,samples,medians)
    np.savez_compressed(OUT/f'{role}_original_heatmaps.npz',heatmaps=np.stack(maps).astype(np.float16),
        sample_ids=np.array([s['row']['sample_id'] for s in samples]))
    all_rows={m:[] for m in METHODS};inventory=[];coverage=[]
    for sample,hm in zip(samples,maps,strict=True):
        r=sample['row'];cs=components(hm)
        x,y,w,h,conf=infer._predict_location(hm);area=choose(cs,'bbox_area')
        assert (int((x+w/2)*1920/512),int((y+h/2)*1080/288))==((area['x_px'],area['y_px']) if area else (0,0))
        assert area is None or abs(conf-area['bbox_mean'])<1e-6
        for c in cs:
            inventory.append(dict(sample_id=r['sample_id'],role=role,candidate_id=r['candidate_id'],local_frame=r['local_frame'],**c))
        near=[math.hypot(c['x_px']-float(r['truth_x_px']),c['y_px']-float(r['truth_y_px'])) for c in cs] if r['truth_visibility']=='visible' else []
        coverage.append(dict(sample_id=r['sample_id'],role=role,truth_visibility=r['truth_visibility'],component_count=len(cs),
            any_component_within16px=int(bool(near) and min(near)<=16),nearest_component_error_px=min(near) if near else ''))
        for method in METHODS:
            c=choose(cs,method);px,py=(c['x_px'],c['y_px']) if c else (0,0)
            all_rows[method].append(dict(sample_id=r['sample_id'],role=role,method=method,candidate_id=r['candidate_id'],
                local_frame=r['local_frame'],source_frame=r['source_frame'],truth_visibility=r['truth_visibility'],
                truth_x_px=r['truth_x_px'],truth_y_px=r['truth_y_px'],x_px=px,y_px=py,visible=int(px!=0 or py!=0),
                confidence=c['bbox_mean'] if c else 0.,ranking_score=c[method] if c else 0.,
                selected_component=c['component_index'] if c else -1,component_count=len(cs)))
    for m,rows in all_rows.items():write(OUT/f'{role}_{m}.csv',rows)
    if inventory:write(OUT/f'{role}_all_components.csv',inventory)
    write(OUT/f'{role}_candidate_coverage.csv',coverage)
    base=all_rows['bbox_area'];assert all([r['visible'] for r in rows]==[r['visible'] for r in base] for rows in all_rows.values())
    return all_rows,coverage

def main():
    import torch
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'One fixed comparison. Preserve completed protocol.'
    manifest=load(DATA/'data_manifest.json');protected=dict(manifest['input_hashes'])
    for role,h in manifest['partition_hashes'].items():
        path=DATA/f'{role}_labels.csv';assert sha(path)==h;protected[str(path)]=h
    for path in [ROOT/'ball_training_case_expansion/manual_center_labels.csv',ROOT/'balltrack_expanded_centers_pilot/decision.json']:
        protected[str(path)]=sha(path)
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth'
    protected[str(weight)]=sha(weight)
    assert all(sha(Path(p))==h for p,h in protected.items())
    protocol=dict(status='fixed_before_new_maps_and_validation_comparison',methods=METHODS,
        source_checkpoint=str(weight),source_checkpoint_sha256=sha(weight),model_changes=False,threshold=.3,
        candidates='Same external contours above0.3, same source bbox centers and mean score; no candidates removed or added',
        component_scores='peak and mean on thresholded pixels within that contour; bbox_mean equals existing confidence; area=w*h',
        tie_breakers='Larger bbox area, then earlier OpenCV contour index, for all methods',
        validation='Fixed120 manually clicked/visibility labels,8 clips; no training or fitting',
        selection='Original bbox_area included. Preserve every original correct validation frame and errors<=3. Max correct16px, then fewer wrong_position+presenceFP. Method order breaks ties, original first.',
        audit='Only after immutable validation choice. If no strict improvement, no audit evaluation; otherwise evaluate selected method against original on fixed60 labels.',
        post_audit='Safety check only; never select another method on audit failures. No automatic operational registry or weight changes.',
        training_examples='A/C/D only qualitative development diagnostics, not counted as validation/audit; B excluded as post-play by user',
        scope='Same-match previously seen sparse labels with preparation and transitions, not in-play-only or external testing',
        no_adaptive_score_threshold_search=True,no_interpolation=True,no_new_user_labels=True,
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)),inference_helper_sha256=sha(ROOT/'run_hard_case_ball_finetune.py'))
    dump(OUT/'protocol.json',protocol)
    sys.path.insert(0,str(modelroot));from inference import BallInferencer
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    medians={};validation=prepare(read(DATA/'validation_labels.csv'),medians)
    results,coverage=evaluate_role(infer,validation,medians,'validation')
    baseline=results['bbox_area'];bm=metrics(baseline);original_ids=good_ids(baseline)
    assert len(original_ids)==80 and err(bm)==3
    previous=read(ROOT/'balltrack_expanded_centers_pilot/validation_original.csv')
    assert all(a['sample_id']==b['sample_id'] and a['visible']==int(b['visible']) and
        abs(a['x_px']-float(b['x_px']))<=2 and abs(a['y_px']-float(b['y_px']))<=2 for a,b in zip(baseline,previous,strict=True))
    summaries=[];selected='bbox_area';best=(80,-3)
    for method in METHODS:
        rows=results[method];m=metrics(rows);lost=sorted(original_ids-good_ids(rows));feasible=not lost and err(m)<=3
        summaries.append(dict(method=method,correct=m['counts']['localized_tp_16px'],errors=err(m),
            wrong_position=m['counts']['wrong_localization'],presence_fp=m['counts']['presence_fp'],
            original_correct_lost=len(lost),lost_sample_ids=';'.join(lost),feasible=int(feasible)))
        if feasible and (m['counts']['localized_tp_16px'],-err(m))>best:
            selected=method;best=(m['counts']['localized_tp_16px'],-err(m))
    write(OUT/'validation_summary.csv',summaries)
    selection=dict(selected_method=selected,strict_validation_improvement=selected!='bbox_area',
        selected_validation=metrics(results[selected]),baseline_validation=bm,audit_used_for_selection=False,
        summaries=summaries,candidate_center_upper_bound=sum(r['any_component_within16px'] for r in coverage),
        upper_bound_warning='Any candidate center near manual point is an oracle diagnostic, not selectable accuracy',automatic_operational_adoption=False)
    dump(OUT/'validation_selection.json',selection);print('validation selection',json.dumps(selection),flush=True)
    changes=[]
    for a,b in zip(baseline,results[selected],strict=True):
        if (a['x_px'],a['y_px'])!=(b['x_px'],b['y_px']):
            changes.append(dict(sample_id=a['sample_id'],candidate_id=a['candidate_id'],local_frame=a['local_frame'],
                truth_visibility=a['truth_visibility'],truth_x_px=a['truth_x_px'],truth_y_px=a['truth_y_px'],
                before_status=classify(a),after_status=classify(b),before_x=a['x_px'],before_y=a['y_px'],after_x=b['x_px'],after_y=b['y_px']))
    if changes:write(OUT/'validation_selected_changes.csv',changes)
    audit_result=None;audit_lost=[];audit_pass=None
    if selected!='bbox_area':
        # The selected rule is already saved. Other audit rankings are not evaluated/selected.
        audit=prepare(read(DATA/'audit_labels.csv'),medians);maps=infer_maps(infer,audit,medians)
        audit_tables={m:[] for m in ['bbox_area',selected]}
        for s,hm in zip(audit,maps,strict=True):
            r=s['row'];cs=components(hm)
            for method in audit_tables:
                c=choose(cs,method);px,py=(c['x_px'],c['y_px']) if c else (0,0)
                audit_tables[method].append(dict(sample_id=r['sample_id'],candidate_id=r['candidate_id'],local_frame=r['local_frame'],
                    source_frame=r['source_frame'],method=method,truth_visibility=r['truth_visibility'],truth_x_px=r['truth_x_px'],truth_y_px=r['truth_y_px'],
                    visible=int(px!=0 or py!=0),x_px=px,y_px=py,confidence=c['bbox_mean'] if c else 0.))
        for m,rows in audit_tables.items():write(OUT/f'audit_{m}.csv',rows)
        ab=metrics(audit_tables['bbox_area']);am=metrics(audit_tables[selected]);assert ab['counts']['localized_tp_16px']==34 and err(ab)==5
        audit_lost=sorted(good_ids(audit_tables['bbox_area'])-good_ids(audit_tables[selected]));audit_pass=not audit_lost and err(am)<=err(ab)
        audit_result=dict(original=ab,selected=am,original_correct_lost=audit_lost,nonregression=audit_pass)
    extra=prepare(read(ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv')[-3:],medians)
    extra_results,_=evaluate_role(infer,extra,medians,'three_training_examples')
    diagnostic={m:metrics(rows) for m,rows in extra_results.items()}
    assert all(sha(Path(p))==h for p,h in protected.items())
    decision=dict(status='fixed_decoder_comparison_complete',selected_method=selected,validation=selection,
        audit=audit_result,audit_skipped=selected=='bbox_area',three_training_example_diagnostics=diagnostic,
        operational_outputs_unchanged=True,all_protected_inputs_unchanged=True,model_and_threshold_unchanged=True,
        selected_rule_passes_this_development_check=selected!='bbox_area' and audit_pass is True,
        independent_external_accuracy=False,main_play_only_accuracy=False,
        next='Review changed annotated frames and temporal context; retain human corrections. No selection using new training examples or audit. Broader in-play evaluation still needed.')
    dump(OUT/'decision.json',decision);print(json.dumps(decision,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test()
    else:main()
