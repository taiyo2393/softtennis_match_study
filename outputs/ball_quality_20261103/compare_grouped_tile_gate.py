"""Fixed cross-scale acceptance rule versus grouped training-only logistic gate."""
import json,math,shutil
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha
from compare_frozen_ball_backbones import fit,predict
from run_manual_ball_finetune import metrics
from run_hard_case_ball_finetune import good_ids,err

BASE=ROOT.parent/'ball_quality_20261109/training_tile_detail_comparison'
OUT=ROOT.parent/'ball_quality_20261109/training_tile_gate_grouped'
F=['native_peak','native_bbox_mean','log_component_pixels','fill_fraction','log_aspect_ratio',
   'native_agreement_fraction16','degraded_peak_within16','degraded_nearest_distance_capped64',
   'tile_edge_distance_normalized','native_peak_margin_outside32']
METHODS=['crossscale_agreement16','logistic10_oof']

def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

def features(c,native,degraded,tiles):
    x,y=c['x_px'],c['y_px']
    distance=lambda q:math.hypot(q['x_px']-x,q['y_px']-y)
    covering={i for i,(a,b,w,h) in enumerate(tiles) if a<=x<a+w and b<=y<b+h}
    assert c['tile_index'] in covering
    agreement={q['tile_index'] for q in native if q['tile_index'] in covering and distance(q)<=16}
    near=[q for q in degraded if distance(q)<=16]
    closest=min((distance(q) for q in degraded),default=64.)
    competitor=max((q['component_peak'] for q in native if distance(q)>32),default=0.)
    edge=min(c['x'],c['y'],512-c['x']-c['w'],288-c['y']-c['h'])/144
    return np.array([c['component_peak'],c['bbox_mean'],math.log1p(c['component_pixels']),
        c['component_pixels']/c['bbox_area'],math.log(c['w']/c['h']),len(agreement)/len(covering),
        max((q['component_peak'] for q in near),default=0.),min(closest/64,1.),edge,
        c['component_peak']-competitor],dtype=np.float64)

def apply_gate(labels,old,native,scores,method):
    out=[]
    for g in labels:
        sid=g['sample_id'];p=native[sid];b=old[sid];score=scores.get(sid,0.);keep=int(int(p['visible']) and score>=.5)
        if int(b['visible']):source='unchanged_original_temporal';actual=b
        elif keep:source='accepted_actual_native_tile';actual=p
        else:source='missing';actual=dict(visible=0,x_px=0,y_px=0,confidence=0.)
        out.append(dict(**{k:g[k] for k in ['sample_id','candidate_id','local_frame','source_frame','truth_visibility','truth_x_px','truth_y_px']},
            method=method,**{k:actual[k] for k in ['visible','x_px','y_px','confidence']},output_source=source,
            proposal_present=p['visible'],proposal_x_px=p['x_px'],proposal_y_px=p['y_px'],
            proposal_component_peak=p['component_peak'],proposal_gate_score=score,proposal_retained=keep))
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists(),'No post-result model/threshold changes.'
    labels_path=ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv';labels=read(labels_path)
    native_path=BASE/'native_peak_predictions.csv';native_rows=read(native_path);native={r['sample_id']:r for r in native_rows}
    strong_path=ROOT/'ball_temporal_path_comparison/train_displacement_0.1.csv';strong=read(strong_path);old={r['sample_id']:r for r in strong}
    assert len(labels)==123 and len(good_ids(strong))==66 and err(metrics(strong))==4
    paths=[labels_path,native_path,strong_path,BASE/'candidate_pools.json',BASE/'protocol.json',BASE/'verification.json',
        BASE/'native_missing_only_verified_predictions.csv',ROOT/'compare_frozen_ball_backbones.py',
        ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',
        SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv',
        SOURCE/'third_party/RacketVision/source/BallTrack/checkpoints/balltrack_best.pth',
        ROOT.parent/'ball_quality_20261104/completed_review/manual_sequence_labels.csv',
        ROOT.parent/'ball_quality_20261104/frozen_predictions.csv',ROOT.parent/'ball_quality_20261104/evaluation_report.json']
    protected={str(p):sha(p) for p in paths}
    groups=sorted({g['candidate_id'] for g in labels});assert len(groups)==8
    protocol=dict(status='fixed_before_new_features_or_fits',date_jst='2026-11-09',role='Training-only grouped internal check',
        scope='Selected native peak candidates from fixed9tile original-model outputs; no detector rerun or new candidate selection.',
        feature_names=F,features='Only candidate detector geometry/response and native/degraded support. No RGB patch,absolute x/y,time,clipID,GT coordinates or future review features as model input.',
        labels='For selected candidates in human-visible frames, positive iff center distance<=16px,otherwise target-position negative. Negative does NOT mean generic non-ball. Unobservable frames excluded from fit,not converted into absence negatives.',
        split='Leave one entire training clip out; compute normalisation and fit using other7clips only. Predict all proposed candidates in held clip,including unobservable stress cases. No frame random split.',
        head='Reuse exact prior class-balanced logistic BCE +0.5*sum(w^2),unpenalised intercept,CPUfloat64,zero init,LBFGS300,strong_wolfe,threshold0.5.',
        scores='Uncalibrated sigmoid,not a correctness probability.',
        reference='Fixed crossscale agreement: native chosen center has ANY degraded candidate center<=16px. No fitted parameter.',
        application='Only add already selected actual native candidate when original output missing and gate accepts. Preserve all70 original output rows exactly.',
        single_class_fold='Reject all proposals in that held clip if the other7 lack either class; do not invent labels.',
        selection='Both methods compared using pooled held-clip predictions. Require all66original correct retained,errors<=4 and >66correct. Max correct,then fewer errors,tie fixed rule first.',
        full_fit='Only if learned gate is selected by held-clip results,fit one head on all visible labelled candidates and require same safety on training outputs. If it fails,stop without alternate.',
        validation='No new validation/audit/challenge/52frame evaluation in this internal experiment. A surviving proposal needs a separate frozen comparison.',
        limitations='Only8known same-match clips,and few localization-negative examples. OOF is internal development,not external accuracy; comparing rules on OOF is also model selection.',
        no_detector_training=True,no_threshold_grid=True,new_human_labels=0,automatic_adoption=False,
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol);shutil.copy2(Path(__file__),OUT/'executed_source.py')
    pools=load(BASE/'candidate_pools.json');tiles=load(BASE/'protocol.json')['tiles_xywh']
    examples=[];all_features=[]
    for g in labels:
        sid=g['sample_id'];p=native[sid]
        if not int(p['visible']):continue
        c=next(c for c in pools['native'][sid] if c['tile_index']==int(p['tile_index']) and c['component_index']==int(p['component_index']))
        assert (c['x_px'],c['y_px'])==(int(p['x_px']),int(p['y_px']))
        x=features(c,pools['native'][sid],pools['degraded'][sid],tiles);assert np.isfinite(x).all()
        error=math.hypot(c['x_px']-float(g['truth_x_px']),c['y_px']-float(g['truth_y_px'])) if g['truth_visibility']=='visible' else None
        target=int(error<=16) if error is not None else None
        examples.append(dict(sample_id=sid,candidate_id=g['candidate_id'],truth_visibility=g['truth_visibility'],
            fit_label=target if target is not None else '',center_error_px=error if error is not None else '',
            label_scope='within16_of_human_center' if target==1 else 'outside16_of_human_center' if target==0 else 'unlabelled_no_observable_center',
            **dict(zip(F,x.tolist()))))
        all_features.append(x)
    X=np.stack(all_features);Y=np.array([float(e['fit_label']) if e['fit_label']!='' else np.nan for e in examples])
    G=np.array([e['candidate_id'] for e in examples]);labelled=np.isfinite(Y)
    assert int((Y==1).sum())==76 and int((Y==0).sum())==5 and int((~labelled).sum())==7
    write(OUT/'candidate_features_and_training_labels.csv',examples)
    fold_records=[];score_rows=[];oof_scores={};fold_models=[]
    torch.set_num_threads(4)
    for held in groups:
        train=(G!=held)&labelled;test=G==held
        train_ids=[examples[i]['sample_id'] for i in np.flatnonzero(train)]
        test_ids=[examples[i]['sample_id'] for i in np.flatnonzero(test)]
        assert not set(train_ids)&set(test_ids)
        tr_hashes=set();te_hashes=set()
        for g in labels:
            if g['sample_id'] in train_ids:tr_hashes.update(json.loads(g['input_frame_sha256_json']))
            if g['sample_id'] in test_ids:te_hashes.update(json.loads(g['input_frame_sha256_json']))
        assert not tr_hashes&te_hashes
        if set(Y[train])=={0.,1.}:
            model=fit(X[train],Y[train]);scores=predict(model,X[test]);name=OUT/('fold_'+held[-3:]+'.npz');np.savez_compressed(name,**model)
            fold_models.append(str(name));status='fitted';model_hash=sha(name)
        else:scores=np.zeros(int(test.sum()));status='rejected_missing_fit_class';model_hash=''
        for i,s in zip(np.flatnonzero(test),scores,strict=True):
            sid=examples[i]['sample_id'];oof_scores[sid]=float(s)
            score_rows.append(dict(sample_id=sid,held_clip=held,proposal_gate_score=float(s),proposal_retained=int(s>=.5),model_sha256=model_hash))
        fold_records.append(dict(held_clip=held,fit_status=status,fit_samples=int(train.sum()),fit_positive=int((Y[train]==1).sum()),
            fit_negative=int((Y[train]==0).sum()),train_sample_ids=train_ids,test_sample_ids=test_ids,
            test_samples=int(test.sum()),test_input_hash_overlap=0,model_sha256=model_hash))
    assert len(oof_scores)==88
    write(OUT/'oof_candidate_scores.csv',score_rows);dump(OUT/'folds.json',fold_records)
    rule_scores={e['sample_id']:float(e['degraded_peak_within16']>0) for e in examples}
    scores_by={'crossscale_agreement16':rule_scores,'logistic10_oof':oof_scores};summaries=[];eligible=[]
    for method in METHODS:
        pred=apply_gate(labels,old,native,scores_by[method],method);write(OUT/(method+'_predictions.csv'),pred)
        ids=good_ids(pred);m=metrics(pred);lost=sorted(good_ids(strong)-ids)
        assert not lost and all(all(p[k]==old[p['sample_id']][k] for k in ['visible','x_px','y_px','confidence']) for p in pred if int(old[p['sample_id']]['visible']))
        fit_kept=[e for e in examples if scores_by[method][e['sample_id']]>=.5]
        summary=dict(method=method,correct=len(ids),errors=err(m),wrong_position=m['counts']['wrong_localization'],output_when_not_observable=m['counts']['presence_fp'],
            old_correct_lost=len(lost),fit_positive_retained=sum(e['fit_label']==1 for e in fit_kept),fit_negative_retained=sum(e['fit_label']==0 for e in fit_kept),
            unobservable_candidates_retained=sum(e['fit_label']=='' for e in fit_kept),eligible=int(not lost and err(m)<=4 and len(ids)>66))
        summaries.append(summary)
        if summary['eligible']:eligible.append(summary)
    write(OUT/'grouped_comparison.csv',summaries)
    eligible.sort(key=lambda r:(-r['correct'],r['errors'],METHODS.index(r['method'])))
    chosen=eligible[0]['method'] if eligible else None;full_safety=None
    if chosen=='logistic10_oof':
        model=fit(X[labelled],Y[labelled]);np.savez_compressed(OUT/'full_training_model.npz',**model)
        full_scores=dict(zip([e['sample_id'] for e in examples],predict(model,X).tolist()))
        pred=apply_gate(labels,old,native,full_scores,'logistic10_full_fit');write(OUT/'full_fit_predictions.csv',pred)
        full_safety=not (good_ids(strong)-good_ids(pred)) and err(metrics(pred))<=4 and len(good_ids(pred))>66
        if not full_safety:chosen=None
    assert all(sha(Path(p))==h for p,h in protected.items())
    result=dict(status='training_only_grouped_gate_comparison_completed',selected_native_candidates=88,
        fit_candidates=81,fit_positive=76,fit_localization_negative=5,unobservable_not_used_in_fit=7,clip_groups=8,
        strongest_original=dict(correct=66,errors=4),ungated_native_fallback=dict(correct=77,errors=14),
        grouped_results=summaries,chosen_proposal=chosen,full_fit_safety=full_safety,
        no_new_validation_inference=True,no_new_detector_training=True,canonical_tracking_changed=False,
        decision='No passing internal proposal; stop before external roles.' if chosen is None else 'Internal proposal only; freeze separate validation before any use.',
        protected_input_hashes=protected,fold_model_files=fold_models)
    dump(OUT/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['protected_input_hashes','fold_model_files']},indent=2))

if __name__=='__main__':main()
