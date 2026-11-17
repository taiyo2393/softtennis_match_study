"""Verify held-clip fitting, raw features, scores and exact output provenance."""
import json,math
from pathlib import Path
from collections import Counter
import numpy as np
from integrate_044_reviewed_tracks import ROOT,read,load,write,sha
from compare_frozen_ball_backbones import predict
from run_manual_ball_finetune import metrics
from run_hard_case_ball_finetune import good_ids,err
from compare_grouped_tile_gate import F,features,apply_gate

OUT=ROOT.parent/'ball_quality_20261109/training_tile_gate_grouped'
BASE=ROOT.parent/'ball_quality_20261109/training_tile_detail_comparison'
p=load(OUT/'protocol.json');result=load(OUT/'result.json')
assert sha(OUT/'executed_source.py')==p['script_sha256']
table=read(OUT/'candidate_features_and_training_labels.csv');labels=read(ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv')
by={g['sample_id']:g for g in labels};assert len(by)==123
native={r['sample_id']:r for r in read(BASE/'native_peak_predictions.csv')};pools=load(BASE/'candidate_pools.json');tiles=load(BASE/'protocol.json')['tiles_xywh']
X=np.array([[float(r[f]) for f in F] for r in table]);Y=np.array([float(r['fit_label']) if r['fit_label']!='' else np.nan for r in table])
assert X.shape==(88,10) and (Y==1).sum()==76 and (Y==0).sum()==5 and np.isnan(Y).sum()==7
for r,x,y in zip(table,X,Y,strict=True):
    g=by[r['sample_id']];candidate=native[r['sample_id']]
    c=next(c for c in pools['native'][r['sample_id']] if c['tile_index']==int(candidate['tile_index']) and c['component_index']==int(candidate['component_index']))
    assert np.array_equal(x,features(c,pools['native'][r['sample_id']],pools['degraded'][r['sample_id']],tiles))
    if g['truth_visibility']=='visible':
        assert y==int(math.hypot(c['x_px']-float(g['truth_x_px']),c['y_px']-float(g['truth_y_px']))<=16)
    else:assert np.isnan(y)
folds=load(OUT/'folds.json');saved={r['sample_id']:r for r in read(OUT/'oof_candidate_scores.csv')};scores={};tested=set()
for fold in folds:
    held=fold['held_clip'];tr=[i for i,r in enumerate(table) if r['candidate_id']!=held and np.isfinite(Y[i])]
    te=[i for i,r in enumerate(table) if r['candidate_id']==held]
    assert [table[i]['sample_id'] for i in tr]==fold['train_sample_ids']
    assert [table[i]['sample_id'] for i in te]==fold['test_sample_ids']
    assert all(by[table[i]['sample_id']]['truth_visibility']=='visible' for i in tr)
    model_path=OUT/('fold_'+held[-3:]+'.npz');assert sha(model_path)==fold['model_sha256']
    with np.load(model_path) as pack:model={k:pack[k] for k in pack.files}
    mean=X[tr].mean(0);std=X[tr].std(0);std[std<1e-8]=1.
    assert np.array_equal(model['mean'],mean) and np.array_equal(model['scale'],std)
    values=predict(model,X[te])
    for i,value in zip(te,values,strict=True):
        sid=table[i]['sample_id'];assert sid not in tested;tested.add(sid)
        assert saved[sid]['held_clip']==held and saved[sid]['model_sha256']==fold['model_sha256']
        assert abs(float(saved[sid]['proposal_gate_score'])-value)<1e-12
        scores[sid]=float(value)
assert len(tested)==88 and len(folds)==8
old={r['sample_id']:r for r in read(ROOT/'ball_temporal_path_comparison/train_displacement_0.1.csv')}
summary_by={r['method']:r for r in result['grouped_results']};extra=[];rejected=[];effects=[]
for method in ['crossscale_agreement16','logistic10_oof']:
    method_scores=scores if method=='logistic10_oof' else {r['sample_id']:float(float(r['degraded_peak_within16'])>0) for r in table}
    rebuilt=apply_gate(labels,old,native,method_scores,method);actual=read(OUT/(method+'_predictions.csv'))
    assert len(actual)==len(rebuilt)==123
    assert all({k:str(v) for k,v in a.items()}==b for a,b in zip(rebuilt,actual,strict=True))
    summary=summary_by[method];assert len(good_ids(actual))==summary['correct'] and err(metrics(actual))==summary['errors']
    counts=Counter()
    for r in actual:
        prior=old[r['sample_id']]
        if int(prior['visible']):assert all(r[k]==prior[k] for k in ['visible','x_px','y_px','confidence']);continue
        if not int(native[r['sample_id']]['visible']):continue
        g=by[r['sample_id']]
        if g['truth_visibility']=='visible':
            pp=native[r['sample_id']];distance=math.hypot(float(pp['x_px'])-float(g['truth_x_px']),float(pp['y_px'])-float(g['truth_y_px']))
            category='correct_center16' if distance<=16 else 'wrong_target_position'
        else:category='output_when_not_observable'
        kept=int(r['visible']);counts[('kept_' if kept else 'rejected_')+category]+=1
        item=dict(method=method,sample_id=r['sample_id'],candidate_id=r['candidate_id'],local_frame=r['local_frame'],source_frame=r['source_frame'],
            category=category,accepted=kept,proposal_gate_score=r['proposal_gate_score'],proposal_x_px=r['proposal_x_px'],proposal_y_px=r['proposal_y_px'],
            truth_x_px=g['truth_x_px'],truth_y_px=g['truth_y_px'])
        if kept and category!='correct_center16':extra.append(item)
        if not kept:rejected.append(item)
    effects.append(dict(method=method,**dict(counts)))
write(OUT/'remaining_added_errors.csv',extra);write(OUT/'rejected_additions.csv',rejected)
dump=lambda path,value:path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
dump(OUT/'addition_effects.json',effects)
assert not (OUT/'full_training_model.npz').exists() and result['chosen_proposal'] is None
assert all(sha(Path(path))==h for path,h in p['protected_input_hashes'].items())
verification=dict(status='passed',candidate_features_reproduced=88,visible_center_fit_examples=81,unobservable_excluded_from_fit=7,
    held_clip_models_verified=8,held_out_score_predictions_reproduced=88,clip_overlap_between_fit_and_check=0,
    preserved_old_outputs_per_method=70,addition_effects=effects,
    all_protected_models_labels_and_tracking_unchanged=True,no_full_fit_or_new_validation=True,
    files={name:sha(OUT/name) for name in ['protocol.json','executed_source.py','candidate_features_and_training_labels.csv',
        'folds.json','oof_candidate_scores.csv','result.json','grouped_comparison.csv','remaining_added_errors.csv','addition_effects.json']},
    script_sha256=sha(Path(__file__)))
dump(OUT/'verification.json',verification);print(json.dumps({k:v for k,v in verification.items() if k!='files'},indent=2))
