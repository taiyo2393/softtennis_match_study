"""Verify all compact maps, geometric mapping, metrics and preserved outputs."""
import json,math
from pathlib import Path
from collections import Counter
import numpy as np
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha
from compare_ball_candidate_decoders import components
from run_manual_ball_finetune import metrics
from run_hard_case_ball_finetune import good_ids,err

OUT=ROOT.parent/'ball_quality_20261109/training_tile_detail_comparison'
BASE=ROOT/'ball_temporal_path_comparison'
protocol=load(OUT/'protocol.json');result=load(OUT/'result.json')
assert sha(OUT/'executed_source.py')==protocol['script_sha256']
rows=read(ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv')
labels={r['sample_id']:r for r in rows};strong=read(BASE/'train_displacement_0.1.csv');old={r['sample_id']:r for r in strong}
cache=load(OUT/'candidate_pools.json');records=read(OUT/'map_records.csv')
with np.load(OUT/'component_rectangles.npz') as packed:
    indexes=packed['indices'];values=packed['values'];offsets=packed['offsets']
assert len(records)==2214 and len(offsets)==2215 and offsets[0]==0 and offsets[-1]==len(indexes)==len(values)
reconstructed={arm:{sid:[] for sid in labels} for arm in ['native','degraded']};seen=set()
for i,r in enumerate(records):
    assert int(r['map_index'])==i
    key=(r['arm'],r['sample_id'],int(r['tile_index']));assert key not in seen;seen.add(key)
    assert r['sample_id'] in labels and int(r['local_frame'])==int(labels[r['sample_id']]['local_frame'])
    a,b=offsets[i:i+2];hm=np.zeros((288,512),np.float32);hm.ravel()[indexes[a:b]]=values[a:b].astype(np.float32)
    cs=components(hm);assert len(cs)==int(r['source_component_count'])
    tile=int(r['tile_index']);x0,y0,w,h=protocol['tiles_xywh'][tile]
    for c in cs:
        x=int((c['x']+c['w']/2)*w/512+x0);y=int((c['y']+c['h']/2)*h/288+y0)
        assert 0<=x<1920 and 0<=y<1080
        reconstructed[r['arm']][r['sample_id']].append(dict(c,tile_index=tile,tile_x=x0,tile_y=y0,x_px=x,y_px=y))
assert reconstructed==cache
assert len(seen)==123*2*9
global_cache=load(BASE/'train_component_cache.json')
for r in read(OUT/'global_replay.csv'):
    assert r['original_components_match']=='True'
    assert int(r['local_frame']) in json.loads(r['batch_frames_json'])
assert len(read(OUT/'global_replay.csv'))==492
checks=[];metadata_notes=[]
for summary in result['summary']:
    method=summary['method'];pred=read(OUT/(method+'_predictions.csv'));assert len(pred)==123
    assert {r['sample_id'] for r in pred}==set(labels)
    assert len(good_ids(pred))==summary['correct'] and err(metrics(pred))==summary['errors']
    assert len(good_ids(strong)-good_ids(pred))==summary['old_correct_lost']
    if method.endswith('_missing_only'):
        arm=method.split('_')[0];raw={r['sample_id']:r for r in read(OUT/(arm+'_peak_predictions.csv'))}
        additions=Counter();corrected=[];preserved=0
        for p in pred:
            sid=p['sample_id'];prior=old[sid];g=labels[sid];q=dict(p)
            # The original inference CSV inherited component_peak from the tile
            # proposal even when retaining an old point. Keep it unchanged and
            # publish a separate table with the actual source peak clarified.
            q['tile_proposal_component_peak']=p['component_peak']
            if int(prior['visible']):
                assert all(p[k]==prior[k] for k in ['visible','x_px','y_px','confidence'])
                source=next(c for c in global_cache[g['candidate_id']][g['local_frame']] if c['component_index']==int(prior['selected_component_index']))
                q['component_peak']=source['component_peak'];preserved+=1
                assert p['output_source']=='unchanged_original_temporal'
            else:
                assert all(float(p[k])==float(raw[sid][k]) for k in ['visible','x_px','y_px','confidence'])
                if int(p['visible']):
                    if g['truth_visibility']!='visible':additions['output_when_not_observable']+=1
                    elif math.hypot(float(p['x_px'])-float(g['truth_x_px']),float(p['y_px'])-float(g['truth_y_px']))<=16:additions['correct']+=1
                    else:additions['wrong_position']+=1
            corrected.append(q)
        assert preserved==70
        assert len(good_ids(corrected))==summary['correct'] and err(metrics(corrected))==summary['errors']
        corrected_path=OUT/(method+'_verified_predictions.csv');write(corrected_path,corrected)
        metadata_notes.append(dict(method=method,original_csv_retained=True,source_peak_clarified_in=str(corrected_path),
            change='Only provenance of component_peak for70preserved points; proposal peak kept in a separate column. All output coordinates/presence/confidence and metrics unchanged.'))
        checks.append(dict(method=method,original_outputs_preserved=preserved,new_correct=additions['correct'],new_wrong_position=additions['wrong_position'],new_output_when_not_observable=additions['output_when_not_observable']))
    elif method in ['native_peak','degraded_peak','global_peak']:
        for p in pred:
            g=labels[p['sample_id']]
            pool=cache[method.split('_')[0]][p['sample_id']] if method!='global_peak' else global_cache[g['candidate_id']][g['local_frame']]
            chosen=max(pool,key=lambda c:(c['component_peak'],c['bbox_area'],-c.get('tile_index',-1),-c['component_index'])) if pool else None
            assert int(p['visible'])==int(chosen is not None)
            if chosen:assert all(float(p[k])==chosen[k] for k in ['x_px','y_px']) and float(p['confidence'])==chosen['bbox_mean']
assert result['chosen_training_proposal'] is None and not result['new_validation_inference']
assert all(sha(Path(p))==h for p,h in result['protected_input_hashes'].items())
assert all(sha(Path(p))==h for p,h in protocol['helper_hashes'].items())
images=load(OUT/'input_image_hashes.json');assert all(sha(Path(p))==h for p,h in images.items())
verification=dict(status='passed',tile_maps_reconstructed=2214,original_contexts_reproduced=492,rows=123,
    input_images_unchanged=len(images),fallback_checks=checks,metadata_clarifications=metadata_notes,
    subthreshold_outside_rectangles_not_reconstructable=True,
    all_canonical_weights_labels_and_outputs_unchanged=True,validation_or_new52_data_used_for_selection=False,
    no_new_training=True,no_parameter_changes_after_results=True,
    artifacts={name:sha(OUT/name) for name in ['protocol.json','executed_source.py','result.json','training_summary.csv','candidate_pools.json',
        'component_rectangles.npz','map_records.csv','native_missing_only_verified_predictions.csv','degraded_missing_only_verified_predictions.csv']},
    script_sha256=sha(Path(__file__)))
(OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in verification.items() if k!='artifacts'},indent=2))
