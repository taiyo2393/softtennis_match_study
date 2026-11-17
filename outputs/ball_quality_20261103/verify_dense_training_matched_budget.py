"""Reconstruct held outputs and audit fit exclusions/compute budgets, no fitting."""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT, SOURCE, read, load, write, sha
from compare_training_tile_detail import TILES, mapped, peak_choice
from compare_ball_candidate_decoders import components
from calibrate_point_supervised_detector import expand

BASE = ROOT.parent / 'ball_quality_20261109'
OUT = BASE / 'dense_training_matched_budget'
DATA = BASE / 'dense_training_sequences/completed_review'
PREV = BASE / 'crossfit_native_candidate_ranker'


def outcome(g, p):
    if g['truth_visibility'] != 'visible':
        return 'output_when_not_observable' if int(p['visible']) else 'no_output_when_not_observable'
    if not int(p['visible']): return 'missing_visible_ball'
    error = math.hypot(float(g['truth_x_px'])-float(p['x_px']), float(g['truth_y_px'])-float(p['y_px']))
    return 'correct16px' if error <= 16 else 'wrong_position'


def main():
    protocol, result = load(OUT/'protocol.json'), load(OUT/'result.json')
    assert sha(OUT/'executed_source.py') == protocol['script_sha256']
    assert all(sha(Path(p)) == h for p,h in protocol['protected_input_hashes'].items())
    assert all(sha(Path(p)) == h for p,h in protocol['helper_hashes'].items())
    assert all(sha(Path(p)) == h for p,h in load(OUT/'input_image_hashes.json').items())
    rows = {r['sample_id']: r for r in read(DATA/'merged_training_observation_states.csv')}
    visible = {r['sample_id']: r for r in read(DATA/'training_visible_centers.csv')}
    dense_labels = read(DATA/'manual_sequence_labels.csv')
    dense_ids = {r['sample_id'] for r in dense_labels}
    original = {r['sample_id']: r for r in read(ROOT/'ball_temporal_path_comparison/train_displacement_0.1.csv')}
    parent = {f['held_clip']: f for f in load(PREV/'folds.json')}
    sparse_controls = {r['sample_id']:r for r in read(PREV/'oof_predictions.csv') if r['method']=='fold_native_peak'}
    sys.path.insert(0, str(SOURCE/'third_party/RacketVision/source/BallTrack'))
    from inference import BallInferencer
    engine = BallInferencer(str(SOURCE/'third_party/RacketVision/source/BallTrack/configs/tracknetv3_base.py'),
        str(SOURCE/'third_party/RacketVision/source/BallTrack/checkpoints/balltrack_best.pth'), device='cpu', thre=.3, batchsize=4)
    initial = engine.model.state_dict()
    train_names = {n for n,_ in engine.model.named_parameters() if n.startswith('up_block_3.') or n.startswith('predictor.')}
    folds = load(OUT/'folds.json')
    changed_counts, total_trace = [], 0
    for fold, budget in zip(folds, protocol['budgets'], strict=True):
        cid = fold['held_clip']; folder = OUT/('fold_'+cid[-3:])
        assert cid == budget['held_clip']
        fit = {sid for sid,r in visible.items() if r['candidate_id']!=cid}
        held = {sid for sid,r in rows.items() if r['candidate_id']==cid}
        assert set(fold['dense_fit_sample_ids'])==fit and set(fold['held_sample_ids'])==held and not fit & held
        assert set(fold['sparse_fit_sample_ids'])==set(parent[cid]['detector_fit_sample_ids'])
        fit_hash = {h for sid in fit for h in json.loads(visible[sid]['input_frame_sha256_json'])}
        held_hash = {h for sid in held for h in json.loads(rows[sid]['input_frame_sha256_json'])}
        assert not fit_hash & held_hash
        checkpoint = torch.load(folder/'detector_final_matched.pth', map_location='cpu', weights_only=False)
        assert checkpoint['meta']['held_clip']==cid and set(checkpoint['meta']['detector_fit_sample_ids'])==fit
        assert checkpoint['meta']['protocol_sha256']==sha(OUT/'protocol.json')
        assert sha(folder/'detector_final_matched.pth')==fold['dense_checkpoint_sha256']
        state = checkpoint['state_dict']
        assert all(torch.equal(state[n], initial[n]) for n in initial if n not in train_names)
        changed_counts.append(sum(not torch.equal(state[n],initial[n]) for n in train_names))
        sample_index = read(folder/'training_sample_index.csv')
        expected_index = [(sid,tile) for sid in visible if sid in fit for tile in range(9)]
        assert [(r['sample_id'],int(r['tile_index'])) for r in sample_index]==expected_index
        assert [int(r['sample_index']) for r in sample_index]==list(range(len(sample_index)))
        sparse_n = parent[cid]['detector_crop_samples']
        sizes = [min(2,sparse_n-j) for _ in range(3) for j in range(0,sparse_n,2)]
        assert sizes==budget['batch_sizes']==fold['batch_sizes']==checkpoint['meta']['batch_sizes']
        assert len(sizes)==fold['update_count']==parent[cid]['detector_updates']
        assert sum(sizes)==fold['presentations']==sparse_n*3
        trace, losses = read(folder/'training_trace.csv'), read(folder/'training_losses.csv')
        assert len(trace)==len(losses)==len(sizes)
        rng=np.random.default_rng(20261109); order=rng.permutation(len(sample_index)); cursor=0; pass_index=0
        for number,(entry,size,loss) in enumerate(zip(trace,sizes,losses,strict=True),1):
            chosen=[]
            for _ in range(size):
                if cursor==len(order): order=rng.permutation(len(sample_index)); cursor=0; pass_index+=1
                chosen.append(int(order[cursor])); cursor+=1
            assert int(entry['update'])==number and int(entry['batch_size'])==size
            assert json.loads(entry['sample_indexes_json'])==chosen and int(entry['data_pass_index'])==pass_index
            assert int(loss['update'])==number and all(math.isfinite(float(loss[k])) for k in ['loss','gradient_norm','grad_scale'])
        total_trace+=len(trace)
    assert all(n==len(train_names) for n in changed_counts) and total_trace==result['training_updates']
    saved_pools=load(OUT/'candidate_pools.json')
    rebuilt={arm:{sid:[] for sid in rows} for arm in ['sparse','dense']}
    maps=read(OUT/'map_records.csv'); packed=np.load(OUT/'component_rectangles.npz')
    assert len(maps)==result['stored_maps']==243*2*9 and len(packed['offsets'])==len(maps)+1
    seen=set()
    for i,m in enumerate(maps):
        assert int(m['map_index'])==i
        key=(m['arm'],m['sample_id'],int(m['tile_index'])); assert key not in seen; seen.add(key)
        lo,hi=map(int,packed['offsets'][i:i+2]); hm=expand(packed['indices'][lo:hi],packed['values'][lo:hi]); cs=components(hm)
        assert len(cs)==int(m['component_count']) and m['held_clip']==rows[m['sample_id']]['candidate_id']
        assert int(rows[m['sample_id']]['local_frame']) in json.loads(m['batch_frames_json'])
        rebuilt[m['arm']][m['sample_id']].extend(mapped(cs,int(m['tile_index'])))
    assert rebuilt==saved_pools
    replayed=0
    for cid in parent:
        old=load(PREV/('fold_'+cid[-3:])/'candidate_pools.json')
        for sid in original:
            if rows[sid]['candidate_id']==cid:
                assert rebuilt['sparse'][sid]==old[sid]; replayed+=1
    predictions=read(OUT/'held_predictions.csv'); assert len(predictions)==result['held_predictions']==732
    by_method={}
    for p in predictions:
        method,sid=p['method'],p['sample_id']; g=rows[sid]; arm=method.split('_')[0]
        assert p['held_clip']==g['candidate_id']
        selected=peak_choice(rebuilt[arm][sid]); presence=int(selected is not None)
        expected=dict(visible=presence,x_px=selected['x_px'] if selected else 0,y_px=selected['y_px'] if selected else 0,confidence=selected['bbox_mean'] if selected else 0.)
        if method.endswith('legacy_missing_only') and int(original[sid]['visible']):
            expected={k:original[sid][k] for k in expected}
        assert all(float(p[k])==float(v) for k,v in expected.items())
        if method=='sparse_legacy_missing_only': assert all(float(p[k])==float(sparse_controls[sid][k]) for k in expected)
        by_method.setdefault(method,{})[sid]=p
    summary_rows=read(OUT/'summary.csv')
    for s in summary_rows:
        scope,method=s['scope'],s['method']; ids=dense_ids if scope=='dense_packet_play' else set(original)
        phase='nonplay' if scope=='legacy_nonplay' else 'play'
        filtered={sid:p for sid,p in by_method[method].items() if sid in ids and rows[sid]['phase_status']==phase}
        counts=Counter(outcome(rows[sid],p) for sid,p in filtered.items())
        assert int(s['targets'])==len(filtered)
        assert all(int(s[k])==counts[k] for k in ['correct16px','wrong_position','missing_visible_ball','output_when_not_observable','no_output_when_not_observable'])
        saved=next(v for v in result['summary'] if v['scope']==scope and v['method']==method)
        assert all(str(saved[k])==str(v) for k,v in s.items())
    paired=read(OUT/'paired_outcomes.csv')
    for c in paired:
        sid=c['sample_id']; suffix='legacy_missing_only' if c['scope']=='legacy_play' else 'raw_native_peak'
        a,b=[outcome(rows[sid],by_method[arm+'_'+suffix][sid]) for arm in ['sparse','dense']]
        assert a==c['sparse_outcome'] and b==c['dense_outcome']
        assert int(c['correct_lost'])==int(a=='correct16px' and b!='correct16px')
    legacy_lost=[c['sample_id'] for c in paired if c['scope']=='legacy_play' and int(c['correct_lost'])]
    dense_lost=[c['sample_id'] for c in paired if c['scope']=='dense_packet_play' and int(c['correct_lost'])]
    def summary(scope,method): return next(s for s in summary_rows if s['scope']==scope and s['method']==method)
    primary=summary('legacy_play','dense_legacy_missing_only')
    secondary_a=summary('dense_packet_play','sparse_raw_native_peak'); secondary_b=summary('dense_packet_play','dense_raw_native_peak')
    safe=(not legacy_lost and int(primary['correct16px'])>=64 and int(primary['wrong_position'])==0 and int(primary['output_when_not_observable'])<=1
        and not dense_lost and int(secondary_b['wrong_position'])<=int(secondary_a['wrong_position']) and int(secondary_b['output_when_not_observable'])<=int(secondary_a['output_when_not_observable']))
    assert legacy_lost==result['legacy_correct_lost'] and dense_lost==result['dense_packet_correct_lost'] and bool(safe)==result['passes_internal_gate']
    oracle=[]
    for arm in ['sparse','dense']:
        for sid,g in rows.items():
            if g['phase_status']!='play' or g['truth_visibility']!='visible': continue
            pool=rebuilt[arm][sid]
            covered=any(math.hypot(float(g['truth_x_px'])-c['x_px'],float(g['truth_y_px'])-c['y_px'])<=16 for c in pool)
            oracle.append(dict(arm=arm,sample_id=sid,candidate_id=g['candidate_id'],candidate_count=len(pool),
                center_candidate_within16px=int(covered),human_center_assisted_upper_bound_not_automatic_accuracy=1))
    write(OUT/'candidate_center_upper_bounds.csv',oracle)
    adjacent=[]
    for wid in 'ABCD':
        group=[r for r in dense_labels if r['window_id']==wid]
        for left,right in zip(group,group[1:]):
            sid1,sid2=left['sample_id'],right['sample_id']
            assert int(right['source_frame'])-int(left['source_frame'])==2
            valid=left['truth_visibility']==right['truth_visibility']=='visible'
            adjacent.append(dict(window_id=wid,left_case=left['case_id'],right_case=right['case_id'],two_human_centers=int(valid),
                sparse_both_correct=int(valid and all(outcome(rows[sid],by_method['sparse_raw_native_peak'][sid])=='correct16px' for sid in [sid1,sid2])),
                dense_both_correct=int(valid and all(outcome(rows[sid],by_method['dense_raw_native_peak'][sid])=='correct16px' for sid in [sid1,sid2]))))
    write(OUT/'adjacent_sampled_pair_outcomes.csv',adjacent)
    assert all(sha(Path(p))==h for p,h in protocol['protected_input_hashes'].items())
    report=dict(status='passed',detectors_checked=8,changed_trainable_tensor_counts=changed_counts,matched_optimizer_updates_verified=total_trace,
        matched_batch_sizes_and_sample_trace_verified=True,held_clip_fit_and_input_hash_exclusion=True,all_frozen_weights_and_bn_exact=True,
        source_heatmap_components_reconstructed=len(maps),legacy_sparse_candidate_replay_targets=replayed,held_outputs_reconstructed=len(predictions),
        summaries_and_paired_outcomes_recomputed=True,dense_sampled_pairs=len(adjacent),pair_observable_count=sum(p['two_human_centers'] for p in adjacent),
        sparse_pair_both_correct=sum(p['sparse_both_correct'] for p in adjacent),dense_pair_both_correct=sum(p['dense_both_correct'] for p in adjacent),
        protected_inputs_unchanged=True,operational_adoption=False,script_sha256=sha(Path(__file__)))
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
