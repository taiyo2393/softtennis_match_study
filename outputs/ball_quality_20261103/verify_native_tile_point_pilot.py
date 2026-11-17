"""Verify supervised crop targets, frozen weights, decoded outputs and changes."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT, read, load, sha, write
from compare_ball_candidate_decoders import components
from compare_training_tile_detail import TILES
from calibrate_point_supervised_detector import expand

BASE = ROOT.parent / 'ball_quality_20261109'
OUT = BASE / 'native_tile_point_pilot'


def main():
    protocol = load(OUT / 'protocol.json')
    result = load(OUT / 'result.json')
    assert all(sha(Path(p)) == h for p, h in protocol['protected_input_hashes'].items())
    assert all(sha(Path(p)) == h for p, h in protocol['helper_hashes'].items())
    assert sha(OUT / 'executed_source.py') == protocol['script_sha256']
    assert sha(Path(result['checkpoint'])) == result['checkpoint_sha256']
    original_path = next(Path(p) for p in protocol['protected_input_hashes'] if p.endswith('checkpoints\\balltrack_best.pth'))
    original = torch.load(original_path, map_location='cpu', weights_only=False)
    original = original.get('state_dict', original)
    original = {(k[7:] if k.startswith('module.') else k): v for k, v in original.items()}
    final = torch.load(result['checkpoint'], map_location='cpu', weights_only=False)['state_dict']
    trainable = set(load(OUT / 'trainable_parameters.json')['names'])
    assert set(original) == set(final)
    frozen = set(original) - trainable
    assert all(torch.equal(original[n], final[n]) for n in frozen)
    changed = [n for n in trainable if not torch.equal(original[n], final[n])]
    assert changed and len(trainable) == 8
    gt = {g['sample_id']: g for g in read(ROOT / 'balltrack_expanded_centers_pilot/expanded_train_labels.csv')}
    phases = {g['sample_id']: g['phase_status'] for g in read(BASE / 'training_phase_scope_review/phase_labels.csv')}
    manifest = read(OUT / 'training_crop_manifest.csv')
    assert len(manifest) == 612 and len({r['sample_id'] for r in manifest}) == 68
    yy, xx = np.ogrid[:288, :512]
    for r in manifest:
        g = gt[r['sample_id']]
        assert phases[r['sample_id']] == 'play' and g['truth_visibility'] == 'visible'
        ox, oy, w, h = TILES[int(r['tile_index'])]
        inside = ox <= float(g['truth_x_px']) < ox + w and oy <= float(g['truth_y_px']) < oy + h
        assert (r['target_center_in_crop'] == 'True') == inside
        if inside:
            x, y = int((float(g['truth_x_px']) - ox) * 512 / w), int((float(g['truth_y_px']) - oy) * 288 / h)
            target = np.exp(-.5 * ((xx - x) ** 2 + (yy - y) ** 2) / 3.5 ** 2).astype(np.float32)[None]
            assert (str(x), str(y)) == (r['gaussian_center_x'], r['gaussian_center_y'])
        else:
            target = np.zeros((1, 288, 512), np.float32)
        assert hashlib.sha256(target.tobytes()).hexdigest() == r['target_sha256']
    history = load(OUT / 'training_history.json')
    assert [r['updates'] for r in history] == [306, 306, 306]
    assert sum(r['updates'] for r in history) == result['updates'] == 918
    maps = read(OUT / 'map_records.csv')
    packed = np.load(OUT / 'component_rectangles.npz')
    pools = load(OUT / 'candidate_pools.json')
    rebuilt = {sid: [] for sid in gt}
    assert len(maps) == 1107 and len(packed['offsets']) == 1108
    for i, r in enumerate(maps):
        lo, hi = packed['offsets'][i:i + 2]
        ti = int(r['tile_index'])
        ox, oy, w, h = TILES[ti]
        for c in components(expand(packed['indices'][lo:hi], packed['values'][lo:hi])):
            c.update(tile_index=ti, tile_x=ox, tile_y=oy, x_px=int((c['x'] + c['w'] / 2) * w / 512 + ox), y_px=int((c['y'] + c['h'] / 2) * h / 288 + oy))
            rebuilt[r['sample_id']].append(c)
    assert pools == rebuilt
    raw = {r['sample_id']: r for r in read(OUT / 'raw_peak_predictions.csv')}
    fallback = {r['sample_id']: r for r in read(OUT / 'missing_only_predictions.csv')}
    baseline = {r['sample_id']: r for r in read(ROOT / 'ball_temporal_path_comparison/train_displacement_0.1.csv')}
    prior = {r['sample_id']: r for r in read(BASE / 'frozen_tile_gate_confirmation/training_predictions.csv')}
    old_pools = load(BASE / 'training_tile_detail_comparison/candidate_pools.json')['native']
    effects = []
    summaries = {(r['method'], r['phase_status']): r for r in read(OUT / 'training_summary.csv')}
    def kind(g, p):
        if g['truth_visibility'] != 'visible':
            return 'output_when_not_observable' if int(p['visible']) else 'no_output_when_not_observable'
        if not int(p['visible']):
            return 'missing_visible_ball'
        return 'correct16px' if math.dist((float(p['x_px']), float(p['y_px'])), (float(g['truth_x_px']), float(g['truth_y_px']))) <= 16 else 'wrong_position'
    for sid, g in gt.items():
        candidates = pools[sid]
        winner = max(candidates, key=lambda c: (c['component_peak'], c['bbox_area'], -c['tile_index'], -c['component_index'])) if candidates else None
        assert bool(int(raw[sid]['visible'])) == (winner is not None)
        if winner:
            assert tuple(float(raw[sid][k]) for k in ['x_px', 'y_px', 'confidence']) == (winner['x_px'], winner['y_px'], winner['bbox_mean'])
        if int(baseline[sid]['visible']):
            assert all(fallback[sid][k] == baseline[sid][k] for k in ['visible', 'x_px', 'y_px', 'confidence'])
        else:
            assert all(fallback[sid][k] == raw[sid][k] for k in ['visible', 'x_px', 'y_px', 'confidence'])
        if phases[sid] == 'play' and kind(g, prior[sid]) != kind(g, fallback[sid]):
            error = math.dist((float(fallback[sid]['x_px']), float(fallback[sid]['y_px'])), (float(g['truth_x_px']), float(g['truth_y_px']))) if g['truth_visibility'] == 'visible' and int(fallback[sid]['visible']) else ''
            close = lambda pool: any(math.dist((c['x_px'], c['y_px']), (float(g['truth_x_px']), float(g['truth_y_px']))) <= 16 for c in pool) if g['truth_visibility'] == 'visible' else ''
            effects.append(dict(sample_id=sid, candidate_id=g['candidate_id'], local_frame=g['local_frame'], source_frame=g['source_frame'],
                before=kind(g, prior[sid]), after=kind(g, fallback[sid]), original_native_has_center16=close(old_pools[sid]),
                final_native_has_center16=close(pools[sid]), selected_error_px=error))
    for name, table in [('original_temporal', baseline), ('previous_static_head', prior), ('final_native_raw_peak', raw), ('final_native_missing_only', fallback)]:
        for phase in ['play', 'nonplay']:
            counts = Counter(kind(gt[sid], p) for sid, p in table.items() if phases[sid] == phase)
            for key in ['correct16px', 'wrong_position', 'missing_visible_ball', 'output_when_not_observable', 'no_output_when_not_observable']:
                assert int(summaries[name, phase][key]) == counts[key]
    write(OUT / 'verified_play_changes.csv', effects)
    change_counts = Counter(r['before'] + '->' + r['after'] for r in effects)
    assert not result['passes_training_gate'] and result['new_validation_inference'] is False
    assert all(sha(Path(p)) == h for p, h in load(OUT / 'input_image_hashes.json').items())
    assert all(sha(Path(p)) == h for p, h in protocol['protected_input_hashes'].items())
    verification = dict(status='passed', human_training_frames=68, verified_crop_targets=612, positive_target_crops=258,
        zero_target_center_crops=354, trainable_tensor_count=8, changed_trainable_tensor_count=len(changed),
        all_frozen_tensors_and_BN_buffers_exact=True, optimizer_updates=918, reconstructed_native_maps=1107,
        raw_and_missing_only_outputs_verified=246, phase_scoped_metrics_recomputed=True,
        play_change_counts=dict(change_counts),
        new_center_candidate_correct_additions=sum(r['after'] == 'correct16px' and r['original_native_has_center16'] is False for r in effects),
        original_candidate_now_selected_correct=sum(r['after'] == 'correct16px' and r['original_native_has_center16'] is True for r in effects),
        source_weights_labels_and_canonical_outputs_unchanged=True, validation_not_run=True,
        operational_adoption=False, script_sha256=sha(Path(__file__)),
        artifacts={p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'verification.json'})
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in verification.items() if k != 'artifacts'}, indent=2))


if __name__ == '__main__':
    main()
