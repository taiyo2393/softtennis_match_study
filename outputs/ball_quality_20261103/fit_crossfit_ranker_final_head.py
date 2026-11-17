"""One conditional full-fit check,never a fallback chosen after crossfit failure."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from integrate_044_reviewed_tracks import ROOT, read, load, write, sha
from compare_training_tile_detail import TILES
from compare_grouped_tile_gate import F, features
from crossfit_native_candidate_ranker import deduplicate, fit_head, predict_head
from evaluate_training_phase_scope import classify

BASE = ROOT.parent / 'ball_quality_20261109'
OUT = BASE / 'crossfit_native_candidate_ranker'
PREV = BASE / 'native_tile_point_pilot'


def main():
    result = load(OUT / 'result.json')
    assert not (OUT / 'full_fit_decision.json').exists()
    if not result['passes_internal_gate']:
        record = dict(status='not_run_crossfit_failed', head_fit_performed=False, new_confirmation_inference=False,
            operational_adoption=False, reason='Honor predeclared stop;no full-fit alternative after failed held-clip safety.')
        (OUT / 'full_fit_decision.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record, indent=2))
        return
    protocol = load(OUT / 'protocol.json')
    assert all(sha(Path(p)) == h for p, h in protocol['protected_input_hashes'].items())
    labels = read(ROOT / 'balltrack_expanded_centers_pilot/expanded_train_labels.csv')
    phases = {r['sample_id']: r['phase_status'] for r in read(BASE / 'training_phase_scope_review/phase_labels.csv')}
    pools = load(PREV / 'candidate_pools.json')
    degraded = load(BASE / 'training_tile_detail_comparison/candidate_pools.json')['degraded']
    by_gt = {g['sample_id']: g for g in labels}
    rows, candidates = [], {}
    for g in labels:
        sid = g['sample_id']
        candidates[sid] = deduplicate(pools[sid])
        fit = phases[sid] == 'play' and g['truth_visibility'] == 'visible'
        for i, c in enumerate(candidates[sid]):
            x = features(c, pools[sid], degraded[sid], TILES)
            error = np.hypot(c['x_px'] - float(g['truth_x_px']), c['y_px'] - float(g['truth_y_px'])) if fit else None
            rows.append(dict(sample_id=sid, candidate_index=i, fit=int(fit), fit_label=int(error <= 16) if fit else '',
                **dict(zip(F, x.tolist()))))
    fit_rows = [r for r in rows if r['fit']]
    X = np.array([[r[f] for f in F] for r in fit_rows], dtype=np.float64)
    Y = np.array([r['fit_label'] for r in fit_rows], dtype=np.float64)
    assert set(Y) == {0., 1.}
    model, weights, normalization = fit_head(X, Y, [r['sample_id'] for r in fit_rows])
    np.savez_compressed(OUT / 'full_fit_head.npz', **model)
    for r, w, n in zip(fit_rows, weights, normalization, strict=True):
        r.update(loss_weight=float(w), normalization_weight=float(n))
    for r in rows:
        r.setdefault('loss_weight', '')
        r.setdefault('normalization_weight', '')
        r['score'] = float(predict_head(model, np.array([[r[f] for f in F]], dtype=np.float64))[0])
    write(OUT / 'full_fit_candidate_features.csv', rows)
    table = {(r['sample_id'], r['candidate_index']): r for r in rows}
    original = {r['sample_id']: r for r in read(ROOT / 'ball_temporal_path_comparison/train_displacement_0.1.csv')}
    raw = {r['sample_id']: r for r in read(PREV / 'missing_only_predictions.csv')}
    output = []
    for g in labels:
        sid = g['sample_id']
        cs = candidates[sid]
        indexes = sorted(range(len(cs)), key=lambda i: (table[sid, i]['score'], cs[i]['component_peak'], cs[i]['bbox_area'], -cs[i]['tile_index'], -cs[i]['component_index']), reverse=True)
        c = cs[indexes[0]] if indexes and table[sid, indexes[0]]['score'] >= .5 else None
        b = original[sid]
        actual = {k: b[k] for k in ['visible', 'x_px', 'y_px', 'confidence']} if int(b['visible']) else dict(visible=int(c is not None), x_px=c['x_px'] if c else 0, y_px=c['y_px'] if c else 0, confidence=c['bbox_mean'] if c else 0.)
        output.append(dict(**{k: g[k] for k in ['sample_id', 'candidate_id', 'local_frame', 'source_frame', 'truth_visibility', 'truth_x_px', 'truth_y_px']}, **actual))
    write(OUT / 'full_fit_predictions.csv', output)
    counts = Counter(classify(g, g)[0] for g in output if phases[g['sample_id']] == 'play')
    lost = [g['sample_id'] for g in output if phases[g['sample_id']] == 'play' and classify(raw[g['sample_id']], raw[g['sample_id']])[0] == 'correct16px' and classify(g, g)[0] != 'correct16px']
    safe = not lost and counts['correct16px'] == 64 and counts['wrong_position'] == 0 and counts['output_when_not_observable'] <= 1
    record = dict(status='full_fit_training_safety_passed' if safe else 'full_fit_training_safety_failed',
        head_fit_performed=True, passed=safe, play_counts=dict(counts), lost_raw_correct_targets=lost,
        head_sha256=sha(OUT / 'full_fit_head.npz'), detector_sha256=sha(PREV / 'final_epoch03.pth'),
        model_recipe_fixed=True, independent_accuracy=False, new_confirmation_inference=False,
        operational_adoption=False, canonical_tracking_changed=False)
    assert all(sha(Path(p)) == h for p, h in protocol['protected_input_hashes'].items())
    (OUT / 'full_fit_decision.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
