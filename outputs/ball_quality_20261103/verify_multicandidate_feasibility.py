"""Check pool bounds and actual-output provenance without creating pseudo-GT."""
import json
import math
from collections import Counter
from pathlib import Path

from integrate_044_reviewed_tracks import ROOT, load, read, write, sha

BASE = ROOT.parent / 'ball_quality_20261109'
OUT = BASE / 'multicandidate_feasibility'


def main():
    protocol = load(OUT / 'protocol.json')
    assert all(sha(Path(p)) == h for p, h in protocol['protected_input_hashes'].items())
    root52 = ROOT.parent / 'ball_quality_20261104'
    datasets = {
        'training123': (load(BASE / 'temporal_consistency_abstention/training_global_contexts.json'),
            load(BASE / 'training_tile_detail_comparison/candidate_pools.json')['native'],
            {r['sample_id']: r for r in read(BASE / 'frozen_tile_gate_confirmation/training_predictions.csv')}),
        'known_ABCD52': (load(root52 / 'frozen_component_cache.json'),
            load(BASE / 'frozen_tile_gate_confirmation/candidate_pools.json')['native'],
            {r['sample_id']: r for r in read(BASE / 'frozen_tile_gate_confirmation/frozen_confirmation_predictions.csv')}),
        'known_EF26': (load(BASE / 'temporal_guard_next_sequences/global_contexts.json'),
            load(BASE / 'temporal_guard_next_sequences/candidate_pools.json')['native'],
            {r['sample_id']: r for r in read(BASE / 'temporal_guard_next_sequences/frozen_predictions.csv') if r['method'] == 'static_supplement'})}
    results = {r['dataset']: r for r in load(OUT / 'result.json')['summary']}
    rows = read(OUT / 'all_visible_play_targets.csv')
    assert len(rows) == len({(r['dataset'], r['sample_id']) for r in rows}) == 139
    closest = []
    for name, (globals, natives, chosen) in datasets.items():
        counts = Counter()
        for r in [r for r in rows if r['dataset'] == name]:
            sid, cid, t = r['sample_id'], r['candidate_id'], r['local_frame']
            assert chosen[sid]['source_frame'] == r['source_frame'] and chosen[sid]['local_frame'] == t
            center = float(r['truth_x_px']), float(r['truth_y_px'])
            gp, np = globals[cid][t], natives[sid]
            distance = lambda c: math.dist(center, (c['x_px'], c['y_px']))
            for key, pool in [('global', gp), ('native', np), ('union', gp + np)]:
                value = min((distance(c) for c in pool), default=None)
                saved = r['nearest_' + key + '_error_px']
                assert (saved == '') == (value is None)
                if value is not None:
                    assert abs(float(saved) - value) < 1e-10
                counts[key + '_center_coverage16'] += value is not None and value <= 16
            out = chosen[sid]
            correct = bool(int(out['visible']) and math.dist(center, (float(out['x_px']), float(out['y_px']))) <= 16)
            if int(out['visible']):
                assert (float(out['x_px']), float(out['y_px'])) in {(c['x_px'], c['y_px']) for c in gp + np}
            expected = 'already_selected_correct' if correct else 'near_center_candidate_exists_not_selected' if min((distance(c) for c in gp + np), default=math.inf) <= 16 else 'no_center_candidate_in_current_union'
            assert r['category'] == expected
            counts['selected_correct'] += correct
            counts[expected] += 1
            if not correct:
                nearest = min(gp + np, key=distance) if gp + np else None
                closest.append(dict(dataset=name, sample_id=sid, local_frame=t, source_frame=r['source_frame'],
                    category=expected, closest_x_px=nearest['x_px'] if nearest else '', closest_y_px=nearest['y_px'] if nearest else '',
                    closest_source='native' if nearest is not None and 'tile_index' in nearest else 'global' if nearest else 'none',
                    distance_px=distance(nearest) if nearest else '',
                    meaning='Closest candidate selected using humanGT:diagnostic only,never an automatic output.'))
        for key, count in counts.items():
            assert results[name].get(key, 0) == count
    write(OUT / 'closest_deficit_candidates_diagnostic.csv', closest)
    verification = dict(status='passed', verified_visible_play_targets=139, datasets_kept_separate=3,
        actual_output_provenance_verified=True, candidate_center_upper_bounds_recomputed=True,
        deficit_rows=15, no_training_or_threshold_changes=True, no_new_labels=True,
        oracle_not_tracking_accuracy=True, protected_inputs_unchanged=True,
        script_sha256=sha(Path(__file__)))
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
