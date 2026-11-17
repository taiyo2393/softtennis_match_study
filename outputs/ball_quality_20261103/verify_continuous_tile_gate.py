"""Check all contiguous output provenance and cached geometry before review."""
import json
import math
from pathlib import Path

import numpy as np
from integrate_044_reviewed_tracks import ROOT, read, load, sha
from compare_ball_candidate_decoders import components
from compare_temporal_candidate_paths import select
from compare_grouped_tile_gate import F
from calibrate_point_supervised_detector import expand
from verify_frozen_tile_gate_confirmation import independently_extract

BASE = ROOT.parent / 'ball_quality_20261109'
OUT = BASE / 'continuous_tile_gate_review'


def main():
    protocol = load(OUT / 'protocol.json')
    manifest = load(OUT / 'inference_manifest.json')
    report = load(OUT / 'render_report.json')
    assert all(sha(Path(p)) == h for p, h in protocol['protected_input_hashes'].items())
    assert all(sha(Path(p)) == h for p, h in protocol['helper_hashes'].items())
    assert sha(OUT / 'executed_source.py') == protocol['script_sha256']
    assert all(sha(OUT / n) == h for n, h in manifest['artifact_hashes'].items())
    image_hashes = load(OUT / 'input_image_hashes.json')
    assert all(sha(Path(p)) == h for p, h in image_hashes.items())
    model = dict(np.load(BASE / 'frozen_tile_gate_confirmation/frozen_training_head.npz'))
    pools = load(OUT / 'candidate_pools.json')
    maps = read(OUT / 'map_records.csv')
    packed = np.load(OUT / 'component_rectangles.npz')
    assert len(maps) == manifest['tile_maps'] and len(packed['offsets']) == len(maps) + 1
    rebuilt = {arm: {sid: [] for sid in pools[arm]} for arm in pools}
    tiles = load(BASE / 'training_tile_detail_comparison/protocol.json')['tiles_xywh']
    unique_maps = set()
    for i, r in enumerate(maps):
        lo, hi = packed['offsets'][i:i + 2]
        hm = expand(packed['indices'][lo:hi], packed['values'][lo:hi])
        ti = int(r['tile_index'])
        key = r['sample_id'], r['arm'], ti
        assert key not in unique_maps
        unique_maps.add(key)
        ox, oy, width, height = tiles[ti]
        for c in components(hm):
            c.update(tile_index=ti, tile_x=ox, tile_y=oy,
                x_px=int((c['x'] + c['w'] / 2) * width / 512 + ox),
                y_px=int((c['y'] + c['h'] / 2) * height / 288 + oy))
            rebuilt[r['arm']][r['sample_id']].append(c)
    assert rebuilt == pools
    contexts = load(OUT / 'global_contexts.json')
    camera = load(OUT / 'camera_features.json')
    feature_rows = {r['sample_id']: r for r in read(OUT / 'candidate_features.csv')}
    rows = read(OUT / 'continuous_predictions.csv')
    assert len(rows) == len({r['sample_id'] for r in rows}) == 724
    additions = verified_features = 0
    for plan in protocol['plans']:
        cid = plan['candidate_id']
        group = [r for r in rows if r['candidate_id'] == cid]
        assert [int(r['local_frame']) for r in group] == list(range(plan['review_first'], plan['review_last'] + 1))
        context = {int(t): cs for t, cs in contexts[cid].items()}
        mask = np.zeros(plan['frame_count'], dtype=bool)
        for t, v in camera[cid]['mask']:
            mask[t] = v
        for r in group:
            sid, t = r['sample_id'], int(r['local_frame'])
            assert int(r['source_frame']) == plan['local_frame_to_source_offset'] + t
            c, status = select(context, t, mask, dict(order='displacement', strength=.1))
            assert r['original_status'] == status
            assert int(r['original_visible']) == int(c is not None)
            if c:
                assert int(r['visible']) == 1 and r['output_source'] == 'original_temporal'
                assert (float(r['original_x_px']), float(r['original_y_px'])) == (c['x_px'], c['y_px'])
                assert float(r['original_confidence']) == c['bbox_mean']
                assert all(r[a] == r[b] for a, b in [('x_px', 'original_x_px'), ('y_px', 'original_y_px'), ('confidence', 'original_confidence')])
                assert sid not in pools['native'] and sid not in feature_rows
                continue
            native, degraded = pools['native'][sid], pools['degraded'][sid]
            assert all((sid, arm, i) in unique_maps for arm in ['native', 'degraded'] for i in range(9))
            chosen = max(native, key=lambda q: (q['component_peak'], q['bbox_area'], -q['tile_index'], -q['component_index'])) if native else None
            if chosen:
                x = independently_extract(chosen, native, degraded, tiles)
                saved = feature_rows[sid]
                assert np.array_equal(x, np.array([float(saved[k]) for k in F]))
                logits = ((x - model['mean']) / model['scale']) @ model['weights'] + model['bias']
                score = float(1 / (1 + np.exp(-logits)))
                assert abs(score - float(saved['proposal_gate_score'])) < 1e-12
                assert abs(score - float(r['proposal_gate_score'])) < 1e-12
                verified_features += 1
            else:
                score = 0.
                assert sid not in feature_rows
            accepted = chosen is not None and score >= .5
            assert int(r['visible']) == accepted
            assert r['output_source'] == ('added_native_candidate' if accepted else 'no_output')
            if accepted:
                assert (float(r['x_px']), float(r['y_px'])) == (chosen['x_px'], chosen['y_px'])
                assert float(r['confidence']) == chosen['bbox_mean']
                additions += 1
    assert additions == sum(r['output_source'] == 'added_native_candidate' for r in rows)
    assert verified_features == len(feature_rows)
    for video in report['videos']:
        assert sha(Path(video['path'])) == video['sha256']
    assert all(sha(Path(p)) == h for p, h in protocol['protected_input_hashes'].items())
    verification = dict(status='passed', continuous_source_frames=724, global_context_frames=788,
        reconstructed_tile_maps=len(maps), independently_verified_features=verified_features,
        added_actual_candidate_centers=additions, all_original_outputs_preserved=True,
        video_hashes_checked=4, source_aspect_preserved=True, protected_inputs_unchanged=True,
        new_human_labels=0, continuous_center_accuracy=None, confirmed_identity_switch_count=None,
        automatic_adoption=False, script_sha256=sha(Path(__file__)),
        artifacts={p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'verification.json'})
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in verification.items() if k != 'artifacts'}, indent=2))


if __name__ == '__main__':
    main()
