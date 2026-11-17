"""Inventory clip clocks and reserve prediction-blind, temporally separate sequences."""
import json, sys
from pathlib import Path
from collections import Counter
from integrate_044_reviewed_tracks import ROOT, SOURCE, load, read, write, sha

OUT = ROOT.parent / 'ball_quality_20261104'
OUT.mkdir(exist_ok=True)
BASE = SOURCE / 'data/balltrack/racketvision/match1'
assert not (OUT / 'inventory_manifest.json').exists(), 'Preserve completed inventory.'
manifest_path = ROOT / 'balltrack_manual_finetune_pilot/data_manifest.json'
manifest = load(manifest_path)
queue_path = SOURCE / 'data/segments/match1_motion_prediction_expand_01_queue.csv'
play_path = SOURCE / 'data/segments/match1_reviewed_play_windows.csv'
queue = {r['candidate_id']: r for r in read(queue_path)}
plays = {r['play_window_id']: r for r in read(play_path)}
protected = {str(p): sha(p) for p in [manifest_path, queue_path, play_path]}
rows = []
for folder in sorted(BASE.iterdir()):
    if not folder.is_dir():
        continue
    path = folder / 'runs/threshold030_median120/run_metadata.json'
    if not path.exists():
        path = folder / 'run_metadata.json'
    if not path.exists():
        continue
    meta = load(path)
    protected[str(path)] = sha(path)
    start = float(meta['clip_start_timestamp_sec'])
    fps, n = float(meta['fps']), int(meta['frame_count'])
    first = round(start * fps)
    pred = path.parent / 'balltrack_predictions.csv'
    if pred.exists():
        source_rows = read(pred)
        assert len(source_rows) == n
        assert all(int(r['local_frame']) == i and int(r['source_frame']) == first + i
                   for i, r in enumerate(source_rows))
        protected[str(pred)] = sha(pred)
    role = 'prior_nonpilot_use_unknown'
    if folder.name.startswith('match1_motion_pilot_'):
        number = int(folder.name[-3:])
        role = next(k for k in ['train', 'validation', 'audit'] if number in manifest[k + '_clips'])
    q = queue.get(folder.name, {})
    p = plays.get(q.get('play_window_id'), {})
    rows.append(dict(candidate_id=folder.name, prior_role=role, clip_start_sec=start,
                     clip_end_exclusive_sec=start + n / fps, first_source_frame=first,
                     last_source_frame=first + n - 1, frame_count=n, fps=fps,
                     clock='legacy_edited_match1', play_window_id=q.get('play_window_id', ''),
                     start_establishment_method=p.get('start_establishment_method', ''),
                     serve_context=p.get('serve_context', ''), metadata_path=str(path),
                     frame_dir=str(folder / 'frames'), median_path=str(path.parent / 'median.npz'),
                     cached_jpg_count=len(list((folder / 'frames').glob('*.jpg'))),
                     previous_generation_or_inspection_known=True,
                     new_independent_external_test=False))
old = [r for r in rows if r['prior_role'] in ['train', 'validation', 'audit']]
assert len(old) == 20
eligible = []
for r in rows:
    overlaps = [x for x in old if r['clip_start_sec'] <= x['clip_end_exclusive_sec'] + 2
                and r['clip_end_exclusive_sec'] >= x['clip_start_sec'] - 2]
    r['old_pilot_overlaps_with_2sec_guard'] = ';'.join(x['candidate_id'] for x in overlaps)
    r['separate_from_old20_with_2sec_guard'] = int(not overlaps)
    ready = (r['candidate_id'] in queue and not overlaps and r['frame_count'] >= 120
             and r['cached_jpg_count'] == r['frame_count'] and Path(r['median_path']).exists())
    r['eligible_for_next_unlabelled_sequence'] = int(ready)
    if ready:
        eligible.append(r)
eligible.sort(key=lambda r: (r['clip_start_sec'], r['candidate_id']))
assert len(eligible) >= 4
# Fixed stratified chronological quantiles; no model scores, outputs or images used.
indexes = [int((j + .5) * len(eligible) / 4) for j in range(4)]
selected = [eligible[i] for i in indexes]
assert len({r['play_window_id'] for r in selected}) == 4
windows = []
for i, r in enumerate(selected):
    t = r['frame_count'] // 2
    windows.append(dict(window_id=chr(65 + i), **r, center_local_frame=t,
                        first_target_local_frame=t - 12, last_target_local_frame=t + 12,
                        target_stride=2, target_count=13,
                        local_frame_to_source_offset=r['first_source_frame'],
                        phase_status='unreviewed', ball_centers_status='unreviewed',
                        role='new_same_match_development_sequence_validation_never_training'))
write(OUT / 'clip_exposure_inventory.csv', rows)
(OUT / 'reserved_windows.json').write_text(json.dumps(windows, indent=2), encoding='utf-8')
summary = dict(status='four_unlabelled_sequences_reserved_before_predictions',
               inventory_clips=len(rows), old20_clip_roles=dict(Counter(r['prior_role'] for r in old)),
               nonoverlap_expand_candidates=len(eligible), eligible_chronological_indexes=indexes,
               reserved_windows=[dict(window_id=r['window_id'], candidate_id=r['candidate_id'],
                                      play_window_id=r['play_window_id'],
                                      center_local_frame=r['center_local_frame']) for r in windows],
               selection_rule='Four equal chronological strata midpoint indexes floor((j+.5)*N/4); clip midpoint +/-12frames step2. No image or prediction based selection.',
               overlap_rule='Exclude entire expanded clip if it touches any full old pilot20 clip interval plus2sec guard.',
               limitations='Same match. Existing expanded clips were generated before; earlier inspection/threshold exposure is not exhaustively known. NOT untouched independent data or confirmed in-play intervals.',
               manual_labels_added=0, training=False, operational_changes=False,
               inventory_sha256=sha(OUT / 'clip_exposure_inventory.csv'),
               reserved_windows_sha256=sha(OUT / 'reserved_windows.json'),
               protected_input_hashes=protected, script_sha256=sha(Path(__file__)))
assert all(sha(Path(p)) == h for p, h in protected.items())
(OUT / 'inventory_manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k != 'protected_input_hashes'}, indent=2))
