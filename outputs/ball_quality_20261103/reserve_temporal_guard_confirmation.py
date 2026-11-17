"""Reserve two further same-match sequences using metadata only, before predictions."""
import json
from pathlib import Path

from integrate_044_reviewed_tracks import ROOT, read, load, write, sha

BASE = ROOT.parent / 'ball_quality_20261109'
STUDY = BASE / 'temporal_consistency_abstention'
OUT = BASE / 'temporal_guard_next_sequences'
CHECK = ROOT.parent / 'ball_quality_20261104'


def main():
    assert load(STUDY / 'verification.json')['status'] == 'passed'
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'reservation.json').exists(), 'Preserve metadata-only selection before inference.'
    inventory = CHECK / 'clip_exposure_inventory.csv'
    old_windows = load(CHECK / 'reserved_windows.json')
    used = {w['candidate_id'] for w in old_windows}
    rows = read(inventory)
    eligible = sorted([r for r in rows if r['eligible_for_next_unlabelled_sequence'] == '1' and r['candidate_id'] not in used],
        key=lambda r: (float(r['clip_start_sec']), r['candidate_id']))
    assert len(eligible) == 16
    indexes = [int((j + .5) * len(eligible) / 2) for j in range(2)]
    selected = [eligible[i] for i in indexes]
    assert len({r['play_window_id'] for r in selected}) == 2
    windows, queries = [], []
    protected = {str(p): sha(p) for p in [inventory, CHECK / 'reserved_windows.json', STUDY / 'protocol.json', STUDY / 'training_decision.json',
        STUDY / 'regression_report.json', STUDY / 'verification.json', ROOT / 'ball_temporal_consistency_rule.py',
        BASE / 'frozen_tile_gate_confirmation/frozen_training_head.npz',
        ROOT / 'reviewed_ball_overrides.json', ROOT / 'reviewed_person_overrides.json', ROOT / 'reviewed_tracking_bundles.json']}
    for wid, r in zip(['E', 'F'], selected, strict=True):
        n = int(r['frame_count'])
        center = n // 2
        first, last = int(r['first_source_frame']), int(r['last_source_frame'])
        for old in old_windows:
            assert last + 120 < old['first_source_frame'] or old['last_source_frame'] + 120 < first
        for old in rows:
            if old['prior_role'] in ['train', 'validation', 'audit']:
                assert last + 120 < int(old['first_source_frame']) or int(old['last_source_frame']) + 120 < first
        assert center - 90 - 16 >= 3 and center + 90 + 16 < n
        meta, median = Path(r['metadata_path']), Path(r['median_path'])
        assert meta.exists() and median.exists()
        protected[str(meta)] = sha(meta)
        protected[str(median)] = sha(median)
        windows.append(dict(r, window_id=wid, center_local_frame=center,
            first_target_local_frame=center - 12, last_target_local_frame=center + 12,
            target_count=13, target_stride=2, continuous_first=center - 90, continuous_last=center + 90,
            phase_status='unreviewed', ball_centers_status='unreviewed', role='prospective_same_match_confirmation_never_training'))
        for number, t in enumerate(range(center - 12, center + 13, 2), 1):
            queries.append(dict(case_id=f'{wid}{number:02d}', sample_id=f'{r["candidate_id"]}_frame_{t:06d}',
                window_id=wid, candidate_id=r['candidate_id'], local_frame=t, source_frame=first + t,
                phase_status='unreviewed', truth_visibility='unreviewed', truth_x_px='', truth_y_px='', review_status='unreviewed'))
    write(OUT / 'reserved_targets.csv', queries)
    write(OUT / 'eligible_metadata.csv', eligible)
    (OUT / 'reserved_windows.json').write_text(json.dumps(windows, indent=2, ensure_ascii=False), encoding='utf-8')
    reservation = dict(status='metadata_only_reservation_before_predictions_and_annotations', date_jst='2026-11-09',
        selection='Remove all4previous reserved clips from20old-pilot-separated eligible clips;sort16remaining chronologically. Select2equal-strata midpoint indexes floor((j+.5)*16/2)=[4,12]. No images or output scores inspected.',
        selected_clips=[w['candidate_id'] for w in windows], window_ids=['E', 'F'], target_count=26,
        target_sampling='Each full-clip midpoint +/-12sourceframes step2. Context +/-90frames for continuous inspection.',
        full_clip_120frame_separation_from_old20_and_prior4=True,
        methods_fixed=['original_temporal', 'static_supplement', 'two_anchor_guard_all_outputs'],
        annotation='Blind human main-rally phase,ball visibility/uncertainty,and center clicks where observable; predictions hidden. No labels assumed from metadata or model.',
        scope_definition='Main-rally valid serve contact through point determination; excludes fault/let attempts. Official in-play begins earlier. Do not infer precise boundaries from momentary phase labels.',
        analysis='All26targets,common play/visibility exclusions,per-window and pooled counts,correct16px,wrong center,output-on-unobservable,missing visible,lost correct,withheld outputs,and sampled-pair differences. No frame-independence significance tests.',
        safety='Guard must lose no static-supplement correct targets and not increase wrong_position/output_when_not_observable; pipeline must retain original correct targets and improve localization without extra errors to show a benefit over original. If guard never acts,no direct new discrimination evidence. Never choose an alternative after failure.',
        decision='Small same-match confirmation only; passing cannot authorize automatic adoption or establish external/full-rally tracking quality.',
        limitations='Clips existed before with prior exposure incompletely known; metadata-only new selection after rule fixed does not make an untouched external test.',
        no_predictions_generated=True, new_human_labels=0, operational_adoption=False,
        protected_input_hashes=protected, reserved_targets_sha256=sha(OUT / 'reserved_targets.csv'),
        reserved_windows_sha256=sha(OUT / 'reserved_windows.json'), script_sha256=sha(Path(__file__)))
    assert all(sha(Path(p)) == h for p, h in protected.items())
    (OUT / 'reservation.json').write_text(json.dumps(reservation, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in reservation.items() if k not in ['protected_input_hashes']}, indent=2))


if __name__ == '__main__':
    main()
