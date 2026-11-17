"""Score the frozen match2 baseline against the completed raw-image references."""
from collections import Counter
from itertools import groupby
import shutil
from statistics import median
import sys

import build_044_spatiotemporal_pilot as common

sys.path.insert(0, str(common.OLD))
from evaluate_reserved_ball_sequences import classify


BASELINE = common.HERE / 'match2_ball_first_point'
REVIEW = common.HERE / 'match2_first_point_ball_review'
OUT = common.HERE / 'match2_first_point_ball_evaluated'


def main():
    protocol = common.load(BASELINE / 'protocol.json')
    assert protocol['frozen_before_ball_outputs']
    protected = common.load(BASELINE / 'publication_hashes.json')
    for name, digest in protected.items():
        assert common.sha(BASELINE / name) == digest, name
    label_path = REVIEW / 'manual_ball_references.csv'
    labels = common.read(label_path)
    assert [int(row['source_frame']) for row in labels] == protocol['reference_frames_selected_before_inference']
    assert all(row['reviewer_id'] == 'human_user' and row['annotated_at_utc'] and
               row['truth_visibility'] in {'visible', 'not_visible', 'uncertain'} for row in labels)
    for row in labels:
        if row['truth_visibility'] == 'visible':
            assert 0 <= int(row['truth_x_px']) < 1920 and 0 <= int(row['truth_y_px']) < 1080
        else:
            assert row['truth_x_px'] == row['truth_y_px'] == ''

    predictions = common.read(BASELINE / 'baseline_unreviewed_candidates.csv')
    core_start, core_end = protocol['core_inclusive']
    assert [int(row['source_frame']) for row in predictions] == list(range(core_start, core_end + 1))
    by_frame = {row['source_frame']: row for row in predictions}
    comparisons = []
    errors = []
    for row in labels:
        pred = by_frame[row['source_frame']]
        # The previously defined interior of the point supplies the evaluation phase.
        # No phase decisions are added to the human reference file.
        status, error = classify(dict(row, phase_status='play'), dict(pred, visible=pred['candidate_present']))
        if error is not None:
            errors.append(error)
        comparisons.append(dict(
            source_frame=row['source_frame'], truth_visibility=row['truth_visibility'],
            truth_x_px=row['truth_x_px'], truth_y_px=row['truth_y_px'],
            candidate_present=pred['candidate_present'], candidate_x_px=pred['x_px'],
            candidate_y_px=pred['y_px'], confidence=pred['confidence'],
            distance_px='' if error is None else error, result=status))

    gaps = []
    for present, group in groupby(predictions, key=lambda row: row['candidate_present']):
        if present == '1':
            continue
        run = list(group)
        gaps.append(dict(first_source_frame=run[0]['source_frame'], last_source_frame=run[-1]['source_frame'],
                         frames=len(run), reason='model_no_candidate; visibility not inferred'))

    counts = Counter(row['result'] for row in comparisons)
    summary = dict(
        match='match2', rally_count=1, point=1, termination='out', reference_times=len(labels),
        visible_reference_times=sum(row['truth_visibility'] == 'visible' for row in labels),
        not_visible_reference_times=sum(row['truth_visibility'] == 'not_visible' for row in labels),
        counts=dict(counts), inherited_position_tolerance_px=16,
        median_distance_px=median(errors) if errors else None,
        maximum_distance_px=max(errors) if errors else None,
        core_frames=len(predictions), core_candidate_output_frames=sum(row['candidate_present'] == '1' for row in predictions),
        core_no_candidate_frames=sum(row['frames'] for row in gaps),
        output_at_not_visible_frames=[int(row['source_frame']) for row in comparisons if row['result'] == 'output_when_not_observable'],
        labels_collected_before_prediction_display=True,
        candidate_identity_for_not_visible_frames='unresolved',
        automated_repair_evaluated=False,
        limitation='The 12 reference frames are 22 source frames apart. They do not establish continuous tracking accuracy or evaluate repair on visible missed balls.',
    )
    OUT.mkdir(exist_ok=False)
    shutil.copy2(label_path, OUT / 'human_references_snapshot.csv')
    assert common.read(OUT / 'human_references_snapshot.csv') == labels
    shutil.copy2(REVIEW / 'rally_boundary_review.json', OUT / 'rally_boundary_review.json')
    diagnostic = BASELINE / 'F12393_raw_context_diagnostic.jpg'
    if diagnostic.exists():
        diagnostic.replace(OUT / diagnostic.name)
    common.write(OUT / 'reference_comparison.csv', comparisons)
    common.write(OUT / 'baseline_candidate_gaps.csv', gaps)
    common.dump(OUT / 'evaluation.json', summary)
    common.dump(OUT / 'input_hashes.json', {
        'baseline_unreviewed_candidates.csv': protected['baseline_unreviewed_candidates.csv'],
        'protocol.json': protected['protocol.json'],
        'human_references_snapshot.csv': common.sha(OUT / 'human_references_snapshot.csv'),
    })
    (OUT / 'README.md').write_text(
        '# match2 第1点の球追跡評価\n\n'
        '予測を表示せずに記録した12時点の参照を、推論済みの基準モデルと照合した。'
        '球が見えた10時点すべてで候補があり、既存の16px基準内だった。'
        f'中心差の中央値は{summary["median_distance_px"]:.2f}px、最大は{summary["maximum_distance_px"]:.2f}px。\n\n'
        '球が見えないと回答した2時点のうち、F12393には候補があり、F12437は出力がなかった。'
        'F12393の前後を拡大すると、黄色い球らしい像がネット上端を通過する。'
        '該当時点の候補の実球同定は未確定で、本人の参照は「見えない」のまま保存した。\n\n'
        '全243フレームでは199時点に候補があり、44時点が未出力だった。'
        '球の有無と位置を照合した範囲は、事前固定した12時点と第1点の1ラリー。'
        '球が見えた参照10時点に欠測がなかったため、この標本では欠測補助の効果を比較できない。\n',
        encoding='utf-8')
    common.dump(OUT / 'publication_hashes.json', {path.name: common.sha(path) for path in sorted(OUT.iterdir()) if path.is_file()})
    print(summary)


if __name__ == '__main__':
    main()
