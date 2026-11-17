"""Score the next two match2 rallies and combine all three point summaries."""
from collections import Counter
from pathlib import Path
import shutil
from statistics import median
import sys

import build_044_spatiotemporal_pilot as common

sys.path.insert(0, str(common.OLD))
from evaluate_reserved_ball_sequences import classify

BASE = common.HERE / 'match2_points_2_3'
FIRST = common.HERE / 'match2_first_point_ball_evaluated'
LABELS = common.HERE / 'match2_points_2_3_ball_review/manual_ball_references.csv'
OUT = common.HERE / 'match2_first_three_points_evaluated'


def summarize(point, comparisons, frame_count, candidate_count):
    counts = Counter(row['result'] for row in comparisons)
    distances = [float(row['distance_px']) for row in comparisons if row['distance_px'] != '']
    return dict(point=point, reference_times=len(comparisons),
        visible_times=sum(row['truth_visibility'] == 'visible' for row in comparisons),
        not_visible_times=sum(row['truth_visibility'] == 'not_visible' for row in comparisons),
        correct16px=counts['correct16px'], wrong_position=counts['wrong_position'],
        missing_visible_ball=counts['missing_visible_ball'],
        output_when_not_observable=counts['output_when_not_observable'],
        no_output_when_not_observable=counts['no_output_when_not_observable'],
        uncertain_times=counts['excluded_ball_uncertain'],
        median_distance_px=median(distances) if distances else None,
        maximum_distance_px=max(distances) if distances else None,
        core_frames=frame_count, core_candidate_output_frames=candidate_count)


def main():
    for folder in (BASE, FIRST):
        for name, digest in common.load(folder/'publication_hashes.json').items():
            assert common.sha(folder/name) == digest, name
    protocol = common.load(BASE/'protocol.json')
    verification = common.load(BASE/'verification.json')
    assert protocol['frozen_before_ball_outputs']
    labels = common.read(LABELS)
    assert [int(row['source_frame']) for row in labels] == protocol['reference_frames_selected_before_inference']
    assert all(row['reviewer_id'] == 'human_user' and row['annotated_at_utc'] for row in labels)
    for row in labels:
        assert row['truth_visibility'] in ('visible', 'not_visible', 'uncertain')
        if row['truth_visibility'] == 'visible':
            assert 0 <= int(row['truth_x_px']) < 1920 and 0 <= int(row['truth_y_px']) < 1080
        else:
            assert row['truth_x_px'] == row['truth_y_px'] == ''
        name = 'F' + row['source_frame'] + '.png'
        assert common.sha(BASE/'raw_reference_frames'/name) == verification['reference_images'][name]
    by_frame = {int(row['source_frame']): row for row in labels}
    first = common.load(FIRST/'evaluation.json')
    combined = [dict(point=1, **row) for row in common.read(FIRST/'reference_comparison.csv')]
    summaries = [summarize(1, combined, first['core_frames'], first['core_candidate_output_frames'])]
    for point, saved in zip(protocol['points'], verification['points'], strict=True):
        number = point['point']
        assert number == saved['point']
        path = BASE/f'point_{number:02}'/'baseline_unreviewed_candidates.csv'
        assert common.sha(path) == saved['baseline_csv_sha256']
        predictions = common.read(path)
        start, end = point['core_inclusive']
        assert [int(row['source_frame']) for row in predictions] == list(range(start, end+1))
        pred_by_frame = {int(row['source_frame']): row for row in predictions}
        comparisons = []
        for frame in point['reference_frames']:
            label, prediction = by_frame[frame], pred_by_frame[frame]
            state, distance = classify(dict(label, phase_status='play'),
                dict(prediction, visible=prediction['candidate_present']))
            comparisons.append(dict(point=number, source_frame=frame,
                truth_visibility=label['truth_visibility'], truth_x_px=label['truth_x_px'],
                truth_y_px=label['truth_y_px'], candidate_present=prediction['candidate_present'],
                candidate_x_px=prediction['x_px'], candidate_y_px=prediction['y_px'],
                confidence=prediction['confidence'], distance_px='' if distance is None else distance, result=state))
        summaries.append(summarize(number, comparisons, len(predictions), saved['candidate_output_frames']))
        combined.extend(comparisons)
    total = summarize('all', combined, sum(row['core_frames'] for row in summaries),
                      sum(row['core_candidate_output_frames'] for row in summaries))
    assert total['reference_times'] == 24
    misses = [row for row in combined if row['result'] == 'missing_visible_ball']
    ambiguous = [row for row in combined if row['result'] == 'output_when_not_observable']
    OUT.mkdir(exist_ok=False)
    shutil.copy2(LABELS, OUT/'points_2_3_human_reference_snapshot.csv')
    assert common.read(OUT/'points_2_3_human_reference_snapshot.csv') == labels
    common.write(OUT/'reference_comparison_all_24_times.csv', combined)
    common.write(OUT/'point_summary.csv', [*summaries, total])
    common.dump(OUT/'evaluation.json', dict(
        match='match2', rallies=3, score_tolerance_px=16,
        per_point=summaries, total=total, visible_misses=misses,
        outputs_at_human_not_visible_times=ambiguous,
        references='Six per later point and twelve for point 1. Raw images, fixed before inference.',
        limitations='One match and sparse sampling within three rallies. Candidate identity at human not-visible times remains unresolved.'))
    report = [
        '# match2 第1〜3点の基準モデル評価', '',
        '同じ試合の3ラリーから、推論前に選んだ24時点を人の入力と照合した。球中心の許容距離は従来と同じ16px。', '',
        '| ラリー | 参照時点 | 見えた球の位置一致 | 見えた球の未検出 | 見えない時点の候補あり |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in [*summaries, total]:
        name = '合計' if row['point'] == 'all' else f'第{row["point"]}点'
        report.append(f'| {name} | {row["reference_times"]} | {row["correct16px"]}/{row["visible_times"]} | '
                      f'{row["missing_visible_ball"]} | {row["output_when_not_observable"]}/{row["not_visible_times"]} |')
    report.extend(['', f'座標を比較できた{total["correct16px"]+total["wrong_position"]}時点の中心差は'
        f'中央値{total["median_distance_px"]:.2f}px、最大{total["maximum_distance_px"]:.2f}px。',
        '見える球を取り逃したのは第3点終盤のF15505。確認位置は元画像の(723, 661)。',
        'F12393とF13896は「見えない」の参照に対して出力があり、候補が実球かどうかは未確定。', '',
        '第2点・第3点の接触と終端は元映像から範囲を見積もった。評価は等間隔に選んだ参照時点に限る。', ''])
    (OUT/'README.md').write_text('\n'.join(report), encoding='utf-8')
    common.dump(OUT/'publication_hashes.json', {p.name: common.sha(p) for p in sorted(OUT.iterdir()) if p.is_file()})
    print(total, flush=True)


if __name__ == '__main__':
    main()
