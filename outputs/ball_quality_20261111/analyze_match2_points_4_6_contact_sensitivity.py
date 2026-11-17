"""Measure receiving-pair geometry variation across source-frame contact windows."""
from collections import defaultdict
from statistics import median

import build_044_spatiotemporal_pilot as common
from summarize_match2_receiving_positions import METRES, summarize


SOURCE = common.HERE / 'match2_points_4_6_stroke_review'
STROKES = SOURCE / 'stroke_observations.json'
POSITIONS = SOURCE / 'guarded_positions.json'
OUT = common.HERE / 'match2_points_4_6_contact_sensitivity'
FEATURES = {'lateral_spacing_m': '横間隔', 'depth_difference_m': '前後差'}


def measure():
    poses = defaultdict(dict)
    for row in common.load(POSITIONS):
        frame, player = row['source_frame'], row['display_id']
        assert player not in poses[frame], (frame, player)
        poses[frame][player] = row['court_xy']
    rows, excluded = [], []
    for point_record in common.load(STROKES)['points']:
        point = point_record['point']
        for ordinal, event in enumerate(point_record['events'], 1):
            status = event['status']
            start, end = event['frames']
            if ordinal == 1 or status not in ('candidate', 'user_confirmed'):
                excluded.append(dict(point=point, stroke=ordinal, candidate_first=start, candidate_last=end,
                                     reason='serve_contact_precedes_tracking' if ordinal == 1 else status))
                continue
            frames = {(point, f): {name: xy for name, xy in poses[f].items() if xy is not None}
                      for f in range(start, end + 1)}
            geometry = summarize(dict(point=point, ordinal=ordinal, hitter=event['hitter'],
                                      first_candidate_frame=start, last_candidate_frame=end), None, frames)
            assert geometry['pair_frames'] + len(geometry['missing']) == geometry['candidate_frames']
            metrics = {}
            if geometry['whole_window_pair']:
                for key in FEATURES:
                    low = min(geometry['per_frame'], key=lambda sample: sample[key])
                    high = max(geometry['per_frame'], key=lambda sample: sample[key])
                    metrics[key] = dict(min=low[key], max=high[key], span=high[key]-low[key],
                                        min_frame=low['source_frame'], max_frame=high['source_frame'])
            rows.append(dict(point=point, stroke=ordinal, hitter=event['hitter'],
                             contact_status=status, result=event.get('result'),
                             confirmed_contact_frame=event.get('contact_frame'),
                             first_frame=start, last_frame=end, elapsed_ms=(end-start)*1000/common.FPS,
                             receiving_players=geometry['receiving_players'],
                             frames_total=geometry['candidate_frames'], frames_with_pair=geometry['pair_frames'],
                             whole_window_pair=geometry['whole_window_pair'], features=metrics,
                             missing=geometry['missing'], per_frame=geometry['per_frame']))
    return rows, excluded


def span_stats(full):
    metrics = {}
    for key in FEATURES:
        largest = max(full, key=lambda row: row['features'][key]['span'])
        metrics[key] = dict(median=median(row['features'][key]['span'] for row in full),
                            maximum=largest['features'][key]['span'],
                            maximum_point=largest['point'], maximum_stroke=largest['stroke'])
    return metrics


def write_report(result):
    stats = result['candidate_span_stats_m']
    lines = [
        '# match2 第4〜6点　接触時刻と守備配置の変動',
        '',
        '打球者の相手側2人について、接触候補区間の各フレームで横間隔（左右方向の距離）と前後差'
        '（ベースラインに垂直な方向の距離）を計算した。画面上の推定足元をコートに投影してメートルに換算した。',
        '区間内のすべてのフレームで2人の位置が得られた場面は、最小値・最大値とその差を示す。',
        '',
        f'打球者候補のあるサーブ後の{result["candidates"]}場面のうち、区間全体で位置が得られたのは'
        f'{result["candidate_full_windows"]}場面。画面外の第4点終盤は接触が未確認のため対象外。',
        '第6点の最終打球は映像レビューで接触・結果を確認し、下段に記載した。',
        '',
        '| 指標 | 区間内の変動幅の中央値 | 最大 | 最大の場面 |',
        '| --- | ---: | ---: | --- |',
    ]
    for key, label in FEATURES.items():
        stat = stats[key]
        lines.append(f'| {label} | {stat["median"]:.2f} m | {stat["maximum"]:.2f} m | '
                     f'第{stat["maximum_point"]}点・候補{stat["maximum_stroke"]} |')
    lines += [
        '',
        '## 場面別の結果',
        '',
        '各セルは区間内の最小〜最大［変動幅］。候補番号は候補表での並び順。',
        '',
        '| 点・候補 | 打球者 | 元映像のフレーム | 守備側2人の位置あり | 横間隔 (m) | 前後差 (m) |',
        '| --- | --- | --- | ---: | ---: | ---: |',
    ]
    for row in result['windows']:
        values = []
        for key in FEATURES:
            metric = row['features'].get(key)
            values.append(f'{metric["min"]:.2f}〜{metric["max"]:.2f}［{metric["span"]:.2f}］' if metric else '—')
        hitter = f'{row["hitter"]}（ネット・確認済み）' if row['result'] == 'net' else row['hitter']
        lines.append(f'| 第{row["point"]}点・{row["stroke"]} | {hitter} | F{row["first_frame"]}〜F{row["last_frame"]} | '
                     f'{row["frames_with_pair"]}/{row["frames_total"]} | {values[0]} | {values[1]} |')
    lines += [
        '',
        '第6点の末尾行はF2が触れた場面で、その打球はネットに掛かった。この行の位置はF2から見た相手'
        '（手前側）の2人。F2の返球失敗時の奥側の守備配置を調べる場合は、直前のN1打球候補時に計測する。',
        '',
        '今回の変動幅には選手の移動と足元位置の推定の揺れが含まれる。固定した追跡結果と接触候補区間に'
        '対する感度であり、座標の精度や配置と得失点の関係はこの集計の範囲外。',
    ]
    (OUT / '接触時刻と守備配置.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    rows, excluded = measure()
    candidates = [r for r in rows if r['contact_status'] == 'candidate']
    complete = [r for r in candidates if r['whole_window_pair']]
    result = dict(
        source_fps='60000/1001', reference_court_size_m=METRES,
        source_sha256={p.name: common.sha(p) for p in (STROKES, POSITIONS)},
        candidates=len(candidates), candidate_full_windows=len(complete),
        candidate_span_stats_m=span_stats(complete), excluded=excluded, windows=rows,
    )
    OUT.mkdir(exist_ok=True)
    common.dump(OUT / 'contact_sensitivity.json', result)
    write_report(result)


if __name__ == '__main__':
    main()
