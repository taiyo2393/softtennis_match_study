"""Filter bbox-bottom proxies, median-smooth accepted runs, then interpolate.

Only files in 056_cleaned_wide_court are written; existing outputs are updated.
Run with study/.venv/Scripts/python.exe -B from the study repository.
"""
from collections import Counter, deque
import argparse
import math
from pathlib import Path
from statistics import median

import build_044_spatiotemporal_pilot as b

HERE = Path(__file__).resolve().parent
PARENT = HERE / '056_reviewed_consolidated'
OUT = HERE / '056_cleaned_wide_court'
FPS = 59.94
COURT_WIDTH_M = 10.97
COURT_LENGTH_M = 23.77
HEIGHT_HISTORY = 15
WARMUP_ACCEPTED_POINTS = 5
HEIGHT_LOW = 0.75
HEIGHT_HIGH = 1.35
MAX_SPEED_M_S = 8.0
JUMP_ALLOWANCE_M = 1.0
MAX_INTERPOLATION_FRAMES = 30
HEIGHT_RESET_FRAMES = 30
SMOOTHING_RADIUS_SOURCE_FRAMES = 8
SMOOTHING_MIN_POINTS = 3
SAMPLE_STEP = 2
N1_REVIEW_INTERVALS = ((154292, 154302), (154310, 154320),
                       (154358, 154368), (154792, 154806))
OBSERVED_STATUSES = ('observed', 'observed_after_height_reset')
STATUSES = (*OBSERVED_STATUSES, 'rejected_box_height', 'rejected_speed', 'interpolated', 'missing')


def counts(rows, field):
    return {slot: {status: sum(r['player_slot'] == slot and r[field] == status for r in rows)
                   for status in STATUSES} for slot in b.SLOTS}


def accepted_runs(accepted):
    """Use the same maximum accepted-endpoint gap as interpolation."""
    runs = []
    for row in accepted:
        if not runs or row['source_frame'] - runs[-1][-1]['source_frame'] > MAX_INTERPOLATION_FRAMES:
            runs.append([])
        runs[-1].append(row)
    return runs


def smooth_accepted(accepted):
    # Freeze every input first: medians must not consume previously smoothed data.
    for row in accepted:
        row.update(raw_court_x=row['court_x'], raw_court_y=row['court_y'])
    for run in accepted_runs(accepted):
        for row in run:
            window = [p for p in run if abs(p['source_frame'] - row['source_frame'])
                      <= SMOOTHING_RADIUS_SOURCE_FRAMES]
            if len(window) >= SMOOTHING_MIN_POINTS:
                row['court_x'] = median(p['raw_court_x'] for p in window)
                row['court_y'] = median(p['raw_court_y'] for p in window)
            row['smoothing_shift_m'] = math.hypot(
                (row['court_x'] - row['raw_court_x']) * COURT_WIDTH_M,
                (row['court_y'] - row['raw_court_y']) * COURT_LENGTH_M)


def position_before_smoothing(row, indexed):
    if row['filter_status'] in OBSERVED_STATUSES:
        return row['raw_court_x'], row['raw_court_y']
    if row['status'] == 'interpolated':
        start, end = row['interpolation_start_frame'], row['interpolation_end_frame']
        left, right = (indexed[f, row['player_slot']] for f in (start, end))
        alpha = (row['source_frame'] - start) / (end - start)
        return tuple(left[k] + alpha * (right[k] - left[k]) for k in ('raw_court_x', 'raw_court_y'))
    return None


def metric_position(x, y):
    return dict(court_x=x, court_y=y, court_x_m=x * COURT_WIDTH_M, court_y_m=y * COURT_LENGTH_M)


def smoothing_report(rows):
    indexed = {(r['source_frame'], r['player_slot']): r for r in rows}
    stats, runs_by_player = {}, {}
    for slot in b.SLOTS:
        accepted = [r for r in rows if r['player_slot'] == slot and r['filter_status'] in OBSERVED_STATUSES]
        shifts = sorted(r['smoothing_shift_m'] for r in accepted)
        rank = .95 * (len(shifts) - 1)
        low = math.floor(rank)
        p95 = shifts[low] + (rank - low) * (shifts[math.ceil(rank)] - shifts[low])
        stats[slot] = dict(accepted_points=len(shifts), changed_points=sum(s > 0 for s in shifts),
                           median_m=median(shifts), percentile_95_m=p95, maximum_m=max(shifts))
        runs_by_player[slot] = [dict(start_frame=run[0]['source_frame'], end_frame=run[-1]['source_frame'],
                                     points=len(run)) for run in accepted_runs(accepted)]
    intervals = []
    for start, end in N1_REVIEW_INTERVALS:
        points = []
        for f in range(start, end + 1, SAMPLE_STEP):
            row = indexed[f, 'near_front']
            before = position_before_smoothing(row, indexed)
            assert before is not None
            points.append(dict(source_frame=f, status=row['status'], filter_status=row['filter_status'],
                               pre_smoothing_court_y_m=before[1] * COURT_LENGTH_M,
                               input_proxy_court_y_m=float(row['court_proxy_y']) * COURT_LENGTH_M,
                               smoothed_court_y_m=row['court_y'] * COURT_LENGTH_M,
                               delta_court_y_m=(row['court_y'] - before[1]) * COURT_LENGTH_M))
        intervals.append(dict(start_frame=start, end_frame=end, points=points))
    strokes = b.read(PARENT / 'stroke_order_confirmed.csv')
    routes = b.read(PARENT / 'schematic_court_routes.csv')
    impacts = []
    assert len(strokes) == 13
    for stroke in strokes:
        f, slot = int(stroke['reference_source_frame']), stroke['hitter_candidate']
        row = indexed[f, slot]
        before = position_before_smoothing(row, indexed)
        assert before is not None
        after = row['court_x'], row['court_y']
        endpoints = []
        for route in routes:
            if route['anchor_id'] != stroke['stroke_id'] or route['anchor_kind'] not in ('hitter_proxy', 'next_hitter_proxy'):
                continue
            x, y = float(route['court_x']), float(route['court_y'])
            endpoints.append(dict(route_stroke_id=route['stroke_id'], anchor_order=int(route['anchor_order']),
                                  anchor_kind=route['anchor_kind'], endpoint=metric_position(x, y),
                                  distance_to_smoothed_hitter_m=math.hypot(
                                      (after[0] - x) * COURT_WIDTH_M, (after[1] - y) * COURT_LENGTH_M),
                                  distance_to_pre_smoothing_hitter_m=math.hypot(
                                      (before[0] - x) * COURT_WIDTH_M, (before[1] - y) * COURT_LENGTH_M)))
        assert endpoints, stroke['stroke_id']
        impacts.append(dict(stroke_id=stroke['stroke_id'], reference_source_frame=f, player_slot=slot,
                            status=row['status'], filter_status=row['filter_status'],
                            pre_smoothing=metric_position(*before), smoothed=metric_position(*after),
                            position_difference_m=math.hypot((after[0] - before[0]) * COURT_WIDTH_M,
                                                             (after[1] - before[1]) * COURT_LENGTH_M),
                            route_endpoints=endpoints))
    assert sum(len(r['route_endpoints']) for r in impacts) == sum(
        r['anchor_kind'] in ('hitter_proxy', 'next_hitter_proxy') for r in routes)
    return dict(statistics_population='Only post-reset accepted observations; interpolated/missing/rejected rows excluded',
                percentile_method='Linear interpolation at rank 0.95*(n-1) in sorted shifts, including zeros',
                shift_statistics_by_player=stats, accepted_runs_by_player=runs_by_player,
                n1_review_intervals=intervals, stroke_hitter_impacts=impacts,
                stroke_position_policy='Exact reference_source_frame; interpolated H09 uses the same endpoints before/after smoothing, not a nearest observation or the stroke-table proxy',
                route_endpoint_policy='Every hitter_proxy/next_hitter_proxy anchor is compared at its anchor_id stroke reference frame; bounce/predicted landing anchors are unchanged and are not hitter positions',
                coordinate_policy='CSV court coordinates remain normalized; metric x=x*10.97, y=y*23.77 (y=0 at near baseline)',
                independent_position_accuracy_validated=False)


def update_smoothing_readme(report):
    path = OUT / 'README.md'
    text = path.read_text(encoding='utf-8') if path.exists() else '# 056：選手代理位置の清掃と平滑化\n'
    marker = '\n<!-- median-smoothing -->\n'
    end_marker = '\n<!-- end-median-smoothing -->\n'
    if marker in text:
        head, tail = text.split(marker, 1)
        text = head + '\n' + tail.split(end_marker, 1)[1]
    lines = [
        '中央値による短い揺れの平滑化：', '',
        '既存の棄却・高さリセット判定を終え、補間する前に、filter_statusがobservedまたは'
        'observed_after_height_resetの点だけを処理する。選手ごとに採用点どうしの間隔が'
        '30元フレーム以下で続く区間に分け、30元フレームより長い欠けで切る（補間と同じ基準）。'
        '各採用点について、同一区間内でsource_frameが±8元フレーム以内（両端を含む）にある'
        '採用点すべてを窓とし、court_xとcourt_yを別々に中央値で置き換える。'
        '窓は点数ではなく時間で取り、欠測・棄却の位置を窓の点として使わない。区間端でも'
        '片側の点をそのまま使い、左右対称への縮小はしない。窓内が3点未満なら元の値を保つ。'
        '偶数点の中央値は中央2値の平均とする。平滑化前の値を一括保存して'
        'から中央値を計算し、処理済みの値を次の窓に使わない。補間の規則は従来どおりで、両端だけ'
        '平滑化後の採用値を使う。左映像の枠座標・実線/破線の描き方は維持した。', '',
        'CSVのraw_court_x/raw_court_yは平滑化前の採用値（正規化座標）、smoothing_shift_mは'
        'その採用点の平滑化による2次元移動距離。3列とも非採用点・補間点は空欄。'
        'court_x/court_yは平滑化後の値またはその両端からの補間値。', '',
        '|選手|採用点数|変化点数|移動中央値 m|95%点 m|最大 m|',
        '|---|---:|---:|---:|---:|---:|']
    for slot, stats in report['shift_statistics_by_player'].items():
        lines.append(f"|{b.NAMES[slot]}|{stats['accepted_points']}|{stats['changed_points']}|"
                     f"{stats['median_m']:.6f}|{stats['percentile_95_m']:.6f}|{stats['maximum_m']:.6f}|")
    lines.extend(['', '95%点はゼロを含む採用点全体の昇順値でrank=0.95×(n−1)を線形補間した。', '',
                  'N1の指定4区間（court_yのm換算、手前ベースラインを0）：', '',
                  '|元フレーム|平滑化前 y m|平滑化後 y m|後−前 m|状態|', '|---|---:|---:|---:|---|'])
    for interval in report['n1_review_intervals']:
        for p in interval['points']:
            lines.append(f"|F{p['source_frame']}|{p['pre_smoothing_court_y_m']:.6f}|{p['smoothed_court_y_m']:.6f}|{p['delta_court_y_m']:+.6f}|{p['status']}|")
    lines.extend(['', 'F154312/F154314は既存の速度棄却による補間点。上表の「前」は従来の補間位置、'
                  '「後」は平滑化した両端からの補間位置であり、これらの棄却点自体には中央値をかけていない。'])
    lines.extend(['', '13打球の打球者位置（reference_source_frameでの値、x/yともm換算）：', '',
                  '|打球|元フレーム|選手|前 (x, y) m|後 (x, y) m|位置差 m|固定模式線端点との差 m|',
                  '|---|---:|---|---|---|---:|---:|'])
    for impact in report['stroke_hitter_impacts']:
        pre, post = impact['pre_smoothing'], impact['smoothed']
        distances = sorted(set(e['distance_to_smoothed_hitter_m'] for e in impact['route_endpoints']))
        endpoint_text = ' / '.join(f'{v:.6f}' for v in distances)
        lines.append(f"|{impact['stroke_id']}|{impact['reference_source_frame']}|{b.NAMES[impact['player_slot']]}|"
                     f"({pre['court_x_m']:.6f}, {pre['court_y_m']:.6f})|"
                     f"({post['court_x_m']:.6f}, {post['court_y_m']:.6f})|{impact['position_difference_m']:.6f}|{endpoint_text}|")
    lines.extend(['', 'H09は棄却点の短区間補間であり、前後とも同じ両端から計算した表示位置。'
                  '打球表に保存された生の代理点とは平滑化前から異なる。固定模式線の各hitter_proxy/'
                  'next_hitter_proxy端点との距離はsummaryに全件記録した。配球線自体は変更していない。', '',
                  '限界：±8元フレームは片側8/59.94=約0.1335秒、全幅16/59.94=約0.2669秒。'
                  '2元フレーム刻みの欠けのない内部窓では9点になるが、欠測・棄却・区間端では'
                  '点数が減り、偶数点にもなる。出っ張りが窓の過半数を占める場合は残りうるため、'
                  '抑えられる長さを時間だけでは保証できない。片側窓や周囲の動きによっても効果が'
                  '変わり、本当の短い前後の動きも消しうる。x/y別の中央値は実際に'
                  '観測されなかった座標の組を作りうる。接触時刻・足元の真値との精度は検証しておらず、'
                  '滑らかさは位置精度の向上を保証しない。', ''])
    section = marker + '\n'.join(lines) + end_marker
    left_marker = '<!-- left-filter-render-verification -->'
    if left_marker in text:
        text = text.replace(left_marker, section + '\n' + left_marker, 1)
    else:
        text += section
    path.write_text(text, encoding='utf-8')


def clean_player(slot, source_rows, display_frames):
    source_rows = sorted(source_rows, key=lambda r: int(r['source_frame']))
    heights = [float(r['y2_px']) - float(r['y1_px']) for r in source_rows]
    assert heights and all(math.isfinite(h) and h > 0 for h in heights)
    global_height = median(heights)
    history = deque(maxlen=HEIGHT_HISTORY)
    accepted = []
    diagnostics = {}
    pending_height = []
    height_resets = []
    previous_raw = None

    def accepted_jump(row):
        if not accepted:
            return '', False
        previous = accepted[-1]
        jump = math.hypot((float(row['court_proxy_x']) - previous['court_x']) * COURT_WIDTH_M,
                          (float(row['court_proxy_y']) - previous['court_y']) * COURT_LENGTH_M)
        limit = MAX_SPEED_M_S * (row['source_frame'] - previous['source_frame']) / FPS + JUMP_ALLOWANCE_M
        return jump, jump > limit

    for raw, height in zip(source_rows, heights):
        frame = int(raw['source_frame'])
        assert frame not in diagnostics, (slot, frame, 'duplicate input')
        x, y = float(raw['court_proxy_x']), float(raw['court_proxy_y'])
        assert math.isfinite(x) and math.isfinite(y)
        reference = global_height if len(accepted) < WARMUP_ACCEPTED_POINTS and not height_resets else median(history)
        jump, speed_rejected = accepted_jump({**raw, 'source_frame': frame})
        raw_jump, raw_limit = '', ''
        if previous_raw is not None:
            raw_jump = math.hypot((x - float(previous_raw['court_proxy_x'])) * COURT_WIDTH_M,
                                  (y - float(previous_raw['court_proxy_y'])) * COURT_LENGTH_M)
            raw_limit = MAX_SPEED_M_S * (frame - previous_raw['source_frame']) / FPS + JUMP_ALLOWANCE_M
        if height < HEIGHT_LOW * reference or height > HEIGHT_HIGH * reference:
            status = 'rejected_box_height'
        elif speed_rejected:
            status = 'rejected_speed'
        else:
            status = 'observed'
        row = dict(source_frame=frame, player_slot=slot, status=status,
                   court_proxy_x=raw['court_proxy_x'], court_proxy_y=raw['court_proxy_y'],
                   court_x=x if status == 'observed' else '',
                   court_y=y if status == 'observed' else '',
                   box_height_px=height, reference_height_px=reference, jump_m=jump,
                   filter_status=status, input_status=status,
                   raw_previous_frame=previous_raw['source_frame'] if previous_raw else '',
                   raw_jump_m=raw_jump, raw_speed_limit_m=raw_limit,
                   height_reset_start_frame='', height_reset_trigger_frame='',
                   interpolation_start_frame='', interpolation_end_frame='')
        diagnostics[frame] = row
        previous_raw = row
        if status == 'observed':
            accepted.append(row)
            history.append(height)
        if status != 'rejected_box_height':
            pending_height.clear()
            continue
        pending_height.append(row)
        if frame - pending_height[0]['source_frame'] <= HEIGHT_RESET_FRAMES:
            continue

        # Retrospective reset: discard the stale accepted-height history. The
        # run is consecutive among raw observations (missing samples are not
        # invented); its duration is measured in original source frames.
        history.clear()
        reset_reference = median(r['box_height_px'] for r in pending_height)
        restored, speed_kept = [], []
        start = pending_height[0]['source_frame']
        for candidate in pending_height:
            jump, accepted_speed_rejected = accepted_jump(candidate)
            raw_speed_rejected = (candidate['raw_jump_m'] != '' and
                                  candidate['raw_jump_m'] > candidate['raw_speed_limit_m'])
            status = 'rejected_speed' if accepted_speed_rejected or raw_speed_rejected else 'observed_after_height_reset'
            candidate.update(status=status, filter_status=status, input_status=status,
                             reference_height_px=reset_reference, jump_m=jump,
                             height_reset_start_frame=start, height_reset_trigger_frame=frame,
                             court_x=float(candidate['court_proxy_x']) if status in OBSERVED_STATUSES else '',
                             court_y=float(candidate['court_proxy_y']) if status in OBSERVED_STATUSES else '')
            if status in OBSERVED_STATUSES:
                accepted.append(candidate)
                history.append(candidate['box_height_px'])
                restored.append(candidate['source_frame'])
            else:
                speed_kept.append(candidate['source_frame'])
        # Even if every point violates speed, replace the stale height
        # reference with the run values; speed still governs acceptance.
        if not history:
            history.extend(r['box_height_px'] for r in pending_height)
        height_resets.append(dict(start_frame=start, trigger_frame=frame,
                                  span_source_frames=frame - start,
                                  candidate_points=len(pending_height),
                                  reset_reference_height_px=reset_reference,
                                  restored_frames=restored, rejected_speed_frames=speed_kept))
        pending_height.clear()

    # Rejection and height-reset decisions are complete before any coordinates change.
    smooth_accepted(accepted)

    # Fill only strictly between two accepted observations; no extrapolation.
    interpolated = {}
    for left, right in zip(accepted, accepted[1:]):
        start, end = left['source_frame'], right['source_frame']
        if end - start > MAX_INTERPOLATION_FRAMES:
            continue
        for frame in display_frames:
            if start < frame < end:
                alpha = (frame - start) / (end - start)
                interpolated[frame] = (
                    left['court_x'] + alpha * (right['court_x'] - left['court_x']),
                    left['court_y'] + alpha * (right['court_y'] - left['court_y']), start, end)

    result = []
    for frame in display_frames:
        row = dict(diagnostics[frame]) if frame in diagnostics else dict(
            source_frame=frame, player_slot=slot, status='missing', court_proxy_x='', court_proxy_y='',
            court_x='', court_y='', box_height_px='', reference_height_px='', jump_m='',
            filter_status='missing', input_status='missing', raw_previous_frame='',
            raw_jump_m='', raw_speed_limit_m='', height_reset_start_frame='', height_reset_trigger_frame='',
            interpolation_start_frame='', interpolation_end_frame='')
        if frame in interpolated:
            assert row['status'] not in OBSERVED_STATUSES
            x, y, start, end = interpolated[frame]
            row.update(status='interpolated', court_x=x, court_y=y,
                       interpolation_start_frame=start, interpolation_end_frame=end)
        for key in ('raw_court_x', 'raw_court_y', 'smoothing_shift_m'):
            row.setdefault(key, '')
        result.append(row)
    return result, global_height, height_resets


def main():
    assert OUT == (HERE / '056_cleaned_wide_court').resolve(), 'Only the authorized output directory may be changed'
    old_rows = b.read(OUT / 'cleaned_player_positions.csv') if (OUT / 'cleaned_player_positions.csv').exists() else []
    old_summary = b.load(OUT / 'cleaning_summary.json') if (OUT / 'cleaning_summary.json').exists() else {}
    protected = b.load(PARENT / 'publication_hashes.json')
    assert all(b.sha(PARENT / name) == digest for name, digest in protected.items())
    # Capture every file in existing 056_* directories and existing sibling scripts.
    existing = [p for folder in HERE.glob('056_*') if folder.is_dir() and folder.resolve() != OUT
                for p in folder.rglob('*') if p.is_file()]
    existing += list(HERE.glob('*.py'))
    before = {str(p.relative_to(HERE)): b.sha(p) for p in sorted(set(existing))}
    source = b.read(PARENT / 'player_court_proxies.csv')
    mapping = b.read(PARENT / 'video_source_frame_map.csv')
    frames = [int(r['source_frame']) for r in mapping]
    assert frames == sorted(set(frames)) and len(frames) == 401
    assert {r['player_slot'] for r in source} == set(b.SLOTS)
    assert all(int(r['source_frame']) in set(frames) for r in source)
    rows, global_heights, height_resets = [], {}, {}
    for slot in b.SLOTS:
        player_rows, global_height, resets = clean_player(
            slot, [r for r in source if r['player_slot'] == slot], frames)
        rows.extend(player_rows)
        global_heights[slot] = global_height
        height_resets[slot] = resets
    rows.sort(key=lambda r: (r['source_frame'], b.SLOTS.index(r['player_slot'])))
    assert len(rows) == 401 * 4
    assert sum(r['input_status'] != 'missing' for r in rows) == len(source)
    for row in rows:
        assert (row['court_x'] != '' and row['court_y'] != '') == (row['status'] in (*OBSERVED_STATUSES, 'interpolated'))
    # All pre-existing non-position fields must retain their exact CSV values.
    coordinate_fields = {'court_x', 'court_y', 'raw_court_x', 'raw_court_y', 'smoothing_shift_m'}
    if old_rows:
        assert len(old_rows) == len(rows)
        for old, new in zip(old_rows, rows):
            assert all(str(new[k]) == value for k, value in old.items() if k not in coordinate_fields), (old['source_frame'], old['player_slot'])
    report = smoothing_report(rows)
    OUT.mkdir(exist_ok=True)
    b.write(OUT / 'cleaned_player_positions.csv', rows)
    b.dump(OUT / 'existing_files_sha256_before.json', before)
    summary = dict(
        display_frames=len(frames), output_rows=len(rows), original_player_rows=len(source),
        rules=dict(source_fps=FPS, court_width_m=COURT_WIDTH_M, court_length_m=COURT_LENGTH_M,
                   height_history_max_accepted_points=HEIGHT_HISTORY,
                   warmup_accepted_points_using_global_median=WARMUP_ACCEPTED_POINTS,
                   height_low_ratio=HEIGHT_LOW, height_high_ratio=HEIGHT_HIGH,
                   height_thresholds='strictly below low or above high; equality accepted',
                   height_reset_after_source_frames=HEIGHT_RESET_FRAMES,
                   height_reset='Consecutive raw height rejections spanning strictly more than 30 source frames are reconsidered from the start; clear stale history and rebuild from restored run heights (last 15). Missing samples do not break a run.',
                   height_reset_speed='Keep rejected_speed if either the original previous-accepted rule or the adjacent-raw-point rule fails, including the raw transition into the run. Raw rule also checks the return from an outlier.',
                   maximum_speed_m_s=MAX_SPEED_M_S, jump_allowance_m=JUMP_ALLOWANCE_M,
                   speed_rule='distance > 8 * (source_frame - previous_accepted_frame) / 59.94 + 1',
                   rejection_priority=['rejected_box_height', 'rejected_speed'],
                   smoothing_max_accepted_gap_source_frames=MAX_INTERPOLATION_FRAMES,
                   smoothing_radius_source_frames=SMOOTHING_RADIUS_SOURCE_FRAMES,
                   smoothing_min_window_points=SMOOTHING_MIN_POINTS,
                   smoothing_sample_step_source_frames=SAMPLE_STEP,
                   smoothing='After all rejections and height resets, before interpolation; only observed/observed_after_height_reset. Split each player only at accepted-frame gaps > 30, as for interpolation. For each accepted point use all accepted points in the same run with absolute source_frame difference <= 8; retain one-sided windows at run ends without symmetric shrinking. Fewer than 3 window points: keep raw coordinates. Separate nonrecursive medians of raw court_x/y; average the middle two values for even windows. Interpolate from smoothed endpoints.',
                   interpolation_max_endpoint_gap_source_frames=MAX_INTERPOLATION_FRAMES,
                   interpolation_max_endpoint_gap_seconds=MAX_INTERPOLATION_FRAMES / FPS,
                   interpolation='linear in normalized court coordinates on display source_frames; no extrapolation',
                   status_policy='status is final display state; filter_status and backward-compatible input_status retain the post-reset, pre-interpolation decision. Rejected speed points may be interpolated under the unchanged short-gap rule.'),
        global_median_box_height_px=global_heights,
        height_resets_by_player=height_resets,
        height_reset_restored_points_by_player={slot: sum(len(r['restored_frames']) for r in resets)
                                              for slot, resets in height_resets.items()},
        height_reset_restored_points_total=sum(r['filter_status'] == 'observed_after_height_reset' for r in rows),
        status_counts_by_player=counts(rows, 'status'),
        filter_status_counts_by_player=counts(rows, 'filter_status'),
        input_status_counts_by_player=counts(rows, 'input_status'),
        status_counts_total=dict(Counter(r['status'] for r in rows)),
        input_sha256={name: b.sha(PARENT / name) for name in
                      ('player_court_proxies.csv', 'video_source_frame_map.csv',
                       'stroke_order_confirmed.csv', 'schematic_court_routes.csv')},
        smoothing=report,
        regression_checks=dict(previous_output_rows_compared=len(old_rows),
                               all_existing_non_position_csv_fields_unchanged=True,
                               protected_files_verified_unchanged=len(before)),
        independent_position_accuracy_validated=False)
    for key in ('height_resets_by_player', 'status_counts_by_player', 'filter_status_counts_by_player',
                'input_status_counts_by_player', 'global_median_box_height_px'):
        if key in old_summary:
            assert summary[key] == old_summary[key], key
    if 'rendering' in old_summary:
        summary['rendering'] = old_summary['rendering']
    assert all(b.sha(HERE / name) == digest for name, digest in before.items())
    b.dump(OUT / 'cleaning_summary.json', summary)
    update_smoothing_readme(report)
    print(summary['status_counts_by_player'], flush=True)
    print(report['shift_statistics_by_player'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT,
                        help='Authorized 056_cleaned_wide_court directory; generated files are updated.')
    OUT = parser.parse_args().output_dir.resolve()
    main()
