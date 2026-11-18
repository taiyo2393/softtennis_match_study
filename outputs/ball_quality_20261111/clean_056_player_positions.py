"""Filter bbox-bottom court proxies, then interpolate short accepted-point gaps.

Only this script and the new 056_cleaned_wide_court directory are written.
Run with study/.venv/Scripts/python.exe from the study repository.
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
OBSERVED_STATUSES = ('observed', 'observed_after_height_reset')
STATUSES = (*OBSERVED_STATUSES, 'rejected_box_height', 'rejected_speed', 'interpolated', 'missing')


def counts(rows, field):
    return {slot: {status: sum(r['player_slot'] == slot and r[field] == status for r in rows)
                   for status in STATUSES} for slot in b.SLOTS}


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
        result.append(row)
    return result, global_height, height_resets


def main():
    assert not OUT.exists() or (OUT.is_dir() and not any(OUT.iterdir())), f'Refusing to overwrite nonempty output: {OUT}'
    protected = b.load(PARENT / 'publication_hashes.json')
    assert all(b.sha(PARENT / name) == digest for name, digest in protected.items())
    # Capture every file in existing 056_* directories and existing sibling scripts.
    existing = [p for folder in HERE.glob('056_*') if folder.is_dir()
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
                      ('player_court_proxies.csv', 'video_source_frame_map.csv')},
        independent_position_accuracy_validated=False)
    b.dump(OUT / 'cleaning_summary.json', summary)
    print(summary['status_counts_by_player'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT,
                        help='New or empty output directory; existing files are never overwritten.')
    OUT = parser.parse_args().output_dir.resolve()
    main()
