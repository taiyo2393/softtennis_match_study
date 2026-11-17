"""Measure interpolation against held-out, unmasked box-bottom proxies."""
from collections import Counter, defaultdict
from itertools import groupby
from math import dist

import numpy as np

import build_044_spatiotemporal_pilot as common
from build_match2_player_ball_review import SCOPES
from review_match2_stroke_routes import METRES


BASE = common.HERE / 'match2_first_three_player_ball_v4'
INPUT = BASE / 'guarded_player_positions.csv'
OUT = common.HERE / 'match2_position_gap_trial'
GAP_LENGTHS = (2, 3, 4, 12, 16, 37, 39, 46, 108)


def available(row):
    return row is not None and row['position_status'] == 'proxy_shown'


def xy(row):
    return np.array([float(row['shown_court_x']), float(row['shown_court_y'])]) * METRES


def series_for_rally():
    tracks = defaultdict(dict)
    for row in common.read(INPUT):
        if row['display_id'].startswith('F'):
            key = int(row['point']), row['display_id']
            tracks[key][int(row['source_frame'])] = row
    for point, (_, start, end) in SCOPES.items():
        for player in ('F1', 'F2'):
            rows = tracks[point, player]
            yield point, player, [(f, rows.get(f)) for f in range(start, end + 1)]


def missing_runs(point, player, series):
    rows = dict(series)
    for visible, group in groupby(series, key=lambda item: available(item[1])):
        frames = list(group)
        if visible:
            continue
        first, last = frames[0][0], frames[-1][0]
        before, after = rows.get(first - 1), rows.get(last + 1)
        bracketed = available(before) and available(after)
        reasons = Counter(row['position_status'].split(':')[0] if row else 'no_detection'
                          for _, row in frames)
        yield dict(point=point, player=player, first_frame=first, last_frame=last,
                   missing_frames=len(frames), missing_seconds=len(frames) / common.FPS,
                   reasons=dict(reasons), bracketed=bracketed,
                   same_track_id=before['track_id'] == after['track_id'] if bracketed else None,
                   endpoint_displacement_m=dist(xy(before), xy(after)) if bracketed else None)


def visible_runs(series):
    for track, group in groupby(series, key=lambda item: item[1]['track_id'] if available(item[1]) else None):
        if track is not None:
            yield list(group)


def test_windows(point, player, series, length):
    # Within one gap length, use disjoint blocks including both endpoints.
    width = length + 2
    t = np.arange(1, length + 1)[:, None] / (length + 1)
    for run in visible_runs(series):
        for offset in range(0, len(run) - width + 1, width):
            block = run[offset:offset + width]
            values = np.array([xy(row) for _, row in block])
            predicted = (1 - t) * values[0] + t * values[-1]
            yield dict(point=point, player=player, first_withheld=block[1][0],
                       last_withheld=block[-2][0], hidden_frames=length,
                       endpoint_displacement_m=dist(values[0], values[-1]),
                       errors_linear_m=np.linalg.norm(predicted - values[1:-1], axis=1).tolist(),
                       errors_hold_last_m=np.linalg.norm(values[0] - values[1:-1], axis=1).tolist())


def stats(windows, key):
    if not windows:
        return None
    errors = np.concatenate([window[key] for window in windows])
    return dict(median_m=float(np.median(errors)), p90_m=float(np.quantile(errors, .9)),
                max_m=float(np.max(errors)),
                windows_all_frames_within_half_m=sum(max(window[key]) <= .5 for window in windows))


def audit_strokes(rows):
    players = defaultdict(set)
    for row in rows:
        if available(row):
            players[int(row['point']), int(row['source_frame'])].add(row['display_id'])
    results = []
    for row in common.read(BASE / 'stroke_candidates.csv'):
        point, ordinal, reference = (int(row[k]) for k in ('point', 'ordinal', 'reference_frame'))
        first, last = (int(row[k]) for k in ('first_candidate_frame', 'last_candidate_frame'))
        _, core_start, core_end = SCOPES[point]
        frames = range(max(core_start, first), min(core_end, last) + 1)
        missing = sorted({'N1', 'N2', 'F1', 'F2'} - players[point, reference])
        results.append(dict(point=point, stroke=ordinal, display_reference_frame=reference,
                            missing_players_at_display_reference=missing,
                            candidate_contact_frames=last - first + 1,
                            contact_frames_inside_tracking=len(frames),
                            contact_frames_with_four_positions=sum(len(players[point, f]) == 4 for f in frames)))
    return results


def main():
    OUT.mkdir(exist_ok=True)
    input_hash = common.sha(INPUT)
    series = list(series_for_rally())
    gaps = [gap for point, player, track in series for gap in missing_runs(point, player, track)]
    trials = [window for length in GAP_LENGTHS for point, player, track in series
              for window in test_windows(point, player, track, length)]
    for gap in gaps:
        observed = [w['endpoint_displacement_m'] for w in trials
                    if w['hidden_frames'] == gap['missing_frames'] and w['point'] == gap['point']]
        gap['clear_test_endpoint_comparators_in_rally'] = len(observed)
        gap['clear_test_endpoint_p95_m'] = float(np.quantile(observed, .95)) if observed else None
    summary = []
    for length in GAP_LENGTHS:
        windows = [window for window in trials if window['hidden_frames'] == length]
        summary.append(dict(gap_frames=length, gap_seconds=length / common.FPS,
                            windows=len(windows), windows_by_point=dict(Counter(w['point'] for w in windows)),
                            linear=stats(windows, 'errors_linear_m'), hold_last=stats(windows, 'errors_hold_last_m')))
    assert sum(g['missing_frames'] for g in gaps) == sum(not available(r) for _, _, s in series for _, r in s)
    assert all(w['last_withheld'] - w['first_withheld'] + 1 == w['hidden_frames'] for w in trials)
    assert common.sha(INPUT) == input_hash
    strokes = audit_strokes(common.read(INPUT))
    common.dump(OUT / 'stroke_position_coverage.json', strokes)
    common.dump(OUT / 'missing_intervals.json', gaps)
    common.dump(OUT / 'pseudo_gap_windows.json', trials)
    common.dump(OUT / 'result.json', dict(
        source_sha256=input_hash, trials=summary, actual_gap_count=len(gaps),
        actual_gap_frames=sum(g['missing_frames'] for g in gaps),
        intervals_with_foot_overlap_flag=sum('foot_overlap' in g['reasons'] for g in gaps),
        short_bracketed_same_track_intervals=sum(
            g['missing_frames'] <= 4 and g['bracketed'] and g['same_track_id'] for g in gaps),
        strokes_total=len(strokes),
        strokes_with_four_at_display_reference=sum(not s['missing_players_at_display_reference'] for s in strokes),
        strokes_with_four_throughout_contact_window=sum(
            s['contact_frames_with_four_positions'] == s['candidate_contact_frames'] for s in strokes),
        processing_decision='Retain missing positions in the reviewed paths. Clear holdouts are not ground truth '
                            'for occluded frames. No short overlap gap has same-ID visible endpoints.',
        target='Visible detector-box bottom proxies. No independent physical ground-position labels.',
        limitations='Three selected rallies. Blocks may be related within a rally and reused across lengths. '
                    'Clear pseudo-gaps do not establish error for actual occlusion or ID switches.',
    ))
    for row in summary:
        print(row)


if __name__ == '__main__':
    main()
