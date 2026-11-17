"""Transfer the original player setup to the next three consecutive rallies."""
from collections import Counter
from itertools import groupby
from math import dist
import sys

import cv2
import numpy as np

import build_044_spatiotemporal_pilot as common
import build_match2_player_ball_review as pilot
from scan_match2_points_4_6 import OUT as BALL_STAGE
from review_match2_stroke_routes import display_guards


OUT = common.HERE / 'match2_points_4_6_player_ball'
BALL_PATH = OUT / 'ball_candidates_for_review.csv'
TRAIL_REVISION = common.HERE / 'match2_points_4_6_ball_trail_v2'
BALL_REVIEW_REVISION = common.HERE / 'match2_points_4_6_ball_review_v3'
VISUAL_EXCLUSIONS = common.HERE / 'match2_points_4_6_ball_trail_audit/visual_exclusions.json'
RECOVERY_REVIEW = common.HERE / 'match2_points_4_6_ball_recovery/visual_recovery_review.json'
RECOVERY_REVISION = common.HERE / 'match2_points_4_6_ball_recovery_v4'
TRAIL_STEP_LIMIT_PX = 100
# Exclude dissolves identified in the source. Retain the preceding terminal review context.
LAST_REVIEW_FRAME = {4: 17299, 5: 19055, 6: 21396}


def cases():
    return common.load(BALL_STAGE / 'protocol.json')['points']


def scopes():
    return {p['point']: (p['cache_span'][0], p['core_inclusive'][0], LAST_REVIEW_FRAME[p['point']])
            for p in cases()}


def track():
    OUT.mkdir(exist_ok=True)
    windows = scopes()
    checked = common.load(BALL_STAGE / 'ball_baseline_verification.json')
    verified = {r['point']: r['candidate_file_sha256'] for r in checked['per_point']}
    rows = []
    for p, (_, first, last) in windows.items():
        path = BALL_STAGE / f'point_{p:02}' / 'baseline_unreviewed_candidates.csv'
        assert common.sha(path) == verified[p]
        selected = [r for r in common.read(path) if first <= int(r['source_frame']) <= last]
        assert [int(r['source_frame']) for r in selected] == list(range(first, last + 1))
        rows.extend(dict(point=p, source_frame=r['source_frame'], current_candidate_present=r['candidate_present'],
                         current_x_px=r['x_px'], current_y_px=r['y_px'],
                         current_position_source='baseline_model_candidate',
                         baseline_identity_confirmed_in_followup=0) for r in selected)
    common.write(BALL_PATH, rows)
    pilot.track_players(OUT, windows, BALL_PATH)


def inspect():
    people = common.grouped(common.read(OUT / 'player_court_proxies.csv'))
    for p, (_, first, last) in scopes().items():
        frames = BALL_STAGE / f'point_{p:02}' / 'frames'
        preview = []
        for f in np.linspace(first, last, 7).round().astype(int):
            im = cv2.imread(str(frames / f'{f}.jpg'))
            assert im is not None
            by_slot = {r['player_slot']: r for r in people[f]}
            cells = []
            for slot in common.SLOTS:
                r = by_slot.get(slot)
                cell = np.zeros((151, 162, 3), np.uint8)
                if r:
                    a, b, c, d = [round(float(r[k])) for k in ('x1', 'y1', 'x2', 'y2')]
                    crop = im[max(b-5, 0):min(d+6, 1080), max(a-6, 0):min(c+7, 1920)]
                    h, w = crop.shape[:2]
                    scale = min(162/w, 125/h)
                    w, h = max(1, round(w*scale)), max(1, round(h*scale))
                    cell[:h, (162-w)//2:(162-w)//2+w] = cv2.resize(crop, (w, h))
                note = f'F{f} {common.NAMES[slot]} [{r["track_id"] if r else "-"}]'
                cv2.putText(cell, note, (4, 145), cv2.FONT_HERSHEY_SIMPLEX, .51, (210, 235, 243), 1)
                cells.append(cell)
            preview.append(np.hstack(cells))
        assert cv2.imwrite(str(OUT / f'identity_p{p}.jpg'), np.vstack(preview))
    visits = Counter((int(r['point']), r['player_slot'], r['track_id']) for rows in people.values() for r in rows)
    for point in scopes():
        print(point, [(slot, tid, count) for (p, slot, tid), count in visits.items() if p == point])


def review(*, split_abrupt_ball_links=False, visual_exclusions=False, visual_recovery=False,
           destination_override=None, additional_ball_exclusions=(), event_footer=None):
    windows = scopes()
    records = common.read(OUT / 'player_court_proxies.csv')
    people = common.grouped(records)
    notes = common.load(OUT / 'visual_review_notes.json')
    stable = {int(p): slots for p, slots in notes['display_slot_map'].items()}
    # Compute guards on the provisional identities used in the display.
    for slots in stable.values():
        assert len({slots.get(slot, slot) for slot in common.SLOTS}) == 4
    named = {f: {common.NAMES[stable[r_point].get(r['player_slot'], r['player_slot'])]: r
                 for r in rows} for r_point, (_, first, last) in windows.items()
             for f in range(first, last + 1) if (rows := people[f])}
    reasons = display_guards(named, windows)
    for status in notes.get('position_withheld', []):
        name = status['player']
        for f in range(status['first_frame'], status['last_frame'] + 1):
            row = named.get(f, {}).get(name)
            if row:
                reasons.setdefault((f, row['player_slot']), status['reason'])
    offscreen = {}
    slots_by_name = {name: slot for slot, name in common.NAMES.items()}
    for run in notes.get('offscreen_reviewed_intervals', []):
        slot = slots_by_name[run['player']]
        for f in range(run['first_frame'], run['last_frame'] + 1):
            assert run['player'] not in named.get(f, {}), (f, run)
            offscreen[f, slot] = '画面外'
    ball_rows = common.read(BALL_PATH)
    ball = {int(r['source_frame']): int(r['current_candidate_present']) for r in ball_rows}
    trail_breaks = []
    destination = TRAIL_REVISION if split_abrupt_ball_links else OUT
    excluded_rows = {}
    replacements = {}
    by_frame = {int(row['source_frame']): row for row in ball_rows}
    if visual_exclusions:
        assert split_abrupt_ball_links
        destination = BALL_REVIEW_REVISION
        audit = common.load(VISUAL_EXCLUSIONS)
        for group in audit['groups']:
            evidence = VISUAL_EXCLUSIONS.parent / group['evidence']
            assert evidence.is_file()
            for f in group['frames']:
                row = by_frame[f]
                assert f not in excluded_rows and row['current_candidate_present'] == '1'
                assert group['point'] == int(row['point'])
                excluded_rows[f] = dict(point=group['point'], source_frame=f, reason=group['reason'],
                                        evidence=evidence.name,
                                        rejected_xy=[float(row[k]) for k in ('current_x_px', 'current_y_px')])
        assert audit['retained_counterexample']['source_frame'] not in excluded_rows
    if visual_recovery:
        assert visual_exclusions
        destination = RECOVERY_REVISION
        recovered = common.load(RECOVERY_REVIEW)
        candidate_rows = common.load(RECOVERY_REVIEW.parent / 'proposals.json')
        proposals = {(r['source_frame'], r['method']): r for r in candidate_rows if r['candidate_x_px'] != ''}
        assert {f for f, _ in proposals} == {r['source_frame'] for group in ('accepted', 'rejected', 'uncertain')
                                             for r in recovered[group]}
        for case in recovered['accepted']:
            f = case['source_frame']
            proposal = proposals[f, case['method']]
            assert f in excluded_rows
            excluded_rows.pop(f)
            replacements[f] = (proposal['candidate_x_px'], proposal['candidate_y_px'])
    if destination_override is not None:
        destination = destination_override
    if additional_ball_exclusions:
        assert split_abrupt_ball_links
    for row in additional_ball_exclusions:
        f = row['source_frame']
        candidate = by_frame[f]
        assert f not in excluded_rows and f not in replacements
        assert candidate['current_candidate_present'] == '1' and int(candidate['point']) == row['point']
        excluded_rows[f] = dict(row, rejected_xy=[float(candidate[k]) for k in ('current_x_px', 'current_y_px')])
    destination.mkdir(exist_ok=True)
    if split_abrupt_ball_links:
        display_rows = []
        for row in ball_rows:
            f = int(row['source_frame'])
            if f in excluded_rows:
                row = dict(row, current_candidate_present='0')
            elif f in replacements:
                x, y = replacements[f]
                row = dict(row, current_x_px=x, current_y_px=y)
            display_rows.append(row)
        for prev, cur in zip(display_rows, display_rows[1:]):
            if (prev['point'] != cur['point'] or int(cur['source_frame']) != int(prev['source_frame']) + 1
                    or prev['current_candidate_present'] != '1' or cur['current_candidate_present'] != '1'):
                continue
            a, b = ([float(r[k]) for k in ('current_x_px', 'current_y_px')] for r in (prev, cur))
            step = dist(a, b)
            if step > TRAIL_STEP_LIMIT_PX:
                trail_breaks.append(dict(point=int(cur['point']), source_frame=int(cur['source_frame']),
                                         prev_source_frame=int(prev['source_frame']), step_px=step,
                                         prev_xy=a, current_xy=b))
    counts, unavailable = [], []
    summary = []
    for case in cases():
        point = case['point']
        first, last = case['core_inclusive']
        for f in range(first, windows[point][2] + 1):
            rows = named.get(f, {})
            available = [name for name, row in rows.items() if (f, row['player_slot']) not in reasons]
            counts.append(dict(point=point, source_frame=f, core=f <= last, detected_slots=len(rows),
                               positions_available=sorted(available), ball_candidate=bool(ball[f])))
        span = [r for r in counts if r['point'] == point and r['core']]
        for slot in common.SLOTS:
            name = common.NAMES[slot]
            for visible, group in groupby(span, key=lambda r: name in r['positions_available']):
                if not visible:
                    seq = list(group)
                    unavailable.append(dict(point=point, player=name, first=seq[0]['source_frame'],
                                            last=seq[-1]['source_frame'], count=len(seq)))
        summary.append(dict(point=point, core_frames=len(span),
                            all_four_boxes_frames=sum(r['detected_slots'] == 4 for r in span),
                            all_four_positions_frames=sum(len(r['positions_available']) == 4 for r in span),
                            all_four_positions_and_ball_frames=sum(
                                len(r['positions_available']) == 4 and r['ball_candidate'] for r in span),
                            ball_candidate_frames=sum(r['ball_candidate'] for r in span),
                            terminal_context_review_end=windows[point][2]))
    common.dump(destination / 'core_summary.json', dict(
        points=summary, unavailable_runs=unavailable,
        limits='Availability only. IDs are visually spot-checked. Ball candidates have not been fully matched to raw evidence. '
               'The occlusion rule is transferred from the previous three points.',
        position_rules=notes,
    ))
    common.dump(destination / 'guarded_positions.json', [
        dict(source_frame=f, display_id=name, source_track_id=row['track_id'],
             court_xy=None if (f, row['player_slot']) in reasons
             else [float(row['court_proxy_x']), float(row['court_proxy_y'])],
             position_status=reasons.get((f, row['player_slot']), 'projected_box_bottom'))
        for f, rows in named.items() for name, row in rows.items()
    ])
    previews = set(notes['preview_frames'])
    if split_abrupt_ball_links:
        previews.update((18723, 18725))
    pilot.render(dict(destination=destination, routes={}, stable_names=stable, masked_positions=set(reasons),
                      guard_reasons=reasons, unavailable_reasons=offscreen, preview_frames=previews,
                      analysis_ends={p['point']: p['core_inclusive'][1] for p in cases()},
                      trail_break_frames={r['source_frame'] for r in trail_breaks},
                      suppressed_ball_frames=set(excluded_rows),
                      replacement_ball=replacements,
                      event_footer=event_footer,
                      id_legend='選手IDはウェア・靴で目視照合',
                      video_name='match2_points_4_6_player_ball.mp4'),
                 tracking_directory=OUT, ball_path=BALL_PATH, scopes=windows)
    if split_abrupt_ball_links:
        verify_trail(destination, trail_breaks, excluded_rows, replacements)
    print(summary, flush=True)


def verify_trail(destination, trail_breaks, excluded_rows, replacements):
    old_map, new_map = [common.read(p / 'video_frame_map.csv') for p in (OUT, destination)]
    asserted_keys = ('point', 'source_frame', 'player_slots', 'masked_player_positions')
    assert all(all(a[k] == b[k] for k in asserted_keys) for a, b in zip(old_map, new_map, strict=True))
    assert common.sha(OUT / 'guarded_positions.json') == common.sha(destination / 'guarded_positions.json')
    flags = {r['source_frame'] for r in trail_breaks}
    for prior, row in zip(old_map, new_map, strict=True):
        end, count = int(row['source_frame']), int(row['ball_trail_samples'])
        assert not flags.intersection(range(end - count + 2, end + 1))
        should_exclude = end in excluded_rows
        assert bool(int(row['ball_candidate_excluded'])) == should_exclude
        assert bool(int(row['ball_candidate_recovered'])) == (end in replacements)
        assert row['ball_candidate_present'] == ('0' if should_exclude else prior['ball_candidate_present'])
        if should_exclude:
            assert count == 0
    common.dump(destination / 'trail_audit.json', dict(
        max_draw_step_native_px=TRAIL_STEP_LIMIT_PX,
        clipped_links_by_point=dict(Counter(r['point'] for r in trail_breaks)),
        clipped_links=trail_breaks,
        displayed_frames_with_shorter_trail=sum(int(b['ball_trail_samples']) < int(a['ball_trail_samples'])
                                               for a, b in zip(old_map, new_map, strict=True)),
        ball_candidates_sha256=common.sha(BALL_PATH),
        visually_excluded_count=len(excluded_rows),
        visually_excluded_by_point=dict(Counter(r['point'] for r in excluded_rows.values())),
        visually_excluded_candidates=list(excluded_rows.values()),
        visually_reviewed_recoveries=[dict(source_frame=f, x_px=xy[0], y_px=xy[1])
                                      for f, xy in sorted(replacements.items())],
        scope='Image display only. Fixed native-pixel gate chosen after inspection of points 4–6. '
              'Selected exclusions and recoveries are visual decisions. Original prediction files were retained. '
              'No full-sequence accuracy or automatic exclusion claim.',
    ))


if __name__ == '__main__':
    {'track': track, 'inspect': inspect, 'review': review,
     'trail': lambda: review(split_abrupt_ball_links=True),
     'cleanup': lambda: review(split_abrupt_ball_links=True, visual_exclusions=True),
     'recovery': lambda: review(split_abrupt_ball_links=True, visual_exclusions=True,
                                visual_recovery=True)}[sys.argv[1]]()
