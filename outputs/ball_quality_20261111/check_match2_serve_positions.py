"""Extend the three original tracking windows backwards over serve candidates."""
import sys
from collections import defaultdict
from math import dist

import cv2
import numpy as np

import build_044_spatiotemporal_pilot as common
import build_match2_player_ball_review as pilot
from review_match2_player_ball_display import IDENTITY
from review_match2_stroke_routes import FRAME_DISPLACEMENT_LIMIT_M, foot_occluders, position_m


BASE = common.HERE / 'match2_first_three_player_ball_v4'
OUT = common.HERE / 'match2_serve_position_trial'
PREVIOUS_AUDIT = common.HERE / 'match2_position_gap_trial/stroke_position_coverage.json'


def ranges():
    serves = [r for r in common.read(BASE / 'stroke_candidates.csv') if r['ordinal'] == '1']
    assert len(serves) == len(pilot.SCOPES)
    return {int(r['point']): (int(r['first_candidate_frame']), int(r['last_candidate_frame']))
            for r in serves}


def track():
    scopes = {}
    for point, (first, last) in ranges().items():
        preroll, original_start, original_end = pilot.SCOPES[point]
        assert preroll <= first <= last < original_start and original_start + 8 <= original_end
        scopes[point] = (preroll, first - 8, original_start + 8)
    pilot.track_players(OUT, scopes)


def index(path):
    frames = defaultdict(dict)
    for row in common.read(path):
        point, frame = int(row['point']), int(row['source_frame'])
        slot = row['player_slot']
        name = common.NAMES[IDENTITY[point].get(slot, slot)]
        assert name not in frames[point, frame]
        frames[point, frame][name] = row
    return frames


def continuity_report(new):
    original = index(pilot.OUT / 'player_court_proxies.csv')
    availability = {(int(r['point']), int(r['source_frame'])): int(r['assigned_slots'])
                    for r in common.read(pilot.OUT / 'player_frame_availability.csv')}
    frozen_fields = ('track_id', 'segment', 'x1', 'y1', 'x2', 'y2', 'court_proxy_x', 'court_proxy_y')
    comparison = []
    for point, (first, last) in ranges().items():
        _, start, _ = pilot.SCOPES[point]
        overlap = range(start, start + 9)
        for frame in overlap:
            a, b = original[point, frame], new[point, frame]
            assert a.keys() == b.keys()
            assert all(all(a[name][field] == b[name][field] for field in frozen_fields) for name in a)
        assert all(availability[point, f] == len(new[point, f]) for f in range(first, last + 1))
        comparison.append(dict(point=point, original_overlap_frames=len(overlap),
                               identical_boxes_ids_proxies=True,
                               candidate_counts_match_original_preroll=True))
    return comparison


def make_review_images(by_frame, protocol):
    capture = cv2.VideoCapture(protocol['source_video'])
    assert capture.isOpened()
    for point, (first, last) in ranges().items():
        images = []
        frames = sorted({first, (first + last) // 2, last, last + 1})
        assert capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        for frame in frames:
            while round(capture.get(cv2.CAP_PROP_POS_FRAMES)) < frame:
                assert capture.grab()
            ok, raw = capture.read()
            assert ok and round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == frame + 1
            crop = raw[380:840].copy()
            annotated = crop.copy()
            for name, row in by_frame[point, frame].items():
                color = next(color for slot, color in common.RGB.items() if common.NAMES[slot] == name)[::-1]
                x0, y0, x1, y1 = [round(float(row[k])) for k in ('x1', 'y1', 'x2', 'y2')]
                cv2.rectangle(annotated, (x0, y0 - 380), (x1, y1 - 380), color, 2)
                if not (point == 3 and name == 'F2'):
                    cv2.circle(annotated, ((x0 + x1) // 2, y1 - 380), 6, color, -1)
                cv2.putText(annotated, name, (x0, y0 - 386), cv2.FONT_HERSHEY_SIMPLEX, .75, color, 2)
            tile = np.hstack([cv2.resize(im, (960, 230), interpolation=cv2.INTER_AREA)
                              for im in (crop, annotated)])
            cv2.putText(tile, f'F{frame}', (15, 220), cv2.FONT_HERSHEY_SIMPLEX, .72, (255, 255, 255), 2)
            images.append(tile)
        assert cv2.imwrite(str(OUT / f'point{point}_serve_review.jpg'), np.vstack(images))
    capture.release()


def inspect():
    protocol = common.load(OUT / 'protocol.json')
    new = index(OUT / 'player_court_proxies.csv')
    checks = continuity_report(new)
    make_review_images(new, protocol)
    common.dump(OUT / 'continuity_verification.json', checks)


def evaluate():
    by_frame = index(OUT / 'player_court_proxies.csv')
    comparison = continuity_report(by_frame)
    previous = common.load(PREVIOUS_AUDIT)
    contact_positions, per_serve = [], []
    for point, (first, last) in ranges().items():
        frames = []
        for frame in range(first, last + 1):
            rows = by_frame[point, frame]
            shown = set()
            for name, row in rows.items():
                position_status = 'available_box_bottom_proxy'
                prior = by_frame[point, frame - 1].get(name)
                if point == 3 and name == 'F2':
                    position_status = 'withheld_airborne_server_visual_review'
                elif name.startswith('F') and any(foot_occluders(row, rows)):
                    position_status = 'withheld_foot_overlap'
                elif prior and dist(position_m(prior), position_m(row)) > FRAME_DISPLACEMENT_LIMIT_M:
                    position_status = 'withheld_unstable_proxy'
                else:
                    shown.add(name)
                contact_positions.append(dict(
                    point=point, source_frame=frame, player=name, status=position_status,
                    shown_court_xy=[float(row['court_proxy_x']), float(row['court_proxy_y'])]
                    if name in shown else None,
                ))
            frames.append(dict(source_frame=frame, boxes=len(rows), available_positions=sorted(shown)))
        per_serve.append(dict(
            point=point, first_candidate_frame=first, last_candidate_frame=last,
            frames=len(frames), all_four_boxes_frames=sum(row['boxes'] == 4 for row in frames),
            all_four_positions_frames=sum(len(row['available_positions']) == 4 for row in frames),
            receiver_pair_positions_frames=sum({'N1', 'N2'} <= set(row['available_positions']) for row in frames),
            availability=frames,
        ))
    n_before = sum(r['candidate_contact_frames'] == r['contact_frames_with_four_positions'] for r in previous)
    updated = []
    for row in previous:
        if row['stroke'] == 1:
            serve = per_serve[row['point'] - 1]
            row = dict(row, contact_frames_inside_tracking=serve['frames'],
                       contact_frames_with_four_positions=serve['all_four_positions_frames'])
        updated.append(row)
    common.dump(OUT / 'contact_positions.json', contact_positions)
    common.dump(OUT / 'stroke_position_coverage.json', updated)
    common.dump(OUT / 'result.json', dict(
        serves=per_serve, continuity_checks=comparison, total_stroke_candidates=len(updated),
        four_positions_whole_contact_window_before=n_before,
        four_positions_whole_contact_window_after=sum(
            r['candidate_contact_frames'] == r['contact_frames_with_four_positions'] for r in updated),
        guard_note='Point 3 F2 is withheld throughout the serve candidate window based on visual jump review. '
                   'A projected bottom point in the air can exaggerate the court depth.',
        validity='Coverage of box-bottom proxies, with approximate visually identified contact windows. '
                 'No new independent ID, foot-position or impact-time reference.',
    ))
    print([(s['point'], s['frames'], s['all_four_boxes_frames'],
            s['all_four_positions_frames'], s['receiver_pair_positions_frames']) for s in per_serve])


if __name__ == '__main__':
    {'track': track, 'inspect': inspect, 'evaluate': evaluate}[sys.argv[1]]()
