"""Build a review video with stroke candidates and conservative foot-position display."""
from collections import Counter, defaultdict
from itertools import groupby
from math import dist

import build_044_spatiotemporal_pilot as common
import build_match2_player_ball_review as pilot
from review_match2_player_ball_display import IDENTITY, OCCLUSION


DEST = common.HERE / 'match2_first_three_player_ball_v3'
CONTACT_REVIEW = common.HERE / 'match2_raw_stroke_review/point3_contact_review.json'
FAR_REVIEW = common.HERE / 'match2_far_bounce_review/point2_ball_identity_review.json'
TIMING_REVIEW = common.HERE / 'match2_point1_contact_timing/review.json'
METRES = (10.97, 23.77)
FRAME_DISPLACEMENT_LIMIT_M = .45
MIN_VISIBLE_RUN = 3
STROKES = {
    1: [(12327, 12324, 12326, 'F1'), (12424, 12419, 12428, 'N1'), (12470, 12466, 12475, 'F2')],
    2: [(13657, 13647, 13656, 'F1'), (13731, 13728, 13734, 'N2'),
        (13761, 13754, 13766, 'F2'), (13855, 13851, 13860, 'N1'),
        (14019, 14016, 14022, 'F2')],
    3: [(15223, 15214, 15222, 'F2'), (15280, 15276, 15285, 'N1'),
        (15317, 15313, 15321, 'F1'), (15381, 15378, 15383, 'N1'),
        (15461, 15457, 15465, 'F1')],
}
# Approximate ball-ground contacts read from frame sequences, in native pixels.
BOUNCES = {
    (1, 1): (12411, 1212, 666), (1, 3): (12571, 1144, 838),
    (2, 1): (13712, 730, 652), (2, 5): (14074, 154, 752),
    (3, 1): (15250, 1345, 650), (3, 3): (15344, 1268, 705), (3, 5): (15477, 1014, 691),
}


def bounds(row):
    return tuple(float(row[key]) for key in ('x1', 'y1', 'x2', 'y2'))


def position_m(row):
    return tuple(float(row[key]) * scale for key, scale in
                 zip(('court_proxy_x', 'court_proxy_y'), METRES))


def foot_occluders(far, others):
    x0, y0, x1, y1 = bounds(far)
    width, height = x1 - x0, y1 - y0
    foot_left, foot_right = x0 + .10 * width, x1 - .10 * width
    foot_top, foot_bottom = y1 - .25 * height, y1 + .05 * height
    for name, other in others.items():
        if other is far:
            continue
        a, b, c, d = bounds(other)
        # A foreground box must extend below the far player's foot region.
        if d < y1 + max(10, .12 * height):
            continue
        overlap_w = min(foot_right, c + 2) - max(foot_left, a - 2)
        overlap_h = min(foot_bottom, d) - max(foot_top, b - 2)
        area = (foot_right - foot_left) * (foot_bottom - foot_top)
        if overlap_w > 0 and overlap_h > 0 and overlap_w * overlap_h / area >= .20:
            yield name


def display_guards(by_frame, scopes=pilot.SCOPES):
    reasons = {}
    for f, rows in by_frame.items():
        for name, row in rows.items():
            if not name.startswith('F'):
                continue
            key = f, row['player_slot']
            blockers = list(foot_occluders(row, rows))
            if blockers:
                reasons[key] = 'foot_overlap:' + '+'.join(blockers)
            elif key in OCCLUSION:
                reasons[key] = 'foot_hidden_visual_review'
            elif name == 'F2' and 15223 <= f <= 15234:
                # Native images show the server above the court following takeoff.
                reasons[key] = 'airborne_serve_visual_review'

    for point, (_, first, last) in scopes.items():
        for name in ('F1', 'F2'):
            clean = [(f, by_frame[f][name]) for f in range(first, last + 1) if name in by_frame[f]
                     and (f, by_frame[f][name]['player_slot']) not in reasons]
            # Split at missing samples and at implausible frame-to-frame changes.
            block_ids = []
            block = 0
            for i, (f, row) in enumerate(clean):
                if i:
                    previous, other = clean[i - 1]
                    if f != previous + 1:
                        block += 1
                    elif dist(position_m(row), position_m(other)) > FRAME_DISPLACEMENT_LIMIT_M:
                        reasons[previous, other['player_slot']] = 'unstable_foot_position'
                        reasons[f, row['player_slot']] = 'unstable_foot_position'
                        block += 1
                block_ids.append(block)
            for _, group in groupby(zip(block_ids, clean), key=lambda item: item[0]):
                sequence = [value for _, value in group]
                if len(sequence) < MIN_VISIBLE_RUN:
                    for f, row in sequence:
                        reasons.setdefault((f, row['player_slot']), 'insufficient_visible_run')
    return reasons


def build_routes(by_frame, reasons, homography, far_candidate=None, timing=None):
    route_groups, stroke_groups = {}, {}
    for point, candidates in STROKES.items():
        _, start, end = pilot.SCOPES[point]
        strokes, routes = [], []
        for n, (frame, low, high, name) in enumerate(candidates, 1):
            timing_updated = timing and (point, n) == (timing['point'], timing['stroke'])
            if timing_updated:
                assert [low, high] == timing['previous_candidate_interval'] and name == timing['hitter']
                low, high = timing['updated_candidate_interval']
                frame = timing['updated_reference_frame']
            eligible = []
            for f in range(max(start, frame - 14), min(end, frame + 14) + 1):
                row = by_frame[f].get(name)
                if row and (f, row['player_slot']) not in reasons:
                    eligible.append((abs(f - frame), f, row))
            position_frame, u, v = None, None, None
            if eligible:
                _, position_frame, row = min(eligible)
                u, v = float(row['court_proxy_x']), float(row['court_proxy_y'])
            strokes.append(dict(point=point, ordinal=n, reference_frame=frame, first_candidate_frame=low,
                                last_candidate_frame=high, hitter=name, position_frame=position_frame,
                                court_x=u, court_y=v, verification=(
                                    'refined visual candidate, exact impact unconfirmed' if timing_updated else
                                    'visual candidate, independent confirmation pending')))
        for n, stroke in enumerate(strokes, 1):
            after = strokes[n] if n < len(strokes) else None
            bounce = BOUNCES.get((point, n))
            if timing and (point, n) == (timing['point'], timing['previous_stroke_bounce']['stroke']):
                reviewed = timing['previous_stroke_bounce']
                assert bounce == (reviewed['frame'], *reviewed['before_image_px'])
                bounce = (reviewed['frame'], *reviewed['after_image_px'])
            is_reviewed_far_candidate = far_candidate and (point, n) == far_candidate['stroke_key']
            if is_reviewed_far_candidate:
                estimate = far_candidate['estimate']
                bounce = (estimate['frame'], estimate['x_px'], estimate['y_px'])
            pts = []
            if stroke['court_x'] is not None:
                pts.append([stroke['court_x'], stroke['court_y']])
            route = dict(id=f'P{point}_H{n:02d}', point=point, ordinal=n, first_frame=stroke['reference_frame'],
                         last_frame=after['reference_frame'] - 1 if after else end,
                         bounce_frame=None, bounce_image_x=None, bounce_image_y=None, bounce_point=None,
                         has_start=bool(pts), has_next_hit=bool(after and after['court_x'] is not None))
            if bounce:
                f, x, y = bounce
                xy = common.project(homography, 2 * x, 2 * y)
                route.update(bounce_frame=f, bounce_image_x=x, bounce_image_y=y, bounce_point=xy,
                             last_frame=max(route['last_frame'], f))
                pts.append(xy)
            if is_reviewed_far_candidate:
                route.update(bounce_evidence='conditional_fit_with_user_reviewed_ball_identity',
                             bounce_identity_review_frame=far_candidate['identity_review']['source_frame'],
                             bounce_contact_validated=False, bounce_in_out_validated=False,
                             bounce_fit_sensitivity=far_candidate['fit_sensitivity'])
            if route['has_next_hit']:
                pts.append([after['court_x'], after['court_y']])
            route['vertices'] = pts
            routes.append(route)
        route_groups[point], stroke_groups[point] = routes, strokes
    return route_groups, stroke_groups


def load_far_bounce_candidate():
    if not FAR_REVIEW.exists():
        return None
    review = common.load(FAR_REVIEW)
    result = common.load(FAR_REVIEW.with_name('result.json'))
    assert review['ball_identity'] == 'real_ball'
    assert not review['ground_contact_frame_reviewed'] and not review['in_out_reviewed']
    assert common.sha(FAR_REVIEW.parent / review['review_clip']) == review['review_clip_sha256']
    assert common.sha(pilot.BALL) == result['input_ball_path_sha256']
    mark = result['visual_streak']
    assert (review['source_frame'], *review['candidate_image_mark_px']) == (
        mark['frame'], mark['x_px'], mark['y_px'])
    return dict(stroke_key=(review['point'], review['stroke']), identity_review=review,
                estimate=result['conditional_model_best'],
                fit_sensitivity=result['sensitivity_within_one_image_px'])


def main():
    contact = common.load(CONTACT_REVIEW) if CONTACT_REVIEW.exists() else None
    far_candidate = load_far_bounce_candidate()
    timing = common.load(TIMING_REVIEW) if TIMING_REVIEW.exists() else None
    destination = DEST
    if contact:
        destination = DEST.with_name('match2_first_three_player_ball_v4')
    if far_candidate:
        destination = DEST.with_name('match2_first_three_player_ball_v5')
    if timing:
        destination = DEST.with_name('match2_first_three_player_ball_v6')
    if contact:
        assert common.sha(CONTACT_REVIEW.parent / contact['review_clip']) == contact['review_clip_sha256']
    destination.mkdir(exist_ok=True)
    people_path = pilot.OUT / 'player_court_proxies.csv'
    people = common.read(people_path)
    by_frame = defaultdict(dict)
    for row in people:
        point, f, slot = int(row['point']), int(row['source_frame']), row['player_slot']
        name = common.NAMES[IDENTITY[point].get(slot, slot)]
        by_frame[f][name] = row
    reasons = display_guards(by_frame)
    homography = common.load(pilot.PILOT / 'protocol.json')['image_to_court_homography_canonical']
    routes, strokes = build_routes(by_frame, reasons, homography, far_candidate, timing)
    for point, group in routes.items():
        assert all(len(route['vertices']) >= 2 for route in group), (point, group)
    common.write(destination / 'stroke_candidates.csv', [row for group in strokes.values() for row in group])
    common.dump(destination / 'schematic_routes.json', routes)
    if far_candidate:
        common.dump(destination / 'far_bounce_display_evidence.json', far_candidate)
    if timing:
        common.dump(destination / 'contact_timing_revision.json', timing)
    common.write(destination / 'guarded_player_positions.csv', [
        dict(row, display_id=name, position_status=reasons.get((f, row['player_slot']), 'proxy_shown'),
             shown_court_x='' if (f, row['player_slot']) in reasons else row['court_proxy_x'],
             shown_court_y='' if (f, row['player_slot']) in reasons else row['court_proxy_y'])
        for f, rows in by_frame.items() for name, row in rows.items()
    ])
    guard_audit = {}
    for point, (_, first, last) in pilot.SCOPES.items():
        counts = Counter()
        jumps_raw, jumps_displayed = [], []
        for f in range(first, last + 1):
            for name, row in by_frame[f].items():
                if not name.startswith('F'):
                    continue
                reason = reasons.get((f, row['player_slot']))
                counts[(reason or 'proxy_shown').split(':')[0]] += 1
                previous = by_frame[f - 1].get(name) if f > first else None
                if previous:
                    distance = dist(position_m(row), position_m(previous))
                    jumps_raw.append(distance)
                    if not reason and (f - 1, previous['player_slot']) not in reasons:
                        jumps_displayed.append(distance)
        guard_audit[point] = dict(counts=counts, raw_largest_adjacent_step_m=max(jumps_raw),
                                  shown_largest_adjacent_step_m=max(jumps_displayed))
    common.dump(destination / 'guard_audit.json', dict(
        original_tracking_sha256=common.sha(people_path), report=guard_audit,
        upper_bound_adjacent_display_step_m=FRAME_DISPLACEMENT_LIMIT_M,
        minimum_visible_run_frames=MIN_VISIBLE_RUN,
        scope='Display rule developed on these three rallies. Covered positions are missing, not ground truth.',
        point3_terminal_contact_review=contact,
        shot_uncertainty=('P3 N2 F15496–15500: user reviewed no contact.' if contact else
                          'P3 around F15496–15500 N2 racket contact is ambiguous; excluded from candidates.'),
    ))
    pilot.render(dict(destination=destination, routes=routes, stable_names=IDENTITY,
                      masked_positions=set(reasons), guard_reasons=reasons, strokes=strokes,
                      contact_review=contact))


if __name__ == '__main__':
    main()
