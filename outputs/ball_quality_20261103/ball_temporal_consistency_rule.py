"""One fixed two-anchor contradiction test; no coordinate interpolation.

The bound is a development heuristic, never a physical speed limit or label.
Absence of anchors is absence of evidence, not evidence of a wrong object.
"""
import math

WINDOW = 8
MARGIN = 16.
RATE = 64. / 2  # Reuse the old 64px per2source-frame displacement scale.
MIN_ANCHOR_SEPARATION = 2


def xy(p):
    return float(p['x_px']), float(p['y_px'])


def decide(t, point, original_present, anchors, camera_mask):
    """anchors contains only immutable original temporal observations, not additions."""
    info = dict(keep=True, reason='no_output_to_test', anchor_a='', anchor_b='',
        pair_distance_px='', pair_bound_px='', distance_a_px='', distance_b_px='',
        bound_a_px='', bound_b_px='')
    if point is None:
        return info
    before = sorted((j for j in anchors if t - WINDOW <= j < t), reverse=True)
    after = sorted(j for j in anchors if t < j <= t + WINDOW)
    pair = None
    if before and after:
        pair = before[0], after[0]
    elif not original_present:
        side = before if before else after
        if side:
            second = next((j for j in side[1:] if abs(j - side[0]) >= MIN_ANCHOR_SEPARATION), None)
            if second is not None:
                pair = side[0], second
    if pair is None:
        info['reason'] = 'keep_insufficient_anchor_evidence'
        return info
    a, b = pair
    info.update(anchor_a=a, anchor_b=b)
    if abs(a - b) < MIN_ANCHOR_SEPARATION:
        info['reason'] = 'keep_anchors_too_close_in_time'
        return info
    if camera_mask[max(0, min(a, b, t)):max(a, b, t) + 1].any():
        info['reason'] = 'keep_camera_transition_evidence_ambiguous'
        return info
    pair_distance = math.dist(xy(anchors[a]), xy(anchors[b]))
    pair_bound = MARGIN + RATE * abs(b - a)
    da, db = math.dist(xy(point), xy(anchors[a])), math.dist(xy(point), xy(anchors[b]))
    ba, bb = MARGIN + RATE * abs(t - a), MARGIN + RATE * abs(t - b)
    info.update(pair_distance_px=pair_distance, pair_bound_px=pair_bound,
        distance_a_px=da, distance_b_px=db, bound_a_px=ba, bound_b_px=bb)
    if pair_distance <= pair_bound and da > ba and db > bb:
        info.update(keep=False, reason='withhold_two_anchor_contradiction')
    elif pair_distance > pair_bound:
        info['reason'] = 'keep_anchors_not_mutually_compatible'
    else:
        info['reason'] = 'keep_not_contradicted_by_both_anchors'
    return info


def self_test():
    import numpy as np
    mask = np.zeros(50, dtype=bool)
    p = lambda x: dict(x_px=x, y_px=0)
    assert not decide(20, p(150), True, {19: p(0), 21: p(0)}, mask)['keep']
    # Fast genuine constant motion is not rejected merely for exceeding RATE.
    assert decide(20, p(0), True, {19: p(-80), 21: p(80)}, mask)['keep']
    assert not decide(20, p(0), False, {24: p(940), 28: p(950)}, mask)['keep']
    assert decide(20, p(0), False, {}, mask)['keep']
    assert decide(20, p(0), True, {24: p(940), 28: p(950)}, mask)['keep']
    assert decide(20, p(0), False, {21: p(940)}, mask)['keep']
    mask[22] = True
    assert decide(20, p(0), False, {24: p(940), 28: p(950)}, mask)['keep']
    print('Passed7synthetic cases: contradiction,fast motion,one-sided new candidate,insufficient evidence,existing one-sided case,camera cut. No research labels written.')


if __name__ == '__main__':
    self_test()
