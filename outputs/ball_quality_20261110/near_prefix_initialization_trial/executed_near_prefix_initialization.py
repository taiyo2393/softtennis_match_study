"""Experimental near-pair prefix; native boxes, past-only, no new-ID recovery.

This supplements the waiting period of an unchanged four-seed pipeline. It does
not independently initialize the far pair, nor backfill its confirmation window.
"""
import itertools
from collections import deque
import numpy as np
from court_initialization_guard import eligible
from confirmed_continuity_guard import old, adaptive

SLOTS = ('near_front', 'near_rear')
PARAMETERS = dict(confirm_source_frames=old.PARAMETERS['initialization_source_frames'],
                  maximum_overlap=old.PARAMETERS['occlusion_intersection_fraction'],
                  maximum_continuation_gap=2, missing_cost=1.05,
                  raw_ID_changes_allowed=False, gallery_updates=False,
                  interpolation=False, backfill=False)


class NearPrefix:
    def __init__(self, homography, adaptive_size=True):
        self.homography = homography
        self.adaptive_size = adaptive_size
        self.window = []
        self.states = {}
        self.events = []
        self.candidate_log = []

    def propose(self, frame, rows, features):
        byid = {int(r['track_id']): (r, f) for r, f in zip(rows, features)}
        assert len(byid) == len(rows), 'Duplicate raw IDs need upstream resolution.'
        near = [r for r in rows if eligible(r, self.homography)
                and float(r['bottom_center_y']) / 2160 >= .61
                and old.a.height(r) / 2160 >= .08]
        options = []
        for a, b in itertools.combinations(near, 2):
            fa, fb = byid[int(a['track_id'])][1], byid[int(b['track_id'])][1]
            if fa is None or fb is None or old.overlap_fraction(a, b) > PARAMETERS['maximum_overlap']:
                continue
            options.append((old.a.app_distance(fa, fb),
                            -(float(a['confidence']) + float(b['confidence'])) / 2, a, b))
        # Pair appearance ranks only when >2 candidates, as in the prior policy.
        options.sort(key=lambda item: item[:2])
        chosen = sorted(options[0][2:], key=lambda r: float(r['bottom_center_y'])) if options else []
        self.candidate_log.append(dict(source_frame=frame, near_candidates=len(near),
                                       selected_raw_ids=[int(r['track_id']) for r in chosen]))
        return dict(zip(SLOTS, (int(r['track_id']) for r in chosen))) if chosen else None, byid

    def step(self, frame, rows, features):
        if not self.states:
            proposed, byid = self.propose(frame, rows, features)
            if proposed is None:
                self.window.clear()
                return []
            if self.window and (self.window[-1]['frame'] != frame - 1 or
                                set(self.window[-1]['seeds'].values()) != set(proposed.values())):
                self.window.clear()
            self.window.append(dict(frame=frame, seeds=proposed, byid=byid))
            if len(self.window) < PARAMETERS['confirm_source_frames']:
                return []
            seeds = self.window[0]['seeds']
            chosen = [byid[t][0] for t in seeds.values()]
            negatives = [f for r, f in zip(rows, features) if f is not None
                         and int(r['track_id']) not in seeds.values()
                         and all(old.overlap_fraction(r, s) < .1 for s in chosen)]
            for slot, tid in seeds.items():
                gallery = [w['byid'][tid][1] for w in self.window]
                heights = [old.a.height(w['byid'][tid][0]) for w in self.window]
                self.states[slot] = dict(row=byid[tid][0], frame=frame, velocity=np.zeros(2),
                    side='near', gallery=gallery, negatives=negatives,
                    reference_height=float(np.median(heights)),
                    heights=deque(heights, maxlen=adaptive.PARAMETERS['history_source_frames']))
            self.events.append(dict(event='near_prefix_initialized', frame=frame,
                window_start=self.window[0]['frame'], seed_ids=seeds,
                identity_status='development_candidate_not_GT'))
            return [(slot, byid[tid][0], 0., 0.) for slot, tid in seeds.items()]

        costs = np.full((2, len(rows)), 1e6)
        for i, slot in enumerate(SLOTS):
            state = self.states[slot]
            if self.adaptive_size:
                state['reference_height'] = adaptive.reference(state['heights'])
            if not 0 < frame - state['frame'] <= PARAMETERS['maximum_continuation_gap']:
                continue
            for j, (row, feature) in enumerate(zip(rows, features)):
                if int(row['track_id']) != int(state['row']['track_id']) or feature is None:
                    continue
                _, template = old.best_appearance(state, feature)
                costs[i, j] = old.previous.continuity_cost(dict(state, template=template), row, feature, frame)
        assignments = old.a.joint_assignment(costs, missing_cost=PARAMETERS['missing_cost'])
        # Never export overlapping near-pair boxes even if both IDs persist.
        if len(assignments) == 2 and old.overlap_fraction(rows[assignments[0][1]], rows[assignments[1][1]]) > PARAMETERS['maximum_overlap']:
            return []
        output = []
        for i, j, margin, value in assignments:
            state = self.states[SLOTS[i]]
            row = rows[j]
            gap = frame - state['frame']
            height = old.a.height(row)
            ratio = height / old.a.height(state['row'])
            predicted = old.a.foot(state['row']) + state['velocity'] * min(gap, 12)
            residual = np.linalg.norm(old.a.foot(row) - predicted) / max(height, old.a.height(state['row']))
            clear = all(old.overlap_fraction(row, rows[k]) < PARAMETERS['maximum_overlap']
                        for _, k, _, _ in assignments if k != j)
            if (self.adaptive_size and .65 <= ratio <= 1.55 and residual < .40 and clear):
                state['heights'].append(height)
            velocity = (old.a.foot(row) - old.a.foot(state['row'])) / gap
            speed = np.linalg.norm(velocity)
            if speed > .10 * height:
                velocity *= .10 * height / speed
            state['velocity'] = .75 * state['velocity'] + .25 * velocity
            state.update(row=row, frame=frame)
            output.append((SLOTS[i], row, margin, value))
        return output
