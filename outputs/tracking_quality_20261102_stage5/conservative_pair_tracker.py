"""Stricter policy for a pair whose initial color features are similar.

If appearance does not separate teammates, require short spatial continuity;
never automatically assign an old identity after a long missing interval.
"""
import copy
import numpy as np
from identity_guard import IdentityGuardTracker,old

PAIR_POLICY = dict(similar_pair_color_distance_max=.60,max_same_color_handover_gap=2,
                   short_handover_motion_max_height=.25,partner_motion_min_height=.75,
                   long_recovery_of_similar_pair='withhold_identity')


class ConservativePairTracker(IdentityGuardTracker):
    def __init__(self):
        super().__init__()
        self.similar_pairs = {}

    def step(self,frame,rows,features):
        before = copy.deepcopy(self.states)
        event_start = len(self.events)
        output = super().step(frame,rows,features)
        if not before and self.states:
            for side in ('near','far'):
                slots = [s for s in self.identity_anchors if s.startswith(side+'_')]
                distances = [old.a.app_distance(x,y) for x in self.identity_anchors[slots[0]]
                             for y in self.identity_anchors[slots[1]]]
                distance = float(np.median(distances))
                self.similar_pairs[side] = dict(distance=distance,
                    not_distinct_by_color=distance <= PAIR_POLICY['similar_pair_color_distance_max'])
            return output
        if not before:
            return output
        accepted,blocked = [],set()
        for i,j,margin,value,origin in output:
            slot = old.a.SLOTS[i]
            side = slot.split('_')[0]
            prior = before[slot]
            row = rows[j]
            gap = frame-prior['frame']
            changed = int(row['track_id']) != int(prior['row']['track_id'])
            reason = None
            if self.similar_pairs[side]['not_distinct_by_color']:
                if gap > PAIR_POLICY['max_same_color_handover_gap']:
                    reason = 'similar_pair_long_gap_identity_unresolved'
                elif changed:
                    prediction = old.a.foot(prior['row'])+prior['velocity']*gap
                    scale = max(old.a.height(row),old.a.height(prior['row']))
                    own_distance = float(np.linalg.norm(old.a.foot(row)-prediction))/scale
                    partner = next(s for s in before if s != slot and s.startswith(side+'_'))
                    other = before[partner]
                    other_gap = frame-other['frame']
                    other_prediction = old.a.foot(other['row'])+other['velocity']*min(other_gap,12)
                    other_distance = float(np.linalg.norm(old.a.foot(row)-other_prediction))/max(scale,old.a.height(other['row']))
                    if (own_distance > PAIR_POLICY['short_handover_motion_max_height']
                        or other_distance < PAIR_POLICY['partner_motion_min_height']):
                        reason = 'similar_pair_short_handover_spatially_ambiguous'
            if reason:
                blocked.add(slot)
                self.states[slot] = prior
                self.pending.pop(slot,None)
                self.guard_events.append(dict(event='handover_withheld_similar_pair',frame=frame,player_slot=slot,
                    proposed_raw_id=int(row['track_id']),last_raw_id=int(prior['row']['track_id']),
                    gap_frames=gap,reason=reason,identity_status='unresolved_not_assigned'))
            else:
                accepted.append((i,j,margin,value,origin))
        for event in self.events[event_start:]:
            if event.get('player_slot') in blocked and event['event'] == 'reacquired_candidate':
                event.update(event='reacquisition_withheld_similar_pair',identity_status='unresolved_not_assigned')
        return accepted
