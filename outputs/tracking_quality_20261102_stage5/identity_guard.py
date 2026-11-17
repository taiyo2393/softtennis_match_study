"""Abstain on same-side identity handovers not distinguished from the partner.

The guard fixes overconfident handovers; it does not solve same-uniform ReID.
Do not change retained stage4 code, output, or human labels.
"""
import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'tracking_quality_20261102_stage4'))
import temporal_tracker as old

GUARD_PARAMETERS = dict(partner_appearance_margin=.12,
                        compare_all_same_side_identities=True,
                        immutable_initial_identity_anchors=True,
                        suppress_handover_when_ambiguous=True)


def identity_separation(anchors,slot,feature):
    own = min(old.a.app_distance(f,feature) for f in anchors[slot])
    partners = [s for s in anchors if s != slot and s.split('_')[0] == slot.split('_')[0]]
    rival = min((old.a.app_distance(f,feature) for s in partners for f in anchors[s]),default=1.)
    return rival-own,own,rival


class IdentityGuardTracker(old.TemporalTracker):
    instances = []

    def __init__(self):
        super().__init__()
        self.identity_anchors = {}
        self.guard_events = []
        self.instances.append(self)

    def step(self,frame,rows,features):
        before = copy.deepcopy(self.states)
        event_start = len(self.events)
        output = super().step(frame,rows,features)
        if not before and self.states:
            # Retain immutable reference images, independently of adaptive gallery.
            self.identity_anchors = {s:tuple(f.copy() for f in st['gallery']) for s,st in self.states.items()}
            self.events[-1]['identity_scope'] = 'clip_local_initial_anchor_not_global_player_name'
            return output
        if not before:
            return output
        blocked = set()
        accepted = []
        for i,j,margin,value,origin in output:
            slot = old.a.SLOTS[i]
            prior = before[slot]
            row = rows[j]
            handover = (int(row['track_id']) != int(prior['row']['track_id'])
                        or frame-prior['frame'] > 2
                        or origin == 'gallery_reacquisition_candidate')
            if handover:
                separation,own,rival = identity_separation(self.identity_anchors,slot,features[j])
                if separation < GUARD_PARAMETERS['partner_appearance_margin']:
                    blocked.add(slot)
                    self.states[slot] = prior
                    self.pending.pop(slot,None)
                    event = dict(event='handover_withheld_partner_ambiguous',frame=frame,player_slot=slot,
                                 proposed_raw_id=int(row['track_id']),last_raw_id=int(prior['row']['track_id']),
                                 proposed_origin=origin,own_appearance_distance=own,
                                 partner_appearance_distance=rival,partner_separation=separation,
                                 identity_status='unresolved_not_assigned')
                    self.guard_events.append(event)
                    continue
            accepted.append((i,j,margin,value,origin))
        # A rejected proposal must not survive as a successful recovery event.
        for event in self.events[event_start:]:
            if event.get('player_slot') in blocked and event['event'] == 'reacquired_candidate':
                event.update(event='reacquisition_withheld_partner_ambiguous',identity_status='unresolved_not_assigned')
        return accepted
