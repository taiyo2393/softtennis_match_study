"""Temporal initialization and conservative gallery-based reacquisition.

All emitted coordinates come from actual detector boxes at that frame.
Confirmation does not fill the preceding gap. Appearance is a color cue,
not trained ReID, and reacquisitions remain unreviewed identity proposals.
"""
import importlib.util
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'tracking_quality_20261102_stage2'))
import improve_players as previous
a = previous.assoc
spec = importlib.util.spec_from_file_location('seed_selector',a.SOURCE/'scripts/96_select_four_players_pilot.py')
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)

PARAMETERS = dict(initialization_source_frames=8,recovery_source_frames=8,min_recovery_gap=6,
                  recovery_appearance_max=.65,recovery_negative_margin=.06,
                  recovery_assignment_margin=.18,recovery_missing_cost=.85,
                  gallery_size=8,gallery_sample_interval=12,gallery_cost_max=.55,
                  gallery_margin_min=.20,occlusion_intersection_fraction=.45,
                  pending_iou_min=.30,pending_motion_max_height=.35,
                  interpolation=False)


def overlap_fraction(x,y):
    x1,y1,x2,y2 = [float(x[k]) for k in ('x1','y1','x2','y2')]
    u1,v1,u2,v2 = [float(y[k]) for k in ('x1','y1','x2','y2')]
    inter = max(0,min(x2,u2)-max(x1,u1))*max(0,min(y2,v2)-max(y1,v1))
    return inter/max(1,min((x2-x1)*(y2-y1),(u2-u1)*(v2-v1)))


def best_appearance(state,feature):
    choices = [(a.app_distance(f,feature),f) for f in state['gallery']]
    return min(choices,key=lambda item:item[0])


def recovery_cost(state,row,feature,frame):
    gap = frame-state['frame']
    if gap < PARAMETERS['min_recovery_gap'] or feature is None:
        return 1e6
    h = a.height(row)
    if h < .50*state['reference_height'] or not .45 <= h/state['reference_height'] <= 2.5:
        return 1e6
    y = a.foot(row)[1]/2160
    if state['side'] == 'near' and y < .585:
        return 1e6
    if state['side'] == 'far' and not .49 < y < .665:
        return 1e6
    app,_ = best_appearance(state,feature)
    if app > PARAMETERS['recovery_appearance_max']:
        return 1e6
    if state['negatives']:
        neg = min(a.app_distance(n,feature) for n in state['negatives'])
        if neg+PARAMETERS['recovery_negative_margin'] < app:
            return 1e6
    # Beyond a short gap, do not assume constant velocity through occlusion.
    distance = float(np.linalg.norm(a.foot(row)-a.foot(state['row'])))/max(h,a.height(state['row']))
    if distance > min(8.,.8+.08*gap):
        return 1e6
    return .70*app+.15*abs(math.log(h/state['reference_height']))+.15*min(distance/8.,1.)


class TemporalTracker:
    def __init__(self):
        self.states = {}
        self.initial_window = []
        self.pending = {}
        self.events = []
        self.initialized_frame = None

    def initialize(self,frame,rows,features):
        try:
            proposed = selector.seed_by_court_geometry(rows,frame,3840,2160)
        except RuntimeError:
            self.initial_window = []
            return []
        byid = {int(r['track_id']):(r,f) for r,f in zip(rows,features)}
        selected = [byid[proposed[s]][0] for s in a.SLOTS]
        # A detected fragment behind another person is a poor initial template.
        if any(overlap_fraction(x,y) > PARAMETERS['occlusion_intersection_fraction']
               for i,x in enumerate(selected) for y in selected[i+1:]):
            self.initial_window = []
            return []
        ids = set(proposed.values())
        if self.initial_window and (self.initial_window[-1]['frame'] != frame-1
                                    or set(self.initial_window[-1]['seeds'].values()) != ids):
            self.initial_window = []
        self.initial_window.append(dict(frame=frame,seeds=proposed,byid=byid))
        if len(self.initial_window) < PARAMETERS['initialization_source_frames']:
            return []
        initial = self.initial_window[0]['seeds']
        negatives = [f for r,f in zip(rows,features) if int(r['track_id']) not in ids and f is not None
                     and all(overlap_fraction(r,s) < .10 for s in selected)]
        assignments = []
        for i,slot in enumerate(a.SLOTS):
            tid = initial[slot]
            row,feature = byid[tid]
            gallery = [item['byid'][tid][1] for item in self.initial_window if item['byid'][tid][1] is not None]
            if not gallery:
                self.initial_window = []
                self.states = {}
                return []
            self.states[slot] = dict(row=row,frame=frame,velocity=np.zeros(2),side=slot.split('_')[0],
                template=feature,gallery=deque(gallery,maxlen=PARAMETERS['gallery_size']),negatives=negatives,
                reference_height=float(np.median([a.height(item['byid'][tid][0]) for item in self.initial_window])),
                last_gallery_frame=frame)
            j = next(j for j,r in enumerate(rows) if int(r['track_id']) == tid)
            assignments.append((i,j,0.,0.,'temporal_initialization'))
        self.initialized_frame = frame
        self.events.append(dict(event='initialized',frame=frame,
                                window_start=self.initial_window[0]['frame'],seed_ids=initial,
                                identity_status='candidate_not_ground_truth'))
        return assignments

    def step(self,frame,rows,features):
        if not self.states:
            return self.initialize(frame,rows,features)
        costs = []
        for slot in a.SLOTS:
            state = self.states[slot]
            line = []
            for row,feature in zip(rows,features):
                _,template = best_appearance(state,feature)
                line.append(previous.continuity_cost(dict(state,template=template),row,feature,frame))
            costs.append(line)
        normal = a.joint_assignment(np.array(costs).reshape(4,len(rows)))
        assignments = [(i,j,margin,cost,'continuous_assignment') for i,j,margin,cost in normal]
        occupied = {j for _,j,_,_,_ in assignments}
        assigned_slots = {i for i,_,_,_,_ in assignments}
        missing = [i for i in range(4) if i not in assigned_slots]
        available = [j for j,r in enumerate(rows) if j not in occupied and
                     all(overlap_fraction(r,rows[k]) < PARAMETERS['occlusion_intersection_fraction'] for k in occupied)]
        if missing and available:
            recovery = np.array([[recovery_cost(self.states[a.SLOTS[i]],rows[j],features[j],frame)
                                  for j in available] for i in missing])
            proposals = a.joint_assignment(recovery,missing_cost=PARAMETERS['recovery_missing_cost'])
            proposed_slots = set()
            for ii,jj,margin,value in proposals:
                i,j = missing[ii],available[jj]
                if margin < PARAMETERS['recovery_assignment_margin']:
                    continue
                slot = a.SLOTS[i]
                proposed_slots.add(slot)
                tid = int(rows[j]['track_id'])
                old = self.pending.get(slot)
                coherent = old and (old['raw_id'] == tid or (
                    a.box_iou(old['row'],rows[j]) >= PARAMETERS['pending_iou_min'] and
                    np.linalg.norm(a.foot(old['row'])-a.foot(rows[j])) < PARAMETERS['pending_motion_max_height']*a.height(rows[j])))
                if old and old['last_frame'] == frame-1 and coherent:
                    count = old['count']+1
                    first = old['first_frame']
                    chain = old['raw_ids']+[tid] if old['raw_ids'][-1] != tid else old['raw_ids']
                else:
                    count,first = 1,frame
                    chain = [tid]
                self.pending[slot] = dict(last_frame=frame,first_frame=first,count=count,raw_id=tid,
                                          raw_ids=chain,row=rows[j])
                if count >= PARAMETERS['recovery_source_frames']:
                    assignments.append((i,j,margin,value,'gallery_reacquisition_candidate'))
                    self.events.append(dict(event='reacquired_candidate',frame=frame,player_slot=slot,
                        last_observed_frame=self.states[slot]['frame'],old_raw_id=int(self.states[slot]['row']['track_id']),
                        new_raw_id=tid,confirmation_start=first,confirmation_frames=count,
                        confirmation_raw_ids=chain,
                        alternative_cost_margin=margin,identity_status='requires_identity_review'))
                    self.pending.pop(slot,None)
            for slot in list(self.pending):
                if slot not in proposed_slots:
                    self.pending.pop(slot,None)
        else:
            self.pending.clear()
        for i,j,margin,value,origin in assignments:
            slot = a.SLOTS[i]
            state = self.states[slot]
            row = rows[j]
            gap = frame-state['frame']
            same = int(row['track_id']) == int(state['row']['track_id'])
            if origin == 'gallery_reacquisition_candidate':
                state['velocity'] = np.zeros(2)
            elif gap > 0:
                velocity = (a.foot(row)-a.foot(state['row']))/gap
                speed = np.linalg.norm(velocity)
                if speed > .10*a.height(row):
                    velocity *= .10*a.height(row)/speed
                state['velocity'] = .75*state['velocity']+.25*velocity
            clear = all(overlap_fraction(row,rows[k]) < PARAMETERS['occlusion_intersection_fraction']
                        for k in range(len(rows)) if k != j and int(rows[k]['track_id']) != int(row['track_id'])
                        and a.height(rows[k]) >= .5*a.height(row))
            if (origin == 'continuous_assignment' and same and gap <= 2 and clear
                    and value < PARAMETERS['gallery_cost_max'] and margin >= PARAMETERS['gallery_margin_min']
                    and frame-state['last_gallery_frame'] >= PARAMETERS['gallery_sample_interval']):
                state['gallery'].append(features[j])
                state['last_gallery_frame'] = frame
            state.update(row=row,frame=frame)
            self.pending.pop(slot,None)
        return assignments
