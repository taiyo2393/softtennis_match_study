"""Adapt near-player size reference from accepted, continuous observations.

Keep the existing fragment rule, with a rolling size reference and a floor.
Do not adapt using missing values, anonymous detections or overlapping players.
"""
import sys
from collections import deque
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'tracking_quality_20261102_stage4'))
import temporal_tracker as old

PARAMETERS=dict(history_source_frames=30,near_reference_floor_fraction_of_image_height=.16,
                min_successive_height_ratio=.65,max_successive_height_ratio=1.55,
                max_update_motion_height=.40,overlap_fraction_max=.45,
                adaptation_requires_same_detector_id=True,interpolation=False)


def reference(heights):
    return max(PARAMETERS['near_reference_floor_fraction_of_image_height']*2160,float(np.median(heights)))


class AdaptiveNearTracker(old.TemporalTracker):
    def __init__(self):
        super().__init__();self.heights={};self.reference_log=[]

    def step(self,frame,rows,features):
        previous={s:dict(st) for s,st in self.states.items() if s.startswith('near_')}
        for s in previous:
            self.states[s]['reference_height']=reference(self.heights[s])
            self.reference_log.append(dict(source_frame=frame,player_slot=s,
                size_reference_height=self.states[s]['reference_height'],
                minimum_candidate_height=.5*self.states[s]['reference_height']))
        output=super().step(frame,rows,features)
        if not previous and self.states:
            ids=self.initial_window[0]['seeds']
            for s in ('near_front','near_rear'):
                samples=[old.a.height(item['byid'][ids[s]][0]) for item in self.initial_window]
                self.heights[s]=deque(samples,maxlen=PARAMETERS['history_source_frames'])
        elif previous:
            for i,j,margin,cost,origin in output:
                slot=old.a.SLOTS[i]
                if slot not in previous:continue
                prior=previous[slot];row=rows[j];h=old.a.height(row)
                gap=frame-prior['frame'];ratio=h/old.a.height(prior['row'])
                predicted=old.a.foot(prior['row'])+prior['velocity']*min(gap,12)
                residual=np.linalg.norm(old.a.foot(row)-predicted)/max(h,old.a.height(prior['row']))
                clear=all(old.overlap_fraction(row,rows[k])<PARAMETERS['overlap_fraction_max']
                          for ii,k,_,_,_ in output if ii!=i)
                if (int(row['track_id'])==int(prior['row']['track_id']) and gap<=2
                    and PARAMETERS['min_successive_height_ratio']<=ratio<=PARAMETERS['max_successive_height_ratio']
                    and residual<PARAMETERS['max_update_motion_height'] and clear):
                    self.heights[slot].append(h)
        return output
