"""Experimental near-ID switch guards, retaining existing thresholds.

No manual player IDs. No changes to seed selection or detector observations.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'tracking_quality_20261102_stage9'))
import adaptive_near as adaptive
old=adaptive.old

MODES=['confirm_new_id','confirm_new_id_opponent_negatives']

def guarded_class(base,mode):
    assert mode in MODES
    class Guarded(base):
        def __init__(self):
            super().__init__();self.guard_events=[];self.opponents_added=False
        def step(self,frame,rows,features):
            original=old.previous.continuity_cost
            def guarded_cost(state,row,feature,f,use_appearance=True):
                if state['side']=='near' and int(row['track_id'])!=int(state['row']['track_id']):
                    return 1e6
                return original(state,row,feature,f,use_appearance)
            old.previous.continuity_cost=guarded_cost
            try:result=super().step(frame,rows,features)
            finally:old.previous.continuity_cost=original
            if self.states and not self.opponents_added:
                # Seed-only features: no future frames, AI ID mapping or outcome labels.
                opponents=[f.copy() for s,st in self.states.items() if s.startswith('far_') for f in st['gallery']]
                if mode=='confirm_new_id_opponent_negatives':
                    for s,st in self.states.items():
                        if s.startswith('near_'):st['negatives']=list(st['negatives'])+opponents
                self.opponents_added=True
                self.guard_events.append(dict(event='guard_initialized',frame=frame,mode=mode,opponent_seed_features=len(opponents) if mode.endswith('opponent_negatives') else 0))
            return result
    return Guarded
