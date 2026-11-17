"""Exploratory follow-up: confirm new IDs with ordinary continuity evidence.

Retains seed opponent negatives and all existing numeric constants.
"""
from automatic_reacquisition_guard import adaptive,old

def confirmed_class(base):
    class Confirmed(base):
        def __init__(self):
            super().__init__();self.normal_pending={};self.seed_opponents=None
        def step(self,frame,rows,features):
            for slot,st in self.states.items():st['_experimental_slot']=slot
            original=old.previous.continuity_cost
            confirmed=set()
            def guarded(state,row,feature,f,use_appearance=True):
                value=original(state,row,feature,f,use_appearance)
                if state['side']!='near' or int(row['track_id'])==int(state['row']['track_id']):return value
                slot=state['_experimental_slot'];key=(slot,int(row['track_id']))
                if value>=1.05 or feature is None:return 1e6
                own,_=old.best_appearance(state,feature)
                if self.seed_opponents and min(old.a.app_distance(n,feature) for n in self.seed_opponents)+old.PARAMETERS['recovery_negative_margin']<own:return 1e6
                previous=self.normal_pending.get(key)
                if previous and previous['last_frame']==frame:count=previous['count']
                else:count=previous['count']+1 if previous and previous['last_frame']==frame-1 else 1
                self.normal_pending[key]=dict(last_frame=frame,count=count)
                if count<old.PARAMETERS['recovery_source_frames']:return 1e6
                confirmed.add(key);return value
            old.previous.continuity_cost=guarded
            try:result=super().step(frame,rows,features)
            finally:old.previous.continuity_cost=original
            if self.states and self.seed_opponents is None:
                self.seed_opponents=[f.copy() for slot,st in self.states.items() if slot.startswith('far_') for f in st['gallery']]
                for slot,st in self.states.items():
                    if slot.startswith('near_'):st['negatives']=list(st['negatives'])+self.seed_opponents
            self.normal_pending={key:value for key,value in self.normal_pending.items() if value['last_frame']==frame}
            renamed=[]
            for i,j,margin,value,origin in result:
                if origin=='continuous_assignment' and (old.a.SLOTS[i],int(rows[j]['track_id'])) in confirmed:origin='confirmed_new_ID_continuity'
                renamed.append((i,j,margin,value,origin))
            return renamed
    return Confirmed
