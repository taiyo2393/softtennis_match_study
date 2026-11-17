"""Transfer legacy image-region gates through court planes, not player positions."""
import itertools,math
import numpy as np
from confirmed_continuity_guard import old
from court_initialization_guard import eligible


class CameraGateMap:
    def __init__(self,new_image_to_court,legacy_image_to_court):
        self.new=np.asarray(new_image_to_court,float)
        self.transform=np.linalg.inv(np.asarray(legacy_image_to_court,float))@self.new
    def legacy_foot(self,row):
        p=self.transform@np.array([float(row['bottom_center_x']),float(row['bottom_center_y']),1.])
        if not np.isfinite(p).all() or abs(p[2])<1e-9:return None
        return p[:2]/p[2]
    def allowed_continuation(self,side,row):
        q=self.legacy_foot(row)
        if q is None:return False
        y=q[1]/2160
        return y>=.585 if side=='near' else .49<y<.665 if side=='far' else True


def calibrated_temporal_class(base,new_homography,legacy_homography):
    mapper=CameraGateMap(new_homography,legacy_homography)
    class CalibratedTemporal(base):
        def initialize(self,frame,rows,features):
            original=old.selector.seed_by_court_geometry
            byid={int(r['track_id']):feature for r,feature in zip(rows,features)}
            def selector(candidates,first,width,height):
                allowed=[r for r in candidates if eligible(r,mapper.new)]
                virtual=[];lookup={}
                for r in allowed:
                    p=mapper.legacy_foot(r)
                    if p is None:continue
                    v=dict(r,bottom_center_x=float(p[0]),bottom_center_y=float(p[1]));virtual.append(v);lookup[int(r['track_id'])]=r
                proposed=original(virtual,first,width,height)
                near=[v for v in virtual if float(v['bottom_center_y'])/height>=.61 and old.a.height(v)/height>=.08]
                if len(near)<=2:return proposed
                far=[lookup[proposed[s]] for s in ['far_rear','far_front']];options=[]
                for va,vb in itertools.combinations(near,2):
                    a,b=lookup[int(va['track_id'])],lookup[int(vb['track_id'])]
                    fa,fb=byid[int(a['track_id'])],byid[int(b['track_id'])]
                    if fa is None or fb is None:continue
                    selected=[a,b]+far
                    if any(old.overlap_fraction(x,y)>.45 for i,x in enumerate(selected) for y in selected[i+1:]):continue
                    distance=old.a.app_distance(fa,fb)
                    if math.isfinite(distance):options.append((distance,-(float(a['confidence'])+float(b['confidence']))/2,va,vb))
                options.sort(key=lambda v:v[:2])
                if not options:raise RuntimeError('No nonoverlapping near pair.')
                chosen=sorted(options[0][2:],key=lambda r:float(r['bottom_center_y']))
                return dict(proposed,near_front=int(chosen[0]['track_id']),near_rear=int(chosen[1]['track_id']))
            old.selector.seed_by_court_geometry=selector
            try:return super().initialize(frame,rows,features)
            finally:old.selector.seed_by_court_geometry=original

        def step(self,frame,rows,features):
            original_cost=old.previous.original_cost;original_recovery=old.recovery_cost
            def cost(state,row,feature,f,use_appearance=True):
                if not mapper.allowed_continuation(state['side'],row):return 1e6
                return original_cost(dict(state,side='camera_gate_checked'),row,feature,f,use_appearance)
            def recovery(state,row,feature,f):
                if not mapper.allowed_continuation(state['side'],row):return 1e6
                return original_recovery(dict(state,side='camera_gate_checked'),row,feature,f)
            old.previous.original_cost=cost;old.recovery_cost=recovery
            try:return super().step(frame,rows,features)
            finally:old.previous.original_cost=original_cost;old.recovery_cost=original_recovery
    return CalibratedTemporal
