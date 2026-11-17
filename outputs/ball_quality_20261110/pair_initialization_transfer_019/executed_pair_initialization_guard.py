"""Development-only clothing-pair ranking before temporal initialization.

Same-uniform near teammates are an explicit assumption, not player identity GT.
No person IDs, absolute frame IDs, outcomes, or later frames enter decisions.
"""
import itertools,math
from court_initialization_guard import eligible,old

PARAMETERS=dict(initialization_only=True,rank_only_when_near_candidates_exceed_two=True,
    absolute_interplayer_appearance_cutoff=None,
    ambiguity_margin=old.PARAMETERS['recovery_negative_margin'],
    maximum_initial_overlap=old.PARAMETERS['occlusion_intersection_fraction'])

def pair_initialization_class(base,homography,abstain=False):
    class PairInitialization(base):
        def __init__(self):
            super().__init__();self.pair_seed_log=[]
        def initialize(self,frame,rows,features):
            original=old.selector.seed_by_court_geometry
            byid={int(r['track_id']):f for r,f in zip(rows,features)}
            def ranked(candidates,first,width,height):
                allowed=[r for r in candidates if eligible(r,homography)]
                proposed=original(allowed,first,width,height)
                near=[r for r in allowed if int(r['source_frame'])==first and int(r['track_id'])>=0 and float(r['bottom_center_y'])/height>=.61 and (float(r['y2'])-float(r['y1']))/height>=.08]
                if len(near)<=2:return proposed
                far=[next(r for r in allowed if int(r['track_id'])==proposed[slot]) for slot in ['far_rear','far_front']]
                options=[]
                for a,b in itertools.combinations(near,2):
                    fa,fb=byid[int(a['track_id'])],byid[int(b['track_id'])]
                    if fa is None or fb is None:continue
                    selected=[a,b]+far
                    if any(old.overlap_fraction(x,y)>PARAMETERS['maximum_initial_overlap'] for i,x in enumerate(selected) for y in selected[i+1:]):continue
                    distance=old.a.app_distance(fa,fb)
                    if math.isfinite(distance):options.append((distance,-(float(a['confidence'])+float(b['confidence']))/2,a,b))
                options.sort(key=lambda item:item[:2])
                margin=options[1][0]-options[0][0] if len(options)>1 else None
                wait=not options or (abstain and margin is not None and margin<PARAMETERS['ambiguity_margin'])
                self.pair_seed_log.append(dict(source_frame=frame,near_candidate_count=len(near),valid_pairs=len(options),abstained=wait,best_pair_ids=[int(r['track_id']) for r in options[0][2:]] if options else [],best_distance=options[0][0] if options else None,second_distance=options[1][0] if len(options)>1 else None,distance_margin=margin))
                if wait:raise RuntimeError('Initial near pair unresolved: retain waiting state.')
                selected=sorted(options[0][2:],key=lambda row:float(row['bottom_center_y']))
                return dict(proposed,near_front=int(selected[0]['track_id']),near_rear=int(selected[1]['track_id']))
            old.selector.seed_by_court_geometry=ranked
            try:return super().initialize(frame,rows,features)
            finally:old.selector.seed_by_court_geometry=original
    return PairInitialization
