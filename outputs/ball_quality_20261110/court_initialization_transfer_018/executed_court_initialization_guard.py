"""Development-only initial-player filter for this calibrated fixed camera.

No raw IDs, outcomes, uniforms, or later tracking states enter selection.
The spatial filter applies only before initialization, never during a rally.
"""
import numpy as np
from confirmed_continuity_guard import old

PARAMETERS=dict(court_x_min=-.15,court_x_max=1.15,court_y_min=-.25,court_y_max=1.25,
                denominator_epsilon=1e-9,initialization_only=True)

def court_point(row,homography):
    q=homography@np.array([float(row['bottom_center_x']),float(row['bottom_center_y']),1.])
    if not np.isfinite(q).all() or abs(q[2])<=PARAMETERS['denominator_epsilon']:return None
    xy=q[:2]/q[2]
    return tuple(map(float,xy)) if np.isfinite(xy).all() else None

def eligible(row,homography):
    xy=court_point(row,homography)
    return xy is not None and PARAMETERS['court_x_min']<=xy[0]<=PARAMETERS['court_x_max'] and PARAMETERS['court_y_min']<=xy[1]<=PARAMETERS['court_y_max']

def court_initialization_class(base,homography):
    h=np.asarray(homography,dtype=float).copy();assert h.shape==(3,3)
    class CourtInitialization(base):
        def __init__(self):
            super().__init__();self.seed_filter_log=[]
        def initialize(self,frame,rows,features):
            original=old.selector.seed_by_court_geometry
            allowed=[r for r in rows if eligible(r,h)]
            self.seed_filter_log.append(dict(source_frame=frame,available=len(rows),eligible=len(allowed),excluded_raw_ids=[int(r['track_id']) for r in rows if not eligible(r,h)]))
            def restricted(candidates,first,width,height):
                return original([r for r in candidates if eligible(r,h)],first,width,height)
            old.selector.seed_by_court_geometry=restricted
            try:return super().initialize(frame,rows,features)
            finally:old.selector.seed_by_court_geometry=original
    return CourtInitialization
