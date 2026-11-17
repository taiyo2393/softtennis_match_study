"""User-authorized exploratory far-bounce/image-path estimates; immutable observations."""
from pathlib import Path
import math,itertools,shutil,sys
import numpy as np
import build_044_spatiotemporal_pilot as b

OUT=b.HERE/'044_estimated_ball_routes'
NEAR=b.HERE/'044_bounce_projection'
CONF=b.HERE/'044_sequence_confirmed'
STROKE=b.HERE/'044_stroke_review'
FAR=[dict(id='FAR01',missing_first=115475,missing_last=115499,next_contact_first=115519,
          after_stroke='H02',before_stroke='H03'),
     dict(id='FAR02',missing_first=115693,missing_last=115711,next_contact_first=115727,
          after_stroke='H04',before_stroke='H05')]

def prepare():
    assert not OUT.exists()
    files=[Path(__file__),b.CAL,b.OUT/'ball_image_observations.csv',b.OUT/'player_court_proxies.csv',b.OUT/'protocol.json',
        NEAR/'publication_hashes.json',NEAR/'conditional_bounce_court_points.csv',CONF/'publication_hashes.json',
        CONF/'confirmed_stroke_sequence.csv',STROKE/'AI_stroke_records.csv']
    OUT.mkdir()
    b.dump(OUT/'user_request_and_scope.json',dict(recorded_on_jst='2026-11-11',
        exact_request='手前の暫定バウンド点はあっていると思う．\n奥のコートの暫定バウンド点も予測でいいから出してほしいかな\nあと，できれば球の軌道も出したい',
        near_point_confirmation='Both displayed near bounce candidates look approximately plausible to user; not exact timing/coordinate GT.',
        authorized_new_work='Predict far bounce candidates and add estimated trajectory presentation in separate outputs.',
        no_changes_to_original_observations_or_labels=True))
    b.dump(OUT/'protocol.json',dict(date_jst='2026-11-11',far_gaps=FAR,
        approach='Exploratory continuous piecewise image-space fit across known ball-center gaps. Not3D dynamics or physical gravity.',
        fixed_fit_window='Up to24sourceframes before missing_first and24after missing_last, clipped before next AI contact first. Visible sampled observations only; no training/GT fitted.',
        fixed_fit_model='For each integer bounce time in original missing interval: x continuous piecewise linear; y continuous piecewise quadratic. Shared bounce x/y. incoming dy/dt>=0,outgoing dy/dt<=0, both y quadratic coefficients>=0 in downward-positive image. Enumerate active inequality bounds by NumPy least squares.',
        selection='Minimum combined x/y squared residual across fixed candidates. No court-side or desired landing coordinate used to select/time/crop/refit. Retain whole time profile. Physical/court plausibility only flagged after fit.',
        sensitivity='All trial times with fitted RMS within1nativepixel of minimum shown as sensitivity range, not confidence interval or ground-truth error.',
        conditional_projection='Only fitted bounce point is transformed with existing native-scale-composed homography, assuming ground level. No projection of airborne curve to court.',
        image_missing='Far gaps use fitted curve only at missing frames. Other gaps use straight image interpolation only if bounding observations and whole gap do not cross any provisional contact/bounce interval. Leave contact gaps and unannotated times blank. Retain original164positions byte-for-byte in observed columns.',
        court_routes='Dashed schematic: previous hitter bbox-bottom proxy -> known/estimated bounce -> next hitter bbox-bottom proxy. Not measured airborne ground track, no moving ground-ball point. Fifth shot endpoint unknown, not invented.',
        independent_accuracy_established=False,canonical_inputs_updated=False,
        protected_input_hashes={str(p):b.sha(p) for p in files}))

def constrained_y(A,y):
    # First coefficient is shared y_b; four nonnegative transformed velocity/curvature parameters.
    best=None
    for zero_bits in itertools.product([0,1],repeat=4):
        free=[0]+[j+1 for j,v in enumerate(zero_bits) if not v]
        value=np.linalg.lstsq(A[:,free],y,rcond=None)[0]
        coef=np.zeros(5);coef[free]=value
        if np.min(coef[1:]) < -1e-8:continue
        error=float(np.sum((A@coef-y)**2))
        if best is None or error<best[0]:best=(error,coef)
    assert best is not None
    return best

def matrices(frames,tau):
    t=(np.array(frames,dtype=float)-tau)/b.FPS;lo=np.minimum(t,0);hi=np.maximum(t,0)
    return np.column_stack([np.ones(len(t)),lo,hi]),np.column_stack([np.ones(len(t)),lo,lo*lo,-hi,hi*hi])

def fit():
    p=b.load(OUT/'protocol.json');assert all(b.sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    raw=b.read(b.OUT/'ball_image_observations.csv');by={int(r['source_frame']):r for r in raw};H=b.load(b.CAL)['image_to_court_homography']
    profile=[];models=[];farrows=[];predicted={}
    for gap in FAR:
        first,last=gap['missing_first'],gap['missing_last'];end=min(last+24,gap['next_contact_first']-2)
        used=[r for r in raw if r['observation_present']=='1' and (first-24<=int(r['source_frame'])<first or last<int(r['source_frame'])<=end)]
        frames=[int(r['source_frame']) for r in used];xx=np.array([float(r['image_x_px']) for r in used]);yy=np.array([float(r['image_y_px']) for r in used]);trials=[]
        assert sum(f<first for f in frames)>=6 and sum(f>last for f in frames)>=6
        for tau in range(first,last+1):
            ax,ay=matrices(frames,tau);cx=np.linalg.lstsq(ax,xx,rcond=None)[0];sy,cy=constrained_y(ay,yy)
            sse=float(np.sum((ax@cx-xx)**2)+sy);rms=math.sqrt(sse/len(frames));u,v=b.project(H,cx[0]*2,cy[0]*2)
            trials.append(dict(candidate_id=gap['id'],source_frame=tau,fit_RMS_image_px=rms,image_x=float(cx[0]),image_y=float(cy[0]),
                assumed_ground_court_x=u,assumed_ground_court_y=v,incoming_image_vx=float(cx[1]),outgoing_image_vx=float(cx[2]),
                incoming_image_vy=float(cy[1]),outgoing_image_vy=float(-cy[3]),x_coefficients=cx.tolist(),y_coefficients=cy.tolist()))
        best=min(trials,key=lambda r:(r['fit_RMS_image_px'],r['source_frame']));sensitivity=[r for r in trials if r['fit_RMS_image_px']<=best['fit_RMS_image_px']+1]
        model=dict(best,fit_source_frames=frames,missing_first=first,missing_last=last,fit_window_first=min(frames),fit_window_last=max(frames),
            after_stroke=gap['after_stroke'],before_stroke=gap['before_stroke'],
            estimated_point_status='model_prediction_no_visible_ball_center_no_human_landing_confirmation',
            far_half_plausibility_flag=bool(-.2<=best['assumed_ground_court_x']<=1.2 and .5<=best['assumed_ground_court_y']<=1.2),
            sensitivity_not_confidence_interval=dict(first_frame=min(r['source_frame'] for r in sensitivity),last_frame=max(r['source_frame'] for r in sensitivity),
                min_court_x=min(r['assumed_ground_court_x'] for r in sensitivity),max_court_x=max(r['assumed_ground_court_x'] for r in sensitivity),
                min_court_y=min(r['assumed_ground_court_y'] for r in sensitivity),max_court_y=max(r['assumed_ground_court_y'] for r in sensitivity)))
        models.append(model)
        for r in trials:profile.append({k:v for k,v in r.items() if not k.endswith('_coefficients')})
        farrows.append({k:v for k,v in model.items() if not isinstance(v,(list,dict))})
        for f in range(first,last+1):
            if f not in by or by[f]['observation_present']=='1':continue
            ax,ay=matrices([f],best['source_frame']);predicted[f]=(float((ax@np.array(best['x_coefficients']))[0]),float((ay@np.array(best['y_coefficients']))[0]),'far_gap_piecewise_curve_prediction')
    b.dump(OUT/'far_fit_models.json',models);b.write(OUT/'all_bounce_time_fit_profiles.csv',profile);b.write(OUT/'estimated_far_bounce_points.csv',farrows)
    # Add only short, within-flight image interpolation. Do not bridge a contact/bounce boundary.
    strokes=b.read(STROKE/'AI_stroke_records.csv');blocked=[(int(r['source_frame_first']),int(r['source_frame_last'])) for r in strokes]+[(115413,115419),(115569,115575)]
    visible=sorted(f for f,r in by.items() if r['observation_present']=='1')
    for a,z in zip(visible,visible[1:]):
        if z-a<=2 or z-a>12:continue
        if any(a<=v and u<=z for u,v in blocked):continue
        for f in range(a+2,z,2):
            if f in predicted:continue
            r=by[f]
            if r['annotation_available']!='1':continue
            t=(f-a)/(z-a);predicted[f]=(float(by[a]['image_x_px'])*(1-t)+float(by[z]['image_x_px'])*t,
                float(by[a]['image_y_px'])*(1-t)+float(by[z]['image_y_px'])*t,'short_within_flight_image_interpolation')
    rows=[]
    for f,r in sorted(by.items()):
        observed=r['observation_present']=='1';estimate=predicted.get(f)
        rows.append(dict(source_frame=f,original_observation_present=r['observation_present'],original_image_x_px=r['image_x_px'],original_image_y_px=r['image_y_px'],
            estimated_image_x_px=estimate[0] if estimate else '',estimated_image_y_px=estimate[1] if estimate else '',estimate_source=estimate[2] if estimate else '',
            display_image_x_px=r['image_x_px'] if observed else estimate[0] if estimate else '',display_image_y_px=r['image_y_px'] if observed else estimate[1] if estimate else '',
            display_position_kind='observed_assisted' if observed else 'estimated' if estimate else 'missing',
            ground_court_x='',ground_court_y='',independent_GT_created=0))
    b.write(OUT/'observed_and_estimated_image_path.csv',rows)
    # Schematic court routes, with explicit anchor types and no fabricated ball height/landing precision.
    near=b.read(NEAR/'conditional_bounce_court_points.csv');landings={r['after_stroke']:dict(id=r['candidate_id'],u=float(r['assumed_ground_court_x']),v=float(r['assumed_ground_court_y']),frame=int(r['source_frame']),kind='user_coarse_plausibility_bounce_candidate') for r in near}
    landings.update({r['after_stroke']:dict(id=r['candidate_id'],u=r['assumed_ground_court_x'],v=r['assumed_ground_court_y'],frame=r['source_frame'],kind='model_predicted_far_bounce') for r in models})
    routes=[]
    for i,r in enumerate(strokes[:-1]):
        nxt=strokes[i+1];landing=landings[r['stroke_id']]
        for order,anchor in enumerate([dict(id=r['stroke_id'],u=float(r['hitter_court_proxy_x']),v=float(r['hitter_court_proxy_y']),frame=int(r['reference_source_frame']),kind='hitter_bbox_bottom_proxy'),landing,
            dict(id=nxt['stroke_id'],u=float(nxt['hitter_court_proxy_x']),v=float(nxt['hitter_court_proxy_y']),frame=int(nxt['reference_source_frame']),kind='next_hitter_bbox_bottom_proxy')]):
            routes.append(dict(stroke_id=r['stroke_id'],anchor_order=order,anchor_id=anchor['id'],court_x=anchor['u'],court_y=anchor['v'],source_frame=anchor['frame'],anchor_kind=anchor['kind'],route_meaning='schematic_delivery_polyline_not_measured_ball_ground_track'))
    b.write(OUT/'schematic_court_routes.csv',routes)
    b.dump(OUT/'result.json',dict(near_bounces_approximately_plausible_user_review=2,estimated_far_bounces=2,
        original_assisted_image_observations=sum(r['display_position_kind']=='observed_assisted' for r in rows),
        estimated_image_positions=sum(r['display_position_kind']=='estimated' for r in rows),remaining_missing_display_times=sum(r['display_position_kind']=='missing' for r in rows),
        physical_3D_ball_positions_created=0,ground_schematic_routes=4,original_observations_or_missing_labels_changed=False))
    print(b.load(OUT/'result.json'),flush=True)
    for m in models:print(m['candidate_id'],'frame',m['source_frame'],'court',m['assumed_ground_court_x'],m['assumed_ground_court_y'],'fitRMS(not accuracy)',m['fit_RMS_image_px'],'farflag',m['far_half_plausibility_flag'],flush=True)

if __name__=='__main__':{'prepare':prepare,'fit':fit}[sys.argv[1]]()
