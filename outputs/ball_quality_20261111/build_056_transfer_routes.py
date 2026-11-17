"""Same frozen image-gap fit/display conventions, AI event proposals for056."""
from pathlib import Path
from collections import Counter
import math,shutil
import numpy as np
import build_044_spatiotemporal_pilot as b
from prepare_056_route_transfer import OUT,FRAMES,PERSON,OFFSET,TIMES
from estimate_044_far_bounces import matrices,constrained_y
from constrain_044_far_bounce_fit import solve

# AI proposals after retained uniform chronology/native grids. Not new human labels.
STROKES=[('H01','near_rear',154081,154086,154086),('H02','far_rear',154152,154160,154156),
 ('H03','near_front',154182,154190,154186),('H04','far_rear',154252,154260,154256),
 ('H05','near_front',154292,154300,154296),('H06','far_front',154344,154352,154348),
 ('H07','near_rear',154468,154474,154470),('H08','far_rear',154508,154516,154512),
 ('H09','near_rear',154634,154642,154638),('H10','far_front',154740,154748,154744),
 ('H11','near_front',154786,154806,154796),('H12','far_rear',154820,154830,154824),
 ('H13','near_front',154842,154852,154846)]
VISIBLE=[dict(id='B01',frame=154120,first=154116,last=154124,after='H01',before='H02',side='far'),
 dict(id='B02',frame=154230,first=154226,last=154234,after='H03',before='H04',side='far'),
 dict(id='B03',frame=154330,first=154328,last=154334,after='H05',before='H06',side='far')]
MISSING=[dict(id='P01',first=154428,last=154444,after='H06',before='H07',side='near'),
 dict(id='P02',first=154592,last=154592,after='H08',before='H09',side='near'),
 dict(id='P03',first=154722,last=154722,after='H09',before='H10',side='far')]

def constraints(H,side):
    bounds=(0,.5) if side=='near' else (.5,1)
    corners=np.array([b.project(np.linalg.inv(H),u,v) for u,v in [[0,bounds[1]],[1,bounds[1]],[1,bounds[0]],[0,bounds[0]]]])/2
    a=corners[1]-corners[0];z=corners[2]-corners[1];sign=1 if a[0]*z[1]-a[1]*z[0]>0 else -1
    C=[];d=[]
    for j in [4,5,6,7]:q=np.zeros(8);q[j]=1;C.append(q);d.append(0.)
    for a,z in zip(corners,np.roll(corners,-1,axis=0)):
        delta=z-a;scale=np.linalg.norm(delta);q=np.zeros(8);q[0]=-delta[1]*sign/scale;q[3]=delta[0]*sign/scale
        C.append(q);d.append((delta[0]*a[1]-delta[1]*a[0])*sign/scale)
    return np.array(C),np.array(d)

def main():
    assert not (OUT/'AI_event_proposals.json').exists()
    proposal=dict(reviewer='AI chronological/native-image inspection,not independent human event GT',strokes=[dict(id=x[0],slot=x[1],first=x[2],last=x[3],reference=x[4]) for x in STROKES],
        visible_bounce_candidates=VISIBLE,predicted_bounce_gaps=MISSING,
        uncertain=['H09 near N2 contact is partly outside left image; timing and proxy are weak.','H11 inferred within ball-missing neighborhood; broad interval retained.','Last net exchange H12/H13 is AI-provisional; original ball annotation ends154838.','Do not extend inherited human valid-serve/two-bounce-end facts to these timings or actors.'],
        not_a_human_confirmed_13stroke_sequence=True,not_independent_accuracy=True)
    b.dump(OUT/'AI_event_proposals.json',proposal)
    b.dump(OUT/'fit_application_protocol.json',dict(fixed_before_056_fit_outputs=True,model='Exact044 matrices/constrained_y/solve functions and thresholds reused.',
        half='Target near/far polygon follows AI recipient side, not044 hardcoded far side. Same inequality mechanism, no parameter tuning.',
        candidates=VISIBLE,gaps=MISSING,windows='24sourceframes before/after gap clipped to adjacent AI contact boundaries; >=6visible points per side.',
        warning='Model estimates are not observations. Every trial retained; boundary-active estimates flagged. No bounce invented on AI volley intervals.',
        source_hashes={str(p):b.sha(p) for p in [Path(__file__),b.HERE/'estimate_044_far_bounces.py',b.HERE/'constrain_044_far_bounce_fit.py',OUT/'AI_event_proposals.json',OUT/'ball_image_observations.csv']}))
    raw=b.read(OUT/'ball_image_observations.csv');by={int(r['source_frame']):r for r in raw};people=b.grouped(b.read(OUT/'player_court_proxies.csv'));H=np.array(b.load(b.CAL)['image_to_court_homography'])
    actors=[]
    for i,(sid,slot,first,last,ref) in enumerate(STROKES):
        person=next((r for r in people[ref] if r['player_slot']==slot),None)
        actors.append(dict(stroke_id=sid,ordinal=i+1,hitter_candidate=slot,source_frame_first=first,source_frame_last=last,reference_source_frame=ref,
            hitter_court_proxy_x=person['court_proxy_x'] if person else '',hitter_court_proxy_y=person['court_proxy_y'] if person else '',
            status='AI_provisional_not_new_human_confirmed',exact_impact_or_position_validated=0))
    b.write(OUT/'AI_stroke_records.csv',actors)
    stroke={r['stroke_id']:r for r in actors};landings=[];models=[];profiles=[];predictions={}
    for v in VISIBLE:
        r=by[v['frame']];assert r['observation_present']=='1';x,y=float(r['image_x_px']),float(r['image_y_px']);u,w=b.project(H,x*2,y*2)
        landings.append(dict(id=v['id'],source_frame=v['frame'],image_x=x,image_y=y,court_x=u,court_y=w,after_stroke=v['after'],before_stroke=v['before'],
            status='AI_visible_bounce_candidate_assisted_center',human_bounce_confirmed=0,boundary_active=False))
    for gap in MISSING:
        first,last=gap['first'],gap['last'];start=max(first-24,int(stroke[gap['after']]['source_frame_last'])+1);end=min(last+24,int(stroke[gap['before']]['source_frame_first'])-2)
        used=[r for r in raw if r['observation_present']=='1' and (start<=int(r['source_frame'])<first or last<int(r['source_frame'])<=end)]
        frames=[int(r['source_frame']) for r in used];assert sum(f<first for f in frames)>=6 and sum(f>last for f in frames)>=6
        xx=np.array([float(r['image_x_px']) for r in used]);yy=np.array([float(r['image_y_px']) for r in used]);target=np.r_[xx,yy];C,d=constraints(H,gap['side']);trials=[]
        for tau in range(first,last+1):
            ax,ay=matrices(frames,tau);cx=np.linalg.lstsq(ax,xx,rcond=None)[0];sy,cy=constrained_y(ay,yy)
            D=np.zeros((len(frames)*2,8));D[:len(frames),:3]=ax;D[len(frames):,3:]=ay
            loss,coef=solve(D,target,C,d);u,w=b.project(H,coef[0]*2,coef[3]*2);old_u,old_v=b.project(H,cx[0]*2,cy[0]*2)
            trials.append(dict(id=gap['id'],source_frame=tau,image_x=float(coef[0]),image_y=float(coef[3]),court_x=u,court_y=w,
                fit_RMS_px=math.sqrt(loss/len(frames)),unconstrained_RMS_px=math.sqrt((float(np.sum((ax@cx-xx)**2))+sy)/len(frames)),
                unconstrained_court_x=old_u,unconstrained_court_y=old_v,x_coefficients=coef[:3].tolist(),y_coefficients=coef[3:].tolist()))
        best=min(trials,key=lambda r:(r['fit_RMS_px'],r['source_frame']));lo,hi=(0,.5) if gap['side']=='near' else (.5,1)
        boundary=min(abs(best['court_x']),abs(best['court_x']-1),abs(best['court_y']-lo),abs(best['court_y']-hi))<1e-5
        model=dict(best,gap=gap,fit_source_frames=frames,boundary_active=bool(boundary),status='curve_model_bounce_prediction_not_landing_GT')
        models.append(model);profiles.extend({k:v for k,v in t.items() if not k.endswith('_coefficients')} for t in trials)
        landings.append({k:v for k,v in model.items() if k in ['id','source_frame','image_x','image_y','court_x','court_y','status','boundary_active']}|dict(after_stroke=gap['after'],before_stroke=gap['before'],human_bounce_confirmed=0))
        for f in range(first,last+1,2):
            if f not in by or by[f]['observation_present']=='1':continue
            ax,ay=matrices([f],best['source_frame']);predictions[f]=(float((ax@np.array(best['x_coefficients']))[0]),float((ay@np.array(best['y_coefficients']))[0]),'same044_piecewise_bounce_model')
    b.dump(OUT/'bounce_fit_models.json',models);b.write(OUT/'all_bounce_fit_trials.csv',profiles);b.write(OUT/'bounce_court_points.csv',landings)
    # Diagnostic case flag only. Do not correct the assisted center or learn a new filter.
    f=154780;a=by[f-2];r=by[f];z=by[f+2];center=(np.array([float(a['image_x_px']),float(a['image_y_px'])])+np.array([float(z['image_x_px']),float(z['image_y_px'])]))/2
    residual=float(np.linalg.norm(np.array([float(r['image_x_px']),float(r['image_y_px'])])-center))
    b.dump(OUT/'AI_ball_position_warning.json',dict(source_frame=f,original_image_x_px=r['image_x_px'],original_image_y_px=r['image_y_px'],
        adjacent_linear_reference_only_not_replacement=center.tolist(),residual_px=residual,
        scope='One AI-discovered suspicious assisted-center jump. Not a validated automatic filter, no human center edit.',
        display='Original point red warning circle; do not draw path segments to it. Never replace it with diagnostic reference.'))
    blocked=[(v[2],v[3]) for v in STROKES]+[(v['first'],v['last']) for v in VISIBLE]+[(g['first'],g['last']) for g in MISSING]
    visible=sorted(k for k,r in by.items() if r['observation_present']=='1')
    for a,z in zip(visible,visible[1:]):
        if z-a<=2 or z-a>12 or a<=154780<=z or any(a<=v and u<=z for u,v in blocked):continue
        for f in range(a+2,z,2):
            if f in predictions:continue
            t=(f-a)/(z-a);predictions[f]=(float(by[a]['image_x_px'])*(1-t)+float(by[z]['image_x_px'])*t,
                float(by[a]['image_y_px'])*(1-t)+float(by[z]['image_y_px'])*t,'same044_short_image_interpolation')
    paths=[]
    for f,r in sorted(by.items()):
        obs=r['observation_present']=='1';pr=predictions.get(f)
        paths.append(dict(source_frame=f,original_observation_present=r['observation_present'],original_image_x_px=r['image_x_px'],original_image_y_px=r['image_y_px'],
            estimated_image_x_px=pr[0] if pr else '',estimated_image_y_px=pr[1] if pr else '',estimate_source=pr[2] if pr else '',
            display_image_x_px=r['image_x_px'] if obs else pr[0] if pr else '',display_image_y_px=r['image_y_px'] if obs else pr[1] if pr else '',
            kind='observed_warning' if f==154780 else 'observed_assisted' if obs else 'estimated' if pr else 'missing',
            ground_court_x='',ground_court_y=''))
    b.write(OUT/'observed_and_estimated_image_path.csv',paths)
    routes=[];lookup={v['after_stroke']:v for v in landings}
    for a,z in zip(actors,actors[1:]):
        anchors=[dict(id=a['stroke_id'],u=a['hitter_court_proxy_x'],v=a['hitter_court_proxy_y'],kind='hitter_proxy')]
        if a['stroke_id'] in lookup:
            v=lookup[a['stroke_id']];anchors.append(dict(id=v['id'],u=v['court_x'],v=v['court_y'],kind=v['status']))
        anchors.append(dict(id=z['stroke_id'],u=z['hitter_court_proxy_x'],v=z['hitter_court_proxy_y'],kind='next_hitter_proxy'))
        for j,v in enumerate(anchors):routes.append(dict(stroke_id=a['stroke_id'],anchor_order=j,anchor_id=v['id'],court_x=v['u'],court_y=v['v'],anchor_kind=v['kind'],meaning='schematic_not_3D_or_airborne_ground_track'))
    b.write(OUT/'schematic_court_routes.csv',routes)
    labels=next(r for r in b.read(b.LABEL) if r['candidate_id']=='match1_rally_window_056')
    b.dump(OUT/'result.json',dict(AI_provisional_strokes=len(actors),human_confirmed_new_stroke_sequences=0,
        visible_bounce_candidates=len(VISIBLE),curve_predicted_bounces=len(models),boundary_active_predictions=sum(m['boundary_active'] for m in models),
        path_counts=dict(Counter(r['kind'] for r in paths)),original_ball_positions=334,original_missing_labels=42,
        inherited_human_winner_side=labels['winner_camera_side'],inherited_end_type=labels['point_end_event_type'],independent_accuracy_established=False))
    shutil.copy2(Path(__file__),OUT/'executed_build.py');print(b.load(OUT/'result.json'),flush=True)
    for m in models:print(m['id'],m['source_frame'],m['court_x'],m['court_y'],'boundary',m['boundary_active'],flush=True)

if __name__=='__main__':main()
