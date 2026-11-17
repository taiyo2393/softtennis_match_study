"""Retain failed unconstrained fit; add explicit far-in-court assumption variant."""
from pathlib import Path
import itertools,math,shutil
import numpy as np
import estimate_044_far_bounces as e
import build_044_spatiotemporal_pilot as b

OUT=e.OUT/'court_constrained_trial'

def solve(D,target,C,d):
    best=None
    for bits in itertools.product([0,1],repeat=len(d)):
        indices=[i for i,v in enumerate(bits) if v]
        if indices:
            A=C[indices];rhs=d[indices];_,s,Vt=np.linalg.svd(A,full_matrices=True);rank=sum(s>1e-9)
            point=np.linalg.lstsq(A,rhs,rcond=None)[0]
            if np.max(np.abs(A@point-rhs))>1e-7:continue
            Z=Vt[rank:].T
            coef=point+Z@np.linalg.lstsq(D@Z,target-D@point,rcond=None)[0] if Z.shape[1] else point
        else:coef=np.linalg.lstsq(D,target,rcond=None)[0]
        if np.min(C@coef-d)<-1e-7:continue
        loss=float(np.sum((D@coef-target)**2))
        if best is None or loss<best[0]:best=(loss,coef)
    assert best is not None
    return best

def main():
    assert not OUT.exists();OUT.mkdir()
    inputs=[Path(__file__),e.OUT/'protocol.json',e.OUT/'far_fit_models.json',e.OUT/'result.json',
        e.OUT/'observed_and_estimated_image_path.csv',e.OUT/'schematic_court_routes.csv',b.CAL]
    b.dump(OUT/'protocol.json',dict(reason='Unconstrained image-curve fits placed both far bounces behind baseline (court y1.097/1.132). Retain failed trial; do not silently clamp positions.',
        new_assumption='Far returns in this continuing sequence are assumed to have bounced inside far doubles half x0..1,y0.5..1,including boundary. This is an assumption,not new human in/out or landing truth.',
        change='Joint constrained least-squares fits both shared bounce image coordinates with four projected polygon inequalities, plus original four image-y sign/curvature inequalities. Enumerate all active sets. Same time candidates, windows, observations and model.',
        warning='A boundary-active solution indicates tension between simple image model and assumption, not a verified line landing. Label low confidence; no physical3D reconstruction.',
        fixed_before_constrained_outputs=True,protected_input_hashes={str(p):b.sha(p) for p in inputs}))
    H=np.array(b.load(b.CAL)['image_to_court_homography']);corners=np.array([b.project(np.linalg.inv(H),u,v) for u,v in [[0,1],[1,1],[1,.5],[0,.5]]])/2
    sign=1 if np.cross(corners[1]-corners[0],corners[2]-corners[1])>0 else -1
    C=[];bounds=[]
    for j in [4,5,6,7]:
        row=np.zeros(8);row[j]=1;C.append(row);bounds.append(0.)
    for a,z in zip(corners,np.roll(corners,-1,axis=0)):
        delta=z-a;scale=np.linalg.norm(delta);row=np.zeros(8)
        row[0]=-delta[1]*sign/scale;row[3]=delta[0]*sign/scale
        C.append(row);bounds.append((delta[0]*a[1]-delta[1]*a[0])*sign/scale)
    C=np.array(C);bounds=np.array(bounds);raw=b.read(b.OUT/'ball_image_observations.csv');models=[];profiles=[]
    for gap in e.FAR:
        first,last=gap['missing_first'],gap['missing_last'];end=min(last+24,gap['next_contact_first']-2)
        used=[r for r in raw if r['observation_present']=='1' and (first-24<=int(r['source_frame'])<first or last<int(r['source_frame'])<=end)]
        frames=[int(r['source_frame']) for r in used];xx=np.array([float(r['image_x_px']) for r in used]);yy=np.array([float(r['image_y_px']) for r in used]);target=np.concatenate([xx,yy]);trials=[]
        for tau in range(first,last+1):
            ax,ay=e.matrices(frames,tau);D=np.zeros((2*len(frames),8));D[:len(frames),:3]=ax;D[len(frames):,3:]=ay
            loss,coef=solve(D,target,C,bounds);u,v=b.project(H,coef[0]*2,coef[3]*2)
            trials.append(dict(candidate_id=gap['id'],source_frame=tau,image_x=float(coef[0]),image_y=float(coef[3]),
                assumed_ground_court_x=u,assumed_ground_court_y=v,fit_RMS_image_px=math.sqrt(loss/len(frames)),
                x_coefficients=coef[:3].tolist(),y_coefficients=coef[3:].tolist(),
                constraint_minimum_margin=float(np.min(C@coef-bounds))))
        best=min(trials,key=lambda v:(v['fit_RMS_image_px'],v['source_frame']));sensitive=[v for v in trials if v['fit_RMS_image_px']<=best['fit_RMS_image_px']+1]
        model=dict(best,missing_first=first,missing_last=last,after_stroke=gap['after_stroke'],before_stroke=gap['before_stroke'],fit_source_frames=frames,
            court_boundary_active=bool(min(abs(best['assumed_ground_court_x']),abs(best['assumed_ground_court_x']-1),abs(best['assumed_ground_court_y']-.5),abs(best['assumed_ground_court_y']-1))<1e-5),
            quality='low_confidence_boundary_constrained_model_prediction',
            timing_sensitivity_not_CI=[min(v['source_frame'] for v in sensitive),max(v['source_frame'] for v in sensitive)])
        models.append(model);profiles.extend({k:v for k,v in r.items() if not k.endswith('_coefficients')} for r in trials)
    b.dump(OUT/'far_fit_models.json',models);b.write(OUT/'all_bounce_time_fit_profiles.csv',profiles)
    b.write(OUT/'estimated_far_bounce_points.csv',[{k:v for k,v in m.items() if not isinstance(v,list)} for m in models])
    path=b.read(e.OUT/'observed_and_estimated_image_path.csv')
    for r in path:
        if r['estimate_source']!='far_gap_piecewise_curve_prediction':continue
        f=int(r['source_frame']);m=next(v for v in models if v['missing_first']<=f<=v['missing_last']);ax,ay=e.matrices([f],m['source_frame'])
        x=float((ax@np.array(m['x_coefficients']))[0]);y=float((ay@np.array(m['y_coefficients']))[0])
        r.update(estimated_image_x_px=x,estimated_image_y_px=y,display_image_x_px=x,display_image_y_px=y,estimate_source='court_constrained_far_gap_curve_prediction')
    b.write(OUT/'observed_and_estimated_image_path.csv',path)
    routes=b.read(e.OUT/'schematic_court_routes.csv');mapping={v['candidate_id']:v for v in models}
    for r in routes:
        if r['anchor_id'] in mapping:
            m=mapping[r['anchor_id']];r.update(court_x=m['assumed_ground_court_x'],court_y=m['assumed_ground_court_y'],source_frame=m['source_frame'],anchor_kind='low_confidence_court_constrained_far_bounce')
    b.write(OUT/'schematic_court_routes.csv',routes)
    b.dump(OUT/'result.json',dict(estimated_far_points=2,boundary_active_solutions=sum(m['court_boundary_active'] for m in models),
        original_unconstrained_trial_retained=True,measured_landing_truth_created=False,physical_3D_positions_created=False,
        display_warning='Both point/time and curve depend on in-court assumption and weak image-space model; boundary solutions are low confidence.'))
    for m in models:print(m['candidate_id'],m['source_frame'],m['assumed_ground_court_x'],m['assumed_ground_court_y'],'RMSfit',m['fit_RMS_image_px'],'boundary',m['court_boundary_active'],flush=True)
    assert all(b.sha(Path(n))==h for n,h in b.load(OUT/'protocol.json')['protected_input_hashes'].items())
    shutil.copy2(Path(__file__),OUT/'executed_source.py')

if __name__=='__main__':main()
