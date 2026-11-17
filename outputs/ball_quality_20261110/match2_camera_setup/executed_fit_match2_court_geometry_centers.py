"""Measure paint-center agreement, rather than confusing stripe width with error."""
import sys
from pathlib import Path
import cv2,numpy as np
import fit_match2_court_geometry as prior
from run_fixed_cohort_player_case_v3 import ROOT,BASE,sha,dump

OUT=BASE/'match2_camera_setup/geometry_center_trial'
FAILED=BASE/'match2_camera_setup/geometry_trial'


def fit_line_centers(image,endpoints):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);white=(hsv[:,:,2]>=210)&(hsv[:,:,1]<=80)
    a,b=np.asarray(endpoints,dtype=float);delta=b-a;length=np.linalg.norm(delta)
    l=max(0,int(min(a[0],b[0])-8));r=min(image.shape[1],int(max(a[0],b[0])+9))
    t=max(0,int(min(a[1],b[1])-8));bottom=min(image.shape[0],int(max(a[1],b[1])+9))
    yy,xx=np.nonzero(white[t:bottom,l:r]);xy=np.column_stack([xx+l,yy+t]).astype(float)
    if not len(xy):return None,dict(pixels=0,bin_coverage=0,accepted=False)
    along=(xy-a)@delta/(length*length)
    dist=np.abs((xy[:,0]-a[0])*delta[1]-(xy[:,1]-a[1])*delta[0])/length
    xy=xy[(dist<=8)&(along>=.02)&(along<=.98)]
    along=(xy-a)@delta/(length*length) if len(xy) else np.array([])
    bins=np.clip(np.floor(along*20).astype(int),0,19)
    centers=np.array([np.median(xy[bins==j],axis=0) for j in sorted(set(bins))])
    coverage=len(centers)/20
    if len(xy)<100 or len(centers)<2:return None,dict(pixels=len(xy),bin_coverage=coverage,accepted=False)
    vx,vy,x,y=cv2.fitLine(centers.astype(np.float32),cv2.DIST_L1,0,.01,.01).ravel().astype(float)
    line=np.array([-vy,vx,vy*x-vx*y])
    center_residual=np.abs(centers@line[:2]+line[2]);pixel_residual=np.abs(xy@line[:2]+line[2])
    median=float(np.median(center_residual))
    return line,dict(pixels=len(xy),bin_coverage=coverage,median_center_residual_px=median,
        median_paint_pixel_spread_px=float(np.median(pixel_residual)),
        accepted=coverage>=.6 and median<=3)


def prepare():
    assert not OUT.exists();prior.OUT=OUT;prior.prepare()
    p=prior.load(OUT/'protocol.json')
    p.update(change='One robust median paint center per existing20 longitudinal bins, fit to centers. Raw paint-pixel spread is reported separately from geometric line residual.',
        reason='First trial rejected all main frames because near baseline paint stripe width produced median raw-pixel distance3.04px versus3px gate. A wide straight stripe is not a misaligned line. Preserve failed trial and constants; fix measured quantity, no threshold loosening.',
        raw_new_match_calibration_development=True,person_or_ball_outputs_still_unseen=True,
        failed_trial_preserved=str(FAILED))
    p['parameters']=dict(p['parameters'],fit='L1 over per-bin median paint centers; report raw paint width separately')
    for n in [Path(__file__),FAILED/'protocol.json',FAILED/'F54006.json']:
        p['protected_input_hashes'][str(n)]=sha(n)
    dump(OUT/'protocol.json',p)


def run():
    prior.OUT=OUT;prior.fit_line=fit_line_centers;prior.run()


if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
