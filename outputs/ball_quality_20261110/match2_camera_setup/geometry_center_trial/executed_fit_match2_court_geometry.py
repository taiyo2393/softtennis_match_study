"""Image-line geometry proposal only; no surveyed or player ground truth."""
import sys,json,shutil
from pathlib import Path
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,sha,dump,write
from prepare_match2_external_validation import load

OUT=BASE/'match2_camera_setup/geometry_trial'
PARENT=BASE/'match2_external_validation'
COARSE=dict(far=[[682,533],[1238,536]],right=[[1238,536],[1880,805]],
            near=[[49,796],[1880,805]],left=[[49,796],[682,533]])
PARAMETERS=dict(line_band_px=8,white_value_min=210,white_saturation_max=80,
    minimum_white_pixels=100,coverage_bins=20,minimum_bin_coverage=.60,
    maximum_line_median_residual_px=3,fit='cv2.fitLine L1',
    description='Engineering proposal fixed before line results, no threshold sweep. Raw main-frame calibration supplied by AI coarse painted-line endpoints.')
TARGET=np.float32([[0,1],[1,1],[1,0],[0,0]])


def fit_line(image,endpoints):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    white=(hsv[:,:,2]>=210)&(hsv[:,:,1]<=80)
    a,b=np.asarray(endpoints,dtype=float);delta=b-a;length=np.linalg.norm(delta)
    margin=8
    l=max(0,int(min(a[0],b[0])-margin));r=min(image.shape[1],int(max(a[0],b[0])+margin+1))
    t=max(0,int(min(a[1],b[1])-margin));bottom=min(image.shape[0],int(max(a[1],b[1])+margin+1))
    yy,xx=np.nonzero(white[t:bottom,l:r]);xy=np.column_stack([xx+l,yy+t]).astype(float)
    if not len(xy):return None,dict(pixels=0,bin_coverage=0,accepted=False)
    along=(xy-a)@delta/(length*length)
    distance=np.abs((xy[:,0]-a[0])*delta[1]-(xy[:,1]-a[1])*delta[0])/length
    xy=xy[(distance<=8)&(along>=.02)&(along<=.98)]
    along=(xy-a)@delta/(length*length) if len(xy) else np.array([])
    coverage=len(set(np.clip(np.floor(along*20).astype(int),0,19)))/20
    if len(xy)<100:return None,dict(pixels=len(xy),bin_coverage=coverage,accepted=False)
    vx,vy,x,y=cv2.fitLine(xy.astype(np.float32),cv2.DIST_L1,0,.01,.01).ravel().astype(float)
    line=np.array([-vy,vx,vy*x-vx*y]);residual=np.abs(xy@line[:2]+line[2])
    median=float(np.median(residual));accepted=coverage>=.6 and median<=3
    return line,dict(pixels=len(xy),bin_coverage=coverage,median_residual_px=median,accepted=accepted)


def intersection(a,b):
    p=np.cross(a,b)
    if abs(p[2])<1e-9:raise ValueError('Parallel or degenerate boundary lines.')
    return p[:2]/p[2]


def geometry(image):
    lines={};diagnostic={}
    for name,points in COARSE.items():lines[name],diagnostic[name]=fit_line(image,points)
    if any(v is None for v in lines.values()) or not all(v['accepted'] for v in diagnostic.values()):
        return None,diagnostic
    corners=np.float32([intersection(lines['far'],lines['left']),intersection(lines['far'],lines['right']),
        intersection(lines['near'],lines['right']),intersection(lines['near'],lines['left'])])
    if not cv2.isContourConvex(corners.reshape(-1,1,2)):return None,diagnostic
    H=cv2.getPerspectiveTransform(corners,TARGET)
    if not np.isfinite(H).all() or abs(np.linalg.det(H))<1e-15:return None,diagnostic
    return dict(image_points_px=corners.tolist(),court_points=TARGET.tolist(),image_to_court_homography=H.tolist(),court_to_image_homography=np.linalg.inv(H).tolist()),diagnostic


def prepare():
    assert not OUT.exists();parent=load(BASE/'match2_camera_setup/protocol.json')
    references=load(PARENT/'raw_reference_pilot/manifest.json')['items']
    cases=[dict(source_frame=r['source_frame'],image=str(PARENT/'raw_reference_pilot'/r['filename']),role='frozen_raw_reference') for r in references]
    cases.append(dict(source_frame=1798,image=str(PARENT/'raw_0030s_F1798.jpg'),role='preview_has30px_header_and640px_resize_not_fit_input'))
    # Only the9 true native PNGs enter numerical geometry; preview stays visual.
    OUT.mkdir();dump(OUT/'protocol.json',dict(calibration_anchor_source_frame=54006,
        cases=cases[:9],parameters=PARAMETERS,AI_proposed_coarse_native_line_endpoints=COARSE,
        corner_order=['far_left','far_right','near_right','near_left'],
        normalized_plane='x0 screen-left doubles sideline,x1 right; y0 near baseline,y1 far baseline',
        raw_image_viewing_used=True,person_ball_predictions_or_outcomes_used=False,
        line_pixel_thresholds_fixed_before_outputs=True,
        reused_old_match_camera_calibration=False,
        output='Geometry proposal and camera-line compatibility flag; not independent camera or tracking accuracy.',
        protected_input_hashes={str(n):sha(n) for n in [Path(__file__),BASE/'match2_camera_setup/protocol.json',PARENT/'publication_hashes.json',Path(parent['calibration_anchor_image'])]}))


def run():
    p=load(OUT/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    reports=[];anchor=None
    for e in p['cases']:
        im=cv2.imread(e['image']);assert im.shape[:2]==(1080,1920)
        value,diagnostic=geometry(im);frame=e['source_frame'];dest=OUT/f'F{frame}.json'
        report=dict(source_frame=frame,all_four_paint_lines_supported=value is not None,line_diagnostics=diagnostic,
            geometry=value,independent_human_calibration=False)
        dump(dest,report);reports.append(report)
        if frame==p['calibration_anchor_source_frame']:anchor=value
        preview=im.copy()
        for name,pts in COARSE.items():cv2.line(preview,tuple(pts[0]),tuple(pts[1]),(0,165,255),2)
        if value:
            corners=np.array(value['image_points_px']);cv2.polylines(preview,[np.rint(corners).astype(np.int32)],True,(40,240,40),3)
            for i,point in enumerate(corners):cv2.circle(preview,tuple(np.rint(point).astype(int)),5,(40,240,40),-1)
            # Projected plane grid. Model foot/ankle positions are never drawn.
            inv=np.array(value['court_to_image_homography'])
            for a,b in [([.5,0],[.5,1]),([0,.5],[1,.5])]:
                q=np.array([a+[1],b+[1]])@inv.T;q=q[:,:2]/q[:,2:];cv2.line(preview,tuple(np.rint(q[0]).astype(int)),tuple(np.rint(q[1]).astype(int)),(255,200,40),2)
        preview=cv2.resize(preview,(1280,720),interpolation=cv2.INTER_AREA)
        assert cv2.imwrite(str(OUT/f'F{frame}_line_review.jpg'),preview)
        print(frame,'accepted',value is not None,{k:round(v['bin_coverage'],2) for k,v in diagnostic.items()},flush=True)
    assert anchor is not None,'Retain failed attempt; do not silently shift anchor or widen bands.'
    for r in reports:
        if r['geometry']:
            r['maximum_corner_displacement_from_anchor_px']=float(np.linalg.norm(np.asarray(r['geometry']['image_points_px'])-np.asarray(anchor['image_points_px']),axis=1).max())
    dump(OUT/'result.json',dict(cases=reports,anchor_geometry=anchor,
        geometry_supported_frames=[r['source_frame'] for r in reports if r['geometry']],
        framing_or_line_failure_frames=[r['source_frame'] for r in reports if not r['geometry']],
        tracking_predictions_created=False,independent_calibration_accuracy_established=False))
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    shutil.copy2(Path(__file__),OUT/('executed_'+Path(__file__).name))


if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
