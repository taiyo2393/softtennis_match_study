"""Conservative yellow-component veto on temporal withholding, for this footage.

This is preservation of an observed candidate, never interpolation or a new
ground-truth label. Never override a camera-transition abstention.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from temporal_consensus_guard import ROOT,OUT,SOURCE,BASE,read,sha,camera_features,filter_rows

DEST=OUT/'appearance_veto_partial_20261103'
AUDIT=OUT/'suppression_audit_20261103'


def evidence(im,x,y):
    x=int(round(float(x)));y=int(round(float(y)))
    left=max(0,x-10);top=max(0,y-10)
    crop=im[top:min(im.shape[0],y+11),left:min(im.shape[1],x+11)]
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    mask=((hsv[:,:,0]>=20)&(hsv[:,:,0]<=42)&(hsv[:,:,1]>=80)&(hsv[:,:,2]>=110)).astype(np.uint8)
    count,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    options=[]
    for k in range(1,count):
        area=int(stats[k,cv2.CC_STAT_AREA]);cx,cy=centers[k]
        distance=float(np.hypot(left+cx-x,top+cy-y))
        if area>=4 and distance<=8:options.append(dict(area=area,distance_px=distance,centroid=[float(left+cx),float(top+cy)]))
    return dict(protect=bool(options),yellow_pixels=int(mask.sum()),components=options)


def protect(input_rows,output_rows,cid):
    result=[];rescued=[]
    for r,p in zip(input_rows,output_rows):
        n=dict(p,appearance_protection=0,temporal_warning=p.get('temporal_reason',''),yellow_component_evidence='')
        if int(p.get('temporal_guard_flag',0)) and p['temporal_reason']=='unsupported_excursion':
            t=int(r['local_frame']);path=SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{t:06d}.jpg'
            im=cv2.imread(str(path));assert im is not None
            ev=evidence(im,r['x_px'],r['y_px']);n['yellow_component_evidence']=json.dumps(ev)
            if ev['protect']:
                for key in ('x_px','y_px','visible','confidence','position_source'):
                    if key in r:n[key]=r[key]
                n.update(temporal_guard_flag=0,appearance_protection=1,temporal_reason='',
                         temporal_warning='retained_despite_trajectory_disagreement_yellow_component')
                rescued.append(dict(candidate_id=cid,local_frame=t,x_px=r['x_px'],y_px=r['y_px'],evidence=ev))
        result.append(n)
    return result,rescued


def main():
    DEST.mkdir(exist_ok=True)
    assert not (DEST/'protocol.json').exists(),'Preserve completed experiment'
    protocol=dict(hypothesis='Preserve a directly visible yellow-ball-like component even when a majority trajectory disagrees.',
        trigger='AI review of all 35 temporal withholdings found plausible real balls at 002/318,014/626,014/630.',
        hsv=dict(h_min=20,h_max=42,s_min=80,v_min=110),patch_radius_px=10,min_component_area_px=4,
        max_centroid_distance_px=8,action='Restore only the input model coordinate; retain temporal warning; never override camera transition.',
        scope='Match-specific conservative preservation, not a universal ball classifier; visual evidence does not certify target-ball identity.',
        evaluation='Development counterexamples; no fresh held-out evaluation or source-label modification.')
    (DEST/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    cases=json.loads((AUDIT/'cases.json').read_text(encoding='utf-8'));inspection=[]
    for c in cases:
        cid=c['candidate_id'];t=c['local_frame']
        im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{t:06d}.jpg'));assert im is not None
        inspection.append(dict(case_id=c['case_id'],candidate_id=cid,local_frame=t,
            **evidence(im,c['withheld_x_px'],c['withheld_y_px'])))
    (DEST/'all_35_candidate_appearance.json').write_text(json.dumps(inspection,indent=2),encoding='utf-8')
    reports=[]
    for cid in ('match1_motion_pilot_002','match1_motion_pilot_014','match1_motion_pilot_018'):
        glove=ROOT/'glove_guard/continuous_predictions'/cid/'balltrack_predictions.csv'
        input_path=glove if glove.exists() else BASE/cid/'balltrack_predictions.csv'
        rows=read(input_path)
        # Freeze the same scene-aware radius128 policy used in the prior video.
        old,diags=filter_rows(rows,camera_features(cid),128,True)
        new,rescued=protect(rows,old,cid)
        folder=DEST/'predictions'/cid;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'balltrack_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(new[0]));writer.writeheader();writer.writerows(new)
        reports.append(dict(candidate_id=cid,frames=len(rows),input_path=str(input_path.resolve()),input_sha256=sha(input_path),
                            original_withheld=len(diags),protected=rescued))
        if cid.endswith('018'):
            assert all(int(new[t]['visible'])==0 for t in (208,229,230))
            assert all(new[t]['x_px']==rows[t]['x_px'] and new[t]['y_px']==rows[t]['y_px'] for t in (219,220))
    for cid,t in (('match1_motion_pilot_014',626),('match1_motion_pilot_014',630)):
        data=read(DEST/'predictions'/cid/'balltrack_predictions.csv')
        assert int(data[t]['appearance_protection'])==1 and int(data[t]['visible'])==1,(cid,t)
    summary=dict(protected_in_all_35=[r['case_id'] for r in inspection if r['protect']],reports=reports,
        user_regressions_208_229_230_still_withheld=True,two_held_ball_counterexamples_preserved=True,
        unresolved_counterexample=dict(candidate_id='match1_motion_pilot_002',local_frame=318,
            reason='Ball-like moving image is desaturated/brownish and has no component in the fixed yellow range. No threshold widening to force success.'),
        safe_for_blanket_adoption=False,
        source_predictions_not_changed=True,source_annotations_not_changed=True,
        warning='Protection means color support, not independent correctness or unique target-ball identity.')
    (DEST/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
