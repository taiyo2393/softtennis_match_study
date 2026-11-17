"""Offline local trajectory consensus and camera-change abstention.

Only withhold raw detections. Never output interpolated coordinates.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'temporal_guard'
sys.path.insert(0,str(ROOT.parent/'tracking_quality_20261102'))
from ball_alignment import SOURCE,read,evaluate
BASE=ROOT.parent/'tracking_quality_20261102/ball_aligned'
LABEL=SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')


def init():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
    save('protocol.json',dict(
        algorithm='offline local line consensus; preserve another supported motion branch; abstain on global image transitions',
        window_frames=30,model_inlier_px=16,min_model_points=6,min_model_span_frames=12,
        own_branch_min_points=4,bridge_min_points_each_side=2,pair_min_dt_frames=3,
        candidate_deviation_thresholds_px=[64,96,128],camera_options=[False,True],
        camera=dict(lag_frames=12,pixel_mean_BRG_difference=24,changed_fraction=.25,expand_before=12,expand_after=4),
        selection='Maximize removed validation nonlocalized outputs with zero lost localized TP; ties fewer total outputs withheld, higher deviation threshold, no camera abstention.',
        evaluation='Raw aligned baseline for component comparison; full two-clip preview may compose after previous glove guard, reported separately.',
        label_sha256=sha(LABEL),fps_reference=59.94005994,
        limitations='Offline lookahead up to 0.5s; same-match repeatedly used development data, not new held-out accuracy. Camera abstention can hide real visible balls during editing.'))
    save('user_reported_cases.json',dict(candidate_id='match1_motion_pilot_018',local_frames=[208,229,230],
        report='User identified player-hand false detections and requested temporal consistency.',
        role='Development regression cases, not independent test labels.'))


def camera_features(cid):
    path=OUT/'camera_features'/f'{cid}.json'
    if path.exists():return json.loads(path.read_text())['changed_fraction']
    rows=read(BASE/cid/'balltrack_predictions.csv');frames=SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'
    images=[];scores=[]
    for t in range(len(rows)):
        im=cv2.imread(str(frames/f'{t:06d}.jpg'),cv2.IMREAD_REDUCED_COLOR_8);assert im is not None
        im=cv2.resize(im,(160,90),interpolation=cv2.INTER_AREA)[8:82].astype(np.int16)
        if t<12:score=0.
        else:score=float((np.abs(im-images[t%12]).mean(2)>24).mean())
        if t<12:images.append(im)
        else:images[t%12]=im
        scores.append(score)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(dict(candidate_id=cid,changed_fraction=scores,baseline_sha256=sha(BASE/cid/'balltrack_predictions.csv'))),encoding='utf-8')
    return scores


def camera_mask(scores,enabled):
    mask=np.zeros(len(scores),bool)
    if enabled:
        for t,score in enumerate(scores):
            if score>.25:mask[max(0,t-12):min(len(scores),t+5)]=True
    return mask


def filter_rows(rows,scores,radius,camera):
    n=len(rows);mask=camera_mask(scores,camera)
    segment=np.cumsum(mask.astype(int))
    times=np.array([int(r['local_frame']) for r in rows if int(r['visible']) and not mask[int(r['local_frame'])]],int)
    positions=np.array([[float(rows[t]['x_px']),float(rows[t]['y_px'])] for t in times],float).reshape(-1,2)
    output=[dict(r,temporal_guard_flag=0,temporal_reason='',temporal_model_error_px='',
                 temporal_camera_score=scores[i]) for i,r in enumerate(rows)]
    diagnostics=[]
    for t,r in enumerate(rows):
        if not int(r['visible']):continue
        if mask[t]:
            output[t].update(visible=0,x_px=0,y_px=0,temporal_guard_flag=1,temporal_reason='camera_transition_abstention')
            diagnostics.append(dict(local_frame=t,reason='camera_transition_abstention'))
            continue
        nearby=(np.abs(times-t)<=30)&(segment[times]==segment[t])
        ts=times[nearby];xy=positions[nearby]
        if len(ts)<7:continue
        aa,bb=np.triu_indices(len(ts),1)
        good=(ts[bb]-ts[aa]>=3)
        aa,bb=aa[good],bb[good]
        if not len(aa):continue
        velocities=(xy[bb]-xy[aa])/(ts[bb]-ts[aa])[:,None]
        at_target=xy[aa]+velocities*(t-ts[aa])[:,None]
        residual=np.linalg.norm(xy[None,:,:]-(at_target[:,None,:]+velocities[:,None,:]*(ts[None,:,None]-t)),axis=2)
        inlier=residual<=16
        count=inlier.sum(1)
        spans=np.where(inlier,ts[None,:],-10**9).max(1)-np.where(inlier,ts[None,:],10**9).min(1)
        before=(inlier&(ts[None,:]<t)).sum(1);after=(inlier&(ts[None,:]>t)).sum(1)
        qualified=(count>=6)&(spans>=12)&(before>=2)&(after>=2)
        if not qualified.any():continue
        model=np.flatnonzero(qualified)[np.argmax(count[qualified])]
        truth_point=np.array([float(r['x_px']),float(r['y_px'])])
        error=float(np.linalg.norm(truth_point-at_target[model]))
        # A second plausible segment at a hit/bounce is preserved even if the majority fits another line.
        own_support=((np.linalg.norm(at_target-truth_point,axis=1)<=16)&(count>=4)).any()
        if error>radius and not own_support:
            output[t].update(visible=0,x_px=0,y_px=0,temporal_guard_flag=1,temporal_reason='unsupported_excursion',temporal_model_error_px=error)
            diagnostics.append(dict(local_frame=t,reason='unsupported_excursion',error_px=error,
                model_points=int(count[model]),support_before=int(before[model]),support_after=int(after[model])))
    for r,p in zip(rows,output):
        p.update(input_x_px=r['x_px'],input_y_px=r['y_px'],input_visible=r['visible'])
        if int(p['temporal_guard_flag']):p['position_source']='withheld_temporal_uncertainty'
        else:assert all(p[k]==r[k] for k in ('visible','x_px','y_px'))
    return output,diagnostics


def phase_predictions(phase,radius,camera,write=False):
    nums=range(1,11) if phase=='validation' else range(11,21)
    predictions={};all_diagnostics=[]
    for num in nums:
        cid=f'match1_motion_pilot_{num:03d}';rows=read(BASE/cid/'balltrack_predictions.csv')
        out,diag=filter_rows(rows,camera_features(cid),radius,camera)
        predictions[cid]={int(r['local_frame']):r for r in out}
        all_diagnostics.extend(dict(candidate_id=cid,**r) for r in diag)
        if write:
            dest=OUT/'component_predictions'/cid;dest.mkdir(parents=True,exist_ok=True)
            with (dest/'balltrack_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    return predictions,all_diagnostics


def extract(phase):
    if phase=='test':assert (OUT/'selection.json').exists()
    for i in (range(1,11) if phase=='validation' else range(11,21)):
        cid=f'match1_motion_pilot_{i:03d}';s=camera_features(cid)
        print(cid,'camera changed max',round(max(s),3),flush=True)


def select():
    assert not (OUT/'selection.json').exists()
    labels=[r for r in read(LABEL) if r['evaluation_split']=='validation']
    baseline={cid:{int(r['local_frame']):r for r in read(BASE/cid/'balltrack_predictions.csv')} for cid in sorted({r['candidate_id'] for r in labels})}
    bm=evaluate(labels,baseline,'validation');reports=[]
    for radius in (64,96,128):
        for camera in (False,True):
            pred,diag=phase_predictions('validation',radius,camera);m=evaluate(labels,pred,'validation')
            lost=bm['counts']['localized_tp_16px']-m['counts'].get('localized_tp_16px',0)
            removed=(bm['counts']['pred_visible']-bm['counts']['localized_tp_16px'])-(m['counts']['pred_visible']-m['counts'].get('localized_tp_16px',0))
            r=dict(radius=radius,camera=camera,metrics=m,lost_localized=lost,removed_nonlocalized=removed,withheld_frames=len(diag));reports.append(r)
            print(json.dumps(r),flush=True)
    eligible=[r for r in reports if r['lost_localized']==0 and r['removed_nonlocalized']>0]
    chosen=max(eligible,key=lambda r:(r['removed_nonlocalized'],-r['withheld_frames'],r['radius'],-int(r['camera']))) if eligible else None
    save('selection.json',dict(config=chosen,baseline=bm,validation_grid=reports,protocol_sha256=sha(OUT/'protocol.json'),
        warning='Chosen using validation only. User-reported test errors already motivated this development; test is not independent.'))
    print('chosen',json.dumps(chosen),flush=True)


def test():
    chosen=json.loads((OUT/'selection.json').read_text())['config'];assert chosen is not None
    labels=[r for r in read(LABEL) if r['evaluation_split']=='test']
    pred,diag=phase_predictions('test',chosen['radius'],chosen['camera'],write=True)
    baseline={cid:{int(r['local_frame']):r for r in read(BASE/cid/'balltrack_predictions.csv')} for cid in sorted(pred)}
    result=dict(baseline=evaluate(labels,baseline,'test'),temporal=evaluate(labels,pred,'test'),config=chosen,
        diagnostics=diag,selection_sha256=sha(OUT/'selection.json'),warning='Reused development test, not independent external performance.')
    save('test_results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='diagnostics'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step',choices=['init','extract-validation','extract-test','select','test']);a=p.parse_args()
    if a.step=='init':init()
    elif a.step=='select':select()
    elif a.step=='test':test()
    else:extract(a.step.removeprefix('extract-'))
