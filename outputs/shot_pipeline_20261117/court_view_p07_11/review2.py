"""Review 2, deterministic rebuild. Original tracking/pose/labels/video are read-only.

Run with study/.venv/Scripts/python.exe -B -X utf8 and --prepare-only to
inspect all numerical outputs before rendering. --render-only resumes rendering.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
from statistics import median
import subprocess

import cv2
import numpy as np
import pandas as pd

import build_court_view as b
import far_depth_revision as far
import review2_envelope as env

OUT=b.OUT
BACKUP=OUT/'review2_before'
SERVE='observed_x_depth_serve_anchor'
LIFT='observed_x_depth_lift_held'
HOLD='offscreen_hold'
VERSION='review2_20261119'
RECHECK_NOTE='最初の回答はN1。2026-11-19 の再確認でN2と確定（名前の取り違え）'

# Refresh copied historical report text without modifying the *_before/ snapshots.
CONFIRMATION_TEXT_UPDATES={
    '25行は自動結果を本人が了承、6行は本人の指定、1行は回答の名前を追跡と照合して解釈した。':
        '25行は自動結果を本人が了承、6行は本人の指定、1行は2026-11-19の本人再確認で確定した。',
    '**第7点6（F23803）は user_answer=N1、confirmed_hitter=N2。画面外にいたのは追跡上 N2。名前の取り違えと判断。要再確認。この1行は直接確認と同等の確定根拠ではない。**':
        '**第7点6（F23803）は user_answer=N1 を履歴として残し、confirmed_hitter=N2。2026-11-19の本人再確認でN2と確定（最初の回答は名前の取り違え）。confirm_source=user_confirmed_after_recheck。**',
    '再解釈1行を除く31行の結果は review2_verification.json にも記録する。':
        '32行すべてを確認済みとして照合し、結果は review2_verification.json にも記録する。',
    '第7点6の名前は要再確認。':'',
    '第7点6は回答の名前を追跡情報から N2 と解釈しており、要再確認。':
        '第7点6（F23803）は2026-11-19の本人再確認でN2と確定。最初の回答N1は名前の取り違えとして履歴に残す。',
    '25打球は旧自動結果を本人了承、6打球は明示回答、1打球は回答を再解釈。元の回答と解釈を別列に保存。':
        '25打球は旧自動結果を本人了承、6打球は明示回答、1打球は本人再確認で確定。元の回答と確定打者・確認根拠を別列に保存。',
}


def update_confirmation_text(text):
    for old,new in CONFIRMATION_TEXT_UPDATES.items():
        text=text.replace(old,new)
    return text


def update_confirmation(summary, truth):
    """Update confirmation provenance only; keep automatic decisions and positions."""
    by_key={(r['point'],r['frame']):r for r in truth}
    for hit in summary['hitter_decisions']:
        row=by_key[hit['point'],hit['frame']]
        assert hit['hitter']==row['confirmed_hitter']
        hit['confirm_source']=row['confirm_source']
    summary['hitter_reference']='human_confirmed_hitters_p07_11.csv; all 32 hitters confirmed by user'
    summary['remaining_issues'].pop('truth',None)


def csv(name, rows):
    table=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    table.to_csv(OUT/name,index=False,encoding='utf-8-sig',lineterminator='\n')


def snapshot():
    manifest=OUT/'review2_before.json'
    if manifest.exists():return json.loads(manifest.read_text(encoding='utf-8'))
    BACKUP.mkdir(exist_ok=True)
    names=['cleaning_summary.json','hitter_auto.csv','identity_summary.json','wrist_identity.csv']
    names += [f'{prefix}_p{p:02d}.csv' for prefix in ('identity_positions','identity_map','cleaned_player_positions') for p in range(7,12)]
    for name in names:shutil.copyfile(OUT/name,BACKUP/name)
    shutil.copyfile(OUT/'hitter_auto.csv',OUT/'hitter_auto_before_review2.csv')
    inputs=set()
    for directory in ('tracking','tracking_v2','pose','labels_human'):
        inputs.update(p for p in (b.PIPE/directory).rglob('*') if p.is_file())
    inputs.add(OUT/'ball_candidates.csv');inputs.add(b.VIDEO)
    # Existing protection manifests also cover the projection and helper inputs.
    old=json.loads((OUT/'input_sha256_before.json').read_text(encoding='utf-8'))
    inputs.update(Path(p) for p in old)
    value=dict(input_sha256={str(p):b.sha(p) for p in sorted(inputs)},
               baseline_sha256={name:b.sha(BACKUP/name) for name in names},version=VERSION)
    b.dump(manifest,value)
    return value


def truth_table(before):
    # Keys are annotation row numbers within each point, never automatic-rule exceptions.
    answers={(7,8):'N1',(7,6):'N1',(8,3):'F1',(8,4):'N2',(8,8):'N1',(9,7):'N1',(10,3):'F1'}
    rows=[]
    for p,g in before.groupby('point',sort=False):
        for no,r in enumerate(g.itertuples(),1):
            answer=answers.get((p,no),'')
            confirmed=answer or r.hitter
            source='user_answered' if answer else 'auto_accepted_by_user'
            note='2026-11-19 ユーザー確認'
            if (p,no)==(7,6):
                confirmed='N2';source='user_confirmed_after_recheck'
                note=RECHECK_NOTE
            rows.append(dict(point=int(p),shot_no=no,frame=int(r.frame),side=r.side,auto_hitter=r.hitter,
                             auto_status=r.status,user_answer=answer,confirmed_hitter=confirmed,
                             confirm_source=source,note=note))
    assert len(rows)==32 and Counter(r['confirm_source'] for r in rows)=={'auto_accepted_by_user':25,'user_answered':6,'user_confirmed_after_recheck':1}
    csv('human_confirmed_hitters_p07_11.csv',rows)
    return rows


def reentry_intervals(p, fixed, mapping):
    old=pd.read_csv(BACKUP/f'identity_map_p{p:02d}.csv').fillna('')
    old=old.set_index(['source_frame','track_id'])
    result=[]
    for start in mapping[mapping.identity_status.eq('reentry_vacant_slot')].itertuples():
        g=fixed[(fixed.rally_no==start.rally_no)&(fixed.player_slot==start.player_slot)&(fixed.track_id==start.track_id)]
        g=g[g.source_frame>=start.source_frame].sort_values('source_frame')
        new=[int(f) for f in g.source_frame if (f,start.track_id) not in old.index or old.loc[(f,start.track_id),'player_slot']!=start.player_slot]
        if not new:continue
        runs=[]
        for f in new:
            if runs and f==runs[-1][-1]+1:runs[-1].append(f)
            else:runs.append([f])
        for run in runs:
            result.append(dict(point=p,rally_no=int(start.rally_no),player=b.NAMES[start.player_slot],
                               player_slot=start.player_slot,track_id=int(start.track_id),
                               start_frame=run[0],end_frame=run[-1],newly_assigned_frames=len(run),
                               assignment_start_frame=int(start.source_frame)))
    return result


def depth_for_court_y(H, image_x, court_y):
    # Homography outputs normalized court units. Solve y, retaining current x.
    return float(((H[1,0]-court_y*H[2,0])*image_x+H[1,2]-court_y*H[2,2])/(court_y*H[2,1]-H[1,1]))


def find_lifts(raw):
    """Stable preceding 12F, >6% height upward excursion, returns within 42F.

    Return band is half the onset threshold to avoid cutting a dip in half.
    Occluded or missing boxes cannot establish a lift/return. Long excursions
    are left unchanged. Every raw frame is searched, independent of shot times.
    """
    overrides={};intervals=[];i=0
    while i<len(raw):
        r=raw[i];f=int(r['source_frame'])
        prior=[z for z in raw[max(0,i-12):i] if f-12<=z['source_frame']<f]
        if len(prior)<4 or any(z['lower_occluded'] for z in prior) or r['lower_occluded']:
            i+=1;continue
        base=median(z['y2'] for z in prior);height=median(z.get('original_box_height_px',z['y2']-z['y1']) for z in prior);threshold=.06*height
        stable=max(z['y2'] for z in prior)-min(z['y2'] for z in prior)<=threshold
        if not stable or base-r['y2']<=threshold:
            i+=1;continue
        a=raw[i-1];return_index=None
        for j in range(i+1,len(raw)):
            z=raw[j]
            if z['source_frame']-f>42:break
            if z['source_frame']-raw[j-1]['source_frame']>1 or z['lower_occluded']:break
            if z['y2']>=base-.5*threshold:
                return_index=j;break
        if return_index is None:
            i+=1;continue
        z=raw[return_index];changed=[]
        for q in raw[i:return_index]:
            n=int(q['source_frame']);t=(n-a['source_frame'])/(z['source_frame']-a['source_frame'])
            depth=(1-t)*a['y2']+t*z['y2']
            overrides[n]=dict(depth_y_px=depth,status=LIFT,start_frame=f,end_frame=int(z['source_frame'])-1,
                              anchor_before_frame=int(a['source_frame']),anchor_after_frame=int(z['source_frame']))
            changed.append(n)
        intervals.append(dict(rally_no=int(r['rally_no']),start_frame=f,end_frame=int(raw[return_index-1]['source_frame']),
                              return_frame=int(z['source_frame']),duration_source_frames=int(z['source_frame'])-f,
                              corrected_frames=len(changed),threshold_px=threshold,baseline_y2_px=base,
                              anchor_before_frame=int(a['source_frame']),anchor_after_frame=int(z['source_frame'])))
        i=return_index+1
    return overrides,intervals


def far_review(p,slot,source,frames,shots):
    project,H=b.abc.geometry(b.load_module)
    baseline,details=far.clean_far(slot,source,frames)
    # Freeze E's horizontal coordinate, including the existing +/-8F median.
    baseline_by={int(r['source_frame']):r for r in baseline}
    x_ref={f:r['effective_court_x'] for f,r in baseline_by.items() if r.get('effective_court_x','')!=''}
    envelopes=env.envelope_rows(source)
    overrides={};lifts=[];serves=[]
    for rally in sorted({int(r['rally_no']) for r in source}):
        raw=sorted((r for r in source if int(r['rally_no'])==rally),key=lambda r:r['source_frame'])
        raw=[dict(r,original_box_height_px=r['y2']-r['y1'],y2=envelopes[int(r['source_frame'])]['envelope_y2_px']) for r in raw]
        ov,intervals=find_lifts(raw)
        serve_shots=[s for s in shots if s['rally_no']==rally and s['is_serve'] and s['hitter_slot']==slot]
        for shot in serve_shots:
            f=shot['frame'];at=[r for r in raw if r['source_frame']==f]
            if not at:
                serves.append(dict(point=p,player= b.NAMES[slot],frame=f,applied=False,reason='no_box_at_serve'));continue
            anchor_y=(23.77+.3)/23.77
            for r in raw:
                n=int(r['source_frame'])
                if n>f:break
                observed_y=project(H,(r['x1']+r['x2'])/2,r['y2'])[1]*23.77
                if n==f or r['lower_occluded'] or n in ov or observed_y>25.27:
                    ov[n]=dict(depth_y_px=depth_for_court_y(H,(r['x1']+r['x2'])/2,anchor_y),status=SERVE,
                               start_frame=n,end_frame=f,anchor_before_frame=f,anchor_after_frame=f)
            # A lift spanning serve uses the mandated serve anchor as its left
            # endpoint after contact; the observed return remains the right one.
            for event in intervals:
                if event['start_frame']<=f<event['return_frame']:
                    end=next(r for r in raw if r['source_frame']==event['return_frame'])
                    start=ov[f]['depth_y_px']
                    for r in raw:
                        n=int(r['source_frame'])
                        if f<n<event['return_frame']:
                            t=(n-f)/(event['return_frame']-f)
                            ov[n]=dict(depth_y_px=(1-t)*start+t*end['y2'],status=LIFT,start_frame=f+1,
                                       end_frame=event['return_frame']-1,anchor_before_frame=f,anchor_after_frame=event['return_frame'])
            serves.append(dict(point=p,rally_no=rally,player=b.NAMES[slot],frame=f,applied=True,court_y_m=24.07))
        overrides.update(ov)
        lifts += [dict(point=p,player=b.NAMES[slot],player_slot=slot,**event) for event in intervals]
    rows,details=far.clean_far(slot,source,frames,depth_overrides=overrides,x_reference=x_ref,envelopes=envelopes,speed_reset=True)
    for r in rows:
        f=int(r['source_frame']);old=baseline_by[f]
        if r['status'] in b.VALID and old.get('court_x','')!='':
            r['court_x']=old['court_x']
        r['review2_depth_correction_m']=''
        if r.get('effective_court_y','')!='' and old.get('effective_court_y','')!='':
            r['review2_depth_correction_m']=(r['effective_court_y']-old['effective_court_y'])*23.77
        if f in overrides:
            r.update(review2_anchor_before_frame=overrides[f]['anchor_before_frame'],
                     review2_anchor_after_frame=overrides[f]['anchor_after_frame'])
            # Keep exact serve anchor and time-linear lift y; do not median it
            # back toward the uncorrected neighbouring airborne boxes.
            if r['filter_status'] in (SERVE,LIFT):r['court_y']=r['effective_court_y']
    for event in lifts:
        relevant=[r for r in rows if event['start_frame']<=r['source_frame']<=event['end_frame'] and r['filter_status']==LIFT]
        event['actually_lift_held_frames']=len(relevant)
        event['max_depth_correction_m']=max([abs(r['review2_depth_correction_m']) for r in relevant if r['review2_depth_correction_m']!=''] or [0.])
        event['max_y2_correction_px']=max([abs(r['depth_y_px']-r['y2']) for r in relevant] or [0.])
    return rows,details,lifts,serves


def hold_missing(rows,fixed,segments):
    lookup={(int(r['source_frame']),r['player_slot']):r for r in rows}
    observed=set(zip(fixed.source_frame,fixed.player_slot))
    for segment in segments:
        for slot in b.SLOTS:
            last=None
            for f in range(segment['start_frame'],segment['end_frame']+1):
                r=lookup[f,slot]
                if (f,slot) in observed:
                    if r['status'] in b.VALID and r['court_x']!='':last=r
                    continue
                if last is not None:
                    # All absence of boxes is explicitly an estimate; this may
                    # also be a detector failure, not proven to be offscreen.
                    r.update(status=HOLD,filter_status=HOLD,court_x=last['court_x'],court_y=last['court_y'],
                             offscreen_anchor_frame=int(last['source_frame']),rally_no=segment['rally_no'])
                    for key in ('x1','x2','y1','y2','track_id'):r[key]=''
    return rows


def baseline_rows():
    project,H=b.abc.geometry(b.load_module);result=[]
    for p in range(7,12):
        old=pd.read_csv(OUT/'hitter_auto.csv');point=old[old.point==p]
        serves=point[point.is_serve]
        frame=int((serves if len(serves) else point).frame.iloc[0])
        rgb=list(b.source_frames(frame,frame))[0][1]
        expected=depth_for_court_y(H,960.,1.)
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        # A broad brightness mask also detects the net mesh. Detect the narrow
        # white ridge against rows +/-3px instead, in the court interior, away
        # from the net-top and service line. Keep the entire score profile.
        lo,hi=int(expected)-15,int(expected)+16
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)
        ridge=gray-.5*(np.roll(gray,3,axis=0)+np.roll(gray,-3,axis=0))
        profile=np.median(ridge[lo:hi,860:1120],axis=1)
        peak=lo+int(np.argmax(profile))
        assert profile.max()>7,(p,'white baseline ridge not resolved')
        samples=[]
        for x in range(860,1120):
            yy=np.arange(peak-2,peak+3)
            good=(hsv[yy,x,1]<85)&(hsv[yy,x,2]>155)
            weights=np.maximum(ridge[yy,x]-7,0)*good
            if weights.sum():samples.append((x,float(np.average(yy,weights=weights))))
        assert len(samples)>=70,(p,'insufficient white baseline pixels')
        data=np.asarray(samples);coef=np.polyfit(data[:,0]-960,data[:,1],1)
        residual=np.abs(data[:,1]-np.polyval(coef,data[:,0]-960))
        kept=data[residual<1.25];coef=np.polyfit(kept[:,0]-960,kept[:,1],1)
        detected=float(coef[1]);delta_m=(project(H,960.,detected)[1]-1.)*23.77
        candidates=[dict(row=lo+i,median_ridge_contrast=float(v)) for i,v in enumerate(profile)]
        result.append(dict(point=p,frame=frame,image_x_px=960.,projected_row_px=expected,detected_white_line_row_px=detected,
                           difference_px=detected-expected,difference_m=delta_m,exceeds_3px=abs(detected-expected)>3,
                           white_line_pixel_columns=len(kept),line_slope=float(coef[0]),detection_candidates_json=json.dumps(candidates)))
        debug=rgb.copy();cv2.line(debug,(600,round(expected)),(1300,round(expected)),(255,60,50),1)
        cv2.line(debug,(860,round(detected-100*coef[0])),(1120,round(detected+160*coef[0])),(0,255,255),1)
        cv2.imwrite(str(OUT/f'baseline_review2_p{p:02d}.jpg'),cv2.cvtColor(debug[lo-35:hi+35,550:1350],cv2.COLOR_RGB2BGR))
    csv('review2_baseline_rows.csv',result)
    return result


def prepare():
    import review2c as r2c
    r2c.snapshot()
    env.snapshot()
    snapshot();before=pd.read_csv(OUT/'hitter_auto_before_review2.csv').fillna('')
    truth=truth_table(before)
    summary=json.loads((BACKUP/'cleaning_summary.json').read_text(encoding='utf-8'))
    segments={p:copy.deepcopy(summary['points'][str(p)]['rendering']['segments']) for p in range(7,12)}
    tracks={};maps={};identities={};reentries=[]
    project,H=b.abc.geometry(b.load_module)
    for p in range(7,12):
        raw=pd.read_csv(b.PIPE/f'tracking_v2/point_{p:02d}/player_court_proxies.csv')
        tracks[p],maps[p],identities[p]=b.abc.relink(b.abc.mark_occlusion(raw),segments[p],
            lambda slot,source,frames:b.clean_player(slot,source,frames,far_revision=False),project,H,vacant_reentry=True)
        reentries+=reentry_intervals(p,tracks[p],maps[p])
        csv(f'identity_positions_p{p:02d}.csv',tracks[p]);csv(f'identity_map_p{p:02d}.csv',maps[p])
        print(f'p{p}: identity rebuilt',flush=True)
    b.dump(OUT/'identity_summary.json',identities)
    # Pose measurements are reused; attach the original measured features to the
    # corrected identity through source_slot. No neural inference is rerun.
    old_wrist=pd.read_csv(BACKUP/'wrist_identity.csv')
    if 'source_slot' in old_wrist:
        old_wrist['player_slot']=old_wrist.source_slot
        old_wrist=old_wrist.drop(columns='source_slot')
    wrist=b.abc.remap_wrist(old_wrist,maps)
    csv('wrist_identity.csv',wrist)
    balls=pd.read_csv(OUT/'ball_candidates.csv')
    # Preserve the exact annotation row ordering frozen in A, for shot numbers.
    hits=b.decide_hitters(before,wrist)
    hits=b.abc.combine_ball(hits,tracks,balls)
    hits=b.retrack.add_lift(hits,tracks)
    for h in hits:
        old=before[(before.point==h['point'])&(before.frame==h['frame'])].iloc[0]
        for key in ('legacy_tracking_issue','tracking_version','pose_recomputed'):
            value=old[key]
            h[key]=value.item() if isinstance(value,np.generic) else value
    csv('hitter_auto.csv',hits)
    rendered_hits=[]
    for h,t in zip(hits,truth):
        assert (h['point'],h['frame'])==(t['point'],t['frame'])
        r=dict(h,auto_hitter=t['auto_hitter'],current_auto_hitter=h['hitter'],confirmed_hitter=t['confirmed_hitter'],
               hitter=t['confirmed_hitter'],hitter_slot={v:k for k,v in b.NAMES.items()}[t['confirmed_hitter']],
               confirm_source=t['confirm_source'],shot_no=t['shot_no'])
        r.pop('ball_evidence_json',None);r.pop('context_evidence_json',None)
        rendered_hits.append(r)
    comparisons=[]
    for label,table in [('before',before.to_dict('records')),('after',hits)]:
        comparisons.append(dict(version=label,total=32,matches=sum(h['hitter']==t['confirmed_hitter'] for h,t in zip(table,truth)),
                                undecided=sum(h['status']=='undecided' for h in table),
                                wrong_decided=sum(h['status']!='undecided' and h['hitter']!=t['confirmed_hitter'] for h,t in zip(table,truth))))
    counts=[];lifts=[];serves=[];envelope_stats=[];envelope_intervals=[]
    prior_counts=pd.read_csv(env.BACKUP/'review2_far_depth_counts.csv').set_index(['point','player_slot'])
    for p in range(7,12):
        old=pd.read_csv(BACKUP/f'cleaned_player_positions_p{p:02d}.csv').fillna('')
        frames=old.source_frame.unique().tolist();fixed=far.mark_occlusion(tracks[p]);rows=[]
        for slot in b.SLOTS:
            source=fixed[fixed.player_slot==slot].to_dict('records')
            if slot.startswith('far_'):
                output,details,events,serve=far_review(p,slot,source,frames,[r for r in rendered_hits if r['point']==p])
                lifts+=events;serves+=serve
            else:
                # Legacy near geometry is unchanged except where D restores boxes.
                output,details=b.clean_player(slot,source,frames,far_revision=False,speed_reset=True)
            rows+=output
            summary['points'][str(p)]['cleaning_diagnostics'][slot]=details
        rows=hold_missing(rows,fixed,segments[p]);rows.sort(key=lambda r:(r['source_frame'],b.SLOTS.index(r['player_slot'])))
        rows=env.freeze_horizontal(p,rows)
        current=pd.DataFrame(rows)
        statistics,intervals=env.summarize(p,current,segments[p])
        envelope_stats+=statistics;envelope_intervals+=intervals
        csv(f'cleaned_player_positions_p{p:02d}.csv',current)
        for slot in b.SLOTS[2:]:
            selected=pd.Series(False,index=current.index)
            for s in segments[p]:selected |= current.source_frame.between(s['start_frame'],s['end_frame'])
            c=current[selected&current.player_slot.eq(slot)]
            a=old[old.source_frame.isin(c.source_frame)&old.player_slot.eq(slot)]
            def number(g):
                y=pd.to_numeric(g.court_y,errors='coerce')*23.77
                return int((g.status.isin(b.VALID)&(y>25.3)).sum())
            counts.append(dict(point=p,player=b.NAMES[slot],player_slot=slot,scope='displayed_all_source_frames',
                               before_above_25_3m=number(a),
                               after_above_25_3m=int(prior_counts.loc[(p,slot),'after_above_25_3m']),
                               after_offscreen_held_above_25_3m=int(prior_counts.loc[(p,slot),'after_offscreen_held_above_25_3m']),
                               review2b_above_25_3m=number(c),
                               review2b_offscreen_held_above_25_3m=number(c[c.status.eq(HOLD)])))
        summary['points'][str(p)].update(status_counts_by_player=b.count_status(rows,'status'),filter_status_counts_by_player=b.count_status(rows,'filter_status'))
        print(f'p{p}: depth and offscreen positions rebuilt',flush=True)
    csv('review2_far_depth_counts.csv',counts)
    csv('review2_lift_intervals.csv',lifts)
    csv('review2_reentry_intervals.csv',reentries)
    csv('review2c_envelope_stats.csv',envelope_stats)
    csv('review2c_envelope_intervals.csv',envelope_intervals)
    checks=[]
    baseline=baseline_rows()
    summary.update(review2=True,review2b=True,review2c=True,review2c_version='review2c_20261119',
                   review2c_envelope_stats=envelope_stats,review2c_envelope_intervals=envelope_intervals,
                   hitter_decisions=rendered_hits,
                   hitter_status_counts=dict(Counter(h['status'] for h in hits)),
                   identity={str(p):{k:v for k,v in x.items() if k!='unlinked_observations'} for p,x in identities.items()},
                   independent_hitter_accuracy_validated=False,
                   hitter_reference='human_confirmed_hitters_p07_11.csv; all 32 hitters confirmed by user',
                   review2_accuracy=comparisons,review2_reentries=reentries,review2_lifts=lifts,
                   review2_serves=serves,review2_baselines=baseline,review2_depth_counts=counts)
    summary['rules']['review2']=dict(version=VERSION,ball_ambiguity_fraction=.25,offscreen_absence_window=[-8,4],
        offscreen_ball_window=[-10,5],offscreen_ball_distance_box_width=1.5,
        serve_anchor_m=24.07,lift_threshold_height_fraction=.06,lift_max_source_frames=42,
        lift_stable_prior_frames=12,lift_return_threshold_fraction=.5,offscreen_hold='last valid observation in the same rally only',
        identity_reentry='last observation gap strictly > FPS source frames: vacant slot; otherwise original time/distance gate; minimum total distance')
    summary['rules']['lift']['correct_position']='far depth only, review2; x unchanged'
    summary['rules']['far_depth_envelope']=env.RULE
    r2c.finalize(summary)
    csv('review2_far_depth_counts.csv',summary['review2_depth_counts'])
    summary['remaining_issues']=dict(positions='Projection and identity accuracy not independently validated',
        generalization='Rules adjusted on these 32 shots; agreement is not held-out performance',
        offscreen='No box includes both out-of-view and detector failure; held point is an estimate')
    for key in ('auto_hitter_decisions','displayed_position_metrics','hitter_changes','changes_from_previous',
                'large_files','side_outside_intervals','raw_side_outside_intervals'):
        summary.pop(key,None)
    summary['requested_checks']=[f'check_review2_p{p:02d}_F{f:05d}.jpg' for p,frames in b.REVIEW2_CHECKS.items() for f in frames]+['check_review2_p07_F23865.jpg']
    b.dump(OUT/'cleaning_summary.json',summary)
    r2c.write_report(summary)
    return summary


def render(summary):
    reentry=[r for r in summary['review2_reentries'] if r['point']==7 and r['player']=='N2']
    if reentry:b.REVIEW2_CHECKS[7]=sorted(set(b.REVIEW2_CHECKS[7]+[min(r['start_frame'] for r in reentry)]))
    balls=pd.read_csv(OUT/'ball_candidates.csv')
    for p in range(7,12):
        rows=pd.read_csv(OUT/f'cleaned_player_positions_p{p:02d}.csv').fillna('').to_dict('records')
        info=summary['points'][str(p)]
        info['rendering']=b.render_point(p,info['rendering']['segments'],rows,
            [s for s in summary['hitter_decisions'] if s['point']==p],True,balls)
        b.dump(OUT/'cleaning_summary.json',summary)
        print(f'p{p}: MP4 rendered and decoded through EOF',flush=True)
    env.attach_visual_review(summary)
    b.dump(OUT/'cleaning_summary.json',summary)


def md_table(rows,columns):
    return ['|'+'|'.join(label for _,label in columns)+'|','|'+'|'.join('---' for _ in columns)+'|']+[
        '|'+'|'.join(f'{r[key]:.3f}' if isinstance(r[key],float) else str(r[key]) for key,_ in columns)+'|' for r in rows]


def write_report(s):
    a=s['review2_accuracy'];lifts=s['review2_lifts'];reentry=s['review2_reentries']
    lines=['# 第7〜11点 review2 報告（2026-11-19）','',
        '## A・B：確認ラベルと自動判定','',
        '打球番号は hitter_auto_before_review2.csv の点ごとの行順（1始まり）。human_confirmed_hitters_p07_11.csv は32行。25行は自動結果を本人が了承、6行は本人の指定、1行は2026-11-19の本人再確認で確定した。auto_hitter / auto_status は変更前の値。動画は confirmed_hitter を使用する。','',
        '**第7点6（F23803）は user_answer=N1 を履歴として残し、confirmed_hitter=N2。2026-11-19の本人再確認でN2と確定（最初の回答は名前の取り違え）。confirm_source=user_confirmed_after_recheck。**','']
    lines+=md_table(a,[('version','判定'),('total','対象'),('matches','一致'),('undecided','未決定'),('wrong_decided','決定したが不一致')])
    lines+=['','一致率は全32打球を分母とし、未決定も不一致に数える。今回は指摘例で規則を調整して同じ32打球に照合した結果であり、独立テストでの精度・汎化性能ではない。32行すべてを確認済みとして照合し、結果は review2_verification.json にも記録する。','',
        '球曖昧なら票を棄権し手首判定を許す。両最小距離がそれぞれの元枠幅の25%未満なら球曖昧。打球Fで同側1人だけ映り、その人の拡張枠に±3Fの球があれば auto_ball_single_visible。−8〜+4Fを通じ片方の枠がなく、相方が映り、−10〜+5Fに相方の元枠から幅1.5倍以内の球がなければ auto_context_offscreen（他の票より優先）。候補球ゼロもこの条件に含むため、誤検出・未検出に弱い。特定Fの判定分岐はない。','',
        '## C：枠がない人物の推定位置','',
        '同一表示ラリー内で最後に得た有効な観測枠の位置を保持し、status/filter_status=offscreen_hold。次の観測では解除。ラリーをまたがず、初観測以前は作らない。破線白抜き円と「画面外・推定」で表示し、打者間の模式線にも使う。枠欠測は画面外か検出漏れかをこの処理だけでは区別できない。','',
        '## D：再認識の原因と再割り当て','',
        '第7点ラリー2 N2：track1813の最後はF23761、track2265はF23865から候補・スロットCSVに存在し、審判除外されていない。間隔104F=1.735秒、距離0.322m。距離条件内だが従来の1秒ゲートで弾かれ、ラリー終端のF23998まで未割り当てだった。新規検出を作らず既存の候補を利用した。','',
        '同側で直前Fから人物名が欠測中なら、同時の既知trackと異なる新trackを空き名へ時間・距離ゲートなしで割り当てる。複数候補は接続数最大、最後の位置への合計距離最小。初回の対応は reentry_vacant_slot、以後は known_track。表は旧対応と比べて新たに割り当てられた連続区間（元F）である。既に接続できていた区間は含めない。','']
    lines+=md_table(reentry,[('point','点'),('rally_no','ラリー'),('player','人物'),('track_id','track'),('start_frame','開始F'),('end_frame','終了F'),('newly_assigned_frames','新規割当F数')])
    lines+=['','## E：奥側のサーブ・持ち上がり','',
        '奥側サーブの打者は接触時 court_y=24.07m を仮定する。開始〜サーブの下隠れ・検出した持ち上がり・25.27m超のFも保持。サーブの補正y2を下隠れ補間の前側アンカーに追加する。サービス規則に基づく仮定であり実測位置ではない。','',
        '全Fで、過去12Fに4観測以上ありy2幅が高さ6%以内の安定区間を探す。その中央値から高さ6%より上へずれ、42F以内に閾値半分以内へ戻る区間を、前後の観測y2で時間線形補間する。遮蔽・枠欠測をまたぐ持ち上がり判定は行わない。42F超の持続変化は補正しない。サーブをまたぐ持ち上がりは接触後の前側をサーブアンカーに置く。xはこの奥行き補正前の当該観測・従来平滑化値を保つ。サーブ・持ち上がり補正Fは、その深さを中央値で再び崩さず直接表示する。','']
    actual=[r for r in lifts if r['actually_lift_held_frames']]
    lines += [f'検出区間 {len(lifts)}、サーブ保持と重複を除き持ち上がり補正が残った区間 {len(actual)}。最長 {max([r["duration_source_frames"] for r in lifts] or [0])} 元F（上限42F）。最大奥行き補正 {max([r["max_depth_correction_m"] for r in actual] or [0.]):.3f}m、最大画像y2補正 {max([r["max_y2_correction_px"] for r in actual] or [0.]):.3f}px。区間別の根拠と補正量は review2_lift_intervals.csv。','',
        '### court_y > 25.3m の元F数','',
        '対象は各動画の表示ラリー区間内の全元F。通常補間・画面外保持を含む有効点。旧版との可用性も変わるため、値の減少だけで精度改善とは判断しない。','']
    lines+=md_table(s['review2_depth_counts'],[('point','点'),('player','人物'),('before_above_25_3m','review2前'),('after_above_25_3m','review2'),('review2b_above_25_3m','今回'),('review2b_offscreen_held_above_25_3m','今回のうち画面外保持')])
    lines+=['','件数減少と位置欠測化を区別するため、同じ表示範囲の有効F数と両版共通の有効Fでの超過数も示す。','']
    lines+=md_table(s['review2_depth_counts'],[('point','点'),('player','人物'),('review2_valid_frames','review2有効F'),('review2b_valid_frames','今回有効F'),('review2b_new_valid_frames','新規有効F'),('review2b_lost_valid_frames','欠測化F'),('common_review2_above_25_3m','共通Fのreview2超過'),('common_review2b_above_25_3m','共通Fの今回超過')])
    lost=[r for r in s['review2_depth_counts'] if r['review2b_lost_valid_frames']]
    for r in lost:
        lines+=['',f'**第{r["point"]}点{r["player"]}は有効位置が{r["review2_valid_frames"]}→{r["review2b_valid_frames"]}F、{r["review2b_lost_valid_frames"]}Fが欠測化した。超過数の減少には欠測化の影響を含む。両版共通Fでの超過数は{r["common_review2_above_25_3m"]}→{r["common_review2b_above_25_3m"]}。**']
    lines+=['','### 奥ベースラインの射影との照合','',
        '点ごとの最初のサーブF（サーブ指定のない点は最初の打球F）で中央x=860〜1120px、予測行±15pxを探索。輝度から上下3pxの平均を引いた細い白線の明るさを行中央値で検出し、HSV（S<85,V>155）の白画素を重み付けして直線を当てる。x=960pxの行を比較。ネット上端やサービス線を探索範囲から外すため、15pxを超えるずれの検出には使えない。線幅・圧縮・遮蔽誤差を含む。射影は変更しない。baseline_review2_p*.jpg の赤は射影、シアンは検出線分。差mは検出行の射影値−23.77m。','']
    lines+=md_table(s['review2_baselines'],[('point','点'),('frame','F'),('projected_row_px','射影行px'),('detected_white_line_row_px','白線行px'),('difference_px','差px'),('difference_m','差m'),('exceeds_3px','3px超')])
    warnings=[str(r['point']) for r in s['review2_baselines'] if r['exceeds_3px']]
    lines+=['',('**3pxを超えた点：'+', '.join(warnings)+'。較正のずれとして要注意。**') if warnings else '検出した白線との差が3pxを超える点はなかった。','',
        '## F：再描画と検証','',
        '5本は1920×1080、15fps、無音。元映像のPTSから原Fを照合し、原則2Fごとと確認用の厳密なFを収録。全編復号・コマ数を検査。確認画像 check_review2_p*.jpg。verify_court_view.py / verify_abc.py は review2 の独立検証へ振り分ける。旧版専用の不変条件（B/C凍結など）は今回の変更条件と両立しないため、旧版の監査関数は保持し、review2用で新規則を検証する。','',
        '入力追跡・ラベル・姿勢・球候補・原動画をSHA-256で前後照合。review2_before/ は変更直前の派生CSV、hitter_auto_before_review2.csv は打者表の元バイトを保存。新規の256KiB超CSVはgit除外。commit/pushは実行しない。','',
        '## 残る問題','',
        '人物再接続は空き名の規則に基づき、外見で本人を確認したものではない。同じtrack内部の人物取り違えは修正しない。画面外保持は移動を表せず、模式線は球軌道ではない。下隠れの長い補間は見えない前後移動を消す。短い本当の前後移動とジャンプはy2だけでは完全に識別できない。サーブ時ベースライン位置は仮定で、投影・選手位置の精度は独立参照で未検証。指摘32例への一致を研究上の汎化性能とみなさない。','']
    lines += ['## 表示上の自動値','',
        '動画帯の「自動: N?」等はAに保存した変更前auto_hitterを示す。改訂後の自動判定はhitter_auto.csvに保存し、正解と区別して照合する。人物再認識Dにより第7点F23934ではN2も映るようになったため、改訂後は単独可視規則ではなく通常のauto_ballでN1となる。','',
        'review2時点では25.3m超が第7点F1に411F、第8点F1に140F、第11点F2に139F残った。今回との差は上表。件数だけでは精度を評価できない。','']
    lines+=['## G：review2b 奥側枠下端の上側包絡','',
        'far選手・ラリーごとに、高さ棄却条件に該当しない元枠（下隠れも含む）の±15元Fのy2を集め、3観測以上なら90パーセンタイル（線形補間）と元y2の大きい方を用いる。3観測未満および高さ棄却枠自身は元値を保つ。元の枠座標・高さ棄却条件は変更しない。このy2を速度ゲート前の非遮蔽奥行き、下隠れアンカー、持ち上がり判定へ渡す。サーブアンカーが優先する。0.5m以上前へ補正した通常観測はobserved_x_depth_envelope、0.5m未満はobservedを保つ。既存の±8Fの深さ中央値は継続する。','',
        'envelope_shift_mは同じ画像xで元y2と包絡y2を射影した前向き差（正値、m）。下隠れ・別アンカー優先・速度棄却の行にも入力段階の候補差を記録する。下表は表示ラリー内の非遮蔽・高さ適格・速度採用・サーブ/持ち上がり非優先の観測だけを数え、1e-9m超を適用とする（envelope_applied）。補間・画面外保持への間接効果は含まない。各点の全far選手を示す。','']
    lines+=md_table(s['review2b_envelope_stats'],[('point','点'),('player','人物'),('applied_frames','適用F数'),('status_envelope_frames','うち0.5m以上F数'),('median_shift_m','中央値m'),('p95_shift_m','95%m'),('max_shift_m','最大m')])
    lines+=['','### 1m以上の連続補正区間：長い順の上位5件','',
        '上記の直接適用観測で1m以上が連続する区間を同一ラリー・人物ごとに抽出し、元F数降順で選定。確認画像は中央Fをファイル名とし、元映像（動画左側のカメラ）のF1/F2を拡大、開始・中央・終了を並べる。行ごとの切り出し位置は固定。橙は元枠下端、シアンは包絡下端。','']
    lines+=md_table(s['review2b_checks'],[('point','点'),('player','人物'),('start_frame','開始F'),('end_frame','終了F'),('duration_source_frames','長さF'),('midpoint_frame','中央F'),('check_image','確認画像'),('visual_assessment','画像確認')])
    lines+=['','包絡は±15F（約±0.25秒）内の大きいy2に寄せるため、本当の後退の開始を遅らせたり、短い後退を小さくする可能性がある。開始・中央・終了の静止画だけでは動きの保存を証明できず、独立した足元ラベルでの時系列評価は未実施。極端な奥位置が減っても位置精度向上・真の後退の保存を保証しない。','',
        'review2b_before/ は今回直前のreview2派生出力を保存。手前選手の既存列は全行一致、両版で表示可能なfar横座標は全行固定して検証する。新しく有効化されたFの横座標は従来の観測x処理を使用する。深さが棄却され欠測化したFでは配置点を表示しない。第11点F2のF32946〜33000は速度棄却で、次の採用点までの間隔が従来の30F補間上限を超えたため欠測になる。速度・高さ・補間上限を今回の改善に合わせて緩めることはしていない。','']
    (OUT/'review2_report.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    readme=['# match2 第7〜11点：人の確認を反映したコート図（review2）','',
        '5本の point_07〜11.mp4 は打者に human_confirmed_hitters_p07_11.csv の confirmed_hitter を使用し「確認済」と表示する。自動判定と異なる打球には自動名も表示。第7点6（F23803）は2026-11-19の本人再確認でN2と確定。最初の回答N1は名前の取り違えとして履歴に残す。時刻・側・ラリーは既存の本人ラベル。位置・人物自動対応・球候補の精度は未検証。','',
        f'正解表32行との一致：変更前 {a[0]["matches"]}/32（未決定{a[0]["undecided"]}）、変更後 {a[1]["matches"]}/32（未決定{a[1]["undecided"]}）。同じ例で規則を調整した照合であり、独立評価ではない。','',
        '## A〜E の規則','',
        '- A：25打球は旧自動結果を本人了承、6打球は明示回答、1打球は本人再確認で確定。元の回答と確定打者・確認根拠を別列に保存。番号は点内の行順。',
        '- B：球曖昧は棄権し手首が決まれば採用。両距離が各枠幅25%未満なら球曖昧。単独可視枠内の球、片方欠測かつ相方の近くに球なし、の一般規則を追加。hitter_auto.csvは自動結果のまま保持。rule_version=review2_20261119。',
        '- C：枠欠測中はラリー内の最終観測位置を保持（offscreen_hold）。右図の破線白抜き円＋画面外・推定。模式線にも使用。次の観測で解除。',
        '- D：同側の人物名が欠測中に非審判の新trackが現れたら、既知の相方との重複を禁じて空き名へ対応。時間・距離ゲートを外し、競合候補は前位置への距離最小。初接続はreentry_vacant_slot。',
        '- E：奥サーブ時24.07mをアンカーとする。開始〜サーブの下隠れ・持ち上がり・25.27m超も保持。枠下端が安定値より高さ6%上がり42F以内に戻る区間は前後y2で時間線形補間。横位置は従来観測のまま。奥側のみ。射影は変更しない。','',
        '既存の下隠れ判定（同Fの手前枠との横重なりが奥枠幅10%以上、手前y1<奥y2）、前後アンカー補間、従来速度・高さフィルタは継続。サーブ保持と持ち上がりは優先し、補正値を中央値で崩さない。横には追加平滑化をしない。打球者間の破線は説明用で、球軌道・バウンド・3D位置ではない。','',
        'review2b：奥側だけ、同一選手・ラリー内の高さ適格枠（下隠れ含む）の±15元F・90パーセンタイルを足元y2の包絡とする。3観測未満は変更せず、元y2より小さくしない。速度判定・下隠れアンカー・持ち上がり判定へ適用し、サーブアンカーを優先する。0.5m以上の直接補正観測はobserved_x_depth_envelope、補正量はenvelope_shift_m。手前側と既存横位置は保持する。短い本当の後退を抑える可能性があり、長い1m以上補正区間の拡大画像と統計をreview2_report.mdに記録する。','',
        '## 再実行','',
        '```powershell',
        '& study/.venv/Scripts/python.exe -B -X utf8 outputs/shot_pipeline_20261117/court_view_p07_11/build_court_view.py --review2',
        '& study/.venv/Scripts/python.exe -B -X utf8 outputs/shot_pipeline_20261117/court_view_p07_11/verify_court_view.py',
        '```','',
        '球候補・姿勢は保存済み出力を使用。元動画は59.94fps、ffmpegは C:/ffmpeg/bin/ffmpeg.exe。表示動画は1920×1080、15fps、無音、原則2元Fごと。正確な元F対応は video_source_frame_map_p*.csv。','',
        '詳細な精度、再認識原因と割り当て区間、奥行き件数、白線との差、持ち上がり補正統計、限界は [review2_report.md](review2_report.md)。検証結果は review2_verification.json。','',
        '入力tracking_v2/・labels_human/・pose/・元動画は読み取りのみ。変更前の派生表はreview2_before/、打者はhitter_auto_before_review2.csv。大きなCSVと画像・動画はgit除外。commit/pushはしない。','']
    (OUT/'README.md').write_text('\n'.join(readme),encoding='utf-8',newline='\n')


def ignore_large():
    path=OUT/'.gitignore';content=path.read_text(encoding='utf-8')
    big=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.stat().st_size>256*1024]
    relative=[p.relative_to(b.ROOT).as_posix() for p in big]
    checked=subprocess.run(['git','check-ignore','-z','--stdin'],input=('\0'.join(relative)+'\0').encode(),cwd=b.ROOT,capture_output=True)
    assert checked.returncode in (0,1)
    ignored=set(checked.stdout.decode().split('\0'))
    additions=['/'+p.relative_to(OUT).as_posix() for p,rel in zip(big,relative) if rel not in ignored]
    if additions:path.write_text(content+'\n# Review2 regenerable artifacts over 256KiB\n'+'\n'.join(additions)+'\n',encoding='utf-8')


def change_list(summary):
    """Include regenerated/ignored artifacts that git status intentionally omits."""
    result={}
    prefix=OUT.relative_to(b.ROOT).as_posix()+'/'
    for args,kind in [(['diff','--name-only'],'変更'),(['ls-files','--others','--exclude-standard'],'追加')]:
        names=subprocess.check_output(['git',*args],cwd=b.ROOT).decode('utf-8').splitlines()
        for name in names:
            if name.startswith(prefix):result[name[len(prefix):]]=kind
    for pattern in ('point_*.mp4','cleaned_player_positions_p*.csv','identity_positions_p*.csv',
                    'identity_map_p*.csv','wrist_identity.csv','video_source_frame_map_p*.csv',
                    'check_review2_*.jpg','check_review2b_*.jpg','check_review2c_*.jpg','review2b_before/*','review2c_before/*','baseline*review2*.jpg','review2_before/*',
                    '*review2.log','*review2c.log','review2_changed_files.md'):
        for path in OUT.glob(pattern):
            if path.is_file():result.setdefault(path.relative_to(OUT).as_posix(),'生成・再生成')
    for p,info in summary['points'].items():
        rendering=info['rendering']
        for check in rendering['checks']:result.setdefault(check['file'],'再生成')
        for segment in rendering['segments']:
            name=f'decode_logs/F{segment["start_frame"]}_{segment["end_frame"]}_step1.log'
            result[name]='再生成'
        frames={7:[23422,23548],8:[25470],10:[30633],11:[32999]}.get(int(p),[])+b.REVIEW2_CHECKS.get(int(p),[])
        if int(p)==7:frames+=[23865]
        for f in frames:
            name=f'check_abc_p{int(p):02d}_F{f}.jpg'
            if (OUT/name).exists():result[name]='生成・再生成'
    for event in summary['review2_baselines']:
        f=event['frame'];result[f'decode_logs/F{f}_{f}_step1.log']='再生成'
    result['review2_changed_files.md']='追加'
    lines=['# review2 変更・追加・再生成ファイル一覧','',
           'すべて court_view_p07_11/ 内。生成・再生成にはgit除外された大容量ファイルと、再出力して内容が同一だったファイルも含む。tracking_v2/、labels_human/、pose/、元動画は変更なし。','',
           '|種別|相対ファイル名|','|---|---|']
    lines += [f'|{kind}|{name}|' for name,kind in sorted(result.items())]
    (OUT/'review2_changed_files.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


def main():
    import review2c as c
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--render-only',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args([] if '--review2' in __import__('sys').argv else None)
    if args.verify:
        from verify_review2 import main as verify
        verify();return
    s=json.loads((OUT/'cleaning_summary.json').read_text(encoding='utf-8')) if args.render_only else prepare()
    if args.render_only:
        truth=truth_table(pd.read_csv(OUT/'hitter_auto_before_review2.csv').fillna(''))
        update_confirmation(s,truth)
    c.finalize(s)
    csv('review2_far_depth_counts.csv',s['review2_depth_counts'])
    b.dump(OUT/'cleaning_summary.json',s)
    ignore_large()
    if not args.prepare_only:
        render(s)
        c.extract_checks(s)
        b.dump(OUT/'cleaning_summary.json',s)
        from verify_review2 import main as verify
        verify()
    c.write_report(s)
    change_list(s)
    csv('file_inventory.csv',b.abc.inventory())
    print(json.dumps(dict(accuracy=s['review2_accuracy'],reentries=s['review2_reentries'],lift_intervals=len(s['review2_lifts'])),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
