"""Camera-specific exploratory phase proposals; never confirms contact/outcome."""
import json,math
from pathlib import Path
import numpy as np
from extract_serve_start_features import SOURCE,ROOT,sha
from propose_serve_starts import tracks,propose,describe

OUT=ROOT/'serve_phase_pilot_v3'
PARAMS=dict(court_envelope_x=[-.25,1.25],court_envelope_y=[-.2,1.2],
    selection='two tallest complete boxes on each court half; ambiguous or missing sides are not imputed',
    local_stationarity_window_sec=.6,min_stationarity_samples=12,min_sample_coverage=.7,
    max_stationary_displacement_by_person_height=.2,server_baseline_distance_max=.18,
    reset_continuous_quiet_sec=2.,ready_context_hold_sec=1.,unknown_is_stationary=False,
    motion_anchor='mean visible hips, confidence>=0.35; missing hips are not stationary; link hips, not arm-dependent box center',
    labels_used=False,serve_validity_inferred=False,
    caution='Fixed-camera development heuristic; box-bottom homography used only as a coarse ROI, not physical position ground truth. Offline gesture/audio proposals use future context.')

def court_frames(frames,H):
    out=[]
    for frame in frames:
        sides=[[],[]]
        for p in frame['people']:
            x1,y1,x2,y2=p['box'];v=H@np.array([x1+x2,2*y2,1.]);x,y=v[:2]/v[2]
            if not (-.25<=x<=1.25 and -.2<=y<=1.2):continue
            sides[int(y>=.5)].append(dict(p,court_roi_xy=[float(x),float(y)]))
        chosen=[p for side in sides for p in sorted(side,key=lambda p:p['box'][3]-p['box'][1],reverse=True)[:2]]
        out.append(dict(local_frame=frame['local_frame'],people=chosen,side_counts=[min(2,len(s)) for s in sides]))
    return out

def phase_tracks(frames):
    states={};next_id=0
    for frame in frames:
        t=frame['local_frame'];points=[]
        for p in frame['people']:
            d=describe(p,t)
            if d is None:continue
            kp=np.asarray(p['keypoints']);hips=kp[[11,12]];hips=hips[hips[:,2]>=.35]
            ankles=kp[[15,16]];ankles=ankles[ankles[:,2]>=.35]
            d['motion_anchor']=np.mean(hips[:,:2],axis=0).tolist() if len(hips) else None
            if len(ankles):d['foot']=np.mean(ankles[:,:2],axis=0).tolist()
            points.append(d)
        links=[]
        for i,p in enumerate(points):
            for tid,rows in states.items():
                q=rows[-1]
                if t-q['local_frame']>12:continue
                if p['motion_anchor'] is None or q['motion_anchor'] is None:continue
                dist=float(np.linalg.norm(np.asarray(p['motion_anchor'])-q['motion_anchor'])/max(p['height'],q['height']))
                if dist<.5:links.append((dist,i,tid))
        used=set();assigned=set()
        for _,i,tid in sorted(links):
            if i in used or tid in assigned:continue
            states[tid].append(points[i]);used.add(i);assigned.add(tid)
        for i,p in enumerate(points):
            if i not in used:states[next_id]=[p];next_id+=1
    return states

def context_series(frames,fps):
    # IDs here are short-term geometric links; no player-identity claim.
    tdata=phase_tracks(frames);by_time={}
    for tid,rows in tdata.items():
        for row in rows:by_time.setdefault(row['local_frame'],[]).append((tid,row))
    series=[];window=round(PARAMS['local_stationarity_window_sec']*fps)
    for frame in frames:
        t=frame['local_frame'];current=by_time.get(t,[]);motions=[];coverage=[]
        complete=frame['side_counts']==[2,2] and len(current)==4
        for tid,p in current:
            prior=[q for q in tdata[tid] if t-window<=q['local_frame']<=t and q['motion_anchor'] is not None]
            feet=np.array([q['motion_anchor'] for q in prior]);height=float(np.median([q['height'] for q in prior])) if prior else 1.
            cover=len(prior)/(window/2+1);coverage.append(cover)
            if len(prior)<PARAMS['min_stationarity_samples'] or cover<PARAMS['min_sample_coverage']:
                complete=False;motions.append(None);continue
            motions.append(float(np.max(np.linalg.norm(feet-np.median(feet,axis=0),axis=1))/height))
        quiet=bool(complete and all(m is not None and m<=PARAMS['max_stationary_displacement_by_person_height'] for m in motions))
        series.append(dict(local_frame=t,four_observed=frame['side_counts']==[2,2],quiet_evidence_complete=complete,
            four_player_quiet=quiet,motion_by_height=motions,coverage=coverage))
    return series,tdata

def phase_filter(proposals,series,fps,H,tdata):
    # A false quiet reset is possible even during a slow rally. No fixed lockout.
    lookup={r['local_frame']:r for r in series};phase='unknown';last_stop=-1;last_t=-2;quiet_start=None;last_quiet=None
    transitions=[];evaluated=[];accepted=[]
    for p in sorted(proposals,key=lambda p:p['raised_first_local_frame']):
        first=p['raised_first_local_frame'];decision_frame=first-(first%2)-2
        # Process past context only up to the gesture; using post-contact motion
        # as a requirement would incorrectly reject aces and short attempts.
        for t in sorted(k for k in lookup if last_t<k<=decision_frame):
            r=lookup[t]
            if phase=='activity_or_unresolved' and t<=last_stop:
                quiet_start=None;continue
            if r['four_player_quiet']:
                last_quiet=t
                if quiet_start is None:quiet_start=t
                needed=PARAMS['reset_continuous_quiet_sec']*fps if phase=='activity_or_unresolved' else 0
                if t-quiet_start>=needed and phase!='ready_candidate':
                    phase='ready_candidate';transitions.append(dict(local_frame=t,state=phase,reason='four_player_quiet_context',quiet_since=quiet_start))
            else:
                quiet_start=None
                moving=any(m is not None and m>PARAMS['max_stationary_displacement_by_person_height'] for m in r['motion_by_height'])
                expired=last_quiet is None or t-last_quiet>PARAMS['ready_context_hold_sec']*fps
                if phase=='ready_candidate' and (moving or expired):
                    phase='unknown';transitions.append(dict(local_frame=t,state=phase,reason='quiet_context_lost'))
        last_t=max(last_t,decision_frame)
        rows=tdata.get(p['person_track_id'],[])
        initial=[r for r in rows if first<=r['local_frame']<=first+6]
        if initial:
            x,y=np.median([r['foot'] for r in initial],axis=0);v=H@np.array([2*x,2*y,1.]);xy=v[:2]/v[2]
            baseline_distance=float(min(abs(xy[1]),abs(1-xy[1])))
        else:baseline_distance=None
        context=lookup.get(decision_frame)
        baseline=baseline_distance is not None and baseline_distance<=PARAMS['server_baseline_distance_max']
        ready=bool(phase=='ready_candidate' and last_quiet is not None and decision_frame-last_quiet<=PARAMS['ready_context_hold_sec']*fps)
        accepted_flag=bool(ready and baseline)
        reason='accepted_start_candidate' if accepted_flag else 'activity_lock' if phase=='activity_or_unresolved' else 'not_near_baseline' if not baseline else 'missing_or_moving_four_player_context'
        row={k:v for k,v in p.items() if k!='evidence_person_boxes'}
        row.update(phase_before=phase,baseline_distance_roi=baseline_distance,quiet_context_frame=decision_frame,
            four_player_quiet=bool(context and context['four_player_quiet']),decision=reason,accepted_candidate=accepted_flag,
            last_four_player_quiet_frame=last_quiet,quiet_evidence_age_sec=(decision_frame-last_quiet)/fps if last_quiet is not None else None,
            outcome='unknown',point_id=None)
        row['review_action']='retain_start_candidate' if accepted_flag else 'suppress_during_activity' if reason=='activity_lock' else 'defer_insufficient_start_evidence'
        evaluated.append(row)
        if accepted_flag:
            accepted.append(row);phase='activity_or_unresolved';last_stop=p['contact_candidate_last_local_frame'];quiet_start=None
            transitions.append(dict(local_frame=first,state=phase,reason='accepted_gesture_proposal_not_confirmed_contact',
                proposed_contact_interval=[p['contact_candidate_first_local_frame'],last_stop]))
    return dict(candidates=evaluated,accepted_candidates=accepted,state_transitions=transitions,
        limitation='Activity lock and quiet reset are candidate phases only; reset does not establish point end, fault, retry number, or winner.')

def main():
    OUT.mkdir(exist_ok=True)
    calpath=SOURCE/'config/match1_court_plane_calibration.json';H=np.array(json.loads(calpath.read_text(encoding='utf-8'))['image_to_court_homography'])
    protocol=dict(parameters=PARAMS,script_sha256=sha(Path(__file__)),
        pose_repair_script_sha256=sha(ROOT/'repair_start_pose_crops.py'),
        gesture_code_sha256=sha(ROOT/'propose_serve_starts.py'),calibration_sha256=sha(calpath),
        development_clips=['016','002'],independent_accuracy_evaluation=False)
    (OUT/'phase_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    summaries=[]
    for cid in ['016','002']:
        folder=OUT/f'match1_motion_pilot_{cid}';d=json.loads((folder/'features.json').read_text());fps=json.loads((folder/'protocol.json').read_text())['fps']
        # Ablation A: repair alone, before any court or phase filtering.
        repaired_proposals,_=propose(tracks(d['frames']),np.load(ROOT/f'serve_start_pilot_tiled_v2/match1_motion_pilot_{cid}/audio_features.npz'),fps)
        selected=court_frames(d['frames'],H);series,tdata=context_series(selected,fps)
        proposals,_=propose(tdata,np.load(ROOT/f'serve_start_pilot_tiled_v2/match1_motion_pilot_{cid}/audio_features.npz'),fps)
        result=phase_filter(proposals,series,fps,H,tdata)
        result.update(labels_read=False,serve_outcome_inferred=False,feature_sha256=sha(folder/'features.json'),
            phase_protocol_sha256=sha(OUT/'phase_protocol.json'),
            repaired_only_candidates=[{k:v for k,v in p.items() if k!='evidence_person_boxes'} for p in repaired_proposals])
        (folder/'phases.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        (folder/'motion_context.json').write_text(json.dumps(dict(frames=series,selection=selected)),encoding='utf-8')
        summary=dict(pilot=cid,repaired_only_candidates=len(repaired_proposals),court_filtered_candidates=len(proposals),
            accepted_candidates=[dict(interval=[p['contact_candidate_first_local_frame'],p['contact_candidate_last_local_frame']],gesture=p['raised_first_local_frame']) for p in result['accepted_candidates']],
            decisions=[dict(interval=[p['contact_candidate_first_local_frame'],p['contact_candidate_last_local_frame']],reason=p['decision'],quiet=p['four_player_quiet'],baseline=p['baseline_distance_roi']) for p in result['candidates']],
            complete_four_samples=sum(r['four_observed'] for r in series),samples=len(series),quiet_samples=sum(r['four_player_quiet'] for r in series))
        summaries.append(summary);print(json.dumps(summary,indent=2),flush=True)
    (OUT/'summary.json').write_text(json.dumps(dict(results=summaries,independent_accuracy=False),indent=2),encoding='utf-8')

if __name__=='__main__':main()
