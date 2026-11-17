"""Generate overhead-action/audio start proposals without reading event labels."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from extract_serve_start_features import OUT,sha


def describe(person,t):
    box=np.array(person['box']);kp=np.array(person['keypoints']);h=box[3]-box[1]
    valid_shoulders=kp[[5,6]][kp[[5,6],2]>=.35]
    if not len(valid_shoulders):return None
    shoulder=float(np.mean(valid_shoulders[:,1]))
    head=float(kp[0,1]) if kp[0,2]>=.35 else shoulder-.15*h
    wrists=kp[[9,10]][kp[[9,10],2]>=.35]
    raised=bool(len(wrists) and np.min(wrists[:,1])<head-.02*h)
    return dict(local_frame=t,box=box.tolist(),height=float(h),foot=[float((box[0]+box[2])/2),float(box[3])],
                raised=raised,head_y=head,shoulder_y=shoulder,wrists=wrists.tolist(),person_confidence=person['confidence'])


def tracks(samples):
    states={};next_id=0
    for sample in samples:
        t=sample['local_frame'];points=[d for p in sample['people'] if (d:=describe(p,t)) is not None]
        links=[]
        for i,p in enumerate(points):
            for tid,entries in states.items():
                q=entries[-1]
                if t-q['local_frame']>12:continue
                d=float(np.linalg.norm(np.array(p['foot'])-q['foot'])/max(p['height'],q['height']))
                if d<.5:links.append((d,i,tid))
        used=set();assigned=set()
        for d,i,tid in sorted(links):
            if i in used or tid in assigned:continue
            states[tid].append(points[i]);used.add(i);assigned.add(tid)
        for i,p in enumerate(points):
            if i not in used:states[next_id]=[p];next_id+=1
    return states


def propose(tdata,audio,fps):
    proposals=[];episodes=[]
    for tid,entries in tdata.items():
        raised=[r for r in entries if r['raised']]
        groups=[]
        for row in raised:
            if not groups or row['local_frame']-groups[-1][-1]['local_frame']>round(.25*fps):groups.append([])
            groups[-1].append(row)
        for group in groups:
            if len(group)<3:continue
            first=group[0]['local_frame'];last=group[-1]['local_frame']
            if not 4<=last-first<=120:continue
            prior=[r for r in entries if first-round(.6*fps)<=r['local_frame']<first]
            if len(prior)<4:continue
            feet=np.array([r['foot'] for r in prior]);h=float(np.median([r['height'] for r in prior]))
            displacement=float(np.max(np.linalg.norm(feet-np.median(feet,axis=0),axis=1))/h)
            if displacement>.45:continue
            begin=first/fps;end=last/fps+.25
            indices=np.flatnonzero((audio['time_sec']>=begin)&(audio['time_sec']<=end))
            if not len(indices):continue
            peak=int(indices[np.argmax(audio['rise_db'][indices])]);when=float(audio['time_sec'][peak]);rise=float(audio['rise_db'][peak])
            # The audio peak is a timing cue, not a verified racket-ball contact.
            proposal=dict(person_track_id=int(tid),raised_first_local_frame=first,raised_last_local_frame=last,
                raised_pose_samples=len(group),prior_body_motion_by_height=displacement,
                audio_peak_local_sec=when,audio_rise_db=rise,
                contact_candidate_first_local_frame=max(0,math.floor((when-.25)*fps)),
                contact_candidate_last_local_frame=math.ceil((when+.25)*fps),
                status='motion_audio_start_proposal_not_confirmed_contact',serve_outcome='unknown',
                evidence_person_boxes=[r['box'] for r in group],
                score=float(max(0.,rise)*min(1.,len(group)/6)))
            episodes.append(proposal)
    # Avoid counting two nearby person's overlapping estimates as separate starts.
    for p in sorted(episodes,key=lambda r:r['score'],reverse=True):
        if any(abs(p['audio_peak_local_sec']-q['audio_peak_local_sec'])<.5 for q in proposals):continue
        proposals.append(p)
    return sorted(proposals,key=lambda r:r['audio_peak_local_sec']),episodes


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',required=True,choices=['016','002']);parser.add_argument('--tiled',action='store_true');a=parser.parse_args()
    feature_root=OUT.parent/'serve_start_pilot_tiled_v2' if a.tiled else OUT
    dest=feature_root/f'match1_motion_pilot_{a.pilot}'
    assert not (dest/'proposals.json').exists()
    protocol=json.loads((dest/'protocol.json').read_text());data=json.loads((dest/'features.json').read_text())
    params=dict(raised='wrist confidence>=0.35 and wrist above nose by0.02*height; fallback head=shoulder-0.15height',
        raised_samples_min=3,raised_span_frames=[4,120],prior_sec=.6,prior_samples_min=4,max_body_motion_by_height=.45,
        gesture_gap_sec=.25,person_link_max_gap_frames=12,person_link_max_displacement_by_height=.5,
        audio_search='raised first frame through last+0.25s; largest positive energy rise',
        audio_timing_uncertainty_sec=.25,global_event_suppression_sec=.5,
        labels_read=False,limitation='Overhead actions only; no underhand-serve model or fault/validity classification.')
    (dest/'proposal_protocol.json').write_text(json.dumps(params,indent=2),encoding='utf-8')
    people=tracks(data['frames']);audio=np.load(dest/'audio_features.npz')
    candidates,episodes=propose(people,audio,protocol['fps'])
    result=dict(pilot=a.pilot,person_tracklets=len(people),proposals=candidates,raw_gesture_episodes=episodes,
        protocol_sha256=sha(dest/'proposal_protocol.json'),feature_sha256=sha(dest/'features.json'),
        code_sha256=sha(Path(__file__)),labels_read=False)
    (dest/'proposals.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(pilot=a.pilot,person_tracklets=len(people),proposals=[{k:v for k,v in r.items() if k!='evidence_person_boxes'} for r in candidates]),indent=2))


if __name__=='__main__':main()
