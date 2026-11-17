"""Propose missing observed boxes using unoccluded torso appearance.

The existing user-reviewed path is immutable. Linear reference coordinates
only gate candidate search, never become output positions.
"""
import json,csv,math
from collections import defaultdict,Counter
from pathlib import Path
import cv2
import numpy as np
from diagnose_044_person_gaps import setup,GAPS
from track_044_far_pair import ROOT,OUT as BASE,FIRST,NEAR,a,g,image_at
from extract_serve_start_features import sha

OUT=ROOT/'person_044_occlusion_recovery'
P=dict(min_box_height_ratio=.60,max_box_height_ratio=1.65,
    min_torso_visible_fraction=.35,min_torso_pixels=20,min_part_pixels=12,
    visible_appearance_max=.65,negative_margin=.04,
    reference_distance_max_height=.75,selection_margin=.08,
    candidate_ids='Only raw IDs of the same reviewed player immediately before or after this missing run',
    reference='linear endpoint reference for candidate gating only; not exported',
    preserve_reviewed_rows=True,interpolation=False,automatic_adoption=False)

def visible_feature(im,row,occluders):
    x1,y1,x2,y2=[float(row[k])*.5 for k in ['x1','y1','x2','y2']];w=x2-x1;h=y2-y1
    parts=[];fractions=[];counts=[]
    for lo,hi in [(.18,.52),(.52,.78)]:
        l=max(0,round(x1+.15*w));r=min(im.shape[1],round(x2-.15*w))
        t=max(0,round(y1+lo*h));b=min(im.shape[0],round(y1+hi*h))
        patch=im[t:b,l:r];mask=np.ones(patch.shape[:2],np.uint8)*255
        if not patch.size:return None,[],[]
        for n in occluders:
            nl,nt,nr,nb=[float(n[k])*.5 for k in ['x1','y1','x2','y2']]
            il=max(l,math.floor(nl));ir=min(r,math.ceil(nr));it=max(t,math.floor(nt));ib=min(b,math.ceil(nb))
            if il<ir and it<ib:mask[it-t:ib-t,il-l:ir-l]=0
        count=int(np.count_nonzero(mask));fraction=count/max(1,mask.size)
        counts.append(count);fractions.append(fraction)
        if count<P['min_part_pixels']:parts.append(None);continue
        hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
        hist=cv2.calcHist([hsv],[0,1,2],mask,[12,4,4],[0,180,0,256,0,256]).ravel()
        parts.append(hist/max(float(hist.sum()),1.))
    return parts,fractions,counts

def distance(anchor,partial):
    ds=[cv2.compareHist(x,y,cv2.HISTCMP_BHATTACHARYYA) for x,y in zip(anchor,partial) if y is not None]
    return float(np.mean(ds)) if ds else 1.

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'proposed_paths.csv').exists()
    source_protocol,raw,by,anchors,negatives=setup()
    registry=json.loads((ROOT/'reviewed_person_overrides.json').read_text())['rallies']['match1_rally_window_044']
    reviewed_path=Path(registry['reviewed_output']);reviewed=a.read(reviewed_path)
    assert sha(reviewed_path)==registry['reviewed_output_sha256']
    lookup={(int(r['local_frame']),r['player_slot']):r for r in reviewed}
    index,counts=g.tracklet_index(raw);owners={}
    for r in reviewed:
        if r['player_slot'].startswith('far_'):owners[r['tracklet_key']]=r['player_slot']
    additions=[];decisions=[];reports=[]
    for slot,start,end in GAPS:
        before=lookup[(start-1,slot)];after=lookup[(end+1,slot)]
        allowed={int(before['track_id']),int(after['track_id'])}
        scale=max(a.height(before),a.height(after));gap_rows=[]
        for t in range(start,end+1):
            f=FIRST+t;im=image_at(f);near=[r for r in by[f] if int(r['track_id']) in NEAR.values()]
            weight=(t-start+1)/(end-start+2);reference=(1-weight)*a.foot(before)+weight*a.foot(after)
            options=[];examined=[]
            for r in by[f]:
                if int(r['track_id']) not in allowed:continue
                reasons=[];height_ratio=a.height(r)/scale
                residual=float(np.linalg.norm(a.foot(r)-reference)/scale)
                key=index[(f,int(r['track_id']))]
                if owners.get(key,slot)!=slot:reasons.append('tracklet_owned_by_other_identity')
                if not P['min_box_height_ratio']<=height_ratio<=P['max_box_height_ratio']:reasons.append('box_size_not_supported')
                if residual>P['reference_distance_max_height']:reasons.append('outside_endpoint_motion_gate')
                x,y=a.foot(r)/[3840,2160]
                if not (.20<x<.90 and .49<y<.665):reasons.append('outside_far_roi')
                feature,fractions,pixels=visible_feature(im,r,near)
                appearance=1.;neg=1.
                if feature is None or fractions[0]<P['min_torso_visible_fraction'] or pixels[0]<P['min_torso_pixels']:
                    reasons.append('insufficient_unoccluded_torso')
                else:
                    appearance=min(distance(z,feature) for z in anchors[slot])
                    neg=min([distance(z,feature) for z in negatives] or [1.])
                    if appearance>P['visible_appearance_max']:reasons.append('visible_appearance_gate')
                    if neg+P['negative_margin']<appearance:reasons.append('visible_negative_template_veto')
                cost=.65*appearance+.35*min(residual,1.)
                diagnostic=dict(local_frame=t,slot=slot,raw_id=r['track_id'],row=r,tracklet_key=key,
                    torso_visible_fraction=fractions[0] if fractions else 0,
                    part_visible_pixels=pixels,appearance=appearance,negative_distance=neg,
                    reference_residual_height=residual,height_ratio=height_ratio,rejection_reasons=reasons,cost=cost)
                examined.append(diagnostic)
                if not reasons:options.append((cost,diagnostic))
            options.sort(key=lambda q:q[0])
            selected=None
            if options and (len(options)==1 or options[1][0]-options[0][0]>=P['selection_margin']):selected=options[0][1]
            if selected:
                r=selected['row']
                q={k:'' for k in reviewed[0]};q.update(r)
                q.update(local_frame=str(t),player_slot=slot,position_source='detected_box_visible_torso_recovery_candidate',
                    identity_status='new_recovery_candidate_requires_review',tracklet_key=selected['tracklet_key'],
                    original_identity_status='new_gap_recovery_not_previously_reviewed',human_video_review_id='',
                    human_video_identity_review='not_reviewed_new_recovery',independent_frame_identity_groundtruth='0')
                additions.append(q);gap_rows.append(t)
            decisions.append(dict(slot=slot,local_frame=t,allowed_raw_ids=sorted(allowed),candidates=examined,
                decision='proposed_observed_box' if selected else 'ambiguous_leave_missing' if len(options)>1 else 'no_supported_observed_box_leave_missing'))
        reports.append(dict(slot=slot,old_missing_local_interval=[start,end],proposed_added_local_frames=gap_rows,
            remaining_missing_local_frames=[t for t in range(start,end+1) if t not in gap_rows]))
    output=reviewed+additions;output.sort(key=lambda r:(int(r['source_frame']),r['player_slot']))
    with (OUT/'proposed_paths.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(reviewed[0]));w.writeheader();w.writerows(output)
    saved=a.read(OUT/'proposed_paths.csv');outlookup={(int(r['source_frame']),r['player_slot']):r for r in saved}
    assert len(outlookup)==len(saved)
    for r in reviewed:assert outlookup[(int(r['source_frame']),r['player_slot'])]==r
    rawlookup={(int(r['source_frame']),r['track_id']):r for r in raw}
    for r in additions:
        assert (int(r['local_frame']),r['player_slot']) not in lookup
        rr=rawlookup[(int(r['source_frame']),r['track_id'])]
        assert all(r[k]==rr[k] for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y','confidence'])
        assert not r['human_video_review_id']
    for frame in set(int(r['source_frame']) for r in output):
        rows=[r for r in output if int(r['source_frame'])==frame]
        assert len({r['track_id'] for r in rows})==len(rows)
    inplay=[r for r in output if 115374<=int(r['source_frame'])<=115775]
    count=Counter(int(r['source_frame']) for r in inplay)
    protocol=dict(parameters=P,reviewed_input_sha256=sha(reviewed_path),reviewed_input=str(reviewed_path),
        original_raw_sha256=source_protocol['raw_source_sha256'],script_sha256=sha(Path(__file__)),
        diagnosis_sha256=sha(ROOT/'person_044_gap_audit/diagnostics.json'),
        scope='One-clip development; preserve reviewed rows. Added rows have no prior human confirmation.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    report=dict(gaps=reports,added_observed_boxes=len(additions),new_frames_with_four_candidates=sum(n==4 for n in count.values()),
        previous_frames_with_four_candidates=315,decisions=decisions,
        human_visibility_or_identity_of_additions='pending',automatic_adoption=False,
        limitation='Box visibility masks and appearance similarity are heuristics, not independent occlusion labels or position accuracy.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(dict(all_reviewed_rows_preserved_exactly=True,
        all_added_coordinates_from_same_frame_raw_detections=True,no_detection_assigned_to_two_identities=True,
        no_interpolation=True,no_human_confirmation_propagated_to_new_rows=True,
        reviewed_input_unchanged=sha(reviewed_path)==registry['reviewed_output_sha256'],accuracy_established=False),indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='decisions'},indent=2))

if __name__=='__main__':main()
