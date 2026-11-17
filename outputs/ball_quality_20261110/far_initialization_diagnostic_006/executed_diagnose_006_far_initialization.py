"""Read-only native-candidate audit; never assigns additional far identities."""
import json
import itertools
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from run_fixed_cohort_player_case_v3 import ROOT, BASE, read, write, sha, dump
from run_near_prefix_initialization_trial import load
from diagnose_initial_pair_appearance import cache
from court_initialization_guard import eligible, court_point, old

OUT=BASE/'far_initialization_diagnostic_006'
PRIOR=BASE/'camera_valid_near_prefix_trial'


def flags(r,H,near):
    x,y=old.a.foot(r)/[3840,2160]; h=old.a.height(r)/2160
    court=eligible(r,H)
    geometry=(.55<=y<.61 and .30<=x<=.63 and h>=.05)
    broad=(.25<x<.75 and .52<y<.665)
    overlap=max([old.overlap_fraction(r,n) for n in near] or [0.])
    failures=[]
    if not court:failures.append('outside_court_envelope')
    if not .55<=y<.61:failures.append('outside_seed_y')
    if not .30<=x<=.63:failures.append('outside_seed_x')
    if h<.05:failures.append('below_seed_height')
    return dict(normalized_foot_x=float(x),normalized_foot_y=float(y),normalized_height=float(h),
        court_eligible=int(court),far_seed_geometry=int(geometry),far_seed_eligible=int(court and geometry),
        broad_far_ROI=int(broad),maximum_near_overlap=float(overlap),
        broad_far_after_near_overlap=int(broad and overlap<=.55),
        seed_rejection_reasons=';'.join(failures),
        court_point_json=json.dumps(court_point(r,H)))


def prepare():
    assert not OUT.exists()
    p=load(PRIOR/'protocol.json');cases=p['cases']
    protected={str(n):sha(n) for n in [Path(__file__),ROOT/'court_initialization_guard.py',
        ROOT/'diagnose_initial_pair_appearance.py',Path(old.selector.__file__),
        Path(old.__file__),Path(p['calibration']),PRIOR/'protocol.json',PRIOR/'publication_hashes.json',
        BASE/'position_reference_evaluation_20261110_r55/human_reference_labels.json']}
    for e in cases:
        for n in [Path(e['raw_csv']),Path(e['cache'])/'cached_detections.csv',Path(e['cache'])/'appearance_cache.npz',
                  PRIOR/f'case_{e["tag"]}/candidate_native_observations.csv',Path(e['temporal_folder'])/'events.json']:
            protected[str(n)]=sha(n)
    for n in (BASE/'current22_player_inputs').iterdir():
        if n.is_file():protected[str(n)]=sha(n)
    OUT.mkdir()
    dump(OUT/'protocol.json',dict(cases=cases,calibration=p['calibration'],source_video=p['source_video'],
        source_video_sha256=p['source_video_sha256'],camera_intervals=p['camera_intervals'],
        primary='006 dense original context start through fixed input end, including pre/post common initialization.',
        controls='All ten existing contexts at all31 frozen input times, unchanged.',
        inference='No detector/pose inference, no weights or thresholds changed, no new identities assigned.',
        gates='Report existing seed .55<=foot_y<.61, .30<=foot_x<=.63,height>=.05 and court envelope, separately from existing downstream far ROI .25<x<.75,.52<y<.665 and near overlap>.55.',
        candidate_counts_are_not_true_player_recall=True,
        review_frames_fixed=[10536,10546,10574,10575,10596],
        additional_review_frames='Earliest eligible main-camera frame in006 context and earliest independent8-frame stable far-seed-pair confirmation; diagnostic algorithmic selection, not accuracy sampling.',
        previous_manual_reference='FrozenP011 includes one unplaceable far reference. Unplaceable is not automatically absent, an occlusion label or detector failure.',
        analysis_inputs_changed=False,independent_GT_added=False,policy_fixed_before_new_audit_outputs=True,
        protected_input_hashes=protected))
    print('Frozen diagnostic: ten fixed windows and dense006 timeline.',flush=True)


def run():
    p=load(OUT/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    H=np.array(load(Path(p['calibration']))['image_to_court_homography'])
    all_rows=[];timeline=[];reports=[];first_independent=None
    for e in p['cases']:
        tag=e['tag']; raw,_,offset=cache(e)
        table=defaultdict(dict)
        for r in read(PRIOR/f'case_{tag}/candidate_native_observations.csv'):
            table[int(r['source_frame'])][r['player_slot']]=r
        fixed=set(e['requested_input_frames']); wanted=sorted(fixed|set(range(e['first'],max(fixed)+1))) if tag=='006' else sorted(fixed)
        window=[];states=[]
        for f in wanted:
            i=f-e['first'];rows=raw[offset[i]:offset[i+1]]
            near=[r for s,r in table[f].items() if s.startswith('near_')]
            classified=[dict(r,tag=tag,**flags(r,H,near)) for r in rows]
            all_rows.extend(classified)
            seeds=sorted([r for r in classified if r['far_seed_eligible']],key=lambda r:float(r['confidence']),reverse=True)[:2]
            valid=(len(seeds)==2 and old.overlap_fraction(*seeds)<=old.PARAMETERS['occlusion_intersection_fraction'])
            ids=tuple(sorted(int(r['track_id']) for r in seeds)) if valid else None
            camera=any(a<=f<=b for a,b in p['camera_intervals'])
            if not camera or not ids or (window and (window[-1][0]!=f-1 or window[-1][1]!=ids)):window=[]
            if camera and ids:window.append((f,ids))
            if tag=='006' and first_independent is None and len(window)>=8:
                first_independent=dict(source_frame=f,window_start=window[0][0],raw_ids=list(ids),not_an_identity_ground_truth=True)
            record=dict(tag=tag,source_frame=f,fixed_requested_time=int(f in fixed),main_camera_interval=int(camera),
                cached_native_candidates=len(classified),
                far_seed_candidates=sum(r['far_seed_eligible'] for r in classified),
                far_seed_top2_raw_ids=';'.join(str(r['track_id']) for r in seeds),
                far_seed_pair_overlap=float(old.overlap_fraction(*seeds)) if len(seeds)==2 else '',
                far_seed_pair_valid=int(valid),independent_pair_consecutive_source_frames=len(window),
                broad_far_candidates=sum(r['broad_far_ROI'] for r in classified),
                broad_far_after_near_overlap=sum(r['broad_far_after_near_overlap'] for r in classified),
                near_output_count=len(near),far_output_count=sum(s.startswith('far_') for s in table[f]))
            timeline.append(record)
        sample=[r for r in timeline if r['tag']==tag and r['fixed_requested_time']]
        report=dict(tag=tag,requested_times=len(sample),
            seed_candidate_count_histogram=dict(Counter(r['far_seed_candidates'] for r in sample)),
            broad_after_near_overlap_count_histogram=dict(Counter(r['broad_far_after_near_overlap'] for r in sample)),
            actual_far_output_count_histogram=dict(Counter(r['far_output_count'] for r in sample)))
        reports.append(report);print(json.dumps(report),flush=True)
    write(OUT/'candidate_gate_audit.csv',all_rows);write(OUT/'frame_gate_audit.csv',timeline)
    e=next(e for e in p['cases'] if e['tag']=='006')
    earliest=min(f for f in range(e['first'],max(e['requested_input_frames'])+1) if any(a<=f<=b for a,b in p['camera_intervals']))
    review=sorted(set(p['review_frames_fixed']+[earliest]+([first_independent['source_frame']] if first_independent else [])))
    labels=load(BASE/'position_reference_evaluation_20261110_r55/human_reference_labels.json')['labels']['P011']
    dump(OUT/'result.json',dict(cases=reports,first_independent_far_seed_confirmation=first_independent,
        review_frames=review,P011_far_reference_states={k:v['state'] for k,v in labels.items() if k.startswith('far_')},
        selected_native_or_ID_outputs_added=0,analysis_inputs_changed=False,independent_accuracy_established=False))
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())


if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()
