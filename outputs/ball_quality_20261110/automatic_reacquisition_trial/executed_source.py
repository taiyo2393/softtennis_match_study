"""Matched replay and two frozen guard variants on five existing development cases."""
import csv,hashlib,json,sys
from pathlib import Path
from collections import defaultdict,Counter
import cv2,numpy as np
from automatic_reacquisition_guard import adaptive,old,guarded_class,MODES
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110';OUT=BASE/'automatic_reacquisition_trial'
a=old.a
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def write(p,rows):
    if not rows:return
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def sources():
    cases=[]
    for tag in ['002','003','005']:
        cases.append((tag,ROOT.parent/'tracking_quality_20261102_stage9'/f'players_{tag}_temporal'))
    cases.append(('025',ROOT.parent/'tracking_quality_20261102_stage10/experiment/tracking_quality_20261102_stage4/players_025_temporal'))
    cases.append(('011',BASE/'cohort_tracking/case_011/tracking_quality_20261102_stage4/players_011_temporal'))
    return cases
def run():
    OUT.mkdir();cases=sources();plans=[];protected={}
    for tag,folder in cases:
        summary=load(folder/'summary.json');path=Path(summary['input_csv']);assert sha(path)==summary['input_sha256']
        plans.append(dict(tag=tag,baseline_folder=str(folder.resolve()),raw_csv=str(path),first=summary['first_frame'],last=summary['last_frame'],adaptive_size='near_size_adaptation' in summary['parameters']))
        for p in [folder/'summary.json',folder/'selected_four.csv',folder/'events.json',path]:protected[str(p.resolve())]=sha(p)
    code=[Path(__file__),ROOT/'automatic_reacquisition_guard.py',Path(adaptive.__file__),Path(old.__file__),Path(old.previous.__file__),Path(a.__file__)]
    for p in code:protected[str(p.resolve())]=sha(p)
    protocol=dict(date_jst='2026-11-10',scope='Same-match development; known011 failure and existing ID-switch controls. Not independent accuracy, human GT or predictive performance.',selection='011 is known failure. Controls002,003,025 contain existing near-ID changes;005 is stable-ID control. All chosen before variant results. No unseen012 inspected.',variants=['baseline']+MODES,changes={'confirm_new_id':'Near new raw IDs cannot enter ordinary continuity; retain existing6-frame minimum gap and8-frame recovery confirmation. Same-raw continuity unchanged.','confirm_new_id_opponent_negatives':'Same confirmation restriction plus automatically selected initial far-player galleries as near negatives; reuse existing appearance thresholds/margins.'},threshold_search=False,manual_ID_binding=False,initialization_changed=False,detector_reused=True,frame_interpolation=False,primary_failure_probe_frames=[27514,27526,27652,27664,27854,27892,27904,27997],failure_probe_provenance='Existing AI image review of wrong N2 binding, not independent human GT; full-rally unseen errors not excluded.',control_switches=[dict(tag='002',slot='near_rear',frame=2452,new_id=78),dict(tag='003',slot='near_front',frame=4572,new_id=202),dict(tag='025',slot='near_front',frame=60372,new_id=107),dict(tag='025',slot='near_rear',frame=60302,new_id=84)],control_switches_status='All existing near raw-ID changes, including ones requiring AI correspondence review; do not presume all are correct.',maximum_allowed_added_confirmation_delay_frames=13,delay_basis='Existing min_recovery_gap6 plus recovery_source_frames8 minus1; development acceptance budget, not a physical rule.',decision='Do not replace retained pipeline if known wrong binding remains, a reviewed valid switch is not reacquired within13 source frames, stable control changes unexpectedly, baseline fails exact replay, or a new obvious wrong identity is seen. All models remain development candidates even if these checks pass.',cases=plans,source_video=str(a.SOURCE/'data/raw/match1.mp4'),protected_input_hashes=protected)
    dump(OUT/'protocol.json',protocol)
    for plan in plans:
        tag=plan['tag'];dest=OUT/f'case_{tag}';dest.mkdir();raw=read(Path(plan['raw_csv']));byframe=defaultdict(list)
        for r in raw:
            if int(r['track_id'])>=0 and float(r['confidence'])>=.25:byframe[int(r['source_frame'])].append(r)
        records=[];features=[];offsets=[0];cap=cv2.VideoCapture(protocol['source_video']);cap.set(1,plan['first'])
        for frame in range(plan['first'],plan['last']+1):
            ok,im=cap.read();assert ok and round(cap.get(1))==frame+1
            im=cv2.resize(im,(1920,1080))
            for r in byframe[frame]:
                feat=a.appearance(im,r);records.append(r);features.append(np.full((2,192),np.nan,np.float32) if feat is None else feat)
            offsets.append(len(records))
        cap.release();array=np.array(features,np.float32);offsets=np.array(offsets,np.int64)
        np.savez_compressed(dest/'appearance_cache.npz',features=array,frame_offsets=offsets)
        write(dest/'cached_detections.csv',records)
        dump(dest/'cache_manifest.json',dict(raw_source_sha256=sha(Path(plan['raw_csv'])),feature_shape=list(array.shape),decoded_frames=plan['last']-plan['first']+1,cache_sha256=sha(dest/'appearance_cache.npz'),detections_sha256=sha(dest/'cached_detections.csv'),resize='cv2.resize1920x1080 default INTER_LINEAR, identical retained appearance pipeline.'))
        base=adaptive.AdaptiveNearTracker if plan['adaptive_size'] else old.TemporalTracker
        trackers={'baseline':base(),**{mode:guarded_class(base,mode)() for mode in MODES}}
        selected={name:[] for name in trackers};traces={name:[] for name in trackers};opponents={}
        for index,frame in enumerate(range(plan['first'],plan['last']+1)):
            rows=records[offsets[index]:offsets[index+1]];fs=[None if np.isnan(f).all() else f for f in array[offsets[index]:offsets[index+1]]]
            for mode,tracker in trackers.items():
                prior={s:dict(st,gallery=list(st['gallery']),velocity=st['velocity'].copy(),negatives=list(st['negatives'])) for s,st in tracker.states.items()}
                if plan['adaptive_size']:
                    for s in prior:
                        if s.startswith('near_'):prior[s]['reference_height']=adaptive.reference(tracker.heights[s])
                assignments=tracker.step(frame,rows,fs)
                if mode not in opponents and tracker.states:opponents[mode]=[f.copy() for s,st in tracker.states.items() if s.startswith('far_') for f in st['gallery']]
                for i,j,margin,value,origin in assignments:
                    r=rows[j];slot=a.SLOTS[i]
                    selected[mode].append(dict(r,player_slot=slot,position_source='detected_box',identity_status='development_candidate_not_GT',identity_source=origin,assignment_cost=round(value,5),alternative_cost_margin=round(margin,5)))
                    if slot in prior and slot.startswith('near_') and (prior[slot]['row']['track_id']!=r['track_id'] or origin=='gallery_reacquisition_candidate'):
                        st=prior[slot];feat=fs[j];own,_=old.best_appearance(st,feat)
                        traces[mode].append(dict(source_frame=frame,player_slot=slot,old_id=st['row']['track_id'],new_id=r['track_id'],old_observed_frame=st['frame'],gap=frame-st['frame'],identity_source=origin,cost=float(value),margin=float(margin),own_gallery_distance=own,closest_initial_opponent_distance=min(a.app_distance(f,feat) for f in opponents[mode]),nearest_distractor_distance=min([a.app_distance(f,feat) for f in st['negatives']],default=None)))
            if index%600==0:print(tag,'replay',index,'/',plan['last']-plan['first']+1,flush=True)
        original=read(Path(plan['baseline_folder'])/'selected_four.csv');keys=['source_frame','player_slot','track_id','x1','y1','x2','y2','identity_source','assignment_cost','alternative_cost_margin']
        assert [[r[k] for k in keys] for r in selected['baseline']]==[[r[k] for k in keys] for r in original],f'{tag}: baseline replay mismatch'
        reports={}
        for mode,tracker in trackers.items():
            folder=dest/mode;folder.mkdir();write(folder/'selected_four.csv',selected[mode]);write(folder/'near_transition_trace.csv',traces[mode]);dump(folder/'events.json',tracker.events)
            near=[r for r in selected[mode] if r['player_slot'].startswith('near_')]
            oldnear=[r for r in selected['baseline'] if r['player_slot'].startswith('near_')]
            key=lambda r:tuple(r[k] for k in ['source_frame','player_slot','track_id','x1','y1','x2','y2'])
            oldset={key(r) for r in oldnear};newset={key(r) for r in near}
            reports[mode]=dict(near_observations=len(near),near_counts=dict(Counter(r['player_slot'] for r in near)),changed_old_observations=len(oldset-newset),added_observations=len(newset-oldset),initialized_frame=tracker.initialized_frame,source_coordinate_interpolation=False,near_ID_transitions=len(traces[mode]))
        dump(dest/'result.json',dict(baseline_replay_exact=True,variants=reports,protected_sources_unchanged=all(sha(Path(n))==h for n,h in protected.items()),independent_identity_accuracy=False));print(tag,json.dumps(reports),flush=True)
    assert all(sha(Path(n))==h for n,h in protected.items())
    dump(OUT/'run_result.json',dict(status='completed_all_five_matched_replays',baseline_exact_cases=5,AI_control_image_review_pending=True,canonical_pipeline_changed=False,current15_inputs_changed=False,independent_accuracy_established=False))
if __name__=='__main__':run()
