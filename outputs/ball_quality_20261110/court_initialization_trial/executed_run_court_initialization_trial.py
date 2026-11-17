"""Matched 15-case development comparison, frozen before new policy outputs."""
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import cv2,numpy as np
from confirmed_continuity_guard import confirmed_class,adaptive,old
from court_initialization_guard import court_initialization_class,PARAMETERS,court_point
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110';OUT=BASE/'court_initialization_trial'
SOURCE=Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def write(p,rows):
    if not rows:return
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
    OUT.mkdir();calibration=SOURCE/'config/match1_court_plane_calibration.json';cal=load(calibration);h=np.array(cal['image_to_court_homography'])
    legacy=load(BASE/'automatic_reacquisition_trial_v2/protocol.json')['cases'];plans=[]
    for tag in ['001','002','003','004','005','006','008','009','010','011','012','013','014','015','025']:
        case=BASE/'cohort_tracking'/f'case_{tag}'
        if case.exists():folder=case/f'tracking_quality_20261102_stage4/players_{tag}_temporal';cache=BASE/f'automatic_reacquisition_transfer_{tag}'
        else:folder=Path(next(x['baseline_folder'] for x in legacy if x['tag']==tag));cache=BASE/'automatic_reacquisition_trial_v2'/f'case_{tag}'
        if not (cache/'appearance_cache.npz').exists():cache=BASE/'automatic_reacquisition_trial_v2'/f'case_{tag}'
        s=load(folder/'summary.json');plans.append(dict(tag=tag,baseline_folder=str(folder),raw_csv=s['input_csv'],first=s['first_frame'],last=s['last_frame'],adaptive_size='near_size_adaptation' in s['parameters'],cache=str(cache) if (cache/'appearance_cache.npz').exists() else None,role='known_AI_diagnosed_initial_nonplayer_failure' if tag in ['009','015'] else 'initialization_correspondence_control_not_certified_whole_tracking_accuracy'))
    protected={str(n):sha(n) for n in [calibration,Path(__file__),ROOT/'court_initialization_guard.py',ROOT/'confirmed_continuity_guard.py',ROOT/'automatic_reacquisition_guard.py',Path(adaptive.__file__),Path(old.__file__),Path(old.previous.__file__),Path(old.a.__file__),SOURCE/'scripts/96_select_four_players_pilot.py']}
    for plan in plans:
        for n in [Path(plan['raw_csv'])]+[Path(plan['baseline_folder'])/x for x in ['summary.json','events.json','selected_four.csv']]:protected[str(n)]=sha(n)
        if plan['cache']:
            for n in ['appearance_cache.npz','cached_detections.csv']:protected[str(Path(plan['cache'])/n)]=sha(Path(plan['cache'])/n)
    for n in (BASE/'current19_player_inputs').iterdir():
        if n.is_file():protected[str(n)]=sha(n)
    video=SOURCE/'data/raw/match1.mp4'
    dump(OUT/'protocol.json',dict(date_jst='2026-11-10',purpose='Develop initial-player exclusion after known009/015 failures. All15 case outputs retained; not independent evaluation.',parameters=PARAMETERS,margin_rationale='Court-width15% per side and court-length25% behind each baseline permit outside-line serving/setup. Engineering choice for development, not measured tracking uncertainty; no numeric sweep.',calibration=str(calibration),calibration_source_frame=cal['source_frame'],calibration_transfer_accuracy_established=False,source_video=str(video),source_video_sha256=sha(video),cases=plans,modes=['baseline','court_seed_only','old_guard','court_seed_plus_old_guard'],manual_ID_binding=False,numeric_search=False,policy_uses_future_frames=False,policy_uses_outcomes_or_appearance_ID_labels=False,initialization_confirmation_frames=8,mask_during_initialized_tracking=False,canonical_pipeline_replaced=False,model_input_snapshot_changed=False,protected_input_hashes=protected))
    results=[]
    keycols=['source_frame','player_slot','track_id','x1','y1','x2','y2','identity_source','assignment_cost','alternative_cost_margin']
    k=lambda r:tuple(str(r[x]) for x in keycols)
    for plan in plans:
        tag=plan['tag'];dest=OUT/f'case_{tag}';dest.mkdir();first,last=plan['first'],plan['last'];raw=read(Path(plan['raw_csv']));raw=[r for r in raw if int(r['track_id'])>=0 and float(r['confidence'])>=.25]
        if plan['cache']:
            cache=Path(plan['cache']);cached=read(cache/'cached_detections.csv');z=np.load(cache/'appearance_cache.npz');array=z['features'];offset=z['frame_offsets'];assert len(offset)==last-first+2
            assert [(r['source_frame'],r['track_id'],r['x1'],r['confidence']) for r in raw]==[(r['source_frame'],r['track_id'],r['x1'],r['confidence']) for r in cached];raw=cached
        else:
            byframe=defaultdict(list)
            for r in raw:byframe[int(r['source_frame'])].append(r)
            cached=[];array=[];offset=[0];cap=cv2.VideoCapture(str(video));cap.set(1,first)
            for f in range(first,last+1):
                ok,im=cap.read();assert ok and round(cap.get(1))==f+1;im=cv2.resize(im,(1920,1080));rows=byframe[f]
                for r in rows:
                    feat=old.a.appearance(im,r);cached.append(r);array.append(np.full((2,192),np.nan,np.float32) if feat is None else feat)
                offset.append(len(cached))
            cap.release();array=np.array(array,np.float32);offset=np.array(offset,np.int64);raw=cached;np.savez_compressed(dest/'appearance_cache.npz',features=array,frame_offsets=offset);write(dest/'cached_detections.csv',cached)
        base=adaptive.AdaptiveNearTracker if plan['adaptive_size'] else old.TemporalTracker
        trackers=dict(baseline=base(),court_seed_only=court_initialization_class(base,h)(),old_guard=confirmed_class(base)(),court_seed_plus_old_guard=court_initialization_class(confirmed_class(base),h)());selected={n:[] for n in trackers}
        for index,f in enumerate(range(first,last+1)):
            rows=raw[offset[index]:offset[index+1]];features=[None if np.isnan(x).all() else x for x in array[offset[index]:offset[index+1]]]
            for name,tr in trackers.items():
                for i,j,margin,value,origin in tr.step(f,rows,features):selected[name].append(dict(rows[j],player_slot=old.a.SLOTS[i],position_source='detected_box',identity_status='automatic_initialization_development_candidate_not_GT',identity_source=origin,assignment_cost=round(value,5),alternative_cost_margin=round(margin,5)))
        original=read(Path(plan['baseline_folder'])/'selected_four.csv');assert [k(r) for r in selected['baseline']]==[k(r) for r in original],tag
        reports={}
        for name,tr in trackers.items():
            folder=dest/name;folder.mkdir();write(folder/'selected_four.csv',selected[name]);dump(folder/'events.json',tr.events)
            if hasattr(tr,'seed_filter_log'):dump(folder/'seed_filter_log.json',tr.seed_filter_log)
            baseline_keys={k(r)[:7] for r in selected['baseline']};new_keys={k(r)[:7] for r in selected[name]};near=[r for r in selected[name] if r['player_slot'].startswith('near_')]
            reports[name]=dict(initialized_frame=tr.initialized_frame,seed_ids=tr.events[0]['seed_ids'] if tr.events else None,near_observations=len(near),near_raw_ID_changes=sum(sum(x['track_id']!=y['track_id'] for x,y in zip(rs,rs[1:])) for slot in ['near_front','near_rear'] for rs in [[r for r in near if r['player_slot']==slot]]),total_temporal_observations=len(selected[name]),removed_assignments=len(baseline_keys-new_keys),added_assignments=len(new_keys-baseline_keys))
        result=dict(tag=tag,role=plan['role'],baseline_replay_exact=True,modes=reports,independent_identity_accuracy_established=False);dump(dest/'result.json',result);results.append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
    assert all(sha(Path(n))==v for n,v in protected.items());dump(OUT/'run_result.json',dict(status='completed15_matched_development_replays',cases=results,baseline_exact_cases=len(results),protected_sources_unchanged=True,canonical_pipeline_changed=False,current19_inputs_changed=False,independent_accuracy_established=False))
if __name__=='__main__':main()
