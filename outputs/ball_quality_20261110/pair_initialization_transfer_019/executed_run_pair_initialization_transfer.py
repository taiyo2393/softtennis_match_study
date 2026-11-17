"""Apply previously frozen pair initialization, without tuning on the new case."""
import argparse,sys,json
from pathlib import Path
from collections import defaultdict,Counter
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,sha,dump,write
from confirmed_continuity_guard import confirmed_class,adaptive,old
from court_initialization_guard import court_initialization_class,PARAMETERS
from pair_initialization_guard import pair_initialization_class,PARAMETERS as PAIR_PARAMETERS
from build_rally_analysis_prototype import pair_features,summarize_prefix
from verify_player_prefix_expansion import calculate_frame,calculate_summary,close
sys.path.insert(0,str(ROOT.parent/'tracking_quality_20261102_stage10'))
import gap_pair_tracking as far
SLOTS=['near_front','near_rear','far_rear','far_front']
MODES=['baseline','court_seed_only','pair_rank']
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def key(r):return tuple(str(r[x]) for x in ['source_frame','player_slot','track_id','x1','y1','x2','y2'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--plan',required=True);args=ap.parse_args();tag=args.tag;assert len(tag)==3 and tag.isdigit()
    case=BASE/'cohort_tracking'/f'case_{tag}';out=BASE/f'pair_initialization_transfer_{tag}';planpath=Path(args.plan).resolve();plan=load(planpath)
    assert plan['next_candidate']==f'match1_rally_window_{tag}' and plan[f'fixed_before_{tag}_inspection'];assert plan['court_parameters']==PARAMETERS and plan['pair_parameters']==PAIR_PARAMETERS
    assert not plan[f'threshold_tuning_on_{tag}'] and not plan['new_person_ID_binding_allowed'] and not plan['extra_ambiguity_margin_applied']
    assert all(sha(Path(n))==h for n,h in plan['frozen_code_and_calibration_hashes'].items());assert plan['selected_pair_policy']=='pair_rank' and plan['modes']==MODES
    p=load(case/'protocol.json');assert p['source_video_sha256']==plan['source_video_sha256'];original=case/f'tracking_quality_20261102_stage4/players_{tag}_temporal';summary=load(original/'summary.json');rawpath=Path(summary['input_csv']);assert sha(rawpath)==summary['input_sha256']
    files=[planpath,Path(__file__),Path(far.__file__),case/'protocol.json',case/'four_player_path.csv',case/'case_result.json',original/'summary.json',original/'events.json',original/'selected_four.csv',rawpath]+[case/n for n in ['prefix_frame_features.csv','snapshot_player_features.csv','prefix_player_summary_features.csv']]+[Path(n) for n in plan['frozen_code_and_calibration_hashes']]
    protected={str(x):sha(x) for x in files};out.mkdir();dump(out/'protocol.json',dict(candidate_id=p['candidate_id'],frozen_plan=str(planpath),frozen_plan_sha256=sha(planpath),source_video_sha256=p['source_video_sha256'],scope='Additional same-match transfer after rule freeze, not independent other-match evaluation.',modes=MODES,individual_ID_binding_added=False,numeric_thresholds_changed=False,transfer_results_used_to_tune=False,canonical_pipeline_replaced=False,protected_input_hashes=protected))
    h=np.array(load(Path(plan['calibration']))['image_to_court_homography']);base=adaptive.AdaptiveNearTracker;trackers=dict(baseline=base(),court_seed_only=court_initialization_class(base,h)(),pair_rank=pair_initialization_class(base,h)());selected={n:[] for n in MODES};byframe=defaultdict(list)
    for r in read(rawpath):
        if int(r['track_id'])>=0 and float(r['confidence'])>=.25:byframe[int(r['source_frame'])].append(r)
    cap=cv2.VideoCapture(p['source_video']);first,last=p['context_first_source_frame'],p['context_last_source_frame'];cap.set(1,first);cached=[];array=[];offset=[0]
    for f in range(first,last+1):
        ok,im=cap.read();assert ok and round(cap.get(1))==f+1;im=cv2.resize(im,(1920,1080));rows=byframe[f];features=[old.a.appearance(im,r) for r in rows]
        for r,feat in zip(rows,features):cached.append(r);array.append(np.full((2,192),np.nan,np.float32) if feat is None else feat)
        offset.append(len(cached))
        for name,tr in trackers.items():
            for i,j,margin,value,origin in tr.step(f,rows,features):selected[name].append(dict(rows[j],player_slot=SLOTS[i],position_source='detected_box',identity_status='automatic_frozen_transfer_candidate_not_GT',identity_source=origin,assignment_cost=round(value,5),alternative_cost_margin=round(margin,5)))
    cap.release();np.savez_compressed(out/'appearance_cache.npz',features=np.array(array,np.float32),frame_offsets=np.array(offset,np.int64));write(out/'cached_detections.csv',cached)
    cols=['source_frame','player_slot','track_id','x1','y1','x2','y2','identity_source','assignment_cost','alternative_cost_margin'];signature=lambda rs:[[str(r[k]) for k in cols] for r in rs]
    assert signature(selected['baseline'])==signature(read(original/'selected_four.csv'))
    reports={};baseline_keys={key(r) for r in read(case/'four_player_path.csv')};native=defaultdict(list)
    for r in read(rawpath):native[int(r['source_frame']),r['track_id']].append(r)
    for mode,tr in trackers.items():
        folder=out/mode;folder.mkdir()
        if selected[mode]:write(folder/'selected_four.csv',selected[mode])
        dump(folder/'events.json',tr.events)
        if hasattr(tr,'seed_filter_log'):dump(folder/'seed_filter_log.json',tr.seed_filter_log)
        if hasattr(tr,'pair_seed_log'):dump(folder/'pair_seed_log.json',tr.pair_seed_log)
        common=dict(initialized_frame=tr.initialized_frame,seed_ids=tr.events[0]['seed_ids'] if tr.events else None,near_observations=sum(r['player_slot'].startswith('near_') for r in selected[mode]),individual_ID_binding_added=False)
        if mode=='baseline':reports[mode]=dict(common,baseline_temporal_replay_exact=True,final_path_source=str(case/'four_player_path.csv'));continue
        if tr.initialized_frame is None:
            reports[mode]=dict(common,status='no_initialization_retained_not_skipped',four_observation_requested_times=0);dump(folder/'verification.json',reports[mode]);continue
        mp=dict(p);mp.update(experiment=f'Frozen {mode} transfer',frozen_plan=str(planpath),mode=mode,AI_identity_binding_added=False,individual_ID_binding_added=False,fully_automatic_result=True,processing_intervention=f'frozen_{mode}_initialization_transfer',canonical_pipeline_changed=False,protected_input_hashes={str(out/'protocol.json'):sha(out/'protocol.json'),**protected});dump(folder/'protocol.json',mp)
        temp=folder/f'tracking_quality_20261102_stage4/players_{tag}_temporal';temp.mkdir(parents=True);write(temp/'selected_four.csv',selected[mode]);dump(temp/'events.json',tr.events);dump(temp/'summary.json',dict(summary,initialized_frame=tr.initialized_frame,counts=dict(Counter(r['player_slot'] for r in selected[mode])),source_warning='Frozen transfer candidate; output availability is not identity accuracy.'))
        far.OUT=folder/'tracking_quality_20261102_stage7';far.OUT.mkdir();saved=sys.argv;sys.argv=[str(far.__file__),'--tag',tag]
        try:far.main()
        finally:sys.argv=saved
        near=[r for r in selected[mode] if r['player_slot'].startswith('near_')];farrows=read(far.OUT/f'players_{tag}_global/far_pair_path.csv');combined=sorted(near+farrows,key=lambda r:(int(r['source_frame']),SLOTS.index(r['player_slot'])));write(folder/'four_player_path.csv',combined);people=defaultdict(dict)
        for r in combined:
            f=int(r['source_frame']);assert r['player_slot'] not in people[f];people[f][r['player_slot']]=r;assert any(all(r[k]==x[k] for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y','confidence']) for x in native[f,r['track_id']])
        for ps in people.values():assert len({r['track_id'] for r in ps.values()})==len(ps)
        times=p['requested_input_frames'];cid=p['candidate_id'];features=[dict(rally_candidate_id=cid,relative_source_frame=f-times[0],**pair_features(people[f])) for f in times];snap,summ,_=summarize_prefix(tag,[dict(r,ball_observation_present=0,ball_x_norm='',ball_y_norm='') for r in features])
        for name,rs in [('prefix_frame_features.csv',features),('snapshot_player_features.csv',[snap]),('prefix_player_summary_features.csv',[summ])]:write(folder/name,rs)
        checked=0;expected=[calculate_frame(cid,people,f,f-times[0]) for f in times]
        for actual,wanted in zip(features,expected):
            for k,v in wanted.items():
                if k!='rally_candidate_id':close(actual[k],v);checked+=1
        es,em=calculate_summary(expected)
        for actual,wanted in [(snap,es),(summ,em)]:
            for k,v in wanted.items():close(actual[k],v);checked+=1
        video=far.OUT/f'players_{tag}_global/global_review_h264.mp4';cap=cv2.VideoCapture(str(video));decoded=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape[:2]==(1080,1920);decoded+=1
        cap.release();assert decoded==last-first+1;new_keys={key(r) for r in combined};temporal_equal=signature(selected[mode])==signature(selected['baseline'])
        report=dict(common,status='passed',baseline_temporal_replay_exact=True,temporal_assignments_equal_baseline=temporal_equal,full_assignments_equal_baseline=new_keys==baseline_keys,removed_full_assignments=len(baseline_keys-new_keys),added_full_assignments=len(new_keys-baseline_keys),fixed31_features_equal_baseline=all(read(folder/n)==read(case/n) for n in ['prefix_frame_features.csv','snapshot_player_features.csv','prefix_player_summary_features.csv']),matched_native_boxes=len(combined),recalculated_feature_values=checked,video_frames_decoded=decoded,four_observation_requested_times=sum(len(people[f])==4 for f in times),requested_times=31,initialization_delay_from_baseline=tr.initialized_frame-trackers['baseline'].initialized_frame,numeric_thresholds_changed=False,independent_identity_or_position_accuracy_established=False)
        dump(folder/'verification.json',report);dump(folder/'case_result.json',dict(status='frozen_transfer_path_constructed',candidate_id=cid,input_times_with_four_observations=report['four_observation_requested_times'],artifact_hashes={x.name:sha(x) for x in folder.glob('*.csv')}));reports[mode]=report;print(json.dumps(dict(mode=mode,**report),ensure_ascii=False),flush=True)
    assert all(sha(Path(n))==h for n,h in protected.items());dump(out/'result.json',dict(status='completed_frozen_same_match_transfer',candidate_id=p['candidate_id'],modes=reports,baseline_temporal_replay_exact=True,protected_sources_unchanged=True,independent_identity_or_position_accuracy_established=False,canonical_pipeline_replaced=False))
if __name__=='__main__':main()
