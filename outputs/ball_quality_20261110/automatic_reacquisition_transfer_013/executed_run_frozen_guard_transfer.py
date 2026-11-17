"""Apply a previously frozen near-ID guard to the next chronological case without tuning."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict,Counter
import cv2,numpy as np
from confirmed_continuity_guard import confirmed_class,adaptive,old
from run_fixed_cohort_player_case_v3 import read,sha,dump,write
from build_rally_analysis_prototype import pair_features,summarize_prefix
from verify_player_prefix_expansion import calculate_frame,calculate_summary,close
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110'
sys.path.insert(0,str(ROOT.parent/'tracking_quality_20261102_stage10'))
import gap_pair_tracking as far
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rowkey(r):return tuple(str(r[k]) for k in ['source_frame','player_slot','track_id','x1','y1','x2','y2'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--plan',required=True);args=ap.parse_args();tag=args.tag;assert len(tag)==3 and tag.isdigit()
    CASE=BASE/'cohort_tracking'/f'case_{tag}';OUT=BASE/f'automatic_reacquisition_transfer_{tag}'
    frozen_path=Path(args.plan).resolve();frozen=load(frozen_path);assert sha(Path(frozen['source_code']))==frozen['source_code_sha256']
    assert frozen['next_candidate']==f'match1_rally_window_{tag}' and frozen[f'fixed_before_{tag}_inspection']
    assert frozen['manual_ID_binding'] is False
    OUT.mkdir();p=load(CASE/'protocol.json');original=CASE/f'tracking_quality_20261102_stage4/players_{tag}_temporal';summary=load(original/'summary.json');rawpath=Path(summary['input_csv']);assert sha(rawpath)==summary['input_sha256']
    files=[frozen_path,Path(frozen['source_code']),ROOT/'automatic_reacquisition_guard.py',ROOT/'run_fixed_cohort_player_case_v3.py',Path(adaptive.__file__),Path(old.__file__),Path(old.previous.__file__),Path(old.a.__file__),Path(far.__file__),Path(__file__),CASE/'protocol.json',CASE/'four_player_path.csv',original/'selected_four.csv',original/'summary.json',original/'events.json',rawpath]+[CASE/n for n in ['prefix_frame_features.csv','snapshot_player_features.csv','prefix_player_summary_features.csv']]
    protected={str(n.resolve()):sha(n) for n in files}
    q=dict(p);q.update(experiment=f'Previously frozen automatic guard transfer to{tag}',frozen_plan=str(frozen_path),frozen_guard_code_sha256=frozen['source_code_sha256'],near_assignment_policy_changed=True,numeric_thresholds_changed=False,manual_ID_binding=False,transfer_results_used_to_tune=False,canonical_pipeline_changed=False,scope='Additional same-match transfer, not independent other-match evaluation.',protected_input_hashes=protected);dump(OUT/'protocol.json',q)
    byframe=defaultdict(list)
    for r in read(rawpath):
        if int(r['track_id'])>=0 and float(r['confidence'])>=.25:byframe[int(r['source_frame'])].append(r)
    trackers={'baseline_replay':adaptive.AdaptiveNearTracker(),'frozen_guard':confirmed_class(adaptive.AdaptiveNearTracker)()};selected={n:[] for n in trackers};cache=[];offset=[0];cached_rows=[]
    cap=cv2.VideoCapture(p['source_video']);cap.set(1,p['context_first_source_frame'])
    for f in range(p['context_first_source_frame'],p['context_last_source_frame']+1):
        ok,im=cap.read();assert ok and round(cap.get(1))==f+1;im=cv2.resize(im,(1920,1080));rows=byframe[f];features=[old.a.appearance(im,r) for r in rows]
        for r,feature in zip(rows,features):cached_rows.append(r);cache.append(np.full((2,192),np.nan,np.float32) if feature is None else feature)
        offset.append(len(cached_rows))
        for name,tracker in trackers.items():
            for i,j,margin,value,origin in tracker.step(f,rows,features):selected[name].append(dict(rows[j],player_slot=old.a.SLOTS[i],position_source='detected_box',identity_status='automatic_transfer_candidate_not_GT',identity_source=origin,assignment_cost=round(value,5),alternative_cost_margin=round(margin,5)))
    cap.release();np.savez_compressed(OUT/'appearance_cache.npz',features=np.array(cache,np.float32),frame_offsets=np.array(offset,np.int64));write(OUT/'cached_detections.csv',cached_rows)
    cols=['source_frame','player_slot','track_id','x1','y1','x2','y2','identity_source','assignment_cost','alternative_cost_margin'];assert [[str(r[k]) for k in cols] for r in selected['baseline_replay']]==[[str(r[k]) for k in cols] for r in read(original/'selected_four.csv')]
    write(OUT/'baseline_temporal_replay.csv',selected['baseline_replay']);write(OUT/'guard_temporal_selected.csv',selected['frozen_guard'])
    temp=OUT/f'tracking_quality_20261102_stage4/players_{tag}_temporal';temp.mkdir(parents=True);write(temp/'selected_four.csv',selected['frozen_guard']);dump(temp/'summary.json',dict(summary,input_csv=str(rawpath),parameters=dict(summary['parameters'],experimental_guard_code_sha256=frozen['source_code_sha256'])));dump(temp/'events.json',trackers['frozen_guard'].events)
    far.OUT=OUT/'tracking_quality_20261102_stage7';far.OUT.mkdir();args=sys.argv;sys.argv=[str(far.__file__),'--tag',tag]
    try:far.main()
    finally:sys.argv=args
    near=[r for r in selected['frozen_guard'] if r['player_slot'].startswith('near_')];farrows=read(far.OUT/f'players_{tag}_global/far_pair_path.csv');slots=['near_front','near_rear','far_rear','far_front'];combined=sorted(near+farrows,key=lambda r:(int(r['source_frame']),slots.index(r['player_slot'])));write(OUT/'four_player_path.csv',combined)
    people=defaultdict(dict);native=defaultdict(list)
    for r in read(rawpath):native[int(r['source_frame']),r['track_id']].append(r)
    for r in combined:
        f=int(r['source_frame']);assert r['player_slot'] not in people[f];people[f][r['player_slot']]=r;assert any(all(r[k]==n[k] for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y','confidence']) for n in native[f,r['track_id']])
    for ps in people.values():assert len(ps)==len({r['track_id'] for r in ps.values()})
    times=p['requested_input_frames'];cid=p['candidate_id'];features=[dict(rally_candidate_id=cid,relative_source_frame=f-times[0],**pair_features(people[f])) for f in times];snap,summ,_=summarize_prefix(tag,[dict(r,ball_observation_present=0,ball_x_norm='',ball_y_norm='') for r in features])
    for name,rows in [('prefix_frame_features.csv',features),('snapshot_player_features.csv',[snap]),('prefix_player_summary_features.csv',[summ])]:write(OUT/name,rows)
    expected=[calculate_frame(cid,people,f,f-times[0]) for f in times];checked=0
    for actual,wanted in zip(features,expected):
        for k,v in wanted.items():
            if k!='rally_candidate_id':close(actual[k],v);checked+=1
    es,em=calculate_summary(expected)
    for actual,wanted in [(snap,es),(summ,em)]:
        for k,v in wanted.items():close(actual[k],v);checked+=1
    transition_counts={}
    for mode,rows in selected.items():
        count=0
        for slot in slots[:2]:
            rs=[r for r in rows if r['player_slot']==slot];count+=sum(a['track_id']!=b['track_id'] for a,b in zip(rs,rs[1:]))
        transition_counts[mode]=count
    baseline_near={rowkey(r) for r in selected['baseline_replay'] if r['player_slot'].startswith('near_')};guard_near={rowkey(r) for r in near};original_full=[rowkey(r) for r in read(CASE/'four_player_path.csv')];new_full=[rowkey(r) for r in combined]
    cap=cv2.VideoCapture(str(far.OUT/f'players_{tag}_global/global_review_h264.mp4'));decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(1080,1920);decoded+=1
    cap.release();assert decoded==p['context_last_source_frame']-p['context_first_source_frame']+1
    assert all(sha(Path(n))==h for n,h in protected.items())
    report=dict(status='passed',candidate_id=cid,previously_frozen_code_hash_matched=True,baseline_temporal_replay_exact=True,baseline_near_observations=len(baseline_near),guard_near_observations=len(guard_near),removed_near_observations=len(baseline_near-guard_near),added_near_observations=len(guard_near-baseline_near),near_raw_ID_transitions=transition_counts,full_four_player_assignments_identical=original_full==new_full,fixed31_features_identical=all(read(OUT/n)==read(CASE/n) for n in ['prefix_frame_features.csv','snapshot_player_features.csv','prefix_player_summary_features.csv']),matched_native_boxes=len(combined),recalculated_feature_values=checked,video_frames_decoded=decoded,four_observation_requested_times=sum(len(people[f])==4 for f in times),requested_times=31,manual_ID_binding=False,threshold_or_confirmation_length_tuned_on_transfer=False,canonical_pipeline_changed=False,independent_identity_or_position_accuracy_established=False,opportunity_has_near_ID_change=any(transition_counts.values()),protected_sources_unchanged=True)
    dump(OUT/'verification.json',report);dump(OUT/'result.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
