"""Export a separate development variant; retain every original rejected path."""
import argparse,sys,json
from pathlib import Path
from collections import defaultdict,Counter
import cv2
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
from build_rally_analysis_prototype import pair_features,summarize_prefix
from verify_player_prefix_expansion import calculate_frame,calculate_summary,close
sys.path.insert(0,str(ROOT.parent/'tracking_quality_20261102_stage10'))
import gap_pair_tracking as far

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True,choices=['012','018']);a=ap.parse_args();tag=a.tag
    trial=BASE/'pair_initialization_development/matched_trial';plan=load(trial/'protocol.json');result=load(trial/'result.json');assert result['status']=='completed18_matched_development_replays';assert all(sha(Path(n))==h for n,h in plan['protected_input_hashes'].items())
    source=BASE/f'cohort_tracking/case_{tag}';out=source/'near_pair_development';out.mkdir();temporal=trial/f'case_{tag}/pair_rank';original=source/f'tracking_quality_20261102_stage4/players_{tag}_temporal';summary=load(original/'summary.json');rows=read(temporal/'selected_four.csv');events=load(temporal/'events.json');rawpath=Path(summary['input_csv']);assert events[0]['event']=='initialized'
    protected={str(n):sha(n) for n in [trial/'protocol.json',trial/'result.json',trial/'invariance_verification.json',ROOT/'pair_initialization_guard.py',Path(__file__),Path(far.__file__),source/'protocol.json',source/'case_result.json',source/'four_player_path.csv',original/'summary.json',rawpath,temporal/'selected_four.csv',temporal/'events.json']}
    p=load(source/'protocol.json');q=dict(p);q.update(experiment='Near-pair appearance ranking, known-case development variant',initialization_selection_changed=True,AI_identity_binding_added=False,individual_ID_binding_added=False,fully_automatic_generation=True,independent_automatic_success_established=False,development_selected_after_known018_failure=True,processing_intervention='automatic_near_pair_initialization_development',initialization_code=str(ROOT/'pair_initialization_guard.py'),fixed_policy_protocol=str(trial/'protocol.json'),fixed_policy_protocol_sha256=sha(trial/'protocol.json'),numeric_tracking_and_recovery_parameters_changed=False,canonical_pipeline_changed=False,protected_input_hashes=protected);dump(out/'protocol.json',q)
    temp=out/f'tracking_quality_20261102_stage4/players_{tag}_temporal';temp.mkdir(parents=True);write(temp/'selected_four.csv',rows);dump(temp/'events.json',events);dump(temp/'summary.json',dict(summary,initialized_frame=events[0]['frame'],counts=dict(Counter(r['player_slot'] for r in rows)),source_warning='Known-case pair initialization development variant, not original or human GT.'))
    far.OUT=out/'tracking_quality_20261102_stage7';far.OUT.mkdir();saved=sys.argv;sys.argv=[str(far.__file__),'--tag',tag]
    try:far.main()
    finally:sys.argv=saved
    near=[r for r in rows if r['player_slot'].startswith('near_')];farrows=read(far.OUT/f'players_{tag}_global/far_pair_path.csv');slots=['near_front','near_rear','far_rear','far_front'];combined=sorted(near+farrows,key=lambda r:(int(r['source_frame']),slots.index(r['player_slot'])));write(out/'four_player_path.csv',combined)
    native=defaultdict(list);people=defaultdict(dict)
    for r in read(rawpath):native[int(r['source_frame']),r['track_id']].append(r)
    for r in combined:
        f=int(r['source_frame']);assert r['player_slot'] not in people[f];people[f][r['player_slot']]=r;assert any(all(r[k]==x[k] for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y','confidence']) for x in native[f,r['track_id']])
    for ps in people.values():assert len({r['track_id'] for r in ps.values()})==len(ps)
    times=p['requested_input_frames'];cid=p['candidate_id'];features=[dict(rally_candidate_id=cid,relative_source_frame=f-times[0],**pair_features(people[f])) for f in times];snap,summ,_=summarize_prefix(tag,[dict(r,ball_observation_present=0,ball_x_norm='',ball_y_norm='') for r in features])
    for name,rs in [('prefix_frame_features.csv',features),('snapshot_player_features.csv',[snap]),('prefix_player_summary_features.csv',[summ])]:write(out/name,rs)
    checked=0;expected=[calculate_frame(cid,people,f,f-times[0]) for f in times]
    for actual,wanted in zip(features,expected):
        for k,value in wanted.items():
            if k!='rally_candidate_id':close(actual[k],value);checked+=1
    es,em=calculate_summary(expected)
    for actual,wanted in [(snap,es),(summ,em)]:
        for k,value in wanted.items():close(actual[k],value);checked+=1
    keys=['source_frame','player_slot','track_id','x1','y1','x2','y2'];key=lambda r:tuple(r[k] for k in keys);original_keys={key(r) for r in read(source/'four_player_path.csv')};new_keys={key(r) for r in combined}
    cap=cv2.VideoCapture(str(far.OUT/f'players_{tag}_global/global_review_h264.mp4'));n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(1080,1920);n+=1
    cap.release();assert n==p['context_last_source_frame']-p['context_first_source_frame']+1;assert all(sha(Path(name))==value for name,value in protected.items())
    v=dict(status='passed',candidate_id=cid,verification_scope='Source boxes, feature calculations, rendering and preservation; person identity needs image review',initialized_frame=events[0]['frame'],seed_ids=events[0]['seed_ids'],manual_ID_binding=False,AI_identity_binding_added=False,development_on_previously_known_case=True,matched_native_boxes=len(combined),recalculated_feature_values=checked,video_frames_decoded=n,fixed31_times_unchanged=True,four_observation_times=sum(len(people[f])==4 for f in times),requested_times=31,missing_requested_times=[dict(source_frame=f,missing_slots=[s for s in slots if s not in people[f]]) for f in times if len(people[f])!=4],full_assignments_equal_original=new_keys==original_keys,removed_original_assignments=len(original_keys-new_keys),added_assignments=len(new_keys-original_keys),prefix_feature_values_equal_original=all(read(out/name)==read(source/name) for name in ['prefix_frame_features.csv','snapshot_player_features.csv','prefix_player_summary_features.csv']),protected_sources_unchanged=True,interpolation=False,independent_identity_or_position_accuracy_established=False,canonical_pipeline_changed=False)
    dump(out/'verification.json',v);dump(out/'case_result.json',dict(status='near_pair_initialization_development_path_constructed',candidate_id=cid,requested_input_frames=times,input_times_with_four_observations=v['four_observation_times'],artifact_hashes={x.name:sha(x) for x in out.glob('*.csv')}));print(json.dumps(v,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
