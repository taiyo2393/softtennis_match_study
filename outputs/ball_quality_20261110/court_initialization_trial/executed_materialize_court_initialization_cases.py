"""Export two predeclared development cases with automatic court-filtered seeds."""
import sys,argparse
from pathlib import Path
from collections import defaultdict,Counter
import cv2
from run_court_initialization_trial import ROOT,BASE,OUT as TRIAL,read,load,sha,dump,write
from build_rally_analysis_prototype import pair_features,summarize_prefix
from verify_player_prefix_expansion import calculate_frame,calculate_summary,close
sys.path.insert(0,str(ROOT.parent/'tracking_quality_20261102_stage10'))
import gap_pair_tracking as far
SLOTS=['near_front','near_rear','far_rear','far_front']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True,choices=['009','015']);tag=ap.parse_args().tag
    source=BASE/'cohort_tracking'/f'case_{tag}';temporal=TRIAL/f'case_{tag}/court_seed_only';out=TRIAL/f'case_{tag}/full_court_seed_only';out.mkdir()
    p=load(source/'protocol.json');plan=load(TRIAL/'protocol.json');assert all(sha(Path(n))==h for n,h in plan['protected_input_hashes'].items())
    original=source/f'tracking_quality_20261102_stage4/players_{tag}_temporal';summary=load(original/'summary.json');events=load(temporal/'events.json');assert events and events[0]['event']=='initialized';rows=read(temporal/'selected_four.csv');rawpath=Path(summary['input_csv'])
    protected={str(n):sha(n) for n in [TRIAL/'protocol.json',TRIAL/'run_result.json',ROOT/'court_initialization_guard.py',Path(__file__),Path(far.__file__),source/'protocol.json',source/'case_result.json',source/'four_player_path.csv',original/'summary.json',rawpath,temporal/'selected_four.csv',temporal/'events.json']}
    q=dict(p);q.update(experiment='Automatic court initialization; development-only known failure probe',initialization_selection_changed=True,AI_identity_binding_added=False,fully_automatic_result=True,manual_ID_binding=False,initialization_code=str(ROOT/'court_initialization_guard.py'),fixed_policy_protocol=str(TRIAL/'protocol.json'),fixed_policy_protocol_sha256=sha(TRIAL/'protocol.json'),numeric_tracking_and_recovery_parameters_changed=False,canonical_pipeline_changed=False,independent_accuracy_established=False,protected_input_hashes=protected);dump(out/'protocol.json',q)
    temp=out/f'tracking_quality_20261102_stage4/players_{tag}_temporal';temp.mkdir(parents=True);write(temp/'selected_four.csv',rows);dump(temp/'events.json',events);dump(temp/'summary.json',dict(summary,initialized_frame=events[0]['frame'],counts=dict(Counter(r['player_slot'] for r in rows)),source_warning='Automatic court-seed development variant, not original or human GT.'))
    far.OUT=out/'tracking_quality_20261102_stage7';far.OUT.mkdir();saved=sys.argv;sys.argv=[str(far.__file__),'--tag',tag]
    try:far.main()
    finally:sys.argv=saved
    near=[r for r in rows if r['player_slot'].startswith('near_')];farrows=read(far.OUT/f'players_{tag}_global/far_pair_path.csv');combined=sorted(near+farrows,key=lambda r:(int(r['source_frame']),SLOTS.index(r['player_slot'])));write(out/'four_player_path.csv',combined)
    native=defaultdict(list);people=defaultdict(dict)
    for r in read(rawpath):native[int(r['source_frame']),r['track_id']].append(r)
    for r in combined:
        f=int(r['source_frame']);assert r['player_slot'] not in people[f];people[f][r['player_slot']]=r;assert any(all(r[k]==n[k] for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y','confidence']) for n in native[f,r['track_id']])
    for ps in people.values():assert len({r['track_id'] for r in ps.values()})==len(ps)
    times=p['requested_input_frames'];cid=p['candidate_id'];features=[dict(rally_candidate_id=cid,relative_source_frame=f-times[0],**pair_features(people[f])) for f in times];snap,summ,_=summarize_prefix(tag,[dict(r,ball_observation_present=0,ball_x_norm='',ball_y_norm='') for r in features])
    for name,rs in [('prefix_frame_features.csv',features),('snapshot_player_features.csv',[snap]),('prefix_player_summary_features.csv',[summ])]:write(out/name,rs)
    checked=0;expected=[calculate_frame(cid,people,f,f-times[0]) for f in times]
    for actual,wanted in zip(features,expected):
        for k,v in wanted.items():
            if k!='rally_candidate_id':close(actual[k],v);checked+=1
    es,em=calculate_summary(expected)
    for actual,wanted in [(snap,es),(summ,em)]:
        for k,v in wanted.items():close(actual[k],v);checked+=1
    reference=source/('AI_seed_binding' if tag=='009' else 'AI_native_binding');assisted=read(reference/'four_player_path.csv');key=lambda r:tuple(r[k] for k in ['source_frame','player_slot','track_id','x1','y1','x2','y2']);point=lambda r:tuple(r[k] for k in ['track_id','x1','y1','x2','y2'])
    aa=[r for r in combined if int(r['source_frame']) in times];bb=[r for r in assisted if int(r['source_frame']) in times]
    near_pair_equal=all({point(r) for r in aa if int(r['source_frame'])==f and r['player_slot'].startswith('near_')}=={point(r) for r in bb if int(r['source_frame'])==f and r['player_slot'].startswith('near_')} for f in times)
    cap=cv2.VideoCapture(str(far.OUT/f'players_{tag}_global/global_review_h264.mp4'));n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(1080,1920);n+=1
    cap.release();assert n==p['context_last_source_frame']-p['context_first_source_frame']+1;assert all(sha(Path(x))==v for x,v in protected.items())
    report=dict(status='passed',candidate_id=cid,manual_ID_binding=False,AI_identity_binding_added=False,development_on_previously_known_failure=True,matched_native_boxes=len(combined),recalculated_feature_values=checked,video_frames_decoded=n,fixed31_times_unchanged=True,four_observation_times=sum(len(people[f])==4 for f in times),requested_times=31,prefix_near_pair_native_boxes_equal_AI_assisted=near_pair_equal,prefix_slot_raw_boxes_equal_AI_assisted=[key(r) for r in aa]==[key(r) for r in bb],prefix_feature_values_equal_AI_assisted=all(read(out/name)==read(reference/name) for name in ['prefix_frame_features.csv','snapshot_player_features.csv','prefix_player_summary_features.csv']),protected_sources_unchanged=True,interpolation=False,independent_identity_or_position_accuracy_established=False,canonical_pipeline_changed=False)
    dump(out/'verification.json',report);dump(out/'case_result.json',dict(status='automatic_court_seed_development_path_constructed',candidate_id=cid,requested_input_frames=times,input_times_with_four_observations=report['four_observation_times'],artifact_hashes={x.name:sha(x) for x in out.glob('*.csv')}));print(__import__('json').dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
